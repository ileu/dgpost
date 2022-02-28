import functools
import logging
import warnings
from typing import Callable, Union

import numpy as np
import pandas as pd
import pint
import uncertainties.unumpy as unp
from scipy.optimize import least_squares, minimize

from .circuit_utils.circuit_parser import parse_circuit
from .helpers import pQ
import numpy
import scipy.optimize

from numpy import asfarray

from scipy.optimize.optimize import (
    _status_message,
    OptimizeResult,
)

logger = logging.getLogger(__name__)


def _neldermead_errors(sim, fsim, func):
    # fit quadratic coefficients
    fun = func

    n = len(sim) - 1

    x = 0.5 * (sim[numpy.mgrid[0:6, 0:6]][1] + sim[numpy.mgrid[0:6, 0:6]][0])

    for i in range(n + 1):
        assert numpy.array_equal(x[i, i], sim[i])
        for j in range(n + 1):
            assert numpy.array_equal(x[i, j], 0.5 * (sim[i] + sim[j]))

    y = numpy.nan * numpy.ones(shape=(n + 1, n + 1))
    for i in range(n + 1):
        y[i, i] = fsim[i]
        for j in range(i + 1, n + 1):
            y[i, j] = y[j, i] = fun(x[i, j])

    y0i = y[numpy.mgrid[0:6, 0:6]][0][1:, 1:, 0]
    for i in range(n):
        for j in range(n):
            assert y0i[i, j] == y[0, i + 1], (i, j)

    y0j = y[numpy.mgrid[0:6, 0:6]][0][0, 1:, 1:]
    for i in range(n):
        for j in range(n):
            assert y0j[i, j] == y[0, j + 1], (i, j)

    b = 2 * (y[1:, 1:] + y[0, 0] - y0i - y0j)
    for i in range(n):
        assert abs(b[i, i] - 2 * (fsim[i + 1] + fsim[0] - 2 * y[0, i + 1])) < 1e-12
        for j in range(n):
            if i == j:
                continue
            assert (
                abs(
                    b[i, j]
                    - 2 * (y[i + 1, j + 1] + fsim[0] - y[0, i + 1] - y[0, j + 1])
                )
                < 1e-12
            )

    q = (sim - sim[0])[1:].T
    for i in range(n):
        assert numpy.array_equal(q[:, i], sim[i + 1] - sim[0])

    varco = numpy.dot(q, numpy.dot(numpy.linalg.inv(b), q.T))
    return numpy.sqrt(numpy.diag(varco))


def _wrap_scalar_function_maxfun_validation(function, args, maxfun):
    # wraps a minimizer function to count number of evaluations
    # and to easily provide an args kwd.
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(x, *wrapper_args):
        if ncalls[0] >= maxfun:
            raise ValueError("Too many function calls")
        ncalls[0] += 1
        # A copy of x is sent to the user function (gh13740)
        fx = function(np.copy(x), *(wrapper_args + args))
        # Ideally, we'd like to a have a true scalar returned from f(x). For
        # backwards-compatibility, also allow np.array([1.3]),
        # np.array([[1.3]]) etc.
        if not np.isscalar(fx):
            try:
                fx = np.asarray(fx).item()
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "The user-provided objective function "
                    "must return a scalar value."
                ) from e
        return fx

    return ncalls, function_wrapper


def minimize_neldermead_witherrors(
    fun,
    x0,
    args=(),
    callback=None,
    xtol=1e-4,
    ftol=1e-4,
    maxiter=None,
    maxfev=None,
    disp=False,
    return_all=False,
    with_errors=True,
    **unknown_options,
):
    """
    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.

    Options for the Nelder-Mead algorithm are:
        disp : bool
            Set to True to print convergence messages.
        xtol : float
            Relative error in solution `xopt` acceptable for convergence.
        ftol : float
            Relative error in ``fun(xopt)`` acceptable for convergence.
        maxiter : int
            Maximum number of iterations to perform.
        maxfev : int
            Maximum number of function evaluations to make.

    This function is called by the `minimize` function with
    `method=minimize_neldermead_with_errors`. It is not supposed to be called directly.
    """
    maxfun = maxfev
    retall = return_all

    x0 = asfarray(x0).flatten()
    N = len(x0)
    rank = len(x0.shape)
    if not -1 < rank < 2:
        raise ValueError("Initial guess must be a scalar or rank-1 sequence.")
    if maxiter is None:
        maxiter = N * 200
    if maxfun is None:
        maxfun = N * 200

    fcalls, func = _wrap_scalar_function_maxfun_validation(fun, args, maxfun)

    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5
    one2np1 = list(range(1, N + 1))

    if rank == 0:
        sim = numpy.zeros((N + 1,), dtype=x0.dtype)
    else:
        sim = numpy.zeros((N + 1, N), dtype=x0.dtype)
    fsim = numpy.zeros((N + 1,), float)
    sim[0] = x0
    if retall:
        allvecs = [sim[0]]
    fsim[0] = func(x0)
    nonzdelt = 0.05
    zdelt = 0.00025
    for k in range(0, N):
        y = numpy.array(x0, copy=True)
        if y[k] != 0:
            y[k] = (1 + nonzdelt) * y[k]
        else:
            y[k] = zdelt

        sim[k + 1] = y
        f = func(y)
        fsim[k + 1] = f

    ind = numpy.argsort(fsim)
    fsim = numpy.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = numpy.take(sim, ind, 0)

    iterations = 1

    while fcalls[0] < maxfun and iterations < maxiter:
        if (
            numpy.max(numpy.ravel(numpy.abs(sim[1:] - sim[0]))) <= xtol
            and numpy.max(numpy.abs(fsim[0] - fsim[1:])) <= ftol
        ):
            break

        xbar = numpy.add.reduce(sim[:-1], 0) / N
        xr = (1 + rho) * xbar - rho * sim[-1]
        fxr = func(xr)
        doshrink = 0

        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            fxe = func(xe)

            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else:  # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else:  # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                    fxc = func(xc)

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    # Perform an inside contraction
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    fxcc = func(xcc)

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        fsim[j] = func(sim[j])

        ind = numpy.argsort(fsim)
        sim = numpy.take(sim, ind, 0)
        fsim = numpy.take(fsim, ind, 0)
        if callback is not None:
            callback(sim[0])
        iterations += 1
        if retall:
            allvecs.append(sim[0])

    x = sim[0]
    fval = numpy.min(fsim)
    warnflag = 0
    errors = None

    if fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message["maxfev"]
        if disp:
            print("Warning: " + msg)
    elif iterations >= maxiter:
        warnflag = 2
        msg = _status_message["maxiter"]
        if disp:
            print("Warning: " + msg)
    else:
        msg = _status_message["success"]
        errors = _neldermead_errors(sim, fsim, func)
        print(errors)
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % iterations)
            print("         Function evaluations: %d" % fcalls[0])

    result = OptimizeResult(
        fun=fval,
        nit=iterations,
        nfev=fcalls[0],
        status=warnflag,
        success=(warnflag == 0),
        message=msg,
        x=x,
        errors=errors,
        sim=sim,
        fsim=fsim,
    )

    if retall:
        result["allvecs"] = allvecs
    return result


def load_data(*cols: str):
    """
    Decorator factory for data loading

    Creates a decorator that will load the coloumns specified in ``cols``
    and calls the wrapped function for each row entry.

    Parameters
    ----------
    cols
        list of strings with the col names used to call the function
    """

    def loading(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # check if there is an output name in kwargs, if not take func name
            if "output" not in kwargs:
                kwargs["output"] = func.__name__

            if len(args) == 0:
                # if there are no argument, it means the function is not called with a dataframe
                return func(**kwargs)
            else:
                # check if the first argument is a dataframe then break and there should only be one kwarg
                # or check if all the arguments are np.ndarray
                for arg in args:
                    if isinstance(arg, pd.DataFrame):
                        if len(args) != 1:
                            raise ValueError(
                                "Only the DataFrame should be given as argument"
                            )
                        df = arg
                        break
                    elif isinstance(arg, np.ndarray):
                        continue
                    else:
                        raise ValueError(
                            "The argument is neither a DataFrame or an ndarray"
                        )
                else:
                    if len(cols) == len(args):
                        return func(*args, **kwargs)
                    else:
                        raise ValueError("Not enought data rows given")

                # check if the dataframe has a units attribute else create it
                if "units" not in df.attrs:
                    df.attrs["units"] = {}

                # each kwarg to get a data colum specified in cols should be a string
                # create a list for each data column that should get past to the function
                raw_data = []
                for col in cols:
                    if isinstance(col, str):
                        raw_data.append(pQ(df, kwargs.pop(col)))
                    else:
                        raise ValueError(f"Provided value for {col} is not a string")

                # transpose the list and iterate over it
                # so that we iterate over each time step
                for index, row in enumerate(zip(*raw_data)):
                    # create kwargs for each data col
                    data = {col: unp.nominal_values(r.m) for col, r in zip(cols, row)}

                    # call the function for each row in the data
                    # the function should return two dicts, which contain the values
                    # and units of the data with the same key
                    vals, units = func(**data, **kwargs)
                    for val_name in vals:
                        # check if the column already exists in the dataframe, if not create empty column
                        if val_name not in df:
                            df[val_name] = ""
                        # fill the column and units attribute with the given values
                        df[val_name].iloc[index] = vals[val_name]
                        df.attrs["units"][val_name] = units[val_name]

        return wrapper

    return loading


def fit_routine(
    opt_func: Callable[[list], float],
    fit_guess: list[float],
    bounds: list[tuple],
    repeat: int = 1,
):
    """
    Fitting routine which uses scipys least_squares and minimize.

    Least_squares is a good fitting method but will get stuck in local minimas.
    For this reason, the Nelder-Mead-Simplex algorithm is used to get out of these local minima.
    The fitting routine is inspired by Relaxis 3 fitting procedure.
    More information about it can be found on page 188 of revison 1.25 of Relaxis User Manual.
    https://www.rhd-instruments.de/download/manuals/relaxis_manual.pdf

    Open issue is estimate the errors of the paramters. For further information look:
    - https://github.com/andsor/notebooks/blob/master/src/nelder-mead.md
    - https://math.stackexchange.com/questions/2447382/nelder-mead-function-fit-error-estimation-via-surface-fit
    - https://stats.stackexchange.com/questions/424073/calculate-the-uncertainty-of-a-mle

    Parameters
    ----------
    opt_func
        function that gets minimized
    fit_guess
        initial guess for minimization
    bounds
        bounds of the fitting parameters
    repeat
        how many times the least squares and minimize step gets repeated

    Returns
    -------
    scipy.optimize.OptimizeResult
        the `OptimizeResult` from the last step
    """
    initial_value = np.array(fit_guess)

    # least squares have different format for bounds
    ls_bounds_lb = [bound[0] for bound in bounds]
    ls_bounds_ub = [bound[1] for bound in bounds]
    ls_bounds = (ls_bounds_lb, ls_bounds_ub)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="overflow encountered in tanh")

        logger.debug(f"Started fitting routine")
        for i in range(repeat):
            logger.debug(f"Fitting routine pass {i}")
            opt_result = least_squares(
                opt_func,
                initial_value,
                bounds=ls_bounds,
                xtol=1e-13,
                max_nfev=1000,
                ftol=1e-9,
            )
            initial_value = opt_result.x
            logger.debug(f"Finished least squares")
            opt_result = minimize(
                opt_func,
                initial_value,
                bounds=bounds,
                tol=1e-13,
                options={"maxiter": 1e4, "fatol": 1e-9},
                method=minimize_neldermead_witherrors,
            )
            initial_value = opt_result.x
            logger.debug(f"Finished Nelder-Mead")
    logger.debug(f"Finished fitting routine")
    return opt_result


def separate_data(
    data: Union[np.ndarray, pint.Quantity], unit: str
) -> tuple[np.ndarray, np.ndarray, str]:
    """Separates the data into values, errors and unit

    Parameters
    ----------
    data
        List of data points which are either floats or ufloats

    unit
        converts the input to this unit
    Returns
    -------
    (values, errors, old_unit)
        returns the values with the corresponding errors and the unit of the input array
    """
    old_unit = ""
    if isinstance(data, pint.Quantity):
        old_unit = data.u
        data = data.to(unit)
        data = data.m

    data_values = unp.nominal_values(data)
    data_errors = unp.std_devs(data)
    return data_values, data_errors, old_unit


@load_data("real", "imag", "freq")
def fit_circuit(
    real: Union[np.ndarray, pint.Quantity],
    imag: Union[np.ndarray, pint.Quantity],
    freq: Union[np.ndarray, pint.Quantity],
    circuit: str,
    initial_values: dict[str, float],
    output: str,
    fit_bounds: dict[str, tuple[float, float]] = None,
    fit_constants: list[str] = None,
    ignore_neg_res: bool = True,
    upper_freq: float = np.inf,
    lower_freq: float = 0,
    repeat: int = 1,
) -> tuple[dict[str, float], dict[str, str]]:
    """
    Fitting function for electrochemical impedance spectroscopy (EIS) data.

    For the fitting a model or equivilant circuit is needed. The equivilant circuit is defined as a string.
    To combine elements in series a dash (-) is used. Elements in parallel are wrapped by p( , ).
    An element is definied by an identifier (usually letters) followed by a digit.
    Already implemented elements are located in :class:`circuit_components<circuit_utils.circuit_components>`:

    +------------------------+--------+-----------+---------------+--------------+
    | Name                   | Symbol | Paramters | Bounds        | Units        |
    +------------------------+--------+-----------+---------------+--------------+
    | Resistor               | R      | R         | (1e-6, 1e6)   | Ohm          |
    +------------------------+--------+-----------+---------------+--------------+
    | Capacitance            | C      | C         | (1e-20, 1)    | Farrad       |
    +------------------------+--------+-----------+---------------+--------------+
    | Constant Phase Element | CPE    | CPE_Q     | (1e-20, 1)    | Ohm^-1 s^a   |
    |                        |        +-----------+---------------+--------------+
    |                        |        | CPE_a     | (0, 1)        |              |
    +------------------------+--------+-----------+---------------+--------------+
    | Warburg element        | W      | W         | (0, 1e10)     | Ohm^-1 s^0.5 |
    +------------------------+--------+-----------+---------------+--------------+
    | Warburg short element  | Ws     | Ws_R      | (0, 1e10)     | Ohm          |
    |                        |        +-----------+---------------+--------------+
    |                        |        | Ws_T      | (1e-10, 1e10) | s            |
    +------------------------+--------+-----------+---------------+--------------+
    | Warburg open elemnt    | Wo     | Wo_R      | (0, 1e10)     | Ohm          |
    |                        |        +-----------+---------------+--------------+
    |                        |        | Wo_T      | (1e-10, 1e10) | s            |
    +------------------------+--------+-----------+---------------+--------------+

    Additionaly an initial guess for the fitting parameters is needed.
    The initial guess is given as a dictionary where each key is the parameters name and
    the coresponding value is the guessed value for the circuit.

    The bounds of each paramater can be customized by the ``fit_bounds`` parameter.
    This parameter is a dictionary, where each key is the parameter name and the value constists of a tuple for the lower and upper bound (lb, ub).

    To hold a parameter constant, add the name of the paramter to a list and pass it as ``fit_constants``

    Parameters
    ----------
    real
        numpy.ndarray or pint.Quantity of a numpy.ndarray containing the real part of the impedance data
        The unit of the provided data should be or gets converted to 'Ω'

    imag
        numpy.ndarray or pint.Quantity of a numpy.ndarray containing the negative imaginary part of the impedance data
        The unit of the provided data should be or gets converted to 'Ω'

    freq
        numpy.ndarray or pint.Quantity of a numpy.ndarray containing the frequency of the impedance data
        The unit of the provided data should be or gets converted to 'Hz'

    circuit
        Equivalent circuit for the fit

    initial_values
        dictionary with initial values
        Structure: {"param name": value, ... }

    output
        the name of the output

    fit_bounds
        Custom bounds for a parameter if default bounds are not wanted
        Structure: {"param name": (lower bound, upper bound), ...}
        Default is ''None''

    fit_constants
        list of parameters which should stay constant during fitting
        Structure: ["param name", ...]
        Default is ''None''

    ignore_neg_res
        ignores impedance values with a negative real part

    upper_freq:
        upper frequency bound to be considered for fitting

    lower_freq:
        lower frequency boudn to be considered for fitting

    repeat
        how many times ``fit_routine`` gets called

    Returns
    -------
    (parameters, units)
        A tuple containing two dicts.
         - parameters, contains all the values of the parameters accessable by their names
         - units, contains all the units of the parameters accessable by their names
    """
    # separate the data into nominal values and uncertainties and normalize the units.
    real_data = separate_data(real, "Ω")[0]
    imag_data = separate_data(imag, "Ω")[0]
    freq_data = separate_data(freq, "Hz")[0]

    # select appropriate data
    # select only frequency in the allowed range
    mask = np.logical_and(lower_freq < freq_data, freq_data < upper_freq)
    if ignore_neg_res:
        # ignore negative real
        mask = np.logical_and(mask, real_data > 0)

    frequency = freq_data[mask]
    z = real_data[mask] - 1j * imag_data[mask]

    # interpret fit circuit
    param_info, circ_calc = parse_circuit(circuit)

    if fit_bounds is None:
        fit_bounds = {}

    if fit_constants is None:
        fit_constants = []

    param_values = initial_values.copy()  # stores all the values of the parameters
    variable_names = []  # name of the parameters that are not fixed
    variable_guess = []  # guesses for the parameters that are not fixed
    variable_bounds = []  # bounds of the parameters that are not fixed

    for p in param_info:
        p_name = p["name"]
        if p_name in initial_values:
            if p_name not in fit_constants:
                variable_bounds.append(fit_bounds.get(p_name, p["bounds"]))
                variable_guess.append(initial_values.get(p_name))
                variable_names.append(p_name)
        else:
            raise ValueError(f"No initial value given for {p_name}")

    # calculate the weight of each datapoint
    def weight(error, value):
        """caluclates the absoulte value squared and devides the error by it"""
        square_value = value.real ** 2 + value.imag ** 2
        return np.true_divide(error, square_value)

    # calculate rmse
    def rmse(predicted, actual):
        """Calculates the root mean squared error between two vectors"""
        e = np.abs(np.subtract(actual, predicted))
        se = np.square(e)
        wse = weight(se, actual)
        mse = np.nansum(wse)
        return np.sqrt(mse)

    # prepare optimizing function:
    def opt_func(x: list[float]):
        params = dict(zip(variable_names, x))
        param_values.update(params)
        predict = circ_calc(param_values, frequency)
        err = rmse(predict, z)
        return err

    # fit
    opt_result = fit_routine(
        opt_func,
        variable_guess,
        variable_bounds,
        repeat=repeat,
    )

    # update values in ParameterList
    param_values.update(dict(zip(variable_names, opt_result.x)))

    # create output dictionaries
    parameters = {f"{output}->circuit": circuit}
    units = {f"{output}->circuit": ""}

    # iterate over param_info to make sure every parameter of the circuit gets added
    # necessary since we need the unit of the parameter
    for p in param_info:
        p_name = p["name"]
        col_name = f"{output}->{p_name}"
        parameters[col_name] = param_values[p_name]
        units[col_name] = p["unit"]

    return parameters, units


@load_data("freq")
def calc_circuit(
    circuit: str,
    parameters: dict[str, float],
    output: str,
    freq: Union[np.ndarray, pint.Quantity],
) -> tuple[dict[str, float], dict[str, str]]:
    """
    Calculate and adds the impedance as columns to the given dataframe for the given parameters, frequency and circuit.

    Parameters
    ----------
    circuit
        string describing the circuit. For more info see impedance.fit_circuit

    parameters
        dictionary with initial values
        Structure: {"param name": value, ... }

    output
        the name of the output

    freq
        numpy.ndarray or pint.Quantity of a numpy.ndarray containing the frequency
        The provided data should be in the unit of "Hz"

    Returns
    -------
    (parameters, units)
        A tuple containing two dicts.
         - values, containing the two lists of the real and the negaitve imaginary part of the impedanc ce
         - units, contains the corresponding units of the impedance
    """
    # separate the freq data into values, errors and normalize the unit
    freq_data, freq_error, freq_unit = separate_data(freq, "Hz")

    # parse the circuit
    param_info, circ_calc = parse_circuit(circuit)

    # check if every parameter has a value
    for p in param_info:
        if p["name"] not in parameters:
            raise ValueError(f"No initial value given for {p['name']}")

    # calculate the impedance
    impedance = circ_calc(parameters, freq_data)

    # generate the output
    real_name = f"{output}->Re(Z)"
    imag_name = f"{output}->-Im(Z)"

    values = {
        real_name: impedance.real.tolist(),
        imag_name: (-impedance.imag).tolist(),
    }
    units = {
        real_name: "Ω",
        imag_name: "Ω",
    }
    return values, units

import os
import pytest
import pandas as pd
import numpy as np

from dgpost.transform import table
from dgpost.utils import transform
from .utils import compare_dfs

from yadg.dgutils import ureg


@pytest.mark.parametrize(
    "a, b, conflicts, output",
    [
        (  # ts0 - sum
            {
                "a": ureg.Quantity(10.0, "ml/min"),
                "b": ureg.Quantity(6.0, "ml/min"),
            },
            {
                "b": ureg.Quantity(0.6, "l/h"),
                "c": ureg.Quantity(6.0, "l/h"),
            },
            "sum",
            {
                "c->a": ureg.Quantity(10.0, "ml/min"),
                "c->b": ureg.Quantity(16.0, "ml/min"),
                "c->c": ureg.Quantity(100.0, "ml/min"),
            },
        ),
        (  # ts1 - replace
            {
                "a": ureg.Quantity(10.0, "mmol/s"),
                "b": ureg.Quantity(6.0, "mmol/s"),
            },
            {
                "b": ureg.Quantity(3.6, "mol/h"),
                "c": ureg.Quantity(72.0, "mol/h"),
            },
            "replace",
            {
                "c->a": ureg.Quantity(10.0, "mmol/s"),
                "c->b": ureg.Quantity(1.0, "mmol/s"),
                "c->c": ureg.Quantity(20.0, "mmol/s"),
            },
        ),
        (  # ts2 - sum with nan
            {
                "a": ureg.Quantity(10.0, "ml/min"),
                "b": ureg.Quantity(float("NaN"), "ml/min"),
            },
            {
                "b": ureg.Quantity(0.6, "l/h"),
                "c": ureg.Quantity(6.0, "l/h"),
            },
            "sum",
            {
                "c->a": ureg.Quantity(10.0, "ml/min"),
                "c->b": ureg.Quantity(10.0, "ml/min"),
                "c->c": ureg.Quantity(100.0, "ml/min"),
            },
        ),
    ],
)
def test_table_combine_namespace_direct(a, b, conflicts, output):
    ret = table.combine_namespaces(a=a, b=b, conflicts=conflicts, output="c")
    for k, v in ret.items():
        assert np.allclose(v, output[k])


@pytest.mark.parametrize(
    "inpath, spec, col",
    [
        (  # ts0 - sum
            "namespaces_1.xlsx",
            [{"a": "rin(GC)", "b": "rin(LC)", "output": "rtot"}],
            "rtot",
        ),
        (  # ts1 - sum
            "namespaces_2.xlsx",
            [{"a": "rin(GC)", "b": "rin(LC)", "output": "rtot", "chemicals": True}],
            "rtot",
        ),
        (  # ts2 - sum with nans
            "namespaces_3.xlsx",
            [{"a": "rin(GC)", "b": "rin(LC)", "output": "rtot", "chemicals": True}],
            "rtot",
        ),
    ],
)
def test_table_combine_namespace_excel(inpath, spec, col, datadir):
    os.chdir(datadir)
    df = pd.read_excel(inpath)
    df = transform(df, "table.combine_namespaces", using=spec)
    compare_dfs(df[f"r{col}"], df[col])

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("Orange")
pytest.importorskip("pandas")

import chem_inf_widgets  # noqa: F401  # triggers package compatibility patches
import Orange.data as orange_data
from Orange.data import Domain, DiscreteVariable, Table
from Orange.data import pandas_compat


def _table_with_discrete_meta() -> Table:
    variable = DiscreteVariable("status", values=["ok", "warn"])
    domain = Domain([], metas=[variable])
    metas = np.array([[0.0], [1.0], [np.nan]], dtype=object)
    return Table.from_numpy(domain, X=np.empty((3, 0)), metas=metas)


def test_table_to_frame_discrete_meta_no_futurewarning():
    table = _table_with_discrete_meta()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        df = pandas_compat.table_to_frame(table, include_metas=True)

    assert list(df["status"].astype(str))[:2] == ["ok", "warn"]
    assert not any(
        "Downcasting object dtype arrays on .fillna" in str(item.message)
        for item in caught
    )


def test_orange_namespace_uses_patched_table_to_frame():
    assert orange_data.table_to_frame is pandas_compat.table_to_frame
    assert getattr(pandas_compat.table_to_frame, "_owcheminf_patched", False) is True

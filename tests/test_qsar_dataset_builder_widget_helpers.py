from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from Orange.data import ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table

from chem_inf_widgets.widgets.ow_qsar_dataset_builder import _is_nan, _orange_value_to_python, _table_to_records


def test_qsar_dataset_builder_nan_helper_accepts_numpy_float():
    assert _is_nan(np.float64("nan")) is True
    assert _is_nan(1.0) is False


def test_qsar_dataset_builder_orange_value_to_python_maps_discrete_values():
    relation = DiscreteVariable("relation", values=["=", "<"])

    assert _orange_value_to_python(relation, 0.0) == "="
    assert _orange_value_to_python(relation, 1.0) == "<"
    assert _orange_value_to_python(relation, 9.0) == 9.0


def test_qsar_dataset_builder_table_to_records_handles_discrete_and_missing_values():
    domain = Domain(
        [ContinuousVariable("pActivity")],
        metas=[DiscreteVariable("relation", values=["=", "<"]), StringVariable("unit")],
    )
    table = Table.from_list(
        domain,
        [
            [6.2, "=", "nM"],
            [float("nan"), "<", "uM"],
        ],
    )

    records = _table_to_records(table)

    assert records == [
        {"pActivity": 6.2, "relation": "=", "unit": "nM"},
        {"pActivity": "", "relation": "<", "unit": "uM"},
    ]

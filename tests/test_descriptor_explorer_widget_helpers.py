from __future__ import annotations

import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from Orange.data import ContinuousVariable, Domain, StringVariable, Table

from chem_inf_widgets.widgets.ow_descriptor_explorer import _df_to_table, _table_to_df


def test_descriptor_explorer_table_to_df_preserves_numeric_and_string_metas():
    domain = Domain(
        [ContinuousVariable("x1")],
        metas=[ContinuousVariable("score"), StringVariable("note")],
    )
    table = Table.from_list(
        domain,
        [
            [1.5, 2.0, "ok"],
            [2.5, 3.5, "needs review"],
        ],
    )

    df = _table_to_df(table)

    assert list(df.columns) == ["x1", "score", "note"]
    assert df["score"].tolist() == [2.0, 3.5]
    assert df["note"].tolist() == ["ok", "needs review"]


def test_descriptor_explorer_table_to_df_falls_back_to_strings_for_non_numeric_meta_values():
    domain = Domain(
        [ContinuousVariable("x1")],
        metas=[ContinuousVariable("score")],
    )
    table = Table.from_numpy(
        domain,
        X=[[1.0], [2.0]],
        metas=[["bad"], ["3.2"]],
    )

    df = _table_to_df(table)

    assert df["score"].tolist() == ["bad", "3.2"]


def test_descriptor_explorer_df_to_table_keeps_non_numeric_columns_as_metas():
    table = _df_to_table(
        _table_to_df(
            Table.from_list(
                Domain([ContinuousVariable("x1")], metas=[StringVariable("label")]),
                [[1.0, "a"], [2.0, "b"]],
            )
        )
    )

    assert [var.name for var in table.domain.attributes] == ["x1"]
    assert [var.name for var in table.domain.metas] == ["label"]

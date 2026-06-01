from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication
from Orange.data import Domain, StringVariable, Table

from chem_inf_widgets.widgets import ow_similarity_search as similarity_widget_module
from chem_inf_widgets.widgets.ow_similarity_search import OWSimilaritySearch


_APP = QApplication.instance() or QApplication([])


def _smiles_table(smiles: str) -> Table:
    return Table.from_list(Domain([], metas=[StringVariable("SMILES")]), [[smiles]])


def test_similarity_search_warns_when_query_and_reference_preparse_fail():
    widget = OWSimilaritySearch()
    widget.auto_run = False
    query = _smiles_table("CCO")
    reference = _smiles_table("CCN")
    warnings: list[str] = []

    with (
        patch.object(
            similarity_widget_module,
            "table_to_chemmols_with_report",
            side_effect=RuntimeError("preparse boom"),
        ),
        patch.object(
            similarity_widget_module,
            "set_widget_warning",
            lambda _w, message: warnings.append(message or ""),
        ),
    ):
        widget.set_query_data(query)
        widget.set_reference_data(reference)
        _APP.processEvents()

    assert warnings
    assert warnings[-1] == (
        "Could not pre-parse query input table: preparse boom; "
        "Could not pre-parse reference input table: preparse boom"
    )

    widget.onDeleteWidget()
    widget.close()

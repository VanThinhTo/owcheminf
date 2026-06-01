from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication
from Orange.data import Domain, StringVariable, Table

from chem_inf_widgets.widgets import ow_drug_filter as drug_filter_widget_module
from chem_inf_widgets.widgets.ow_drug_filter import OWDrugFilter


_APP = QApplication.instance() or QApplication([])


def test_drug_filter_set_data_warns_when_table_preparse_fails():
    widget = OWDrugFilter()
    warnings: list[str] = []
    table = Table.from_list(Domain([], metas=[StringVariable("SMILES")]), [["CCO"]])

    with (
        patch.object(drug_filter_widget_module, "clear_widget_messages", lambda _widget: None),
        patch.object(
            drug_filter_widget_module,
            "table_to_chemmols_with_report",
            side_effect=RuntimeError("preparse boom"),
        ),
        patch.object(
            drug_filter_widget_module,
            "set_widget_warning",
            lambda _w, message: warnings.append(message or ""),
        ),
    ):
        widget.set_data(table)
        _APP.processEvents()

    assert warnings
    assert warnings[-1] == "Could not pre-parse input table: preparse boom"

    widget.onDeleteWidget()
    widget.close()

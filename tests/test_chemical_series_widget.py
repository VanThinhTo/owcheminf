from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")
pytest.importorskip("rdkit")

from AnyQt.QtWidgets import QApplication
from Orange.data import ContinuousVariable, Domain, StringVariable, Table

from chem_inf_widgets.widgets import ow_chemical_series_explorer as series_widget_module
from chem_inf_widgets.widgets.ow_chemical_series_explorer import OWChemicalSeriesExplorer

_APP = QApplication.instance() or QApplication([])


def _demo_table() -> Table:
    domain = Domain(
        [ContinuousVariable("activity")],
        metas=[StringVariable("SMILES"), StringVariable("Name")],
    )
    return Table.from_numpy(
        domain,
        X=np.asarray([[1.1], [2.2], [3.3], [4.4], [5.5], [6.6]], dtype=float),
        metas=np.asarray(
            [
                ["Cc1ccccc1", "toluene"],
                ["Oc1ccccc1", "phenol"],
                ["CC1CCCCC1", "ethyl_cyclohexane"],
                ["OC1CCCCC1", "cyclohexanol"],
                ["CCO", "ethanol"],
                ["not-a-smiles", "broken"],
            ],
            dtype=object,
        ),
    )


def _table_without_smiles() -> Table:
    return Table.from_numpy(
        Domain([ContinuousVariable("activity")]),
        X=np.asarray([[1.0], [2.0]], dtype=float),
    )


def _patch_outputs(widget: OWChemicalSeriesExplorer, sent: list[tuple[str, object]]):
    return [
        patch.object(widget.Outputs.series_table, "send", lambda value: sent.append(("series", value))),
        patch.object(widget.Outputs.members_table, "send", lambda value: sent.append(("members", value))),
        patch.object(widget.Outputs.summary_table, "send", lambda value: sent.append(("summary", value))),
        patch.object(widget.Outputs.selected_data, "send", lambda value: sent.append(("selected", value))),
        patch.object(widget.Outputs.rgroup_table, "send", lambda value: sent.append(("rgroup", value))),
    ]


def test_chemical_series_widget_runs_and_populates_views():
    widget = OWChemicalSeriesExplorer()
    sent: list[tuple[str, object]] = []

    with ExitStack() as stack:
        for output_patch in _patch_outputs(widget, sent):
            stack.enter_context(output_patch)
        widget.set_data(_demo_table())
        _APP.processEvents()

    assert [name for name, _value in sent[:5]] == ["series", "members", "summary", "selected", "rgroup"]
    assert all(value is not None for _name, value in sent[:5])
    assert "Chemical Series Explorer" in widget._report_browser.toHtml()
    assert widget._series_table_widget.rowCount() >= 1
    assert widget._members_table_widget.rowCount() >= 1
    assert widget._rgroup_table_widget.rowCount() >= 1
    assert widget._target_combo.count() >= 2
    assert "Done:" in widget._status_label.text()

    widget.onDeleteWidget()
    widget.close()


def test_chemical_series_widget_updates_selected_data_for_series_row():
    widget = OWChemicalSeriesExplorer()
    sent: list[tuple[str, object]] = []

    with ExitStack() as stack:
        for output_patch in _patch_outputs(widget, sent):
            stack.enter_context(output_patch)
        widget.set_data(_demo_table())
        _APP.processEvents()

        scaffold_row = None
        scaffold_column = None
        for column_index in range(widget._series_table_widget.columnCount()):
            header_item = widget._series_table_widget.horizontalHeaderItem(column_index)
            if header_item is not None and header_item.text() == "scaffold":
                scaffold_column = column_index
                break

        assert scaffold_column is not None
        for row_index in range(widget._series_table_widget.rowCount()):
            item = widget._series_table_widget.item(row_index, scaffold_column)
            if item is not None and item.text() == "c1ccccc1":
                scaffold_row = row_index
                break

        assert scaffold_row is not None
        widget._series_table_widget.setCurrentCell(scaffold_row, scaffold_column)
        widget._series_table_widget.selectRow(scaffold_row)
        _APP.processEvents()

    selected_outputs = [value for name, value in sent if name == "selected" and value is not None]
    assert selected_outputs
    last_selected = selected_outputs[-1]
    assert len(last_selected) == 2
    selected_names = {str(row["Name"]) for row in last_selected}
    assert selected_names == {"toluene", "phenol"}
    rgroup_outputs = [value for name, value in sent if name == "rgroup" and value is not None]
    assert rgroup_outputs
    assert len(rgroup_outputs[-1]) == 2
    assert widget._rgroup_status_label.text().startswith("Core:")

    widget.onDeleteWidget()
    widget.close()


def test_chemical_series_widget_surfaces_service_errors():
    widget = OWChemicalSeriesExplorer()
    warnings: list[str] = []

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                series_widget_module,
                "set_widget_warning",
                lambda _w, message: warnings.append(str(message or "")),
            )
        )
        widget.set_data(_table_without_smiles())
        _APP.processEvents()

    assert warnings
    assert warnings[-1].startswith("1 chemical series explorer issue(s).")

    widget.onDeleteWidget()
    widget.close()


def test_chemical_series_widget_clears_outputs_without_input():
    widget = OWChemicalSeriesExplorer()
    sent: list[tuple[str, object]] = []

    with ExitStack() as stack:
        for output_patch in _patch_outputs(widget, sent):
            stack.enter_context(output_patch)
        widget.set_data(None)
        _APP.processEvents()

    assert sent == [("series", None), ("members", None), ("summary", None), ("selected", None), ("rgroup", None)]
    assert widget._report_browser.toPlainText() == ""
    assert widget._series_table_widget.rowCount() == 0
    assert widget._members_table_widget.rowCount() == 0
    assert widget._rgroup_table_widget.rowCount() == 0

    widget.onDeleteWidget()
    widget.close()

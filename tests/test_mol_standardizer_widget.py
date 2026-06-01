# ruff: noqa: I001

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication
from Orange.data import Domain, StringVariable, Table

from chem_inf_widgets.chemcore import molecule_contract as molecule_contract_module
from chem_inf_widgets.widgets import ow_mol_standardizer as standardizer_widget_module
from chem_inf_widgets.widgets.ow_mol_standardizer import OWMolStandardizer


_APP = QApplication.instance() or QApplication([])


def _noop(_value=None):
    return None


def _patch_output_sends(widget: OWMolStandardizer):
    return [
        patch.object(widget.Outputs.modeling_data, "send", _noop),
        patch.object(widget.Outputs.data, "send", _noop),
        patch.object(widget.Outputs.molecules, "send", _noop),
        patch.object(widget.Outputs.qsar_ready_data, "send", _noop),
        patch.object(widget.Outputs.qsar_ready_molecules, "send", _noop),
        patch.object(widget.Outputs.standardization_failed_data, "send", _noop),
        patch.object(widget.Outputs.standardization_failed_molecules, "send", _noop),
        patch.object(widget.Outputs.standardization_report, "send", _noop),
        patch.object(widget.Outputs.curation_summary, "send", _noop),
    ]


def test_mol_standardizer_keeps_output_molecule_and_reports_contract_warning():
    widget = OWMolStandardizer()

    with patch.object(
        molecule_contract_module,
        "ensure_contract_props",
        side_effect=RuntimeError("contract boom"),
    ):
        mols, warnings = widget._chemmols_from_smiles(["CCO"], ["No changes"])

    assert len(mols) == 1
    assert warnings == ["Could not attach contract metadata for standardized molecule row 1: contract boom"]
    assert mols[0].get_prop("SMILES") == "CCO"

    widget.onDeleteWidget()
    widget.close()


def test_mol_standardizer_widget_surfaces_runtime_warning():
    widget = OWMolStandardizer()
    warnings: list[str] = []
    payload = (
        None,
        [],
        None,
        None,
        [],
        None,
        [],
        None,
        None,
        ["Could not attach contract metadata for standardized molecule row 1: contract boom"],
        1,
        1,
    )

    with ExitStack() as stack:
        stack.enter_context(patch.object(widget, "progressBarFinished", lambda: None))
        for output_patch in _patch_output_sends(widget):
            stack.enter_context(output_patch)
        stack.enter_context(
            patch.object(
                standardizer_widget_module,
                "set_widget_warning",
                lambda _w, message: warnings.append(message or ""),
            )
        )
        widget._apply_outputs(payload)
        _APP.processEvents()

    assert warnings
    assert warnings[-1] == "Could not attach contract metadata for standardized molecule row 1: contract boom"

    widget.onDeleteWidget()
    widget.close()


def test_mol_standardizer_set_data_warns_when_table_preparse_fails():
    widget = OWMolStandardizer()
    warnings: list[str] = []
    table = Table.from_list(Domain([], metas=[StringVariable("SMILES")]), [["CCO"]])

    with (
        patch.object(
            standardizer_widget_module,
            "table_to_chemmols_with_report",
            side_effect=RuntimeError("preparse boom"),
        ),
        patch.object(
            standardizer_widget_module,
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


def test_mol_standardizer_no_input_clears_all_outputs():
    widget = OWMolStandardizer()
    sent: list[tuple[str, object]] = []

    with ExitStack() as stack:
        stack.enter_context(patch.object(widget.Outputs.modeling_data, "send", lambda value: sent.append(("modeling_data", value))))
        stack.enter_context(patch.object(widget.Outputs.data, "send", lambda value: sent.append(("data", value))))
        stack.enter_context(patch.object(widget.Outputs.molecules, "send", lambda value: sent.append(("molecules", value))))
        stack.enter_context(patch.object(widget.Outputs.qsar_ready_data, "send", lambda value: sent.append(("qsar_ready_data", value))))
        stack.enter_context(patch.object(widget.Outputs.qsar_ready_molecules, "send", lambda value: sent.append(("qsar_ready_molecules", value))))
        stack.enter_context(patch.object(widget.Outputs.standardization_failed_data, "send", lambda value: sent.append(("failed_data", value))))
        stack.enter_context(patch.object(widget.Outputs.standardization_failed_molecules, "send", lambda value: sent.append(("failed_molecules", value))))
        stack.enter_context(patch.object(widget.Outputs.standardization_report, "send", lambda value: sent.append(("report", value))))
        stack.enter_context(patch.object(widget.Outputs.curation_summary, "send", lambda value: sent.append(("curation", value))))
        widget._on_run()
        _APP.processEvents()

    assert sent == [
        ("modeling_data", None),
        ("data", None),
        ("molecules", []),
        ("qsar_ready_data", None),
        ("qsar_ready_molecules", []),
        ("failed_data", None),
        ("failed_molecules", []),
        ("report", None),
        ("curation", None),
    ]

    widget.onDeleteWidget()
    widget.close()

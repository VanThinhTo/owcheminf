from __future__ import annotations

import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")
pytest.importorskip("rdkit")

from AnyQt.QtWidgets import QApplication

from chem_inf_widgets.chemcore.services.orange_table_utils import records_to_orange_table
from chem_inf_widgets.chemcore.services import mol_depict
from chem_inf_widgets.widgets.ow_pair_viewer import OWPairViewer, _mol_pixmap


_APP = QApplication.instance() or QApplication([])


def _standardization_table():
    records = [
        {
            "compound_id": "cmpd-1",
            "SMILES": "CCO.Cl",
            "SMILES_STD": "CCO",
            "standardization_input_smiles": "CCO.Cl",
            "standardization_output_smiles": "CCO",
            "standardization_changed": "1",
        },
        {
            "compound_id": "cmpd-2",
            "SMILES": "CCN",
            "SMILES_STD": "CCN",
            "standardization_input_smiles": "CCN",
            "standardization_output_smiles": "CCN",
            "standardization_changed": "0",
        },
    ]
    return records_to_orange_table(
        records,
        meta_columns=list(records[0].keys()),
        name="Standardization Compare",
    )


def _canonicalization_table():
    records = [
        {
            "compound_id": "cmpd-can-1",
            "input_smiles": "C(C)O",
            "canonical_smiles": "CCO",
        },
        {
            "compound_id": "cmpd-can-2",
            "input_smiles": "N(C)C",
            "canonical_smiles": "CNC",
        },
    ]
    return records_to_orange_table(
        records,
        meta_columns=list(records[0].keys()),
        name="Canonicalization Compare",
    )


def _numbered_standardization_table():
    records = [
        {
            "compound_id": "cmpd-num-1",
            "standardization_input_smiles_2": "CCO.Cl",
            "SMILES_STD_2": "CCO",
        },
        {
            "compound_id": "cmpd-num-2",
            "standardization_input_smiles_2": "CCN.Cl",
            "SMILES_STD_2": "CCN",
        },
    ]
    return records_to_orange_table(
        records,
        meta_columns=list(records[0].keys()),
        name="Repeated Standardization Compare",
    )


def test_pair_viewer_auto_detects_standardization_columns():
    widget = OWPairViewer()
    try:
        widget.set_data(_standardization_table())
        _APP.processEvents()

        assert widget.smiles_a_var_name == "standardization_input_smiles"
        assert widget.smiles_b_var_name == "SMILES_STD"
        assert widget.name_a_var_name == "compound_id"
        assert widget.name_b_var_name == "compound_id"
        assert widget.panel_a.title_label.text() == "Original Structure"
        assert widget.panel_b.title_label.text() == "Standardized Structure"
        assert widget.pair_list.item(0).text() == "1. cmpd-1"
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_pair_viewer_auto_detects_input_to_canonical_comparison():
    widget = OWPairViewer()
    try:
        widget.set_data(_canonicalization_table())
        _APP.processEvents()

        assert widget.smiles_a_var_name == "input_smiles"
        assert widget.smiles_b_var_name == "canonical_smiles"
        assert widget.panel_a.title_label.text() == "Input Structure"
        assert widget.panel_b.title_label.text() == "Output Structure"
        assert widget.pair_list.item(0).text() == "1. cmpd-can-1"
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_pair_viewer_matches_numbered_standardization_audit_columns():
    widget = OWPairViewer()
    try:
        widget.set_data(_numbered_standardization_table())
        _APP.processEvents()

        assert widget.smiles_a_var_name == "standardization_input_smiles_2"
        assert widget.smiles_b_var_name == "SMILES_STD_2"
        assert widget.panel_a.title_label.text() == "Original Structure"
        assert widget.panel_b.title_label.text() == "Standardized Structure"
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_pair_viewer_shows_original_and_standardized_smiles_side_by_side():
    widget = OWPairViewer()
    try:
        widget.set_data(_standardization_table())
        widget.pair_list.setCurrentRow(0)
        _APP.processEvents()

        assert widget.panel_a.smiles_label.text() == "SMILES: CCO.Cl"
        assert widget.panel_b.smiles_label.text() == "SMILES: CCO"
        assert not widget.panel_a.activity_label.isVisible()
        assert not widget.panel_b.activity_label.isVisible()
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_pair_viewer_uses_display_cleanup_for_isolated_metal_radicals(monkeypatch):
    calls = []
    original = mol_depict.render_mol_png

    def _spy_render(mol, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(mol, *args, **kwargs)

    monkeypatch.setattr(mol_depict, "render_mol_png", _spy_render)

    pixmap = _mol_pixmap("CC[O-].[Fe+]")

    assert pixmap is not None
    assert calls
    assert calls[-1].get("suppress_isolated_metal_radicals") is True

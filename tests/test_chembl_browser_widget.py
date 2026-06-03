from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication
from Orange.data import Domain, StringVariable, Table

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.chembl_models import ChemBLMoleculeRecord
from chem_inf_widgets.widgets.ow_chembl_browser import OWChemBLBrowser


_APP = QApplication.instance() or QApplication([])


class _DoneFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _PendingFuture:
    def add_done_callback(self, _callback):
        return None


def _demo_output_table() -> Table:
    smiles_var = StringVariable("SMILES")
    smiles_var.attributes["format"] = "SMILES"
    domain = Domain([], metas=[StringVariable("molecule_chembl_id"), smiles_var])
    return Table.from_numpy(
        domain,
        X=np.empty((1, 0), dtype=float),
        metas=np.asarray([["CHEMBL1", "CCO"]], dtype=object),
    )


def test_chembl_browser_molecule_outputs_ready_sends_default_outputs():
    widget = OWChemBLBrowser()
    sent: list[tuple[str, object]] = []
    table = _demo_output_table()
    molecules = [ChemMol.from_smiles("CCO", name="CHEMBL1")]

    with ExitStack() as stack:
        stack.enter_context(patch.object(widget.Outputs.data, "send", lambda value: sent.append(("data", value))))
        stack.enter_context(patch.object(widget.Outputs.molecules, "send", lambda value: sent.append(("molecules", value))))
        widget._on_molecule_outputs_ready(_DoneFuture((table, molecules, "")))
        _APP.processEvents()

    assert sent == [("data", table), ("molecules", molecules)]
    assert widget._last_table is table
    assert widget._last_molecules == molecules
    assert widget.lbl_status.text() == "Ready: 1 rows, 1 molecules."

    widget.onDeleteWidget()
    widget.close()


def test_chembl_browser_update_molecules_sends_quick_outputs_before_enrichment():
    widget = OWChemBLBrowser()
    sent: list[tuple[str, object]] = []
    molecules = [ChemBLMoleculeRecord(chembl_id="CHEMBL1", pref_name="demo", canonical_smiles="CCO")]

    with ExitStack() as stack:
        stack.enter_context(patch.object(widget.Outputs.data, "send", lambda value: sent.append(("data", value))))
        stack.enter_context(patch.object(widget.Outputs.molecules, "send", lambda value: sent.append(("molecules", value))))
        stack.enter_context(patch.object(widget.executor, "submit", lambda *args, **kwargs: _PendingFuture()))
        widget._update_molecules(molecules, 42)
        _APP.processEvents()

    assert [name for name, _value in sent] == ["data", "molecules"]
    quick_table = sent[0][1]
    quick_molecules = sent[1][1]
    assert quick_table is not None
    assert len(quick_table) == 1
    assert quick_table.domain.metas[0].name == "ChEMBL ID"
    assert len(quick_molecules) == 1
    assert quick_molecules[0].name == "CHEMBL1"
    assert widget.lbl_status.text() == "Loaded 1 molecules in 42 ms."

    widget.onDeleteWidget()
    widget.close()

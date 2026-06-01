from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication
from Orange.data import Domain, Table

from chem_inf_widgets.chemcore.result import ServiceIssue
from chem_inf_widgets.chemcore.services.molecule_import_service import (
    MoleculeImportResult,
    MoleculeImportSummary,
)
from chem_inf_widgets.widgets import ow_molecule_import_hub as import_widget_module
from chem_inf_widgets.widgets.ow_molecule_import_hub import OWMoleculeImportHub


_APP = QApplication.instance() or QApplication([])


def _noop(_value=None):
    return None


def _patch_output_sends(widget: OWMoleculeImportHub):
    return [
        patch.object(widget.Outputs.data, "send", _noop),
        patch.object(widget.Outputs.molecules, "send", _noop),
        patch.object(widget.Outputs.accepted_data, "send", _noop),
        patch.object(widget.Outputs.accepted_molecules, "send", _noop),
        patch.object(widget.Outputs.rejected_records, "send", _noop),
        patch.object(widget.Outputs.import_report, "send", _noop),
        patch.object(widget.Outputs.failed_records, "send", _noop),
        patch.object(widget.Outputs.import_summary, "send", _noop),
        patch.object(widget.Outputs.curation_summary, "send", _noop),
    ]


def _demo_table() -> Table:
    return Table.from_numpy(Domain([]), X=np.empty((1, 0), dtype=float))


def _result_with_issues(*, issues: list[ServiceIssue]) -> MoleculeImportResult:
    return MoleculeImportResult(
        mols=[],
        records=[],
        summary=MoleculeImportSummary(
            source_path="demo.csv",
            source_format="table",
            total_records=1,
            valid_records=1,
            failed_records=0,
            accepted_records=1,
            rejected_records=0,
            duplicate_groups=0,
            duplicate_records=0,
            smiles_column="smiles",
            name_column="name",
            columns=["name", "smiles"],
        ),
        issues=issues,
    )


def test_molecule_import_hub_surfaces_backend_service_warnings():
    widget = OWMoleculeImportHub()
    warnings: list[str] = []
    table = _demo_table()
    payload = (
        table,
        [],
        table,
        [],
        table,
        table,
        table,
        table,
        table,
        _result_with_issues(
            issues=[
                ServiceIssue(
                    code="delimiter_detection_failed",
                    message="Could not auto-detect delimiter for 'demo.csv': sniff boom. Falling back to ','.",
                    severity="warning",
                )
            ]
        ),
    )

    with ExitStack() as stack:
        stack.enter_context(patch.object(widget, "progressBarFinished", lambda: None))
        for output_patch in _patch_output_sends(widget):
            stack.enter_context(output_patch)
        stack.enter_context(
            patch.object(import_widget_module, "set_widget_warning", lambda _w, message: warnings.append(message or ""))
        )
        widget._apply_outputs(payload)
        _APP.processEvents()

    assert warnings
    assert warnings[-1] == "1 import warning(s). Could not auto-detect delimiter for 'demo.csv': sniff boom. Falling back to ','."

    widget.onDeleteWidget()
    widget.close()


def test_molecule_import_hub_clears_warning_after_clean_run():
    widget = OWMoleculeImportHub()
    warnings: list[str] = []
    table = _demo_table()

    with ExitStack() as stack:
        stack.enter_context(patch.object(widget, "progressBarFinished", lambda: None))
        for output_patch in _patch_output_sends(widget):
            stack.enter_context(output_patch)
        stack.enter_context(
            patch.object(import_widget_module, "set_widget_warning", lambda _w, message: warnings.append(message or ""))
        )
        widget._apply_outputs(
            (
                table,
                [],
                table,
                [],
                table,
                table,
                table,
                table,
                table,
                _result_with_issues(issues=[]),
            )
        )
        _APP.processEvents()

    assert warnings
    assert warnings[-1] == ""

    widget.onDeleteWidget()
    widget.close()

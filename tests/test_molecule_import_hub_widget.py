# ruff: noqa: I001

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
from chem_inf_widgets.widgets.utils import summarize_service_issues
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
            patch.object(
                import_widget_module,
                "show_service_issues",
                lambda _w, issues, subject="service", issue_label="warning": warnings.append(
                    summarize_service_issues(issues, subject=subject, issue_label=issue_label)
                ),
            )
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
            patch.object(
                import_widget_module,
                "show_service_issues",
                lambda _w, issues, subject="service", issue_label="warning": warnings.append(
                    summarize_service_issues(issues, subject=subject, issue_label=issue_label)
                ),
            )
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


def test_molecule_import_hub_send_empty_clears_outputs_and_role_summary():
    widget = OWMoleculeImportHub()
    sent: list[tuple[str, object]] = []
    widget.roles_label.setText("Something loaded")

    with ExitStack() as stack:
        stack.enter_context(patch.object(widget.Outputs.data, "send", lambda value: sent.append(("data", value))))
        stack.enter_context(patch.object(widget.Outputs.molecules, "send", lambda value: sent.append(("molecules", value))))
        stack.enter_context(patch.object(widget.Outputs.accepted_data, "send", lambda value: sent.append(("accepted_data", value))))
        stack.enter_context(patch.object(widget.Outputs.accepted_molecules, "send", lambda value: sent.append(("accepted_molecules", value))))
        stack.enter_context(patch.object(widget.Outputs.rejected_records, "send", lambda value: sent.append(("rejected_records", value))))
        stack.enter_context(patch.object(widget.Outputs.import_report, "send", lambda value: sent.append(("import_report", value))))
        stack.enter_context(patch.object(widget.Outputs.failed_records, "send", lambda value: sent.append(("failed_records", value))))
        stack.enter_context(patch.object(widget.Outputs.import_summary, "send", lambda value: sent.append(("import_summary", value))))
        stack.enter_context(patch.object(widget.Outputs.curation_summary, "send", lambda value: sent.append(("curation_summary", value))))
        stack.enter_context(patch.object(import_widget_module, "set_widget_warning", lambda _w, message: None))
        widget._send_empty()
        _APP.processEvents()

    assert sent == [
        ("data", None),
        ("molecules", []),
        ("accepted_data", None),
        ("accepted_molecules", []),
        ("rejected_records", None),
        ("import_report", None),
        ("failed_records", None),
        ("import_summary", None),
        ("curation_summary", None),
    ]
    assert widget.roles_label.text() == "Import a file to inspect attributes, class variables, and metas."

    widget.onDeleteWidget()
    widget.close()

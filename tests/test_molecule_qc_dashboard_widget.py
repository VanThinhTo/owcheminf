# ruff: noqa: I001

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication

from chem_inf_widgets.chemcore.result import ServiceIssue
from chem_inf_widgets.chemcore.services.molecule_qc_service import (
    MoleculeQCResult,
    MoleculeQCSummary,
)
from chem_inf_widgets.widgets import ow_molecule_qc_dashboard as qc_widget_module
from chem_inf_widgets.widgets.utils import summarize_service_issues
from chem_inf_widgets.widgets.ow_molecule_qc_dashboard import OWMoleculeQCDashboard


_APP = QApplication.instance() or QApplication([])


def _noop(_value=None):
    return None


def _patch_output_sends(widget: OWMoleculeQCDashboard):
    return [
        patch.object(widget.Outputs.modeling_data, "send", _noop),
        patch.object(widget.Outputs.annotated_data, "send", _noop),
        patch.object(widget.Outputs.clean_data, "send", _noop),
        patch.object(widget.Outputs.problem_data, "send", _noop),
        patch.object(widget.Outputs.rejected_data, "send", _noop),
        patch.object(widget.Outputs.qc_report, "send", _noop),
        patch.object(widget.Outputs.qc_summary, "send", _noop),
        patch.object(widget.Outputs.curation_summary, "send", _noop),
        patch.object(widget.Outputs.annotated_molecules, "send", _noop),
        patch.object(widget.Outputs.clean_molecules, "send", _noop),
        patch.object(widget.Outputs.problem_molecules, "send", _noop),
        patch.object(widget.Outputs.rejected_molecules, "send", _noop),
    ]


def _result_with_issues(*, issues: list[ServiceIssue]) -> MoleculeQCResult:
    return MoleculeQCResult(
        records=[],
        clean_indices=[],
        problem_indices=[],
        summary=MoleculeQCSummary(
            total=1,
            clean=1,
            problem=0,
            invalid=0,
            warnings=0,
            errors=0,
            duplicate_groups=0,
            duplicate_records=0,
            issue_counts={},
        ),
        issues=issues,
    )


def test_molecule_qc_dashboard_surfaces_backend_service_warnings():
    widget = OWMoleculeQCDashboard()
    warnings: list[str] = []
    payload = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        [],
        [],
        [],
        [],
        _result_with_issues(
            issues=[
                ServiceIssue(
                    code="molecular_weight_computation_failed",
                    message="Could not compute molecular weight: mw boom",
                    severity="warning",
                    row_index=1,
                )
            ]
        ),
    )

    with ExitStack() as stack:
        stack.enter_context(patch.object(widget, "progressBarFinished", lambda: None))
        for output_patch in _patch_output_sends(widget):
            stack.enter_context(output_patch)
        stack.enter_context(
            patch.object(qc_widget_module, "show_service_issues", lambda _w, issues, subject="service", issue_label="warning": warnings.append(
                summarize_service_issues(issues, subject=subject, issue_label=issue_label)
            ))
        )
        widget._apply_outputs(payload)
        _APP.processEvents()

    assert warnings
    assert warnings[-1] == "1 QC backend warning(s). Row 1. Could not compute molecular weight: mw boom"

    widget.onDeleteWidget()
    widget.close()


def test_molecule_qc_dashboard_clears_backend_warning_after_clean_run():
    widget = OWMoleculeQCDashboard()
    warnings: list[str] = []

    with ExitStack() as stack:
        stack.enter_context(patch.object(widget, "progressBarFinished", lambda: None))
        for output_patch in _patch_output_sends(widget):
            stack.enter_context(output_patch)
        stack.enter_context(
            patch.object(qc_widget_module, "show_service_issues", lambda _w, issues, subject="service", issue_label="warning": warnings.append(
                summarize_service_issues(issues, subject=subject, issue_label=issue_label)
            ))
        )
        widget._apply_outputs(
            (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                [],
                [],
                [],
                [],
                _result_with_issues(issues=[]),
            )
        )
        _APP.processEvents()

    assert warnings
    assert warnings[-1] == ""

    widget.onDeleteWidget()
    widget.close()

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

from chem_inf_widgets.widgets import ow_dataset_profiler as profiler_widget_module
from chem_inf_widgets.widgets.ow_dataset_profiler import OWDatasetProfiler
from chem_inf_widgets.widgets.utils import summarize_service_issues

_APP = QApplication.instance() or QApplication([])


def _demo_table() -> Table:
    domain = Domain(
        [ContinuousVariable("activity")],
        metas=[StringVariable("SMILES"), StringVariable("Name")],
    )
    return Table.from_numpy(
        domain,
        X=np.asarray([[1.0], [2.0], [np.nan]], dtype=float),
        metas=np.asarray(
            [
                ["CCO", "ethanol_a"],
                ["OCC", "ethanol_b"],
                ["not-a-smiles", "broken"],
            ],
            dtype=object,
        ),
    )


def _patch_outputs(widget: OWDatasetProfiler, sent: list[tuple[str, object]]):
    return [
        patch.object(widget.Outputs.profiled_table, "send", lambda value: sent.append(("profiled", value))),
        patch.object(widget.Outputs.problematic_compounds, "send", lambda value: sent.append(("problematic", value))),
        patch.object(widget.Outputs.summary_table, "send", lambda value: sent.append(("summary", value))),
    ]


def test_dataset_profiler_widget_runs_and_populates_views():
    widget = OWDatasetProfiler()
    sent: list[tuple[str, object]] = []

    with ExitStack() as stack:
        for output_patch in _patch_outputs(widget, sent):
            stack.enter_context(output_patch)
        widget.set_data(_demo_table())
        _APP.processEvents()

    assert [name for name, _value in sent] == ["profiled", "problematic", "summary"]
    assert all(value is not None for _name, value in sent)
    assert "Dataset Profile" in widget._report_browser.toHtml()
    assert widget._descriptor_table.rowCount() >= 1
    assert widget._problem_table.rowCount() >= 1
    assert "Done:" in widget._status_label.text()

    widget.onDeleteWidget()
    widget.close()


def test_dataset_profiler_widget_surfaces_service_issues():
    widget = OWDatasetProfiler()
    warnings: list[str] = []
    table = Table.from_numpy(
        Domain([ContinuousVariable("activity")]),
        X=np.asarray([[1.0], [2.0]], dtype=float),
    )

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                profiler_widget_module,
                "show_service_issues",
                lambda _w, issues, subject="service", issue_label="warning": warnings.append(
                    summarize_service_issues(
                        issues,
                        subject=subject,
                        issue_label=issue_label,
                    )
                ),
            )
        )
        widget.set_data(table)
        _APP.processEvents()

    assert warnings
    assert warnings[-1].startswith("1 dataset profiler issue(s).")

    widget.onDeleteWidget()
    widget.close()


def test_dataset_profiler_widget_clears_outputs_without_input():
    widget = OWDatasetProfiler()
    sent: list[tuple[str, object]] = []

    with ExitStack() as stack:
        for output_patch in _patch_outputs(widget, sent):
            stack.enter_context(output_patch)
        widget.set_data(None)
        _APP.processEvents()

    assert sent == [("profiled", None), ("problematic", None), ("summary", None)]
    assert widget._report_browser.toPlainText() == ""
    assert widget._descriptor_table.rowCount() == 0
    assert widget._problem_table.rowCount() == 0

    widget.onDeleteWidget()
    widget.close()

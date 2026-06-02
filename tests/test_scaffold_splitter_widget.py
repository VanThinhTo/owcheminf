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

from chem_inf_widgets.widgets import ow_scaffold_splitter as splitter_widget_module
from chem_inf_widgets.widgets.ow_scaffold_splitter import OWScaffoldSplitter
from chem_inf_widgets.widgets.utils import summarize_service_issues

_APP = QApplication.instance() or QApplication([])


def _random_table() -> Table:
    domain = Domain([ContinuousVariable("x1")])
    return Table.from_numpy(domain, X=np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float))


def _scaffold_table() -> Table:
    domain = Domain(
        [ContinuousVariable("activity")],
        metas=[StringVariable("SMILES"), StringVariable("Name")],
    )
    return Table.from_list(
        domain,
        [
            [5.1, "c1ccccc1O", "phenol"],
            [5.2, "c1ccccc1N", "aniline"],
            [5.3, "CCO", "ethanol"],
            [5.4, "CCCO", "propanol"],
        ],
    )


def _targetless_table() -> Table:
    domain = Domain([], metas=[StringVariable("SMILES")])
    return Table.from_list(domain, [["CCO"], ["CCN"], ["CCC"]])


def _patch_outputs(widget: OWScaffoldSplitter, sent: list[tuple[str, object]]):
    return [
        patch.object(widget.Outputs.train_data, "send", lambda value: sent.append(("train", value))),
        patch.object(widget.Outputs.validation_data, "send", lambda value: sent.append(("validation", value))),
        patch.object(widget.Outputs.test_data, "send", lambda value: sent.append(("test", value))),
        patch.object(widget.Outputs.summary, "send", lambda value: sent.append(("summary", value))),
    ]


def test_scaffold_splitter_random_mode_works_without_smiles_column():
    widget = OWScaffoldSplitter()
    sent: list[tuple[str, object]] = []
    widget.auto_run_check.setChecked(False)
    widget.method_combo.setCurrentIndex(1)

    with ExitStack() as stack:
        for output_patch in _patch_outputs(widget, sent):
            stack.enter_context(output_patch)
        widget.set_data(_random_table())
        widget.commit()
        _APP.processEvents()

    assert [name for name, _value in sent] == ["train", "validation", "test", "summary"]
    assert all(value is not None for _name, value in sent)
    assert "Done:" in widget.status_label.text()

    widget.onDeleteWidget()
    widget.close()


def test_scaffold_splitter_default_scaffold_mode_still_succeeds():
    widget = OWScaffoldSplitter()
    sent: list[tuple[str, object]] = []
    widget.auto_run_check.setChecked(False)

    with ExitStack() as stack:
        for output_patch in _patch_outputs(widget, sent):
            stack.enter_context(output_patch)
        widget.set_data(_scaffold_table())
        widget.commit()
        _APP.processEvents()

    summary = sent[-1][1]
    assert summary is not None
    assert "Done:" in widget.status_label.text()

    widget.onDeleteWidget()
    widget.close()


def test_scaffold_splitter_stratified_mode_warns_without_target():
    widget = OWScaffoldSplitter()
    sent: list[tuple[str, object]] = []
    warnings: list[str] = []
    widget.auto_run_check.setChecked(False)
    widget.method_combo.setCurrentIndex(2)

    with ExitStack() as stack:
        for output_patch in _patch_outputs(widget, sent):
            stack.enter_context(output_patch)
        stack.enter_context(
            patch.object(
                splitter_widget_module,
                "show_service_issues",
                lambda _w, issues, subject="service", issue_label="warning": warnings.append(
                    summarize_service_issues(issues, subject=subject, issue_label=issue_label)
                ),
            )
        )
        widget.set_data(_targetless_table())
        widget.commit()
        _APP.processEvents()

    assert sent == [("train", None), ("validation", None), ("test", None), ("summary", None)]
    assert warnings
    assert "target column" in warnings[-1].lower()
    assert widget.status_label.text().startswith("Failed:")

    widget.onDeleteWidget()
    widget.close()

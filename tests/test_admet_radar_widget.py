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

from chem_inf_widgets.widgets import ow_admet_radar as admet_widget_module
from chem_inf_widgets.widgets.ow_admet_radar import OWAdmetRadar
from chem_inf_widgets.widgets.utils import summarize_service_issues

_APP = QApplication.instance() or QApplication([])


def _demo_table() -> Table:
    domain = Domain(
        [ContinuousVariable("activity")],
        metas=[StringVariable("SMILES"), StringVariable("Name")],
    )
    return Table.from_numpy(
        domain,
        X=np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=float),
        metas=np.asarray(
            [
                ["CCO", "ethanol"],
                ["CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", "long_chain"],
                ["c1ccccc1N=Nc2ccccc2", "azo_probe"],
                ["not-a-smiles", "broken"],
            ],
            dtype=object,
        ),
    )


def _table_without_smiles() -> Table:
    return Table.from_numpy(
        Domain([ContinuousVariable("x1")]),
        X=np.asarray([[1.0], [2.0]], dtype=float),
    )


def _patch_outputs(widget: OWAdmetRadar, sent: list[tuple[str, object]]):
    return [
        patch.object(widget.Outputs.admet_table, "send", lambda value: sent.append(("admet", value))),
        patch.object(widget.Outputs.flagged_compounds, "send", lambda value: sent.append(("flagged", value))),
        patch.object(widget.Outputs.summary_table, "send", lambda value: sent.append(("summary", value))),
    ]


def test_admet_radar_widget_runs_and_populates_views():
    widget = OWAdmetRadar()
    sent: list[tuple[str, object]] = []

    with ExitStack() as stack:
        for output_patch in _patch_outputs(widget, sent):
            stack.enter_context(output_patch)
        widget.set_data(_demo_table())
        _APP.processEvents()

    assert [name for name, _value in sent] == ["admet", "flagged", "summary"]
    assert all(value is not None for _name, value in sent)
    assert "ADMET Radar" in widget._report_browser.toHtml()
    assert widget._summary_table_widget.rowCount() >= 1
    assert widget._flagged_table_widget.rowCount() >= 1
    assert "Done:" in widget._status_label.text()

    widget.onDeleteWidget()
    widget.close()


def test_admet_radar_widget_surfaces_service_errors():
    widget = OWAdmetRadar()
    warnings: list[str] = []

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                admet_widget_module,
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
        widget.set_data(_table_without_smiles())
        _APP.processEvents()

    assert warnings
    assert warnings[-1].startswith("1 admet radar issue(s).")

    widget.onDeleteWidget()
    widget.close()


def test_admet_radar_widget_clears_outputs_without_input():
    widget = OWAdmetRadar()
    sent: list[tuple[str, object]] = []

    with ExitStack() as stack:
        for output_patch in _patch_outputs(widget, sent):
            stack.enter_context(output_patch)
        widget.set_data(None)
        _APP.processEvents()

    assert sent == [("admet", None), ("flagged", None), ("summary", None)]
    assert widget._report_browser.toPlainText() == ""
    assert widget._summary_table_widget.rowCount() == 0
    assert widget._flagged_table_widget.rowCount() == 0

    widget.onDeleteWidget()
    widget.close()

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication
from Orange.data import ContinuousVariable, Domain, StringVariable, Table

from chem_inf_widgets.widgets import ow_molecular_space_map as space_widget_module
from chem_inf_widgets.widgets.ow_molecular_space_map import OWMolecularSpaceMap
from chem_inf_widgets.widgets.utils import summarize_service_issues

_APP = QApplication.instance() or QApplication([])


def _demo_table() -> Table:
    domain = Domain(
        [ContinuousVariable("d1"), ContinuousVariable("d2"), ContinuousVariable("d3")],
        metas=[StringVariable("Name")],
    )
    return Table.from_numpy(
        domain,
        X=np.asarray(
            [
                [0.0, 1.0, 0.5],
                [1.0, 0.5, 1.5],
                [2.0, 0.0, 2.5],
                [3.0, 1.5, 3.5],
            ],
            dtype=float,
        ),
        metas=np.asarray([["mol_1"], ["mol_2"], ["mol_3"], ["mol_4"]], dtype=object),
    )


def _no_feature_table() -> Table:
    return Table.from_numpy(
        Domain([], metas=[StringVariable("Name")]),
        X=np.empty((2, 0), dtype=float),
        metas=np.asarray([["mol_1"], ["mol_2"]], dtype=object),
    )


def _patch_outputs(widget: OWMolecularSpaceMap, sent: list[tuple[str, object]]):
    return [
        patch.object(widget.Outputs.coordinates, "send", lambda value: sent.append(("coordinates", value))),
        patch.object(widget.Outputs.summary_table, "send", lambda value: sent.append(("summary", value))),
    ]


def test_molecular_space_widget_runs_and_emits_outputs():
    widget = OWMolecularSpaceMap()
    sent: list[tuple[str, object]] = []

    with ExitStack() as stack:
        for output_patch in _patch_outputs(widget, sent):
            stack.enter_context(output_patch)
        widget.set_data(_demo_table())
        _APP.processEvents()

    assert [name for name, _value in sent] == ["coordinates", "summary"]
    coordinates = sent[0][1]
    assert coordinates is not None
    assert coordinates.X.shape == (4, 2)
    assert coordinates.domain.metas[0].name == "Name"
    assert sent[1][1] is not None
    assert "Molecular Space Map" in widget._summary_browser.toHtml()
    assert widget._coordinates_table_widget.rowCount() == 4
    assert "Done:" in widget._status_label.text()

    widget.onDeleteWidget()
    widget.close()


def test_molecular_space_widget_reports_service_errors():
    widget = OWMolecularSpaceMap()
    sent: list[tuple[str, object]] = []
    warnings: list[str] = []

    with ExitStack() as stack:
        for output_patch in _patch_outputs(widget, sent):
            stack.enter_context(output_patch)
        stack.enter_context(
            patch.object(
                space_widget_module,
                "show_service_issues",
                lambda _w, issues, subject="service", issue_label="warning": warnings.append(
                    summarize_service_issues(issues, subject=subject, issue_label=issue_label)
                ),
            )
        )
        widget.set_data(_no_feature_table())
        _APP.processEvents()

    assert sent[0] == ("coordinates", None)
    assert sent[1][0] == "summary"
    assert warnings
    assert "empty" in warnings[-1].lower()
    assert widget._status_label.text().startswith("Failed:")

    widget.onDeleteWidget()
    widget.close()

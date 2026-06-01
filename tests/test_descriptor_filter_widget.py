# ruff: noqa: I001

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication

from chem_inf_widgets.widgets.ow_descriptor_filter import OWDescriptorFilter


_APP = QApplication.instance() or QApplication([])


def _noop(_value=None):
    return None


def test_descriptor_filter_clears_outputs_when_input_is_missing():
    widget = OWDescriptorFilter()
    sent: list[tuple[str, object]] = []

    with (
        patch.object(widget.Outputs.filtered_data, "send", lambda value: sent.append(("filtered", value))),
        patch.object(widget.Outputs.modeling_data, "send", lambda value: sent.append(("modeling", value))),
        patch.object(widget.Outputs.report, "send", lambda value: sent.append(("report", value))),
    ):
        widget.set_data(None)
        _APP.processEvents()

    assert sent == [
        ("filtered", None),
        ("modeling", None),
        ("report", None),
    ]
    assert widget._lbl_status.text() == "No data."

    widget.onDeleteWidget()
    widget.close()

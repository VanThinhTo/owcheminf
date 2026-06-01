from __future__ import annotations

from unittest.mock import patch

from chem_inf_widgets.chemcore.result import ServiceIssue
from chem_inf_widgets.widgets.utils import (
    combine_messages,
    require_table,
    send_empty,
    send_output_values,
    summarize_service_issues,
)
from chem_inf_widgets.widgets.utils import messages as message_utils


class _DummyWidget:
    def __init__(self) -> None:
        self.status_calls: list[tuple[str, bool]] = []

    def _set_status(self, message: str, ok: bool = False) -> None:
        self.status_calls.append((message, ok))


class _DummyOutput:
    def __init__(self) -> None:
        self.values: list[object] = []

    def send(self, value) -> None:
        self.values.append(value)


def test_require_table_updates_widget_status_for_missing_input():
    widget = _DummyWidget()

    assert require_table(None, widget, message="No input data.") is False
    assert widget.status_calls == [("No input data.", False)]


def test_send_helpers_forward_expected_values():
    output_a = _DummyOutput()
    output_b = _DummyOutput()

    send_empty(output_a)
    send_output_values((output_a, "demo"), (output_b, []))

    assert output_a.values == [None, "demo"]
    assert output_b.values == [[]]


def test_service_issue_helpers_format_and_show_warning():
    issues = [
        ServiceIssue(
            code="demo_failed",
            message="Demo warning",
            severity="warning",
            row_index=3,
        )
    ]
    warnings: list[str] = []

    assert combine_messages("", "first", None, "second") == "first second"
    assert summarize_service_issues(issues, subject="QC backend") == "1 QC backend warning(s). Row 3. Demo warning"

    with patch.object(message_utils, "set_widget_warning", lambda _w, message: warnings.append(message)):
        message_utils.show_service_issues(object(), issues, subject="QC backend")

    assert warnings == ["1 QC backend warning(s). Row 3. Demo warning"]

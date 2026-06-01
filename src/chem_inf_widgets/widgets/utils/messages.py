from __future__ import annotations

from collections.abc import Sequence

from chem_inf_widgets.chemcore.result import ServiceIssue
from chem_inf_widgets.widgets.ui_helpers import set_widget_warning


def combine_messages(*messages: str | None) -> str:
    return " ".join(str(message).strip() for message in messages if str(message or "").strip())


def summarize_service_issues(
    issues: Sequence[ServiceIssue],
    *,
    subject: str = "service",
    issue_label: str = "warning",
) -> str:
    issue_list = list(issues or [])
    if not issue_list:
        return ""
    first = issue_list[0]
    row_text = f" Row {first.row_index}." if first.row_index is not None else ""
    return f"{len(issue_list)} {subject} {issue_label}(s).{row_text} {first.message}".strip()


def show_service_issues(
    widget,
    issues: Sequence[ServiceIssue],
    *,
    subject: str = "service",
    issue_label: str = "warning",
) -> str:
    message = summarize_service_issues(issues, subject=subject, issue_label=issue_label)
    set_widget_warning(widget, message)
    return message

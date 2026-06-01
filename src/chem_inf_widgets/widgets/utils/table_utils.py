from __future__ import annotations

from typing import Any

from Orange.data import Table


def require_table(data: Table | None, widget=None, message: str = "No input data.") -> bool:
    if data is not None:
        return True
    if widget is not None and hasattr(widget, "_set_status"):
        try:
            widget._set_status(message, ok=False)
        except TypeError:
            widget._set_status(message)
    return False


def send_empty(output: Any) -> None:
    output.send(None)


def send_output_values(*pairs: tuple[Any, Any]) -> None:
    for output, value in pairs:
        output.send(value)

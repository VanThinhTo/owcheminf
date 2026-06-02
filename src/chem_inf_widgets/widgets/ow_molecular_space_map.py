from __future__ import annotations

from html import escape

import numpy as np
from AnyQt.QtWidgets import (
    QLabel,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
)
from Orange.data import ContinuousVariable, Domain, Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.molecular_space_service import (
    MolecularSpaceConfig,
    MolecularSpaceResult,
    compute_molecular_space,
)
from chem_inf_widgets.chemcore.services.orange_table_utils import (
    records_to_orange_table,
    safe_table_from_numpy,
)
from chem_inf_widgets.widgets.ui_helpers import (
    clear_widget_messages,
    format_done_status,
    format_failed_status,
    format_loaded_status,
    format_no_input_status,
)
from chem_inf_widgets.widgets.utils import send_output_values, show_service_issues


def _count_card(label: str, value: str) -> str:
    return (
        "<div style='display:inline-block; min-width:150px; margin:6px; padding:10px 12px; "
        "border:1px solid #D0D5DD; border-radius:10px; background:#F8FAFC;'>"
        f"<div style='font-size:12px; color:#475467;'>{escape(label)}</div>"
        f"<div style='font-size:22px; font-weight:600; color:#101828;'>{escape(value)}</div>"
        "</div>"
    )


def _render_summary_html(
    result: MolecularSpaceResult,
    *,
    n_rows: int,
    n_features: int,
) -> str:
    warning_count = sum(1 for issue in result.issues if issue.severity == "warning")
    error_count = sum(1 for issue in result.issues if issue.severity == "error")
    cards = "".join(
        [
            _count_card("Rows", str(n_rows)),
            _count_card("Input features", str(n_features)),
            _count_card("Method", result.method.upper()),
            _count_card("Dimensions", str(result.coordinates.shape[1] if result.coordinates.ndim == 2 else 0)),
            _count_card("Warnings", str(warning_count)),
            _count_card("Errors", str(error_count)),
        ]
    )

    explained = result.explained_variance or []
    explained_html = "".join(
        f"<li>Component {index + 1}: {value * 100:.2f}% variance</li>"
        for index, value in enumerate(explained)
    ) or "<li>Explained variance is not available for this method.</li>"

    issue_html = "".join(
        f"<li>{escape(issue.severity.upper())}: {escape(issue.message)}</li>"
        for issue in result.issues
    ) or "<li>No service issues recorded.</li>"

    return (
        "<html><body style='font-family: sans-serif;'>"
        "<h2>Molecular Space Map</h2>"
        f"{cards}"
        "<h3>Explained variance</h3>"
        f"<ul>{explained_html}</ul>"
        "<h3>Service issues</h3>"
        f"<ul>{issue_html}</ul>"
        "</body></html>"
    )


def _summary_rows(
    result: MolecularSpaceResult,
    *,
    n_rows: int,
    n_features: int,
) -> list[dict[str, str]]:
    rows = [
        {"metric": "method", "value": result.method},
        {"metric": "n_rows", "value": str(n_rows)},
        {"metric": "n_features", "value": str(n_features)},
        {
            "metric": "n_components",
            "value": str(result.coordinates.shape[1] if result.coordinates.ndim == 2 else 0),
        },
        {
            "metric": "warning_count",
            "value": str(sum(1 for issue in result.issues if issue.severity == "warning")),
        },
        {
            "metric": "error_count",
            "value": str(sum(1 for issue in result.issues if issue.severity == "error")),
        },
    ]
    rows.extend(
        {
            "metric": f"explained_variance_{index + 1}",
            "value": f"{value:.6f}",
        }
        for index, value in enumerate(result.explained_variance or [])
    )
    return rows


def _summary_table(
    result: MolecularSpaceResult,
    *,
    n_rows: int,
    n_features: int,
) -> Table | None:
    return records_to_orange_table(
        _summary_rows(result, n_rows=n_rows, n_features=n_features),
        meta_columns=["metric", "value"],
        name="Molecular Space Summary",
    )


def _coordinates_table(data: Table, result: MolecularSpaceResult) -> Table | None:
    coords = np.asarray(result.coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return None

    coordinate_vars = [
        ContinuousVariable(f"Space {index + 1}")
        for index in range(coords.shape[1])
    ]
    domain = Domain(coordinate_vars, class_vars=data.domain.class_vars, metas=data.domain.metas)
    return safe_table_from_numpy(
        domain,
        X=coords,
        Y=data.Y if len(data.domain.class_vars) else None,
        metas=data.metas if len(data.domain.metas) else None,
        name="Molecular Space Coordinates",
    )


def _set_table_rows(widget: QTableWidget, rows: list[dict[str, object]]) -> None:
    if not rows:
        widget.clear()
        widget.setRowCount(0)
        widget.setColumnCount(0)
        return

    headers = list(rows[0].keys())
    widget.setColumnCount(len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        for column_index, header in enumerate(headers):
            item = QTableWidgetItem("" if row.get(header) is None else str(row.get(header)))
            widget.setItem(row_index, column_index, item)
    widget.resizeColumnsToContents()


class OWMolecularSpaceMap(OWWidget):
    name = "Molecular Space Map"
    description = "Project descriptor or fingerprint matrices into a low-dimensional molecular space."
    icon = "icons/modeling/qsar_regression.png"
    priority = 145
    keywords = ["molecular space", "pca", "umap", "embedding", "projection"]
    want_main_area = True

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        coordinates = Output("Coordinates", Table, default=True)
        summary_table = Output("Summary Table", Table)

    auto_run = Setting(True)
    method_idx = Setting(0)
    n_components = Setting(2)
    random_state = Setting(0)

    _METHODS = [("PCA", "pca"), ("UMAP", "umap")]

    def __init__(self) -> None:
        super().__init__()
        self.data: Table | None = None
        self._build_ui()
        self._set_status("Waiting for data.", ok=True)

    def _build_ui(self) -> None:
        options_box = gui.widgetBox(self.controlArea, "Options")
        gui.comboBox(
            options_box,
            self,
            "method_idx",
            label="Method",
            items=[label for label, _method in self._METHODS],
            callback=self._on_settings_changed,
        )

        self._components_spin = QSpinBox()
        self._components_spin.setRange(1, 10)
        self._components_spin.setValue(int(self.n_components))
        self._components_spin.valueChanged.connect(self._on_components_changed)
        gui.indentedBox(options_box).layout().addWidget(QLabel("Components"))
        gui.indentedBox(options_box).layout().addWidget(self._components_spin)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 1_000_000)
        self._seed_spin.setValue(int(self.random_state))
        self._seed_spin.valueChanged.connect(self._on_random_state_changed)
        gui.indentedBox(options_box).layout().addWidget(QLabel("Random seed"))
        gui.indentedBox(options_box).layout().addWidget(self._seed_spin)

        gui.checkBox(
            options_box,
            self,
            "auto_run",
            "Auto-run on input changes",
            callback=self._on_settings_changed,
        )
        gui.button(options_box, self, "Compute space", callback=self.commit)

        self._status_label = QLabel("Waiting for data.")
        self._status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self._status_label)

        tabs = QTabWidget()
        self.mainArea.layout().addWidget(tabs)

        self._summary_browser = QTextBrowser()
        self._summary_browser.setOpenExternalLinks(False)
        tabs.addTab(self._summary_browser, "Summary")

        self._coordinates_table_widget = QTableWidget()
        tabs.addTab(self._coordinates_table_widget, "Coordinates")

    def _set_status(self, text: str, ok: bool = False) -> None:
        color = "#027A48" if ok else "#475467"
        self._status_label.setStyleSheet(f"color:{color};")
        self._status_label.setText(text)

    def _config(self) -> MolecularSpaceConfig:
        return MolecularSpaceConfig(
            method=self._METHODS[int(self.method_idx)][1],
            n_components=int(self.n_components),
            random_state=int(self.random_state),
        )

    def _on_components_changed(self, value: int) -> None:
        self.n_components = int(value)
        self._on_settings_changed()

    def _on_random_state_changed(self, value: int) -> None:
        self.random_state = int(value)
        self._on_settings_changed()

    def _on_settings_changed(self) -> None:
        if self.auto_run and self.data is not None:
            self.commit()

    @Inputs.data
    def set_data(self, data: Table | None) -> None:
        self.data = data
        if data is None:
            self._send_empty()
            self._set_status(format_no_input_status(), ok=False)
            return

        self._set_status(format_loaded_status(len(data)), ok=True)
        if self.auto_run:
            self.commit()

    def _send_empty(self) -> None:
        clear_widget_messages(self)
        self._summary_browser.clear()
        self._coordinates_table_widget.clear()
        self._coordinates_table_widget.setRowCount(0)
        self._coordinates_table_widget.setColumnCount(0)
        send_output_values(
            (self.Outputs.coordinates, None),
            (self.Outputs.summary_table, None),
        )

    def commit(self) -> None:
        if self.data is None:
            self._send_empty()
            self._set_status(format_no_input_status(), ok=False)
            return

        clear_widget_messages(self)
        matrix = np.asarray(self.data.X, dtype=float)
        result = compute_molecular_space(matrix, self._config())
        show_service_issues(self, result.issues, subject="molecular space", issue_label="issue")

        if any(issue.severity == "error" for issue in result.issues):
            self._summary_browser.setHtml(_render_summary_html(result, n_rows=len(self.data), n_features=matrix.shape[1]))
            self._coordinates_table_widget.clear()
            self._coordinates_table_widget.setRowCount(0)
            self._coordinates_table_widget.setColumnCount(0)
            send_output_values(
                (self.Outputs.coordinates, None),
                (self.Outputs.summary_table, _summary_table(result, n_rows=len(self.data), n_features=matrix.shape[1])),
            )
            self._set_status(format_failed_status(result.issues[0].message), ok=False)
            return

        coordinates = _coordinates_table(self.data, result)
        summary = _summary_table(result, n_rows=len(self.data), n_features=matrix.shape[1])
        send_output_values(
            (self.Outputs.coordinates, coordinates),
            (self.Outputs.summary_table, summary),
        )

        self._summary_browser.setHtml(
            _render_summary_html(result, n_rows=len(self.data), n_features=matrix.shape[1])
        )
        preview_rows = [
            {
                **{f"space_{index + 1}": f"{value:.6f}" for index, value in enumerate(row)},
                "row_index": row_index,
            }
            for row_index, row in enumerate(result.coordinates[:200])
        ]
        _set_table_rows(self._coordinates_table_widget, preview_rows)

        self._set_status(
            format_done_status(
                f"method={result.method}",
                f"rows={len(self.data)}",
                f"dims={result.coordinates.shape[1]}",
            ),
            ok=not any(issue.severity == "error" for issue in result.issues),
        )


__all__ = ["OWMolecularSpaceMap"]

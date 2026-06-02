from __future__ import annotations

from html import escape

from AnyQt.QtWidgets import (
    QComboBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
)
from Orange.data import ContinuousVariable, Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.chemical_series_service import (
    ChemicalSeriesConfig,
    ChemicalSeriesResult,
    chemical_series_members_as_dicts,
    chemical_series_members_table,
    chemical_series_summary_as_rows,
    chemical_series_summary_table,
    chemical_series_table,
    run_chemical_series_explorer,
    series_rows_as_dicts,
)
from chem_inf_widgets.widgets.ui_helpers import (
    clear_widget_messages,
    format_done_status,
    format_loaded_status,
    format_no_input_status,
)
from chem_inf_widgets.widgets.utils import send_output_values, show_service_issues


def _count_card(label: str, value: int | str) -> str:
    return (
        "<div style='display:inline-block; min-width:150px; margin:6px; padding:10px 12px; "
        "border:1px solid #D0D5DD; border-radius:10px; background:#F8FAFC;'>"
        f"<div style='font-size:12px; color:#475467;'>{escape(label)}</div>"
        f"<div style='font-size:24px; font-weight:600; color:#101828;'>{escape(str(value))}</div>"
        "</div>"
    )


def _render_report_html(result: ChemicalSeriesResult) -> str:
    summary = result.summary
    top_series = series_rows_as_dicts(result)[:5]
    issue_rows = [
        row for row in chemical_series_summary_as_rows(result) if str(row.get("metric", "")).startswith("issue_")
    ][:5]

    cards = "".join(
        [
            _count_card("Rows", summary.n_rows),
            _count_card("Valid molecules", summary.n_valid_molecules),
            _count_card("Invalid molecules", summary.n_invalid_molecules),
            _count_card("Series", summary.n_series),
            _count_card("Singleton series", summary.n_singleton_series),
            _count_card("Acyclic rows", summary.n_acyclic_rows),
        ]
    )

    top_series_html = "".join(
        (
            f"<li>{escape(str(row['scaffold']))}: count={int(row['count'])}, "
            f"fraction={float(row['fraction']) * 100:.1f}%, "
            f"mean_activity={row['mean_activity']}</li>"
        )
        for row in top_series
    ) or "<li>No scaffold series detected.</li>"

    issue_html = "".join(
        f"<li>{escape(str(row['description']))}</li>"
        for row in issue_rows
    ) or "<li>No service issues recorded.</li>"

    target_html = (
        f"<p><b>Activity column:</b> {escape(summary.target_column)}</p>"
        if summary.target_column
        else "<p><b>Activity column:</b> not used</p>"
    )

    return (
        "<html><body style='font-family: sans-serif;'>"
        "<h2>Chemical Series Explorer</h2>"
        f"{cards}"
        f"{target_html}"
        "<h3>Top series</h3>"
        f"<ul>{top_series_html}</ul>"
        "<h3>Service issues</h3>"
        f"<ul>{issue_html}</ul>"
        "</body></html>"
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


def _numeric_target_names(data: Table | None) -> list[str]:
    if data is None:
        return []
    variables = list(data.domain.class_vars) + list(data.domain.attributes) + list(data.domain.metas)
    return [var.name for var in variables if isinstance(var, ContinuousVariable)]


class OWChemicalSeriesExplorer(OWWidget):
    name = "Chemical Series Explorer"
    description = "Group compounds into scaffold-defined series and summarize per-series activity trends."
    icon = "icons/analysis/owscaffoldanalysiswidget.svg"
    priority = 147
    keywords = ["series", "scaffold", "sar", "murcko", "activity"]
    want_main_area = True

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        series_table = Output("Series Table", Table, default=True)
        members_table = Output("Members Table", Table)
        summary_table = Output("Summary Table", Table)
        selected_data = Output("Selected Data", Table)

    auto_run = Setting(True)
    scaffold_kind_idx = Setting(0)
    activity_log_scale = Setting(False)
    target_var_name = Setting("")

    _KINDS = [("Murcko", "murcko"), ("Generic Murcko", "generic")]

    def __init__(self) -> None:
        super().__init__()
        self.data: Table | None = None
        self._result: ChemicalSeriesResult | None = None
        self._series_scaffolds: list[str] = []
        self._selected_scaffold = ""
        self._build_ui()
        self._set_status("Waiting for data.", ok=True)
        self._refresh_target_combo()

    def _build_ui(self) -> None:
        options_box = gui.widgetBox(self.controlArea, "Options")
        gui.comboBox(
            options_box,
            self,
            "scaffold_kind_idx",
            label="Scaffold type",
            items=[label for label, _kind in self._KINDS],
            callback=self._on_settings_changed,
        )
        self._target_combo = QComboBox()
        self._target_combo.currentTextChanged.connect(self._on_target_changed)
        options_box.layout().addWidget(QLabel("Activity column"))
        options_box.layout().addWidget(self._target_combo)
        gui.checkBox(
            options_box,
            self,
            "activity_log_scale",
            "Treat activity as log scale",
            callback=self._on_settings_changed,
        )
        gui.checkBox(
            options_box,
            self,
            "auto_run",
            "Auto-run on input changes",
            callback=self._on_settings_changed,
        )
        gui.button(options_box, self, "Run series explorer", callback=self.commit)

        self._status_label = QLabel("Waiting for data.")
        self._status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self._status_label)

        tabs = QTabWidget()
        self.mainArea.layout().addWidget(tabs)

        self._report_browser = QTextBrowser()
        self._report_browser.setOpenExternalLinks(False)
        tabs.addTab(self._report_browser, "Summary")

        self._series_table_widget = QTableWidget()
        self._series_table_widget.setSelectionBehavior(QTableWidget.SelectRows)
        self._series_table_widget.setSelectionMode(QTableWidget.SingleSelection)
        self._series_table_widget.itemSelectionChanged.connect(self._on_series_selection_changed)
        tabs.addTab(self._series_table_widget, "Series")

        self._members_table_widget = QTableWidget()
        tabs.addTab(self._members_table_widget, "Members")

    def _set_status(self, text: str, ok: bool = False) -> None:
        color = "#027A48" if ok else "#475467"
        self._status_label.setStyleSheet(f"color:{color};")
        self._status_label.setText(text)

    def _refresh_target_combo(self) -> None:
        names = _numeric_target_names(self.data)
        current = self.target_var_name if self.target_var_name in names else ""
        self._target_combo.blockSignals(True)
        self._target_combo.clear()
        self._target_combo.addItem("Auto")
        for name in names:
            self._target_combo.addItem(name)
        if current:
            self._target_combo.setCurrentText(current)
        else:
            self._target_combo.setCurrentIndex(0)
            self.target_var_name = ""
        self._target_combo.setEnabled(bool(names))
        self._target_combo.blockSignals(False)

    def _on_target_changed(self, text: str) -> None:
        self.target_var_name = "" if text == "Auto" else str(text)
        self._on_settings_changed()

    def _on_settings_changed(self) -> None:
        if self.auto_run and self.data is not None:
            self.commit()

    def _config(self) -> ChemicalSeriesConfig:
        return ChemicalSeriesConfig(
            scaffold_kind=self._KINDS[int(self.scaffold_kind_idx)][1],
            target_column=self.target_var_name or None,
            activity_log_scale=bool(self.activity_log_scale),
        )

    @Inputs.data
    def set_data(self, data: Table | None) -> None:
        self.data = data
        self._refresh_target_combo()
        if data is None:
            self._send_empty()
            self._set_status(format_no_input_status(), ok=False)
            return

        self._set_status(format_loaded_status(len(data)), ok=True)
        if self.auto_run:
            self.commit()

    def _send_empty(self) -> None:
        self._result = None
        self._series_scaffolds = []
        self._selected_scaffold = ""
        clear_widget_messages(self)
        self._report_browser.clear()
        self._series_table_widget.clear()
        self._series_table_widget.setRowCount(0)
        self._series_table_widget.setColumnCount(0)
        self._members_table_widget.clear()
        self._members_table_widget.setRowCount(0)
        self._members_table_widget.setColumnCount(0)
        send_output_values(
            (self.Outputs.series_table, None),
            (self.Outputs.members_table, None),
            (self.Outputs.summary_table, None),
            (self.Outputs.selected_data, None),
        )

    def _send_selected_data(self) -> None:
        if self.data is None or self._result is None or not self._selected_scaffold:
            self.Outputs.selected_data.send(None)
            return

        row_indices = [
            int(record.row_index) - 1
            for record in self._result.members
            if record.series_scaffold == self._selected_scaffold and record.valid_molecule
        ]
        if not row_indices:
            self.Outputs.selected_data.send(None)
            return

        self.Outputs.selected_data.send(self.data[row_indices])

    def _on_series_selection_changed(self) -> None:
        if not self._series_scaffolds:
            self._selected_scaffold = ""
            self._send_selected_data()
            return

        row_index = self._series_table_widget.currentRow()
        if row_index < 0:
            selected_ranges = self._series_table_widget.selectedRanges()
            if not selected_ranges:
                self._selected_scaffold = ""
                self._send_selected_data()
                return
            row_index = selected_ranges[0].topRow()

        if 0 <= row_index < len(self._series_scaffolds):
            self._selected_scaffold = self._series_scaffolds[row_index]
        else:
            self._selected_scaffold = ""
        self._send_selected_data()

    def commit(self) -> None:
        if self.data is None:
            self._send_empty()
            self._set_status(format_no_input_status(), ok=False)
            return

        clear_widget_messages(self)
        result = run_chemical_series_explorer(self.data, self._config())
        self._result = result

        series_table = chemical_series_table(result)
        members_table = chemical_series_members_table(result)
        summary = chemical_series_summary_table(result)
        send_output_values(
            (self.Outputs.series_table, series_table),
            (self.Outputs.members_table, members_table),
            (self.Outputs.summary_table, summary),
        )

        self._report_browser.setHtml(_render_report_html(result))
        series_rows = series_rows_as_dicts(result)[:100]
        _set_table_rows(self._series_table_widget, series_rows)
        _set_table_rows(self._members_table_widget, chemical_series_members_as_dicts(result)[:200])
        self._series_scaffolds = [str(row.get("scaffold", "")) for row in series_rows]
        if self._series_scaffolds:
            self._series_table_widget.selectRow(0)
            self._selected_scaffold = self._series_scaffolds[0]
        else:
            self._selected_scaffold = ""
        self._send_selected_data()

        show_service_issues(self, result.issues, subject="chemical series explorer", issue_label="issue")
        self._set_status(
            format_done_status(
                f"rows={result.summary.n_rows}",
                f"series={result.summary.n_series}",
                f"valid={result.summary.n_valid_molecules}",
            ),
            ok=not any(issue.severity == "error" for issue in result.issues),
        )


__all__ = ["OWChemicalSeriesExplorer"]

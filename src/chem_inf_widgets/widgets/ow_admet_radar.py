from __future__ import annotations

from html import escape

from AnyQt.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.admet_radar_service import (
    AdmetRadarConfig,
    AdmetRadarResult,
    admet_flagged_records_as_dicts,
    admet_flagged_table,
    admet_radar_records_table,
    admet_radar_summary_table,
    admet_summary_as_rows,
    run_admet_radar,
)
from chem_inf_widgets.widgets.ui_helpers import (
    clear_widget_messages,
    format_done_status,
    format_loaded_status,
    format_no_input_status,
)
from chem_inf_widgets.widgets.utils import send_output_values, show_service_issues


def _count_card(label: str, value: int) -> str:
    return (
        "<div style='display:inline-block; min-width:150px; margin:6px; padding:10px 12px; "
        "border:1px solid #D0D5DD; border-radius:10px; background:#F8FAFC;'>"
        f"<div style='font-size:12px; color:#475467;'>{escape(label)}</div>"
        f"<div style='font-size:24px; font-weight:600; color:#101828;'>{int(value)}</div>"
        "</div>"
    )


def _render_report_html(result: AdmetRadarResult) -> str:
    summary = result.summary
    summary_rows = admet_summary_as_rows(result)
    issue_rows = [
        row
        for row in summary_rows
        if str(row.get("metric", "")).startswith("issue_")
    ][:5]
    flagged_rows = admet_flagged_records_as_dicts(result)[:5]

    cards = "".join(
        [
            _count_card("Rows", summary.n_rows),
            _count_card("Valid molecules", summary.n_valid_molecules),
            _count_card("Invalid molecules", summary.n_invalid_molecules),
            _count_card("Lipinski pass", summary.n_lipinski_pass),
            _count_card("Veber pass", summary.n_veber_pass),
            _count_card("QED / alerts flagged", summary.n_pains_matches + summary.n_brenk_matches),
        ]
    )

    issue_html = "".join(
        f"<li>{escape(str(row['description']))}</li>"
        for row in issue_rows
    ) or "<li>No service issues recorded.</li>"

    flagged_html = "".join(
        "<li>"
        f"{escape(str(row.get('name') or row.get('input_smiles') or 'compound'))}: "
        f"Lipinski={int(row.get('lipinski_pass', 0))}, "
        f"Veber={int(row.get('veber_pass', 0))}, "
        f"PAINS={int(row.get('pains_match', 0))}, "
        f"Brenk={int(row.get('brenk_match', 0))}"
        "</li>"
        for row in flagged_rows
    ) or "<li>No flagged compounds detected.</li>"

    return (
        "<html><body style='font-family: sans-serif;'>"
        "<h2>ADMET Radar</h2>"
        f"{cards}"
        "<h3>Service issues</h3>"
        f"<ul>{issue_html}</ul>"
        "<h3>Flagged compounds preview</h3>"
        f"<ul>{flagged_html}</ul>"
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


class OWAdmetRadar(OWWidget):
    name = "ADMET Radar"
    description = "Summarize drug-likeness rules and structural alerts for molecular datasets."
    icon = "icons/standardization_filtering/owdrugfilterwidget.png"
    priority = 146
    keywords = ["admet", "lipinski", "veber", "muegge", "pains", "brenk", "qed"]
    want_main_area = True

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        admet_table = Output("ADMET Table", Table, default=True)
        flagged_compounds = Output("Flagged Compounds", Table)
        summary_table = Output("Summary Table", Table)

    auto_run = Setting(True)
    compute_pains = Setting(True)
    compute_brenk = Setting(True)

    def __init__(self) -> None:
        super().__init__()
        self.data: Table | None = None
        self._result: AdmetRadarResult | None = None
        self._build_ui()
        self._set_status("Waiting for data.", ok=True)

    def _build_ui(self) -> None:
        options_box = gui.widgetBox(self.controlArea, "Options")
        gui.checkBox(
            options_box,
            self,
            "compute_pains",
            "Compute PAINS alerts",
            callback=self._on_settings_changed,
        )
        gui.checkBox(
            options_box,
            self,
            "compute_brenk",
            "Compute Brenk alerts",
            callback=self._on_settings_changed,
        )
        gui.checkBox(
            options_box,
            self,
            "auto_run",
            "Auto-run on input changes",
            callback=self._on_settings_changed,
        )
        gui.button(options_box, self, "Run ADMET radar", callback=self.commit)

        self._status_label = QLabel("Waiting for data.")
        self._status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self._status_label)

        tabs = QTabWidget()
        self.mainArea.layout().addWidget(tabs)

        self._report_browser = QTextBrowser()
        self._report_browser.setOpenExternalLinks(False)
        tabs.addTab(self._report_browser, "Summary")

        self._summary_table_widget = QTableWidget()
        tabs.addTab(self._summary_table_widget, "Rule Summary")

        self._flagged_table_widget = QTableWidget()
        tabs.addTab(self._flagged_table_widget, "Flagged Compounds")

    def _set_status(self, text: str, ok: bool = False) -> None:
        color = "#027A48" if ok else "#475467"
        self._status_label.setStyleSheet(f"color:{color};")
        self._status_label.setText(text)

    def _config(self) -> AdmetRadarConfig:
        return AdmetRadarConfig(
            compute_pains=bool(self.compute_pains),
            compute_brenk=bool(self.compute_brenk),
        )

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
        self._result = None
        clear_widget_messages(self)
        self._report_browser.clear()
        self._summary_table_widget.clear()
        self._summary_table_widget.setRowCount(0)
        self._summary_table_widget.setColumnCount(0)
        self._flagged_table_widget.clear()
        self._flagged_table_widget.setRowCount(0)
        self._flagged_table_widget.setColumnCount(0)
        send_output_values(
            (self.Outputs.admet_table, None),
            (self.Outputs.flagged_compounds, None),
            (self.Outputs.summary_table, None),
        )

    def commit(self) -> None:
        if self.data is None:
            self._send_empty()
            self._set_status(format_no_input_status(), ok=False)
            return

        clear_widget_messages(self)
        result = run_admet_radar(self.data, self._config())
        self._result = result

        admet_table = admet_radar_records_table(result)
        flagged_table = admet_flagged_table(result)
        summary = admet_radar_summary_table(result)
        send_output_values(
            (self.Outputs.admet_table, admet_table),
            (self.Outputs.flagged_compounds, flagged_table),
            (self.Outputs.summary_table, summary),
        )

        self._report_browser.setHtml(_render_report_html(result))
        _set_table_rows(self._summary_table_widget, admet_summary_as_rows(result))
        _set_table_rows(self._flagged_table_widget, admet_flagged_records_as_dicts(result)[:100])

        show_service_issues(self, result.issues, subject="admet radar", issue_label="issue")
        self._set_status(
            format_done_status(
                f"rows={result.summary.n_rows}",
                f"valid={result.summary.n_valid_molecules}",
                f"flagged={len(admet_flagged_records_as_dicts(result))}",
            ),
            ok=not any(issue.severity == "error" for issue in result.issues),
        )


__all__ = ["OWAdmetRadar"]

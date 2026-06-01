from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from matplotlib.figure import Figure


@dataclass(frozen=True)
class ReportContext:
    model_name: str
    total_descriptors: int
    descriptors_used: int
    cv_score: Optional[float]
    train_metrics: dict
    test_metrics: dict
    external_metrics: dict


def build_report_html(
    *,
    model_name: str,
    total_descriptors: int,
    descriptors_used: int,
    cv_score: Optional[float],
    train_metrics: dict,
    test_metrics: dict,
    external_metrics: dict,
) -> str:
    cv_text = f"{cv_score:.3f}" if cv_score is not None else "N/A"
    metrics_list = ["R²", "RMSE", "MAE", "Median AE", "Explained Variance"]

    html = f"""
    <html>
      <head>
        <style>
          body {{ font-family: Arial, sans-serif; font-size: 12pt; color: #333; }}
          h2 {{ color: #444; }}
          table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
          th, td {{ border: 1px solid #ccc; padding: 5px; text-align: center; }}
          th {{ background-color: #f0f0f0; }}
        </style>
      </head>
      <body>
        <h2>Model Report</h2>
        <p><b>Model:</b> {model_name}</p>
        <p><b>Total Descriptors:</b> {total_descriptors}</p>
        <p><b>Descriptors Used:</b> {descriptors_used}</p>
        <p><b>CV R²:</b> {cv_text}</p>
        <h3>Metrics</h3>
        <table>
          <tr>
            <th>Metric</th>
            <th>Training</th>
            <th>Test</th>
            <th>External</th>
          </tr>
    """
    for metric in metrics_list:
        train_val = f"{train_metrics[metric]:.3f}" if metric in train_metrics else "N/A"
        test_val = f"{test_metrics[metric]:.3f}" if metric in test_metrics else "N/A"
        ext_val = f"{external_metrics[metric]:.3f}" if external_metrics and metric in external_metrics else "N/A"
        html += f"""
          <tr>
            <td>{metric}</td>
            <td>{train_val}</td>
            <td>{test_val}</td>
            <td>{ext_val}</td>
          </tr>
        """
    html += """
        </table>
      </body>
    </html>
    """
    return html


def build_report_context(
    *,
    model_name: str,
    total_descriptors: int,
    descriptors_used: int,
    cv_score: Optional[float],
    train_metrics: dict,
    test_metrics: dict,
    external_metrics: dict,
) -> ReportContext:
    return ReportContext(
        model_name=model_name,
        total_descriptors=total_descriptors,
        descriptors_used=descriptors_used,
        cv_score=cv_score,
        train_metrics=dict(train_metrics),
        test_metrics=dict(test_metrics),
        external_metrics=dict(external_metrics),
    )


def build_report_html_from_context(context: ReportContext) -> str:
    return build_report_html(
        model_name=context.model_name,
        total_descriptors=context.total_descriptors,
        descriptors_used=context.descriptors_used,
        cv_score=context.cv_score,
        train_metrics=context.train_metrics,
        test_metrics=context.test_metrics,
        external_metrics=context.external_metrics,
    )


def build_waiting_status_text(model_name: str) -> str:
    return f"Please wait calculation {model_name} is started"


def build_waiting_report_html() -> str:
    return (
        '<div style="text-align: center; font-weight: bold; font-size: 14pt;">'
        "Please wait, calculation of the QSAR model in progress"
        "</div>"
    )


def build_cancelled_status_text() -> str:
    return "Calculation cancelled."


def build_completed_status_text(model_name: str, performance_text: str) -> str:
    return f"Calculation {model_name} is completed.\n{performance_text}"


def build_error_status_text(error_msg: str) -> str:
    return "Error: " + str(error_msg)


def build_pdf_export_success_status_text() -> str:
    return "PDF Exported Successfully."


def build_pdf_export_empty_status_text() -> str:
    return "No QSAR results available to export."


def build_pdf_export_error_status_text(error_msg: str) -> str:
    return "Error exporting PDF: " + str(error_msg)


def build_pdf_report_text(
    *,
    model_name: str,
    total_descriptors: int,
    descriptors_used: int,
    cv_score: Optional[float],
    train_metrics: dict,
    test_metrics: dict,
    external_metrics: dict,
) -> str:
    cv_info = f"CV R²: {cv_score:.3f}\n\n" if cv_score is not None else "CV R²: N/A\n\n"
    report_text = (
        f"Model: {model_name}\n"
        f"Total Descriptors: {total_descriptors}\n"
        f"Descriptors Used: {descriptors_used}\n"
        f"{cv_info}"
    )

    report_text += "Training Metrics:\n"
    for key, value in train_metrics.items():
        report_text += f"  {key}: {value:.3f}\n"

    report_text += "\nTest Metrics:\n"
    for key, value in test_metrics.items():
        report_text += f"  {key}: {value:.3f}\n"

    report_text += "\nExternal Metrics:\n"
    for key, value in external_metrics.items():
        report_text += f"  {key}: {value:.3f}\n"

    return report_text


def build_pdf_report_text_from_context(context: ReportContext) -> str:
    return build_pdf_report_text(
        model_name=context.model_name,
        total_descriptors=context.total_descriptors,
        descriptors_used=context.descriptors_used,
        cv_score=context.cv_score,
        train_metrics=context.train_metrics,
        test_metrics=context.test_metrics,
        external_metrics=context.external_metrics,
    )


def build_pdf_report_figure_from_context(context: ReportContext) -> Figure:
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.axis("off")
    report_text = build_pdf_report_text_from_context(context)
    ax.text(0, 1, report_text, va="top", ha="left", fontsize=10, wrap=True)
    return fig


def collect_pdf_export_figures(*figures) -> tuple:
    return tuple(fig for fig in figures if fig is not None)


__all__ = [
    "ReportContext",
    "build_cancelled_status_text",
    "build_completed_status_text",
    "build_error_status_text",
    "build_pdf_export_empty_status_text",
    "build_pdf_export_error_status_text",
    "build_pdf_export_success_status_text",
    "build_pdf_report_figure_from_context",
    "build_pdf_report_text",
    "build_pdf_report_text_from_context",
    "build_report_context",
    "build_report_html",
    "build_report_html_from_context",
    "build_waiting_report_html",
    "build_waiting_status_text",
    "collect_pdf_export_figures",
]

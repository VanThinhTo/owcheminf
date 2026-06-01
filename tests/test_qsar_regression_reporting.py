from __future__ import annotations

from chem_inf_widgets.chemcore.qsar.reporting import (
    build_cancelled_status_text,
    build_completed_status_text,
    build_pdf_report_figure_from_context,
    build_report_context,
    build_report_html_from_context,
    build_waiting_status_text,
    collect_pdf_export_figures,
)


def test_qsar_reporting_context_round_trip_and_pdf_figure():
    context = build_report_context(
        model_name="Random Forest",
        total_descriptors=100,
        descriptors_used=12,
        cv_score=0.82,
        train_metrics={"R²": 0.9},
        test_metrics={"R²": 0.8},
        external_metrics={"MAE": 0.7},
    )

    html = build_report_html_from_context(context)
    fig = build_pdf_report_figure_from_context(context)

    assert "Random Forest" in html
    assert "Descriptors Used" in html
    assert len(fig.axes) == 1


def test_qsar_reporting_status_and_figure_collection_helpers():
    assert "Model X" in build_waiting_status_text("Model X")
    assert "completed" in build_completed_status_text("Model X", "metrics")
    assert build_cancelled_status_text() == "Calculation cancelled."
    assert collect_pdf_export_figures(None, "a", None, "b") == ("a", "b")

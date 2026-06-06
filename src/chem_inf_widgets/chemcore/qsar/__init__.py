"""QSAR core utilities (dataset prep, feature filtering, diagnostics, reporting, AD).

This module intentionally exposes its public API lazily. A few service-layer
modules import one small QSAR constant or helper, and eager importing the whole
stack here can create circular imports through the applicability-domain
workbench path.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "CompoundPreview",
    "DiagnosticDatasetPayload",
    "DiagnosticPlotData",
    "DiagnosticPlotSpec",
    "DiagnosticSeries",
    "FeatureInspectionPayload",
    "LEAKY_OR_NON_DESCRIPTOR_ATTRIBUTE_NAMES",
    "QSARRunConfig",
    "RDKit_DESCRIPTOR_NAMES",
    "ReportContext",
    "SelectionGalleryPayload",
    "SelectionPublishPayload",
    "TARGET_COLUMN_CANDIDATES",
    "TORCH_AVAILABLE",
    "TorchRegressor",
    "_build_modeling_pipeline",
    "_make_safe_regressor",
    "_run_auto_qsar_model_selection",
    "_rdkit_descriptor_row",
    "_transform_features",
    "available_algorithms",
    "build_applicability_domain_table",
    "build_cancelled_status_text",
    "build_completed_status_text",
    "build_compound_previews",
    "build_diagnostic_plot_spec",
    "build_error_status_text",
    "build_feature_inspection_payload",
    "build_pdf_export_empty_status_text",
    "build_pdf_export_error_status_text",
    "build_pdf_export_success_status_text",
    "build_pdf_report_figure_from_context",
    "build_pdf_report_text",
    "build_pdf_report_text_from_context",
    "build_qsar_modeling_summary_table",
    "build_report_context",
    "build_report_html",
    "build_report_html_from_context",
    "build_run_config",
    "build_selection_gallery_payload",
    "build_selection_publish_payload",
    "build_waiting_report_html",
    "build_waiting_status_text",
    "cap_qsar_descriptor_matrix",
    "clean_qsar_descriptor_matrix",
    "collect_pdf_export_figures",
    "compute_coef_stats",
    "compute_vif",
    "diagnostic_payloads_from_result",
    "extract_descriptor_coefficients",
    "find_name_var",
    "find_smiles_var",
    "lasso_selection_indices",
    "prepare_diagnostic_plot_data",
    "prepare_qsar_model_matrix",
    "rectangle_selection_indices",
    "run_qsar_regression",
    "selection_overlay_offsets",
    "selection_status_text",
]

_EXPORTS = {
    "LEAKY_OR_NON_DESCRIPTOR_ATTRIBUTE_NAMES": ".dataset",
    "RDKit_DESCRIPTOR_NAMES": ".dataset",
    "TARGET_COLUMN_CANDIDATES": ".dataset",
    "_rdkit_descriptor_row": ".dataset",
    "cap_qsar_descriptor_matrix": ".dataset",
    "clean_qsar_descriptor_matrix": ".dataset",
    "find_name_var": ".dataset",
    "find_smiles_var": ".dataset",
    "prepare_qsar_model_matrix": ".dataset",
    "QSARRunConfig": ".algorithms",
    "TORCH_AVAILABLE": ".algorithms",
    "TorchRegressor": ".algorithms",
    "_build_modeling_pipeline": ".algorithms",
    "_make_safe_regressor": ".algorithms",
    "_run_auto_qsar_model_selection": ".algorithms",
    "available_algorithms": ".algorithms",
    "build_run_config": ".algorithms",
    "build_applicability_domain_table": ".applicability_domain",
    "CompoundPreview": ".diagnostics",
    "DiagnosticDatasetPayload": ".diagnostics",
    "DiagnosticPlotData": ".diagnostics",
    "DiagnosticPlotSpec": ".diagnostics",
    "DiagnosticSeries": ".diagnostics",
    "FeatureInspectionPayload": ".diagnostics",
    "SelectionGalleryPayload": ".diagnostics",
    "SelectionPublishPayload": ".diagnostics",
    "_transform_features": ".diagnostics",
    "build_compound_previews": ".diagnostics",
    "build_diagnostic_plot_spec": ".diagnostics",
    "build_feature_inspection_payload": ".diagnostics",
    "build_selection_gallery_payload": ".diagnostics",
    "build_selection_publish_payload": ".diagnostics",
    "compute_coef_stats": ".diagnostics",
    "compute_vif": ".diagnostics",
    "diagnostic_payloads_from_result": ".diagnostics",
    "extract_descriptor_coefficients": ".diagnostics",
    "lasso_selection_indices": ".diagnostics",
    "prepare_diagnostic_plot_data": ".diagnostics",
    "rectangle_selection_indices": ".diagnostics",
    "selection_overlay_offsets": ".diagnostics",
    "selection_status_text": ".diagnostics",
    "run_qsar_regression": ".regression_service",
    "ReportContext": ".reporting",
    "build_cancelled_status_text": ".reporting",
    "build_completed_status_text": ".reporting",
    "build_error_status_text": ".reporting",
    "build_pdf_export_empty_status_text": ".reporting",
    "build_pdf_export_error_status_text": ".reporting",
    "build_pdf_export_success_status_text": ".reporting",
    "build_pdf_report_figure_from_context": ".reporting",
    "build_pdf_report_text": ".reporting",
    "build_pdf_report_text_from_context": ".reporting",
    "build_report_context": ".reporting",
    "build_report_html": ".reporting",
    "build_report_html_from_context": ".reporting",
    "build_waiting_report_html": ".reporting",
    "build_waiting_status_text": ".reporting",
    "collect_pdf_export_figures": ".reporting",
    "build_qsar_modeling_summary_table": ".validation",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

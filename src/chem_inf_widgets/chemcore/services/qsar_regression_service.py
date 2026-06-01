from __future__ import annotations

import io
import json
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib.path import Path
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning

from Orange.data import ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table
from rdkit.Chem import Draw
from chem_inf_widgets.chemcore.qsar.algorithms import (
    QSARRunConfig,
    _build_modeling_pipeline,
    _run_auto_qsar_model_selection,
    available_algorithms,
    build_run_config,
)
from chem_inf_widgets.chemcore.qsar.dataset import (
    LEAKY_OR_NON_DESCRIPTOR_ATTRIBUTE_NAMES,
    RDKit_DESCRIPTOR_NAMES,
    TARGET_COLUMN_CANDIDATES,
    _rdkit_descriptor_row,
    cap_qsar_descriptor_matrix,
    clean_qsar_descriptor_matrix,
    find_name_var,
    find_smiles_var,
    prepare_qsar_model_matrix,
)

from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles
from chem_inf_widgets.chemcore.services.orange_table_utils import records_to_orange_table, safe_table_from_numpy


@dataclass(frozen=True)
class CompoundPreview:
    title: str
    png_bytes: bytes


@dataclass(frozen=True)
class DiagnosticPlotData:
    preds: np.ndarray
    actuals: np.ndarray
    residuals: np.ndarray
    inlier_mask: np.ndarray
    outlier_mask: np.ndarray
    is_classification: bool


@dataclass(frozen=True)
class DiagnosticSeries:
    x: np.ndarray
    y: np.ndarray
    color: str
    label: str


@dataclass(frozen=True)
class DiagnosticPlotSpec:
    left_series: tuple[DiagnosticSeries, ...]
    right_series: tuple[DiagnosticSeries, ...]
    diagonal_min: float
    diagonal_max: float
    left_title: str
    left_xlabel: str
    left_ylabel: str
    right_title: str
    right_xlabel: str
    right_ylabel: str
    show_legends: bool


@dataclass(frozen=True)
class DiagnosticDatasetPayload:
    dataset_type: str
    X: np.ndarray
    y: np.ndarray
    pipeline: object
    is_classification: bool
    result_table: Optional[Table]


@dataclass(frozen=True)
class ReportContext:
    model_name: str
    total_descriptors: int
    descriptors_used: int
    cv_score: Optional[float]
    train_metrics: dict
    test_metrics: dict
    external_metrics: dict


@dataclass(frozen=True)
class FeatureInspectionPayload:
    available: bool
    message_html: str
    value_label: str
    names: tuple[str, ...]
    values: Optional[np.ndarray]
    ses: Optional[np.ndarray]
    ts: Optional[np.ndarray]
    ps: Optional[np.ndarray]
    vifs: Optional[np.ndarray]
    chart_names: tuple[str, ...]
    chart_values: Optional[np.ndarray]
    chart_colors: tuple[str, ...]
    chart_title: str
    subtitle: str
    tab_title: str


@dataclass(frozen=True)
class SelectionGalleryPayload:
    placeholder_text: Optional[str]
    previews: tuple[CompoundPreview, ...]
    more_count: int


@dataclass(frozen=True)
class SelectionPublishPayload:
    selected_table: Table
    gallery: SelectionGalleryPayload
    status_text: str


def build_applicability_domain_table(result: dict) -> Table | None:
    """Compute a Williams/leverage and kNN-distance AD table for train/test/external rows."""
    if not result or result.get("pipeline") is None:
        return None
    if result.get("is_classification"):
        return None
    try:
        pipeline = result["pipeline"]
        X_train_t = np.asarray(_transform_features(pipeline, result["X_train"]), dtype=float)
        y_train = np.asarray(result["y_train"], dtype=float)
        train_pred = np.asarray(pipeline.predict(result["X_train"]), dtype=float).ravel()
        n_train, p_feat = X_train_t.shape
        if n_train < 3 or p_feat < 1:
            return None

        X_aug = np.column_stack([np.ones(n_train), X_train_t])
        xtx_inv = np.linalg.pinv(X_aug.T @ X_aug)
        h_star = float(3.0 * (p_feat + 1) / max(n_train, 1))

        k = max(1, min(5, n_train))
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_train_t)
        train_dist, _ = nn.kneighbors(X_train_t)
        # For training points, the first neighbor is usually itself. Exclude it when possible.
        train_knn = train_dist[:, 1:].mean(axis=1) if train_dist.shape[1] > 1 else train_dist.mean(axis=1)
        dist_threshold = float(np.nanmean(train_knn) + 3.0 * np.nanstd(train_knn)) if train_knn.size else float("nan")

        rows: list[dict] = []

        def add_dataset(dataset: str, X_raw: np.ndarray, y: np.ndarray, pred: np.ndarray) -> None:
            X_t = np.asarray(_transform_features(pipeline, X_raw), dtype=float)
            X_t_aug = np.column_stack([np.ones(X_t.shape[0]), X_t])
            leverage = np.sum((X_t_aug @ xtx_inv) * X_t_aug, axis=1)
            dist, _ = nn.kneighbors(X_t)
            knn = dist[:, 1:].mean(axis=1) if dataset == "train" and dist.shape[1] > 1 else dist.mean(axis=1)
            for i, (actual, predicted, lev, d) in enumerate(zip(y, pred, leverage, knn)):
                in_lev = bool(np.isfinite(lev) and lev <= h_star)
                in_dist = bool((not np.isfinite(dist_threshold)) or d <= dist_threshold)
                rows.append({
                    "dataset": dataset,
                    "row_index": int(i),
                    "actual": float(actual),
                    "predicted": float(predicted),
                    "residual": float(actual - predicted),
                    "abs_residual": float(abs(actual - predicted)),
                    "ad_leverage": float(lev),
                    "ad_leverage_threshold": h_star,
                    "ad_knn_distance": float(d),
                    "ad_distance_threshold": dist_threshold,
                    "ad_in_domain": int(in_lev and in_dist),
                    "ad_outlier": int((not in_lev) or (not in_dist)),
                })

        add_dataset("train", result["X_train"], y_train, train_pred)
        test_pred = np.asarray(pipeline.predict(result["X_test"]), dtype=float).ravel()
        add_dataset("test", result["X_test"], np.asarray(result["y_test"], dtype=float), test_pred)
        if result.get("X_ext") is not None and result.get("y_ext") is not None:
            ext_pred = np.asarray(pipeline.predict(result["X_ext"]), dtype=float).ravel()
            add_dataset("external", result["X_ext"], np.asarray(result["y_ext"], dtype=float), ext_pred)
        return records_to_orange_table(rows, name="QSAR Applicability Domain") if rows else None
    except Exception:
        return None

def build_qsar_modeling_summary_table(result: dict) -> Table | None:
    """Build a compact modeling audit table for downstream inspection."""
    if not result:
        return None
    cleanup = dict(result.get("descriptor_cleanup") or {})
    records = [
        {
            "section": "dataset",
            "metric": "target_column",
            "value": result.get("target_column", ""),
            "numeric_value": "",
        },
        {
            "section": "dataset",
            "metric": "usable_rows",
            "value": str(result.get("usable_row_count", "")),
            "numeric_value": result.get("usable_row_count", ""),
        },
        {
            "section": "dataset",
            "metric": "removed_rows",
            "value": str(result.get("removed_row_count", "")),
            "numeric_value": result.get("removed_row_count", ""),
        },
        {
            "section": "descriptors",
            "metric": "input_descriptor_count",
            "value": str(cleanup.get("input_descriptor_count", "")),
            "numeric_value": cleanup.get("input_descriptor_count", ""),
        },
        {
            "section": "descriptors",
            "metric": "descriptor_count_used",
            "value": str(cleanup.get("descriptor_count", len(result.get("feature_names", [])))),
            "numeric_value": cleanup.get("descriptor_count", len(result.get("feature_names", []))),
        },
        {
            "section": "descriptors",
            "metric": "removed_all_missing_count",
            "value": str(cleanup.get("removed_all_missing_count", 0)),
            "numeric_value": cleanup.get("removed_all_missing_count", 0),
        },
        {
            "section": "descriptors",
            "metric": "removed_constant_count",
            "value": str(cleanup.get("removed_constant_count", 0)),
            "numeric_value": cleanup.get("removed_constant_count", 0),
        },
        {
            "section": "descriptors",
            "metric": "qsar_cap_limit",
            "value": str(cleanup.get("qsar_cap_limit", "")),
            "numeric_value": cleanup.get("qsar_cap_limit", ""),
        },
        {
            "section": "descriptors",
            "metric": "removed_qsar_cap_count",
            "value": str(cleanup.get("removed_qsar_cap_count", 0)),
            "numeric_value": cleanup.get("removed_qsar_cap_count", 0),
        },
        {
            "section": "model",
            "metric": "cv_score",
            "value": str(result.get("cv_score", "")),
            "numeric_value": result.get("cv_score", ""),
        },
    ]
    if cleanup.get("removed_all_missing"):
        records.append({
            "section": "descriptors",
            "metric": "removed_all_missing_examples",
            "value": ", ".join(cleanup.get("removed_all_missing", [])),
            "numeric_value": "",
        })
    if cleanup.get("removed_constant"):
        records.append({
            "section": "descriptors",
            "metric": "removed_constant_examples",
            "value": ", ".join(cleanup.get("removed_constant", [])),
            "numeric_value": "",
        })
    if cleanup.get("removed_qsar_cap"):
        records.append({
            "section": "descriptors",
            "metric": "removed_qsar_cap_examples",
            "value": ", ".join(cleanup.get("removed_qsar_cap", [])),
            "numeric_value": "",
        })
    return records_to_orange_table(
        records,
        attribute_columns=["numeric_value"],
        meta_columns=["section", "metric", "value"],
        name="QSAR Modeling Summary",
    )


def _result_domain(source_data: Table, feature_names: Sequence[str], target_var, *, is_classification: bool = False) -> Domain:
    attributes = [ContinuousVariable(str(name)) for name in feature_names]
    pred_var = ContinuousVariable("Predicted") if not is_classification else DiscreteVariable("Predicted")
    class_vars = [ContinuousVariable(target_var.name)] if isinstance(target_var, ContinuousVariable) else [target_var]
    return Domain(attributes + [pred_var], class_vars, source_data.domain.metas)


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


def build_compound_previews(selected_table: Optional[Table], *, max_preview: int = 12) -> list[CompoundPreview]:
    if selected_table is None or len(selected_table) == 0:
        return []

    smiles_var = find_smiles_var(selected_table)
    if smiles_var is None:
        return []

    name_var = find_name_var(selected_table)
    smiles_col = selected_table.get_column(smiles_var)
    names_col = selected_table.get_column(name_var) if name_var is not None else None

    previews: list[CompoundPreview] = []
    for i in range(min(len(selected_table), max_preview)):
        smiles = "" if smiles_col[i] is None else str(smiles_col[i]).strip()
        if not smiles:
            continue

        mol = safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True).mol
        if mol is None:
            continue

        img = Draw.MolToImage(mol, size=(150, 110))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        title = ""
        if names_col is not None and names_col[i] is not None:
            title = str(names_col[i]).strip()
        if not title:
            title = f"Row {i + 1}"

        previews.append(CompoundPreview(title=title, png_bytes=buf.getvalue()))

    return previews


def rectangle_selection_indices(preds, ys, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    preds_arr = np.asarray(preds, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    x_min, x_max = sorted([float(x0), float(x1)])
    y_min, y_max = sorted([float(y0), float(y1)])
    mask = (preds_arr >= x_min) & (preds_arr <= x_max) & (ys_arr >= y_min) & (ys_arr <= y_max)
    return np.flatnonzero(mask)


def lasso_selection_indices(preds, ys, vertices) -> np.ndarray:
    if not vertices:
        return np.array([], dtype=int)
    preds_arr = np.asarray(preds, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    points = np.column_stack([preds_arr, ys_arr])
    mask = Path(vertices).contains_points(points)
    return np.flatnonzero(mask)


def selection_overlay_offsets(preds, y, residuals, selected_idx) -> tuple[np.ndarray, np.ndarray]:
    preds_arr = np.asarray(preds, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    residuals_arr = np.asarray(residuals, dtype=float)
    idx = np.asarray(selected_idx, dtype=int)
    if idx.size == 0:
        empty = np.empty((0, 2))
        return empty, empty
    left_offsets = np.column_stack([preds_arr[idx], y_arr[idx]])
    right_offsets = np.column_stack([preds_arr[idx], residuals_arr[idx]])
    return left_offsets, right_offsets


def selection_status_text(model_name: str, dataset_type: str, count: int) -> str:
    return (
        f"Calculation {model_name} is completed.\n"
        f"Selected {int(count)} compounds from {dataset_type} diagnostics."
    )


def build_selection_gallery_payload(
    selected_table: Optional[Table],
    dataset_type: str,
    *,
    max_preview: int = 12,
) -> SelectionGalleryPayload:
    if selected_table is None or len(selected_table) == 0:
        return SelectionGalleryPayload(
            placeholder_text=f"No compounds selected in {dataset_type} diagnostics.",
            previews=tuple(),
            more_count=0,
        )

    previews = build_compound_previews(selected_table, max_preview=max_preview)
    if not previews:
        return SelectionGalleryPayload(
            placeholder_text=(
                f"Selected {len(selected_table)} rows from {dataset_type}, but no valid molecules could be rendered."
            ),
            previews=tuple(),
            more_count=0,
        )

    shown = len(previews)
    more_count = max(0, len(selected_table) - shown)
    return SelectionGalleryPayload(
        placeholder_text=None,
        previews=tuple(previews),
        more_count=more_count,
    )


def build_selection_publish_payload(
    *,
    model_name: str,
    dataset_type: str,
    table: Table,
    selected_idx,
    max_preview: int = 12,
) -> SelectionPublishPayload:
    idx = np.asarray(selected_idx, dtype=int)
    selected_table = table[idx.tolist()] if idx.size else table[:0]
    gallery = build_selection_gallery_payload(selected_table, dataset_type, max_preview=max_preview)
    return SelectionPublishPayload(
        selected_table=selected_table,
        gallery=gallery,
        status_text=selection_status_text(model_name, dataset_type, int(idx.size)),
    )


def prepare_diagnostic_plot_data(X, y, pipeline, *, is_classification: bool = False) -> DiagnosticPlotData:
    preds = np.asarray(pipeline.predict(X), dtype=float)
    actuals = np.asarray(y, dtype=float)

    if not is_classification:
        residuals = actuals - preds
        threshold = 2 * np.std(residuals)
        inlier_mask = np.abs(residuals) <= threshold
        outlier_mask = np.abs(residuals) > threshold
    else:
        residuals = (actuals != preds).astype(float)
        inlier_mask = np.ones_like(preds, dtype=bool)
        outlier_mask = np.zeros_like(preds, dtype=bool)

    return DiagnosticPlotData(
        preds=preds,
        actuals=actuals,
        residuals=residuals,
        inlier_mask=inlier_mask,
        outlier_mask=outlier_mask,
        is_classification=bool(is_classification),
    )


def build_diagnostic_plot_spec(diagnostic: DiagnosticPlotData) -> DiagnosticPlotSpec:
    diagonal_min = float(np.min(diagnostic.preds)) if diagnostic.preds.size else 0.0
    diagonal_max = float(np.max(diagnostic.preds)) if diagnostic.preds.size else 1.0

    if not diagnostic.is_classification:
        left_series = [
            DiagnosticSeries(
                x=diagnostic.preds[diagnostic.inlier_mask],
                y=diagnostic.actuals[diagnostic.inlier_mask],
                color="blue",
                label="Inliers",
            )
        ]
        right_series = [
            DiagnosticSeries(
                x=diagnostic.preds[diagnostic.inlier_mask],
                y=diagnostic.residuals[diagnostic.inlier_mask],
                color="blue",
                label="Inliers",
            )
        ]
        if np.any(diagnostic.outlier_mask):
            left_series.append(
                DiagnosticSeries(
                    x=diagnostic.preds[diagnostic.outlier_mask],
                    y=diagnostic.actuals[diagnostic.outlier_mask],
                    color="red",
                    label="Outliers",
                )
            )
            right_series.append(
                DiagnosticSeries(
                    x=diagnostic.preds[diagnostic.outlier_mask],
                    y=diagnostic.residuals[diagnostic.outlier_mask],
                    color="red",
                    label="Outliers",
                )
            )
        return DiagnosticPlotSpec(
            left_series=tuple(left_series),
            right_series=tuple(right_series),
            diagonal_min=diagonal_min,
            diagonal_max=diagonal_max,
            left_title="Predicted vs Actual",
            left_xlabel="Predicted",
            left_ylabel="Actual",
            right_title="Residuals vs Predicted",
            right_xlabel="Predicted",
            right_ylabel="Residuals",
            show_legends=True,
        )

    return DiagnosticPlotSpec(
        left_series=(
            DiagnosticSeries(
                x=diagnostic.preds,
                y=diagnostic.actuals,
                color="green",
                label="Observations",
            ),
        ),
        right_series=(
            DiagnosticSeries(
                x=diagnostic.preds,
                y=diagnostic.residuals,
                color="green",
                label="Observations",
            ),
        ),
        diagonal_min=diagonal_min,
        diagonal_max=diagonal_max,
        left_title="Predicted vs Actual",
        left_xlabel="Predicted",
        left_ylabel="Actual",
        right_title="Misclassifications (1 if error)",
        right_xlabel="Predicted",
        right_ylabel="Error Indicator",
        show_legends=False,
    )


def diagnostic_payloads_from_result(result: dict, *, include_external: bool = False) -> list[DiagnosticDatasetPayload]:
    payloads = [
        DiagnosticDatasetPayload(
            dataset_type="train",
            X=result["X_train"],
            y=result["y_train"],
            pipeline=result["pipeline"],
            is_classification=result["is_classification"],
            result_table=result["train_table"],
        ),
        DiagnosticDatasetPayload(
            dataset_type="test",
            X=result["X_test"],
            y=result["y_test"],
            pipeline=result["pipeline"],
            is_classification=result["is_classification"],
            result_table=result["test_table"],
        ),
    ]

    if include_external and result.get("external_table") is not None:
        payloads.append(
            DiagnosticDatasetPayload(
                dataset_type="external",
                X=result["X_ext"],
                y=result["y_ext"],
                pipeline=result["pipeline"],
                is_classification=result["is_classification"],
                result_table=result["external_table"],
            )
        )

    return payloads


def build_feature_inspection_payload(result: dict, *, model_name: str) -> FeatureInspectionPayload:
    pipeline = result.get("pipeline")
    feature_names = list(result.get("feature_names", []))

    if pipeline is None or not feature_names:
        return FeatureInspectionPayload(
            available=False,
            message_html="No feature information available.",
            value_label="value",
            names=tuple(),
            values=None,
            ses=None,
            ts=None,
            ps=None,
            vifs=None,
            chart_names=tuple(),
            chart_values=None,
            chart_colors=tuple(),
            chart_title="",
            subtitle="",
            tab_title="Features",
        )

    names = list(feature_names)
    if "feature_selection" in pipeline.named_steps:
        try:
            mask = pipeline.named_steps["feature_selection"].get_support()
            names = [n for n, keep in zip(names, mask) if keep]
        except Exception:
            pass

    estimator = pipeline.named_steps.get("regressor", pipeline[-1])
    values: np.ndarray | None = None
    value_label = "value"

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_).ravel()
        if len(coef) == len(names):
            values = coef
            value_label = "Coefficient"
    if values is None and hasattr(estimator, "feature_importances_"):
        imp = np.asarray(estimator.feature_importances_).ravel()
        if len(imp) == len(names):
            values = imp
            value_label = "Importance"

    if values is None:
        return FeatureInspectionPayload(
            available=True,
            message_html=(
                f"<b>{model_name}</b> does not expose coefficients or "
                "feature importances directly.<br>"
                "Selected descriptors are listed in the <i>Descriptor Coefficients</i> output."
            ),
            value_label=value_label,
            names=tuple(names),
            values=None,
            ses=None,
            ts=None,
            ps=None,
            vifs=None,
            chart_names=tuple(),
            chart_values=None,
            chart_colors=tuple(),
            chart_title="",
            subtitle=f"{len(names)} selected descriptors (no {value_label.lower()} values for this model type)",
            tab_title=f"Features ({len(names)})",
        )

    order = np.argsort(np.abs(values))[::-1]
    names = [names[i] for i in order]
    values = np.asarray(values[order], dtype=float)

    coef_stats = result.get("coef_stats")
    ses_s = ts_s = ps_s = None
    if coef_stats is not None:
        beta_d = coef_stats["beta"][1:]
        if len(beta_d) == len(names):
            ses_s = np.asarray(coef_stats["se"][1:][order], dtype=float)
            ts_s = np.asarray(coef_stats["t"][1:][order], dtype=float)
            ps_s = np.asarray(coef_stats["p"][1:][order], dtype=float)

    vifs_all = result.get("vifs")
    vifs_s = np.asarray(vifs_all[order], dtype=float) if (vifs_all is not None and len(vifs_all) == len(names)) else None

    top_n = min(30, len(names))
    chart_names = tuple(names[:top_n])
    chart_values = np.asarray(values[:top_n], dtype=float)
    chart_colors = tuple("#16a34a" if v >= 0 else "#dc2626" for v in chart_values)

    total_feature_count = len(feature_names)
    suffix = f" ({total_feature_count - len(names)} dropped by SelectKBest)" if len(names) < total_feature_count else ""

    return FeatureInspectionPayload(
        available=True,
        message_html="",
        value_label=value_label,
        names=tuple(names),
        values=values,
        ses=ses_s,
        ts=ts_s,
        ps=ps_s,
        vifs=vifs_s,
        chart_names=chart_names,
        chart_values=chart_values,
        chart_colors=chart_colors,
        chart_title=(
            f"{model_name} — {value_label}s  "
            f"(top {top_n} of {len(names)} {'selected ' if len(names) < total_feature_count else ''}descriptors)"
        ),
        subtitle=f"All {len(names)} descriptors — sorted by |{value_label.lower()}|",
        tab_title=f"Features ({len(names)}){suffix}",
    )


def run_qsar_regression(
    data: Table,
    external_data: Optional[Table],
    config: QSARRunConfig,
    *,
    interruption_requested: Optional[Callable[[], bool]] = None,
):
    def cancelled() -> bool:
        return bool(interruption_requested and interruption_requested())

    if cancelled():
        return None

    prepared = prepare_qsar_model_matrix(data)
    X_all = prepared["X"]
    y_all = prepared["y"]
    metas_all = prepared["metas"]
    feature_names = prepared["feature_names"]
    target_var = prepared["target_var"]
    generated_descriptors = bool(prepared["generated_descriptors"])

    finite_y = np.isfinite(y_all)
    # A row is usable if: target is finite AND at least one descriptor value is finite.
    # Rows with some NaN descriptors are kept here — the imputation step in the pipeline
    # will fill them. Requiring ALL descriptors to be finite would drop every row when
    # descriptor sets like Mordred contain even a single NaN per molecule.
    if X_all.ndim == 2 and X_all.shape[1]:
        has_any_finite_x = np.any(np.isfinite(X_all), axis=1)
        finite_x_rows = np.all(np.isfinite(X_all), axis=1)  # for diagnostic message only
    else:
        has_any_finite_x = np.zeros(len(y_all), dtype=bool)
        finite_x_rows = has_any_finite_x
    finite_rows = finite_y & has_any_finite_x
    if np.count_nonzero(finite_rows) < 3:
        raise ValueError(
            "Too few rows with valid target and descriptor values for QSAR regression "
            f"({np.count_nonzero(finite_rows)} usable rows).\n"
            f"Rows with numeric target '{target_var.name}': {np.count_nonzero(finite_y)} / {len(y_all)}.\n"
            f"Rows with at least one finite descriptor: {np.count_nonzero(has_any_finite_x)} / {len(y_all)}.\n"
            f"Rows with all descriptors finite: {np.count_nonzero(finite_x_rows)} / {len(y_all)}.\n"
            f"Descriptor columns used: {len(feature_names)} ({', '.join(feature_names[:8])}"
            + ("..." if len(feature_names) > 8 else "")
            + ").\n"
            "Fix: in Select Columns, keep pActivity as Target/class variable and keep descriptor columns as Features. "
            "Alternatively keep a SMILES column so QSAR Regression can compute RDKit descriptors automatically."
        )
    original_row_count = int(len(y_all))
    usable_row_count = int(np.count_nonzero(finite_rows))
    X_all = X_all[finite_rows]
    y_all = y_all[finite_rows]
    metas_all = metas_all[finite_rows]

    X_all, feature_names, descriptor_cleanup = clean_qsar_descriptor_matrix(X_all, feature_names)
    X_all, feature_names, cap_cleanup = cap_qsar_descriptor_matrix(
        X_all,
        y_all,
        feature_names,
        max_features=getattr(config, "max_model_features", 1000),
    )
    descriptor_cleanup.update(cap_cleanup)
    descriptor_cleanup["descriptor_count"] = int(X_all.shape[1])
    if X_all.shape[1] == 0:
        raise ValueError(
            "No informative descriptor columns remain after removing empty/constant descriptors. "
            "Compute descriptor/fingerprint columns before QSAR or keep a valid SMILES column for RDKit fallback descriptors."
        )

    X_train, X_test, y_train, y_test, metas_train, metas_test = train_test_split(
        X_all,
        y_all,
        metas_all,
        test_size=config.test_size,
        random_state=42,
    )
    if cancelled():
        return None

    algo_name, algo_class = config.algorithms[config.selected_algorithm]
    is_classification = False
    scoring = "r2"
    if algo_name == "Logistic Regression":
        if len(np.unique(y_train)) == 2:
            is_classification = True
            scoring = "accuracy"

    # Avoid CV errors when the user requests more folds than training rows.
    cv_folds = max(2, min(int(config.cv_folds), int(len(y_train))))

    model_ranking_table = None
    auto_mode = bool(getattr(config, "enable_auto_qsar", False)) and not is_classification
    if auto_mode:
        best_pipeline, cv_score, algo_name, model_ranking_table, tuning_info = _run_auto_qsar_model_selection(
            X_train,
            y_train,
            config=config,
            cv_folds=cv_folds,
            scoring=scoring,
        )
        pipeline = best_pipeline
        hp = {}
    else:
        pipeline = _build_modeling_pipeline(
            algo_class=algo_class,
            imputation_method=config.imputation_method,
            normalization_method=config.normalization_method,
            enable_feature_selection=config.enable_feature_selection,
            num_features=config.num_features,
            n_available_features=int(X_train.shape[1]),
            is_classification=is_classification,
        )
        hp = {}
    if (not auto_mode) and config.hyperparameters.strip():
        try:
            hp = json.loads(config.hyperparameters)
        except Exception as exc:  # pragma: no cover - UI surfaced
            raise Exception("Error parsing hyperparameters: " + str(exc)) from exc

    if auto_mode:
        best_pipeline = pipeline
    elif config.tuning_method == 1 and hp:
        tuner = GridSearchCV(
            pipeline,
            param_grid=hp,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            tuner.fit(X_train, y_train)
        if cancelled():
            return None
        best_pipeline = tuner.best_estimator_
        cv_score = tuner.best_score_
        tuning_info = f"Grid Search best CV {scoring}: {cv_score:.3f}\n"
    elif config.tuning_method == 2 and hp:
        tuner = RandomizedSearchCV(
            pipeline,
            param_distributions=hp,
            cv=cv_folds,
            scoring=scoring,
            n_iter=config.n_iter,
            n_jobs=1,
            random_state=42,
            error_score="raise",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            tuner.fit(X_train, y_train)
        if cancelled():
            return None
        best_pipeline = tuner.best_estimator_
        cv_score = tuner.best_score_
        tuning_info = f"Randomized Search best CV {scoring}: {cv_score:.3f}\n"
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring=scoring)
        cv_score = np.mean(cv_scores)
        if cancelled():
            return None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            best_pipeline = pipeline.fit(X_train, y_train)
        tuning_info = f"CV {scoring} (no tuning): {cv_score:.3f}\n"

    if cancelled():
        return None

    if not is_classification:
        train_preds = best_pipeline.predict(X_train)
        test_preds = best_pipeline.predict(X_test)
        train_metrics = {
            "R²": r2_score(y_train, train_preds),
            "RMSE": np.sqrt(mean_squared_error(y_train, train_preds)),
            "MAE": mean_absolute_error(y_train, train_preds),
            "Median AE": median_absolute_error(y_train, train_preds),
            "Explained Variance": explained_variance_score(y_train, train_preds),
        }
        test_metrics = {
            "R²": r2_score(y_test, test_preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, test_preds)),
            "MAE": mean_absolute_error(y_test, test_preds),
            "Median AE": median_absolute_error(y_test, test_preds),
            "Explained Variance": explained_variance_score(y_test, test_preds),
        }
        performance_text = (
            f"{algo_name}: \n \n {tuning_info}\n"
            f"Train R²: {train_metrics['R²']:.3f}, RMSE: {train_metrics['RMSE']:.3f}, MAE: {train_metrics['MAE']:.3f},"
            f" MedAE: {train_metrics['Median AE']:.3f}, Expl.Var: {train_metrics['Explained Variance']:.3f}\n"
            f"Test R²: {test_metrics['R²']:.3f}, RMSE: {test_metrics['RMSE']:.3f}, MAE: {test_metrics['MAE']:.3f}, "
            f"MedAE: {test_metrics['Median AE']:.3f}, Expl.Var: {test_metrics['Explained Variance']:.3f}\n"
        )
    else:
        train_preds = best_pipeline.predict(X_train)
        test_preds = best_pipeline.predict(X_test)
        test_score = best_pipeline.score(X_test, y_test)
        performance_text = f"{algo_name}: {tuning_info} | Test Accuracy: {test_score:.3f}"
        train_metrics = {}
        test_metrics = {"Accuracy": test_score}

    ext_table = None
    external_metrics = {}
    X_ext = y_ext = None
    if external_data is not None:
        ext_prepared = prepare_qsar_model_matrix(external_data, feature_names=feature_names)
        X_ext = ext_prepared["X"]
        y_ext = ext_prepared["y"]
        metas_ext = ext_prepared["metas"]
        finite_ext = np.isfinite(y_ext) & np.any(np.isfinite(X_ext), axis=1)
        X_ext = X_ext[finite_ext]
        y_ext = y_ext[finite_ext]
        metas_ext = metas_ext[finite_ext]
        ext_preds = best_pipeline.predict(X_ext).reshape(-1, 1)
        ext_domain = _result_domain(external_data, feature_names, ext_prepared["target_var"], is_classification=is_classification)
        ext_table = safe_table_from_numpy(ext_domain, X=np.hstack([X_ext, ext_preds]), Y=y_ext.reshape(-1, 1), metas=metas_ext, name="External Results")
        if not is_classification and len(y_ext) > 1:
            ext_preds_full = best_pipeline.predict(X_ext)
            external_metrics = {
                "R²": r2_score(y_ext, ext_preds_full),
                "RMSE": np.sqrt(mean_squared_error(y_ext, ext_preds_full)),
                "MAE": mean_absolute_error(y_ext, ext_preds_full),
                "Median AE": median_absolute_error(y_ext, ext_preds_full),
                "Explained Variance": explained_variance_score(y_ext, ext_preds_full),
            }

    if cancelled():
        return None

    new_domain = _result_domain(data, feature_names, target_var, is_classification=is_classification)
    train_table = safe_table_from_numpy(new_domain, X=np.hstack([X_train, train_preds.reshape(-1, 1)]), Y=y_train.reshape(-1, 1), metas=metas_train, name="QSAR Train Results")
    test_table = safe_table_from_numpy(new_domain, X=np.hstack([X_test, test_preds.reshape(-1, 1)]), Y=y_test.reshape(-1, 1), metas=metas_test, name="QSAR Test Results")

    # ── Applicability domain, VIF, coef stats, permutation test ──────────
    ad_info: dict = {}
    try:
        if not is_classification:
            X_train_t = _transform_features(best_pipeline, X_train)
            estimator = best_pipeline.named_steps.get("regressor", best_pipeline[-1])
            n_obs, n_feat = X_train_t.shape
            linear_diag_notes = []
            if isinstance(estimator, PLSRegression):
                linear_diag_notes.append("PLS latent-space model: skipped OLS coefficient statistics and VIF diagnostics.")
            elif isinstance(estimator, (Lasso, Ridge, ElasticNet)):
                linear_diag_notes.append(
                    f"{type(estimator).__name__} regularized model: skipped OLS coefficient statistics and VIF diagnostics."
                )
            elif n_feat > 256:
                linear_diag_notes.append(
                    f"Skipped OLS coefficient statistics and VIF diagnostics for {n_feat} features (>256 safety limit)."
                )
            elif n_obs <= (n_feat + 1):
                linear_diag_notes.append(
                    f"Skipped OLS coefficient statistics because training rows ({n_obs}) are not greater than features + intercept ({n_feat + 1})."
                )
            else:
                ad_info["vifs"] = compute_vif(X_train_t)
                coef_stats = compute_coef_stats(X_train_t, y_train, estimator)
                if coef_stats is not None:
                    ad_info["coef_stats"] = coef_stats
            if linear_diag_notes:
                ad_info["linear_diagnostics_note"] = " ".join(linear_diag_notes)
    except Exception:
        pass

    result = {
        "model": best_pipeline,
        "model_name": algo_name,
        "requested_model_name": config.algorithms[config.selected_algorithm][0],
        "auto_qsar_used": bool(auto_mode),
        "train_table": train_table,
        "test_table": test_table,
        "external_table": ext_table,
        "pipeline": best_pipeline,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "is_classification": is_classification,
        "performance_text": performance_text,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "external_metrics": external_metrics,
        "cv_score": cv_score,
        "feature_names": feature_names,
        "generated_descriptors": generated_descriptors,
        "target_column": target_var.name,
        "original_row_count": original_row_count,
        "usable_row_count": usable_row_count,
        "removed_row_count": int(original_row_count - usable_row_count),
        "descriptor_cleanup": descriptor_cleanup,
        "model_ranking_table": model_ranking_table,
        **ad_info,
    }
    if bool(getattr(config, "enable_applicability_domain", True)):
        result["applicability_domain_table"] = build_applicability_domain_table(result)
    else:
        result["applicability_domain_table"] = None
    result["modeling_summary_table"] = build_qsar_modeling_summary_table(result)
    if external_data is not None:
        result["X_ext"] = X_ext
        result["y_ext"] = y_ext

    return result


# ── Statistical diagnostics helpers ──────────────────────────────────────────

def _transform_features(pipeline, X: np.ndarray) -> np.ndarray:
    """Apply all pipeline steps except the final estimator."""
    pre = Pipeline(pipeline.steps[:-1]) if len(pipeline.steps) > 1 else None
    return pre.transform(X) if pre is not None else X


def compute_vif(X_t: np.ndarray) -> np.ndarray:
    """Variance Inflation Factor per feature column."""
    from sklearn.linear_model import LinearRegression as _LR
    n, p = X_t.shape
    if p > 256:
        raise ValueError("VIF is disabled for descriptor spaces wider than 256 features.")
    if p < 2:
        return np.ones(p)
    vifs = np.zeros(p)
    for j in range(p):
        Xo = np.delete(X_t, j, axis=1)
        reg = _LR().fit(Xo, X_t[:, j])
        r2  = float(r2_score(X_t[:, j], reg.predict(Xo)))
        vifs[j] = 1.0 / max(1.0 - r2, 1e-12)
    return vifs


def compute_coef_stats(
    X_t: np.ndarray,
    y: np.ndarray,
    estimator,
) -> dict | None:
    """OLS-style SE / t / p for linear models (coef_ attribute). None for non-linear."""
    if isinstance(estimator, (PLSRegression, Lasso, Ridge, ElasticNet)):
        return None
    from scipy.stats import t as _t_dist
    if not hasattr(estimator, "coef_"):
        return None
    coef = np.asarray(estimator.coef_).ravel()
    n, p = X_t.shape
    if p > 256 or n <= (p + 1):
        return None
    if len(coef) != p:
        return None
    intercept = float(getattr(estimator, "intercept_", 0.0))
    beta = np.concatenate([[intercept], coef])
    X1   = np.column_stack([np.ones(n), X_t])
    y_pred = estimator.predict(X_t)
    resid  = y - y_pred
    dof    = max(n - (p + 1), 1)
    sigma2 = float((resid @ resid) / dof)
    XtX = X1.T @ X1
    try:
        inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(XtX)
    se    = np.sqrt(np.maximum(np.diag(inv) * sigma2, 1e-300))
    tvals = beta / np.maximum(se, 1e-300)
    pvals = 2.0 * (1.0 - _t_dist.cdf(np.abs(tvals), df=dof))
    return {"beta": beta, "se": se, "t": tvals, "p": pvals}


def extract_descriptor_coefficients(pipeline, feature_names: Sequence[str]) -> Table | None:
    """Return an Orange Table with one row per (selected) descriptor and its coefficient or importance.

    Columns: descriptor (string), value (float), abs_value (float).
    The 'value' column is labelled 'coefficient' for linear models and 'importance' for tree models.
    Returns None when the estimator exposes neither coef_ nor feature_importances_.
    """
    names = list(feature_names)

    # Narrow names to those that survived feature selection
    if "feature_selection" in pipeline.named_steps:
        selector = pipeline.named_steps["feature_selection"]
        try:
            mask = selector.get_support()
            names = [n for n, keep in zip(names, mask) if keep]
        except Exception:
            pass

    # Locate the final estimator (named 'regressor' or last step)
    estimator = pipeline.named_steps.get("regressor", pipeline[-1])

    values: np.ndarray | None = None
    col_label = "value"

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_).ravel()
        if len(coef) == len(names):
            values = coef
            col_label = "coefficient"
    if values is None and hasattr(estimator, "feature_importances_"):
        imp = np.asarray(estimator.feature_importances_).ravel()
        if len(imp) == len(names):
            values = imp
            col_label = "importance"

    if values is None:
        return None

    # Sort by |value| descending
    order = np.argsort(np.abs(values))[::-1]
    names_sorted  = [names[i]          for i in order]
    values_sorted = [float(values[i])  for i in order]
    abs_sorted    = [abs(float(values[i])) for i in order]

    domain = Domain(
        [ContinuousVariable(col_label), ContinuousVariable("abs_" + col_label)],
        metas=[StringVariable("descriptor")],
    )
    X = np.column_stack([values_sorted, abs_sorted]).astype(float)
    M = np.array([[n] for n in names_sorted], dtype=object)
    return Table.from_numpy(domain, X=X, Y=None, metas=M)

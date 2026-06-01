from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from matplotlib.path import Path
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from rdkit.Chem import Draw
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from chem_inf_widgets.chemcore.qsar.dataset import find_name_var, find_smiles_var
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles


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


def _transform_features(pipeline, X: np.ndarray) -> np.ndarray:
    """Apply all pipeline steps except the final estimator."""
    pre = Pipeline(pipeline.steps[:-1]) if len(pipeline.steps) > 1 else None
    return pre.transform(X) if pre is not None else X


def compute_vif(X_t: np.ndarray) -> np.ndarray:
    """Variance Inflation Factor per feature column."""
    from sklearn.linear_model import LinearRegression as _LR

    _, p = X_t.shape
    if p > 256:
        raise ValueError("VIF is disabled for descriptor spaces wider than 256 features.")
    if p < 2:
        return np.ones(p)
    vifs = np.zeros(p)
    for j in range(p):
        Xo = np.delete(X_t, j, axis=1)
        reg = _LR().fit(Xo, X_t[:, j])
        r2 = float(r2_score(X_t[:, j], reg.predict(Xo)))
        vifs[j] = 1.0 / max(1.0 - r2, 1e-12)
    return vifs


def compute_coef_stats(
    X_t: np.ndarray,
    y: np.ndarray,
    estimator,
) -> dict | None:
    """OLS-style SE / t / p for linear models (coef_ attribute). None for non-linear."""
    from scipy.stats import t as _t_dist

    if isinstance(estimator, (PLSRegression, Lasso, Ridge, ElasticNet)):
        return None
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
    X1 = np.column_stack([np.ones(n), X_t])
    y_pred = estimator.predict(X_t)
    resid = y - y_pred
    dof = max(n - (p + 1), 1)
    sigma2 = float((resid @ resid) / dof)
    xtx = X1.T @ X1
    try:
        inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(xtx)
    se = np.sqrt(np.maximum(np.diag(inv) * sigma2, 1e-300))
    tvals = beta / np.maximum(se, 1e-300)
    pvals = 2.0 * (1.0 - _t_dist.cdf(np.abs(tvals), df=dof))
    return {"beta": beta, "se": se, "t": tvals, "p": pvals}


def extract_descriptor_coefficients(pipeline, feature_names: Sequence[str]) -> Table | None:
    """Return an Orange Table with one row per (selected) descriptor and its coefficient or importance."""
    names = list(feature_names)

    if "feature_selection" in pipeline.named_steps:
        selector = pipeline.named_steps["feature_selection"]
        try:
            mask = selector.get_support()
            names = [n for n, keep in zip(names, mask) if keep]
        except Exception:
            pass

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

    order = np.argsort(np.abs(values))[::-1]
    names_sorted = [names[i] for i in order]
    values_sorted = [float(values[i]) for i in order]
    abs_sorted = [abs(float(values[i])) for i in order]

    domain = Domain(
        [ContinuousVariable(col_label), ContinuousVariable("abs_" + col_label)],
        metas=[StringVariable("descriptor")],
    )
    X = np.column_stack([values_sorted, abs_sorted]).astype(float)
    M = np.array([[n] for n in names_sorted], dtype=object)
    return Table.from_numpy(domain, X=X, Y=None, metas=M)


__all__ = [
    "CompoundPreview",
    "DiagnosticDatasetPayload",
    "DiagnosticPlotData",
    "DiagnosticPlotSpec",
    "DiagnosticSeries",
    "FeatureInspectionPayload",
    "SelectionGalleryPayload",
    "SelectionPublishPayload",
    "_transform_features",
    "build_compound_previews",
    "build_diagnostic_plot_spec",
    "build_feature_inspection_payload",
    "build_selection_gallery_payload",
    "build_selection_publish_payload",
    "compute_coef_stats",
    "compute_vif",
    "diagnostic_payloads_from_result",
    "extract_descriptor_coefficients",
    "lasso_selection_indices",
    "prepare_diagnostic_plot_data",
    "rectangle_selection_indices",
    "selection_overlay_offsets",
    "selection_status_text",
]

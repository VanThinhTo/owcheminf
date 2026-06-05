from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class QSARValidationConfig:
    observed_column: str = "observed"
    predicted_column: str = "predicted"
    split_column: str = "split"
    id_column: str = "compound_id"
    residual_threshold: float | None = None
    z_threshold: float = 3.0


@dataclass(frozen=True)
class QSARValidationResult:
    metrics: pd.DataFrame
    diagnostics: pd.DataFrame
    outliers: pd.DataFrame
    summary: dict[str, Any]



def _resolve_column(data: pd.DataFrame, requested: str, aliases: tuple[str, ...]) -> str:
    """Resolve a user-selected column with robust common QSAR aliases."""
    if requested in data.columns:
        return requested
    normalized = {str(c).strip().lower().replace(" ", "_").replace("-", "_"): c for c in data.columns}
    req_key = str(requested).strip().lower().replace(" ", "_").replace("-", "_")
    if req_key in normalized:
        return normalized[req_key]
    for alias in aliases:
        key = str(alias).strip().lower().replace(" ", "_").replace("-", "_")
        if key in normalized:
            return normalized[key]
    raise ValueError(
        f"Required column '{requested}' not found. Tried aliases: {', '.join(aliases)}. "
        f"Available columns: {', '.join(map(str, data.columns))}"
    )


def _optional_column(data: pd.DataFrame, requested: str, aliases: tuple[str, ...]) -> str | None:
    try:
        return _resolve_column(data, requested, aliases)
    except ValueError:
        return None


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _safe_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or float(np.std(x)) <= 1e-12:
        return (float("nan"), float("nan"))
    slope, intercept = np.polyfit(x, y, deg=1)
    return (float(slope), float(intercept))


def _ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin concordance correlation coefficient for observed/predicted agreement."""
    if len(y_true) < 2:
        return float("nan")
    mean_true = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))
    var_true = float(np.var(y_true, ddof=0))
    var_pred = float(np.var(y_pred, ddof=0))
    cov = float(np.mean((y_true - mean_true) * (y_pred - mean_pred)))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom <= 1e-300:
        return float("nan")
    return float(2.0 * cov / denom)


def _ad_columns(data: pd.DataFrame) -> tuple[str | None, str | None]:
    lowered = {str(c).strip().lower(): c for c in data.columns}
    flag = None
    for name in ("ad_in_domain", "in_domain", "within_ad", "applicability_domain"):
        if name in lowered:
            flag = lowered[name]
            break
    confidence = None
    for name in ("ad_confidence", "confidence", "reliability", "ad_reliability"):
        if name in lowered:
            confidence = lowered[name]
            break
    return flag, confidence


def _as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(float) > 0
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y", "in", "inside", "in_domain", "within"})


def _metrics_for_group(group_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return {
            "group": group_name,
            "n": 0,
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "ccc": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
        }
    residual = yt - yp
    slope, intercept = _safe_slope_intercept(yp, yt)
    return {
        "group": group_name,
        "n": int(len(yt)),
        "r2": float(r2_score(yt, yp)) if len(yt) > 1 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "bias": float(np.mean(residual)),
        "residual_sd": float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
        "median_abs_error": float(np.median(np.abs(residual))),
        "p95_abs_residual": float(np.quantile(np.abs(residual), 0.95)),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "pearson_r": _safe_corrcoef(yt, yp),
        "ccc": _ccc(yt, yp),
        "slope": slope,
        "intercept": intercept,
    }


def validate_qsar_predictions(
    df: pd.DataFrame,
    config: QSARValidationConfig | None = None,
) -> QSARValidationResult:
    config = config or QSARValidationConfig()
    data = df.copy()
    observed_col = _resolve_column(
        data,
        config.observed_column,
        ("observed", "actual", "actual_value", "y_true", "experimental", "measured", "pActivity", "activity"),
    )
    predicted_col = _resolve_column(
        data,
        config.predicted_column,
        ("predicted", "prediction", "predicted_value", "y_pred", "predicted_pActivity", "predicted_activity"),
    )
    split_col = _optional_column(data, config.split_column, ("split", "dataset", "partition", "subset", "set"))
    id_col = _optional_column(data, config.id_column, ("compound_id", "id", "name", "molecule_id", "mol_id"))

    # Keep canonical column names so downstream Orange widgets and reports work
    # even when the input uses names such as actual_value or predicted_pActivity.
    data["observed"] = pd.to_numeric(data[observed_col], errors="coerce")
    data["predicted"] = pd.to_numeric(data[predicted_col], errors="coerce")
    if split_col is not None and "split" not in data.columns:
        data["split"] = data[split_col].astype(str)
    if id_col is not None and "compound_id" not in data.columns:
        data["compound_id"] = data[id_col].astype(str)

    data["residual"] = data["observed"] - data["predicted"]
    data["abs_residual"] = data["residual"].abs()
    data["signed_error"] = data["predicted"] - data["observed"]
    residual_sd = float(data["residual"].std(ddof=1)) if data["residual"].notna().sum() > 1 else 0.0
    residual_mean = float(data["residual"].mean()) if data["residual"].notna().sum() else 0.0
    if residual_sd > 0:
        data["residual_z"] = (data["residual"] - residual_mean) / residual_sd
    else:
        data["residual_z"] = 0.0
    data["abs_residual_z"] = data["residual_z"].abs()
    if config.residual_threshold is None:
        # Data-adaptive default: do not let the threshold collapse for nearly perfect
        # examples, but otherwise follow roughly a 2-sigma review band.
        threshold = float(max(1.0, 2.0 * residual_sd)) if residual_sd > 0 else 1.0
    else:
        threshold = float(config.residual_threshold)
    data["large_residual"] = data["abs_residual"] > threshold
    data["z_outlier"] = data["abs_residual_z"] > float(config.z_threshold)
    data["within_residual_threshold"] = (~data["large_residual"]).astype(int)

    ad_flag_col, ad_conf_col = _ad_columns(data)
    if ad_flag_col is not None:
        data["ad_in_domain_bool"] = _as_bool_series(data[ad_flag_col])
        data["ad_flag"] = np.where(data["ad_in_domain_bool"], "in-domain", "out-of-domain")
    else:
        data["ad_in_domain_bool"] = True
        data["ad_flag"] = "not provided"

    if ad_conf_col is not None:
        data["ad_confidence_source"] = data[ad_conf_col].astype(str)
        data["ad_confidence_normalized"] = data[ad_conf_col].astype(str).str.strip().str.lower()
        data["low_ad_confidence"] = data["ad_confidence_normalized"].isin({"low", "poor", "outside", "out-of-domain"})
    else:
        data["low_ad_confidence"] = False

    data["validation_flag"] = np.where(
        data["large_residual"] | data["z_outlier"] | (~data["ad_in_domain_bool"]) | data["low_ad_confidence"],
        "review",
        "ok",
    )
    data["review_reason"] = ""
    data.loc[data["large_residual"], "review_reason"] += "large residual; "
    data.loc[data["z_outlier"], "review_reason"] += "residual z-outlier; "
    data.loc[~data["ad_in_domain_bool"], "review_reason"] += "outside applicability domain; "
    data.loc[data["low_ad_confidence"], "review_reason"] += "low AD confidence; "
    data["review_reason"] = data["review_reason"].str.strip().str.rstrip(";").replace("", "ok")

    severity_score = (
        data["large_residual"].astype(int)
        + data["z_outlier"].astype(int)
        + (~data["ad_in_domain_bool"]).astype(int)
        + data["low_ad_confidence"].astype(int)
    )
    data["validation_severity"] = np.select(
        [severity_score >= 2, severity_score == 1],
        ["critical", "warning"],
        default="ok",
    )

    metric_rows = [_metrics_for_group("all", data["observed"].to_numpy(float), data["predicted"].to_numpy(float))]
    if "split" in data.columns:
        for split, sub in data.groupby("split"):
            metric_rows.append(_metrics_for_group(str(split), sub["observed"].to_numpy(float), sub["predicted"].to_numpy(float)))
    metrics = pd.DataFrame(metric_rows)
    outliers = data[data["validation_flag"] == "review"].copy()
    ad_coverage = float(data["ad_in_domain_bool"].mean()) if len(data) and ad_flag_col is not None else None
    summary = {
        "n_rows": int(len(data)),
        "n_outliers": int(len(outliers)),
        "n_large_residuals": int(data["large_residual"].sum()),
        "n_z_outliers": int(data["z_outlier"].sum()),
        "n_outside_ad": int((~data["ad_in_domain_bool"]).sum()) if ad_flag_col is not None else 0,
        "n_low_ad_confidence": int(data["low_ad_confidence"].sum()),
        "n_warning": int((data["validation_severity"] == "warning").sum()),
        "n_critical": int((data["validation_severity"] == "critical").sum()),
        "ad_coverage": ad_coverage,
        "residual_threshold": threshold,
        "z_threshold": float(config.z_threshold),
        "observed_column_used": str(observed_col),
        "predicted_column_used": str(predicted_col),
        "split_column_used": str(split_col) if split_col is not None else None,
        "id_column_used": str(id_col) if id_col is not None else None,
        "overall_metrics": metric_rows[0],
    }
    return QSARValidationResult(metrics=metrics, diagnostics=data, outliers=outliers, summary=summary)


def write_qsar_validation_outputs(result: QSARValidationResult, out_prefix: str | Path) -> dict[str, str]:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics_csv": str(prefix.with_suffix(".validation_metrics.csv")),
        "diagnostics_csv": str(prefix.with_suffix(".residual_diagnostics.csv")),
        "outliers_csv": str(prefix.with_suffix(".outliers.csv")),
        "summary_json": str(prefix.with_suffix(".validation_summary.json")),
    }
    result.metrics.to_csv(paths["metrics_csv"], index=False)
    result.diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    result.outliers.to_csv(paths["outliers_csv"], index=False)
    Path(paths["summary_json"]).write_text(json.dumps(result.summary, indent=2, sort_keys=True), encoding="utf-8")
    return paths

from __future__ import annotations

import numpy as np
import pandas as pd
from Orange.data import Table

from chem_inf_widgets.chemcore.qsar.diagnostics import _transform_features
from chem_inf_widgets.chemcore.services.orange_table_utils import records_to_orange_table


def _as_prediction_vector(pipeline, X: np.ndarray | None) -> np.ndarray:
    if X is None:
        return np.asarray([], dtype=float)
    return np.asarray(pipeline.predict(X), dtype=float).ravel()


def _model_feature_frame(X_t: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(X_t, dtype=float), columns=feature_names)


def _append_dataset_rows(
    frames: list[pd.DataFrame],
    *,
    X_t: np.ndarray,
    feature_names: list[str],
    dataset: str,
    y: np.ndarray | None,
    predicted: np.ndarray,
) -> None:
    if X_t is None or len(X_t) == 0:
        return
    frame = _model_feature_frame(X_t, feature_names)
    n_rows = len(frame)
    actual = np.full(n_rows, np.nan, dtype=float) if y is None else np.asarray(y, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    frame.insert(0, "row_index", np.arange(n_rows, dtype=int))
    frame.insert(0, "dataset", dataset)
    frame["actual"] = actual
    frame["observed"] = actual
    frame["predicted"] = predicted
    finite_actual = np.isfinite(actual)
    residual = np.full(n_rows, np.nan, dtype=float)
    residual[finite_actual] = actual[finite_actual] - predicted[finite_actual]
    frame["residual"] = residual
    frame["abs_residual"] = np.abs(residual)
    frames.append(frame)


def _workbench_config(feature_names: list[str], n_rows: int):
    from chem_inf_widgets.chemcore.services.ad_workbench_service import ADWorkbenchConfig

    # Mahalanobis is valuable but can become unstable in very high-dimensional
    # transformed feature spaces. Enable it when the feature space is compact
    # enough for a useful covariance estimate; Williams + kNN remain active.
    use_mahalanobis = len(feature_names) <= max(3, min(50, n_rows - 2))
    return ADWorkbenchConfig(
        combine_mode="and",
        use_williams=True,
        use_knn=True,
        use_mahalanobis=bool(use_mahalanobis),
        knn_k=5,
        knn_quantile=0.95,
        maha_alpha=0.95,
        maha_use_chi2=True,
        feature_columns=tuple(feature_names),
    )


def build_applicability_domain_table(result: dict) -> Table | None:
    """Build the QSAR Applicability Domain table using the AD Workbench engine.

    QSAR previously used a smaller local Williams/kNN implementation.  The
    Workbench service is now the canonical implementation, so QSAR outputs get
    the same richer diagnostics: Williams leverage, kNN distance, optional
    Mahalanobis distance, per-method flags, normalized ratios, AD margin,
    confidence tier and review reason.
    """
    if not result or result.get("pipeline") is None:
        return None
    if result.get("is_classification"):
        return None

    try:
        pipeline = result["pipeline"]
        X_train_t = np.asarray(_transform_features(pipeline, result["X_train"]), dtype=float)
        n_train, n_features = X_train_t.shape
        if n_train < 3 or n_features < 1:
            return None

        feature_names = [f"model_feature_{i + 1:04d}" for i in range(n_features)]
        y_train = np.asarray(result["y_train"], dtype=float)
        train_pred = _as_prediction_vector(pipeline, result["X_train"])

        query_frames: list[pd.DataFrame] = []
        _append_dataset_rows(
            query_frames,
            X_t=X_train_t,
            feature_names=feature_names,
            dataset="train",
            y=y_train,
            predicted=train_pred,
        )

        X_test = result.get("X_test")
        if X_test is not None:
            X_test_t = np.asarray(_transform_features(pipeline, X_test), dtype=float)
            _append_dataset_rows(
                query_frames,
                X_t=X_test_t,
                feature_names=feature_names,
                dataset="test",
                y=np.asarray(result.get("y_test"), dtype=float) if result.get("y_test") is not None else None,
                predicted=_as_prediction_vector(pipeline, X_test),
            )

        X_ext = result.get("X_ext")
        if X_ext is not None:
            X_ext_t = np.asarray(_transform_features(pipeline, X_ext), dtype=float)
            _append_dataset_rows(
                query_frames,
                X_t=X_ext_t,
                feature_names=feature_names,
                dataset="external",
                y=np.asarray(result.get("y_ext"), dtype=float) if result.get("y_ext") is not None else None,
                predicted=_as_prediction_vector(pipeline, X_ext),
            )

        if not query_frames:
            return None

        reference_df = _model_feature_frame(X_train_t, feature_names)
        query_df = pd.concat(query_frames, ignore_index=True)
        from chem_inf_widgets.chemcore.services.ad_workbench_service import evaluate_applicability_domain_workbench

        wb_result = evaluate_applicability_domain_workbench(
            reference_df,
            query_df,
            _workbench_config(feature_names, n_train),
        )
        scored = wb_result.scored_query.copy()

        # Lowercase aliases make the table immediately usable by the QSAR
        # Validation Dashboard and report generator, while preserving Workbench
        # column names for compatibility with the standalone AD Workbench.
        scored["ad_in_domain"] = scored["AD_in_domain"].astype(int)
        scored["ad_outlier"] = (~scored["AD_in_domain"].astype(bool)).astype(int)
        scored["ad_confidence"] = scored["AD_confidence"].astype(str)
        scored["ad_reason"] = scored["AD_reason"].astype(str)
        scored["ad_leverage_ratio"] = scored["AD_leverage_ratio"]
        scored["ad_distance_ratio"] = scored["AD_knn_ratio"]
        scored["ad_mahalanobis_ratio"] = scored["AD_maha_ratio"]
        scored["ad_margin"] = scored["AD_margin"]

        drop_features = [c for c in feature_names if c in scored.columns]
        records = scored.drop(columns=drop_features, errors="ignore").to_dict(orient="records")
        return records_to_orange_table(
            records,
            meta_columns=(
                "dataset",
                "AD_dataset_role",
                "AD_confidence",
                "AD_reason",
                "ad_confidence",
                "ad_reason",
            ),
            name="QSAR Applicability Domain",
        )
    except Exception:
        return None


__all__ = ["build_applicability_domain_table"]

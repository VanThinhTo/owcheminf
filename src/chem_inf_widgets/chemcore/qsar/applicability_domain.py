from __future__ import annotations

import numpy as np
from Orange.data import Table
from sklearn.neighbors import NearestNeighbors

from chem_inf_widgets.chemcore.qsar.diagnostics import _transform_features
from chem_inf_widgets.chemcore.services.orange_table_utils import records_to_orange_table


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
                rows.append(
                    {
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
                    }
                )

        add_dataset("train", result["X_train"], y_train, train_pred)
        test_pred = np.asarray(pipeline.predict(result["X_test"]), dtype=float).ravel()
        add_dataset("test", result["X_test"], np.asarray(result["y_test"], dtype=float), test_pred)
        if result.get("X_ext") is not None and result.get("y_ext") is not None:
            ext_pred = np.asarray(pipeline.predict(result["X_ext"]), dtype=float).ravel()
            add_dataset("external", result["X_ext"], np.asarray(result["y_ext"], dtype=float), ext_pred)
        return records_to_orange_table(rows, name="QSAR Applicability Domain") if rows else None
    except Exception:
        return None


__all__ = ["build_applicability_domain_table"]

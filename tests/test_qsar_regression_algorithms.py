from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("Orange")

from chem_inf_widgets.chemcore.qsar.algorithms import (
    _build_modeling_pipeline,
    _run_auto_qsar_model_selection,
    available_algorithms,
    build_run_config,
)


def test_build_modeling_pipeline_adds_safe_fallback_imputer_and_feature_selection():
    pipeline = _build_modeling_pipeline(
        algo_class=available_algorithms()[0][1],
        imputation_method=0,
        normalization_method=1,
        enable_feature_selection=True,
        num_features=2,
        n_available_features=3,
        is_classification=False,
    )

    assert list(pipeline.named_steps) == ["imputer", "scaler", "feature_selection", "regressor"]
    assert pipeline.named_steps["imputer"].strategy == "mean"
    assert pipeline.named_steps["feature_selection"].k == 2


def test_run_auto_qsar_model_selection_returns_fitted_pipeline_and_ranking_table():
    X_train = np.array(
        [
            [0.1, 1.0, 2.0],
            [0.2, 0.9, 2.1],
            [0.3, 0.8, 2.2],
            [0.4, 0.7, 2.3],
            [0.5, 0.6, 2.4],
            [0.6, 0.5, 2.5],
        ],
        dtype=float,
    )
    y_train = np.array([4.1, 4.3, 4.5, 4.8, 5.0, 5.2], dtype=float)
    config = build_run_config(
        selected_algorithm=0,
        normalization_method=0,
        imputation_method=1,
        cv_folds=2,
        test_size=0.3,
        tuning_method=0,
        n_iter=5,
        hyperparameters="",
        enable_feature_selection=False,
        num_features=3,
        algorithms=available_algorithms(),
        enable_auto_qsar=True,
    )

    pipeline, cv_score, model_name, ranking_table, info = _run_auto_qsar_model_selection(
        X_train,
        y_train,
        config=config,
        cv_folds=2,
        scoring="r2",
    )

    assert pipeline is not None
    assert np.isfinite(cv_score)
    assert model_name
    assert ranking_table is not None
    assert len(ranking_table) >= 1
    assert model_name in info

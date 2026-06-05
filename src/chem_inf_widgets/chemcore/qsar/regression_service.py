from __future__ import annotations

import json
import warnings
from typing import Callable, Optional, Sequence

import numpy as np
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, Table
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split

from chem_inf_widgets.chemcore.qsar.algorithms import (
    QSARRunConfig,
    _build_modeling_pipeline,
    _run_auto_qsar_model_selection,
)
from chem_inf_widgets.chemcore.qsar.applicability_domain import build_applicability_domain_table
from chem_inf_widgets.chemcore.qsar.dataset import (
    cap_qsar_descriptor_matrix,
    clean_qsar_descriptor_matrix,
    prepare_qsar_model_matrix,
)
from chem_inf_widgets.chemcore.qsar.diagnostics import _transform_features, compute_coef_stats, compute_vif
from chem_inf_widgets.chemcore.qsar.validation import build_qsar_modeling_summary_table
from chem_inf_widgets.chemcore.services.orange_table_utils import safe_table_from_numpy


def _result_domain(source_data: Table, feature_names: Sequence[str], target_var, *, is_classification: bool = False) -> Domain:
    attributes = [ContinuousVariable(str(name)) for name in feature_names]
    pred_var = ContinuousVariable("Predicted") if not is_classification else DiscreteVariable("Predicted")
    class_vars = [ContinuousVariable(target_var.name)] if isinstance(target_var, ContinuousVariable) else [target_var]
    return Domain(attributes + [pred_var], class_vars, source_data.domain.metas)


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
    if X_all.ndim == 2 and X_all.shape[1]:
        has_any_finite_x = np.any(np.isfinite(X_all), axis=1)
        finite_x_rows = np.all(np.isfinite(X_all), axis=1)
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
    if algo_name == "Logistic Regression" and len(np.unique(y_train)) == 2:
        is_classification = True
        scoring = "accuracy"

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
        ext_table = safe_table_from_numpy(
            ext_domain,
            X=np.hstack([X_ext, ext_preds]),
            Y=y_ext.reshape(-1, 1),
            metas=metas_ext,
            name="External Results",
        )
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
    train_table = safe_table_from_numpy(
        new_domain,
        X=np.hstack([X_train, train_preds.reshape(-1, 1)]),
        Y=y_train.reshape(-1, 1),
        metas=metas_train,
        name="QSAR Train Results",
    )
    test_table = safe_table_from_numpy(
        new_domain,
        X=np.hstack([X_test, test_preds.reshape(-1, 1)]),
        Y=y_test.reshape(-1, 1),
        metas=metas_test,
        name="QSAR Test Results",
    )

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
    if external_data is not None:
        result["X_ext"] = X_ext
        result["y_ext"] = y_ext

    if bool(getattr(config, "enable_applicability_domain", True)):
        result["applicability_domain_table"] = build_applicability_domain_table(result)
    else:
        result["applicability_domain_table"] = None
    result["modeling_summary_table"] = build_qsar_modeling_summary_table(result)

    return result


__all__ = ["run_qsar_regression"]

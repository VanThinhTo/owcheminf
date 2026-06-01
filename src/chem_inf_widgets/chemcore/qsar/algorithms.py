from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from Orange.data import Table
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from chem_inf_widgets.chemcore.services.orange_table_utils import records_to_orange_table
from chem_inf_widgets.chemcore.services.safe_feature_selection import safe_f_regression

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    optim = None
    TORCH_AVAILABLE = False


class TorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layer_size=256, epochs=200, lr=0.01, batch_size=32, random_state=42):
        self.hidden_layer_size = hidden_layer_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise ImportError("Deep Learning Regression requires the optional 'torch' package.")
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        input_dim = X_tensor.shape[1]

        self.model_ = nn.Sequential(
            nn.Linear(input_dim, self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, 1),
        )
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for _epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        if not TORCH_AVAILABLE:
            raise ImportError("Deep Learning Regression requires the optional 'torch' package.")
        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model_(X_tensor)
        return predictions.numpy().ravel()


@dataclass(frozen=True)
class QSARRunConfig:
    selected_algorithm: int
    normalization_method: int
    imputation_method: int
    cv_folds: int
    test_size: float
    tuning_method: int
    n_iter: int
    hyperparameters: str
    enable_feature_selection: bool
    num_features: int
    max_model_features: int
    enable_applicability_domain: bool
    enable_auto_qsar: bool
    algorithms: Sequence[tuple[str, type]]


def available_algorithms():
    algorithms = [
        ("Random Forest", RandomForestRegressor),
        ("Support Vector Regression", SVR),
        ("Gradient Boosting", GradientBoostingRegressor),
        ("PLS Regression", PLSRegression),
        ("Decision Tree Regression", DecisionTreeRegressor),
        ("Lasso Regression", Lasso),
        ("Ridge Regression", Ridge),
        ("Elastic Net", ElasticNet),
    ]
    if TORCH_AVAILABLE:
        algorithms.append(("Deep Learning Regression", TorchRegressor))
    return algorithms


def build_run_config(
    *,
    selected_algorithm: int,
    normalization_method: int,
    imputation_method: int,
    cv_folds: int,
    test_size: float,
    tuning_method: int,
    n_iter: int,
    hyperparameters: str,
    enable_feature_selection: bool,
    num_features: int,
    algorithms: Sequence[tuple[str, type]],
    max_model_features: int = 1000,
    enable_applicability_domain: bool = True,
    enable_auto_qsar: bool = False,
) -> QSARRunConfig:
    return QSARRunConfig(
        selected_algorithm=selected_algorithm,
        normalization_method=normalization_method,
        imputation_method=imputation_method,
        cv_folds=cv_folds,
        test_size=test_size,
        tuning_method=tuning_method,
        n_iter=n_iter,
        hyperparameters=hyperparameters,
        enable_feature_selection=enable_feature_selection,
        num_features=num_features,
        max_model_features=int(max_model_features or 0),
        enable_applicability_domain=bool(enable_applicability_domain),
        enable_auto_qsar=bool(enable_auto_qsar),
        algorithms=algorithms,
    )


def _make_safe_regressor(algo_class: type):
    model = algo_class()
    if isinstance(model, (Lasso, ElasticNet)):
        model.set_params(max_iter=10000, random_state=42)
    elif isinstance(model, RandomForestRegressor):
        model.set_params(n_jobs=1, random_state=42)
    elif isinstance(model, GradientBoostingRegressor):
        model.set_params(random_state=42)
    elif isinstance(model, DecisionTreeRegressor):
        model.set_params(random_state=42)
    return model


def _build_modeling_pipeline(
    *,
    algo_class: type,
    imputation_method: int,
    normalization_method: int,
    enable_feature_selection: bool,
    num_features: int,
    n_available_features: int,
    is_classification: bool = False,
) -> Pipeline:
    steps = []
    if imputation_method != 0:
        strat = {1: "mean", 2: "median", 3: "most_frequent"}.get(imputation_method, "mean")
        steps.append(("imputer", SimpleImputer(strategy=strat)))
    else:
        steps.append(("imputer", SimpleImputer(strategy="mean")))

    if normalization_method == 1:
        steps.append(("scaler", StandardScaler()))
    elif normalization_method == 2:
        steps.append(("scaler", MinMaxScaler()))

    if enable_feature_selection:
        score_func = mutual_info_classif if is_classification else safe_f_regression
        k_features = max(1, min(int(num_features), int(n_available_features))) if n_available_features else int(num_features)
        steps.append(("feature_selection", SelectKBest(score_func=score_func, k=k_features)))

    steps.append(("regressor", _make_safe_regressor(algo_class)))
    return Pipeline(steps)


def _auto_qsar_candidates() -> list[tuple[str, type]]:
    return [
        ("Random Forest", RandomForestRegressor),
        ("Gradient Boosting", GradientBoostingRegressor),
        ("Ridge Regression", Ridge),
        ("Elastic Net", ElasticNet),
        ("Support Vector Regression", SVR),
        ("PLS Regression", PLSRegression),
    ]


def _run_auto_qsar_model_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: QSARRunConfig,
    cv_folds: int,
    scoring: str,
) -> tuple[Pipeline, float, str, Table | None, str]:
    rows = []
    best_score = -np.inf
    best_name = ""
    best_pipe: Pipeline | None = None

    for model_name, algo_class in _auto_qsar_candidates():
        pipe = _build_modeling_pipeline(
            algo_class=algo_class,
            imputation_method=config.imputation_method,
            normalization_method=config.normalization_method,
            enable_feature_selection=config.enable_feature_selection,
            num_features=config.num_features,
            n_available_features=int(X_train.shape[1]),
            is_classification=False,
        )
        if algo_class is PLSRegression:
            n_comp = max(1, min(2, int(X_train.shape[1]), max(1, int(len(y_train)) - 1)))
            pipe.named_steps["regressor"].set_params(n_components=n_comp)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=1, error_score=np.nan)
            mean_score = float(np.nanmean(scores)) if np.any(np.isfinite(scores)) else float("nan")
            std_score = float(np.nanstd(scores)) if np.any(np.isfinite(scores)) else float("nan")
        except Exception as exc:
            mean_score = float("nan")
            std_score = float("nan")
            rows.append({
                "rank": "",
                "model": model_name,
                "cv_score_mean": mean_score,
                "cv_score_std": std_score,
                "selected": 0,
                "status": f"failed: {exc}",
            })
            continue

        rows.append({
            "rank": "",
            "model": model_name,
            "cv_score_mean": mean_score,
            "cv_score_std": std_score,
            "selected": 0,
            "status": "ok" if np.isfinite(mean_score) else "failed/no finite CV score",
        })
        if np.isfinite(mean_score) and mean_score > best_score:
            best_score = mean_score
            best_name = model_name
            best_pipe = pipe

    if best_pipe is None:
        raise ValueError("Auto QSAR failed: no candidate model produced a finite CV score.")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        best_pipe.fit(X_train, y_train)

    ok_rows = [r for r in rows if np.isfinite(float(r["cv_score_mean"]))]
    ranked_names = {id(r): i + 1 for i, r in enumerate(sorted(ok_rows, key=lambda r: float(r["cv_score_mean"]), reverse=True))}
    for r in rows:
        r["rank"] = ranked_names.get(id(r), "")
        r["selected"] = 1 if r["model"] == best_name else 0
    table = records_to_orange_table(rows, name="Auto QSAR Model Ranking") if rows else None
    info = f"Auto QSAR selected {best_name}; best CV {scoring}: {best_score:.3f}\n"
    return best_pipe, best_score, best_name, table, info


__all__ = [
    "QSARRunConfig",
    "TORCH_AVAILABLE",
    "TorchRegressor",
    "_build_modeling_pipeline",
    "_make_safe_regressor",
    "_run_auto_qsar_model_selection",
    "available_algorithms",
    "build_run_config",
]

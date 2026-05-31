from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from chem_inf_widgets.chemcore.services.orange_table_utils import records_to_orange_table, safe_table_from_numpy
from chem_inf_widgets.chemcore.services.report_table_utils import summary_rows_to_table


_PREFERRED_TARGET_CANDIDATES = (
    "pactivity",
    "activity",
    "target",
    "y",
    "response",
    "value",
)


@dataclass(frozen=True)
class SymbolicRegressionConfig:
    max_features: int = 6
    max_terms: int = 4
    cv_folds: int = 5
    include_square: bool = True
    include_cube: bool = False
    include_log: bool = True
    include_sqrt: bool = True
    include_inverse: bool = True
    include_interactions: bool = True
    random_state: int = 0
    min_cv_improvement: float = 1e-4


@dataclass(frozen=True)
class SymbolicTerm:
    label: str
    kind: str
    feature_indices: tuple[int, ...]
    feature_names: tuple[str, ...]
    complexity: int


@dataclass
class SymbolicRegressionModel:
    target_name: str
    input_feature_names: tuple[str, ...]
    selected_terms: tuple[SymbolicTerm, ...]
    coefficients: tuple[float, ...]
    intercept: float
    imputer_statistics: tuple[float, ...]

    def predict(self, data: Table | np.ndarray) -> np.ndarray:
        if isinstance(data, Table):
            X = _extract_feature_matrix(data, self.input_feature_names)
        else:
            X = np.asarray(data, dtype=float)
        X_imp = _apply_imputer_statistics(X, self.imputer_statistics)
        design = _design_matrix_from_terms(X_imp, self.selected_terms)
        if design.shape[1] == 0:
            return np.full(X_imp.shape[0], float(self.intercept), dtype=float)
        coef = np.asarray(self.coefficients, dtype=float)
        return (design @ coef + float(self.intercept)).astype(float)

    @property
    def expression(self) -> str:
        return build_expression(self.target_name, self.intercept, self.coefficients, self.selected_terms)


@dataclass
class SymbolicRegressionResult:
    model: SymbolicRegressionModel
    expression: str
    target_name: str
    candidate_terms: tuple[SymbolicTerm, ...]
    selected_terms: tuple[SymbolicTerm, ...]
    selected_feature_names: tuple[str, ...]
    train_metrics: dict[str, float]
    cv_metrics: dict[str, float]
    predictions: np.ndarray
    predictions_table: Table
    term_table: Table
    summary_table: Table


def continuous_target_candidates(data: Table) -> list[ContinuousVariable]:
    out: list[ContinuousVariable] = []
    for variable in list(data.domain.class_vars) + list(data.domain.attributes):
        if getattr(variable, "is_continuous", False):
            out.append(variable)
    seen: set[str] = set()
    unique: list[ContinuousVariable] = []
    for variable in out:
        if variable.name not in seen:
            unique.append(variable)
            seen.add(variable.name)
    return unique


def preferred_target_name(data: Table) -> str:
    candidates = continuous_target_candidates(data)
    if not candidates:
        return ""
    if data.domain.class_vars:
        first_class = data.domain.class_vars[0]
        if getattr(first_class, "is_continuous", False):
            return first_class.name
    by_name = {variable.name.lower(): variable.name for variable in candidates}
    for candidate in _PREFERRED_TARGET_CANDIDATES:
        if candidate in by_name:
            return by_name[candidate]
    return candidates[0].name


def fit_symbolic_regression(
    data: Table,
    *,
    target_name: str,
    config: SymbolicRegressionConfig | None = None,
) -> SymbolicRegressionResult:
    if data is None or len(data) == 0:
        raise ValueError("No input data provided.")

    cfg = config or SymbolicRegressionConfig()
    target_var = _resolve_target_variable(data, target_name)
    X_raw, y, feature_names = _extract_xy(data, target_var.name)
    if X_raw.shape[1] == 0:
        raise ValueError("No continuous descriptor columns found in attributes.")

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw).astype(float)

    valid_mask = np.nanstd(X_imp, axis=0) > 1e-12
    if not np.any(valid_mask):
        raise ValueError("All continuous descriptor columns are constant or empty.")
    X_valid = X_imp[:, valid_mask]
    valid_feature_names = [name for name, keep in zip(feature_names, valid_mask) if keep]

    ranked_idx = _rank_feature_indices(X_valid, y)
    top_n = max(1, min(int(cfg.max_features), len(ranked_idx)))
    selected_feature_idx = ranked_idx[:top_n]
    X_top = X_valid[:, selected_feature_idx]
    top_feature_names = [valid_feature_names[idx] for idx in selected_feature_idx]

    candidate_terms = build_candidate_terms(top_feature_names, cfg)
    candidate_values = [_evaluate_term(term, X_top) for term in candidate_terms]
    usable_terms: list[SymbolicTerm] = []
    usable_values: list[np.ndarray] = []
    for term, values in zip(candidate_terms, candidate_values):
        if np.nanstd(values) <= 1e-12:
            continue
        usable_terms.append(term)
        usable_values.append(values.astype(float))

    baseline_cv = _cross_validated_metrics(np.empty((len(y), 0), dtype=float), y, cfg.cv_folds, cfg.random_state)
    current_score = _selection_score(baseline_cv)
    current_r2 = baseline_cv.get("r2", float("nan"))

    chosen_indices: list[int] = []
    max_terms = max(1, int(cfg.max_terms))
    for _ in range(min(max_terms, len(usable_terms))):
        best_idx: Optional[int] = None
        best_metrics: Optional[dict[str, float]] = None
        best_score = current_score
        for idx, values in enumerate(usable_values):
            if idx in chosen_indices:
                continue
            cols = [usable_values[j] for j in chosen_indices] + [values]
            design = np.column_stack(cols).astype(float)
            metrics = _cross_validated_metrics(design, y, cfg.cv_folds, cfg.random_state)
            score = _selection_score(metrics)
            if score > best_score + cfg.min_cv_improvement:
                best_idx = idx
                best_metrics = metrics
                best_score = score
        if best_idx is None or best_metrics is None:
            break
        chosen_indices.append(best_idx)
        current_score = best_score
        current_r2 = best_metrics.get("r2", current_r2)

    selected_terms = tuple(usable_terms[idx] for idx in chosen_indices)
    design = _design_matrix_from_terms(X_top, selected_terms)

    if design.shape[1] == 0:
        intercept = float(np.mean(y))
        coefficients = np.empty((0,), dtype=float)
        predictions = np.full(len(y), intercept, dtype=float)
    else:
        model = LinearRegression()
        model.fit(design, y)
        intercept = float(model.intercept_)
        coefficients = np.asarray(model.coef_, dtype=float)
        predictions = model.predict(design).astype(float)

    train_metrics = _regression_metrics(y, predictions)
    cv_metrics = dict(baseline_cv if design.shape[1] == 0 else _cross_validated_metrics(design, y, cfg.cv_folds, cfg.random_state))
    if not np.isfinite(cv_metrics.get("r2", float("nan"))):
        cv_metrics["r2"] = current_r2

    fitted_model = SymbolicRegressionModel(
        target_name=target_var.name,
        input_feature_names=tuple(top_feature_names),
        selected_terms=selected_terms,
        coefficients=tuple(float(value) for value in coefficients.tolist()),
        intercept=intercept,
        imputer_statistics=tuple(float(value) for value in np.asarray(imputer.statistics_, dtype=float)[valid_mask][selected_feature_idx].tolist()),
    )
    expression = fitted_model.expression

    return SymbolicRegressionResult(
        model=fitted_model,
        expression=expression,
        target_name=target_var.name,
        candidate_terms=tuple(usable_terms),
        selected_terms=selected_terms,
        selected_feature_names=tuple(top_feature_names),
        train_metrics=train_metrics,
        cv_metrics=cv_metrics,
        predictions=predictions,
        predictions_table=predictions_table(data, target_var.name, predictions),
        term_table=term_table(selected_terms, coefficients, y, design),
        summary_table=summary_table(
            target_name=target_var.name,
            expression=expression,
            n_rows=len(data),
            n_input_features=len(feature_names),
            n_screened_features=len(top_feature_names),
            n_candidate_terms=len(usable_terms),
            n_selected_terms=len(selected_terms),
            train_metrics=train_metrics,
            cv_metrics=cv_metrics,
        ),
    )


def build_candidate_terms(feature_names: Sequence[str], config: SymbolicRegressionConfig) -> list[SymbolicTerm]:
    terms: list[SymbolicTerm] = []
    for idx, name in enumerate(feature_names):
        terms.append(SymbolicTerm(name, "linear", (idx,), (name,), 1))
        if config.include_square:
            terms.append(SymbolicTerm(f"{name}^2", "square", (idx,), (name,), 2))
        if config.include_cube:
            terms.append(SymbolicTerm(f"{name}^3", "cube", (idx,), (name,), 3))
        if config.include_sqrt:
            terms.append(SymbolicTerm(f"sign({name})*sqrt(abs({name}))", "signed_sqrt", (idx,), (name,), 2))
        if config.include_log:
            terms.append(SymbolicTerm(f"sign({name})*log1p(abs({name}))", "signed_log", (idx,), (name,), 2))
        if config.include_inverse:
            terms.append(SymbolicTerm(f"1/(1+abs({name}))", "inverse_abs", (idx,), (name,), 2))

    if config.include_interactions:
        for left in range(len(feature_names)):
            for right in range(left + 1, len(feature_names)):
                left_name = feature_names[left]
                right_name = feature_names[right]
                terms.append(
                    SymbolicTerm(
                        f"{left_name}*{right_name}",
                        "interaction",
                        (left, right),
                        (left_name, right_name),
                        3,
                    )
                )
    return terms


def build_expression(
    target_name: str,
    intercept: float,
    coefficients: Sequence[float],
    selected_terms: Sequence[SymbolicTerm],
) -> str:
    parts = [f"{target_name} = {intercept:.5g}"]
    for coefficient, term in zip(coefficients, selected_terms):
        sign = "+" if coefficient >= 0 else "-"
        parts.append(f" {sign} {abs(float(coefficient)):.5g}*{term.label}")
    return "".join(parts)


def predictions_table(data: Table, target_name: str, y_pred: np.ndarray) -> Table:
    used = {var.name for var in list(data.domain.attributes) + list(data.domain.class_vars) + list(data.domain.metas)}
    pred_name = _unique_name("symbolic_prediction", used)
    resid_name = _unique_name("symbolic_residual", used)
    attrs = list(data.domain.attributes) + [ContinuousVariable(pred_name), ContinuousVariable(resid_name)]
    domain = Domain(attrs, data.domain.class_vars, data.domain.metas)
    X = np.asarray(data.X, dtype=float)
    y_true = _target_column(data, target_name)
    residual = (y_true - y_pred).reshape(-1, 1)
    X_out = np.hstack([X, y_pred.reshape(-1, 1), residual]).astype(float)
    Y = np.asarray(data.Y, dtype=float) if data.domain.class_vars else None
    if Y is not None and Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return safe_table_from_numpy(
        domain,
        X=X_out,
        Y=Y,
        metas=np.asarray(data.metas, dtype=object),
        name="Symbolic Regression Predictions",
    )


def term_table(
    selected_terms: Sequence[SymbolicTerm],
    coefficients: Sequence[float],
    y_true: np.ndarray,
    design: np.ndarray,
) -> Table:
    rows: list[dict[str, Any]] = []
    for idx, (term, coefficient) in enumerate(zip(selected_terms, coefficients), start=1):
        term_values = design[:, idx - 1] if design.ndim == 2 and design.shape[1] >= idx else np.zeros(len(y_true), dtype=float)
        corr = _safe_corr(term_values, y_true)
        rows.append(
            {
                "order": idx,
                "term": term.label,
                "coefficient": float(coefficient),
                "abs_coefficient": float(abs(coefficient)),
                "train_correlation": float(corr),
                "complexity": float(term.complexity),
            }
        )
    return records_to_orange_table(
        rows,
        attribute_columns=["order", "coefficient", "abs_coefficient", "train_correlation", "complexity"],
        meta_columns=["term"],
        name="Symbolic Regression Terms",
    ) or safe_table_from_numpy(Domain([], metas=[StringVariable("term")]), X=np.empty((0, 0)), metas=np.empty((0, 1), dtype=object), name="Symbolic Regression Terms")


def summary_table(
    *,
    target_name: str,
    expression: str,
    n_rows: int,
    n_input_features: int,
    n_screened_features: int,
    n_candidate_terms: int,
    n_selected_terms: int,
    train_metrics: dict[str, float],
    cv_metrics: dict[str, float],
) -> Table:
    return summary_rows_to_table(
        [
            {"metric": "target", "value": target_name, "description": "Continuous target used for symbolic regression."},
            {"metric": "expression", "value": expression, "description": "Fitted symbolic expression."},
            {"metric": "rows", "value": float(n_rows), "description": "Rows used for training."},
            {"metric": "input_features", "value": float(n_input_features), "description": "Continuous descriptor columns seen at input."},
            {"metric": "screened_features", "value": float(n_screened_features), "description": "Top-ranked descriptor columns considered for term generation."},
            {"metric": "candidate_terms", "value": float(n_candidate_terms), "description": "Generated symbolic basis terms after filtering constant terms."},
            {"metric": "selected_terms", "value": float(n_selected_terms), "description": "Terms retained in the final symbolic expression."},
            {"metric": "train_r2", "value": float(train_metrics.get("r2", float("nan"))), "description": "Training R-squared."},
            {"metric": "train_rmse", "value": float(train_metrics.get("rmse", float("nan"))), "description": "Training RMSE."},
            {"metric": "train_mae", "value": float(train_metrics.get("mae", float("nan"))), "description": "Training MAE."},
            {"metric": "cv_r2", "value": float(cv_metrics.get("r2", float("nan"))), "description": "Cross-validated R-squared."},
            {"metric": "cv_rmse", "value": float(cv_metrics.get("rmse", float("nan"))), "description": "Cross-validated RMSE."},
            {"metric": "cv_mae", "value": float(cv_metrics.get("mae", float("nan"))), "description": "Cross-validated MAE."},
        ],
        name="Symbolic Regression Summary",
    )


def _resolve_target_variable(data: Table, target_name: str) -> ContinuousVariable:
    for variable in continuous_target_candidates(data):
        if variable.name == target_name:
            return variable
    preferred = preferred_target_name(data)
    for variable in continuous_target_candidates(data):
        if variable.name == preferred:
            return variable
    raise ValueError("No continuous target variable is available.")


def _extract_xy(data: Table, target_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_vars = [var for var in data.domain.attributes if getattr(var, "is_continuous", False) and var.name != target_name]
    if not feature_vars:
        raise ValueError("No continuous descriptor columns found in attributes.")
    X = np.column_stack([data.get_column(var).astype(float) for var in feature_vars]).astype(float)
    y = _target_column(data, target_name)
    return X, y, [var.name for var in feature_vars]


def _extract_feature_matrix(data: Table, feature_names: Sequence[str]) -> np.ndarray:
    variables = {var.name: var for var in data.domain.attributes if getattr(var, "is_continuous", False)}
    missing = [name for name in feature_names if name not in variables]
    if missing:
        raise ValueError(f"Input table is missing required descriptor columns: {', '.join(missing)}")
    return np.column_stack([data.get_column(variables[name]).astype(float) for name in feature_names]).astype(float)


def _target_column(data: Table, target_name: str) -> np.ndarray:
    target_var = next((var for var in list(data.domain.class_vars) + list(data.domain.attributes) if getattr(var, "is_continuous", False) and var.name == target_name), None)
    if target_var is None:
        raise ValueError(f"Target column not found: {target_name}")
    return data.get_column(target_var).astype(float)


def _apply_imputer_statistics(X: np.ndarray, statistics: Sequence[float]) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float).copy()
    stats = np.asarray(statistics, dtype=float)
    if X_arr.shape[1] != len(stats):
        raise ValueError("Predictor matrix does not match the fitted symbolic regression model.")
    missing = ~np.isfinite(X_arr)
    if np.any(missing):
        rows, cols = np.where(missing)
        X_arr[rows, cols] = stats[cols]
    return X_arr


def _design_matrix_from_terms(X: np.ndarray, terms: Sequence[SymbolicTerm]) -> np.ndarray:
    if not terms:
        return np.empty((X.shape[0], 0), dtype=float)
    cols = [_evaluate_term(term, X).reshape(-1, 1) for term in terms]
    return np.hstack(cols).astype(float)


def _evaluate_term(term: SymbolicTerm, X: np.ndarray) -> np.ndarray:
    if term.kind == "linear":
        return X[:, term.feature_indices[0]]
    if term.kind == "square":
        return X[:, term.feature_indices[0]] ** 2
    if term.kind == "cube":
        return X[:, term.feature_indices[0]] ** 3
    if term.kind == "signed_sqrt":
        values = X[:, term.feature_indices[0]]
        return np.sign(values) * np.sqrt(np.abs(values))
    if term.kind == "signed_log":
        values = X[:, term.feature_indices[0]]
        return np.sign(values) * np.log1p(np.abs(values))
    if term.kind == "inverse_abs":
        values = X[:, term.feature_indices[0]]
        return 1.0 / (1.0 + np.abs(values))
    if term.kind == "interaction":
        left = X[:, term.feature_indices[0]]
        right = X[:, term.feature_indices[1]]
        return left * right
    raise ValueError(f"Unsupported symbolic term kind: {term.kind}")


def _rank_feature_indices(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = np.array([abs(_safe_corr(X[:, idx], y)) for idx in range(X.shape[1])], dtype=float)
    return np.argsort(-scores)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size == 0 or y_arr.size == 0:
        return 0.0
    if np.nanstd(x_arr) <= 1e-12 or np.nanstd(y_arr) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _effective_cv_folds(n_rows: int, requested: int) -> int:
    if n_rows < 4:
        return 0
    return max(2, min(int(requested), n_rows))


def _cross_validated_metrics(X: np.ndarray, y: np.ndarray, cv_folds: int, random_state: int) -> dict[str, float]:
    n_rows = len(y)
    effective_folds = _effective_cv_folds(n_rows, cv_folds)
    if effective_folds <= 1:
        baseline = np.full(n_rows, float(np.mean(y)), dtype=float)
        metrics = _regression_metrics(y, baseline)
        metrics["folds"] = float(effective_folds)
        return metrics

    splitter = KFold(n_splits=effective_folds, shuffle=True, random_state=int(random_state))
    preds = np.zeros(n_rows, dtype=float)
    for train_idx, test_idx in splitter.split(X if X.size else np.zeros((n_rows, 1), dtype=float)):
        if X.shape[1] == 0:
            preds[test_idx] = float(np.mean(y[train_idx]))
        else:
            model = LinearRegression()
            model.fit(X[train_idx], y[train_idx])
            preds[test_idx] = model.predict(X[test_idx]).astype(float)
    metrics = _regression_metrics(y, preds)
    metrics["folds"] = float(effective_folds)
    return metrics


def _selection_score(metrics: dict[str, float]) -> float:
    r2 = metrics.get("r2", float("nan"))
    if np.isfinite(r2):
        return float(r2)
    return -float(metrics.get("rmse", float("inf")))


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if len(y_true_arr) < 2 or np.nanstd(y_true_arr) <= 1e-12:
        r2 = float("nan")
    else:
        r2 = float(r2_score(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    return {"r2": r2, "rmse": rmse, "mae": mae}


def _unique_name(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    i = 2
    while f"{name}_{i}" in used:
        i += 1
    out = f"{name}_{i}"
    used.add(out)
    return out


__all__ = [
    "SymbolicRegressionConfig",
    "SymbolicRegressionModel",
    "SymbolicRegressionResult",
    "SymbolicTerm",
    "build_candidate_terms",
    "build_expression",
    "continuous_target_candidates",
    "fit_symbolic_regression",
    "preferred_target_name",
    "predictions_table",
    "summary_table",
    "term_table",
]

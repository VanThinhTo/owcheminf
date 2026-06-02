from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from Orange.data import DiscreteVariable, Table
from sklearn.model_selection import train_test_split

from chem_inf_widgets.chemcore.qsar.dataset import find_smiles_var
from chem_inf_widgets.chemcore.result import ServiceIssue
from chem_inf_widgets.chemcore.services.scaffold_service import NO_SCAFFOLD_LABEL, analyze_scaffolds
from chem_inf_widgets.chemcore.services.scaffold_splitter_service import split_by_scaffold


@dataclass(frozen=True)
class SplitConfig:
    method: str = "random"
    test_size: float = 0.2
    validation_size: float = 0.0
    random_state: int = 0
    target_column: str | None = None
    scaffold_kind: str = "murcko"


@dataclass(frozen=True)
class SplitResult:
    train_indices: list[int]
    test_indices: list[int]
    validation_indices: list[int]
    issues: list[ServiceIssue] = field(default_factory=list)


def _empty_result(issues: list[ServiceIssue]) -> SplitResult:
    return SplitResult(
        train_indices=[],
        test_indices=[],
        validation_indices=[],
        issues=issues,
    )


def _all_vars(data: Table):
    return list(data.domain.attributes) + list(data.domain.class_vars) + list(data.domain.metas)


def _find_var_by_name(data: Table, name: str | None):
    wanted = str(name or "").strip().lower()
    if not wanted:
        return None
    for var in _all_vars(data):
        if var.name.strip().lower() == wanted:
            return var
    return None


def _resolve_target_var(data: Table, target_column: str | None):
    explicit = _find_var_by_name(data, target_column)
    if explicit is not None:
        return explicit
    if len(data.domain.class_vars) == 1:
        return data.domain.class_vars[0]
    return None


def _split_counts(n_rows: int, test_size: float, validation_size: float) -> tuple[int, int]:
    n_test = int(round(float(test_size) * n_rows))
    n_validation = int(round(float(validation_size) * n_rows))

    if test_size > 0 and n_test == 0 and n_rows >= 2:
        n_test = 1
    if validation_size > 0 and n_validation == 0 and (n_rows - n_test) >= 2:
        n_validation = 1

    if n_rows > 1 and n_test + n_validation >= n_rows:
        overflow = (n_test + n_validation) - (n_rows - 1)
        if overflow > 0:
            reduce_validation = min(overflow, n_validation)
            n_validation -= reduce_validation
            overflow -= reduce_validation
        if overflow > 0:
            n_test = max(0, n_test - overflow)

    return max(0, n_test), max(0, n_validation)


def _random_split(
    indices: np.ndarray,
    *,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(int(random_state))
    shuffled = np.asarray(indices, dtype=int).copy()
    rng.shuffle(shuffled)

    n_test, n_validation = _split_counts(len(shuffled), test_size, validation_size)
    test_indices = shuffled[:n_test].tolist()
    validation_indices = shuffled[n_test:n_test + n_validation].tolist()
    train_indices = shuffled[n_test + n_validation:].tolist()
    return train_indices, test_indices, validation_indices


def _rank_binned_labels(values: np.ndarray, n_bins: int = 5) -> np.ndarray:
    finite_mask = np.isfinite(values)
    if not finite_mask.all():
        raise ValueError("Target column contains missing values.")

    if values.size == 0:
        raise ValueError("Target column is empty.")

    order = np.argsort(values, kind="mergesort")
    n_effective_bins = max(2, min(int(n_bins), int(values.size)))
    labels = np.zeros(len(values), dtype=int)
    for rank, original_index in enumerate(order):
        labels[int(original_index)] = int(np.floor(rank * n_effective_bins / len(values)))
    return labels


def _stratify_labels(data: Table, target_var) -> np.ndarray:
    column = np.asarray(data.get_column(target_var))
    if isinstance(target_var, DiscreteVariable):
        return np.asarray([str(value) for value in column], dtype=object)

    values = np.asarray(column, dtype=float)
    return _rank_binned_labels(values)


def _stratified_split(
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple[list[int], list[int], list[int]]:
    idx = np.asarray(indices, dtype=int)
    strat_labels = np.asarray(labels)

    remaining_indices = idx
    remaining_labels = strat_labels
    if test_size > 0:
        remaining_indices, test_indices, remaining_labels, _test_labels = train_test_split(
            remaining_indices,
            remaining_labels,
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=remaining_labels,
        )
    else:
        test_indices = np.empty((0,), dtype=int)

    if validation_size > 0:
        relative_validation_size = float(validation_size) / max(1.0 - float(test_size), 1e-12)
        train_indices, validation_indices, _train_labels, _validation_labels = train_test_split(
            remaining_indices,
            remaining_labels,
            test_size=relative_validation_size,
            random_state=int(random_state),
            stratify=remaining_labels,
        )
    else:
        train_indices = remaining_indices
        validation_indices = np.empty((0,), dtype=int)

    return (
        sorted(int(index) for index in train_indices.tolist()),
        sorted(int(index) for index in test_indices.tolist()),
        sorted(int(index) for index in validation_indices.tolist()),
    )


def _split_by_scaffold_method(
    data: Table,
    *,
    test_size: float,
    validation_size: float,
    random_state: int,
    scaffold_kind: str,
    issues: list[ServiceIssue],
) -> SplitResult:
    smiles_var = find_smiles_var(data)
    if smiles_var is None:
        issues.append(
            ServiceIssue(
                code="missing_smiles_column",
                message="Scaffold split requires a SMILES column.",
                severity="error",
            )
        )
        return _empty_result(issues)

    smiles_values = ["" if value is None else str(value).strip() for value in data.get_column(smiles_var)]
    result = split_by_scaffold(
        smiles_values,
        train_fraction=1.0 - float(test_size) - float(validation_size),
        validation_fraction=float(validation_size),
        test_fraction=float(test_size),
        scaffold_kind=str(scaffold_kind or "murcko"),
        random_seed=int(random_state),
    )

    train_indices = [assignment.index for assignment in result.assignments if assignment.split == "train"]
    validation_indices = [assignment.index for assignment in result.assignments if assignment.split == "validation"]
    test_indices = [assignment.index for assignment in result.assignments if assignment.split == "test"]
    invalid_indices = [assignment.index for assignment in result.assignments if assignment.split == "invalid"]

    if invalid_indices:
        issues.append(
            ServiceIssue(
                code="invalid_scaffold_rows_skipped",
                message=f"{len(invalid_indices)} row(s) could not be scaffold-assigned and were excluded.",
                severity="warning",
                details={"row_indices": invalid_indices[:20]},
            )
        )

    return SplitResult(
        train_indices=sorted(train_indices),
        test_indices=sorted(test_indices),
        validation_indices=sorted(validation_indices),
        issues=issues,
    )


def split_dataset(
    data: Table | None,
    config: SplitConfig | None = None,
) -> SplitResult:
    cfg = config or SplitConfig()
    issues: list[ServiceIssue] = []

    if data is None or len(data) == 0:
        issues.append(
            ServiceIssue(
                code="no_input_data",
                message="No input data to split.",
                severity="error",
            )
        )
        return _empty_result(issues)

    test_size = float(cfg.test_size)
    validation_size = float(cfg.validation_size)
    if test_size < 0 or validation_size < 0 or (test_size + validation_size) >= 1.0:
        issues.append(
            ServiceIssue(
                code="invalid_split_sizes",
                message="test_size and validation_size must be >= 0 and sum to less than 1.",
                severity="error",
            )
        )
        return _empty_result(issues)

    indices = np.arange(len(data), dtype=int)
    method = str(cfg.method or "random").strip().lower()

    if method == "random":
        train_indices, test_indices, validation_indices = _random_split(
            indices,
            test_size=test_size,
            validation_size=validation_size,
            random_state=int(cfg.random_state),
        )
        return SplitResult(
            train_indices=sorted(train_indices),
            test_indices=sorted(test_indices),
            validation_indices=sorted(validation_indices),
            issues=issues,
        )

    if method == "scaffold":
        return _split_by_scaffold_method(
            data,
            test_size=test_size,
            validation_size=validation_size,
            random_state=int(cfg.random_state),
            scaffold_kind=cfg.scaffold_kind,
            issues=issues,
        )

    if method in {"activity_stratified", "stratified"}:
        target_var = _resolve_target_var(data, cfg.target_column)
        if target_var is None:
            issues.append(
                ServiceIssue(
                    code="missing_target_column",
                    message="Activity-stratified split requires a target column or class variable.",
                    severity="error",
                )
            )
            return _empty_result(issues)

        try:
            labels = _stratify_labels(data, target_var)
            train_indices, test_indices, validation_indices = _stratified_split(
                indices,
                labels,
                test_size=test_size,
                validation_size=validation_size,
                random_state=int(cfg.random_state),
            )
        except ValueError as exc:
            issues.append(
                ServiceIssue(
                    code="stratified_split_fallback",
                    message=f"Could not create stratified split ({exc}). Falling back to random split.",
                    severity="warning",
                )
            )
            train_indices, test_indices, validation_indices = _random_split(
                indices,
                test_size=test_size,
                validation_size=validation_size,
                random_state=int(cfg.random_state),
            )

        return SplitResult(
            train_indices=sorted(train_indices),
            test_indices=sorted(test_indices),
            validation_indices=sorted(validation_indices),
            issues=issues,
        )

    issues.append(
        ServiceIssue(
            code="unsupported_split_method",
            message=f"Unsupported split method: {cfg.method!r}.",
            severity="error",
        )
    )
    return _empty_result(issues)


def scaffold_groups_by_split(data: Table, result: SplitResult) -> dict[str, set[str]]:
    smiles_var = find_smiles_var(data)
    if smiles_var is None:
        return {"train": set(), "validation": set(), "test": set()}

    smiles_values = ["" if value is None else str(value).strip() for value in data.get_column(smiles_var)]
    analysis = analyze_scaffolds(smiles_values)
    by_index = {
        annotation.index: (
            annotation.murcko
            or annotation.generic
            or NO_SCAFFOLD_LABEL
        )
        for annotation in analysis.annotations
        if annotation.status != "invalid"
    }
    return {
        "train": {by_index[index] for index in result.train_indices if index in by_index},
        "validation": {by_index[index] for index in result.validation_indices if index in by_index},
        "test": {by_index[index] for index in result.test_indices if index in by_index},
    }


__all__ = [
    "SplitConfig",
    "SplitResult",
    "scaffold_groups_by_split",
    "split_dataset",
]

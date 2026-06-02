from __future__ import annotations

import importlib
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA

from chem_inf_widgets.chemcore.result import ServiceIssue


@dataclass(frozen=True)
class MolecularSpaceConfig:
    method: str = "pca"
    n_components: int = 2
    random_state: int = 0


@dataclass(frozen=True)
class MolecularSpaceResult:
    coordinates: np.ndarray
    method: str
    explained_variance: list[float] | None
    issues: list[ServiceIssue] = field(default_factory=list)


def _empty_result(
    *,
    method: str,
    issues: list[ServiceIssue],
) -> MolecularSpaceResult:
    return MolecularSpaceResult(
        coordinates=np.empty((0, 0), dtype=float),
        method=method,
        explained_variance=None,
        issues=issues,
    )


def _normalise_matrix(matrix: np.ndarray | list[list[float]] | list[float]) -> np.ndarray:
    X = np.asarray(matrix, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _effective_components(
    X: np.ndarray,
    requested: int,
) -> tuple[int, ServiceIssue | None]:
    max_components = min(int(X.shape[0]), int(X.shape[1]))
    if max_components <= 0:
        return 0, None
    wanted = max(1, int(requested))
    if wanted <= max_components:
        return wanted, None
    return (
        max_components,
        ServiceIssue(
            code="reduced_n_components",
            message=(
                f"Requested {wanted} components, but only {max_components} are possible "
                "for the given matrix shape."
            ),
            severity="warning",
            details={
                "requested_components": wanted,
                "used_components": max_components,
            },
        ),
    )


def _compute_pca(
    X: np.ndarray,
    *,
    n_components: int,
    random_state: int,
) -> MolecularSpaceResult:
    pca = PCA(n_components=n_components, random_state=int(random_state))
    coords = pca.fit_transform(X)
    explained_variance = [float(value) for value in pca.explained_variance_ratio_.tolist()]
    return MolecularSpaceResult(
        coordinates=np.asarray(coords, dtype=float),
        method="pca",
        explained_variance=explained_variance,
        issues=[],
    )


def _compute_umap(
    X: np.ndarray,
    *,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    umap_module = importlib.import_module("umap")
    reducer = umap_module.UMAP(
        n_components=int(n_components),
        random_state=int(random_state),
    )
    return np.asarray(reducer.fit_transform(X), dtype=float)


def compute_molecular_space(
    matrix: np.ndarray | list[list[float]] | list[float] | None,
    config: MolecularSpaceConfig | None = None,
) -> MolecularSpaceResult:
    cfg = config or MolecularSpaceConfig()
    method = str(cfg.method or "pca").strip().lower()
    issues: list[ServiceIssue] = []

    if matrix is None:
        issues.append(
            ServiceIssue(
                code="empty_input_matrix",
                message="No descriptor or fingerprint matrix was provided.",
                severity="error",
            )
        )
        return _empty_result(method=method, issues=issues)

    try:
        X = _normalise_matrix(matrix)
    except (TypeError, ValueError) as exc:
        issues.append(
            ServiceIssue(
                code="invalid_input_matrix",
                message=f"Could not convert input matrix to numeric array: {exc}",
                severity="error",
            )
        )
        return _empty_result(method=method, issues=issues)

    if X.ndim != 2:
        issues.append(
            ServiceIssue(
                code="invalid_matrix_shape",
                message=f"Expected a 2D matrix, got {X.ndim}D input.",
                severity="error",
            )
        )
        return _empty_result(method=method, issues=issues)

    if X.shape[0] == 0 or X.shape[1] == 0:
        issues.append(
            ServiceIssue(
                code="empty_input_matrix",
                message="Descriptor or fingerprint matrix is empty.",
                severity="error",
            )
        )
        return _empty_result(method=method, issues=issues)

    if not np.isfinite(X).all():
        issues.append(
            ServiceIssue(
                code="non_finite_values",
                message="Input matrix contains NaN or infinite values.",
                severity="error",
            )
        )
        return _empty_result(method=method, issues=issues)

    n_components, reduction_issue = _effective_components(X, int(cfg.n_components))
    if reduction_issue is not None:
        issues.append(reduction_issue)

    if n_components <= 0:
        issues.append(
            ServiceIssue(
                code="invalid_n_components",
                message="Could not determine a valid number of embedding components.",
                severity="error",
            )
        )
        return _empty_result(method=method, issues=issues)

    if method == "pca":
        result = _compute_pca(X, n_components=n_components, random_state=int(cfg.random_state))
        return MolecularSpaceResult(
            coordinates=result.coordinates,
            method=result.method,
            explained_variance=result.explained_variance,
            issues=issues,
        )

    if method == "umap":
        try:
            coordinates = _compute_umap(
                X,
                n_components=n_components,
                random_state=int(cfg.random_state),
            )
            return MolecularSpaceResult(
                coordinates=coordinates,
                method="umap",
                explained_variance=None,
                issues=issues,
            )
        except ModuleNotFoundError:
            issues.append(
                ServiceIssue(
                    code="umap_not_available",
                    message="UMAP is not installed. Falling back to PCA.",
                    severity="warning",
                )
            )
            result = _compute_pca(X, n_components=n_components, random_state=int(cfg.random_state))
            return MolecularSpaceResult(
                coordinates=result.coordinates,
                method=result.method,
                explained_variance=result.explained_variance,
                issues=issues,
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            issues.append(
                ServiceIssue(
                    code="umap_embedding_failed",
                    message=f"UMAP embedding failed: {exc}",
                    severity="error",
                )
            )
            return _empty_result(method="umap", issues=issues)

    issues.append(
        ServiceIssue(
            code="unsupported_method",
            message=f"Unsupported molecular space method: {cfg.method!r}.",
            severity="error",
        )
    )
    return _empty_result(method=method, issues=issues)


__all__ = [
    "MolecularSpaceConfig",
    "MolecularSpaceResult",
    "compute_molecular_space",
]

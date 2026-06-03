from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

from chem_inf_widgets.chemcore.services.molecular_space_service import (
    MolecularSpaceConfig,
    compute_molecular_space,
)
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles

DiversityMethod = Literal["maxmin", "sphere_exclusion", "butina"]

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@dataclass(frozen=True)
class DiversityMetrics:
    n_compounds: int
    mean_nn_distance: float
    mean_pairwise_dist: float
    n_singletons: int
    diversity_score: float


@dataclass(frozen=True)
class DiversitySelectionResult:
    method: DiversityMethod
    selected_indices: list[int]
    valid_indices: list[int]
    failed_indices: list[int]
    metrics_input: DiversityMetrics
    metrics_selected: DiversityMetrics
    coordinates: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))
    explained_variance: list[float] | None = None
    selection_ranks: list[int | None] = field(default_factory=list)


def _compute_fps(smiles_list: list[str]) -> tuple[list, list[int], np.ndarray]:
    fps = []
    valid_indices = []
    rows: list[np.ndarray] = []
    for index, smiles in enumerate(smiles_list):
        mol = safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True).mol
        if mol is None:
            continue
        fp = _MORGAN_GEN.GetFingerprint(mol)
        arr = np.zeros((2048,), dtype=float)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(fp)
        valid_indices.append(index)
        rows.append(arr)
    matrix = np.vstack(rows).astype(float, copy=False) if rows else np.empty((0, 2048), dtype=float)
    return fps, valid_indices, matrix


def _distance_function(fps: list):
    cache: dict[tuple[int, int], float] = {}

    def _dist(i: int, j: int) -> float:
        key = (i, j) if i <= j else (j, i)
        if key not in cache:
            cache[key] = 1.0 - float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        return cache[key]

    return _dist


def maxmin_selection(
    smiles_list: list[str],
    n_select: int,
    seed_idx: int = 0,
    random_seed: int = 42,
) -> list[int]:
    fps, valid_indices, _matrix = _compute_fps(smiles_list)
    n_fps = len(fps)
    if n_fps == 0:
        return []

    n_select = min(max(1, int(n_select)), n_fps)
    rng = random.Random(random_seed)
    first_picks: list[int] = []
    if 0 <= seed_idx < n_fps:
        first_picks = [int(seed_idx)]
    elif n_fps:
        first_picks = [rng.randint(0, n_fps - 1)]

    picker = MaxMinPicker()
    selected = list(
        picker.LazyPick(
            _distance_function(fps),
            n_fps,
            n_select,
            firstPicks=first_picks,
            seed=int(random_seed),
        )
    )
    return [valid_indices[idx] for idx in selected]


def sphere_exclusion(
    smiles_list: list[str],
    radius: float = 0.35,
    random_seed: int = 42,
) -> list[int]:
    fps, valid_indices, _matrix = _compute_fps(smiles_list)
    n_fps = len(fps)
    if n_fps == 0:
        return []

    rng = random.Random(random_seed)
    order = list(range(n_fps))
    rng.shuffle(order)

    similarity_threshold = 1.0 - float(radius)
    excluded = set()
    selected = []

    for idx in order:
        if idx in excluded:
            continue
        selected.append(idx)
        sims = DataStructs.BulkTanimotoSimilarity(fps[idx], fps)
        for other_idx, similarity in enumerate(sims):
            if other_idx != idx and similarity >= similarity_threshold:
                excluded.add(other_idx)

    return [valid_indices[idx] for idx in selected]


def butina_cluster_selection(
    smiles_list: list[str],
    n_clusters: int = 10,
    threshold: float = 0.4,
) -> list[int]:
    fps, valid_indices, _matrix = _compute_fps(smiles_list)
    n_fps = len(fps)
    if n_fps == 0:
        return []

    n_clusters = min(max(1, int(n_clusters)), n_fps)
    dists = []
    for idx in range(1, n_fps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[idx], fps[:idx])
        dists.extend(1.0 - similarity for similarity in sims)

    clusters = Butina.ClusterData(dists, n_fps, float(threshold), isDistData=True)
    sorted_clusters = sorted(clusters, key=len, reverse=True)

    selected = []
    for cluster in sorted_clusters[:n_clusters]:
        if cluster:
            selected.append(cluster[0])
    return [valid_indices[idx] for idx in selected]


def diversity_metrics(
    smiles_list: list[str],
    sample_size: int = 500,
    random_seed: int = 42,
) -> DiversityMetrics:
    fps, _valid_indices, _matrix = _compute_fps(smiles_list)
    n_fps = len(fps)
    if n_fps < 2:
        return DiversityMetrics(
            n_compounds=n_fps,
            mean_nn_distance=1.0,
            mean_pairwise_dist=1.0,
            n_singletons=n_fps,
            diversity_score=1.0,
        )

    rng = random.Random(random_seed)
    fps_sample = fps
    if n_fps > int(sample_size):
        sample_indices = rng.sample(range(n_fps), int(sample_size))
        fps_sample = [fps[idx] for idx in sample_indices]

    nn_distances = []
    pairwise_sample = []
    for idx, fp in enumerate(fps_sample):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps_sample)
        sims_without_self = [similarity for j, similarity in enumerate(sims) if j != idx]
        nn_sim = max(sims_without_self) if sims_without_self else 0.0
        nn_distances.append(1.0 - nn_sim)
        pairwise_sample.extend(1.0 - similarity for similarity in sims_without_self[:20])

    mean_nn = sum(nn_distances) / len(nn_distances)
    mean_pairwise = sum(pairwise_sample) / len(pairwise_sample) if pairwise_sample else 0.0
    n_singletons = sum(1 for distance in nn_distances if distance > 0.5)

    return DiversityMetrics(
        n_compounds=n_fps,
        mean_nn_distance=round(mean_nn, 4),
        mean_pairwise_dist=round(mean_pairwise, 4),
        n_singletons=n_singletons,
        diversity_score=round(mean_pairwise, 4),
    )


def _project_fingerprint_matrix(
    matrix: np.ndarray,
    *,
    full_size: int,
    valid_indices: list[int],
    random_seed: int,
) -> tuple[np.ndarray, list[float] | None]:
    coordinates = np.full((int(full_size), 2), np.nan, dtype=float)
    if matrix.shape[0] == 0:
        return coordinates, None

    projection = compute_molecular_space(
        matrix,
        MolecularSpaceConfig(method="pca", n_components=2, random_state=int(random_seed)),
    )
    valid_coords = np.asarray(projection.coordinates, dtype=float)
    if valid_coords.ndim != 2 or valid_coords.shape[0] == 0:
        return coordinates, projection.explained_variance

    if valid_coords.shape[1] == 1:
        valid_coords = np.column_stack([valid_coords[:, 0], np.zeros(valid_coords.shape[0], dtype=float)])
    elif valid_coords.shape[1] > 2:
        valid_coords = valid_coords[:, :2]

    for row_index, original_index in enumerate(valid_indices):
        coordinates[int(original_index), :] = valid_coords[row_index, :]
    return coordinates, projection.explained_variance


def _selection_ranks(
    total_count: int,
    selected_indices: list[int],
) -> list[int | None]:
    ranks: list[int | None] = [None] * int(total_count)
    for rank, index in enumerate(selected_indices, start=1):
        if 0 <= int(index) < len(ranks):
            ranks[int(index)] = rank
    return ranks


def select_diverse_subset(
    smiles_list: list[str],
    *,
    method: DiversityMethod = "maxmin",
    n_select: int = 25,
    seed_idx: int = 0,
    radius: float = 0.35,
    n_clusters: int | None = None,
    threshold: float = 0.4,
    random_seed: int = 42,
) -> DiversitySelectionResult:
    method_name = method.strip().lower()
    metrics_input = diversity_metrics(smiles_list, random_seed=random_seed)
    _fps, valid_indices, matrix = _compute_fps(smiles_list)
    failed_indices = [idx for idx in range(len(smiles_list)) if idx not in set(valid_indices)]

    if method_name == "maxmin":
        selected_indices = maxmin_selection(
            smiles_list,
            n_select=n_select,
            seed_idx=seed_idx,
            random_seed=random_seed,
        )
    elif method_name == "sphere_exclusion":
        selected_indices = sphere_exclusion(
            smiles_list,
            radius=radius,
            random_seed=random_seed,
        )
    elif method_name == "butina":
        selected_indices = butina_cluster_selection(
            smiles_list,
            n_clusters=n_clusters if n_clusters is not None else n_select,
            threshold=threshold,
        )
    else:
        raise ValueError(f"Unsupported diversity method: {method!r}")

    selected_smiles = [smiles_list[idx] for idx in selected_indices]
    metrics_selected = diversity_metrics(selected_smiles, random_seed=random_seed)
    coordinates, explained_variance = _project_fingerprint_matrix(
        matrix,
        full_size=len(smiles_list),
        valid_indices=valid_indices,
        random_seed=random_seed,
    )
    return DiversitySelectionResult(
        method=method_name,  # type: ignore[arg-type]
        selected_indices=selected_indices,
        valid_indices=valid_indices,
        failed_indices=failed_indices,
        metrics_input=metrics_input,
        metrics_selected=metrics_selected,
        coordinates=coordinates,
        explained_variance=explained_variance,
        selection_ranks=_selection_ranks(len(smiles_list), selected_indices),
    )

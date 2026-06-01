from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from Orange.data import ContinuousVariable, DiscreteVariable, StringVariable, Table

from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
except Exception:  # pragma: no cover - optional RDKit guard
    Chem = None
    Descriptors = None
    Crippen = None
    Lipinski = None
    rdMolDescriptors = None


TARGET_COLUMN_CANDIDATES = {
    "pactivity",
    "p_activity",
    "pchembl_value",
    "pic50",
    "pki",
    "pkd",
    "pec50",
    "activity",
}

LEAKY_OR_NON_DESCRIPTOR_ATTRIBUTE_NAMES = {
    "row_index",
    "activity",
    "activity_value",
    "pactivity",
    "p_activity",
    "pactivity_raw",
    "pactivity_min",
    "pactivity_max",
    "pactivity_std",
    "pchembl_value",
    "pic50",
    "pki",
    "pkd",
    "pec50",
    "n_measurements",
    "duplicate_group",
}

RDKit_DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "LabuteASA",
]


def find_smiles_var(data: Table):
    wanted = {"smiles", "canonical_smiles", "smile"}
    variables = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)
    preferred = [var for var in variables if isinstance(var, StringVariable) and var.name.strip().lower() in wanted]
    if preferred:
        return preferred[0]
    return next((var for var in variables if isinstance(var, StringVariable)), None)


def find_name_var(data: Table):
    wanted = {"name", "title", "compound", "compound_name"}
    variables = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)
    return next(
        (var for var in variables if isinstance(var, StringVariable) and var.name.strip().lower() in wanted),
        None,
    )


def _norm_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _is_numeric_variable(var) -> bool:
    return isinstance(var, ContinuousVariable) or isinstance(var, DiscreteVariable)


def _target_candidate_var(data: Table):
    variables = list(data.domain.class_vars) + list(data.domain.attributes) + list(data.domain.metas)
    for var in variables:
        if _norm_name(var.name) in TARGET_COLUMN_CANDIDATES and _is_numeric_variable(var):
            return var
    for var in variables:
        if _norm_name(var.name) in TARGET_COLUMN_CANDIDATES:
            try:
                col = data.get_column(var)
                vals = np.asarray([float(str(v).strip().replace(",", ".")) for v in col if str(v).strip()], dtype=float)
                if vals.size:
                    return var
            except Exception:
                pass
    return None


def _column_as_float(data: Table, var) -> np.ndarray:
    col = data.get_column(var)
    out = []
    for value in col:
        if value is None:
            out.append(np.nan)
            continue
        try:
            out.append(float(value))
        except Exception:
            try:
                out.append(float(str(value).strip().replace(",", ".")))
            except Exception:
                out.append(np.nan)
    return np.asarray(out, dtype=float)


def _numeric_columns_from_vars(data: Table, variables: Sequence) -> tuple[np.ndarray, list[str]]:
    cols = []
    names = []
    for var in variables:
        arr = _column_as_float(data, var)
        if np.any(np.isfinite(arr)):
            cols.append(arr)
            names.append(var.name)
    if not cols:
        return np.empty((len(data), 0), dtype=float), []
    return np.asarray(np.column_stack(cols), dtype=float), names


def _descriptor_attribute_vars(data: Table, target_var=None) -> list:
    out = []
    for var in data.domain.attributes:
        if target_var is not None and var.name == target_var.name:
            continue
        if not isinstance(var, ContinuousVariable):
            continue
        if _norm_name(var.name) in LEAKY_OR_NON_DESCRIPTOR_ATTRIBUTE_NAMES:
            continue
        out.append(var)
    return out


def _smiles_column_values(data: Table) -> list[str]:
    var = find_smiles_var(data)
    if var is None:
        return []
    col = data.get_column(var)
    values = []
    for value in col:
        if value is None:
            values.append("")
        else:
            text = str(value).strip()
            values.append("" if text.lower() == "nan" else text)
    return values


def _rdkit_descriptor_row(smiles: str) -> list[float]:
    if Chem is None or Descriptors is None:
        raise ValueError("No descriptor columns found and RDKit is not available to compute all-in-one QSAR descriptors.")
    mol = safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True).mol
    if mol is None:
        return [np.nan] * len(RDKit_DESCRIPTOR_NAMES)
    return [
        float(Descriptors.MolWt(mol)),
        float(Crippen.MolLogP(mol)),
        float(rdMolDescriptors.CalcTPSA(mol)),
        float(Lipinski.NumHDonors(mol)),
        float(Lipinski.NumHAcceptors(mol)),
        float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        float(rdMolDescriptors.CalcNumRings(mol)),
        float(rdMolDescriptors.CalcFractionCSP3(mol)),
        float(mol.GetNumHeavyAtoms()),
        float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        float(rdMolDescriptors.CalcLabuteASA(mol)),
    ]


def _compute_rdkit_descriptor_matrix(data: Table) -> tuple[np.ndarray, list[str]]:
    smiles_values = _smiles_column_values(data)
    if not smiles_values:
        raise ValueError(
            "No usable descriptor attributes were found and no SMILES column was found for automatic descriptor calculation. "
            "Connect QSAR Dataset Builder output with SMILES metas, or compute descriptors before QSAR Regression."
        )
    X = np.asarray([_rdkit_descriptor_row(smiles) for smiles in smiles_values], dtype=float)
    return X, list(RDKit_DESCRIPTOR_NAMES)


def _finite_unique_count(values: np.ndarray) -> int:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0
    return int(np.unique(finite).size)


def clean_qsar_descriptor_matrix(
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    remove_constant: bool = True,
) -> tuple[np.ndarray, list[str], dict]:
    X_arr = np.asarray(X, dtype=float)
    names = list(feature_names)
    if X_arr.ndim != 2:
        X_arr = X_arr.reshape(len(X_arr), -1)

    if X_arr.shape[1] != len(names):
        names = [f"descriptor_{i + 1}" for i in range(X_arr.shape[1])]

    finite_col_mask = np.any(np.isfinite(X_arr), axis=0) if X_arr.size else np.zeros(X_arr.shape[1], dtype=bool)
    unique_counts = np.asarray([_finite_unique_count(X_arr[:, j]) for j in range(X_arr.shape[1])], dtype=int)
    constant_mask = unique_counts <= 1
    keep_mask = finite_col_mask.copy()
    if remove_constant:
        keep_mask &= ~constant_mask

    removed_all_missing = [name for name, keep, finite in zip(names, keep_mask, finite_col_mask) if not finite]
    removed_constant = [
        name for name, keep, finite, const in zip(names, keep_mask, finite_col_mask, constant_mask)
        if finite and const and remove_constant
    ]

    if not np.any(keep_mask):
        fallback = finite_col_mask
        if np.any(fallback):
            keep_mask = fallback
            removed_constant = []

    X_clean = X_arr[:, keep_mask] if np.any(keep_mask) else X_arr[:, :0]
    names_clean = [name for name, keep in zip(names, keep_mask) if keep]
    cleanup = {
        "input_descriptor_count": int(X_arr.shape[1]),
        "descriptor_count": int(X_clean.shape[1]),
        "removed_all_missing_count": int(len(removed_all_missing)),
        "removed_constant_count": int(len(removed_constant)),
        "removed_all_missing": removed_all_missing[:30],
        "removed_constant": removed_constant[:30],
    }
    return X_clean, names_clean, cleanup


def _feature_relevance_scores_for_cap(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n_features = int(X_arr.shape[1]) if X_arr.ndim == 2 else 0
    scores = np.zeros(n_features, dtype=float)
    for j in range(n_features):
        x = X_arr[:, j]
        mask = np.isfinite(x) & np.isfinite(y_arr)
        if np.count_nonzero(mask) >= 3:
            xv = x[mask]
            yv = y_arr[mask]
            xstd = float(np.std(xv))
            ystd = float(np.std(yv))
            if xstd > 0 and ystd > 0:
                corr = np.corrcoef(xv, yv)[0, 1]
                if np.isfinite(corr):
                    scores[j] = abs(float(corr))
                    continue
        finite = x[np.isfinite(x)]
        if finite.size > 1:
            var = float(np.nanvar(finite))
            scores[j] = var if np.isfinite(var) else 0.0
    return scores


def cap_qsar_descriptor_matrix(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    max_features: int,
) -> tuple[np.ndarray, list[str], dict]:
    X_arr = np.asarray(X, dtype=float)
    names = list(feature_names)
    max_features = int(max_features or 0)
    if max_features <= 0 or X_arr.ndim != 2 or X_arr.shape[1] <= max_features:
        return X_arr, names, {
            "qsar_cap_applied": False,
            "qsar_cap_limit": max_features,
            "removed_qsar_cap_count": 0,
            "removed_qsar_cap": [],
        }

    scores = _feature_relevance_scores_for_cap(X_arr, y)
    order = np.argsort(scores, kind="mergesort")[::-1]
    keep_idx = np.sort(order[:max_features])
    keep_mask = np.zeros(X_arr.shape[1], dtype=bool)
    keep_mask[keep_idx] = True
    removed = [name for name, keep in zip(names, keep_mask) if not keep]
    kept_names = [name for name, keep in zip(names, keep_mask) if keep]
    return X_arr[:, keep_mask], kept_names, {
        "qsar_cap_applied": True,
        "qsar_cap_limit": max_features,
        "removed_qsar_cap_count": int(len(removed)),
        "removed_qsar_cap": removed[:30],
    }


def prepare_qsar_model_matrix(data: Table, *, feature_names: Optional[Sequence[str]] = None) -> dict:
    if data is None or len(data) == 0:
        raise ValueError("No rows are available for QSAR regression.")

    target_var = data.domain.class_var or _target_candidate_var(data)
    if target_var is None:
        attr_names = [var.name for var in data.domain.attributes]
        meta_names = [var.name for var in data.domain.metas]
        class_names = [var.name for var in data.domain.class_vars]
        raise ValueError(
            "No numeric target variable found. Expected a class variable or a pActivity/activity-like column.\n"
            f"Class variables: {class_names or 'none'}\n"
            f"Attributes: {attr_names[:20] or 'none'}\n"
            f"Metas: {meta_names[:20] or 'none'}"
        )

    y = _column_as_float(data, target_var)
    generated_descriptors = False
    attr_vars = _descriptor_attribute_vars(data, target_var=target_var)

    if feature_names is not None:
        by_name = {var.name: var for var in attr_vars}
        if all(name in by_name for name in feature_names):
            attr_vars = [by_name[name] for name in feature_names]
            X = np.asarray(np.column_stack([_column_as_float(data, var) for var in attr_vars]), dtype=float)
            names = [var.name for var in attr_vars]
        else:
            candidate_vars = [
                var for var in list(data.domain.attributes) + list(data.domain.metas)
                if var.name in set(feature_names)
                and var.name != target_var.name
                and _norm_name(var.name) not in LEAKY_OR_NON_DESCRIPTOR_ATTRIBUTE_NAMES
            ]
            X, names = _numeric_columns_from_vars(data, candidate_vars)
            if len(names) != len(feature_names):
                X, names = _compute_rdkit_descriptor_matrix(data)
                generated_descriptors = True
        finite_y = np.isfinite(y)
        if not np.any(finite_y):
            raise ValueError(f"Target column '{target_var.name}' contains no numeric values.")
        if X.shape[1] == 0:
            raise ValueError("No descriptor columns are available for QSAR regression.")
        return {
            "X": X,
            "y": y,
            "metas": np.array(data.metas),
            "target_var": target_var,
            "feature_names": names,
            "generated_descriptors": generated_descriptors,
        }

    if attr_vars:
        X = np.asarray(np.column_stack([_column_as_float(data, var) for var in attr_vars]), dtype=float)
        names = [var.name for var in attr_vars]
    else:
        meta_vars = [
            var for var in data.domain.metas
            if var.name != target_var.name
            and _norm_name(var.name) not in LEAKY_OR_NON_DESCRIPTOR_ATTRIBUTE_NAMES
            and _norm_name(var.name) not in TARGET_COLUMN_CANDIDATES
            and _norm_name(var.name) not in {"smiles", "smile", "canonicalsmiles", "canonical_smiles"}
        ]
        X, names = _numeric_columns_from_vars(data, meta_vars)
        if X.shape[1] == 0:
            X, names = _compute_rdkit_descriptor_matrix(data)
            generated_descriptors = True

    finite_y = np.isfinite(y)
    if not np.any(finite_y):
        raise ValueError(f"Target column '{target_var.name}' contains no numeric values.")
    if X.shape[1] == 0:
        raise ValueError("No descriptor columns are available for QSAR regression.")

    finite_descriptor_cols = np.any(np.isfinite(X), axis=0)
    if not np.any(finite_descriptor_cols):
        raise ValueError(
            "Descriptor columns were found, but all descriptor values are missing/non-numeric. "
            "Check Select Columns: descriptor columns must remain numeric attributes, or keep SMILES for automatic RDKit descriptors."
        )
    if not np.all(finite_descriptor_cols):
        X = X[:, finite_descriptor_cols]
        names = [name for name, keep in zip(names, finite_descriptor_cols) if keep]

    return {
        "X": X,
        "y": y,
        "metas": np.array(data.metas),
        "target_var": target_var,
        "feature_names": names,
        "generated_descriptors": generated_descriptors,
    }


__all__ = [
    "LEAKY_OR_NON_DESCRIPTOR_ATTRIBUTE_NAMES",
    "RDKit_DESCRIPTOR_NAMES",
    "TARGET_COLUMN_CANDIDATES",
    "_rdkit_descriptor_row",
    "cap_qsar_descriptor_matrix",
    "clean_qsar_descriptor_matrix",
    "find_name_var",
    "find_smiles_var",
    "prepare_qsar_model_matrix",
]

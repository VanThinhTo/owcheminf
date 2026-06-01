"""QSAR core utilities (dataset prep, feature filtering, MLR selection, applicability domain)."""

from .dataset import (
    LEAKY_OR_NON_DESCRIPTOR_ATTRIBUTE_NAMES,
    RDKit_DESCRIPTOR_NAMES,
    TARGET_COLUMN_CANDIDATES,
    _rdkit_descriptor_row,
    cap_qsar_descriptor_matrix,
    clean_qsar_descriptor_matrix,
    find_name_var,
    find_smiles_var,
    prepare_qsar_model_matrix,
)
from .algorithms import (
    QSARRunConfig,
    TORCH_AVAILABLE,
    TorchRegressor,
    _build_modeling_pipeline,
    _make_safe_regressor,
    _run_auto_qsar_model_selection,
    available_algorithms,
    build_run_config,
)

__all__ = [
    "LEAKY_OR_NON_DESCRIPTOR_ATTRIBUTE_NAMES",
    "QSARRunConfig",
    "RDKit_DESCRIPTOR_NAMES",
    "TARGET_COLUMN_CANDIDATES",
    "TORCH_AVAILABLE",
    "TorchRegressor",
    "_build_modeling_pipeline",
    "_make_safe_regressor",
    "_run_auto_qsar_model_selection",
    "_rdkit_descriptor_row",
    "available_algorithms",
    "build_run_config",
    "cap_qsar_descriptor_matrix",
    "clean_qsar_descriptor_matrix",
    "find_name_var",
    "find_smiles_var",
    "prepare_qsar_model_matrix",
]

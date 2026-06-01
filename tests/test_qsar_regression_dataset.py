from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("Orange")
pytest.importorskip("rdkit")

from Orange.data import ContinuousVariable, Domain, StringVariable, Table

from chem_inf_widgets.chemcore.qsar.dataset import (
    cap_qsar_descriptor_matrix,
    clean_qsar_descriptor_matrix,
    prepare_qsar_model_matrix,
)


def test_clean_qsar_descriptor_matrix_removes_all_missing_and_constant_columns():
    X = np.array(
        [
            [1.0, np.nan, 5.0, 0.1],
            [2.0, np.nan, 5.0, 0.2],
            [3.0, np.nan, 5.0, 0.3],
        ],
        dtype=float,
    )

    X_clean, names_clean, cleanup = clean_qsar_descriptor_matrix(
        X,
        ["varying", "all_missing", "constant", "signal"],
    )

    assert X_clean.shape == (3, 2)
    assert names_clean == ["varying", "signal"]
    assert cleanup["removed_all_missing_count"] == 1
    assert cleanup["removed_constant_count"] == 1
    assert cleanup["removed_all_missing"] == ["all_missing"]
    assert cleanup["removed_constant"] == ["constant"]


def test_cap_qsar_descriptor_matrix_keeps_best_features_in_original_order():
    X = np.array(
        [
            [0.0, 0.2, 10.0],
            [1.0, 0.2, 9.0],
            [2.0, 0.2, 8.0],
            [3.0, 0.2, 7.0],
        ],
        dtype=float,
    )
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)

    X_cap, names_cap, cleanup = cap_qsar_descriptor_matrix(
        X,
        y,
        ["best", "flat", "inverse_best"],
        max_features=2,
    )

    assert X_cap.shape == (4, 2)
    assert names_cap == ["best", "inverse_best"]
    assert cleanup["qsar_cap_applied"] is True
    assert cleanup["removed_qsar_cap_count"] == 1
    assert cleanup["removed_qsar_cap"] == ["flat"]


def test_prepare_qsar_model_matrix_falls_back_to_rdkit_descriptors_from_smiles_meta():
    domain = Domain(
        [],
        class_vars=[ContinuousVariable("pActivity")],
        metas=[StringVariable("SMILES"), StringVariable("compound_id")],
    )
    table = Table.from_list(
        domain,
        [
            [5.1, "CCO", "M001"],
            [5.4, "CCN", "M002"],
            [5.8, "c1ccccc1", "M003"],
        ],
    )

    prepared = prepare_qsar_model_matrix(table)

    assert prepared["generated_descriptors"] is True
    assert prepared["feature_names"]
    assert prepared["target_var"].name == "pActivity"
    assert prepared["X"].shape[0] == len(table)
    assert prepared["X"].shape[1] == len(prepared["feature_names"])

from __future__ import annotations

import numpy as np
from Orange.data import Table
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from chem_inf_widgets.chemcore.qsar.applicability_domain import build_applicability_domain_table
from chem_inf_widgets.chemcore.qsar.validation import build_qsar_modeling_summary_table


def test_build_qsar_modeling_summary_table_includes_cleanup_metrics():
    result = {
        "target_column": "pActivity",
        "usable_row_count": 12,
        "removed_row_count": 3,
        "feature_names": ["d1", "d2"],
        "cv_score": 0.81,
        "descriptor_cleanup": {
            "input_descriptor_count": 10,
            "descriptor_count": 2,
            "removed_all_missing_count": 1,
            "removed_constant_count": 2,
            "qsar_cap_limit": 1000,
            "removed_qsar_cap_count": 5,
            "removed_all_missing": ["bad1"],
            "removed_constant": ["bad2"],
            "removed_qsar_cap": ["bad3"],
        },
    }

    table = build_qsar_modeling_summary_table(result)

    assert isinstance(table, Table)
    metrics = [str(row[table.domain.metas[1]]) for row in table]
    assert "target_column" in metrics
    assert "removed_qsar_cap_examples" in metrics


def test_build_applicability_domain_table_returns_train_and_test_rows():
    X_train = np.array(
        [
            [0.1, 1.0],
            [0.2, 0.9],
            [0.3, 0.8],
            [0.4, 0.7],
        ],
        dtype=float,
    )
    y_train = np.array([1.0, 1.2, 1.4, 1.6], dtype=float)
    X_test = np.array([[0.15, 0.95], [0.35, 0.75]], dtype=float)
    y_test = np.array([1.1, 1.5], dtype=float)
    pipeline = Pipeline([("regressor", LinearRegression())]).fit(X_train, y_train)

    table = build_applicability_domain_table(
        {
            "pipeline": pipeline,
            "is_classification": False,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
    )

    assert isinstance(table, Table)
    assert len(table) == 6
    dataset_names = {str(row[table.domain["dataset"]]) for row in table}
    assert dataset_names == {"train", "test"}

from __future__ import annotations

import numpy as np
from Orange.data import Table
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from chem_inf_widgets.chemcore.qsar.diagnostics import compute_vif, extract_descriptor_coefficients


def test_compute_vif_returns_one_for_single_feature():
    X_t = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=float)

    vifs = compute_vif(X_t)

    assert vifs.shape == (1,)
    assert vifs[0] == 1.0


def test_extract_descriptor_coefficients_sorts_by_absolute_value():
    X = np.array(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.5],
            [4.0, 2.0],
            [5.0, 1.5],
        ],
        dtype=float,
    )
    y = np.array([2.0, 3.0, 4.0, 6.0, 7.0], dtype=float)
    pipeline = Pipeline([("regressor", LinearRegression())]).fit(X, y)

    table = extract_descriptor_coefficients(pipeline, ["d1", "d2"])

    assert isinstance(table, Table)
    assert table.domain.attributes[0].name == "coefficient"
    assert table.domain.attributes[1].name == "abs_coefficient"
    names = [str(row[table.domain.metas[0]]) for row in table]
    abs_values = table.X[:, 1]
    assert names == ["d1", "d2"]
    assert abs_values[0] >= abs_values[1]

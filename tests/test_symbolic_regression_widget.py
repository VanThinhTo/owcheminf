from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication
from Orange.data import ContinuousVariable, Domain, StringVariable, Table

from chem_inf_widgets.widgets.ow_symbolic_regression import OWSymbolicRegression


_APP = QApplication.instance() or QApplication([])


def _demo_table(n: int = 48) -> Table:
    rng = np.random.default_rng(29)
    x1 = rng.uniform(-2.0, 2.0, size=n)
    x2 = rng.uniform(-1.5, 1.5, size=n)
    x3 = rng.normal(0.0, 0.8, size=n)
    y = 1.4 * x1 + 0.75 * (x2 ** 2) + rng.normal(0.0, 0.05, size=n)
    domain = Domain(
        [ContinuousVariable("x1"), ContinuousVariable("x2"), ContinuousVariable("x3")],
        class_vars=[ContinuousVariable("target")],
        metas=[StringVariable("compound_id")],
    )
    return Table.from_numpy(
        domain,
        X=np.column_stack([x1, x2, x3]).astype(float),
        Y=y.reshape(-1, 1).astype(float),
        metas=np.array([[f"C{i:03d}"] for i in range(n)], dtype=object),
    )


def test_symbolic_regression_widget_fits_and_renders_expression():
    widget = OWSymbolicRegression()
    try:
        widget.auto_run = False
        widget.set_data(_demo_table())
        widget.commit()
        _APP.processEvents()

        assert "target =" in widget._expression_browser.toPlainText()
        assert "Train R" in widget._summary_browser.toPlainText()
        assert widget._terms_table.rowCount() >= 1
        assert widget._predictions_table.rowCount() == 48
    finally:
        widget.onDeleteWidget()
        widget.close()

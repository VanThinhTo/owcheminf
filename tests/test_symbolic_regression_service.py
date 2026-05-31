from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from Orange.data import ContinuousVariable, Domain, StringVariable, Table
except Exception:  # pragma: no cover
    ContinuousVariable = Domain = StringVariable = Table = None  # type: ignore

from chem_inf_widgets.chemcore.services.symbolic_regression_service import (  # noqa: E402
    SymbolicRegressionConfig,
    fit_symbolic_regression,
)


@unittest.skipIf(Table is None, "Orange is required for symbolic regression tests")
class SymbolicRegressionServiceTests(unittest.TestCase):
    @staticmethod
    def _demo_table(n: int = 72) -> Table:
        rng = np.random.default_rng(19)
        x1 = rng.uniform(-2.0, 2.0, size=n)
        x2 = rng.uniform(-1.5, 1.5, size=n)
        x3 = rng.normal(0.0, 0.8, size=n)
        y = 1.25 * x1 + 0.9 * (x2 ** 2) + rng.normal(0.0, 0.05, size=n)
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

    def test_symbolic_regression_finds_compact_expression(self):
        result = fit_symbolic_regression(
            self._demo_table(),
            target_name="target",
            config=SymbolicRegressionConfig(
                max_features=3,
                max_terms=4,
                cv_folds=5,
                include_square=True,
                include_cube=False,
                include_log=False,
                include_sqrt=False,
                include_inverse=False,
                include_interactions=False,
            ),
        )

        self.assertGreater(result.train_metrics["r2"], 0.95)
        self.assertGreater(result.cv_metrics["r2"], 0.90)
        self.assertIn("x1", result.expression)
        self.assertIn("x2^2", result.expression)

    def test_symbolic_regression_predictions_preserve_payload(self):
        table = self._demo_table(n=24)
        result = fit_symbolic_regression(table, target_name="target")
        out = result.predictions_table

        self.assertIsNotNone(out)
        self.assertIn("symbolic_prediction", [var.name for var in out.domain.attributes])
        self.assertIn("symbolic_residual", [var.name for var in out.domain.attributes])
        np.testing.assert_array_equal(out.metas, table.metas)
        np.testing.assert_allclose(out.Y, table.Y)


if __name__ == "__main__":
    unittest.main()

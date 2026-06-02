from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.molecular_space_service import (  # noqa: E402
    MolecularSpaceConfig,
    compute_molecular_space,
)


class MolecularSpaceServiceTests(unittest.TestCase):
    @staticmethod
    def _demo_matrix() -> np.ndarray:
        return np.asarray(
            [
                [0.0, 1.0, 0.5],
                [1.0, 0.5, 1.5],
                [2.0, 0.0, 2.5],
                [3.0, 1.5, 3.5],
            ],
            dtype=float,
        )

    def test_pca_returns_coordinates_and_explained_variance(self):
        result = compute_molecular_space(
            self._demo_matrix(),
            MolecularSpaceConfig(method="pca", n_components=2, random_state=7),
        )

        self.assertEqual(result.method, "pca")
        self.assertEqual(result.coordinates.shape, (4, 2))
        self.assertIsNotNone(result.explained_variance)
        assert result.explained_variance is not None
        self.assertEqual(len(result.explained_variance), 2)
        self.assertTrue(all(value >= 0.0 for value in result.explained_variance))
        self.assertFalse(result.issues)

    def test_umap_falls_back_to_pca_when_optional_dependency_is_missing(self):
        with mock.patch(
            "chem_inf_widgets.chemcore.services.molecular_space_service.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'umap'"),
        ):
            result = compute_molecular_space(
                self._demo_matrix(),
                MolecularSpaceConfig(method="umap", n_components=2, random_state=3),
            )

        self.assertEqual(result.method, "pca")
        self.assertEqual(result.coordinates.shape, (4, 2))
        self.assertIsNotNone(result.explained_variance)
        self.assertEqual([issue.code for issue in result.issues], ["umap_not_available"])

    def test_requested_component_count_is_reduced_with_warning(self):
        result = compute_molecular_space(
            self._demo_matrix()[:, :1],
            MolecularSpaceConfig(method="pca", n_components=3, random_state=0),
        )

        self.assertEqual(result.coordinates.shape, (4, 1))
        self.assertEqual(result.method, "pca")
        self.assertEqual([issue.code for issue in result.issues], ["reduced_n_components"])


if __name__ == "__main__":
    unittest.main()

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
    from Orange.data import ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table
except Exception:  # pragma: no cover
    ContinuousVariable = DiscreteVariable = Domain = StringVariable = Table = None  # type: ignore

from chem_inf_widgets.chemcore.services.splitter_service import (  # noqa: E402
    SplitConfig,
    scaffold_groups_by_split,
    split_dataset,
)


@unittest.skipIf(Table is None, "Orange is required for splitter service tests")
class SplitterServiceTests(unittest.TestCase):
    @staticmethod
    def _random_demo_table(n_rows: int = 10) -> Table:
        domain = Domain([ContinuousVariable("x1")])
        X = np.arange(float(n_rows), dtype=float).reshape(-1, 1)
        return Table.from_numpy(domain, X=X)

    @staticmethod
    def _scaffold_demo_table() -> Table:
        domain = Domain(
            [ContinuousVariable("activity")],
            metas=[StringVariable("SMILES"), StringVariable("Name")],
        )
        return Table.from_list(
            domain,
            [
                [5.1, "c1ccccc1O", "phenol"],
                [5.2, "c1ccccc1N", "aniline"],
                [5.3, "c1ccncc1", "pyridine"],
                [5.4, "c1ccncc1O", "hydroxypyridine"],
                [5.5, "CCO", "ethanol"],
                [5.6, "CCCO", "propanol"],
            ],
        )

    @staticmethod
    def _stratified_demo_table() -> Table:
        domain = Domain(
            [ContinuousVariable("x1")],
            class_vars=[DiscreteVariable("Series", values=("A", "B"))],
            metas=[StringVariable("SMILES")],
        )
        X = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]], dtype=float)
        Y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
        metas = np.asarray(
            [["CCO"], ["CCN"], ["CCC"], ["CCCl"], ["c1ccccc1"], ["c1ccncc1"], ["CCCO"], ["CCBr"]],
            dtype=object,
        )
        return Table.from_numpy(domain, X=X, Y=Y, metas=metas)

    def test_random_split_is_reproducible_and_non_overlapping(self):
        table = self._random_demo_table()
        config = SplitConfig(method="random", test_size=0.2, validation_size=0.2, random_state=7)

        first = split_dataset(table, config)
        second = split_dataset(table, config)

        self.assertEqual(first.train_indices, second.train_indices)
        self.assertEqual(first.test_indices, second.test_indices)
        self.assertEqual(first.validation_indices, second.validation_indices)

        train = set(first.train_indices)
        test = set(first.test_indices)
        validation = set(first.validation_indices)
        self.assertFalse(train & test)
        self.assertFalse(train & validation)
        self.assertFalse(test & validation)
        self.assertEqual(len(train | test | validation), len(table))

    def test_scaffold_split_keeps_scaffolds_in_single_subset(self):
        table = self._scaffold_demo_table()
        result = split_dataset(
            table,
            SplitConfig(method="scaffold", test_size=0.2, validation_size=0.2, random_state=3),
        )

        split_scaffolds = scaffold_groups_by_split(table, result)
        self.assertFalse(split_scaffolds["train"] & split_scaffolds["test"])
        self.assertFalse(split_scaffolds["train"] & split_scaffolds["validation"])
        self.assertFalse(split_scaffolds["test"] & split_scaffolds["validation"])
        self.assertEqual(
            len(result.train_indices) + len(result.test_indices) + len(result.validation_indices),
            len(table),
        )

    def test_activity_stratified_split_uses_target_column(self):
        table = self._stratified_demo_table()
        result = split_dataset(
            table,
            SplitConfig(
                method="activity_stratified",
                test_size=0.25,
                validation_size=0.25,
                random_state=11,
                target_column="Series",
            ),
        )

        self.assertFalse(result.issues)
        self.assertEqual(
            len(result.train_indices) + len(result.test_indices) + len(result.validation_indices),
            len(table),
        )

        target_values = np.asarray(table.Y, dtype=float).reshape(-1)
        train_labels = {int(target_values[index]) for index in result.train_indices}
        test_labels = {int(target_values[index]) for index in result.test_indices}
        validation_labels = {int(target_values[index]) for index in result.validation_indices}
        self.assertEqual(train_labels, {0, 1})
        self.assertEqual(test_labels, {0, 1})
        self.assertEqual(validation_labels, {0, 1})

    def test_scaffold_split_reports_missing_smiles_column(self):
        table = self._random_demo_table()
        result = split_dataset(
            table,
            SplitConfig(method="scaffold", test_size=0.2, validation_size=0.2, random_state=0),
        )

        self.assertEqual(result.train_indices, [])
        self.assertEqual(result.test_indices, [])
        self.assertEqual(result.validation_indices, [])
        self.assertEqual([issue.code for issue in result.issues], ["missing_smiles_column"])


if __name__ == "__main__":
    unittest.main()

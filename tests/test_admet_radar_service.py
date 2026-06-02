from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from Orange.data import ContinuousVariable, Domain, StringVariable, Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.admet_radar_service import (  # noqa: E402
    AdmetRadarConfig,
    admet_flagged_records_as_dicts,
    admet_flagged_table,
    admet_radar_records_table,
    admet_radar_summary_table,
    admet_summary_as_rows,
    run_admet_radar,
)


def _demo_table() -> Table:
    domain = Domain(
        [ContinuousVariable("activity")],
        metas=[StringVariable("SMILES"), StringVariable("Name")],
    )
    return Table.from_numpy(
        domain,
        X=np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=float),
        metas=np.asarray(
            [
                ["CCO", "ethanol"],
                ["CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", "long_chain"],
                ["c1ccccc1N=Nc2ccccc2", "azo_probe"],
                ["not-a-smiles", "broken"],
            ],
            dtype=object,
        ),
    )


class AdmetRadarServiceTests(unittest.TestCase):
    def test_admet_radar_computes_flags_and_invalid_rows(self):
        result = run_admet_radar(_demo_table(), AdmetRadarConfig(compute_pains=True, compute_brenk=True))

        self.assertEqual(result.summary.n_rows, 4)
        self.assertEqual(result.summary.n_valid_molecules, 3)
        self.assertEqual(result.summary.n_invalid_molecules, 1)
        self.assertEqual(len(result.records), 4)
        self.assertTrue(any(issue.code == "invalid_molecule" for issue in result.issues))

        ethanol = result.records[0]
        self.assertTrue(ethanol.valid_molecule)
        self.assertTrue(ethanol.lipinski_pass)
        self.assertTrue(ethanol.veber_pass)
        self.assertGreater(ethanol.qed_score, 0.0)

        long_chain = result.records[1]
        self.assertFalse(long_chain.lipinski_pass)
        self.assertFalse(long_chain.ghose_pass)
        self.assertTrue(long_chain.brenk_match)

        azo_probe = result.records[2]
        self.assertTrue(azo_probe.pains_match)
        self.assertTrue(azo_probe.brenk_match)

        broken = result.records[3]
        self.assertFalse(broken.valid_molecule)
        self.assertEqual(broken.status, "Invalid")

    def test_admet_radar_table_exports_work(self):
        result = run_admet_radar(_demo_table())
        records_table = admet_radar_records_table(result)
        flagged_table = admet_flagged_table(result)
        summary_table = admet_radar_summary_table(result)
        flagged_rows = admet_flagged_records_as_dicts(result)
        summary_rows = admet_summary_as_rows(result)

        self.assertIsNotNone(records_table)
        self.assertIsNotNone(flagged_table)
        self.assertIsNotNone(summary_table)
        assert records_table is not None
        assert flagged_table is not None
        assert summary_table is not None
        self.assertEqual(len(records_table), 4)
        self.assertGreaterEqual(len(summary_table), 10)
        self.assertGreaterEqual(len(flagged_table), 2)
        self.assertGreaterEqual(len(flagged_rows), 2)
        self.assertEqual(summary_rows[0]["metric"], "n_rows")
        self.assertIn("qed_score", [var.name for var in records_table.domain.attributes])
        self.assertIn("status", [var.name for var in records_table.domain.metas])

    def test_admet_radar_empty_input_reports_error(self):
        result = run_admet_radar(None)
        self.assertEqual(result.summary.n_rows, 0)
        self.assertEqual([issue.code for issue in result.issues], ["no_input_data"])

    def test_admet_radar_reports_missing_smiles_column(self):
        table = Table.from_numpy(
            Domain([ContinuousVariable("x1")]),
            X=np.asarray([[1.0], [2.0]], dtype=float),
        )
        result = run_admet_radar(table)
        self.assertEqual(result.summary.n_rows, 2)
        self.assertEqual([issue.code for issue in result.issues], ["table_to_molecule_conversion_failed"])


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.descriptors import fingerprints as fingerprint_module  # noqa: E402
from chem_inf_widgets.chemcore.descriptors.fingerprints import compute_fingerprints_from_smiles  # noqa: E402


class FingerprintsTests(unittest.TestCase):
    def test_compute_fingerprints_skips_invalid_smiles(self):
        result = compute_fingerprints_from_smiles(["CCO", "not_a_smiles", ""], fp_type="morgan", bit_size=64)

        self.assertEqual(result.valid_indices, [0])
        self.assertEqual(result.failed_indices, [1, 2])
        self.assertEqual(result.X.shape[0], 1)
        self.assertEqual(result.X.shape[1], 64)
        self.assertEqual([issue.code for issue in result.issues], [
            "fingerprint_input_invalid",
            "fingerprint_input_invalid",
        ])
        self.assertEqual([issue.row_index for issue in result.issues], [2, 3])

    def test_compute_fingerprints_reports_backend_failure_details(self):
        with patch.object(fingerprint_module, "pyAvalonTools", None):
            result = compute_fingerprints_from_smiles(["CCO"], fp_type="avalon", bit_size=64)

        self.assertEqual(result.valid_indices, [])
        self.assertEqual(result.failed_indices, [0])
        self.assertTrue(result.errors)
        self.assertIn("Avalon fingerprint support is unavailable", result.errors[0])
        self.assertEqual(len(result.issues), 1)
        self.assertEqual(result.issues[0].code, "fingerprint_generation_failed")
        self.assertEqual(result.issues[0].row_index, 1)

    def test_compute_fingerprints_honors_cancel(self):
        seen = []

        def progress_cb(pct: int):
            seen.append(pct)

        calls = {"n": 0}

        def cancel_cb():
            calls["n"] += 1
            return calls["n"] > 1

        result = compute_fingerprints_from_smiles(
            ["CCO", "CCN", "CCC"],
            fp_type="rdkit",
            bit_size=32,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )

        self.assertEqual(result.valid_indices, [0])
        self.assertEqual(result.failed_indices, [])
        self.assertTrue(seen)


if __name__ == "__main__":
    unittest.main()

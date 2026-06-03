import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.chembl_bioactivity_service import (  # noqa: E402
    ChemBLBioactivityService,
    canonical_smiles_no_h,
)


class ChemblBioactivityServiceTests(unittest.TestCase):
    def test_canonical_smiles_no_h_returns_canonical_smiles(self):
        self.assertEqual(canonical_smiles_no_h("C(C)O"), "CCO")

    def test_canonical_smiles_no_h_keeps_invalid_input_as_text(self):
        self.assertEqual(canonical_smiles_no_h("not-a-smiles"), "not-a-smiles")

    @patch("chem_inf_widgets.chemcore.services.chembl_bioactivity_service.requests.get")
    def test_fetch_for_target_follows_relative_next_links(self, mock_get):
        first = Mock()
        first.status_code = 200
        first.json.return_value = {
            "activities": [
                {
                    "molecule_chembl_id": "CHEMBL1",
                    "canonical_smiles": "C(C)O",
                    "standard_type": "IC50",
                    "standard_value": "12.0",
                    "standard_units": "nM",
                    "pchembl_value": "8.1",
                }
            ],
            "page_meta": {"next": "/chembl/api/data/activity.json?limit=1&offset=1&target_chembl_id=CHEMBL203&standard_type=IC50"},
        }

        second = Mock()
        second.status_code = 200
        second.json.return_value = {
            "activities": [
                {
                    "molecule_chembl_id": "CHEMBL2",
                    "canonical_smiles": "CCN",
                    "standard_type": "IC50",
                    "standard_value": "18.0",
                    "standard_units": "nM",
                    "pchembl_value": "7.9",
                }
            ],
            "page_meta": {"next": None},
        }

        mock_get.side_effect = [first, second]

        records = ChemBLBioactivityService().fetch_for_target("CHEMBL203", standard_type="IC50", limit=2)

        self.assertEqual([record.molecule_chembl_id for record in records], ["CHEMBL1", "CHEMBL2"])
        self.assertEqual(mock_get.call_args_list[1].args[0], "https://www.ebi.ac.uk/chembl/api/data/activity.json?limit=1&offset=1&target_chembl_id=CHEMBL203&standard_type=IC50")
        self.assertIsNone(mock_get.call_args_list[1].kwargs["params"])


if __name__ == "__main__":
    unittest.main()

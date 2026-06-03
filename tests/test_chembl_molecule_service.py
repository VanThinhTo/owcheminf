import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chem_inf_widgets.chemcore.services.chembl_molecule_service import ChemBLMoleculeService  # noqa: E402


class ChemblMoleculeServiceTests(unittest.TestCase):
    @patch("chem_inf_widgets.chemcore.services.chembl_molecule_service.requests.get")
    def test_fetch_molecules_with_properties_uses_semicolon_set_endpoint(self, mock_get):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "molecules": [
                {
                    "molecule_chembl_id": "CHEMBL25",
                    "pref_name": "Aspirin",
                    "molecule_structures": {"canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O"},
                    "molecule_properties": {"alogp": "1.31"},
                }
            ]
        }
        mock_get.return_value = response

        records = ChemBLMoleculeService().fetch_molecules_with_properties(["CHEMBL25", "CHEMBL1201179"])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].chembl_id, "CHEMBL25")
        self.assertEqual(
            mock_get.call_args.args[0],
            "https://www.ebi.ac.uk/chembl/api/data/molecule/set/CHEMBL25;CHEMBL1201179.json",
        )


if __name__ == "__main__":
    unittest.main()

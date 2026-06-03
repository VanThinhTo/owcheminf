import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chem_inf_widgets.chemcore.services.chembl_target_service import ChemBLTargetService  # noqa: E402
from chem_inf_widgets.chemcore.services.disk_cache import CachePolicy  # noqa: E402


class ChemblTargetServiceTests(unittest.TestCase):
    @patch("chem_inf_widgets.chemcore.services.chembl_target_service.requests.get")
    def test_search_reranks_exact_single_protein_alias_above_complex(self, mock_get):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "targets": [
                {
                    "target_chembl_id": "CHEMBL4523747",
                    "pref_name": "EGFR/PPP1CA",
                    "organism": "Homo sapiens",
                    "target_type": "PROTEIN-PROTEIN INTERACTION",
                    "score": 17.0,
                    "target_components": [
                        {
                            "component_description": "Epidermal growth factor receptor",
                            "target_component_synonyms": [{"component_synonym": "EGFR"}],
                        }
                    ],
                },
                {
                    "target_chembl_id": "CHEMBL203",
                    "pref_name": "Epidermal growth factor receptor",
                    "organism": "Homo sapiens",
                    "target_type": "SINGLE PROTEIN",
                    "score": 9.0,
                    "target_components": [
                        {
                            "component_description": "Epidermal growth factor receptor",
                            "target_component_synonyms": [{"component_synonym": "EGFR"}],
                        }
                    ],
                },
            ],
            "page_meta": {"next": None},
        }
        mock_get.return_value = response

        service = ChemBLTargetService(cache_policy=CachePolicy(enabled=False, ttl_s=0))
        records = service.search("EGFR", limit=2)

        self.assertEqual([record.chembl_id for record in records], ["CHEMBL203", "CHEMBL4523747"])
        self.assertEqual(mock_get.call_args.kwargs["params"]["limit"], 25)


if __name__ == "__main__":
    unittest.main()

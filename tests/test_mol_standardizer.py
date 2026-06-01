import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.mol import ChemMol  # noqa: E402
from chem_inf_widgets.chemcore.molecule_contract import INPUT_SMILES  # noqa: E402
from chem_inf_widgets.chemcore.services import mol_standardizer as mol_standardizer_module  # noqa: E402
from chem_inf_widgets.chemcore.services.mol_standardizer import MolStandardizer  # noqa: E402


class MolStandardizerTests(unittest.TestCase):
    def test_standardize_smiles_invalid(self):
        service = MolStandardizer()

        result = service.standardize_smiles("not_a_smiles")

        self.assertFalse(result.ok)
        self.assertEqual(result.output_smiles, "")

    def test_standardize_chemmols_uses_rdkit_fallback_smiles(self):
        service = MolStandardizer()
        chem_mol = ChemMol.from_smiles("C(C)O", name="ethanol")
        chem_mol.set_prop("SMILES", "")

        out_mols, results = service.standardize_chemmols([chem_mol])

        self.assertEqual(len(out_mols), 1)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].ok)
        self.assertTrue(bool(out_mols[0].get_prop("SMILES_STD")))

    def test_standardize_chemmols_reports_audit_write_failure(self):
        service = MolStandardizer()
        chem_mol = ChemMol.from_smiles("CCO", name="ethanol")
        chem_mol.set_prop("SMILES", "CCO")
        original_ensure_contract_props = mol_standardizer_module.ensure_contract_props
        call_count = 0

        def flaky_ensure_contract_props(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("audit contract boom")
            return original_ensure_contract_props(*args, **kwargs)

        with mock.patch.object(
            mol_standardizer_module,
            "ensure_contract_props",
            side_effect=flaky_ensure_contract_props,
        ):
            out_mols, results = service.standardize_chemmols([chem_mol])

        self.assertEqual(len(out_mols), 1)
        self.assertTrue(results[0].ok)
        self.assertEqual(len(results[0].issues), 1)
        self.assertEqual(results[0].issues[0].code, "standardization_audit_write_failed")
        self.assertEqual(results[0].issues[0].row_index, 1)
        self.assertIn("audit contract boom", results[0].issues[0].message)
        self.assertIn("Warning: Could not write standardization audit fields", results[0].log)
        self.assertEqual(out_mols[0].get_prop(INPUT_SMILES), "CCO")

    def test_standardize_chemmols_reports_structure_update_failure(self):
        service = MolStandardizer(profile="qsar_ready")
        chem_mol = ChemMol.from_smiles("CCO.Cl", name="ethanol_salt")
        chem_mol.set_prop("SMILES", "CCO.Cl")

        with mock.patch.object(
            mol_standardizer_module.ChemMol,
            "from_rdkit",
            side_effect=RuntimeError("rdkit rebuild boom"),
        ):
            out_mols, results = service.standardize_chemmols([chem_mol])

        self.assertEqual(len(out_mols), 1)
        self.assertTrue(results[0].ok)
        self.assertEqual(len(results[0].issues), 1)
        self.assertEqual(results[0].issues[0].code, "standardized_structure_update_failed")
        self.assertEqual(results[0].issues[0].row_index, 1)
        self.assertIn("rdkit rebuild boom", results[0].issues[0].message)
        self.assertIn("Warning: Could not update standardized ChemMol structure", results[0].log)
        self.assertEqual(out_mols[0].get_prop("SMILES_STD"), "CCO")


if __name__ == "__main__":
    unittest.main()

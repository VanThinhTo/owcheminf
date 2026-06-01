from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("rdkit")

from chem_inf_widgets.chemcore.services.rdkit_descriptor_service import RdkitDescriptorService


def test_rdkit_descriptor_service_reports_descriptor_failures_with_row_mapping():
    service = RdkitDescriptorService()
    mol = service.smiles_to_mols(["CCO"])[0][0]
    assert mol is not None

    with mock.patch.dict(service._functions, {"MolWt": lambda _mol: (_ for _ in ()).throw(RuntimeError("mw boom"))}):
        result = service.compute_with_issues([mol], ["MolWt"], row_indices=[4])

    assert list(result.frame.columns) == ["MolWt"]
    assert result.frame.iloc[0]["MolWt"] != result.frame.iloc[0]["MolWt"]  # NaN
    assert len(result.issues) == 1
    assert result.issues[0].code == "rdkit_descriptor_computation_failed"
    assert result.issues[0].row_index == 5
    assert result.issues[0].details["descriptor_name"] == "MolWt"
    assert "mw boom" in result.issues[0].message

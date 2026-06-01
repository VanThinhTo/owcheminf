from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("rdkit")

from chem_inf_widgets.chemcore.services import molecule_qc_service
from chem_inf_widgets.chemcore.services.molecule_qc_service import (
    MoleculeQCConfig,
    qc_records_as_dicts,
    qc_summary_as_rows,
    run_molecule_qc,
)


def test_molecule_qc_flags_invalid_and_duplicates():
    result = run_molecule_qc([
        "CCO",
        "CCO",
        "[Na+].[O-]C(=O)C",
        "",
    ])
    assert result.summary.total == 4
    assert result.summary.invalid == 1
    assert result.summary.duplicate_groups >= 1
    assert result.summary.issue_counts.get("DUPLICATE_STRUCTURE", 0) >= 2
    assert result.summary.issue_counts.get("MULTI_FRAGMENT", 0) >= 1


def test_molecule_qc_summary_rows_are_exportable():
    result = run_molecule_qc(["CCO", "C[N+](C)(C)C"])
    rows = qc_summary_as_rows(result.summary)
    assert rows
    assert any(row["metric"] == "total_records" for row in rows)


def test_molecule_qc_record_dicts_have_expected_columns():
    result = run_molecule_qc([""])
    rows = qc_records_as_dicts(result.records)
    assert "canonical_smiles" in rows[0]
    assert "issue_codes" in rows[0]
    assert "molecular_weight" in rows[0]
    assert rows[0]["qc_flags"] == "invalid_structure"
    assert rows[0]["dropped_reason"] == "invalid_structure"


def test_molecule_qc_config_can_relax_charge_flag():
    strict = run_molecule_qc(["C[N+](C)(C)C"])
    relaxed = run_molecule_qc(["C[N+](C)(C)C"], MoleculeQCConfig(flag_formal_charge=False))
    assert strict.summary.issue_counts.get("NET_FORMAL_CHARGE", 0) == 1
    assert relaxed.summary.issue_counts.get("NET_FORMAL_CHARGE", 0) == 0


def test_molecule_qc_reports_molecular_weight_computation_failure():
    with mock.patch.object(
        molecule_qc_service.Descriptors,
        "MolWt",
        side_effect=RuntimeError("mw boom"),
    ):
        result = run_molecule_qc(["CCO"])

    assert result.summary.issue_counts.get("MOLECULAR_WEIGHT_COMPUTATION_FAILED", 0) == 1
    assert result.records[0].severity == "WARNING"
    assert "MOLECULAR_WEIGHT_COMPUTATION_FAILED" in result.records[0].issue_codes
    assert any("mw boom" in message for message in result.records[0].issues)
    assert len(result.issues) == 1
    assert result.issues[0].code == "molecular_weight_computation_failed"
    assert result.issues[0].row_index == 1


def test_molecule_qc_reports_fragment_analysis_failure():
    with mock.patch.object(
        molecule_qc_service.Chem,
        "GetMolFrags",
        side_effect=RuntimeError("frag boom"),
    ):
        result = run_molecule_qc(["CCO"])

    assert result.summary.issue_counts.get("FRAGMENT_ANALYSIS_FAILED", 0) == 1
    assert "FRAGMENT_ANALYSIS_FAILED" in result.records[0].issue_codes
    assert any("frag boom" in message for message in result.records[0].issues)
    assert len(result.issues) == 1
    assert result.issues[0].code == "fragment_analysis_failed"
    assert result.issues[0].row_index == 1

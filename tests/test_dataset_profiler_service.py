from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("Orange")
pytest.importorskip("rdkit")

from Orange.data import ContinuousVariable, Domain, StringVariable, Table

from chem_inf_widgets.chemcore.services.dataset_profiler_service import (
    dataset_profile_summary_as_rows,
    descriptor_summary_as_rows,
    problematic_compounds_as_dicts,
    profiled_records_table,
    run_dataset_profiler,
    summary_table,
)


def _demo_table() -> Table:
    domain = Domain(
        [ContinuousVariable("activity")],
        metas=[
            StringVariable("SMILES"),
            StringVariable("Name"),
            StringVariable("Series"),
        ],
    )
    return Table.from_numpy(
        domain,
        X=np.asarray([[1.0], [np.nan], [2.5], [3.0], [4.0]], dtype=float),
        metas=np.asarray(
            [
                ["CCO", "ethanol_a", "A"],
                ["OCC", "ethanol_b", "A"],
                ["not-a-smiles", "broken", "B"],
                ["c1ccccc1N=Nc2ccccc2", "azo", ""],
                ["CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", "wax", "C"],
            ],
            dtype=object,
        ),
    )


def test_dataset_profiler_reports_core_counts_and_summaries():
    result = run_dataset_profiler(_demo_table())

    assert result.summary.n_rows == 5
    assert result.summary.n_valid_molecules == 4
    assert result.summary.n_invalid_molecules == 1
    assert result.summary.duplicate_smiles_count == 1
    assert result.summary.duplicate_smiles_groups == 1
    assert result.summary.rows_with_missing_values == 2
    assert result.summary.n_lipinski_fail >= 1
    assert result.summary.n_pains_matches >= 1
    assert result.summary.missing_value_counts["activity"] == 1
    assert result.summary.missing_value_counts["Series"] == 1
    assert result.summary.issue_counts["invalid_structure"] == 1
    assert result.summary.issue_counts["duplicate_smiles"] == 2
    assert any(record.canonical_smiles == "CCO" and record.duplicate_smiles for record in result.records)
    assert any(record.pains_match for record in result.records if record.valid_molecule)
    assert any(not record.lipinski_pass for record in result.records if record.valid_molecule)

    descriptor_rows = descriptor_summary_as_rows(result.descriptor_summary)
    assert any(row["descriptor"] == "molecular_weight" for row in descriptor_rows)
    assert any(row["count"] == 4 for row in descriptor_rows if row["descriptor"] == "molecular_weight")

    summary_rows = dataset_profile_summary_as_rows(result.summary)
    assert any(row["metric"] == "duplicate_smiles_count" and row["value"] == 1 for row in summary_rows)
    assert any(row["metric"] == "missing_activity" and row["value"] == 1 for row in summary_rows)


def test_dataset_profiler_problematic_rows_and_tables_are_exportable():
    result = run_dataset_profiler(_demo_table())

    problematic = problematic_compounds_as_dicts(result.records)
    problematic_names = {row["name"] for row in problematic}
    assert {"ethanol_a", "ethanol_b", "broken", "azo", "wax"} <= problematic_names

    profiled = profiled_records_table(result)
    summary = summary_table(result)

    assert profiled is not None
    assert summary is not None
    assert len(profiled) == 5
    assert "canonical_smiles" in [var.name for var in profiled.domain.metas]
    assert "value" in [var.name for var in summary.domain.attributes]


def test_dataset_profiler_returns_error_without_smiles_column():
    table = Table.from_numpy(
        Domain([ContinuousVariable("activity")]),
        X=np.asarray([[1.0], [2.0]], dtype=float),
    )

    result = run_dataset_profiler(table)

    assert result.summary.n_valid_molecules == 0
    assert result.summary.issue_counts["dataset_profile_failed"] == 1
    assert len(result.issues) == 1
    assert result.issues[0].severity == "error"

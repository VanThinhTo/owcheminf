from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("Orange")
pytest.importorskip("rdkit")

from Orange.data import ContinuousVariable, Domain, StringVariable, Table

from chem_inf_widgets.chemcore.services.chemical_series_service import (
    ChemicalSeriesConfig,
    chemical_series_members_as_dicts,
    chemical_series_members_table,
    chemical_series_summary_as_rows,
    chemical_series_summary_table,
    chemical_series_table,
    run_chemical_series_explorer,
    series_rows_as_dicts,
)
from chem_inf_widgets.chemcore.services.scaffold_service import NO_SCAFFOLD_LABEL


def _demo_table() -> Table:
    domain = Domain(
        [ContinuousVariable("activity")],
        metas=[StringVariable("SMILES"), StringVariable("Name")],
    )
    return Table.from_numpy(
        domain,
        X=np.asarray([[1.1], [2.2], [3.3], [4.4], [5.5], [6.6]], dtype=float),
        metas=np.asarray(
            [
                ["Cc1ccccc1", "toluene"],
                ["Oc1ccccc1", "phenol"],
                ["CC1CCCCC1", "ethyl_cyclohexane"],
                ["OC1CCCCC1", "cyclohexanol"],
                ["CCO", "ethanol"],
                ["not-a-smiles", "broken"],
            ],
            dtype=object,
        ),
    )


def test_chemical_series_service_groups_members_and_summaries():
    result = run_chemical_series_explorer(_demo_table())

    assert result.summary.n_rows == 6
    assert result.summary.n_valid_molecules == 5
    assert result.summary.n_invalid_molecules == 1
    assert result.summary.n_series == 3
    assert result.summary.n_singleton_series == 1
    assert result.summary.n_acyclic_rows == 1
    assert result.summary.target_column == "activity"
    assert result.summary.issue_counts["invalid_molecule"] == 1

    series_rows = series_rows_as_dicts(result)
    assert series_rows[0]["count"] == 2
    assert any(row["scaffold"] == "c1ccccc1" and row["count"] == 2 for row in series_rows)
    assert any(row["scaffold"] == "C1CCCCC1" and row["count"] == 2 for row in series_rows)
    assert any(row["scaffold"] == NO_SCAFFOLD_LABEL and row["count"] == 1 for row in series_rows)
    assert any(row["scaffold"] == "c1ccccc1" and row["mean_activity"] == 1.65 for row in series_rows)

    members = chemical_series_members_as_dicts(result)
    broken = next(row for row in members if row["name"] == "broken")
    assert broken["status"] == "invalid"
    assert broken["series_size"] == 0
    assert "Could not parse molecule" in str(broken["issues"])


def test_chemical_series_service_supports_generic_scaffolds_and_exports():
    result = run_chemical_series_explorer(
        _demo_table(),
        ChemicalSeriesConfig(scaffold_kind="generic"),
    )

    assert any(row.scaffold == "C1CCCCC1" for row in result.series_rows)
    assert result.summary.n_series == 2

    members_table = chemical_series_members_table(result)
    series_table = chemical_series_table(result)
    summary_table = chemical_series_summary_table(result)
    summary_rows = chemical_series_summary_as_rows(result)

    assert members_table is not None
    assert series_table is not None
    assert summary_table is not None
    assert len(members_table) == 6
    assert len(series_table) == 2
    assert any(row["metric"] == "n_series" and row["value"] == 2 for row in summary_rows)


def test_chemical_series_service_reports_missing_smiles_column():
    table = Table.from_numpy(
        Domain([ContinuousVariable("activity")]),
        X=np.asarray([[1.0], [2.0]], dtype=float),
    )

    result = run_chemical_series_explorer(table)

    assert result.summary.n_valid_molecules == 0
    assert result.summary.n_series == 0
    assert len(result.issues) == 1
    assert result.issues[0].code == "chemical_series_failed"
    assert result.issues[0].severity == "error"

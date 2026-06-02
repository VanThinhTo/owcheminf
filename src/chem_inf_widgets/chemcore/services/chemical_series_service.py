from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Literal

from Orange.data import ContinuousVariable, Table, Variable

from chem_inf_widgets.chemcore.qsar import TARGET_COLUMN_CANDIDATES
from chem_inf_widgets.chemcore.result import ServiceIssue, count_issues
from chem_inf_widgets.chemcore.services.from_orange import table_to_chemmols_with_report
from chem_inf_widgets.chemcore.services.orange_table_utils import records_to_orange_table
from chem_inf_widgets.chemcore.services.scaffold_service import (
    analyze_scaffolds,
    build_scaffold_summary,
)

ScaffoldKind = Literal["murcko", "generic"]


@dataclass(frozen=True)
class ChemicalSeriesConfig:
    scaffold_kind: ScaffoldKind = "murcko"
    target_column: str | None = None
    activity_log_scale: bool = False


@dataclass(frozen=True)
class ChemicalSeriesMemberRecord:
    row_index: int
    name: str
    input_smiles: str
    series_scaffold: str
    status: str
    valid_molecule: bool
    series_size: int
    activity: float = float("nan")
    issue_codes: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChemicalSeriesSummaryRow:
    scaffold: str
    kind: ScaffoldKind
    count: int
    fraction: float
    mean_activity: float = float("nan")
    best_activity: float = float("nan")
    worst_activity: float = float("nan")
    std_activity: float = float("nan")


@dataclass(frozen=True)
class ChemicalSeriesSummary:
    n_rows: int
    n_valid_molecules: int
    n_invalid_molecules: int
    n_series: int
    n_singleton_series: int
    n_acyclic_rows: int
    target_column: str = ""
    issue_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ChemicalSeriesResult:
    summary: ChemicalSeriesSummary
    members: tuple[ChemicalSeriesMemberRecord, ...]
    series_rows: tuple[ChemicalSeriesSummaryRow, ...]
    issues: tuple[ServiceIssue, ...] = ()


def _iter_all_vars(table: Table) -> list[Variable]:
    return list(table.domain.attributes) + list(table.domain.class_vars) + list(table.domain.metas)


def _find_var_by_name(table: Table, name: str | None) -> Variable | None:
    wanted = str(name or "").strip().lower()
    if not wanted:
        return None
    for var in _iter_all_vars(table):
        if var.name.strip().lower() == wanted:
            return var
    return None


def _column_strings(table: Table, var_name: str | None) -> list[str]:
    var = _find_var_by_name(table, var_name)
    if var is None:
        return [""] * len(table)
    return ["" if value is None else str(value) for value in table.get_column(var)]


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _auto_target_var(table: Table) -> Variable | None:
    candidates = list(table.domain.class_vars) + list(table.domain.attributes) + list(table.domain.metas)
    for var in candidates:
        if isinstance(var, ContinuousVariable) and _normalize_name(var.name) in TARGET_COLUMN_CANDIDATES:
            return var
    for var in table.domain.class_vars:
        if isinstance(var, ContinuousVariable):
            return var
    return None


def _column_floats(table: Table, var: Variable | None) -> list[float]:
    if var is None:
        return [float("nan")] * len(table)

    values: list[float] = []
    for value in table.get_column(var):
        if value is None:
            values.append(float("nan"))
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            try:
                number = float(str(value).strip().replace(",", "."))
            except (TypeError, ValueError):
                number = float("nan")
        values.append(number)
    return values


def _series_activity_stats(values: list[float], *, activity_log_scale: bool) -> tuple[float, float, float, float]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        nan = float("nan")
        return nan, nan, nan, nan

    best_value = max(finite_values) if activity_log_scale else min(finite_values)
    worst_value = min(finite_values) if activity_log_scale else max(finite_values)
    std_value = statistics.stdev(finite_values) if len(finite_values) > 1 else 0.0
    return (
        round(statistics.mean(finite_values), 4),
        round(best_value, 4),
        round(worst_value, 4),
        round(std_value, 4),
    )


def _empty_result(
    *,
    n_rows: int,
    target_column: str = "",
    issues: list[ServiceIssue],
) -> ChemicalSeriesResult:
    return ChemicalSeriesResult(
        summary=ChemicalSeriesSummary(
            n_rows=n_rows,
            n_valid_molecules=0,
            n_invalid_molecules=n_rows,
            n_series=0,
            n_singleton_series=0,
            n_acyclic_rows=0,
            target_column=target_column,
            issue_counts=count_issues(issues),
        ),
        members=(),
        series_rows=(),
        issues=tuple(issues),
    )


def run_chemical_series_explorer(
    data: Table | None,
    config: ChemicalSeriesConfig | None = None,
) -> ChemicalSeriesResult:
    cfg = config or ChemicalSeriesConfig()
    issues: list[ServiceIssue] = []
    if data is None or len(data) == 0:
        issues.append(
            ServiceIssue(
                code="no_input_data",
                message="No input data for chemical series exploration.",
                severity="error",
            )
        )
        return _empty_result(n_rows=0, issues=issues)

    try:
        _mols, report = table_to_chemmols_with_report(data)
    except (ImportError, ValueError) as exc:
        issues.append(
            ServiceIssue(
                code="chemical_series_failed",
                message=str(exc),
                severity="error",
            )
        )
        return _empty_result(n_rows=len(data), issues=issues)

    names = _column_strings(data, report.name_column)
    smiles = _column_strings(data, report.smiles_column)
    scaffold_result = analyze_scaffolds(smiles)
    target_var = _find_var_by_name(data, cfg.target_column) if cfg.target_column else _auto_target_var(data)
    if cfg.target_column and target_var is None:
        issues.append(
            ServiceIssue(
                code="target_column_missing",
                message=f"Target column '{cfg.target_column}' was not found; activity summaries were skipped.",
                severity="warning",
                field="target_column",
            )
        )
    activity_values = _column_floats(data, target_var)

    series_counts = (
        dict(scaffold_result.murcko_counts)
        if cfg.scaffold_kind == "murcko"
        else dict(scaffold_result.generic_counts)
    )
    activity_by_scaffold: dict[str, list[float]] = {}
    members: list[ChemicalSeriesMemberRecord] = []

    for annotation in scaffold_result.annotations:
        row_index = int(annotation.index) + 1
        series_scaffold_raw = annotation.murcko if cfg.scaffold_kind == "murcko" else annotation.generic
        series_scaffold = str(series_scaffold_raw or "")
        activity_value = activity_values[annotation.index] if annotation.index < len(activity_values) else float("nan")
        if annotation.status == "invalid":
            message = f"Could not parse molecule from row {row_index}."
            issues.append(
                ServiceIssue(
                    code="invalid_molecule",
                    message=message,
                    severity="warning",
                    row_index=row_index,
                )
            )
            members.append(
                ChemicalSeriesMemberRecord(
                    row_index=row_index,
                    name=names[annotation.index] if annotation.index < len(names) else "",
                    input_smiles=smiles[annotation.index] if annotation.index < len(smiles) else "",
                    series_scaffold="",
                    status="invalid",
                    valid_molecule=False,
                    series_size=0,
                    activity=activity_value,
                    issue_codes=("INVALID_MOLECULE",),
                    issues=(message,),
                )
            )
            continue

        if series_scaffold:
            activity_by_scaffold.setdefault(series_scaffold, [])
            if math.isfinite(activity_value):
                activity_by_scaffold[series_scaffold].append(activity_value)

        members.append(
            ChemicalSeriesMemberRecord(
                row_index=row_index,
                name=names[annotation.index] if annotation.index < len(names) else "",
                input_smiles=smiles[annotation.index] if annotation.index < len(smiles) else "",
                series_scaffold=series_scaffold,
                status=str(annotation.status),
                valid_molecule=True,
                series_size=int(series_counts.get(series_scaffold, 0)),
                activity=activity_value,
            )
        )

    series_rows: list[ChemicalSeriesSummaryRow] = []
    for row in build_scaffold_summary(scaffold_result, kind=cfg.scaffold_kind):
        mean_activity, best_activity, worst_activity, std_activity = _series_activity_stats(
            activity_by_scaffold.get(row.scaffold, []),
            activity_log_scale=cfg.activity_log_scale,
        )
        series_rows.append(
            ChemicalSeriesSummaryRow(
                scaffold=row.scaffold,
                kind=cfg.scaffold_kind,
                count=row.count,
                fraction=row.fraction,
                mean_activity=mean_activity,
                best_activity=best_activity,
                worst_activity=worst_activity,
                std_activity=std_activity,
            )
        )

    summary = ChemicalSeriesSummary(
        n_rows=len(data),
        n_valid_molecules=scaffold_result.valid_count,
        n_invalid_molecules=len(scaffold_result.failed_indices),
        n_series=len(series_rows),
        n_singleton_series=sum(1 for row in series_rows if row.count == 1),
        n_acyclic_rows=sum(1 for annotation in scaffold_result.annotations if annotation.status == "acyclic"),
        target_column="" if target_var is None else target_var.name,
        issue_counts=count_issues(issues),
    )
    return ChemicalSeriesResult(
        summary=summary,
        members=tuple(members),
        series_rows=tuple(series_rows),
        issues=tuple(issues),
    )


def chemical_series_members_as_dicts(result: ChemicalSeriesResult) -> list[dict[str, object]]:
    return [
        {
            "row_index": record.row_index,
            "name": record.name,
            "input_smiles": record.input_smiles,
            "series_scaffold": record.series_scaffold,
            "status": record.status,
            "valid_molecule": float(record.valid_molecule),
            "series_size": record.series_size,
            "activity": record.activity,
            "issues": " | ".join(record.issues),
        }
        for record in result.members
    ]


def chemical_series_summary_as_rows(result: ChemicalSeriesResult) -> list[dict[str, object]]:
    rows = [
        {
            "metric": "n_rows",
            "value": result.summary.n_rows,
            "description": "All input rows.",
        },
        {
            "metric": "n_valid_molecules",
            "value": result.summary.n_valid_molecules,
            "description": "Rows with a valid parsed molecule.",
        },
        {
            "metric": "n_invalid_molecules",
            "value": result.summary.n_invalid_molecules,
            "description": "Rows that could not be parsed as molecules.",
        },
        {
            "metric": "n_series",
            "value": result.summary.n_series,
            "description": "Unique scaffold-defined chemical series.",
        },
        {
            "metric": "n_singleton_series",
            "value": result.summary.n_singleton_series,
            "description": "Series containing a single molecule.",
        },
        {
            "metric": "n_acyclic_rows",
            "value": result.summary.n_acyclic_rows,
            "description": "Valid molecules without a Murcko scaffold.",
        },
    ]
    if result.summary.target_column:
        rows.append(
            {
                "metric": "target_column",
                "value": result.summary.target_column,
                "description": "Numeric activity column used for per-series statistics.",
            }
        )
    for issue in result.issues:
        rows.append(
            {
                "metric": f"issue_{issue.code}",
                "value": 1,
                "description": issue.message,
            }
        )
    return rows


def series_rows_as_dicts(result: ChemicalSeriesResult) -> list[dict[str, object]]:
    return [
        {
            "scaffold": row.scaffold,
            "kind": row.kind,
            "count": row.count,
            "fraction": row.fraction,
            "mean_activity": row.mean_activity,
            "best_activity": row.best_activity,
            "worst_activity": row.worst_activity,
            "std_activity": row.std_activity,
        }
        for row in result.series_rows
    ]


def chemical_series_members_table(result: ChemicalSeriesResult) -> Table | None:
    return records_to_orange_table(
        chemical_series_members_as_dicts(result),
        meta_columns=["row_index", "name", "input_smiles", "series_scaffold", "status", "issues"],
        name="Chemical Series Members",
    )


def chemical_series_table(result: ChemicalSeriesResult) -> Table | None:
    return records_to_orange_table(
        series_rows_as_dicts(result),
        meta_columns=["scaffold", "kind"],
        name="Chemical Series Summary",
    )


def chemical_series_summary_table(result: ChemicalSeriesResult) -> Table | None:
    return records_to_orange_table(
        chemical_series_summary_as_rows(result),
        meta_columns=["metric", "description"],
        name="Chemical Series Service Summary",
    )


__all__ = [
    "ChemicalSeriesConfig",
    "ChemicalSeriesMemberRecord",
    "ChemicalSeriesResult",
    "ChemicalSeriesSummary",
    "ChemicalSeriesSummaryRow",
    "chemical_series_members_as_dicts",
    "chemical_series_members_table",
    "chemical_series_summary_as_rows",
    "chemical_series_summary_table",
    "chemical_series_table",
    "run_chemical_series_explorer",
    "series_rows_as_dicts",
]

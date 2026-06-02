from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from Orange.data import Table, Variable

from chem_inf_widgets.chemcore.molecule_contract import SOURCE_ROW_INDEX
from chem_inf_widgets.chemcore.result import ServiceIssue, count_issues
from chem_inf_widgets.chemcore.services.drug_filter_service import (
    FilterConfig,
    lipinski_stats,
    pains_match_info,
    veber_stats,
)
from chem_inf_widgets.chemcore.services.from_orange import (
    TableMolConversionReport,
    table_to_chemmols_with_report,
)
from chem_inf_widgets.chemcore.services.orange_table_utils import (
    is_missing,
    records_to_orange_table,
)


@dataclass(frozen=True)
class DatasetProfilerConfig:
    compute_pains: bool = True


@dataclass(frozen=True)
class MissingValueStat:
    column: str
    missing_count: int
    missing_fraction: float


@dataclass(frozen=True)
class DescriptorSummaryStat:
    descriptor: str
    count: int
    mean: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class DatasetProfilerRecord:
    row_index: int
    name: str
    input_smiles: str
    canonical_smiles: str
    status: str
    valid_molecule: bool
    duplicate_smiles: bool
    duplicate_group_size: int
    missing_value_count: int
    missing_fields: tuple[str, ...] = ()
    lipinski_violations: float = float("nan")
    lipinski_pass: bool = False
    pains_match: bool = False
    pains_regid: str = ""
    molecular_weight: float = float("nan")
    logp: float = float("nan")
    hbd: float = float("nan")
    hba: float = float("nan")
    tpsa: float = float("nan")
    rotatable_bonds: float = float("nan")
    issue_codes: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetProfilerSummary:
    n_rows: int
    n_valid_molecules: int
    n_invalid_molecules: int
    duplicate_smiles_count: int
    duplicate_smiles_groups: int
    rows_with_missing_values: int
    n_lipinski_pass: int
    n_lipinski_fail: int
    n_pains_matches: int
    missing_value_counts: dict[str, int] = field(default_factory=dict)
    issue_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetProfilerResult:
    summary: DatasetProfilerSummary
    records: tuple[DatasetProfilerRecord, ...]
    missing_value_summary: tuple[MissingValueStat, ...]
    descriptor_summary: tuple[DescriptorSummaryStat, ...]
    issues: tuple[ServiceIssue, ...] = ()


_DESCRIPTOR_FIELDS = (
    "molecular_weight",
    "logp",
    "hbd",
    "hba",
    "tpsa",
    "rotatable_bonds",
)

_ROW_ERROR_RE = re.compile(r"^Row\s+(?P<row>\d+):\s*(?P<message>.+)$")
_BASE_FILTER_CFG = FilterConfig(
    filter_rule="Lipinski",
    selection_mode="Forward All Molecules",
    compute_qed=False,
    highlight_pains_atoms=False,
)


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


def _missing_fields_for_row(table: Table, row_index: int) -> tuple[str, ...]:
    missing: list[str] = []
    row = table[row_index]
    for var in _iter_all_vars(table):
        if is_missing(row[var]):
            missing.append(var.name)
    return tuple(missing)


def _missing_value_stats(table: Table) -> tuple[MissingValueStat, ...]:
    rows = max(len(table), 1)
    stats: list[MissingValueStat] = []
    for var in _iter_all_vars(table):
        values = table.get_column(var)
        missing_count = sum(1 for value in values if is_missing(value))
        stats.append(
            MissingValueStat(
                column=var.name,
                missing_count=int(missing_count),
                missing_fraction=float(missing_count) / float(rows),
            )
        )
    return tuple(stats)


def _invalid_row_messages(report: TableMolConversionReport) -> dict[int, str]:
    by_row: dict[int, str] = {}
    for message in report.errors:
        match = _ROW_ERROR_RE.match(str(message or "").strip())
        if match is None:
            continue
        by_row[int(match.group("row"))] = match.group("message")
    return by_row


def _nan_descriptor_values() -> dict[str, float]:
    return {field: float("nan") for field in _DESCRIPTOR_FIELDS}


def _finite_descriptor_values(records: Iterable[DatasetProfilerRecord], field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = getattr(record, field)
        if value is None:
            continue
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_float):
            values.append(value_float)
    return values


def _build_descriptor_summary(records: Iterable[DatasetProfilerRecord]) -> tuple[DescriptorSummaryStat, ...]:
    stats: list[DescriptorSummaryStat] = []
    record_list = list(records)
    for descriptor_name in _DESCRIPTOR_FIELDS:
        values = _finite_descriptor_values(record_list, descriptor_name)
        if not values:
            stats.append(
                DescriptorSummaryStat(
                    descriptor=descriptor_name,
                    count=0,
                    mean=float("nan"),
                    minimum=float("nan"),
                    maximum=float("nan"),
                )
            )
            continue
        stats.append(
            DescriptorSummaryStat(
                descriptor=descriptor_name,
                count=len(values),
                mean=sum(values) / len(values),
                minimum=min(values),
                maximum=max(values),
            )
        )
    return tuple(stats)


def _build_valid_record(
    *,
    row_index: int,
    name: str,
    input_smiles: str,
    canonical_smiles: str,
    missing_fields: tuple[str, ...],
    duplicate_group_size: int,
    lipinski_violations: float,
    lipinski_pass: bool,
    pains_match: bool,
    pains_regid: str,
    descriptor_values: dict[str, float],
    issue_codes: list[str],
    issue_messages: list[str],
) -> DatasetProfilerRecord:
    return DatasetProfilerRecord(
        row_index=row_index,
        name=name,
        input_smiles=input_smiles,
        canonical_smiles=canonical_smiles,
        status="Valid",
        valid_molecule=True,
        duplicate_smiles=duplicate_group_size > 1,
        duplicate_group_size=duplicate_group_size,
        missing_value_count=len(missing_fields),
        missing_fields=missing_fields,
        lipinski_violations=lipinski_violations,
        lipinski_pass=bool(lipinski_pass),
        pains_match=bool(pains_match),
        pains_regid=pains_regid,
        molecular_weight=descriptor_values["molecular_weight"],
        logp=descriptor_values["logp"],
        hbd=descriptor_values["hbd"],
        hba=descriptor_values["hba"],
        tpsa=descriptor_values["tpsa"],
        rotatable_bonds=descriptor_values["rotatable_bonds"],
        issue_codes=tuple(issue_codes),
        issues=tuple(issue_messages),
    )


def _build_invalid_record(
    *,
    row_index: int,
    name: str,
    input_smiles: str,
    missing_fields: tuple[str, ...],
    error_message: str,
) -> DatasetProfilerRecord:
    return DatasetProfilerRecord(
        row_index=row_index,
        name=name,
        input_smiles=input_smiles,
        canonical_smiles="",
        status="Invalid",
        valid_molecule=False,
        duplicate_smiles=False,
        duplicate_group_size=0,
        missing_value_count=len(missing_fields),
        missing_fields=missing_fields,
        pains_regid="",
        issue_codes=("INVALID_STRUCTURE",),
        issues=(error_message,),
        **_nan_descriptor_values(),
    )


def run_dataset_profiler(
    data: Table | None,
    config: DatasetProfilerConfig | None = None,
) -> DatasetProfilerResult:
    cfg = config or DatasetProfilerConfig()
    if data is None:
        issue = ServiceIssue(
            code="no_input_data",
            message="No input data.",
            severity="error",
        )
        return DatasetProfilerResult(
            summary=DatasetProfilerSummary(
                n_rows=0,
                n_valid_molecules=0,
                n_invalid_molecules=0,
                duplicate_smiles_count=0,
                duplicate_smiles_groups=0,
                rows_with_missing_values=0,
                n_lipinski_pass=0,
                n_lipinski_fail=0,
                n_pains_matches=0,
                issue_counts={issue.code: 1},
            ),
            records=(),
            missing_value_summary=(),
            descriptor_summary=(),
            issues=(issue,),
        )

    issues: list[ServiceIssue] = []
    try:
        mols, report = table_to_chemmols_with_report(data)
    except (ImportError, ValueError) as exc:
        issue = ServiceIssue(
            code="dataset_profile_failed",
            message=str(exc),
            severity="error",
        )
        return DatasetProfilerResult(
            summary=DatasetProfilerSummary(
                n_rows=len(data),
                n_valid_molecules=0,
                n_invalid_molecules=len(data),
                duplicate_smiles_count=0,
                duplicate_smiles_groups=0,
                rows_with_missing_values=0,
                n_lipinski_pass=0,
                n_lipinski_fail=0,
                n_pains_matches=0,
                issue_counts={issue.code: 1},
            ),
            records=(),
            missing_value_summary=_missing_value_stats(data),
            descriptor_summary=(),
            issues=(issue,),
        )

    missing_stats = _missing_value_stats(data)
    missing_counts = {stat.column: stat.missing_count for stat in missing_stats if stat.missing_count}
    rows_with_missing_values = sum(1 for row_index in range(len(data)) if _missing_fields_for_row(data, row_index))

    input_smiles_values = _column_strings(data, report.smiles_column)
    name_values = _column_strings(data, report.name_column)
    invalid_messages = _invalid_row_messages(report)
    valid_by_row = {
        int(cm.props.get(SOURCE_ROW_INDEX) or 0): cm
        for cm in mols
    }
    canonical_counts = Counter(
        str(cm.props.get("canonical_smiles") or cm.props.get("SMILES") or "").strip()
        for cm in mols
        if str(cm.props.get("canonical_smiles") or cm.props.get("SMILES") or "").strip()
    )

    records: list[DatasetProfilerRecord] = []
    lipinski_pass_count = 0
    lipinski_fail_count = 0
    pains_match_count = 0

    for row_idx in range(1, len(data) + 1):
        missing_fields = _missing_fields_for_row(data, row_idx - 1)
        input_smiles = input_smiles_values[row_idx - 1] if row_idx - 1 < len(input_smiles_values) else ""
        name = name_values[row_idx - 1] if row_idx - 1 < len(name_values) else ""
        cm = valid_by_row.get(row_idx)

        if cm is None:
            error_message = invalid_messages.get(row_idx, "Could not parse structure.")
            issues.append(
                ServiceIssue(
                    code="invalid_structure",
                    message=error_message,
                    severity="warning",
                    row_index=row_idx,
                    molecule_id=name or None,
                    field=report.smiles_column,
                )
            )
            records.append(
                _build_invalid_record(
                    row_index=row_idx,
                    name=name,
                    input_smiles=input_smiles,
                    missing_fields=missing_fields,
                    error_message=error_message,
                )
            )
            continue

        canonical_smiles = str(cm.props.get("canonical_smiles") or cm.props.get("SMILES") or "").strip()
        descriptor_values = _nan_descriptor_values()
        issue_codes: list[str] = []
        issue_messages: list[str] = []
        lipinski_violations = float("nan")
        lipinski_pass = False
        pains_regid = ""
        pains_match = False

        try:
            lip_vio, mw, logp, hbd, hba = lipinski_stats(cm.mol, _BASE_FILTER_CFG)
            _veber_ok, rotb, tpsa = veber_stats(cm.mol, _BASE_FILTER_CFG)
            descriptor_values.update(
                {
                    "molecular_weight": float(mw),
                    "logp": float(logp),
                    "hbd": float(hbd),
                    "hba": float(hba),
                    "tpsa": float(tpsa),
                    "rotatable_bonds": float(rotb),
                }
            )
            lipinski_violations = float(lip_vio)
            lipinski_pass = bool(lip_vio <= 1)
        except (RuntimeError, ValueError) as exc:
            issues.append(
                ServiceIssue(
                    code="descriptor_computation_failed",
                    message=f"Descriptor calculation failed: {exc}",
                    severity="warning",
                    row_index=row_idx,
                    molecule_id=name or None,
                )
            )
            issue_codes.append("DESCRIPTOR_COMPUTATION_FAILED")
            issue_messages.append(f"Descriptor calculation failed: {exc}")

        if cfg.compute_pains:
            try:
                pains_flag, pains_regid, _atoms = pains_match_info(
                    cm.mol,
                    FilterConfig(
                        filter_rule="None",
                        selection_mode="Forward All Molecules",
                        compute_qed=False,
                        compute_pains=True,
                        highlight_pains_atoms=False,
                    ),
                )
                pains_match = bool(pains_flag)
            except (RuntimeError, ValueError) as exc:
                issues.append(
                    ServiceIssue(
                        code="pains_flagging_failed",
                        message=f"PAINS flagging failed: {exc}",
                        severity="warning",
                        row_index=row_idx,
                        molecule_id=name or None,
                    )
                )
                issue_codes.append("PAINS_FLAGGING_FAILED")
                issue_messages.append(f"PAINS flagging failed: {exc}")

        duplicate_group_size = int(canonical_counts.get(canonical_smiles, 0))
        if duplicate_group_size > 1:
            issues.append(
                ServiceIssue(
                    code="duplicate_smiles",
                    message=f"Canonical SMILES occurs {duplicate_group_size} times in the dataset.",
                    severity="warning",
                    row_index=row_idx,
                    molecule_id=name or None,
                    field="canonical_smiles",
                )
            )
            issue_codes.append("DUPLICATE_SMILES")
            issue_messages.append(
                f"Canonical SMILES occurs {duplicate_group_size} times in the dataset."
            )

        if math.isfinite(lipinski_violations):
            if lipinski_pass:
                lipinski_pass_count += 1
            else:
                lipinski_fail_count += 1
                issues.append(
                    ServiceIssue(
                        code="lipinski_rule_of_five_failed",
                        message=f"Lipinski rule-of-five failed with {int(lipinski_violations)} violation(s).",
                        severity="warning",
                        row_index=row_idx,
                        molecule_id=name or None,
                    )
                )
                issue_codes.append("LIPINSKI_RULE_OF_FIVE_FAILED")
                issue_messages.append(
                    f"Lipinski rule-of-five failed with {int(lipinski_violations)} violation(s)."
                )

        if pains_match:
            pains_match_count += 1
            issues.append(
                ServiceIssue(
                    code="pains_match",
                    message=f"Matched PAINS alert(s): {pains_regid or 'PAINS'}.",
                    severity="warning",
                    row_index=row_idx,
                    molecule_id=name or None,
                )
            )
            issue_codes.append("PAINS_MATCH")
            issue_messages.append(f"Matched PAINS alert(s): {pains_regid or 'PAINS'}.")

        records.append(
            _build_valid_record(
                row_index=row_idx,
                name=name or (cm.name or ""),
                input_smiles=input_smiles,
                canonical_smiles=canonical_smiles,
                missing_fields=missing_fields,
                duplicate_group_size=duplicate_group_size,
                lipinski_violations=lipinski_violations,
                lipinski_pass=lipinski_pass,
                pains_match=pains_match,
                pains_regid=pains_regid,
                descriptor_values=descriptor_values,
                issue_codes=issue_codes,
                issue_messages=issue_messages,
            )
        )

    valid_records = [record for record in records if record.valid_molecule]
    summary = DatasetProfilerSummary(
        n_rows=len(data),
        n_valid_molecules=len(valid_records),
        n_invalid_molecules=len(data) - len(valid_records),
        duplicate_smiles_count=sum(max(count - 1, 0) for count in canonical_counts.values()),
        duplicate_smiles_groups=sum(1 for count in canonical_counts.values() if count > 1),
        rows_with_missing_values=rows_with_missing_values,
        n_lipinski_pass=lipinski_pass_count,
        n_lipinski_fail=lipinski_fail_count,
        n_pains_matches=pains_match_count,
        missing_value_counts=missing_counts,
        issue_counts=count_issues(issues),
    )
    return DatasetProfilerResult(
        summary=summary,
        records=tuple(records),
        missing_value_summary=missing_stats,
        descriptor_summary=_build_descriptor_summary(valid_records),
        issues=tuple(issues),
    )


def profiled_records_as_dicts(records: Iterable[DatasetProfilerRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "row_index": record.row_index,
                "name": record.name,
                "input_smiles": record.input_smiles,
                "canonical_smiles": record.canonical_smiles,
                "status": record.status,
                "valid_molecule": int(record.valid_molecule),
                "duplicate_smiles": int(record.duplicate_smiles),
                "duplicate_group_size": record.duplicate_group_size,
                "missing_value_count": record.missing_value_count,
                "missing_fields": " | ".join(record.missing_fields),
                "lipinski_violations": record.lipinski_violations,
                "lipinski_pass": int(record.lipinski_pass),
                "pains_match": int(record.pains_match),
                "pains_regid": record.pains_regid,
                "molecular_weight": record.molecular_weight,
                "logp": record.logp,
                "hbd": record.hbd,
                "hba": record.hba,
                "tpsa": record.tpsa,
                "rotatable_bonds": record.rotatable_bonds,
                "issue_codes": ";".join(record.issue_codes),
                "issues": " | ".join(record.issues),
            }
        )
    return rows


def problematic_compounds_as_dicts(records: Iterable[DatasetProfilerRecord]) -> list[dict[str, Any]]:
    problematic = [
        record
        for record in records
        if (
            not record.valid_molecule
            or record.duplicate_smiles
            or record.missing_value_count > 0
            or bool(record.issue_codes)
        )
    ]
    return profiled_records_as_dicts(problematic)


def dataset_profile_summary_as_rows(summary: DatasetProfilerSummary) -> list[dict[str, Any]]:
    rows = [
        {
            "metric": "n_rows",
            "value": summary.n_rows,
            "description": "All input rows.",
        },
        {
            "metric": "n_valid_molecules",
            "value": summary.n_valid_molecules,
            "description": "Rows with a valid parsed molecule.",
        },
        {
            "metric": "n_invalid_molecules",
            "value": summary.n_invalid_molecules,
            "description": "Rows that could not be parsed as molecules.",
        },
        {
            "metric": "duplicate_smiles_count",
            "value": summary.duplicate_smiles_count,
            "description": "Duplicate valid molecules beyond the first occurrence.",
        },
        {
            "metric": "duplicate_smiles_groups",
            "value": summary.duplicate_smiles_groups,
            "description": "Distinct canonical-SMILES duplicate groups.",
        },
        {
            "metric": "rows_with_missing_values",
            "value": summary.rows_with_missing_values,
            "description": "Rows containing at least one missing value.",
        },
        {
            "metric": "n_lipinski_pass",
            "value": summary.n_lipinski_pass,
            "description": "Valid molecules passing Lipinski rule-of-five.",
        },
        {
            "metric": "n_lipinski_fail",
            "value": summary.n_lipinski_fail,
            "description": "Valid molecules failing Lipinski rule-of-five.",
        },
        {
            "metric": "n_pains_matches",
            "value": summary.n_pains_matches,
            "description": "Valid molecules matching at least one PAINS alert.",
        },
    ]
    for code, count in summary.issue_counts.items():
        rows.append(
            {
                "metric": f"issue_{code}",
                "value": count,
                "description": f"Rows/issues flagged as {code}.",
            }
        )
    for column, count in sorted(summary.missing_value_counts.items()):
        rows.append(
            {
                "metric": f"missing_{column}",
                "value": count,
                "description": f"Missing values in column {column}.",
            }
        )
    return rows


def descriptor_summary_as_rows(stats: Iterable[DescriptorSummaryStat]) -> list[dict[str, Any]]:
    return [
        {
            "descriptor": stat.descriptor,
            "count": stat.count,
            "mean": stat.mean,
            "minimum": stat.minimum,
            "maximum": stat.maximum,
        }
        for stat in stats
    ]


def missing_value_summary_as_rows(stats: Iterable[MissingValueStat]) -> list[dict[str, Any]]:
    return [
        {
            "column": stat.column,
            "missing_count": stat.missing_count,
            "missing_fraction": stat.missing_fraction,
        }
        for stat in stats
    ]


def profiled_records_table(result: DatasetProfilerResult) -> Table | None:
    return records_to_orange_table(
        profiled_records_as_dicts(result.records),
        meta_columns=[
            "name",
            "input_smiles",
            "canonical_smiles",
            "status",
            "missing_fields",
            "pains_regid",
            "issue_codes",
            "issues",
        ],
        name="Profiled Molecules",
    )


def problematic_compounds_table(result: DatasetProfilerResult) -> Table | None:
    return records_to_orange_table(
        problematic_compounds_as_dicts(result.records),
        meta_columns=[
            "name",
            "input_smiles",
            "canonical_smiles",
            "status",
            "missing_fields",
            "pains_regid",
            "issue_codes",
            "issues",
        ],
        name="Problematic Compounds",
    )


def summary_table(result: DatasetProfilerResult) -> Table | None:
    return records_to_orange_table(
        dataset_profile_summary_as_rows(result.summary),
        meta_columns=["metric", "description"],
        attribute_columns=["value"],
        name="Dataset Profile Summary",
    )


__all__ = [
    "DatasetProfilerConfig",
    "DatasetProfilerRecord",
    "DatasetProfilerResult",
    "DatasetProfilerSummary",
    "DescriptorSummaryStat",
    "MissingValueStat",
    "dataset_profile_summary_as_rows",
    "descriptor_summary_as_rows",
    "missing_value_summary_as_rows",
    "problematic_compounds_as_dicts",
    "problematic_compounds_table",
    "profiled_records_as_dicts",
    "profiled_records_table",
    "run_dataset_profiler",
    "summary_table",
]

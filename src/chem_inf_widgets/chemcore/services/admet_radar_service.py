from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from Orange.data import Table, Variable

from chem_inf_widgets.chemcore.result import ServiceIssue
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
from chem_inf_widgets.chemcore.services.orange_table_utils import records_to_orange_table

try:
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    from rdkit.Chem.QED import qed
except Exception:  # pragma: no cover
    Crippen = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    FilterCatalog = None  # type: ignore[assignment]
    FilterCatalogParams = None  # type: ignore[assignment]
    qed = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]


@dataclass(frozen=True)
class AdmetRadarConfig:
    compute_pains: bool = True
    compute_brenk: bool = True


@dataclass(frozen=True)
class AdmetRadarRecord:
    row_index: int
    name: str
    input_smiles: str
    canonical_smiles: str
    status: str
    valid_molecule: bool
    molecular_weight: float = float("nan")
    logp: float = float("nan")
    molar_refractivity: float = float("nan")
    hbd: float = float("nan")
    hba: float = float("nan")
    tpsa: float = float("nan")
    rotatable_bonds: float = float("nan")
    ring_count: float = float("nan")
    heavy_atom_count: float = float("nan")
    qed_score: float = float("nan")
    lipinski_violations: float = float("nan")
    lipinski_pass: bool = False
    veber_pass: bool = False
    ghose_pass: bool = False
    egan_pass: bool = False
    muegge_pass: bool = False
    pains_match: bool = False
    pains_regid: str = ""
    brenk_match: bool = False
    brenk_description: str = ""
    issue_codes: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdmetRadarSummary:
    n_rows: int
    n_valid_molecules: int
    n_invalid_molecules: int
    n_lipinski_pass: int
    n_veber_pass: int
    n_ghose_pass: int
    n_egan_pass: int
    n_muegge_pass: int
    n_pains_matches: int
    n_brenk_matches: int


@dataclass(frozen=True)
class AdmetRadarResult:
    summary: AdmetRadarSummary
    records: tuple[AdmetRadarRecord, ...]
    issues: tuple[ServiceIssue, ...] = ()


_FILTER_CFG = FilterConfig(
    filter_rule="Lipinski + Veber",
    selection_mode="Forward All Molecules",
    compute_qed=True,
    compute_pains=True,
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


def _invalid_row_messages(report: TableMolConversionReport) -> dict[int, str]:
    out: dict[int, str] = {}
    for message in report.errors:
        text = str(message or "").strip()
        if not text.startswith("Row "):
            continue
        prefix, _, tail = text.partition(":")
        try:
            row_index = int(prefix.replace("Row", "").strip())
        except ValueError:
            continue
        out[row_index] = tail.strip() or "Could not parse molecule."
    return out


@lru_cache(maxsize=1)
def _brenk_catalog():
    if FilterCatalog is None or FilterCatalogParams is None:
        return None
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog(params)


def _brenk_match(mol) -> tuple[bool, str]:
    catalog = _brenk_catalog()
    if catalog is None:
        return False, ""
    match = catalog.GetFirstMatch(mol)
    if match is None:
        return False, ""
    return True, str(match.GetDescription() or "Brenk")


def _ghose_pass(
    *,
    molecular_weight: float,
    logp: float,
    molar_refractivity: float,
    heavy_atom_count: float,
) -> bool:
    return (
        160.0 <= float(molecular_weight) <= 480.0
        and -0.4 <= float(logp) <= 5.6
        and 40.0 <= float(molar_refractivity) <= 130.0
        and 20.0 <= float(heavy_atom_count) <= 70.0
    )


def _egan_pass(*, logp: float, tpsa: float) -> bool:
    return float(logp) <= 5.88 and float(tpsa) <= 131.6


def _muegge_pass(
    *,
    molecular_weight: float,
    logp: float,
    tpsa: float,
    hbd: float,
    hba: float,
    rotatable_bonds: float,
    ring_count: float,
    heavy_atom_count: float,
) -> bool:
    return (
        200.0 <= float(molecular_weight) <= 600.0
        and -2.0 <= float(logp) <= 5.0
        and float(tpsa) <= 150.0
        and float(hbd) <= 5.0
        and float(hba) <= 10.0
        and float(rotatable_bonds) <= 15.0
        and 1.0 <= float(ring_count) <= 7.0
        and float(heavy_atom_count) >= 10.0
    )


def _build_invalid_record(
    *,
    row_index: int,
    name: str,
    input_smiles: str,
    message: str,
) -> AdmetRadarRecord:
    return AdmetRadarRecord(
        row_index=row_index,
        name=name,
        input_smiles=input_smiles,
        canonical_smiles="",
        status="Invalid",
        valid_molecule=False,
        issue_codes=("INVALID_MOLECULE",),
        issues=(message,),
    )


def _build_valid_record(
    *,
    row_index: int,
    name: str,
    input_smiles: str,
    canonical_smiles: str,
    molecular_weight: float,
    logp: float,
    molar_refractivity: float,
    hbd: float,
    hba: float,
    tpsa: float,
    rotatable_bonds: float,
    ring_count: float,
    heavy_atom_count: float,
    qed_score: float,
    lipinski_violations: int,
    lipinski_pass: bool,
    veber_pass: bool,
    ghose_pass: bool,
    egan_pass: bool,
    muegge_pass: bool,
    pains_match: bool,
    pains_regid: str,
    brenk_match: bool,
    brenk_description: str,
) -> AdmetRadarRecord:
    issue_codes: list[str] = []
    issue_messages: list[str] = []
    if pains_match:
        issue_codes.append("PAINS_MATCH")
        issue_messages.append(f"Matched PAINS alert(s): {pains_regid or 'PAINS'}.")
    if brenk_match:
        issue_codes.append("BRENK_MATCH")
        issue_messages.append(f"Matched Brenk alert: {brenk_description}.")
    return AdmetRadarRecord(
        row_index=row_index,
        name=name,
        input_smiles=input_smiles,
        canonical_smiles=canonical_smiles,
        status="Valid",
        valid_molecule=True,
        molecular_weight=float(molecular_weight),
        logp=float(logp),
        molar_refractivity=float(molar_refractivity),
        hbd=float(hbd),
        hba=float(hba),
        tpsa=float(tpsa),
        rotatable_bonds=float(rotatable_bonds),
        ring_count=float(ring_count),
        heavy_atom_count=float(heavy_atom_count),
        qed_score=float(qed_score),
        lipinski_violations=float(lipinski_violations),
        lipinski_pass=bool(lipinski_pass),
        veber_pass=bool(veber_pass),
        ghose_pass=bool(ghose_pass),
        egan_pass=bool(egan_pass),
        muegge_pass=bool(muegge_pass),
        pains_match=bool(pains_match),
        pains_regid=str(pains_regid or ""),
        brenk_match=bool(brenk_match),
        brenk_description=str(brenk_description or ""),
        issue_codes=tuple(issue_codes),
        issues=tuple(issue_messages),
    )


def _empty_result(
    *,
    n_rows: int,
    issues: list[ServiceIssue],
) -> AdmetRadarResult:
    return AdmetRadarResult(
        summary=AdmetRadarSummary(n_rows, 0, n_rows, 0, 0, 0, 0, 0, 0, 0),
        records=(),
        issues=tuple(issues),
    )


def run_admet_radar(
    data: Table | None,
    config: AdmetRadarConfig | None = None,
) -> AdmetRadarResult:
    cfg = config or AdmetRadarConfig()
    issues: list[ServiceIssue] = []
    if data is None or len(data) == 0:
        issues.append(
            ServiceIssue(
                code="no_input_data",
                message="No input data for ADMET radar analysis.",
                severity="error",
            )
        )
        return _empty_result(n_rows=0, issues=issues)

    if any(obj is None for obj in (Descriptors, Crippen, rdMolDescriptors, qed)):
        issues.append(
            ServiceIssue(
                code="rdkit_unavailable",
                message="RDKit descriptor functionality is unavailable.",
                severity="error",
            )
        )
        return _empty_result(n_rows=len(data), issues=issues)

    try:
        mols, report = table_to_chemmols_with_report(data)
    except (ImportError, ValueError) as exc:
        issues.append(
            ServiceIssue(
                code="table_to_molecule_conversion_failed",
                message=str(exc),
                severity="error",
            )
        )
        return _empty_result(n_rows=len(data), issues=issues)

    name_values = _column_strings(data, report.name_column)
    smiles_values = _column_strings(data, report.smiles_column)
    invalid_messages = _invalid_row_messages(report)
    mol_by_row = {
        int(chem_mol.get_prop("source_row_index", chem_mol.get_prop("row_index", index + 1))): chem_mol
        for index, chem_mol in enumerate(mols)
    }

    if cfg.compute_brenk and _brenk_catalog() is None:
        issues.append(
            ServiceIssue(
                code="brenk_unavailable",
                message="Brenk alerts are unavailable in this RDKit build.",
                severity="warning",
            )
        )

    records: list[AdmetRadarRecord] = []
    for row_index in range(1, len(data) + 1):
        name = name_values[row_index - 1] if row_index - 1 < len(name_values) else ""
        input_smiles = smiles_values[row_index - 1] if row_index - 1 < len(smiles_values) else ""
        chem_mol = mol_by_row.get(row_index)
        if chem_mol is None:
            message = invalid_messages.get(row_index, "Could not parse molecule.")
            issues.append(
                ServiceIssue(
                    code="invalid_molecule",
                    message=message,
                    severity="warning",
                    row_index=row_index,
                )
            )
            records.append(
                _build_invalid_record(
                    row_index=row_index,
                    name=name,
                    input_smiles=input_smiles,
                    message=message,
                )
            )
            continue

        mol = chem_mol.to_rdkit()
        if mol is None:
            issues.append(
                ServiceIssue(
                    code="rdkit_conversion_failed",
                    message="Could not convert molecule to RDKit Mol.",
                    severity="warning",
                    row_index=row_index,
                )
            )
            records.append(
                _build_invalid_record(
                    row_index=row_index,
                    name=name,
                    input_smiles=input_smiles,
                    message="Could not convert molecule to RDKit Mol.",
                )
            )
            continue

        lipinski_violations, molecular_weight, logp, hbd, hba = lipinski_stats(mol, _FILTER_CFG)
        veber_ok, rotatable_bonds, tpsa = veber_stats(mol, _FILTER_CFG)
        molar_refractivity = float(Crippen.MolMR(mol))
        ring_count = float(rdMolDescriptors.CalcNumRings(mol))
        heavy_atom_count = float(mol.GetNumHeavyAtoms())
        qed_score = float(qed(mol))

        pains_flag = False
        pains_regid = ""
        if cfg.compute_pains:
            pains_value, pains_regid, _pains_atoms = pains_match_info(mol, _FILTER_CFG)
            pains_flag = bool(pains_value)

        brenk_flag = False
        brenk_description = ""
        if cfg.compute_brenk:
            brenk_flag, brenk_description = _brenk_match(mol)

        lipinski_pass = int(lipinski_violations) <= 1
        ghose_ok = _ghose_pass(
            molecular_weight=molecular_weight,
            logp=logp,
            molar_refractivity=molar_refractivity,
            heavy_atom_count=heavy_atom_count,
        )
        egan_ok = _egan_pass(logp=logp, tpsa=tpsa)
        muegge_ok = _muegge_pass(
            molecular_weight=molecular_weight,
            logp=logp,
            tpsa=tpsa,
            hbd=hbd,
            hba=hba,
            rotatable_bonds=rotatable_bonds,
            ring_count=ring_count,
            heavy_atom_count=heavy_atom_count,
        )

        records.append(
            _build_valid_record(
                row_index=row_index,
                name=name,
                input_smiles=input_smiles,
                canonical_smiles=chem_mol.canonical_smiles(remove_hs=False),
                molecular_weight=molecular_weight,
                logp=logp,
                molar_refractivity=molar_refractivity,
                hbd=hbd,
                hba=hba,
                tpsa=tpsa,
                rotatable_bonds=rotatable_bonds,
                ring_count=ring_count,
                heavy_atom_count=heavy_atom_count,
                qed_score=qed_score,
                lipinski_violations=lipinski_violations,
                lipinski_pass=lipinski_pass,
                veber_pass=veber_ok,
                ghose_pass=ghose_ok,
                egan_pass=egan_ok,
                muegge_pass=muegge_ok,
                pains_match=pains_flag,
                pains_regid=pains_regid,
                brenk_match=brenk_flag,
                brenk_description=brenk_description,
            )
        )

    summary = AdmetRadarSummary(
        n_rows=len(data),
        n_valid_molecules=sum(1 for record in records if record.valid_molecule),
        n_invalid_molecules=sum(1 for record in records if not record.valid_molecule),
        n_lipinski_pass=sum(1 for record in records if record.lipinski_pass),
        n_veber_pass=sum(1 for record in records if record.veber_pass),
        n_ghose_pass=sum(1 for record in records if record.ghose_pass),
        n_egan_pass=sum(1 for record in records if record.egan_pass),
        n_muegge_pass=sum(1 for record in records if record.muegge_pass),
        n_pains_matches=sum(1 for record in records if record.pains_match),
        n_brenk_matches=sum(1 for record in records if record.brenk_match),
    )
    return AdmetRadarResult(
        summary=summary,
        records=tuple(records),
        issues=tuple(issues),
    )


def admet_radar_records_table(result: AdmetRadarResult) -> Table | None:
    rows = [
        {
            "row_index": record.row_index,
            "name": record.name,
            "input_smiles": record.input_smiles,
            "canonical_smiles": record.canonical_smiles,
            "status": record.status,
            "molecular_weight": record.molecular_weight,
            "logp": record.logp,
            "molar_refractivity": record.molar_refractivity,
            "hbd": record.hbd,
            "hba": record.hba,
            "tpsa": record.tpsa,
            "rotatable_bonds": record.rotatable_bonds,
            "ring_count": record.ring_count,
            "heavy_atom_count": record.heavy_atom_count,
            "qed_score": record.qed_score,
            "lipinski_violations": record.lipinski_violations,
            "lipinski_pass": float(record.lipinski_pass),
            "veber_pass": float(record.veber_pass),
            "ghose_pass": float(record.ghose_pass),
            "egan_pass": float(record.egan_pass),
            "muegge_pass": float(record.muegge_pass),
            "pains_match": float(record.pains_match),
            "pains_regid": record.pains_regid,
            "brenk_match": float(record.brenk_match),
            "brenk_description": record.brenk_description,
            "issues": " | ".join(record.issues),
        }
        for record in result.records
    ]
    return records_to_orange_table(
        rows,
        meta_columns=[
            "row_index",
            "name",
            "input_smiles",
            "canonical_smiles",
            "status",
            "pains_regid",
            "brenk_description",
            "issues",
        ],
        name="ADMET Radar Records",
    )


def admet_flagged_records_as_dicts(result: AdmetRadarResult) -> list[dict[str, object]]:
    return [
        {
            "row_index": record.row_index,
            "name": record.name,
            "input_smiles": record.input_smiles,
            "canonical_smiles": record.canonical_smiles,
            "status": record.status,
            "lipinski_pass": float(record.lipinski_pass),
            "veber_pass": float(record.veber_pass),
            "ghose_pass": float(record.ghose_pass),
            "egan_pass": float(record.egan_pass),
            "muegge_pass": float(record.muegge_pass),
            "pains_match": float(record.pains_match),
            "pains_regid": record.pains_regid,
            "brenk_match": float(record.brenk_match),
            "brenk_description": record.brenk_description,
            "issues": " | ".join(record.issues),
        }
        for record in result.records
        if (
            not record.valid_molecule
            or not record.lipinski_pass
            or not record.veber_pass
            or not record.ghose_pass
            or not record.egan_pass
            or not record.muegge_pass
            or record.pains_match
            or record.brenk_match
            or bool(record.issue_codes)
        )
    ]


def admet_flagged_table(result: AdmetRadarResult) -> Table | None:
    return records_to_orange_table(
        admet_flagged_records_as_dicts(result),
        meta_columns=[
            "row_index",
            "name",
            "input_smiles",
            "canonical_smiles",
            "status",
            "pains_regid",
            "brenk_description",
            "issues",
        ],
        name="ADMET Radar Flagged Compounds",
    )


def admet_summary_as_rows(result: AdmetRadarResult) -> list[dict[str, object]]:
    summary = result.summary
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
            "metric": "n_lipinski_pass",
            "value": summary.n_lipinski_pass,
            "description": "Valid molecules passing Lipinski rule-of-five.",
        },
        {
            "metric": "n_veber_pass",
            "value": summary.n_veber_pass,
            "description": "Valid molecules passing Veber rule.",
        },
        {
            "metric": "n_ghose_pass",
            "value": summary.n_ghose_pass,
            "description": "Valid molecules passing Ghose filter.",
        },
        {
            "metric": "n_egan_pass",
            "value": summary.n_egan_pass,
            "description": "Valid molecules passing Egan filter.",
        },
        {
            "metric": "n_muegge_pass",
            "value": summary.n_muegge_pass,
            "description": "Valid molecules passing Muegge filter.",
        },
        {
            "metric": "n_pains_matches",
            "value": summary.n_pains_matches,
            "description": "Valid molecules matching at least one PAINS alert.",
        },
        {
            "metric": "n_brenk_matches",
            "value": summary.n_brenk_matches,
            "description": "Valid molecules matching at least one Brenk alert.",
        },
    ]
    for issue in result.issues:
        rows.append(
            {
                "metric": f"issue_{issue.code}",
                "value": 1,
                "description": issue.message,
            }
        )
    return rows


def admet_radar_summary_table(result: AdmetRadarResult) -> Table | None:
    return records_to_orange_table(
        admet_summary_as_rows(result),
        meta_columns=["metric", "description"],
        name="ADMET Radar Summary",
    )


__all__ = [
    "AdmetRadarConfig",
    "AdmetRadarRecord",
    "AdmetRadarResult",
    "AdmetRadarSummary",
    "admet_flagged_records_as_dicts",
    "admet_flagged_table",
    "admet_radar_records_table",
    "admet_radar_summary_table",
    "admet_summary_as_rows",
    "run_admet_radar",
]

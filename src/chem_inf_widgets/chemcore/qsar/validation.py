from __future__ import annotations

from Orange.data import Table

from chem_inf_widgets.chemcore.services.orange_table_utils import records_to_orange_table


def build_qsar_modeling_summary_table(result: dict) -> Table | None:
    """Build a compact modeling audit table for downstream inspection."""
    if not result:
        return None
    cleanup = dict(result.get("descriptor_cleanup") or {})
    records = [
        {
            "section": "dataset",
            "metric": "target_column",
            "value": result.get("target_column", ""),
            "numeric_value": "",
        },
        {
            "section": "dataset",
            "metric": "usable_rows",
            "value": str(result.get("usable_row_count", "")),
            "numeric_value": result.get("usable_row_count", ""),
        },
        {
            "section": "dataset",
            "metric": "removed_rows",
            "value": str(result.get("removed_row_count", "")),
            "numeric_value": result.get("removed_row_count", ""),
        },
        {
            "section": "descriptors",
            "metric": "input_descriptor_count",
            "value": str(cleanup.get("input_descriptor_count", "")),
            "numeric_value": cleanup.get("input_descriptor_count", ""),
        },
        {
            "section": "descriptors",
            "metric": "descriptor_count_used",
            "value": str(cleanup.get("descriptor_count", len(result.get("feature_names", [])))),
            "numeric_value": cleanup.get("descriptor_count", len(result.get("feature_names", []))),
        },
        {
            "section": "descriptors",
            "metric": "removed_all_missing_count",
            "value": str(cleanup.get("removed_all_missing_count", 0)),
            "numeric_value": cleanup.get("removed_all_missing_count", 0),
        },
        {
            "section": "descriptors",
            "metric": "removed_constant_count",
            "value": str(cleanup.get("removed_constant_count", 0)),
            "numeric_value": cleanup.get("removed_constant_count", 0),
        },
        {
            "section": "descriptors",
            "metric": "qsar_cap_limit",
            "value": str(cleanup.get("qsar_cap_limit", "")),
            "numeric_value": cleanup.get("qsar_cap_limit", ""),
        },
        {
            "section": "descriptors",
            "metric": "removed_qsar_cap_count",
            "value": str(cleanup.get("removed_qsar_cap_count", 0)),
            "numeric_value": cleanup.get("removed_qsar_cap_count", 0),
        },
        {
            "section": "model",
            "metric": "cv_score",
            "value": str(result.get("cv_score", "")),
            "numeric_value": result.get("cv_score", ""),
        },
    ]
    if cleanup.get("removed_all_missing"):
        records.append(
            {
                "section": "descriptors",
                "metric": "removed_all_missing_examples",
                "value": ", ".join(cleanup.get("removed_all_missing", [])),
                "numeric_value": "",
            }
        )
    if cleanup.get("removed_constant"):
        records.append(
            {
                "section": "descriptors",
                "metric": "removed_constant_examples",
                "value": ", ".join(cleanup.get("removed_constant", [])),
                "numeric_value": "",
            }
        )
    if cleanup.get("removed_qsar_cap"):
        records.append(
            {
                "section": "descriptors",
                "metric": "removed_qsar_cap_examples",
                "value": ", ".join(cleanup.get("removed_qsar_cap", [])),
                "numeric_value": "",
            }
        )
    return records_to_orange_table(
        records,
        attribute_columns=["numeric_value"],
        meta_columns=["section", "metric", "value"],
        name="QSAR Modeling Summary",
    )


__all__ = ["build_qsar_modeling_summary_table"]

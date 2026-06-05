from __future__ import annotations

import pandas as pd

from chem_inf_widgets.chemcore.services.qsar_validation_dashboard_service import (
    validate_qsar_predictions,
)


def test_validation_dashboard_resolves_common_qsar_column_aliases_and_ad_confidence():
    df = pd.DataFrame(
        {
            "molecule_id": ["C001", "C002", "C003"],
            "dataset": ["train", "test", "test"],
            "actual_value": [5.0, 6.0, 7.0],
            "predicted_pActivity": [5.1, 6.8, 6.9],
            "ad_in_domain": [1, 0, 1],
            "ad_confidence": ["high", "low", "medium"],
        }
    )

    result = validate_qsar_predictions(df)

    assert result.summary["observed_column_used"] == "actual_value"
    assert result.summary["predicted_column_used"] == "predicted_pActivity"
    assert result.summary["split_column_used"] == "dataset"
    assert result.summary["id_column_used"] == "molecule_id"
    assert result.summary["n_outside_ad"] == 1
    assert result.summary["n_low_ad_confidence"] == 1
    assert "validation_severity" in result.diagnostics.columns
    assert "signed_error" in result.diagnostics.columns
    assert result.outliers["review_reason"].str.contains("low AD confidence").any()

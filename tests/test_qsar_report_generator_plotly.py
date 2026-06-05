import pandas as pd
import pytest

from chem_inf_widgets.chemcore.services.qsar_report_generator_service import generate_qsar_report

pytest.importorskip("plotly")


def test_qsar_report_contains_interactive_plotly_graphs():
    predictions = pd.DataFrame(
        {
            "compound_id": ["cmp-1", "cmp-2", "cmp-3", "cmp-4"],
            "observed": [1.0, 2.0, 3.0, 4.0],
            "predicted": [1.1, 1.8, 3.2, 3.7],
            "split": ["train", "train", "test", "test"],
        }
    )
    metrics = pd.DataFrame(
        {
            "split": ["train", "test", "test"],
            "metric": ["r2", "r2", "rmse"],
            "value": [0.91, 0.74, 0.42],
        }
    )
    ad_summary = pd.DataFrame(
        {
            "compound_id": ["cmp-1", "cmp-2", "cmp-3", "cmp-4"],
            "ad_leverage_ratio": [0.4, 0.7, 1.2, 0.9],
            "ad_distance_ratio": [0.5, 0.8, 1.4, 0.95],
            "ad_confidence": ["high", "high", "low", "medium"],
            "ad_in_domain": [True, True, False, True],
        }
    )
    explanation = pd.DataFrame(
        {
            "feature": ["MolWt", "LogP", "TPSA"],
            "importance": [0.4, -0.2, 0.7],
        }
    )

    result = generate_qsar_report(
        predictions=predictions,
        metrics=metrics,
        ad_summary=ad_summary,
        explanation_summary=explanation,
    )

    assert "Plotly.newPlot" in result.html
    assert "Interactive QSAR visual analytics" in result.html
    assert "observed_predicted" in result.summary["interactive_graphs"]
    assert "ad_boundary_map" in result.summary["interactive_graphs"]
    assert "feature_importance" in result.summary["interactive_graphs"]

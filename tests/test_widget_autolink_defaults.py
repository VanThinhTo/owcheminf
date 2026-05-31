from __future__ import annotations

import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")
pytest.importorskip("rdkit")

from orangecanvas.scheme.node import SchemeNode
from orangecanvas.scheme.scheme import Scheme

from chem_inf_widgets.widgets import _widget_desc_from_local_module
from chem_inf_widgets.widgets import ow_activity_cliff_finder
from chem_inf_widgets.widgets import ow_applicability_domain
from chem_inf_widgets.widgets import ow_descriptor_filter
from chem_inf_widgets.widgets import ow_matched_molecular_pairs
from chem_inf_widgets.widgets import ow_mol_standardizer
from chem_inf_widgets.widgets import ow_molecule_import_hub
from chem_inf_widgets.widgets import ow_molecule_qc_dashboard
from chem_inf_widgets.widgets import ow_pair_viewer
from chem_inf_widgets.widgets import ow_qsar_dataset_builder
from chem_inf_widgets.widgets import ow_qsar_model_hub
from chem_inf_widgets.widgets import ow_reaction_enumerator


def _proposed_links(source_module, sink_module):
    scheme = Scheme()
    source = SchemeNode(_widget_desc_from_local_module(source_module))
    sink = SchemeNode(_widget_desc_from_local_module(sink_module))
    return scheme.propose_links(source, sink)


@pytest.mark.parametrize(
    ("source_module", "sink_module", "expected_output", "expected_input"),
    [
        (ow_molecule_import_hub, ow_molecule_qc_dashboard, "Data", "Data"),
        (ow_molecule_qc_dashboard, ow_mol_standardizer, "Clean Data", "Data"),
        (ow_mol_standardizer, ow_pair_viewer, "Data", "Data"),
        (ow_qsar_dataset_builder, ow_qsar_model_hub, "QSAR Ready Data", "Data"),
        (ow_descriptor_filter, ow_qsar_model_hub, "Modeling Data", "Data"),
        (ow_qsar_model_hub, ow_applicability_domain, "Predictions", "Data"),
        (ow_activity_cliff_finder, ow_pair_viewer, "Cliff Pairs", "Data"),
        (ow_matched_molecular_pairs, ow_pair_viewer, "Pair Table", "Data"),
        (ow_reaction_enumerator, ow_mol_standardizer, "Products", "Data"),
    ],
)
def test_autolink_prefers_primary_workflow_signals(
    source_module,
    sink_module,
    expected_output: str,
    expected_input: str,
) -> None:
    links = _proposed_links(source_module, sink_module)

    assert links, "Expected at least one compatible connection"
    best_output, best_input, best_weight = links[0]

    assert best_output.name == expected_output
    assert best_input.name == expected_input
    assert best_weight > 0

    if len(links) > 1:
        assert best_weight >= links[1][2]

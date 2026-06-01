from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.result import ServiceIssue
from chem_inf_widgets.chemcore.services.rdkit_descriptor_service import RdkitDescriptorComputeResult
from chem_inf_widgets.widgets import ow_rdkit_descriptors as rdkit_widget_module
from chem_inf_widgets.widgets.ow_rdkit_descriptors import OWRdkitDescriptors


_APP = QApplication.instance() or QApplication([])


def _noop(_value=None):
    return None


def test_rdkit_descriptors_widget_surfaces_backend_service_warning():
    widget = OWRdkitDescriptors()
    widget.set_molecules([ChemMol.from_smiles("CCO", name="ethanol")])
    warnings: list[str] = []

    with ExitStack() as stack:
        stack.enter_context(patch.object(widget.Outputs.data, "send", _noop))
        stack.enter_context(patch.object(widget.Outputs.molecules, "send", _noop))
        stack.enter_context(patch.object(widget, "_read_selected_names", lambda: ["MolWt"]))
        stack.enter_context(
            patch.object(
                widget._service,
                "compute_with_issues",
                return_value=RdkitDescriptorComputeResult(
                    frame=pd.DataFrame({"MolWt": [46.07]}),
                    issues=[
                        ServiceIssue(
                            code="rdkit_descriptor_computation_failed",
                            message="RDKit descriptor 'MolWt' failed: mw boom",
                            severity="warning",
                            row_index=1,
                        )
                    ],
                ),
            )
        )
        stack.enter_context(
            patch.object(rdkit_widget_module, "set_widget_warning", lambda _w, message: warnings.append(message or ""))
        )
        widget.commit()
        _APP.processEvents()

    assert warnings
    assert warnings[-1] == "1 descriptor warning(s). Row 1. RDKit descriptor 'MolWt' failed: mw boom"

    widget.onDeleteWidget()
    widget.close()


def test_rdkit_descriptors_widget_clears_warning_when_service_is_clean():
    widget = OWRdkitDescriptors()
    widget.set_molecules([ChemMol.from_smiles("CCO", name="ethanol")])
    warnings: list[str] = []

    with ExitStack() as stack:
        stack.enter_context(patch.object(widget.Outputs.data, "send", _noop))
        stack.enter_context(patch.object(widget.Outputs.molecules, "send", _noop))
        stack.enter_context(patch.object(widget, "_read_selected_names", lambda: ["MolWt"]))
        stack.enter_context(
            patch.object(
                widget._service,
                "compute_with_issues",
                return_value=RdkitDescriptorComputeResult(
                    frame=pd.DataFrame({"MolWt": [46.07]}),
                    issues=[],
                ),
            )
        )
        stack.enter_context(
            patch.object(rdkit_widget_module, "set_widget_warning", lambda _w, message: warnings.append(message or ""))
        )
        widget.commit()
        _APP.processEvents()

    assert warnings
    assert warnings[-1] == ""

    widget.onDeleteWidget()
    widget.close()

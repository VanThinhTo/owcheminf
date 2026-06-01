from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication

from chem_inf_widgets.chemcore.descriptors.fingerprints import FingerprintResult
from chem_inf_widgets.widgets import ow_fingerprint_generator as fp_widget_module
from chem_inf_widgets.widgets.ow_fingerprint_generator import OWFingerprintGenerator


_APP = QApplication.instance() or QApplication([])


def test_fingerprint_generator_warns_with_first_failure_reason():
    widget = OWFingerprintGenerator()
    worker = object()
    widget._worker = worker

    result = FingerprintResult(
        X=np.zeros((1, 8), dtype=np.float32),
        smiles=["CCO"],
        valid_indices=[0],
        failed_indices=[1],
        bit_names=[f"morgan_{i:04d}" for i in range(8)],
        fp_type="morgan",
        errors=["Fingerprint computation failed: demo backend failure"],
    )

    warnings: list[str] = []
    with (
        patch.object(widget, "progressBarFinished", lambda: None),
        patch.object(widget, "_update_buttons", lambda: None),
        patch.object(widget, "_restart_if_pending", lambda: None),
        patch.object(widget.Outputs.fingerprints, "send", lambda value: None),
        patch.object(widget.Outputs.molecules, "send", lambda value: None),
        patch.object(fp_widget_module, "set_widget_warning", lambda _w, message: warnings.append(message or "")),
    ):
        widget._on_finished(
            worker,
            (result, ["CCO"]),
            source_table=None,
            source_molecules=None,
            spec=fp_widget_module._FPJobSpec(
                fp_type="morgan",
                bit_size=8,
                radius=2,
                sanitize=True,
            ),
        )
        _APP.processEvents()

    assert warnings
    assert warnings[-1] == "Fingerprint computation failed: demo backend failure"

    widget.onDeleteWidget()
    widget.close()

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication
from Orange.data import Domain, StringVariable, Table

from chem_inf_widgets.widgets.ow_diversity_picker import OWDiversityPicker

_APP = QApplication.instance() or QApplication([])


def _demo_table() -> Table:
    smiles_var = StringVariable("SMILES")
    smiles_var.attributes["format"] = "SMILES"
    name_var = StringVariable("Name")
    domain = Domain([], metas=[name_var, smiles_var])
    metas = np.asarray(
        [
            ["ethanol", "CCO"],
            ["propanol", "CCCO"],
            ["benzene", "c1ccccc1"],
            ["pyridine", "c1ccncc1"],
            ["acetic acid", "CC(=O)O"],
            ["bad", "not_a_smiles"],
        ],
        dtype=object,
    )
    table = Table.from_numpy(domain, X=np.empty((len(metas), 0), dtype=float), metas=metas)
    table.name = "Diversity demo"
    return table


def test_diversity_picker_outputs_selected_subset_and_annotated_full_table():
    widget = OWDiversityPicker()
    sent: list[tuple[str, object]] = []
    table = _demo_table()
    widget.auto_run = False
    widget.n_select = 2
    widget.n_select_spin.setValue(2)
    widget.method_idx = 0
    widget.method_combo.setCurrentIndex(0)

    with ExitStack() as stack:
        stack.enter_context(patch.object(widget.Outputs.selected_data, "send", lambda value: sent.append(("selected", value))))
        stack.enter_context(patch.object(widget.Outputs.annotated_data, "send", lambda value: sent.append(("annotated", value))))
        stack.enter_context(patch.object(widget.Outputs.remainder_data, "send", lambda value: sent.append(("remainder", value))))
        stack.enter_context(patch.object(widget.Outputs.inspected_data, "send", lambda value: sent.append(("inspected", value))))
        stack.enter_context(patch.object(widget.Outputs.selected_molecules, "send", lambda value: sent.append(("selected_molecules", value))))
        stack.enter_context(patch.object(widget.Outputs.remainder_molecules, "send", lambda value: sent.append(("remainder_molecules", value))))
        widget.set_data(table)
        widget.commit()
        _APP.processEvents()

    selected = next(value for name, value in sent if name == "selected")
    annotated = next(value for name, value in sent if name == "annotated")
    remainder = next(value for name, value in sent if name == "remainder")

    assert selected is not None
    assert annotated is not None
    assert remainder is not None
    assert len(selected) == 2
    assert len(annotated) == len(table)
    assert len(remainder) == len(table) - 2
    attr_names = [var.name for var in annotated.domain.attributes]
    assert "chem_space_x" in attr_names
    assert "chem_space_y" in attr_names
    assert "diversity_selected" in attr_names
    assert "diversity_rank" in attr_names
    selected_flag = annotated.get_column("diversity_selected")
    assert float(np.nansum(selected_flag)) == 2.0
    rank_values = annotated.get_column("diversity_rank")
    finite_ranks = sorted(int(value) for value in rank_values if np.isfinite(value))
    assert finite_ranks == [1, 2]
    assert widget._all_points_item is not None
    assert widget._selected_points_item is not None
    assert widget._inspection_points_item is not None
    assert "PCA diversity" in widget.status_label.text()

    widget.onDeleteWidget()
    widget.close()


def test_diversity_picker_plot_inspection_sends_inspected_subset():
    widget = OWDiversityPicker()
    sent: list[tuple[str, object]] = []
    table = _demo_table()
    widget.auto_run = False
    widget.n_select = 2
    widget.n_select_spin.setValue(2)

    with ExitStack() as stack:
        stack.enter_context(patch.object(widget.Outputs.selected_data, "send", lambda value: sent.append(("selected", value))))
        stack.enter_context(patch.object(widget.Outputs.annotated_data, "send", lambda value: sent.append(("annotated", value))))
        stack.enter_context(patch.object(widget.Outputs.remainder_data, "send", lambda value: sent.append(("remainder", value))))
        stack.enter_context(patch.object(widget.Outputs.inspected_data, "send", lambda value: sent.append(("inspected", value))))
        stack.enter_context(patch.object(widget.Outputs.selected_molecules, "send", lambda value: sent.append(("selected_molecules", value))))
        stack.enter_context(patch.object(widget.Outputs.remainder_molecules, "send", lambda value: sent.append(("remainder_molecules", value))))
        widget.set_data(table)
        widget.commit()
        widget._publish_inspection([0, 2])
        _APP.processEvents()

    inspected_payloads = [value for name, value in sent if name == "inspected"]
    inspected = inspected_payloads[-1]
    assert inspected is not None
    assert len(inspected) == 2
    assert widget._inspection_list.count() == 2
    assert widget._inspection_list.currentItem() is not None
    assert "ethanol" in widget._inspection_browser.toPlainText().lower()
    assert widget._inspected_indices == [0, 2]

    widget.onDeleteWidget()
    widget.close()


def test_diversity_picker_hover_text_describes_compound():
    widget = OWDiversityPicker()
    table = _demo_table()
    widget.auto_run = False
    widget.n_select = 2
    widget.n_select_spin.setValue(2)

    widget.set_data(table)
    widget.commit()
    _APP.processEvents()

    text = widget._hover_text(0)
    assert "ethanol" in text.lower()
    assert "picked" in text.lower() or "not picked" in text.lower()
    assert "(" in text and ")" in text

    widget.onDeleteWidget()
    widget.close()

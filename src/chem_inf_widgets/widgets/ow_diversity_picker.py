from __future__ import annotations

from html import escape

import numpy as np
import pyqtgraph as pg
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.diversity_service import (
    DiversitySelectionResult,
    select_diverse_subset,
)
from chem_inf_widgets.chemcore.services.orange_table_utils import safe_table_from_numpy
from chem_inf_widgets.widgets.ui_helpers import (
    clear_widget_messages,
    format_done_status,
    format_error_status,
    format_no_input_status,
    set_widget_error,
)


def _find_smiles_vars(data: Table) -> list[StringVariable]:
    wanted = {"smiles", "canonical_smiles", "smile"}
    variables = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)

    preferred = [var for var in variables if isinstance(var, StringVariable) and var.name.strip().lower() in wanted]
    if preferred:
        return preferred + [var for var in variables if isinstance(var, StringVariable) and var not in preferred]
    return [var for var in variables if isinstance(var, StringVariable)]


def _table_smiles(data: Table, var_name: str) -> list[str]:
    variables = _find_smiles_vars(data)
    selected_var = next((var for var in variables if var.name == var_name), None)
    if selected_var is None:
        raise ValueError("No SMILES column selected.")

    col = data.get_column(selected_var)
    return ["" if value is None else str(value).strip() for value in col]


def _molecule_smiles(molecules: list[ChemMol]) -> list[str]:
    smiles = []
    for molecule in molecules:
        value = molecule.get_prop("SMILES") or molecule.get_prop("smiles")
        if isinstance(value, str) and value.strip():
            smiles.append(value.strip())
            continue
        try:
            smiles.append(molecule.canonical_smiles())
        except Exception:
            smiles.append("")
    return smiles


def _styled_plot(title: str = "") -> pg.PlotWidget:
    plot_widget = pg.PlotWidget(title=title)
    plot_widget.setBackground("#FFFFFF")
    plot_widget.getPlotItem().getAxis("left").setPen(pg.mkPen("#CBD5E1"))
    plot_widget.getPlotItem().getAxis("bottom").setPen(pg.mkPen("#CBD5E1"))
    plot_widget.getPlotItem().getAxis("left").setTextPen(pg.mkPen("#475569"))
    plot_widget.getPlotItem().getAxis("bottom").setTextPen(pg.mkPen("#475569"))
    plot_widget.showGrid(x=True, y=True, alpha=0.18)
    plot_widget.setMenuEnabled(False)
    return plot_widget


def _set_plot_range(plot: pg.PlotWidget, x_values: np.ndarray, y_values: np.ndarray) -> None:
    if x_values.size == 0 or y_values.size == 0:
        return
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_pad = max(x_span * 0.08, 0.15 if x_span == 0 else 0.0)
    y_pad = max(y_span * 0.08, 0.15 if y_span == 0 else 0.0)
    plot.setXRange(x_min - x_pad, x_max + x_pad, padding=0.0)
    plot.setYRange(y_min - y_pad, y_max + y_pad, padding=0.0)


def _count_card(label: str, value: str) -> str:
    return (
        "<div style='display:inline-block; min-width:160px; margin:6px; padding:10px 12px; "
        "border:1px solid #D0D5DD; border-radius:10px; background:#F8FAFC;'>"
        f"<div style='font-size:12px; color:#475467;'>{escape(label)}</div>"
        f"<div style='font-size:22px; font-weight:600; color:#101828;'>{escape(value)}</div>"
        "</div>"
    )


def _summary_html(result: DiversitySelectionResult) -> str:
    cards = "".join(
        [
            _count_card("Fingerprint", "Morgan r=2, 2048 bits"),
            _count_card("Projection", "PCA"),
            _count_card("Picker", "MaxMinPicker" if result.method == "maxmin" else result.method.replace("_", " ").title()),
            _count_card("Valid", str(result.metrics_input.n_compounds)),
            _count_card("Selected", str(len(result.selected_indices))),
            _count_card("Invalid skipped", str(len(result.failed_indices))),
        ]
    )
    explained = result.explained_variance or []
    explained_html = "".join(
        f"<li>PC{index + 1}: {value * 100:.2f}% variance</li>"
        for index, value in enumerate(explained[:2])
    ) or "<li>Explained variance unavailable.</li>"
    return (
        "<html><body style='font-family: sans-serif;'>"
        "<h2>Diversity Picker</h2>"
        f"{cards}"
        "<h3>Diversity</h3>"
        "<ul>"
        f"<li>Input diversity score: {result.metrics_input.diversity_score:.4f}</li>"
        f"<li>Selected diversity score: {result.metrics_selected.diversity_score:.4f}</li>"
        f"<li>Mean nearest-neighbour distance: {result.metrics_selected.mean_nn_distance:.4f}</li>"
        "</ul>"
        "<h3>PCA projection</h3>"
        f"<ul>{explained_html}</ul>"
        "<h3>Display legend</h3>"
        "<ul><li>Blue circles: all valid compounds</li><li>Orange stars: selected compounds</li></ul>"
        "</body></html>"
    )


def _unique_variable_name(existing_names: set[str], wanted: str) -> str:
    if wanted not in existing_names:
        existing_names.add(wanted)
        return wanted
    index = 2
    while f"{wanted}_{index}" in existing_names:
        index += 1
    out = f"{wanted}_{index}"
    existing_names.add(out)
    return out


def _annotation_arrays(result: DiversitySelectionResult, total_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.asarray(result.coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[0] != int(total_count):
        coordinates = np.full((int(total_count), 2), np.nan, dtype=float)
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(coordinates.shape[0], dtype=float)])
    elif coordinates.shape[1] == 0:
        coordinates = np.full((int(total_count), 2), np.nan, dtype=float)
    elif coordinates.shape[1] > 2:
        coordinates = coordinates[:, :2]

    selected_mask = np.zeros(int(total_count), dtype=float)
    for index in result.selected_indices:
        if 0 <= int(index) < len(selected_mask):
            selected_mask[int(index)] = 1.0

    rank = np.full(int(total_count), np.nan, dtype=float)
    for index, value in enumerate(result.selection_ranks):
        if value is not None and 0 <= int(index) < len(rank):
            rank[int(index)] = float(value)

    return coordinates[:, 0], coordinates[:, 1], selected_mask, rank


def _annotated_table_from_data(data: Table, result: DiversitySelectionResult) -> Table:
    total_count = len(data)
    x_col, y_col, selected_col, rank_col = _annotation_arrays(result, total_count)

    existing_names = {var.name for var in list(data.domain.attributes) + list(data.domain.class_vars) + list(data.domain.metas)}
    x_var = ContinuousVariable(_unique_variable_name(existing_names, "chem_space_x"))
    y_var = ContinuousVariable(_unique_variable_name(existing_names, "chem_space_y"))
    selected_var = ContinuousVariable(_unique_variable_name(existing_names, "diversity_selected"))
    rank_var = ContinuousVariable(_unique_variable_name(existing_names, "diversity_rank"))

    attrs = list(data.domain.attributes) + [x_var, y_var, selected_var, rank_var]
    domain = Domain(attrs, data.domain.class_vars, data.domain.metas)
    x_base = np.asarray(data.X, dtype=float)
    x_extra = np.column_stack([x_col, y_col, selected_col, rank_col]).astype(float, copy=False)
    x_out = np.hstack([x_base, x_extra]) if x_base.size else x_extra
    y_out = data.Y if len(data.domain.class_vars) else None
    metas_out = data.metas if len(data.domain.metas) else None
    return safe_table_from_numpy(
        domain,
        X=x_out,
        Y=y_out,
        metas=metas_out,
        name=getattr(data, "name", "Diversity Annotated Data") or "Diversity Annotated Data",
    )


def _annotated_table_from_molecules(molecules: list[ChemMol], result: DiversitySelectionResult) -> Table:
    total_count = len(molecules)
    x_col, y_col, selected_col, rank_col = _annotation_arrays(result, total_count)
    attrs = [
        ContinuousVariable("chem_space_x"),
        ContinuousVariable("chem_space_y"),
        ContinuousVariable("diversity_selected"),
        ContinuousVariable("diversity_rank"),
    ]
    name_var = StringVariable("Name")
    smiles_var = StringVariable("SMILES")
    smiles_var.attributes["format"] = "SMILES"
    domain = Domain(attrs, metas=[name_var, smiles_var])

    metas = np.empty((total_count, 2), dtype=object)
    for index, molecule in enumerate(molecules):
        metas[index, 0] = str(molecule.name or "")
        metas[index, 1] = str(_molecule_smiles([molecule])[0] if molecules else "")

    x_out = np.column_stack([x_col, y_col, selected_col, rank_col]).astype(float, copy=False)
    return safe_table_from_numpy(
        domain,
        X=x_out,
        metas=metas,
        name="Diversity Annotated Data",
    )


class OWDiversityPicker(OWWidget):
    name = "Diversity Picker"
    description = "Select a diverse subset of compounds using MaxMin, sphere exclusion, or Butina clustering."
    icon = "icons/analysis/owdiversitypickerwidget.svg"
    priority = 134
    want_main_area = True

    class Inputs:
        data = Input("Data", Table)
        molecules = Input("Molecules", list, auto_summary=False)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output("Annotated Data", Table)
        remainder_data = Output("Remainder Data", Table)
        selected_molecules = Output("Selected Molecules", list, auto_summary=False)
        remainder_molecules = Output("Remainder Molecules", list, auto_summary=False)

    method_idx: int = Setting(0)
    smiles_var_name: str = Setting("")
    n_select: int = Setting(25)
    seed_idx: int = Setting(0)
    sphere_radius: float = Setting(0.35)
    butina_threshold: float = Setting(0.40)
    random_seed: int = Setting(42)
    auto_run: bool = Setting(True)

    _METHODS = [
        ("MaxMin", "maxmin"),
        ("Sphere exclusion", "sphere_exclusion"),
        ("Butina clusters", "butina"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.data: Table | None = None
        self.molecules: list[ChemMol] = []
        self._last_result: DiversitySelectionResult | None = None
        self._all_points_item: pg.ScatterPlotItem | None = None
        self._selected_points_item: pg.ScatterPlotItem | None = None

        root = QWidget(self.controlArea)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)
        self.controlArea.layout().addWidget(root)

        self.status_label = QLabel("Waiting for input…")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)

        self.smiles_combo = QComboBox()
        self.smiles_combo.currentTextChanged.connect(self._on_smiles_changed)
        form.addRow("SMILES column:", self.smiles_combo)

        self.method_combo = QComboBox()
        self.method_combo.addItems([label for label, _method in self._METHODS])
        self.method_combo.setCurrentIndex(int(self.method_idx))
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        form.addRow("Method:", self.method_combo)

        self.n_select_spin = QSpinBox()
        self.n_select_spin.setRange(1, 100000)
        self.n_select_spin.setValue(int(self.n_select))
        self.n_select_spin.valueChanged.connect(self._on_n_select_changed)
        form.addRow("Target count / clusters:", self.n_select_spin)

        self.seed_idx_spin = QSpinBox()
        self.seed_idx_spin.setRange(0, 100000)
        self.seed_idx_spin.setValue(int(self.seed_idx))
        self.seed_idx_spin.valueChanged.connect(self._on_seed_idx_changed)
        form.addRow("Seed index (MaxMin):", self.seed_idx_spin)

        self.sphere_radius_spin = QDoubleSpinBox()
        self.sphere_radius_spin.setRange(0.01, 0.99)
        self.sphere_radius_spin.setSingleStep(0.01)
        self.sphere_radius_spin.setDecimals(2)
        self.sphere_radius_spin.setValue(float(self.sphere_radius))
        self.sphere_radius_spin.valueChanged.connect(self._on_sphere_radius_changed)
        form.addRow("Sphere radius:", self.sphere_radius_spin)

        self.butina_threshold_spin = QDoubleSpinBox()
        self.butina_threshold_spin.setRange(0.01, 1.00)
        self.butina_threshold_spin.setSingleStep(0.01)
        self.butina_threshold_spin.setDecimals(2)
        self.butina_threshold_spin.setValue(float(self.butina_threshold))
        self.butina_threshold_spin.valueChanged.connect(self._on_butina_threshold_changed)
        form.addRow("Butina distance threshold:", self.butina_threshold_spin)

        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(0, 1_000_000)
        self.random_seed_spin.setValue(int(self.random_seed))
        self.random_seed_spin.valueChanged.connect(self._on_random_seed_changed)
        form.addRow("Random seed:", self.random_seed_spin)

        layout.addWidget(form_widget)

        self.auto_run_check = QCheckBox("Auto-run")
        self.auto_run_check.setChecked(bool(self.auto_run))
        self.auto_run_check.toggled.connect(self._on_auto_run_toggled)
        layout.addWidget(self.auto_run_check)

        self.run_button = QPushButton("Select diverse subset")
        self.run_button.clicked.connect(self.commit)
        layout.addWidget(self.run_button)
        layout.addStretch(1)

        self._summary_browser = QTextBrowser()
        self._summary_browser.setOpenExternalLinks(False)
        self._plot_widget = _styled_plot("Chemical Space Projection")

        tabs = QTabWidget()
        tabs.addTab(self._summary_browser, "Summary")
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        plot_layout.addWidget(self._plot_widget, 1)
        self._plot_legend = QLabel("Circles = all valid compounds, stars = selected compounds.")
        self._plot_legend.setWordWrap(True)
        plot_layout.addWidget(self._plot_legend)
        tabs.addTab(plot_tab, "Projection")
        self.mainArea.layout().addWidget(tabs)

        self._update_smiles_controls()
        self._update_method_controls()

    @Inputs.data
    def set_data(self, data: Table | None) -> None:
        self.data = data
        self._populate_smiles_combo()
        self._set_status(self._input_summary())
        self._maybe_autorun()

    @Inputs.molecules
    def set_molecules(self, molecules: list | None) -> None:
        self.molecules = [molecule for molecule in (molecules or []) if isinstance(molecule, ChemMol)]
        self._set_status(self._input_summary())
        self._maybe_autorun()

    def _populate_smiles_combo(self) -> None:
        self.smiles_combo.blockSignals(True)
        try:
            self.smiles_combo.clear()
            if self.data is None:
                self._update_smiles_controls()
                return

            smiles_vars = _find_smiles_vars(self.data)
            self.smiles_combo.addItems([var.name for var in smiles_vars])
            if smiles_vars:
                names = [var.name for var in smiles_vars]
                if self.smiles_var_name and self.smiles_var_name in names:
                    self.smiles_combo.setCurrentText(self.smiles_var_name)
                else:
                    self.smiles_var_name = smiles_vars[0].name
                    self.smiles_combo.setCurrentText(self.smiles_var_name)
        finally:
            self.smiles_combo.blockSignals(False)
        self._update_smiles_controls()

    def _update_smiles_controls(self) -> None:
        self.smiles_combo.setEnabled(self.data is not None)

    def _update_method_controls(self) -> None:
        method = self._METHODS[self.method_idx][1]
        self.seed_idx_spin.setEnabled(method == "maxmin")
        self.sphere_radius_spin.setEnabled(method == "sphere_exclusion")
        self.butina_threshold_spin.setEnabled(method == "butina")

    def _on_smiles_changed(self, text: str) -> None:
        self.smiles_var_name = text
        self._maybe_autorun()

    def _on_method_changed(self, index: int) -> None:
        self.method_idx = int(index)
        self._update_method_controls()
        self._maybe_autorun()

    def _on_n_select_changed(self, value: int) -> None:
        self.n_select = int(value)
        self._maybe_autorun()

    def _on_seed_idx_changed(self, value: int) -> None:
        self.seed_idx = int(value)
        self._maybe_autorun()

    def _on_sphere_radius_changed(self, value: float) -> None:
        self.sphere_radius = float(value)
        self._maybe_autorun()

    def _on_butina_threshold_changed(self, value: float) -> None:
        self.butina_threshold = float(value)
        self._maybe_autorun()

    def _on_random_seed_changed(self, value: int) -> None:
        self.random_seed = int(value)
        self._maybe_autorun()

    def _on_auto_run_toggled(self, checked: bool) -> None:
        self.auto_run = bool(checked)
        self._maybe_autorun()

    def _input_summary(self) -> str:
        table_rows = 0 if self.data is None else len(self.data)
        return f"Input: Table rows={table_rows}, Molecules={len(self.molecules)}"

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _maybe_autorun(self) -> None:
        if bool(self.auto_run) and (self.data is not None or self.molecules):
            self.commit()

    def _input_smiles(self) -> list[str]:
        if self.data is not None:
            return _table_smiles(self.data, self.smiles_var_name)
        return _molecule_smiles(self.molecules)

    @staticmethod
    def _subset_table(data: Table | None, indices: list[int]) -> Table | None:
        if data is None:
            return None
        return data[indices] if indices else data[:0]

    @staticmethod
    def _subset_molecules(molecules: list[ChemMol], indices: list[int]) -> list[ChemMol]:
        return [molecules[idx] for idx in indices if 0 <= idx < len(molecules)]

    def _clear_visuals(self) -> None:
        self._summary_browser.clear()
        self._plot_widget.clear()
        self._all_points_item = None
        self._selected_points_item = None

    def _update_plot(self, result: DiversitySelectionResult) -> None:
        self._plot_widget.clear()
        self._all_points_item = None
        self._selected_points_item = None

        coordinates = np.asarray(result.coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[0] == 0:
            return
        if coordinates.shape[1] == 1:
            coordinates = np.column_stack([coordinates[:, 0], np.zeros(coordinates.shape[0], dtype=float)])

        valid_mask = np.isfinite(coordinates[:, 0]) & np.isfinite(coordinates[:, 1])
        if not np.any(valid_mask):
            return

        x_values = coordinates[valid_mask, 0]
        y_values = coordinates[valid_mask, 1]
        self._all_points_item = pg.ScatterPlotItem(
            x=x_values,
            y=y_values,
            size=8,
            symbol="o",
            pen=pg.mkPen(None),
            brush=pg.mkBrush(59, 130, 246, 180),
        )
        self._plot_widget.addItem(self._all_points_item)

        selected_rows = [index for index in result.selected_indices if 0 <= index < coordinates.shape[0]]
        if selected_rows:
            selected_coords = coordinates[selected_rows, :]
            finite_selected = np.isfinite(selected_coords[:, 0]) & np.isfinite(selected_coords[:, 1])
            self._selected_points_item = pg.ScatterPlotItem(
                x=selected_coords[finite_selected, 0],
                y=selected_coords[finite_selected, 1],
                size=16,
                symbol="star",
                pen=pg.mkPen("#C2410C", width=1.5),
                brush=pg.mkBrush(251, 146, 60, 230),
            )
            self._plot_widget.addItem(self._selected_points_item)

        self._plot_widget.setLabel("bottom", "chem_space_x")
        self._plot_widget.setLabel("left", "chem_space_y")
        _set_plot_range(self._plot_widget, x_values, y_values)

    def _annotated_table(self, result: DiversitySelectionResult) -> Table | None:
        if self.data is not None:
            return _annotated_table_from_data(self.data, result)
        if self.molecules:
            return _annotated_table_from_molecules(self.molecules, result)
        return None

    def commit(self) -> None:
        clear_widget_messages(self)
        if self.data is None and not self.molecules:
            self._clear_visuals()
            self._set_status(format_no_input_status())
            self.Outputs.selected_data.send(None)
            self.Outputs.annotated_data.send(None)
            self.Outputs.remainder_data.send(None)
            self.Outputs.selected_molecules.send([])
            self.Outputs.remainder_molecules.send([])
            return

        try:
            smiles = self._input_smiles()
        except ValueError as exc:
            self._clear_visuals()
            set_widget_error(self, str(exc))
            self._set_status(format_error_status(str(exc)))
            self.Outputs.selected_data.send(None)
            self.Outputs.annotated_data.send(None)
            self.Outputs.remainder_data.send(None)
            self.Outputs.selected_molecules.send([])
            self.Outputs.remainder_molecules.send([])
            return

        method = self._METHODS[self.method_idx][1]
        result = select_diverse_subset(
            smiles,
            method=method,
            n_select=int(self.n_select),
            seed_idx=int(self.seed_idx),
            radius=float(self.sphere_radius),
            n_clusters=int(self.n_select),
            threshold=float(self.butina_threshold),
            random_seed=int(self.random_seed),
        )
        self._last_result = result
        self._send_outputs(result, len(smiles))

    def _send_outputs(self, result: DiversitySelectionResult, total_count: int) -> None:
        selected_indices = list(result.selected_indices)
        remainder_indices = [idx for idx in range(total_count) if idx not in set(selected_indices)]

        annotated = self._annotated_table(result)
        selected_data = self._subset_table(annotated, selected_indices)
        remainder_data = self._subset_table(annotated, remainder_indices)

        aligned_molecules = self.molecules if len(self.molecules) == total_count else []
        selected_molecules = self._subset_molecules(aligned_molecules, selected_indices)
        remainder_molecules = self._subset_molecules(aligned_molecules, remainder_indices)

        self.Outputs.selected_data.send(selected_data)
        self.Outputs.annotated_data.send(annotated)
        self.Outputs.remainder_data.send(remainder_data)
        self.Outputs.selected_molecules.send(selected_molecules)
        self.Outputs.remainder_molecules.send(remainder_molecules)

        self._summary_browser.setHtml(_summary_html(result))
        self._update_plot(result)

        input_metrics = result.metrics_input
        selected_metrics = result.metrics_selected
        self._set_status(
            format_done_status(
                f"selected={len(selected_indices)}/{input_metrics.n_compounds}",
                f"invalid skipped={len(result.failed_indices)}",
                f"PCA diversity {input_metrics.diversity_score:.4f}->{selected_metrics.diversity_score:.4f}",
                prefix="Selected",
            )
        )


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWDiversityPicker).run()

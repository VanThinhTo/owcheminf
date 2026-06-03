# Diversity Picker

## Status

Current widget in the package.

Source:
- [ow_diversity_picker.py](../../src/chem_inf_widgets/widgets/ow_diversity_picker.py)

## Purpose

`Diversity Picker` selects a maximally diverse subset of compounds using fingerprint-based algorithms.

## Input

- `Data` — Orange `Table` with a SMILES column
- `Molecules` — `ChemMol` list

## Output

- `Selected Data` — diverse subset as Orange `Table`
- `Annotated Data` — full table with chemical-space and diversity annotations
- `Remainder Data` — compounds not selected
- `Inspected Data` — compounds currently selected in the projection for inspection
- `Inspected Molecules` — interactive plot selection as `ChemMol` list for downstream viewers
- `Selected Molecules` — diverse subset as `ChemMol` list
- `Remainder Molecules` — remainder as `ChemMol` list

## Supported algorithms

- MaxMin (default, deterministic)
- Sphere exclusion
- Butina clustering

## Chemical-space view

- Morgan fingerprints are computed internally with radius `2` and `2048` bits.
- The widget projects valid compounds into `2D` with `PCA`.
- All valid compounds are shown as circles.
- Selected compounds are overlaid as stars.
- Hovering over a point shows a quick compound/rank preview.
- Clicked compounds are outlined and shown in the `Inspection` tab with structure previews.
- `Ctrl`-click adds or removes compounds from the current inspection selection.
- The `Inspection` list supports multi-selection and stays synced with the plot overlay and inspection outputs.
- `Inspect picked subset` loads the current diversity-picked star set into the inspection view in one step.
- The annotated full-table output includes:
  - `chem_space_x`
  - `chem_space_y`
  - `diversity_selected`
  - `diversity_rank`

## Typical workflow

1. `Mol Standardizer`
2. `Fingerprint Generator`
3. `Diversity Picker`
4. `Molecular Viewer`

## Notes

- MaxMin is the recommended default for teaching because it is fast and reproducible.
- The subset size is set directly in the widget controls.

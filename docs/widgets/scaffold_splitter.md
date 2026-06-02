# Scaffold Splitter

## Status

Current widget in the package.

Source:
- [ow_scaffold_splitter.py](../../src/chem_inf_widgets/widgets/ow_scaffold_splitter.py)
- [splitter_service.py](../../src/chem_inf_widgets/chemcore/services/splitter_service.py)
- [scaffold_splitter_service.py](../../src/chem_inf_widgets/chemcore/services/scaffold_splitter_service.py)

## Purpose

`Scaffold Splitter` separates a dataset into train, validation and test partitions. The default mode is scaffold-aware splitting, and the widget now also supports plain random splitting and activity-stratified splitting.

This is especially useful for QSAR workflows, where random splitting often gives over-optimistic results.

## Input

- Orange `Table`
- molecule structures as SMILES-containing rows for `Scaffold` mode
- any Orange table for `Random` mode

## Output

- split subsets such as train / validation / test
- split summary output

## Typical workflow

1. `Mol Standardizer`
2. `Scaffold Splitter`
3. descriptor widget
4. `QSAR Regression`
5. `Applicability Domain`

## Notes

- `Scaffold` mode is still the default and is usually the best choice for realistic QSAR validation.
- `Random` mode is useful as a baseline or for non-chemical tables.
- `Activity-stratified` mode keeps class/target balance more even when a target column is available.
- It also works well as a teaching example for why chemically aware evaluation matters.

# Chemical Series Explorer

## Status

Current widget in the package.

Source:
- [ow_chemical_series_explorer.py](../../src/chem_inf_widgets/widgets/ow_chemical_series_explorer.py)
- [chemical_series_service.py](../../src/chem_inf_widgets/chemcore/services/chemical_series_service.py)

## Purpose

`Chemical Series Explorer` groups compounds into scaffold-defined series and summarizes early SAR context.

The current version is intentionally lightweight:
- groups by `Murcko` or `Generic Murcko` scaffold
- exports per-series summary rows and per-member annotations
- computes per-series activity statistics when a numeric target is available
- reports invalid structures through the standard `ServiceIssue` path

## Input

- Orange `Table`
- molecule structures as SMILES-containing rows

## Output

- `Series Table`
- `Members Table`
- `Summary Table`

## Notes

- This widget is currently in `Cheminf - Development`.
- The first version focuses on scaffold series grouping rather than full R-group or matched-pair workflows.
- It pairs naturally with `Scaffold Analysis`, `Activity Cliff Finder`, and `R-Group Decomposition`.

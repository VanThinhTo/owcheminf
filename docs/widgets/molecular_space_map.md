# Molecular Space Map

## Status

Current widget in the package.

Source:
- [ow_molecular_space_map.py](../../src/chem_inf_widgets/widgets/ow_molecular_space_map.py)
- [molecular_space_service.py](../../src/chem_inf_widgets/chemcore/services/molecular_space_service.py)

## Purpose

`Molecular Space Map` projects descriptor or fingerprint matrices into a low-dimensional embedding that is easier to inspect.

The first version is intentionally lightweight:
- `PCA` is available by default
- `UMAP` is optional and falls back gracefully when unavailable
- the widget emits coordinate and summary tables

## Input

- Orange `Table`
- numeric attribute matrix such as descriptors or fingerprints

## Output

- coordinate table for downstream visualization or export
- summary table with method and variance metadata

## Notes

- This widget is currently in `Cheminf - Development`.
- It is a backend-first skeleton and does not yet include a rich interactive scatter plot.
- It works well after descriptor or fingerprint generation steps.

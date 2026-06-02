# ADMET Radar

## Status

Current widget in the package.

Source:
- [ow_admet_radar.py](../../src/chem_inf_widgets/widgets/ow_admet_radar.py)
- [admet_radar_service.py](../../src/chem_inf_widgets/chemcore/services/admet_radar_service.py)

## Purpose

`ADMET Radar` summarizes drug-likeness rules and common structural alerts across a molecular dataset.

The current version is intentionally backend-first:
- computes `Lipinski`, `Veber`, `Ghose`, `Egan`, and `Muegge`
- reports `QED`
- optionally reports `PAINS` and `Brenk` alerts
- exports Orange tables for downstream filtering and reporting

## Input

- Orange `Table`
- molecule structures as SMILES-containing rows

## Output

- `ADMET Table`
- `Flagged Compounds`
- `Summary Table`

## Notes

- This widget is currently in `Cheminf - Development`.
- The first version focuses on summary/report outputs and does not yet include a true radar chart visualization.
- It pairs naturally with `Dataset Profiler`, `Drug Filter`, and descriptor widgets.

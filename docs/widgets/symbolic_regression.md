# Symbolic Regression

## Status

Experimental widget in the package.

Source:
- [ow_symbolic_regression.py](../../src/chem_inf_widgets/widgets/ow_symbolic_regression.py)

## Purpose

`Symbolic Regression` searches for a short, human-readable regression expression from continuous descriptor columns. It is meant as a lightweight exploratory alternative to black-box QSAR models when you want a compact formula you can inspect directly.

## Input

- Orange `Table`
- expected descriptors: continuous attributes
- expected target: one continuous class variable or continuous attribute

## Output

- `Model` — fitted symbolic regression model bundle
- `Predictions` — input rows with added prediction and residual columns
- `Term Table` — selected symbolic terms and coefficients
- `Modeling Summary` — compact metrics/report table
- `Expression` — plain-text fitted formula

## Search space

The current experimental implementation builds expressions from a sparse basis search over:

- linear terms
- squares
- cubes
- sign-preserving `log1p(abs(x))`
- sign-preserving `sqrt(abs(x))`
- `1 / (1 + abs(x))`
- pairwise interactions

## Typical workflow

1. `QSAR Dataset Builder` or descriptor-ready table
2. `Symbolic Regression`
3. compare the compact expression with `QSAR/QSPR Model Hub` or `MLR Model Selection`

## Notes

- This widget is intentionally experimental and focuses on interpretability, not exhaustive global symbolic search.
- It works best on small-to-medium descriptor sets where you want a short formula rather than the highest possible predictive power.

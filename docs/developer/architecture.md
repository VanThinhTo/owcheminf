# Package Architecture

`owcheminf` is an Orange3 add-on with two main layers:

- `src/chem_inf_widgets/widgets`
- `src/chem_inf_widgets/chemcore`

The goal is to keep widgets focused on Orange UI concerns and move chemistry, data, and reporting logic into reusable Python services.

## Layer split

### `widgets/`

Widgets should own:

- Orange inputs and outputs
- settings and control widgets
- progress, status, and warning display
- conversion of service outputs into visible UI state
- orchestration of background work

Widgets should avoid owning:

- RDKit business rules
- descriptor calculations
- duplicate chemistry logic that already exists in `chemcore`
- large data transformations that can be unit-tested separately

### `chemcore/`

`chemcore` should own:

- molecule parsing and `ChemMol` helpers
- Orange `Table <-> ChemMol` conversion
- standardization, QC, descriptors, fingerprints, and search services
- QSAR data preparation, validation, diagnostics, and reporting
- report-oriented table builders
- small utilities that are reusable from widgets, CLI tools, and tests

### `tests/`

Tests are intentionally mixed by scope:

- pure service tests
- Orange table contract tests
- widget smoke tests
- workflow regression tests
- packaging and wheel smoke tests

Prefer testing pure services first. Add widget tests when behavior depends on Orange UI, signals, or output wiring.

## Recommended module shape

A good service module usually contains:

- one config dataclass
- one result dataclass
- a small public entrypoint such as `run_*`, `fit_*`, `compute_*`, or `split_*`
- a few private helpers for parsing, validation, and transformation

Example:

```python
from dataclasses import dataclass, field

from chem_inf_widgets.chemcore.result import ServiceIssue


@dataclass(frozen=True)
class ExampleConfig:
    threshold: float = 0.5


@dataclass(frozen=True)
class ExampleResult:
    rows: list[dict]
    issues: list[ServiceIssue] = field(default_factory=list)


def run_example(data, config: ExampleConfig | None = None) -> ExampleResult:
    cfg = config or ExampleConfig()
    ...
```

## Shared patterns already in use

The current codebase already has a few preferred building blocks:

- `chemcore/result.py`
  - `ServiceIssue`
  - `ServiceResult`
- `chemcore/services/from_orange.py`
  - `table_to_chemmols_with_report`
  - `chemmols_to_table`
- `chemcore/services/orange_table_utils.py`
  - shared Orange table builders
- `widgets/utils/`
  - widget-side output and warning helpers

When adding a new feature, prefer extending these pieces instead of creating a new parallel pattern.

## Public API guidance

Public behavior that widgets or external scripts already use should remain stable unless there is a strong reason to change it.

When refactoring:

- keep old import paths working where practical
- turn large modules into compatibility wrappers instead of deleting them
- keep widget names and output names stable unless a deliberate migration is planned

The QSAR regression split is a good reference:

- new focused modules live under `chemcore/qsar/`
- the old service import path remains available as a wrapper

## Optional dependency policy

Optional dependencies are allowed, but the base install should remain usable without them.

Rules:

- required runtime dependencies belong in `pyproject.toml` and `environment.yml`
- optional capabilities should fail gracefully with a clear warning or error
- widget imports should not crash the whole add-on just because an optional backend is missing
- docs should name the optional package explicitly

Good examples:

- `mordred`
- `padelpy`
- `py3Dmol`
- `torch`
- `umap-learn`

## Clean coding conventions

Preferred conventions for this package:

- keep functions small and single-purpose
- prefer explicit names over clever compression
- avoid silent `except: pass`
- return structured issues instead of hiding row failures
- reuse existing helpers before adding a new utility layer
- keep UI code and computation separate
- make code review easy by shipping small commits

## Practical file map

Useful places to inspect before adding new work:

- `src/chem_inf_widgets/widgets/__init__.py`
- `src/chem_inf_widgets/widgets/utils/`
- `src/chem_inf_widgets/chemcore/result.py`
- `src/chem_inf_widgets/chemcore/services/`
- `src/chem_inf_widgets/chemcore/qsar/`
- `tests/`

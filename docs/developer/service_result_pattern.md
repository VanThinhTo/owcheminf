# Service Result Pattern

The package is moving away from silent failures and toward explicit, structured service results.

The key types live in [src/chem_inf_widgets/chemcore/result.py](../../src/chem_inf_widgets/chemcore/result.py).

## Core types

### `ServiceIssue`

Use `ServiceIssue` for warnings and errors that should survive the service call.

Fields:

- `code`
- `message`
- `severity`
- `row_index`
- `molecule_id`
- `field`
- `details`

Typical use cases:

- invalid molecules
- failed descriptor calculations
- fallback behavior
- optional dependency absence
- skipped rows

### `ServiceResult[T]`

`ServiceResult` is a generic wrapper for services that return one main payload plus issues and summary data.

Use it when:

- a service has one clear `data` payload
- callers need both the data and the warnings
- you want `.ok`, `.warnings`, and `.errors`

Use a custom dataclass instead when:

- the result has many named payloads
- a generic `data` field would make the call site less readable

Both approaches are used in this package.

## Preferred pattern

Instead of:

```python
try:
    value = risky_operation()
except Exception:
    pass
```

prefer:

```python
issues: list[ServiceIssue] = []

try:
    value = risky_operation()
except ValueError as exc:
    issues.append(
        ServiceIssue(
            code="operation_failed",
            message=str(exc),
            severity="warning",
            row_index=row_index,
        )
    )
    value = default_value
```

This keeps the service usable while preserving the reason something went wrong.

## Row-level vs service-level issues

### Row-level

Use row-level issues when the service can continue for other rows.

Examples:

- one bad SMILES in a dataset
- one descriptor failing on one molecule
- one imported record needing fallback parsing

### Service-level

Use service-level errors when the whole operation cannot continue.

Examples:

- no input data
- missing required target column
- unsupported split method
- input matrix is empty or non-numeric

## Severity guidance

- `info`
  - rare; use for notable but harmless context
- `warning`
  - partial failure, fallback, or row exclusion
- `error`
  - the requested operation could not produce a meaningful result

When in doubt:

- use `warning` if there is still a useful payload
- use `error` if the output is effectively unusable

## Widget handoff

Widgets should not rebuild warning logic by hand when a service already returns issues.

Preferred widget pattern:

```python
from chem_inf_widgets.widgets.utils import show_service_issues

result = run_service(...)
show_service_issues(self, result.issues, subject="descriptor service")
```

If the service uses a custom result dataclass, expose its `issues` attribute in the same way.

## Summary helpers

Many services also expose compact report-friendly helpers such as:

- `*_summary_as_rows(...)`
- `*_records_as_dicts(...)`
- `records_to_orange_table(...)`

This makes it easier for widgets to:

- send Orange tables
- render report tabs
- keep UI code thin

## Good current examples

- [src/chem_inf_widgets/chemcore/services/molecule_qc_service.py](../../src/chem_inf_widgets/chemcore/services/molecule_qc_service.py)
- [src/chem_inf_widgets/chemcore/services/molecule_import_service.py](../../src/chem_inf_widgets/chemcore/services/molecule_import_service.py)
- [src/chem_inf_widgets/chemcore/services/dataset_profiler_service.py](../../src/chem_inf_widgets/chemcore/services/dataset_profiler_service.py)
- [src/chem_inf_widgets/chemcore/services/splitter_service.py](../../src/chem_inf_widgets/chemcore/services/splitter_service.py)

## Anti-patterns to avoid

- `except Exception: pass`
- returning `None` for multiple different failure modes without explanation
- hiding invalid rows without reporting how many were skipped
- raising a broad exception after already doing useful partial work
- embedding user-facing warning strings only in widgets instead of the service

## Design checklist for a new service

Before you finish a new service, check:

- does it report partial failures explicitly?
- are row-level issues attached where possible?
- does the result shape make sense for downstream widgets?
- can a widget show warnings without reverse-engineering the failure?
- is there a test for both success and failure paths?

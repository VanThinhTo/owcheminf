# Widget Template

This page gives a minimal template for new Orange widgets in `owcheminf`.

The goal is not to copy-paste a huge starter file. The goal is to keep new widgets consistent with the package style:

- thin UI layer
- service-driven backend
- explicit warnings
- simple outputs

## Minimal widget template

```python
from __future__ import annotations

from typing import Optional

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from chem_inf_widgets.widgets.ui_helpers import clear_widget_messages, format_no_input_status


class OWExampleWidget(OWWidget):
    name = "Example Widget"
    description = "Short description."

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        result = Output("Result", Table)

    class Warning(OWWidget.Warning):
        issues = Msg("{}")

    def __init__(self):
        super().__init__()
        self.data: Optional[Table] = None
        self._build_ui()

    def _build_ui(self):
        box = gui.widgetBox(self.controlArea, "Run")
        gui.button(box, self, "Apply", callback=self.commit)

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.commit.now()

    @gui.deferred
    def commit(self):
        clear_widget_messages(self)
        if self.data is None:
            self.Outputs.result.send(None)
            self.setStatusMessage(format_no_input_status())
            return
        ...
```

## Preferred responsibilities

Widgets should own:

- Orange input/output declarations
- settings
- controls and layouts
- status text
- progress bars and cancellation hooks
- calling services
- sending output tables or objects

Widgets should not own:

- descriptor math
- scaffold chemistry
- dataset splitting rules
- report-row business logic
- duplicate versions of shared Orange conversion helpers

## Typical widget flow

The most common structure is:

1. receive Orange inputs
2. cache input data on `self`
3. clear warnings/errors
4. validate required inputs
5. call one `chemcore` service
6. surface `ServiceIssue` warnings
7. send outputs
8. update status text

## Status and message helpers

Prefer shared helpers instead of custom inline strings when possible.

Useful places:

- [src/chem_inf_widgets/widgets/ui_helpers.py](../../src/chem_inf_widgets/widgets/ui_helpers.py)
- [src/chem_inf_widgets/widgets/utils/](../../src/chem_inf_widgets/widgets/utils/__init__.py)

Common helpers:

- `clear_widget_messages(...)`
- `format_no_input_status(...)`
- `format_done_status(...)`
- `show_service_issues(...)`
- `send_output_values(...)`

## Handling service issues

If the backend returns structured issues, do not hide them.

Pattern:

```python
result = run_service(...)
show_service_issues(self, result.issues, subject="dataset profiler")
```

If there are no issues, the warning message should clear cleanly.

## Output conventions

When a widget has multiple related outputs, prefer grouped sending with the shared helper:

```python
send_output_values(
    (self.Outputs.cleaned_data, cleaned),
    (self.Outputs.report, report),
)
```

When there is no usable input:

```python
self.Outputs.cleaned_data.send(None)
self.Outputs.report.send(None)
```

or use helper methods already present in the widget.

## Background work

If the task can block the UI, move the heavy part out of the widget method.

Recommended pattern:

- widget method builds config
- background function runs service
- completion handler updates UI and outputs

This is already used in several larger widgets such as:

- [OWMolStandardizer](../../src/chem_inf_widgets/widgets/ow_mol_standardizer.py)
- [OWMoleculeQCDashboard](../../src/chem_inf_widgets/widgets/ow_molecule_qc_dashboard.py)

## Registration checklist

After creating a new widget:

1. add it to `src/chem_inf_widgets/widgets/__init__.py`
2. make sure `name`, `description`, `icon`, and `priority` are set
3. choose the correct category
4. add at least a minimal widget test
5. update widget docs if the widget is user-facing

## Common mistakes

- doing chemistry directly in `commit()`
- reimplementing `table_to_chemmols_with_report(...)`
- swallowing exceptions instead of surfacing warnings
- adding too many settings before the workflow is proven
- making widget tests depend on unrelated heavy backends

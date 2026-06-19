from __future__ import annotations

import html
import sys
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList

sys.path.insert(0, str(Path(__file__).parent))

from widget_catalog import CATEGORIES, WIDGETS, WIDGETS_BY_SLUG


project = "OWChemInf"
author = "OWChemInf contributors"
copyright = "2026, OWChemInf contributors"
release = "0.3.0"

extensions = []
templates_path = ["_templates"]
exclude_patterns = []
source_suffix = ".rst"
master_doc = "index"

html_theme = "alabaster"
html_static_path = ["_static"]
html_extra_path = ["widget-help.html"]
html_css_files = ["owcheminf.css"]
html_title = "OWChemInf Documentation"
html_short_title = "OWChemInf"
html_theme_options = {
    "description": "Chemoinformatics widgets for Orange Data Mining",
    "github_user": "VanThinhTo",
    "github_repo": "owcheminf",
    "github_button": True,
    "github_type": "star",
    "fixed_sidebar": True,
    "sidebar_width": "260px",
    "page_width": "1120px",
}
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "searchbox.html",
    ]
}


def _signal_description(label: str, type_name: str, direction: str) -> str:
    if type_name == "Table":
        object_name = "Orange data table"
    elif type_name == "list":
        object_name = "molecule collection"
    elif type_name == "str":
        object_name = "text value"
    elif type_name == "int":
        object_name = "integer value"
    elif type_name == "ChemMol":
        object_name = "single chemical molecule"
    else:
        object_name = "Python object"
    verb = "received by the widget" if direction == "input" else "produced by the widget"
    return f"{object_name} {verb}."


def _render_widget_rst(widget: dict) -> str:
    icon_url = (
        "https://raw.githubusercontent.com/VanThinhTo/owcheminf/main/"
        f"src/chem_inf_widgets/widgets/{widget['icon']}"
    )
    lines = [
        ".. raw:: html",
        "",
        '   <div class="widget-lead">',
        f'     <img src="{html.escape(icon_url)}" alt="" class="widget-lead-icon">',
        f'     <p>{html.escape(widget["description"])}</p>',
        "   </div>",
        "",
        ".. rubric:: Inputs",
        "",
    ]
    if widget["inputs"]:
        for label, type_name in widget["inputs"]:
            description = _signal_description(label, type_name, "input")
            lines.append(f"* **{label}** ({type_name}): {description}")
    else:
        lines.append(
            "This widget has no Orange input signal. It reads from a file, an editor, "
            "or a remote service configured in the widget."
        )
    lines.extend(["", ".. rubric:: Outputs", ""])
    if widget["outputs"]:
        for label, type_name in widget["outputs"]:
            description = _signal_description(label, type_name, "output")
            lines.append(f"* **{label}** ({type_name}): {description}")
    else:
        lines.append(
            "This widget has no Orange output signal. It writes to disk or presents "
            "an interactive view."
        )
    lines.extend(
        [
            "",
            ".. rubric:: Overview",
            "",
            widget["overview"],
            "",
            ".. rubric:: Key controls",
            "",
        ]
    )
    for index, control in enumerate(widget["controls"], start=1):
        lines.append(f"{index}. {control}")
    lines.extend(
        [
            "",
            ".. rubric:: Example",
            "",
            "A typical workflow is:",
            "",
            ".. code-block:: text",
            "",
        ]
    )
    for line in widget["workflow"].splitlines():
        lines.append(f"   {line}")
    lines.extend(
        [
            "",
            widget["note"],
            "",
            f"`View the widget source on GitHub <https://github.com/VanThinhTo/owcheminf/blob/main/src/chem_inf_widgets/widgets/{widget['module']}.py>`_.",
        ]
    )
    return "\n".join(lines)


class WidgetDocDirective(Directive):
    required_arguments = 1
    has_content = False

    def run(self):
        slug = self.arguments[0].strip()
        widget = WIDGETS_BY_SLUG.get(slug)
        if widget is None:
            raise self.error(f"Unknown widget slug: {slug}")
        container = nodes.container(classes=["widget-documentation"])
        self.state.nested_parse(
            StringList(_render_widget_rst(widget).splitlines()),
            self.content_offset,
            container,
        )
        return [container]


class WidgetGridDirective(Directive):
    required_arguments = 1
    has_content = False

    def run(self):
        category_key = self.arguments[0].strip()
        category = CATEGORIES.get(category_key)
        if category is None:
            raise self.error(f"Unknown category: {category_key}")
        cards = []
        for widget in WIDGETS:
            if widget["category"] != category_key:
                continue
            icon_url = (
                "https://raw.githubusercontent.com/VanThinhTo/owcheminf/main/"
                f"src/chem_inf_widgets/widgets/{widget['icon']}"
            )
            href = f"{widget['slug']}.html"
            cards.append(
                '<a class="widget-card" href="{href}">'
                '<img src="{icon}" alt="">'
                '<span>{name}</span>'
                "</a>".format(
                    href=html.escape(href),
                    icon=html.escape(icon_url),
                    name=html.escape(widget["name"]),
                )
            )
        raw = (
            f'<div class="category-banner category-{category_key}">'
            f'<span>{html.escape(category["title"])}</span></div>'
            f'<div class="widget-grid">{"".join(cards)}</div>'
        )
        return [nodes.raw("", raw, format="html")]


def setup(app):
    app.add_directive("widget-doc", WidgetDocDirective)
    app.add_directive("widget-grid", WidgetGridDirective)
    return {"version": "1.0", "parallel_read_safe": True}

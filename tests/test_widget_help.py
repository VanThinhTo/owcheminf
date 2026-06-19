from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytest.importorskip("Orange")

from AnyQt.QtCore import QUrl
from orangecanvas.help.provider import SimpleHelpProvider

import chem_inf_widgets.widgets as widget_package


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_default_palette_widgets_have_documentation_refs():
    default_modules = {
        module_name
        for spec in widget_package.LIGHT_CATEGORY_SPECS
        for module_name in spec["modules"]
    }

    assert set(widget_package.WIDGET_HELP_REFS) == default_modules


def test_widget_descriptions_resolve_to_live_documentation_paths():
    provider = SimpleHelpProvider(baseurl=QUrl(widget_package.DOCUMENTATION_ROOT))

    for spec in widget_package.LIGHT_CATEGORY_SPECS:
        for description in widget_package._iter_widget_descriptions(spec):
            assert description.project_name == widget_package.PROJECT_NAME
            assert description.help_ref
            assert provider.search(description).toString() == (
                f"{widget_package.DOCUMENTATION_ROOT}{description.help_ref}.html"
            )


def test_project_registers_orange_html_help_provider():
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["entry-points"]["orange.canvas.help"] == {
        "html-simple": "chem_inf_widgets.widgets:WIDGET_HELP_PATH"
    }

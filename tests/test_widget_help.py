from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytest.importorskip("Orange")

from AnyQt.QtCore import QUrl
from orangecanvas.help.provider import HtmlIndexProvider

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
    index_source = (PROJECT_ROOT / "docs" / "source" / "index.rst").read_text(
        encoding="utf-8"
    )

    for spec in widget_package.LIGHT_CATEGORY_SPECS:
        for description in widget_package._iter_widget_descriptions(spec):
            assert description.project_name == widget_package.PROJECT_NAME
            assert description.help_ref
            assert description.name
            assert (
                f":doc:`{description.name} <{description.help_ref}>`"
                in index_source
            )


def test_project_registers_orange_html_help_provider():
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["entry-points"]["orange.canvas.help"] == {
        "html-index": "chem_inf_widgets.widgets:WIDGET_HELP_PATH"
    }


def test_help_provider_indexes_widget_links_from_machine_readable_index():
    target, xpath = widget_package.WIDGET_HELP_PATH[0]

    assert target == f"{widget_package.DOCUMENTATION_ROOT}widget-help.html"
    assert xpath == ".//*[@id='widgets']//li/a"

    local_index = PROJECT_ROOT / "docs" / "source" / "widget-help.html"
    provider = HtmlIndexProvider(
        inventory=QUrl.fromLocalFile(str(local_index)),
        xpathquery=xpath,
    )
    for spec in widget_package.LIGHT_CATEGORY_SPECS:
        for description in widget_package._iter_widget_descriptions(spec):
            url = provider.search(description)
            assert url.path().endswith(f"/{description.help_ref}.html")

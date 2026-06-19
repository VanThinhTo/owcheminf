from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytest.importorskip("Orange")

from AnyQt.QtCore import QUrl
from orangecanvas.help.provider import HtmlIndexProvider
from orangecanvas.registry import WidgetRegistry
from orangewidget.workflow.discovery import WidgetDiscovery

import chem_inf_widgets.widgets as widget_package


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_default_palette_widgets_have_documentation_refs():
    default_modules = {
        module_name
        for spec in widget_package.LIGHT_CATEGORY_SPECS
        for module_name in spec["modules"]
    }

    assert set(widget_package.WIDGET_HELP_REFS) == default_modules


def test_widget_descriptions_resolve_to_documentation_paths():
    help_index = (
        PROJECT_ROOT / "docs" / "source" / "widget-help.html"
    ).read_text(encoding="utf-8")

    for spec in widget_package.LIGHT_CATEGORY_SPECS:
        for description in widget_package._iter_widget_descriptions(spec):
            assert description.project_name == widget_package.PROJECT_NAME
            assert description.help_ref
            assert description.name
            documentation_page = (
                PROJECT_ROOT
                / "docs"
                / "source"
                / f"{description.help_ref}.rst"
            )
            assert documentation_page.is_file()
            assert f'href="{description.help_ref}.html"' in help_index


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


def test_combined_help_index_resolves_standard_orange_widgets():
    import Orange.widgets

    local_index = PROJECT_ROOT / "docs" / "source" / "widget-help.html"
    provider = HtmlIndexProvider(
        inventory=QUrl.fromLocalFile(str(local_index)),
        xpathquery=widget_package.WIDGET_HELP_INDEX_XPATH,
    )
    registry = WidgetRegistry()
    Orange.widgets.widget_discovery(WidgetDiscovery(registry))

    active_widgets = [
        description
        for description in registry.widgets()
        if description.category != "Orange Obsolete"
    ]
    resolved = [provider.search(description) for description in active_widgets]

    assert len(active_widgets) == 104
    assert len(resolved) == 104
    assert all("/orange/source/widgets/" in url.path() for url in resolved)


def test_cheminf_discovery_overrides_orange_help_path():
    import Orange.widgets

    original = Orange.widgets.WIDGET_HELP_PATH
    try:
        Orange.widgets.WIDGET_HELP_PATH = ()
        registry = WidgetRegistry()
        widget_package.widget_discovery(WidgetDiscovery(registry))
        assert Orange.widgets.WIDGET_HELP_PATH == widget_package.WIDGET_HELP_PATH
    finally:
        Orange.widgets.WIDGET_HELP_PATH = original

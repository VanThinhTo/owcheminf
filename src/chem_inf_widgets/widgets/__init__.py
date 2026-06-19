"""Orange widget discovery for the Chemoinformatics add-on.

The default Orange sidebar exposes a light palette with the most stable and
widely used widgets. Overlapping, optional-dependency-heavy, diagnostic, or
experimental tools remain available in the code base and can be re-enabled
with ``OWCHEMINF_PALETTE=full`` before launching Orange.
"""

from __future__ import annotations

import inspect
import os
from importlib import import_module

from orangecanvas.registry import CategoryDescription, WidgetDescription
from orangewidget.workflow.discovery import widget_desc_from_module

PACKAGE_NAME = __name__
PROJECT_NAME = "chem-inf-widgets"
PALETTE_ENV_VAR = "OWCHEMINF_PALETTE"

NAME = "Chemoinformatics"
DESCRIPTION = "Chemoinformatics widgets for Orange Data Mining."
PRIORITY = 1

DOCUMENTATION_ROOT = "https://owcheminf.readthedocs.io/en/latest/"
WIDGET_HELP_INDEX_XPATH = ".//*[@id='widgets']//li/a"
WIDGET_HELP_PATH = (
    (f"{DOCUMENTATION_ROOT}widget-help.html", WIDGET_HELP_INDEX_XPATH),
)

# Orange's context-menu Help action resolves a widget through its installed
# distribution. The HTML-index provider matches each widget name against the
# links under the documentation index's ``Widgets`` section. Keep help_ref as
# explicit metadata as well so the intended page remains testable and can be
# reused by other Orange help providers.
WIDGET_HELP_REFS = {
    "ow_molecule_import_hub": "widgets/molecule-import-hub",
    "ow_molecule_export_hub": "widgets/molecule-export-hub",
    "ow_sdf_reader": "widgets/sdf-reader",
    "ow_sdf_writer": "widgets/sdf-writer",
    "ow_chembl_browser": "widgets/chembl-browser",
    "ow_chembl_dataretriever": "widgets/chembl-bioactivity-retriever",
    "ow_molecule_qc_dashboard": "widgets/molecule-qc-dashboard",
    "ow_mol_standardizer": "widgets/mol-standardizer",
    "ow_mol_editor": "widgets/mol-editor",
    "ow_compound_detail_card": "widgets/compound-detail-card",
    "ow_mol_viewer": "widgets/molecular-viewer",
    "ow_substructure_search": "widgets/substructure-similarity-search",
    "ow_similarity_search": "widgets/similarity-search",
    "ow_fingerprint_generator": "widgets/fingerprint-generator",
    "ow_rdkit_descriptors": "widgets/rdkit-descriptors",
    "ow_mol_descriptor": "widgets/mol-descriptors-2",
    "ow_scaffold_analysis": "widgets/scaffold-analysis",
    "ow_scaffold_splitter": "widgets/scaffold-splitter",
    "ow_diversity_picker": "widgets/diversity-picker",
    "ow_activity_cliff_finder": "widgets/activity-cliff-finder",
    "ow_rgroup_decomposition": "widgets/r-group-decomposition",
    "ow_matched_molecular_pairs": "widgets/matched-molecular-pairs",
    "ow_pair_viewer": "widgets/pair-viewer",
    "ow_drug_filter": "widgets/drug-filter",
    "ow_qsar_dataset_builder": "widgets/qsar-dataset-builder",
    "ow_descriptor_explorer": "widgets/qsar-descriptor-explorer",
    "ow_descriptor_filter": "widgets/descriptor-pre-selector",
    "ow_qsar_model_hub": "widgets/qsar-model-hub",
    "ow_qsar_validation_dashboard": "widgets/qsar-validation-dashboard",
    "ow_applicability_domain": "widgets/applicability-domain",
    "ow_model_explanation": "widgets/model-explanation",
    "ow_qsar_report_generator": "widgets/qsar-report-generator",
    "ow_qsar_prediction_packager": "widgets/qsar-prediction-packager",
    "ow_reactionviewer": "widgets/reaction-viewer",
    "ow_reactor": "widgets/rdkit-reactor",
    "ow_reaction_enumerator": "widgets/reaction-enumerator",
}


def install_combined_orange_help() -> None:
    """Route Orange's built-in widget Help action through this documentation.

    Orange resolves help providers lazily from ``Orange.widgets.WIDGET_HELP_PATH``.
    Replacing that value during add-on discovery keeps the standard Orange
    widgets and the Cheminf widgets on the same deployed documentation domain.
    """
    try:
        import Orange.widgets as orange_widgets
        from orangecanvas.help import manager as help_manager
    except ImportError:
        return

    orange_widgets.WIDGET_HELP_PATH = WIDGET_HELP_PATH
    help_manager._providers_cache.pop("Orange3", None)


# Curated categories for the v0.3.0 light-by-default layout.
#
# Design rule:
# - Core: robust entry-point widgets used in most workflows.
# - Search & Analysis: broadly useful cheminformatics analysis widgets.
# - Filters & Alerts: rule-based filters and structural alerts.
# - QSAR: compact public modeling workflow; older/overlapping QSAR widgets are
#   kept in Development until they are merged or retired.
# - Reactions: reaction-specific widgets.
# - Development: useful but experimental, diagnostic, optional-dependency-heavy,
#   unstable-on-some-systems, or overlapping widgets that should not clutter
#   the main user-facing palette.
_LIGHT_CATEGORY_SPECS = (
    {
        "name": "Cheminf - Core",
        "description": "Import, clean, standardize, inspect, and visualize molecular datasets.",
        "icon": "icons/categories/cheminf_data.svg",
        "background": "#EFF6FF",
        "priority": 1000,
        "modules": (
            "ow_molecule_import_hub",
            "ow_molecule_export_hub",
            "ow_sdf_reader",
            "ow_sdf_writer",
            "ow_chembl_browser",
            "ow_chembl_dataretriever",
            "ow_molecule_qc_dashboard",
            "ow_mol_standardizer",
            "ow_mol_editor",
            "ow_compound_detail_card",
            "ow_mol_viewer",
        ),
    },
    {
        "name": "Cheminf - Search & Analysis",
        "description": "Search, featurize, filter, cluster, and analyze molecular libraries.",
        "icon": "icons/categories/cheminf_processing.svg",
        "background": "#F0FDF4",
        "priority": 1001,
        "modules": (
            "ow_substructure_search",
            "ow_similarity_search",
            "ow_fingerprint_generator",
            "ow_rdkit_descriptors",
            "ow_mol_descriptor",
            "ow_scaffold_analysis",
            "ow_scaffold_splitter",
            "ow_diversity_picker",
            "ow_activity_cliff_finder",
            "ow_rgroup_decomposition",
            "ow_matched_molecular_pairs",
            "ow_pair_viewer",
        ),
    },
    {
        "name": "Cheminf - Filters & Alerts",
        "description": "Rule-based drug-likeness filters, structural alerts, and library triage tools.",
        "icon": "icons/categories/cheminf_processing.svg",
        "background": "#FEFCE8",
        "priority": 1002,
        "modules": (
            "ow_drug_filter",
        ),
    },
    {
        "name": "Cheminf - QSAR",
        "description": "Build QSAR-ready datasets, audit descriptors, train and validate models, and assemble reports or prediction packages.",
        "icon": "icons/categories/cheminf_modeling.svg",
        "background": "#FFF7ED",
        "priority": 1003,
        "modules": (
            "ow_qsar_dataset_builder",
            "ow_descriptor_explorer",
            "ow_descriptor_filter",
            "ow_qsar_model_hub",
            "ow_qsar_validation_dashboard",
            "ow_applicability_domain",
            "ow_model_explanation",
            "ow_qsar_report_generator",
            "ow_qsar_prediction_packager",
        ),
    },
    {
        "name": "Cheminf - Reactions",
        "description": "Inspect, enumerate, and apply RDKit reaction workflows.",
        "icon": "icons/categories/cheminf_processing.svg",
        "background": "#F5F3FF",
        "priority": 1004,
        "modules": (
            "ow_reactionviewer",
            "ow_reactor",
            "ow_reaction_enumerator",
        ),
    },
)

_DEVELOPMENT_CATEGORY_SPECS = (
    {
        "name": "Cheminf - Development",
        "description": "Experimental, duplicate, optional-dependency-heavy, unstable, and legacy widgets.",
        "icon": "icons/categories/cheminf_modeling.svg",
        "background": "#F8FAFC",
        "priority": 1999,
        "modules": (
            "ow_mol_ketcher_editor",
            "ow_mol3d_viewer",
            "ow_widget_smoke_tester",
            "ow_audit_trail_viewer",
            "ow_pharmafp_search",
            "ow_cyclic_registry_fingerprint",
            "ow_isida_descriptors",
            "ow_padel_descriptors",
            "ow_ad_workbench",
            "ow_atom_contribution_map",
            "ow_qsar_regression",
            "ow_mlr_model_selection",
            "ow_dataset_profiler",
            "ow_chemical_series_explorer",
            "ow_admet_radar",
            "ow_molecular_space_map",
            "ow_symbolic_regression",
        ),
    },
)

LIGHT_CATEGORY_SPECS = _LIGHT_CATEGORY_SPECS
FULL_CATEGORY_SPECS = _LIGHT_CATEGORY_SPECS + _DEVELOPMENT_CATEGORY_SPECS

# Backward-compatible alias for the default user-facing palette.
_CATEGORY_SPECS = LIGHT_CATEGORY_SPECS


def _normalize_palette_name(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"full", "all", "dev", "development", "labs"}:
        return "full"
    return "light"


def get_category_specs(palette: str | None = None):
    palette_name = _normalize_palette_name(
        os.environ.get(PALETTE_ENV_VAR, "light") if palette is None else palette
    )
    if palette_name == "full":
        return FULL_CATEGORY_SPECS
    return LIGHT_CATEGORY_SPECS


def _category_description(spec: dict[str, object]) -> CategoryDescription:
    return CategoryDescription(
        name=spec["name"],
        qualified_name=PACKAGE_NAME,
        package=PACKAGE_NAME,
        description=spec["description"],
        priority=spec["priority"],
        icon=spec["icon"],
        background=spec["background"],
    )


def _iter_widget_descriptions(spec: dict[str, object]):
    category_name = spec["name"]
    for module_name in spec["modules"]:
        module = import_module(f"{PACKAGE_NAME}.{module_name}")
        desc = _widget_desc_from_local_module(module)
        desc.category = category_name
        desc.project_name = PROJECT_NAME
        desc.help_ref = WIDGET_HELP_REFS.get(module_name)
        yield desc


def _widget_desc_from_local_module(module) -> WidgetDescription:
    """Return the widget description for a class defined in ``module``.

    Some widgets import other OWWidget classes at module scope for internal
    workflow smoke tests. Orange's default helper accepts the first class with
    ``get_widget_description`` it encounters, even if that class was merely
    imported from another module. Filtering by ``__module__`` ensures the
    registered widget always matches the current file.
    """
    for _, widget_class in inspect.getmembers(module, inspect.isclass):
        if widget_class.__module__ != module.__name__:
            continue
        if not hasattr(widget_class, "get_widget_description"):
            continue

        description = widget_class.get_widget_description()
        if description is None:
            continue

        desc = WidgetDescription(**description)
        desc.package = module.__package__
        desc.category = widget_class.category
        return desc

    return widget_desc_from_module(module)


def widget_discovery(discovery) -> None:
    """Register the curated Orange categories for this add-on."""
    from chem_inf_widgets.widgets.theme import apply_theme

    install_combined_orange_help()
    apply_theme()

    for spec in get_category_specs():
        discovery.handle_category(_category_description(spec))
        for desc in _iter_widget_descriptions(spec):
            discovery.handle_widget(desc)

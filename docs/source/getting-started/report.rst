Report
======

A useful chemoinformatics report records data provenance, molecular curation,
descriptor generation, model validation, applicability domain, and failures.

:doc:`../widgets/qsar-report-generator` accepts dataset, metric, prediction,
validation, feature-importance, model-summary, applicability-domain, and
explanation tables. It produces Markdown, HTML, PDF-path, section, and summary
outputs.

.. code-block:: text

   QSAR/QSPR Model Hub ──────────┐
   QSAR Validation Dashboard ────┤
   Applicability Domain ─────────┼→ QSAR Report Generator
   Model Explanation ────────────┘

Retain the underlying tables with the rendered report. The report should make
the target definition, split strategy, metrics, thresholds, software version,
and excluded records explicit.


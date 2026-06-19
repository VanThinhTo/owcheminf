Exporting Visualizations
========================

OWChemInf visualization widgets provide export controls appropriate to their
content. :doc:`../widgets/reaction-viewer` exports reaction depictions as PNG
or SVG. QSAR reporting widgets can export rendered diagnostic content through
:doc:`../widgets/qsar-report-generator`.

For a reproducible figure:

1. Preserve the input table and widget settings.
2. Use vector output such as SVG when the widget supports it.
3. Include axis labels, units, split labels, and sample counts.
4. Record selection and threshold choices in the project report.

Generic Orange visualization widgets can also consume OWChemInf tables. Keep
chemical identifiers and SMILES fields as metadata so plotted selections can
be traced back to compounds.


Building Workflows
==================

Orange workflows pass data through connections between widget outputs and
compatible inputs. Drag a connection from the output side of one widget to the
input side of the next widget, then inspect the signal-selection dialog when
several connections are possible.

A robust molecular workflow usually has four stages:

1. Import and audit the source records.
2. Standardize structures and curate activities.
3. Calculate fingerprints or descriptors.
4. Search, analyze, model, validate, or export the result.

Example QSAR workflow
---------------------

.. code-block:: text

   ChEMBL Browser
       → QSAR Dataset Builder
       → Molecule QC Dashboard
       → Mol Standardizer
       → Mol Descriptors 2
       → QSAR Descriptor Explorer
       → Descriptor Pre-selector
       → QSAR/QSPR Model Hub
       → QSAR Validation Dashboard

Keep preprocessing choices identical for training, validation, and external
prediction branches. Connect report outputs to preserve curation and modeling
decisions alongside the final result.


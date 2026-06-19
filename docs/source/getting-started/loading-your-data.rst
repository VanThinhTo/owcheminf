Loading your Data
=================

OWChemInf accepts molecular data as Orange tables and as ``ChemMol`` object
collections. Most table-based widgets detect a text field named ``SMILES`` or
``canonical_smiles`` automatically.

Choose an importer
------------------

Use :doc:`../widgets/molecule-import-hub` for CSV, TSV, TXT, SMI, SMILES, SDF,
or SD files when you need row-level acceptance, rejection, and import reports.
Use :doc:`../widgets/sdf-reader` for a direct, lightweight SDF workflow. Use
:doc:`../widgets/chembl-browser` or
:doc:`../widgets/chembl-bioactivity-retriever` to retrieve public ChEMBL data.

Recommended first workflow
--------------------------

.. code-block:: text

   Molecule Import Hub
       → Molecule QC Dashboard
       → Mol Standardizer
       → Molecular Viewer

Check the import and QC reports before modeling. Invalid structures, duplicate
identities, mixtures, salts, activity units, and endpoint definitions can all
change downstream conclusions.


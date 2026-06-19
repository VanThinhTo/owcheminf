Exporting Models
================

:doc:`../widgets/qsar-model-hub` can train a prediction-ready model bundle.
Exported bundles include the fitted model, a manifest, the feature contract,
and selected-feature information.

Before exporting:

* confirm that validation metrics come from an appropriate split;
* record descriptor and standardization settings;
* verify the exact target definition and unit;
* keep the feature contract with the fitted model.

Use :doc:`../widgets/qsar-prediction-packager` to apply the model to an
external descriptor table and produce predictions, feature-alignment results,
failed records, and a package manifest.

.. warning::

   Serialized Python model files can execute code when loaded. Only open model
   packages from trusted sources.


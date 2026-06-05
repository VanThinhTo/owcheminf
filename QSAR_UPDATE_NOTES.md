# QSAR module update notes

Updated areas:

1. `chemcore/services/qsar_validation_dashboard_service.py`
   - Added additional validation metrics: Pearson r, CCC, slope, intercept, median absolute error, P95 absolute residual.
   - Added AD-aware validation flags when prediction tables contain `ad_in_domain`, `in_domain`, or similar columns.
   - Added `review_reason` and separate summary counts for large residuals, residual z-outliers, and outside-AD records.
   - Added `ad_coverage` to the validation summary.

2. `chemcore/qsar/applicability_domain.py`
   - Extended AD output table with leverage and distance ratios.
   - Added method-specific flags: `ad_in_leverage`, `ad_in_distance`.
   - Added residual scale diagnostics and confidence tiers (`high`, `medium`, `low`).
   - Added readable AD reason labels for reporting and triage.

3. `chemcore/services/qsar_report_generator_service.py`
   - Improved applicability-domain profiling in the report generator.
   - Report now summarizes AD coverage, out-of-domain counts, leverage/distance boundary ratios, confidence distribution, and top review reasons.
   - Validation dashboard summary is now integrated into the report narrative.
   - Added agreement diagnostics (CCC, slope, bias) when present.
   - Slightly improved generated HTML layout.

4. `widgets/ow_qsar_validation_dashboard.py`
   - Outlier table now displays AD flag, validation flag, and review reason.

Validation performed:

- Python syntax compilation passed for all changed files.
- Service-level smoke test passed for QSAR validation and report generation.
- Full Qt/Orange widget tests could not be run in this sandbox because `Orange` is not installed in the execution environment.

## v3 continuation: QSAR AD now uses AD Workbench engine

- Replaced the simplified QSAR-only Applicability Domain implementation with the richer `ad_workbench_service` scoring path.
- QSAR Applicability Domain output now includes Workbench-compatible diagnostics:
  - Williams leverage: `AD_leverage`, `AD_h_star`, `AD_leverage_ratio`, `AD_in_williams`
  - kNN domain: `AD_knn_dist`, `AD_knn_threshold`, `AD_knn_ratio`, `AD_in_knn`
  - Mahalanobis domain when the transformed model feature space is compact enough: `AD_maha_d2`, `AD_maha_threshold`, `AD_maha_ratio`, `AD_in_mahalanobis`
  - combined flag: `AD_in_domain`
  - method counts: `AD_enabled_methods`, `AD_failed_methods`
  - interpretability fields: `AD_margin`, `AD_confidence`, `AD_reason`
- Added lowercase aliases (`ad_in_domain`, `ad_outlier`, `ad_confidence`, `ad_reason`, `ad_leverage_ratio`, `ad_distance_ratio`, `ad_mahalanobis_ratio`, `ad_margin`) so the QSAR Validation Dashboard and report generator can consume QSAR AD tables directly.
- Fixed QSAR regression result assembly so external prediction sets are added to the result before the AD table is generated. External rows can therefore appear in the QSAR Applicability Domain output.
- Kept Williams + kNN always active for regression AD. Mahalanobis is enabled only for compact transformed feature spaces to avoid unstable high-dimensional covariance diagnostics.

## v4 — QSAR reporting visual upgrade

- Added optional Plotly-based interactive visual analytics to the QSAR report HTML export.
- New interactive report figures are generated when the required columns are detected:
  - observed vs predicted scatter plot with split-aware traces, hover labels and y=x reference line;
  - residuals vs predicted plot with residual histogram and ±2 SD guide lines;
  - grouped validation-metric bars for R²/Q², RMSE, MAE and CCC when available;
  - applicability-domain boundary map using leverage ratio vs kNN distance ratio;
  - AD ratio distribution box plots and AD confidence distribution;
  - top descriptor / feature-importance horizontal bar chart.
- Plotly is treated as an optional dependency. If it is unavailable, report generation still succeeds and explains how to enable interactive graphs.
- Added `interactive_graphs` metadata to the QSAR report summary JSON.
- Added service-level regression test for interactive Plotly report generation.

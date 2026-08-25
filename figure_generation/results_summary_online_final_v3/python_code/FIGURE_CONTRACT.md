# Figure contract: locked one-to-one event evaluation

## Core conclusion

Under deterministic maximum-cardinality one-to-one matching over any
positive-duration temporal overlap, each predicted event and each catalog event
can contribute to at most one true positive.  This removes the degeneracy in
which one overlong prediction could be credited repeatedly.  With the
validation-locked static thresholds and the single common SAOCP configuration,
the held-out test results must be shown exactly as obtained: SAOCP improves F1
for all four BCE backbones and matched BCE feature/window ablations, while the
relative effect of magnetic pressure and sequence length remains
configuration-dependent.

## Archetype and backend

- Archetype: quantitative grid for the global comparisons; aligned time-series
  and event-interval grid for the selected example.
- Backend: Python/matplotlib only.
- Final width: 183 mm (double column).
- Vector outputs: PDF and SVG with editable text.
- Raster outputs: PNG and LZW-compressed TIFF at 600 dpi.
- Color is redundant with printed values, line style, row names, and event-bar
  positions; no conclusion depends on color alone.

## Panel map

1. `Figure_backbones_static_vs_saocp_one_to_one`
   - One quantitative matrix: LSTM, CNN-LSTM, U-net, and RU-net.
   - Paired Static/SAOCP rows; columns are precision, recall, and F1.
   - Printed values and within-run delta F1 are mandatory.
2. `Figure_ablation_Pm_window_one_to_one`
   - Panel a: 33 channels without Pm versus 34 channels with Pm at 64 samples.
   - Panel b: 32, 64, and 128 samples with 34 channels.
   - Paired Static/SAOCP rows; printed P/R/F1 and delta F1 are mandatory.
3. `Figure_four_models_eventXX`
   - Panels a-d: LSTM, CNN-LSTM, U-net, and RU-net for one common catalog window.
   - Each panel shows the unchanged frozen probability, the newly locked static
     threshold, the newly locked SAOCP threshold series, and Catalog/Static/SAOCP
     event bars.
   - Insets state that P/R/F1 are full-test values, not scores for the one shown
     case.

## Source data and metric provenance

- Test metric source: `_submission_code_candidate/locked_one_to_one_results/test_event_metrics.csv`.
- Validation lock: `validation_selection_audit/VALIDATION_SELECTION_COMPLETE.json`.
- Locked-test audit: `locked_one_to_one_results/LOCKED_TEST_AUDIT.json`.
- Locked decisions: `locked_one_to_one_results/locked_decisions/{run}/`.
- Frozen probabilities/time arrays are accepted only when their SHA-256 values
  match `test_input_hashes.csv`.
- Matching: deterministic maximum-cardinality one-to-one assignment over
  positive-duration overlap edges.
- False-positive rule: every unmatched constructed prediction is a false
  positive.
- P, R, and F1 are independently recomputed from integer TP, FP, and FN before
  plotting.
- Seed: one frozen training seed (42) per run; no uncertainty interval is
  claimed.

## Representative-window rule

The example window is chosen post hoc for display after the locked test outputs
exist, using the prespecified high-contrast rule from the earlier figure
workflow: catalog isolation of at least 48 h; SAOCP coverage of at least 0.5 in
every backbone; positive coverage gain in every backbone; no unrelated
constructed event in the fixed plus/minus 12 h display; ranking by minimum
gain, minimum SAOCP coverage, mean gain, and then earliest UTC.  All 230 catalog
events are retained in `selection_audit.csv`.  This is post hoc display
selection, not a validation-independent sample.  The selected window is an
illustration and is not used to compute, tune, or select the global test
metrics.

## Review risks and required disclosures

- Single seed: comparisons are descriptive; no variance or significance claim.
- Representative case: selected post hoc for display on the locked test period
  using the prespecified high-contrast rule, so the caption must not present it
  as a typical, random, or validation-independent event.
- Boundary quality: any positive temporal overlap establishes an eligible match;
  onset/offset accuracy and boundary agreement are not measured by the reported
  F1.
- The event example cannot be used to claim global superiority; the quantitative
  matrices carry that evidence.
- Static thresholds and the common SAOCP configuration must match the pre-test
  validation lock exactly; no test-time re-selection is permitted.

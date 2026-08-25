# Code and result QA report

Final clean-copy QA date: 2026-08-25 (Asia/Shanghai)

## Outcome

The candidate code package implements the locked primary protocol: gap-aware
event construction, any-positive-overlap candidate edges, deterministic
maximum-cardinality one-to-one matching, and all unmatched constructed
predictions counted as FP.  The historical many-to-many/2.5 h evaluator is
retained only as a clearly marked legacy comparator.

In manuscript-facing results, **Online** denotes sequential decision-threshold
calibration with a frozen probability backbone.  The internal method value
`saocp`, related filenames, and `src/saocp.py` identify the study-specific,
SAOCP-inspired implementation used for that condition.  They do not denote an
included upstream reference release or a claim that SAOCP itself is new here.

Validation selection completed before the test evaluator was run.  The
validation lock records `test_files_opened=false`; its input manifest contains
only the catalog and `probability_val.npy`, `Y_val_aligned.npy`, and
`time_val.npy` for the eight frozen runs.  The locked selections are static
thresholds `0.80, 0.82, 0.90, 0.90, 0.66, 0.62, 0.64, 0.66` in the documented
run order and shared Online-implementation `coverage=0.95`, `lifetime=16`.  No
candidate was added after observing test results.

## Automated checks

- Python byte compilation: passed for all eight source modules and the test
  module.
- Unit tests: 10/10 passed, including the reporting-label/implementation-ID
  distinction.
- Locked-result arithmetic: 16/16 rows passed TP/FN, TP/FP, precision, recall,
  and F1 checks.
- Compact/source equality: both compact result tables exactly match the locked
  evaluator outputs on run, method, counts, precision, recall, and F1.
- CNN--LSTM command-line import smoke test: recorded as passed under TensorFlow
  2.20.0 in the original full-environment QA.
- Static validation selector reproduced its complete pre-registered grid and
  locked tie-break order.
- The Online implementation's validation selector evaluated exactly the
  pre-existing 14 coverage by 4 lifetime candidates and wrote the full
  macro/per-model grids.
- All-positive regression: passed.  The actual all-positive test label stream
  produced 143 gap-aware events and 77 TP, never more TP than predictions.  A
  single prediction spanning the full test interval produced exactly one TP,
  229 FN, and F1 0.008658.
- Matching cardinality: every evaluation was cross-checked by a second,
  independent augmenting-path implementation.
- Path scan: no user-specific absolute path is embedded in source, configs, or
  README commands.
- Third-party boundary scan: no complete Chen U-net/RU-net, upstream common
  evaluator, or `online_conformal` implementation is included.
- Recursive output hashes: present for all validation artifacts and 48 locked
  test artifacts, including all 40 per-run decision/threshold/event files.

## Locked primary test F1

The Online column corresponds to the internal `saocp` implementation ID.

| Run | Static | Online |
|---|---:|---:|
| LSTM | 0.602888 | 0.729958 |
| CNN--LSTM | 0.567863 | 0.740741 |
| U-net | 0.685841 | 0.754564 |
| RU-net | 0.658824 | 0.687090 |
| CNN--LSTM f33/w64 | 0.673961 | 0.735294 |
| CNN--LSTM f34/w32 | 0.516484 | 0.760181 |
| CNN--LSTM f34/w64 | 0.567863 | 0.740741 |
| CNN--LSTM f34/w128 | 0.601329 | 0.724576 |

All principal and ablation rows use BCE. The common Online configuration was
selected on the four BCE validation streams and transferred unchanged; the
34-feature, 64-observation row is shared across the benchmark and ablations.

## Recorded full environment

- Python 3.13.5
- NumPy 2.3.3
- pandas 2.3.2
- SciPy 1.16.2
- scikit-learn 1.7.2
- PyArrow 21.0.0
- TensorFlow 2.20.0

## Final clean-copy recheck

The final copy was rechecked locally with Python 3.8.10, NumPy 1.24.4, pandas
2.0.3, SciPy 1.10.1, scikit-learn 1.3.2, and PyArrow 17.0.0.  Byte compilation,
all 10 unit tests, and all 16 compact-result arithmetic checks passed.  The
locked result and validation artifacts remain byte-for-byte identical to the
author source copy.  TensorFlow training, model inference, the external
backbones, and the validation/test searches were not rerun during this final
copy audit.

## Remaining limitations and release actions

- Model training was not repeated during final packaging; evaluation used the
  already frozen probability streams whose hashes are recorded outside this
  compact package boundary.
- Only training seed 42 is recorded.  The results are not a multi-seed
  uncertainty analysis.
- The Online implementation uses delayed block labels after each complete block
  is predicted.  This is label-causal at block level, not a label-free
  deployment experiment or a new proof of conformal coverage.
- Raw observations, model weights, full probability arrays, and third-party
  comparison-model training code are not redistributed.
- Before public archival, the authors must choose a public license and confirm
  rights for inherited team code, source data, catalog, and included derived
  decision arrays.

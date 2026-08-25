# CNN--LSTM + Online decision-calibration code for ICME event detection

This package contains the study's data preparation, CNN--LSTM probability
backbone, label-causal Online decision layer, locked one-to-one event
evaluation, and the audit artifacts needed to reproduce the reported counts.
The Online layer is implemented here by a study-specific, SAOCP-inspired
blockwise quantile-expert mixture.  Complete third-party comparison-model
implementations are deliberately excluded; see `THIRD_PARTY.md`.

## Terminology

- **Static** is the validation-selected fixed-threshold comparator.
- **Online** is the paper's reporting label for sequential decision-threshold
  calibration while the probability backbone remains frozen.
- **SAOCP** is the published strongly adaptive online conformal prediction
  framework that motivates the study's Online implementation.  This package
  does not include an upstream reference SAOCP release or claim numerical
  equivalence to one.
- The machine-readable method value `saocp` and filenames such as
  `saocp_labels.npy` are retained for traceability.  In manuscript-facing
  comparisons, those artifacts correspond to the **Online** condition.

## Primary event metric

Thresholded sample labels first pass through the fixed gap-aware constructor:

- correct runs of at most two observations;
- retain positive runs strictly longer than 48 observations;
- merge events separated by less than 12 h;
- never correct or merge across an observation gap longer than 30 min.

A predicted interval and catalog interval form a candidate edge when their
temporal intersection has positive duration.  A deterministic
maximum-cardinality bipartite matching then assigns at most one catalog event
to each prediction and at most one prediction to each catalog event.  TP is the
number of matched pairs, FP is **every** unmatched constructed prediction, and
FN is every unmatched catalog event.  There is no IoG/IoU cutoff and no 2.5 h
FP exemption in the primary metric.

`src/event_f1.py` also retains the old many-to-many, 2.5 h-filtered function as
an explicitly named legacy comparator for internal audit.  It is not used by
the primary configuration or result files.

## Leakage-controlled selection

Configuration selection and test evaluation are separate executables.

1. `src/select_validation_config.py` knows only the validation filenames.  It
   writes `validation_selection_audit/VALIDATION_SELECTION_COMPLETE.json` and
   hashes every validation input before any test file is opened.
2. `src/evaluate_locked_test.py` first verifies that lock and the validation
   hashes.  Only then does it open the test streams and evaluate each frozen
   configuration once.  It contains no test-time search or ranking step.

Static thresholds use the pre-existing grid 0.10--0.90 in increments of 0.02.
The locked validation tie-break is event F1, precision, recall, proximity to
0.5, then the larger threshold.  Selected thresholds are:

| Run | Threshold |
|---|---:|
| LSTM | 0.80 |
| CNN--LSTM | 0.82 |
| U-net | 0.90 |
| RU-net | 0.90 |
| CNN--LSTM, 33 channels, 64 observations | 0.66 |
| CNN--LSTM, 34 channels, 32 observations | 0.62 |
| CNN--LSTM, 34 channels, 64 observations | 0.64 |
| CNN--LSTM, 34 channels, 128 observations | 0.66 |

The common configuration search for the SAOCP-inspired Online implementation
uses only the original candidate set: coverage `0.30, 0.35, ..., 0.95` and
lifetime `4, 8, 16, 32`, with `positive_singleton`, 64-observation blocks, and
a 128-block validation warm-up.  The four BCE backbones select
`coverage=0.95`, `lifetime=16` by macro validation event F1.  Ties are resolved
by minimum backbone F1, macro precision, macro recall, and fixed lexical
candidate order.  This pair is transferred unchanged to every Pm/window
ablation.  The selection has macro validation F1 0.796685 and
minimum-backbone validation F1 0.761905.

## Locked test results

The primary four-backbone event F1 changes are (Online is the SAOCP-inspired
implementation described above):

| Backbone | Static | Online | Difference |
|---|---:|---:|---:|
| LSTM | 0.602888 | 0.729958 | +0.127070 |
| CNN--LSTM | 0.567863 | 0.740741 | +0.172878 |
| U-net | 0.685841 | 0.754564 | +0.068723 |
| RU-net | 0.658824 | 0.687090 | +0.028266 |

The Pm/window audit uses the same BCE training family as the principal
benchmark. The 34-channel, 64-observation row reuses the principal CNN--LSTM
checkpoint and probability stream:

| Configuration | Static | Online | Difference |
|---|---:|---:|---:|
| 33 channels, 64 observations | 0.673961 | 0.735294 | +0.061334 |
| 34 channels, 32 observations | 0.516484 | 0.760181 | +0.243697 |
| 34 channels, 64 observations | 0.567863 | 0.740741 | +0.172878 |
| 34 channels, 128 observations | 0.601329 | 0.724576 | +0.123247 |

At 64 observations, adding `Pm` changes Static F1 from 0.673961 to 0.567863 and
Online F1 from 0.735294 to 0.740741.  This interaction is mixed and does not
support an unconditional claim that `Pm` improves every decision method.
Among the 34-channel Online window audit, 32 observations gives the largest
F1, while 64 observations remains the reference setting for earlier-model
comparability and lower latency/compute; it is not claimed to be globally
optimal.

## Included audit artifacts

- `validation_selection_audit/`: compact Static and Online-implementation
  validation grids, selected metrics, per-event validation matching, blockwise
  thresholds, input hashes, output hashes, and the selection lock. Redundant
  per-observation NumPy threshold arrays are omitted from this GitHub package.
- `locked_one_to_one_results/`: compact test metrics, TP/FP/FN event records,
  all-positive regression test, and input / output hashes. The 30-MB blockwise
  threshold table and redundant per-observation decision arrays are omitted;
  they can be regenerated with `src/evaluate_locked_test.py` from authorized
  frozen probability streams.
- `results/`: compact manuscript-facing result extracts.
- `configs/`: human-readable locked protocol, thresholds, and feature schema.
- `training_run_records/`: BCE model configurations, training histories,
  runtimes, and model summaries for the shared 34/64 baseline and the three
  matched ablation runs. Large weights and probability arrays are not bundled.
- `figure_generation/`: plotting scripts and compact source-data tables. The
  generated PDF/SVG/PNG/TIFF files are omitted because the scripts reproduce
  them and the publication package already contains the final figures.

See `LARGE_ARTIFACTS.md` for the exact exclusion policy. No reported metric in
the manuscript-facing CSV tables was changed when preparing this compact
repository.

The all-positive regression passes the same gap-aware constructor and enforces
that one prediction cannot receive more than one TP.  Matching cardinality is
also cross-checked by an independent augmenting-path implementation on every
evaluation.

## Study code

- `src/prepare_data.py`: reads the source Parquet table, appends `Beta`, `Pdyn`,
  `RmsBob`, and magnetic pressure `Pm`, makes chronological splits, and fits the
  scaler on 1998--2009 training observations only.  `Pm[Pa] = 1e-18 * B[nT]^2 /
  (2 * mu0)`.
- `src/cnn_lstm_pipeline.py`: trains the CNN--LSTM with selectable feature and
  window settings and BCE by default, and exports frozen validation/test
  probabilities without threshold tuning.
- `src/saocp.py`: implements the study-specific SAOCP-inspired expert mixture
  used by the Online layer.  It emits the threshold for a complete
  64-observation block before that block's labels update the next-block state.
  It changes decisions, not backbone probabilities or network weights.
- `src/event_f1.py`: gap-aware construction, deterministic matching, primary
  P/R/F1, legacy audit comparator, and command-line evaluation.
- `src/select_validation_config.py`: validation-only selection and lock for
  Static thresholds and the shared Online-implementation configuration.
- `src/evaluate_locked_test.py`: lock-verified one-time test evaluation.
- `src/verify_locked_results.py`: arithmetic audit of compact TP/FP/FN tables.

## Environment and commands

The recorded environment is Python 3.13.5 with exact versions in
`requirements.txt`.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Prepare data and train the study's CNN--LSTM:

```bash
python src/prepare_data.py \
  --parquet /authorized/data/datasetWithSpectro.parquet \
  --catalog /authorized/data/listOfICMEs.csv \
  --output-dir work/prepared_data

python src/cnn_lstm_pipeline.py all \
  --data-dir work/prepared_data \
  --run-dir work/cnn_lstm_f34_w64 \
  --window 64 --feature-count 34 --loss bce \
  --steps-per-epoch 100 --validation-batch-size 32
```

For the matched BCE window ablation, use 200, 100, and 50 steps per epoch for
32, 64, and 128 observations, respectively. This keeps nominal observation
exposure per epoch equal; the 64-observation run above is shared rather than
trained again.

Run validation-only selection by supplying all eight frozen run directories:

```bash
python src/select_validation_config.py \
  --catalog /authorized/data/listOfICMEs.csv \
  --run lstm=/authorized/runs/lstm \
  --run cnn_lstm=/authorized/runs/cnn_lstm \
  --run unet=/authorized/runs/unet \
  --run runet=/authorized/runs/runet \
  --run cnn_lstm_f33_w64=/authorized/runs/f33_w64 \
  --run cnn_lstm_f34_w32=/authorized/runs/f34_w32 \
  --run cnn_lstm_f34_w64=/authorized/runs/f34_w64 \
  --run cnn_lstm_f34_w128=/authorized/runs/f34_w128 \
  --output-dir work/validation_selection_audit
```

After inspecting the completed validation lock, run the separate test command
with the same eight `--run` arguments:

```bash
python src/evaluate_locked_test.py \
  --catalog /authorized/data/listOfICMEs.csv \
  --validation-lock-dir work/validation_selection_audit \
  --run lstm=/authorized/runs/lstm \
  --run cnn_lstm=/authorized/runs/cnn_lstm \
  --run unet=/authorized/runs/unet \
  --run runet=/authorized/runs/runet \
  --run cnn_lstm_f33_w64=/authorized/runs/f33_w64 \
  --run cnn_lstm_f34_w32=/authorized/runs/f34_w32 \
  --run cnn_lstm_f34_w64=/authorized/runs/f34_w64 \
  --run cnn_lstm_f34_w128=/authorized/runs/f34_w128 \
  --output-dir work/locked_one_to_one_results
```

Verify the compact tables and unit tests:

```bash
python src/verify_locked_results.py \
  results/backbone_event_f1.csv \
  results/pm_window_ablation_event_f1.csv
python -m unittest discover -s tests -v
```

## Required inputs and limitations

The source solar-wind observations, catalog redistribution rights, model
weights, and full probability arrays are not bundled.  End-to-end reproduction
therefore requires authorized copies of those inputs.  The package records a
single training seed (`42`) and does not claim a multi-seed uncertainty study.
The Online implementation uses delayed catalog labels after a block has already
been predicted; it is label-causal at block level but is not a label-free
deployment experiment.  This package does not establish formal conformal
coverage or numerical equivalence to an upstream SAOCP reference release.

Before public archival, the authors should confirm redistribution permission
for the included derived decision arrays, choose an appropriate public code
license, and confirm rights to any line-level code inherited from earlier team
work.

# Large and generated artifacts excluded from the GitHub package

This compact repository keeps the study code, locked configuration, tests,
training records, compact validation/test audit tables, figure-generation
scripts, and figure source-data tables. It intentionally excludes redundant or
regenerable files that are unsuitable for GitHub's browser uploader.

Excluded from the full internal package:

- `locked_one_to_one_results/test_saocp_threshold_blocks.csv` (about 30 MB);
- `locked_one_to_one_results/locked_decisions/**` NumPy label, threshold, and
  radius arrays;
- `validation_selection_audit/validation_threshold_series/**` NumPy arrays;
- generated figure files in PDF, SVG, PNG, and TIFF formats;
- Python bytecode and cache directories;
- model weights, full probability arrays, and source solar-wind observations,
  which were already not redistributed in the internal package.

The excluded decision and threshold artifacts are outputs rather than source
code. They can be regenerated with the documented validation-selection and
locked-test commands when the authorized catalog, prepared data, frozen model
outputs, and model weights are available. The compact CSV files under
`results/`, `validation_selection_audit/`, and
`locked_one_to_one_results/` retain the reported metrics and the evidence used
for the manuscript tables.

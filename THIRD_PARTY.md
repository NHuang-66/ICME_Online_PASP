# Third-party and upstream boundaries

## Excluded source code

The following files or implementations from the larger working directory are
not part of this submission candidate:

- `chen2022_Unet_model_original.py` (the supplied Chen et al. U-net/RU-net
  source snapshot);
- the unified LSTM, U-net, and RU-net training implementation used to create
  comparison probability streams;
- complete third-party `online_conformal` repositories or build directories;
- raw solar-wind data, catalogs whose redistribution terms have not been
  confirmed, trained model weights, and full frozen probability arrays.

The result directories contain derived counts, decisions, threshold sequences,
and event intervals, but no comparison-model source or raw probability stream.
The authors should still confirm redistribution permission for these derived
arrays before a public repository release.

## Scientific antecedents

- The CNN--LSTM architecture follows the team's earlier ICME detector described
  by Li et al., *A Novel Convolutional Neural Network--Long Short-term Memory
  Model for Interplanetary Coronal Mass Ejection Detection*, ApJS 279(1), 2025.
- The U-net/RU-net comparison is associated with Chen et al., *RU-net: A
  Residual U-net for Automatic Interplanetary Coronal Mass Ejection Detection*,
  ApJS 259(1), 8 (2022), DOI: 10.3847/1538-4365/ac4587.
- SAOCP is an existing strongly adaptive online conformal prediction framework;
  the manuscript bibliography should be used for its method citation.  The
  paper's **Online** condition is the authors' application of sequential
  threshold calibration to frozen ICME probability backbones.  The code in
  `src/saocp.py` is a study-specific, SAOCP-inspired blockwise implementation,
  not an included upstream reference release or a claim that SAOCP itself was
  invented in this work.

These acknowledgements do not determine source-code licensing.  Before public
release, the authors must confirm whether any line-level code inherited from
earlier team work requires a copyright notice or separate license.

## Python dependencies

This package imports NumPy, pandas, SciPy, scikit-learn, PyArrow, and TensorFlow.
They remain governed by their own licenses and are installed separately through
`requirements.txt`; no dependency source code is vendored here.

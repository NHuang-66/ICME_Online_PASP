# Figure 1 v5 contract

- Core conclusion: The fixed probability backbone is followed by an online decision layer that updates its blockwise threshold from completed-block errors, after which gap-aware event construction and maximum-cardinality one-to-one matching provide event-level evaluation.
- Figure archetype: schematic-led composite with panel b as the hero panel.
- Target/output: PASP/AASTeX double-column figure, 183 mm x 132 mm; editable SVG/PDF plus high-resolution RGB PNG/TIFF.
- Backend: Python/Matplotlib only.
- Panel map:
  - a, `Generate ICME probabilities`: chronological solar-wind inputs enter a fixed sequence model and produce a probability stream. The old `CNN-LSTM` box heading and header lock are absent; the centred `fixed model weights` label retains generous whitespace, and the input is labelled consistently as a `64-observation window`.
  - b, `Adapt the threshold online`: a distinct 8 x 8 observation card visualizes the 64-observation block without repeating the probability curve; experts use completed-block errors, decisions precede current labels, and the update applies to the next block. `Online implementation: SAOCP` appears once; the study contribution is explicitly the application of online decision updating under drift.
  - c, `Construct ICME events`: enlarged discrete decisions undergo concise gap-aware before/after correction, become enlarged event intervals, and are compared through enlarged maximum-cardinality one-to-one links for event-level precision, recall, and F1. Redundant block labels and the explanatory matching headline are removed from the artwork.
- Evidence hierarchy: panel b is the main methodological contribution; panels a and c establish the unchanged backbone and the auditable event-level output protocol.
- Reviewer risks: no claim that SAOCP was invented in this study; no online weight update; no current-block label leakage; no reuse of the probability/threshold trace in panel c; no many-to-one event credit; schematic glyphs are not presented as data.

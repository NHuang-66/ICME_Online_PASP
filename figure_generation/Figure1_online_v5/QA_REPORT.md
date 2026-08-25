# Figure 1 v5 QA report

- Scope: candidate figure only; no manuscript source or earlier candidate was modified.
- Backend: Python/Matplotlib exclusively for drawing, previewing, exporting, and QA.
- Panel-a audit: the lock preceding `fixed model weights` was removed; the retained label is centred in its own header space and does not overlap the model glyph.
- Panel-b audit: the stage-1 lock was retained. Its full visual centre, including the shackle, is aligned numerically with the vertical centre of `labels hidden` at y = 41.85.
- Panel-c reduction audit: the small block labels were removed from c1; c2 retains only `before`, `after`, and `fix short glitches · preserve gaps`; c3 retains the enlarged event capsules and `merge nearby runs`; c4 removes its lock and the `maximum-cardinality matching` artwork line, retaining the enlarged one-to-one graph plus two metric-explanation lines.
- Logic audit: the enlarged c4 node-link diagram still represents maximum-cardinality one-to-one event matching, and the suggested caption states the full technical term. Gap-aware correction and event-level precision, recall, and F1 remain unchanged.
- Claim audit: `Online implementation: SAOCP` occurs exactly once in the artwork; the contribution remains the Online application under distribution drift.
- Automated containment and collision checks embedded in the script passed: every visible text bounding box remains inside the canvas, with zero text-pair intersections above 10% of the smaller label area.
- Full-resolution visual QA passed: zero text/frame overlap, zero icon/text overlap, and clear whitespace around all effective labels.
- Bottom safety margin: `Precision · Recall · F1` remains approximately 7.24 pt above the frame border.
- Physical size: 183 mm x 132 mm; PDF MediaBox 518.7402 pt x 374.1732 pt.
- PNG: 3242 x 2338 px, RGB, approximately 450 dpi.
- TIFF: 4322 x 3118 px, RGB, 600 dpi, LZW compression.
- SVG: 87 editable text nodes, no embedded raster image, one occurrence of `SAOCP`, and zero occurrences of the deleted matching headline.
- PDF: no Type 3 fonts.


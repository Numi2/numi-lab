# Publication 024 independent byte audit

This is a read-only verification of the published 024 archive against its 023 prefix and the prepared v2 inventory. It checks source and archived payload hashes (including decompression), summary-prefix identity, cumulative pathspec bytes, source142 selected-source binding, and six-axis verdict identity. It does not modify the publication or claim physical qualification.

Run with `/Users/home/.pyenv/versions/3.10.12/bin/python3 audit.py`. The verified binding is `publication-binding-reference024.json` SHA-256 `635ec94166fa02e415dfaa8df7be54a7fb78e0abc4c4661a912d449c58cd2d29`. Results are in `report.json`.

The first consumer attempt compared the binding verdicts with `summary.qualification`, which is explanatory prose; the correct verdict object is `summary.verdicts`. This was an audit-script defect, not a publication mismatch. That attempt and the subsequent bytes-vs-Path report-assembly error are preserved with their original scripts and receipts. The corrected audit found exact equality between the binding and `summary.verdicts`.

The 024 suffix has no forbidden source/base/build/cache copies. Two `base/` paths from the 023 prefix remain as historical exceptions and were verified unchanged; they were not rewritten or silently excluded.

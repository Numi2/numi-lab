# P176: finite-pair SAT routing trial

P176 follows P175 at main commit `9aa8b3ad02826cb73a4445664bf15cd083cdd961`. This package records a tested selector/adapter trial on the 697-crossing state and its exact construction checks.

## Results

The selector chose 680 of 697 typed crossing pairs. The fit solved and emitted a proposal; after the existing 582 retention, the full checks found 725 surface pairs, 0 contained pairs, 0 self pairs across all 185 banks. This is a regression against the 697-pair baseline and is retained as a negative result.

A separate component-merit extraction produced 695 surface pairs, 0 contained pairs, 0 self pairs, with all 185 banks checked. Its only bankwise change against the 697 baseline is -2 in bank 114/117. This is an unadmitted construction candidate. All evidence records `accepted_physical_steps=0` and `native_admission=false`; no physical qualification is claimed.

The source and test suite are under `sources/component-sat-selector-638/`. Selector, fit, initial failed launches, exact 130/159 reports, and lossless report restoration proof are under `evidence/component-sat-selector-638/`. The compressed full fit report is independently restorable to the source SHA in the package manifest.

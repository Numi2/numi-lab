# Native strict child selection — evidence package

This package prepares a narrow source publication against Lab parent `06fef224d0e9b9af684956bb22cc0ccc3e9aadf3`. `manifest-input.json` limits the proposed source change to three owner files; `prepare_publication.py` is copied from publication-646 and is not run here.

The source-only census changed strict-fail rows from **31,916** (selector-007 baseline) to **24,354** on 56,054 detail rows. The compressed current census is `detail-embedding-census.csv.gz`; its full restore hash is verified in `evidence/census-gzip-restore-receipt-010.json`. The remaining native rejection is the declared effective tet-weight bound. This count is mapping evidence, not physical qualification.

The 010 focused check and the independent check receipt both pass with assertions active. The parity receipt reports zero mismatches in the historical first 45 columns across all 56,054 rows. The prior replay-002 producer-face/native-selector mismatch is retained in `evidence/prior-selector-mismatch-receipt.json`; later selection advanced past that mismatch but still fails the weight bound.

All native work here was source-only: **runtime 0, accepted steps 0, physical admission false**. No physical or simulation qualification is claimed. The top-level `root-focused-check.json` belongs to the independent check and was only read for this package.

The independent check was also recompiled against the exact published importer header. All 31 local included headers match the proposed three-file tree or parent main; no unrelated dirty importer diagnostics are required. See root-published-focused-check.json and dependency-tree-parity.json.

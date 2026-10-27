# P164: matched refinement preparation and retained construction trial

P164 extends P163 commit 77f648a774f5d83d7f8092b4a191eccac85e6331 (manifest SHA-256 c1fa8d07ce4e56b6dc3a1ea7aa1b46d1f92b91c600edc5188f2fc8c2c5747af9). It records the frozen 603 component backtracking entry and its failed/filtered trials, 604 matched preparation, all retained 606 target versions and correction census, and 607 matched fit evidence. P162 and P163 source-pin indexes are carried as exact references; P164 resolves current source pins by included snapshot or exact SHA entry in those indexes.

## Construction evidence

- The 603 filtered baseline-599 affine attempt is retained with its unfiltered failed predecessor, filtering implementations, full audit reports, and 413 depth lineage. Its reported count is 2,629 surface crossings, 422 contained pairs, and 0 self pairs. It did not replace the prior published checkpoint.
- The 607 direct full-coordinate fit is a preserved failure: exact self-pair checking found 3,940 pairs.
- The 607 retained 582/P175 final is a complete exact construction check at 2,614 surface crossings, 392 contained pairs, 0 self pairs, 185 banks, and 237 anchors. Candidate 59e20ae8b6bb8a0768686f015a7180d89aeb23c05777ab5b437737a1e617885e is temporary and unselected; the prior P163 602 checkpoint remains the best published construction at 1,576/0/0.
- The 603 component helper receipt records six tests passed. 604 and 606 source versions, tests, correction census, preparation outputs, and stage receipts are retained with original run evidence.

These are offline construction results. native_admission=false and accepted_physical_steps=0 remain in force. No physical, anatomical, or native qualification is claimed. Active 608 cutting-plane work is excluded.

## Restoring the 607 fit report

The original 607 fit/report.json is retained on disk and packaged as fit-report.json.zst. Zstandard integrity, streamed decompression, and a fresh destination restoration passed exact SHA-256, byte-count, mode and mtime checks. The temporary restoration tree was removed. The original remains on disk until independent remote verification and a fresh consumer check.

    python3 evidence/compression-preservation/607-fit-report/restore_compressed_text_mirrors.py \
      --receipt evidence/compression-preservation/607-fit-report/receipt.json \
      --compressed-root <P164-package-root> \
      --destination-root <empty-restoration-root>

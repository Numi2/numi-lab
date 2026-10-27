# P163: completed 595 and 602 construction evidence

P163 adds the frozen 595 `run-002` branch and the completed 602 saved-field merge with fresh full checks. Its parent is P162, commit `e86aa13955400d36167c02a1975ee6779184ec75`, manifest SHA-256 `2f00a11832725cc0292e48e7174729ee490054a1260cd7c8d2f18b274cce4b02`. The parent manifest and source-pin index are preserved under `evidence/prior-publications/publication-162/`.

## Construction checkpoints

- 595 retained 582/545 candidate `b2366af63df68ba5796a6d54ed2af917a90cd99914cdf919b21ee9cee0b5a70e`: exact full check passed with 1,674 surface crossings, 8 contained node-bank pairs, 0 self pairs, 185 banks and 237 anchors. This is temporary, unselected point-continuation construction. The preceding joint-fit proposal and its failed/raw evidence are retained.
- 602 saved-field merge candidate `e7c2bfa57ab4e7311f85b441597141b8bcc840f7f51b147da43a73f254d0e9b3`: exact full check passed with 1,576 surface crossings, 0 contained pairs, 0 self pairs, 185 banks and 237 anchors. This is the lowest exact-count candidate in this publication scope; it remains unselected.
- The 595 test receipt records PASS. Its original bytes end with the literal characters backslash and `n` after the valid JSON object; P163 preserves the exact receipt and records this format note. The 602 safety suite records 4 tests passed.

These are offline construction results. `native_admission=false` and `accepted_physical_steps=0` throughout; no physical or anatomical completion is claimed. Active 603/604/606 and SCP-draft work is excluded.

## Restoring the two large 595 fit text files

The original 595 `run-002/fit-stage.log` and `run-002/fit/report.json` are stored as adjacent `.zst` files. Both archives have passed zstd integrity checks and streamed decompression to the exact original SHA-256 and byte count. Fresh restoration-helper tests with mode and mtime checks are pending until the bounded geometry fit releases scratch capacity; both original files remain present. Before publication, restore with:

```sh
python3 evidence/compression-preservation/595-run-002/restore_compressed_text_mirrors.py \
  --receipt evidence/compression-preservation/595-run-002/receipt.json \
  --compressed-root <P163-package-root> \
  --destination-root <empty-restoration-root>
```

The helper recreates the original relative paths under the destination. The original payloads are preserved exactly; no summary replaces either file.

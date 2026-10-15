# Rich regional-capture stride correction

This is a capture-only fix layered on the source-098 runtime plus the restart-augmentation-007 runtime overlay. Source 098's runtime is byte-identical to overlay 007's declared source base (`d2ea134b...`); applying the runtime portion of overlay 007 reproduces its augmented runtime SHA `b7d85d45...`, then `capture-stride.patch` produces `46e7208c...`.

The bug is in the rich RAP sidecar, not the runtime snapshot readback. With restart augmentation, `femRegionalEffectiveBasis` and `femRegionalAP` use environment-major storage with `baseModeCount + 1` rows per environment. The sidecar still declares compact `[environment, baseMode, node]` arrays. A single contiguous copy incorrectly includes environment 0's padded restart mode before environment 1's base modes. The patch records validated source segment size/stride/count for these two arrays and emits one blit per environment into the existing compact destination. Without augmentation, the same segmented blits are byte-equivalent to the former contiguous copy.

No solver, kernel, tolerance, coefficient, snapshot, capture schema, or physics behavior changes. The existing 32 MiB combined readback cap and payload dimensions remain unchanged. The public `RuntimeStateSnapshot` compaction is already correct and untouched.

## Reproduction and checks

Base snapshot:

`/Users/home/cardboard-implementation-20261014/snapshots/source-private-mortar-witness-098`

Required predecessor overlay:

`/Users/home/cardboard-implementation-20261014/analysis/restart-augmentation-two-env-007/source-overlay.patch`

Apply the `matter/src/runtime.mm` portion of the predecessor overlay to the exact base, then apply this package's `capture-stride.patch`. The reconstructed patched runtime SHA must equal the value in `source-overlay.json`.

Run the layout contract tests with:

```sh
python3 -m unittest discover -s /Users/home/cardboard-implementation-20261014/analysis/regional-rich-capture-stride-098-001/tests -v
```

The tests cover a two-environment padded mode array against the existing snapshot compaction, the unaugmented compact case, and source-level checks for validated segment metadata and segmented blits. These are CPU layout tests, not a runtime compile or native capture result. Objective-C++ candidate compilation and native rich-sidecar comparison remain pending for the integrating source build.

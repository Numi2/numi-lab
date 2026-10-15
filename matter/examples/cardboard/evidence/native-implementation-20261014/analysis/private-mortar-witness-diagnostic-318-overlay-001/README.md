# Private mortar CCD witness diagnostic overlay

This package is an observational overlay on frozen source 097. It adds a second, test-only Metal kernel that runs the same mortar partition and private CCD helpers on the already-uploaded fixture inputs, then exports the selected complement-cell pair and CCD witness. The production validator still runs first. The diagnostic kernel only reads the inputs and writes a separate audit buffer; it does not write contact status, candidate state, accepted history, or cohesive history. The `contact.metalinc` edits preserve the original boolean tests and return paths; they only retain the booleans/witness metadata for export.

The overlay was compiled, not executed on GPU:

```sh
cmake --build /Users/home/cardboard-implementation-20261014/analysis/private-mortar-witness-diagnostic-318-overlay-001/build-host \
  --target numi-matter-cardboard-full-contact-armijo-check -j2
```

The target completed successfully. Apply-check also passed against the frozen 097 snapshot:

```sh
git -C /Users/home/cardboard-implementation-20261014/snapshots/source-exact-mortar-seam-diagnostics-097 \
  apply --check /Users/home/cardboard-implementation-20261014/analysis/private-mortar-witness-diagnostic-318-overlay-001/overlay.patch
```

After integration, the checker emits `witness_audit` for every case. Rows are:

0. left/right partition cell counts, valid-partition flag, all-CCD-certified flag;
1. selected left/right cell indices and their covered patch IDs (`-1` means uncovered);
2. witness kind, paired-boundary validity, witness-on-boundary, common-translation;
3. nonapproaching flag, TOI, separation, authored patch count;
4–5. left and right witness barycentrics;
6. witness normal;
7. witness point. If partition construction itself fails, row 7 contains left/right partition failure stages and patch count. No GPU result is claimed by this package; the run of source 097 predates the diagnostic and contains no witness rows.

## Exact map audit for native 318 inputs

`mortar_exact_map_audit.py` decodes the retained `cohesive_faces_buffer_9` bytes from the native 318 fixture log, reads actual binary32 barycentric maps, widens each binary32 exactly to a rational, and performs exact convex clipping. It found:

- Three-patch fan: parent XY area `1/2`; patch areas `5592405/33554432`, `11184811/67108864`, `11184811/67108864`; pairwise positive-area intersections `0`; total area exactly `1/2`.
- Four-patch partition: four patch areas `1/8`; pairwise positive-area intersections `0`; total area exactly `1/2`.

Thus the captured planar x/y maps have no true complement sliver in exact rational geometry. The float32 clipping residue seen in the exploratory CPU clipping diagnostic is numerical clipping error, not an authored planar map gap. The three-patch center row's x/y/z weights sum to `33554433/33554432 = 1 + 2^-25`; that is a separate barycentric normalization discrepancy in the stored xyz row. The contact partition uses x/y and reconstructs z, so do not silently renormalize that source map or conflate the xyz sum discrepancy with a planar gap.

This input-geometry result does not explain the four source-097 rejections. The native 318 failures have contact diagnostics `[candidate 0,4, TOI 0, separation 0]`, indicating a first-touch event in the old run, but the precise selected partition cells/features were not captured. The new witness overlay is intended to localize that event. Keep the four mismatches as failures until a source-bound native run records the witness and the repair preserves the off-edge and cross-identity negative controls.

The native 318 raw log, receipt, prediction, and source-build record remain at `/Users/home/cardboard-implementation-20261014/native/exact-mortar-seam-stage-diagnostics-318/`; the exact-map audit is retained as `exact-map-audit.json`. `manifest.json` binds those files by SHA-256 and binds this patch to source 097.

# Selector-5 five-kernel arithmetic check

This folder holds a source-bound host harness and an independent full-patch arithmetic audit for native selector-5 coarse/patch kernels. Start with `lineage-erratum.md`; the JSON record `lineage-manifest.json` binds both the original missing-symbol run 209 and the fresh source-067 run 210. The original failed run remains at its own path and has not been overwritten.

Run 210 dispatched the five named production Metal kernels and returned native status zero. The all-patch audit is retained in `native-output-audit.json`; all 13 patch records are present. One arithmetic gate fails: patch 7's local solve residual is 1.394965249e-3 against the unchanged 1e-3 criterion. The FP64 diagnosis separates factorization and triangular-solve effects and is included per patch. The checker never drops later patch results after a failure.

The verifier's scalar reductions use `math.fsum`; the suite is run with warnings promoted to errors:

```sh
python3 -W error test_verify_kernel_output.py
python3 -W error verify_kernel_output.py \
  canonical182-input/selector5-kernel-input.bin native-output.bin \
  --bundle-manifest canonical182-input/bundle.json \
  --report native-output-audit.json
```

The second command returns exit 2 because the patch-7 threshold is not met. That is the expected honest verdict. The checker is a prepared-kernel arithmetic instrument, not a full solver or selector-5 FGMRES qualification. It bypasses production FEM patch assembly by feeding prepared matrices, and it is not material or physical validation.

`source-build-067.json` is the successful build receipt copied from run 210. `native-output.bin` and the 210 receipt are retained copies; `lineage-manifest.json` binds both copies and originals by hash. The metallib itself remains at the absolute path in that manifest; it is not duplicated here.

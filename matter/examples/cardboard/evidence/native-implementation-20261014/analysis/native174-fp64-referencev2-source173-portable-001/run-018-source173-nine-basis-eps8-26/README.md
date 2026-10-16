# Run 018: full nine-basis captured-center FP64 FD audit

This CPU-only run used the exact pre-run portable v2 module snapshot under
`portable-tools-used/matter/tools/` and the frozen source173 equation module.
The script hash is `e106dd3d42737b2421430ce4e1754f78c1c81fd859a6bf06d5f5ccfe5ac61e5c`.
It completed in 121.25 seconds; the main NPZ hash is
`626ad3fe6d6bd147b3b18e8d8a688d63a430a43af33c2aeb9ca4f103a8811808`.

Reproduction command from the package directory:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 qualify_nine_basis_source173.py \
  --source-reference /Users/home/cardboard-implementation-20261014/snapshots/source-checkpoint-warm-breakdown-173/matter/tools/cardboard_bilinear_fp64_reference.py \
  --capture /Users/home/cardboard-implementation-20261014/analysis/native469-bilinear-layer-replay-002/arrays.npz \
  --mp-arrays /Users/home/cardboard-implementation-20261014/analysis/native469-mp-fallback-005/run-002/mixed_reference_arrays.npz \
  --portable-tools /Users/home/cardboard-implementation-20261014/analysis/native174-fp64-referencev2-source173-portable-001/run-018-source173-nine-basis-eps8-26/portable-tools-used/matter/tools \
  --output /Users/home/cardboard-implementation-20261014/analysis/native174-fp64-referencev2-source173-portable-001/run-018-source173-nine-basis-eps8-26-reproduction-001
```

Bindings: source173 equation base SHA-256
`a910339573822bf32c7ea895c295a0a51b8b5a99d901b1e3ab1c788c094f6995`; native469
capture SHA-256 `4361b6aa9f0d269a9733b0bd8f508dbc1204624451611816512dad02c303d27c`;
independent per-tet parameter arrays SHA-256
`107e942cc2e131eed8f8db7759f8deb3e86fcdc42d8da249996daf51d6ed11ad`.

All 1,728 captured paper centers pass the unchanged `2e-12` scaled-residual and
`2e-11` dynamic-correction gates. The run tested each of the nine independent
entries of F on all 1,728 rows at epsilon powers 8, 10, …, 26. All 311,040
positive and negative endpoints pass the strict gates and retain the captured
center's lambda/y/representation signature. No branch crossing occurred in
this particular center set; plus and minus one-sided comparisons are still
recorded separately.

Using realized FP64 component displacements and the unchanged `1e-6`
relative-L2 derivative gate, 15,536 of 15,552 row/direction combinations have
at least two adjacent sampled epsilon points passing. Sixteen combinations do
not: four have no passing point on this grid and twelve have only one. Their
full epsilon-wise relative/absolute errors, reference-action norms, strict
metrics, iterations, branch signatures, realized displacements, and one-sided
errors are in `unresolved-row-directions.json`. These values are finite and
their reference actions and endpoint deltas are nonzero; an absolute numerical
noise floor was not independently measured. They remain unqualified at the
sampled-grid criterion, not silently passed or attributed to a measured floor.

The run is an independent FP64 local derivative check on captured states. It is
not a native Metal finite-difference result, coupon convergence result,
calibration, or physical box qualification.

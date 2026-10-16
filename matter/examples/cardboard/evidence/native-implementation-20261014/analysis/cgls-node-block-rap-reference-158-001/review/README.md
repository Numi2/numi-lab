# Independent review: node-block plus RAP CGLS CPU comparison

This review binds the immutable 158 numeric output without editing it. It checks the formula and its transpose against an explicit dense small nonsymmetric system, verifies the source/array bindings, and exercises invalid-factor rejection. It does not rerun the six-case CGLS experiment.

## Finding

The implemented map is `S = H + w (P - H A P) R^-1 P^T`, with `H` built from the diagonal 3x3 inverse blocks of the 419 free CSR operator. Its transpose is correctly composed as `S^T = H^T + w P R^-T (P^T - P^T A^T H^T)`: the code uses `R^-T`, the block inverse transpose, and the nonsymmetric coarse correction in reverse order. The synthetic test exercises the actual sealed `BlockRAP.apply` and `BlockRAP.transpose` methods with nonsymmetric `A` and `R`; max absolute dense-action error is at most `5.6e-17`. The invalid RAP reference solve and study wrapper reject an invalid factor.

The 464 capture is same-generation joined to the 419 CSR input by the independent 464 audit: candidate/accepted nodes, accepted history, tangent cache and validity, RHS, original correction, parameters, topology and free mask match. Captured `AP` versus 419 `A P` differs by `1.043e-7` relative L2; base RAP LU reconstruction versus 419 `P^T A P` differs by `1.227e-7` relative L2. The environment-0 base factor is marked valid and its pivots are a permutation.

The measured grid at 64 iterations reaches relative true residuals (FP64 / CPU FP32 arithmetic): `w=0`: 0.0884539 / 0.0897914; `w=0.1`: 0.0880093 / 0.0894090; `w=1`: 0.1006064 / 0.1135656. Every case reports `iteration_cap`, so these are conditioning comparisons, not solved systems. The small `w=0.1` change is marginal; `w=1` worsens the residual.

## Captured-map limit and numerical warnings

The capture says restart augmentation was enabled, but the consumer restart cycle was not recorded, so the augmented selector-5 view is explicitly unadmitted. This experiment uses the captured same-Newton base `P/AP/RAP` with a separately reconstructed block3 `H`; it does not establish the map consumed by CGLS or FGMRES. The `completed_tail_latch` clone has no role in the `BlockRAP` arithmetic, which never invokes graph Schwarz.

The raw result contains 3,300 NumPy RuntimeWarnings: divide-by-zero, overflow, and invalid-value in `matmul`. All serialized numeric outputs are finite, and the source recurrence checks its main iterates/actions for finiteness. A separate single-vector probe reproduces the warnings with finite map outputs; with `np.seterr(all='raise')`, `P.T @ x` raises a divide-by-zero `FloatingPointError`. The cause is unresolved. Warnings are retained and are not described as benign.

## Reproduction

The original run's exact command, supplied by the operator after the run, was:

```sh
cd /Users/home
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/cgls-node-block-rap-reference-158-001/run.py \
  > /Users/home/cardboard-implementation-20261014/analysis/cgls-node-block-rap-reference-158-001/attempt002-console.txt 2>&1
```

This invocation is parent-tool-supplied evidence; the original folder has no attempt002 command receipt. The output is bound by its hash in `manifest.json`. Attempt001's missing optional `threadpoolctl` import and failure before data read/recurrence remain preserved.

Run the added independent source tests with:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/cgls-node-block-rap-reference-158-001/review/test_review.py
```

The warning probe is separate and does not run CGLS:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/cgls-node-block-rap-reference-158-001/review/warning_probe.py
```

The manifest records hashes for the reviewed scripts, raw outputs, exact input payloads, and source/readers. No GPU, build, production source edit, or solver qualification is included.

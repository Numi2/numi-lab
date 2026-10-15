# Selector-5 patch solve roundoff audit

This is a CPU-only replay of the source-bound native kernel-check bundle. It does
not modify solver code and does not qualify the preconditioner or a full native
simulation.

## Inputs

- Prepared input bundle SHA-256: `b2619d5d7120f1e6cae40869af0a37ce14b86b29ce0a51961c95ec2c84c3de60`
- Native output from `selector5-coarse-patch-kernels-fresh067-canonical182-210`
  SHA-256: `2b391c0a6feb0c5a52c9cfb61ffdc12f7734a124d322338e136854076c15a6fd`
- Output verifier SHA-256: `0a4935689449d11a30d1600d585f88a6d62edc65d00feb2fcd2ea813bcb83028`
- The prepared inputs come from a selector-3 captured operator with a graph
  layout reconstructed from the source-bound topology. The selector-5 point
  kernels were run natively, but their full preconditioner action was not
  captured from the same production Newton generation.

## Finding

The native run completed, but the independent output verifier rejects patch 7:
its local residual is `0.00139497`, above the unchanged `1e-3` limit. The other
12 patches are below the limit.

Patch 7 has 256 nodes / 768 scalar unknowns and a source-matrix 2-norm condition
estimate of `381038.5`. Its stored FP32 LU reconstructs the source matrix with
relative Frobenius error `3.91e-7`, which passes the separate factor-reconstruction
check. The native solution is close to the literal FP32 sequential triangular
recurrence (relative solution difference `1.27e-5`), so the native solve appears
to be executing that recurrence as intended.

The CPU variants isolate the remaining error:

| Patch 7 solve path | Relative residual against source matrix |
| --- | ---: |
| Native correction, FP64 residual evaluation | `1.394965e-3` |
| FP32 sequential triangular dots | `1.425850e-3` |
| FP32 fused-like triangular dots | `1.314827e-3` |
| FP32 correctly accumulated dots, rounded once per row | `1.296836e-3` |
| FP64 triangular solve through the same FP32 LU | `1.296359e-3` |
| Direct FP64 solve of the source matrix | `2.13e-12` |

Thus better triangular dot accumulation helps, but does not bring patch 7 below
the gate. Even an FP64 triangular solve through the captured FP32 factors remains
over the threshold. This points to factor perturbation amplified by the patch's
conditioning as the first remaining issue; it is not explained by sequential
dot cancellation alone. The direct FP64 solve is a reference calculation, not a
candidate production change.

An offline iterative-refinement experiment using the original source matrix and
the captured FP32 LU as a correction solver brought every patch below `1e-3` on
its first correction. With FP32 `fsum`-style triangular corrections, the largest
first-correction residual across all patches was about `7.9e-5`. The original
matrix is preserved in this diagnostic input; production currently overwrites
the patch matrix with LU, so this result only motivates a separately bounded
native experiment. It is not a native refinement result.

## Reproduction

Run `python3 audit.py`. The script writes `report.json` and `per_patch.csv` with
all 13 patches and the input/output hashes. It distinguishes the native-like
FP32 sequential RHS from an FP64 `math.fsum` RHS, and keeps the offline
refinement result labeled as diagnostic-only.


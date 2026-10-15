# Bounded base-CGLS candidate generator (source 139 overlay)

This package contains a private, opt-in CGLS recurrence for a later runtime integration against the frozen source-139 FEM operator. The Metal helper only updates private vectors and per-environment status. It does not write production FGMRES state, physical histories, acceptance metrics, or certificates. The runtime adapter and final candidate installer remain parent-owned.

The method is unregularized truncated CGLS on the original possibly nonsymmetric operator, using separate actions `A p` and `A^T r`. A cap of 32 or 64 iterations is accepted. Breakdown, nonfinite values, invalid shape/masks, nonpositive denominators, and inconsistent environment mapping fail closed. The only supported vector layout is `U = 2 * E * N`: mechanics first, zero-valued field backing second. Quiet environments must have exact-zero RHS, transpose RHS, and row mask. Fixed rows must carry zero input and remain zero. The helper expects the runtime to construct private selected-environment RHS and free-row mask exactly; it validates rather than repairs nonzero quiet/field/fixed data.

## API

Nine kernels are added in `base_cgls.metalinc`; exact binding slots are in `core-v1/README.md`. Reductions dispatch `E` threadgroups of 32 threads. Vector updates dispatch `ceil(U/64)` groups of 64 threads. Initialization, iteration completion, and finalization dispatch environment-sized work. `ATb` and `s` may alias during initialization. The helper has no convergence tolerance and produces a bounded candidate only; the caller must freshly apply the production operator and independently check descent/line-search acceptance. It must never transfer an FGMRES/Newton convergence certificate to the candidate.

## Validation completed

- Six CPU recurrence/admission tests pass; they cover a nonsymmetric matrix with an explicit transpose, invalid caps and masks, constrained-row rejection, and exact-zero quiet-environment data.
- The source-bound capture-419 CPU replay runs caps 32 and 64. It is only a CPU FP32 recurrence reference: residual ratios were 0.2996 and 0.2355 respectively; both reached the iteration cap. It is not a native action parity or nonlinear-solve result.
- `native/compile_only.sh` compiles the Metal helper/test actions into a standalone metallib and runs Objective-C++ syntax checking for the direct-kernel checker.

No native checker executable was run and no GPU result is claimed. The direct checker’s sparse CSR A/A-transpose kernels are test-only and do not validate the production FEM operator. Production integration must bind the existing source-139 production forward/transpose actions, source hash, vector sizes, and environment masks, then run native parity and the existing nonlinear replay before any solver/performance claim.

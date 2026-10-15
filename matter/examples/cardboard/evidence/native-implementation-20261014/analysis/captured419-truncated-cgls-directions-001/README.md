# Truncated CGLS directions from captured 419

This package performs a bounded, CPU-only CGLS recurrence on the exact source125 unsymmetric CSR matrix A from captured Newton state 1:48. It starts from p=0 with lambda=0, applies A and A-transpose as sparse products without forming A-transpose-A, and exports both FP64-recurrence and explicitly emulated-FP32-recurrence directions after 8, 16, 32, 64, and 128 iterations. Every exported full E*N float4 direction is rounded to FP32 for the existing native replay helper, with quiet environment, fixed nodes, and w lane zero. Only the direction region of the exact 424 base replay payload changes.

Run the generator and independent output verifier with one CPU thread:

    python3 /Users/home/cardboard-implementation-20261014/analysis/captured419-truncated-cgls-directions-001/generate.py
    python3 /Users/home/cardboard-implementation-20261014/analysis/captured419-truncated-cgls-directions-001/verify.py

The bound CSR has shape 2538x2538 and 71,046 nonzeros. Its source125 capture, exact 424 base payload, base RHS, compiled package, source build, helper, and assembly module are SHA-bound in report.json. The generator validates those source inputs before creating outputs; the manifest hashes each exported direction and full replay payload.

## CPU linear results

The table gives the exact CSR residual norm after each direction has been exported to FP32. Ratios are relative to the captured RHS norm 1.8444380407784322e-4 Ns. The FP32 recurrence uses float32 sparse matvecs, vector updates, and deterministic sequential float32 dot accumulation. It is an arithmetic emulation, not the Metal GPU reduction order.

| Iterations | FP64 recurrence ratio | FP32-emulated recurrence ratio | FP64 direction norm (m/s) | FP32 direction norm (m/s) | Exported direction relative L2 difference |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.545846 | 0.545846 | 1.02419e-3 | 1.02419e-3 | 1.97e-5 |
| 16 | 0.407694 | 0.407697 | 1.68951e-3 | 1.68949e-3 | 1.60e-4 |
| 32 | 0.299390 | 0.300545 | 2.44377e-3 | 2.43343e-3 | 8.46e-3 |
| 64 | 0.230056 | 0.236494 | 3.43982e-3 | 3.30304e-3 | 7.24e-2 |
| 128 | 0.152613 | 0.160119 | 5.73140e-3 | 5.45401e-3 | 8.03e-2 |

All ten exported directions have positive g dot p (so the exact least-squares merit derivative at the origin is negative). The FP32-exported exact-CSR residual ratios at 32, 64, and 128 iterations are respectively 0.300545, 0.236494, and 0.160119 for the emulated FP32 recurrence; the FP64 recurrence followed by FP32 export gives 0.299390, 0.230056, and 0.152613. These are linear-model results only.

The discrepancy between FP64 and emulated FP32 directions grows after 16 iterations, consistent with loss of finite-precision conjugacy in this ill-conditioned problem. The 128-step norm difference is 8.03% even though their exact-CSR residual ratios remain relatively close. That is a warning against interpreting the recurrence emulation as a native Metal prediction.

## Scope

The directions are truncated least-squares proposals, not certified Newton corrections. No physical transaction, native CGLS solve, nonlinear residual replay, Armijo/CCD acceptance, or terminal certificate is present. The original Newton certificate does not apply. These files only justify which bounded native residual replays may be informative.

CGLS applies the normal equations and squares the conditioning. This CPU experiment does not show that CGLS is preferable to LSQR/LSMR or that it will improve the nonlinear residual. Any native candidate must be replayed through the source-bound production residual and the unchanged acceptance, rollback, and final certification path before it can be considered.

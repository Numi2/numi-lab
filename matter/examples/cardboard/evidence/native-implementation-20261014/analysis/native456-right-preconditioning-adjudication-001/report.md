# Native 456/457 nonlinear residual adjudication

This CPU-only analysis recomputes all 300 environment/sample residual norms from the immutable native outputs. The GPU helper evaluated candidate directions through production FEM preparation, geometry validation, internal-force projection, and mechanical-residual kernels. No physical timestep was advanced: both receipts record zero physical transactions.

## Binding and computation

456 and 457 each contain 10 successful native residual calls plus 10 successful payload preflights. There were no timeouts, all receipt command exit codes are zero, and both runners record that source/helper inputs were unchanged. Every output hash and byte count matches its capture manifest. The two packages use the same source125 build receipt, compiled world (SHA-256 `cafbf36052bbfa72e8be49fdba8b23e900469c9f15136a14a4ec1e4c0b5ccf7c`), native helper executable (SHA-256 `cc872337366821d0752665ea0feb913f38c1ff7cd921696c395c9861fabccbec`), dimensions `2×882 nodes ×1920 tets`, and `dt=9.999999747378752e-5 s`.

Every input payload has bitwise-identical candidate, RHS, accepted material history, cached tangent and validity flags, parameters, captured fresh action, tet audit, and accepted nodes across all 20 run payloads. Only the exported correction-direction segment differs. Alpha-zero residual bytes and candidate-node bytes are identical across all runs. The source and binary identities, all payload paths/hashes, and exact helper/source lines are in `source_bindings.json`.

The helper sets `metrics.z=epsilon`. It negates the exported direction only for `side=1`; therefore `side=0` is the positive correction direction. The production candidate kernel adds `alpha*correction` to velocity, then uses accepted displacement plus `dt*new_velocity` for reference-displacement FEM. It runs geometry validation, material/internal-force evaluation, and the production mechanical-residual kernel. The helper verifies that accepted nodes and accepted material history remain byte-identical after each sample.

The reported residual is the xyz norm of the FEM mechanical impulse residual (Ns): production computes `dt*(internal force + gravity + external force) - mass*(candidate velocity - accepted velocity)`. The helper sets `hasExternalForces=0` and this residual kernel does not include contact contributions. This is an isolated FEM residual comparison, not the full-contact Armijo merit. The fixed-row mask is taken from `restAndFixed.w`: exact values 1 or 3 are fixed and excluded; all other legal zero-tag rows are included. Each environment has 846 free rows and 36 fixed rows. In this captured payload all fixed rows are tag 1; none is tag 3.

For every one of the 300 samples, the two environment status codes are `[0,0]`, and the helper’s accepted-history-unchanged marker is true. All output floats are finite. Environment 1 has an all-zero direction segment and its node/residual buffers are bitwise unchanged across every sample; the measured change is in environment 0.

## Residual results

Norms are recomputed from the stored FP32 residual vectors using `math.fsum` for squared terms. The baseline alpha-zero free residual is `1.8444380407784322e-4 Ns` in env0 and `6.780005479517216e-8 Ns` in env1. Table entries are env0 free residuals in Ns; the complete alpha/sign table for both environments, status/history markers and output hashes is in `residuals.csv`.

| Direction source | alpha 1 | alpha 0.5 | alpha 0.25 | alpha 2 |
|---|---:|---:|---:|---:|
| None, 64 (`456/none_fp64_iter064`) | 4.838781318830856e-05 | 9.99125434217383e-05 | 0.00014124333674971027 | 0.00018966752760509153 |
| Column-norm, 64 (`456/column_fp64_iter064`) | 5.673587119092096e-05 | 9.44080138900689e-05 | 0.00013875283722164446 | 0.00019962364432955942 |
| Column-norm, 128 (`456/column_fp64_iter128`) | 6.0791158167957065e-05 | 9.247655351159943e-05 | 0.00013792906916774392 | 0.00022642792708980542 |
| Gram-block inverse square root, 64 (`456/block3_fp64_iter064`) | 5.887063016244249e-05 | 9.352039325826205e-05 | 0.00013805998554021083 | 0.000223970519908027 |
| Gram-block inverse square root, 128 (`456/block3_fp64_iter128`) | 6.0001224362274965e-05 | 9.448288025931009e-05 | 0.00013806765270856235 | 0.00020930396250694395 |
| Diagonal inverse Aii, 64 (`457/diag_Aii_fp64_iter064`) | 6.1015639958117306e-05 | 9.628383850277875e-05 | 0.00013921212318442059 | 0.00019552432144590332 |
| Diagonal inverse Aii, 128 (`457/diag_Aii_fp64_iter128`) | 6.0405562008313214e-05 | 9.421869194183823e-05 | 0.00013846761747064992 | 0.0002091846147032424 |
| Nonsymmetric 3x3 Aii inverse, 64 (`457/block_Aii_fp64_iter064`) | 5.8870549628624755e-05 | 9.35204809777023e-05 | 0.00013806001044507413 | 0.00022397046021537916 |
| Nonsymmetric 3x3 Aii inverse, 128 (`457/block_Aii_fp64_iter128`) | 6.000085839748998e-05 | 9.448288687294483e-05 | 0.0001380677180339502 | 0.00020930532080028699 |

At alpha 1, unpreconditioned CGLS64 gives the smallest env0 residual, `4.8387813e-5 Ns`. Column-norm64 gives `5.6735871e-5`, Gram-block64 `5.8870630e-5`, diagonal inverse Aii64 `6.1015640e-5`, and nonsymmetric 3×3 Aii inverse64 `5.8870550e-5`. All reduce the center residual at that tested positive alpha, but right scaling does not beat the unpreconditioned direction at alpha 1.

At alpha 0.5, right-scaled directions are modestly better than the unpreconditioned direction (`9.25–9.64e-5 Ns` versus `9.99125e-5 Ns`). At alpha 0.25, the block and column variants are also modestly better (`1.379–1.388e-4 Ns` versus `1.41243e-4 Ns`). At alpha 2, every positive direction is at or above the center residual; the block directions are substantially worse than the unpreconditioned direction. Every negative-side sample increases the residual above the center. The alpha grid’s lowest residual for each direction is alpha 1, among tested points.

Increasing CGLS from 64 to 128 does not improve the tested alpha-1 nonlinear residual for any compared family: column-norm goes from `5.67359e-5` to `6.07912e-5 Ns`, Gram-block from `5.88706e-5` to `6.00012e-5 Ns`, diagonal Aii inverse from `6.10156e-5` to `6.04056e-5 Ns`, and 3×3 Aii inverse from `5.88705e-5` to `6.00009e-5 Ns`. Some alpha 0.5/2 comparisons vary, so there is no uniform 128-iteration benefit. The 128-iteration candidates were generated offline and passed through the native residual helper; they are not production CGLS runs, since the source-bound native CGLS cap is 64.

## Recommendation and limits

These results do not justify enabling diagonal or 3×3 Aii right scaling by default. They show a valid FEM residual decrease at tested positive alphas, but the unpreconditioned direction is stronger at alpha 1 and lower-alphas only slightly favor some scaled candidates. Crucially, this helper omits contact force/residual terms and does not perform the full-contact selector, rollback, or nonlinear terminal certificate. Treat this as candidate-direction evidence for one captured state only. If pursued, evaluate the same candidates through full production Armijo with unchanged physical operator, history/rollback, and strict final-certificate gates against the unpreconditioned control.

The 146 studies also bind the direction generation: `column` is `S[j,j]=1/||A[:,j]||₂`; `block3` is per-free-node SPD `Sₙ=(AₙₙᵀAₙₙ)^(-1/2)`; 457 `diag_Aii` is signed `1/A[j,j]`; 457 `block_Aii` is the nonsymmetric principal-block inverse `Sₙ=(Aₙₙ)^(-1)` with adjoint recurrence applying `Sₙᵀ`. No operator shift or regularization was applied. The prior 146-001 report’s operator sign/unit typo is corrected in 146-002; this analysis consumes its exported direction bytes without changing the captured matrix.

This does not establish a complete coupon, physical calibration, or production convergence. It is a zero-transaction FEM residual replay at one captured state; 128 is exploratory; no full contact or rigid-body residual is present.

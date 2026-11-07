# All-path contact response containment (research only)

This is a containment experiment, not a converged contact solver or a promotion recommendation. It is based on 40ba4f37 and keeps the existing distributed certificate while adding the same final-state normal complementarity check to the non-distributed wave solver. The threshold is 1e-4 m/s plus 1% of the larger pre-solve or target normal speed. This is an interim publication gate, not a general force-accuracy specification. Both paths check finiteness and normal impulse bounds; an upper impulse bound has the opposite complementarity sign. Status 11 and the first failing constraint are propagated to the world transaction. The corrected non-distributed error status reports the normal residual in its normal component.

## Clean baseline and independent reference

An independently linked 304fd4be world owner and retained baseline metallib accepted a frictionless 33-contact fixture with two production sweeps even though its normal complementarity residual was 0.1200054 m/s. Its pre-contact normal velocity was -0.08175 m/s, its published center velocity -0.0491071 m/s, and its normal impulse 0.0326429 N s. For a 1 kg sphere-body with 0.1 kg m² moment of inertia and contacts at x = 0, 0.05, ... m, the FP64 normal operator is W_ij = 1 + 10 x_i x_j. Each row has target 0 and pre-solve velocity -0.08175 m/s. The exact unilateral solution is 0.08175 N s at x=0 and zero elsewhere, final normal and angular velocities zero, and contact kinetic work -0.00334153125 J. At 257 contacts the two-sweep baseline accepted zero impulse and residual 0.08175 m/s; one sweep accepted a 77.47 m/s residual and +2.075 J kinetic work. The geometry is a synthetic stress fixture, not a representative object.

The 257-contact captured candidate rows reproduce the GPU complementarity residual in independent FP64 arithmetic: 0.35849949 versus 0.35849962 m/s at two sweeps. See owner-row-reconstruction/verify_rows.py and raw captures in the accompanying archive.

## Containment results and limits

In the 44-case owner-shaped control set the candidate rejects both 33 and 257 contact incoming impacts at all tested 1, 2, 3, 8, 16, 32, 64 sweeps, with exact state restoration and deterministic replay. No-contact, resting and separating cases pass and produce zero contact impulse. The independent 24-case control has 33-contact incoming response accepted at 2, 3 and 16 sweeps; the one-sweep result is rejected (0.02081 m/s residual). All incoming 257-contact cases reject. Both revised native acceptance tests passed. Thus it is useful fail-closed containment but does not provide the requested useful-simulation repair and must not be promoted.

The preceding first candidate reported the non-distributed normal residual in the wrong diagnostic component. Its results are retained separately under all-path-containment, while all-path-containment-v2 contains the corrected status rerun.

## Evidence and reproducibility

The retained archive [ccd-research-evidence-20261107.tar.gz](evidence/ccd-research-evidence-20261107.tar.gz) has SHA-256 66071749f28860da9b8b2e6005eb17ea6c495420068b3f899120f4361affc594. It contains raw outputs, fixtures, source patches and SHA-256 identities. Metal libraries, object files and executable binaries are retained only on the host; their hashes are in the archive. Hashes and the exact fixture sources bind each native result. Runs were serialized with other GPU jobs. No performance claim is made.

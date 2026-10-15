# Coupon case 160 Newton 23 audit

**Classification:** diagnostic analysis of a failed native solver attempt; not a successful coupon cycle or physical validation. Case 160 exits 2, is not timed out, and retains three accepted steps before rejection at loading control step 3.

## Binding and paired history check

The two predictions use the same physical argv after replacing only `--output`; both cite source snapshot `/Users/home/cardboard-implementation-20261014/snapshots/source-tangent-cache-trial-scratch-and-exact-AP-integrated-054` and build `/Users/home/numi-cardboard-20261006/build-cardboard-provenance-054/source-build.json`. The executable SHA-256 is `16d3ff0ec3b9a8895750f738f392ea2ca479d79367ec65ef5aa8895bf90b8273` and Metal library SHA-256 is `89dd021ff5c57985ce15c16e5d3eafbec3534cb3b733e16cd2254bc848722e84`. Input receipt says `source_inputs_unchanged=true`. Case 160 adds only selected diagnostics (true-residual cycles, Newton/Armijo traces and selected operator capture) to case 152.

All 24 corresponding files under the two `output/` directories have identical sizes and SHA-256 digests; no output file differs. This includes the full observation CSV, accepted OBJ checkpoints 0–2, compiled package, material state, manifest, result, and rejected conditioning arena. Every CSV field in all eight rows is also identical. Thus the diagnostics did not alter the recorded production history in this paired run. Both runs still fail at step 3; the identity does not promote either to a complete cycle.

## Newton 23 linear solve and material return maps

For root 3, zero-based Newton iteration 23, the trace contains 64 restart-cycle measurements (restart length 16; configured cap 1024). Freshly assembled `b-Ax` is `4.40767462e-06 N·s` at the first captured cycle and `4.40767417e-06 N·s` after cycle 63, against `rhs_l2=4.40768054e-06 N·s`. The last Arnoldi estimate is `4.40767462e-06 N·s`; its relative difference from the fresh residual is `1.02e-07`, and the largest relative discrepancy over all 64 cycle snapshots is `2.06e-07`. The estimate tracks the recomputed residual closely. The residual itself falls only `1.02e-07` relative from captured cycle 0 to 63 and remains `0.999998555` of the RHS norm. This is measured stagnation through the configured Krylov work, not evidence that the Arnoldi estimate was falsely optimistic.

All captured paper material center and four side returns have valid projection flags and zero terminal failure. Center projections used 1–4 local iterations for both grades; sides used up to 4 liner and 5 medium iterations. Their worst dimensionless residuals are near, but below, the accurate local tolerance: liner center ratio `0.998449`, medium center `0.991347`; over finite-difference sides the maxima are `0.999370` and `0.999312`. This capture does not show a local return-map rejection, although the small tolerance margin should remain visible.

## Armijo result and next diagnosis

The root-3 Newton-23 free-node residual starts at `4.40768281e-06 N·s`. Armijo exhausts 24 trials to `alpha=1.1920929e-07` (`2^-23`); the last trial residual is unchanged at printed precision, `J_min=0.996224165`, and the trial status is `0`. No decrease is resolved, so `accepted=false` and `selected_alpha=0`. This is consistent with the stagnant linear solve direction but does not prove why that direction stagnates.

Next inspect the regional block-Galerkin/FGMRES action and effective coarse basis at the first failing Newton step. Keep fresh `b-Ax` recomputation as the reference; capture the linear residual trajectory and preconditioned operator behavior without increasing iteration budgets or relaxing the certificate. Because local returns succeeded and true/Arnoldi residuals agree, changing the local material tolerance or merely raising the Krylov cap is not the first evidence-supported fix.

Full machine-readable fields, output hashes, and limitations are in [`report.json`](report.json). The native raw trace remains at `/Users/home/cardboard-implementation-20261014/solver/cached-coupon-rejection-linear-and-local-capture-160`.

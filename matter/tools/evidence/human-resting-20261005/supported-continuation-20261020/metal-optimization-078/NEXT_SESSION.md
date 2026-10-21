# Next session: Numi Human mechanics performance
Session closed 2026-10-21 at user request. All execution belongs on SSH Numi. Do not run builds or physics on the Air. No visual verification requested.

## Start here
Repository: /Users/numi/numi-human-aortic-contact-opt-20261019 (Numi2/numi-lab).
Published base for this round: 28ed85f7. Read the latest commit and this checkpoint before editing.
Build directory: build-integrated-validation-20261019/profile-build-039.
Evidence directory: matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-078.
Check git status, origin/main, disk and active GPU/build processes before restarting. Preserve unrelated work. Run one substantial GPU benchmark at a time.

## Outcome and limits
The 0.1x end-to-end requirement is NOT achieved. Best recent three-root startup mechanics result is approximately 0.0000317x at 0.5 ms, not sustained coupled throughput. No supported settled checkpoint or 10/60/330-second qualification was established.
Published 28ed85f7 retains tested empty-work removal, coupled initialization derivatives and telemetry; see build214-219-results.md and its independently verified archive.
This final increment fixes the opt-in terminal cache observer and adds exact sparse coefficient metadata/CPU construction. Runtime GPU consumers remain dense; sparse metadata is not a measured speedup.

## Final bounded runs
Build224 Human-only row equilibration:
- 125 us, three roots: exit 0, 30.070917 wall seconds.
- 1 ms, one root: exit 1, 27.033901 wall seconds, mechanical convergence failure at root 0; nonvascular norm 0.000165635880 versus threshold 0.0000125. This does not qualify 1 ms.
Global unscoped build223 trials failed auxiliary respiratory initialization, not mechanics. The new explicit Human-only scope retains the original targeted-world admission checks. It remains default off.
Build224 native FGMRES: 77,465 checks PASS. Source certification contracts: 34 PASS. Refinement selector: 3 PASS. Build223 sparse basis fixture PASS; fixture is opticOnlyWorld (120 modes / 10 regions), not full Human fidelity.

## Observer finding and unresolved skip failure
Build223 control/skipall terminal snapshots now include graph factors and FEM candidate/residual/solution/FGMRES state. Payload SHA and each member SHA were independently reopened and verified. These captured physical/cache arrays match byte for byte; range-plan metadata intentionally differs.
This does NOT resolve the failed next-root 1% endpoint gate for producer skipping. Keep the optimization default off; investigate uncaptured post-begin/tail state and dispatch effects. Do not assert a stale final-cache cause.
Original malformed build221 JSON is retained, with an explicitly labeled forensic comparison. Build222+ native JSON is valid. Capture uses the existing bounded artifact schema and cap.

## Unbuilt private work (do not treat as implemented)
1. Resting-owner accepted FEM-load recruitment handoff:
   retained-tmp/perf080-resting-owner-load-recruitment.patch in the session archive.
   SHA256 3381abbcff6c39e2ec4cd6ab3a585bf4e4f452cd318eabe0726dc07e194889f5.
   Read its companion checkpoint. Correct owner is RestingCoupling; do not use the incompatible standalone standing Brain.
   Apply/review, compile, test excitation-only writes and rollback, then matched supported-prelude runs. Frozen-pose recruitment is not equilibrium.
2. Sparse runtime consumers:
   /tmp/perf080-sparse-basis-private-20261021/base, also archived below as private-sparse/.
   Manifest SHA256 f537a01a7b0d59ba85a11a77e6d5433ac7fb9319013f8d42f561353b4d4d3104.
   Unbuilt and untested. Verify bindings/extents and sparse tangent/projection consumers before integration.
   Still needs high-mode construction without dense B/global Gram allocations. Existing regional815 candidate FAILED attachment (~4.13%) and strain-energy (~195.4%) gates; no accepted coordinate reduction.
3. Full-stage graph/preconditioner ICB: no functional new patch was produced. Private runtime/base copies are identical. Existing aggregate cycle is not a completed whole-Newton ICB.

## Next implementation sequence
- Review and validate the two private patches separately; publish only passing increments.
- Establish actual supported load transfer and a rested checkpoint before larger-step throughput comparisons.
- Resolve producer-skip numerical mismatch before enabling it.
- Finish sparse high-mode construction and fidelity-driven local refinement; retain full residual/contact/rollback gates.
- Then qualify timesteps and measure sustained whole-scene throughput with Brain, physiology, scheduling and presentation. Do not promote startup body timing.

## Resources
Verified duplicate-output APFS cloning reclaimed about 15.4 GiB without deleting unique evidence; receipt verified-output-clones-incremental-20261021-104814.json.
Session leaves no native benchmark or build running. Check fresh df before new qualification; preserve sources, current evidence and private recovery snapshots.

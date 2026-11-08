# Research-only monotonic box-plane CCD certificate

The isolated candidate adds a whole-sweep clearance test before the existing
subdivision search. It requires a positive projected center displacement,
positive hemisphere-aligned quaternion dot, positive denominator Bernstein
bounds, and either positive clearance bounds or positive initial guarded gap
and positive derivative bounds for every box corner. Failure of any condition
falls through to the unchanged search. Both existing experimental selectors
remain disabled, and no production source or runtime was modified.

For q(alpha)=(1-alpha)q0+alpha*q1, D=q dot q and quadratic unnormalized
vertex projection R, the ideal physical-time map has derivative kappa/D,
with kappa >= dot(q0,q1) for hemisphere-aligned endpoints. For nonnegative
normal center displacement c, the clearance derivative numerator is
R'D-RD' + c*kappa*D, bounded below by replacing kappa with dot(q0,q1).
The implementation evaluates Bernstein coefficients with the existing
outward interval primitives and reuses the existing geometric guard. This
reasoning still needs independent review of the actual integrated path,
rounding, guard and nonfinite input handling before production use.

The direct native scalar screen ran 2,048 deterministic projected sweeps,
including unused rotations, scales, offsets, approaching motions and thin
boxes. Of these, 95 used the new whole-sweep clearance path; an independent
FP64 sampled Lipschitz enclosure stayed above the requested tolerance in all
95. Among the other 1,953 cases, 1,607 had a sampled intersection and were
left to the original fallback. This finite screen, including a scaled 0.2 µm
rounding allowance, is not a proof over all input geometries or Metal
operations. It does not qualify arbitrary moving planes or articulated paths.

At production cap 16, the certificate linked with the previously unpromoted
four-contact research solver completed all four thin impacts, retained four
force-free near misses, and passed the existing single-box accepted-segment
observer across all 12 fixtures. Trace on/off and repeated exposed q/v
matched. However the thick scale-1 impact had 0.4411 m/s independently
reconstructed normal residual against the 0.002 m/s gate. The solver also has
a previously retained mixed rod-rigid response failure. Since every fixture in
this arm was accepted, its rollback assertion did not exercise a rejected step.

Linked instead to the existing main contact solver, the candidate still
rejected all four thin impacts; near misses stayed force-free and accepted
segments passed the scoped observer. Successful intermediate event rows had
normal residuals up to 2.153 m/s for a thin impact and 4.306 m/s for a thicker
impact. Rejected q/v and the fixture-exposed cache restoration checks passed,
not a complete owner-state rollback qualification. These findings separate
certificate progress from the still-unresolved response problem. Neither arm
is promotable; no force-accuracy or timing claim follows.

The initial host harness build failed due to a C++ auto declaration; its
compiler output is retained. The first scalar analysis failed when a NumPy
boolean was serialized; its error and corrected script are retained. The
first main-response JSON had a wrong descriptive solver label; the retained
v1 and corrected analysis differ only in that text. No physical thresholds
or fixture settings were changed for these corrections. Source, patches,
commands, input/output hashes and raw data are retained; full-step runtime and
AIR dependencies are identified by hash but are not all in this directory.

The scalar instrument source is reconstructed from main commit 48d061e3b
Collision.metal copied as Collision.before.metal, the candidate generator,
and scalar-kernel.metal; build_scalar.sh records the native commands. The
first preregistered plan is retained as PLAN-v1.md because the current plan
was amended before the separately controlled main-solver run.

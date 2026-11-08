# Corrected capped-cone Newton curvature

The owning Metal projector's tangential ball root uses
y_i = -c_i / (d_i + alpha). The norm-constraint multiplier is alpha times
the radius, making its tangential curvature alpha times (I - u u^T).
The previous code divided alpha by the radius a second time. The one-line
production correction removes that extra division.

The sealed old/corrected native comparison covers 33 projection inputs:
isotropic closed forms at radii .25,1,2,4, an adhesive case, 24 anisotropic
three/four/six-dimensional cases, and four unaffected branch controls.
The independent weighted-ball derivative is calibrated by finite differences
(maximum error 1.82e-10). Corrected native derivative error is at most
8.5901128e-8. Every non-unit closed-form baseline error exceeds .0236.
Unit-radius controls agree, explaining why such a control alone missed it.

A nearby tangential perturbation exposes the derivative through the full
solver, not just an exported matrix. With one Newton update, the baseline
passes only the two unit-radius checks and rejects the other eight; corrected
direct and PCG paths pass all ten. Both arms repeat their binary projections
and typed solver outputs exactly. Projected impulses/objective/feasibility/
complementarity are byte-identical across arms; unaffected derivatives are
also byte-identical. These are numerical engineering cases, not ecological
or independent learning replicates.

The correction is integrated in src/metal/MetalUnifiedQualitySolver.metal.
The permanent metal_unified_quality_probe includes all ten direct/PCG
near-solution cases and independently checks the published velocity using
an FP64 circular-ball projection. Full Release, the expanded quality probe,
ConstraintIR and native world-contact owners pass.

Before/after native world checks preserve all physical fields in 576 motor
and 48 zero-gain records (GPU timing is excluded where present). The default
crow fixture completes 640 control transitions with all ten physical files
byte-identical. The experimental coupled crow reproduces its original
substep-seven rejection capture exactly. It uses uncapped contacts, so this
fix does not resolve its precision failure. No tolerance, policy, deployment
or physics configuration changed. The full lifelong embodied goal remains open.

Plans, source deltas, compiler/runtime identities, all paired outputs and
production regressions are retained through the content-deduplicated Git
payload. The companion Git-recovery study executes the actual archived
runtimes before exact hash/mode/mtime/no-handle cleanup.

# Dormant Matter-to-slider midpoint bridge

Matter's coupled-contact candidate and publication requests now select the
owner's midpoint-aware operations when its midpoint mode and an articulated
candidate are both present. Their operation values are checked against the
MetalWorld API at compile time. The existing fixed-plane admission guard still
rejects reacting bodies, so these new branches have not been exercised in a
native coupled solve and do not enable the feature.

The explicit moving-plane guard is now a reproducible CTest. It and four
owner/legacy rollback tests passed. One accepted fixed-plane midpoint step
also passed the probe's replay, continuation, forced-rollback and identity
checks; it was a noncontact step and does not establish force accuracy.
See `slider-matter-bridge-ctest-001.log`,
`slider-matter-bridge-fixed-001.json` and
`slider-matter-bridge-inputs.sha256`.

Before admitting a reacting body, the coupled nonlinear residual must
include slider gravity with the correct stage impulse; the tangent, physical
full-step residual, midpoint and endpoint clearance, independent impulse and
torque, work and whole-environment rollback must all pass the preregistered
`SLIDER_NATIVE_QUALIFICATION_PLAN.md`. No completion, speedup or
spatial-convergence claim follows from the bridge.

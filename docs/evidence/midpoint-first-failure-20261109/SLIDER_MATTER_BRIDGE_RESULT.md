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

## Remaining owner interfaces identified by source audit

The generic primal residual rebuild subtracts articulated mass action and
adds contact, but does not add the slider gravity impulse. It does this at the
Newton iterate and final certificate; private Armijo trials have their own
assembly path. A future admitted slider path must include the same stage
force at every relevant evaluation and use consistent stage-to-full-step
impulse scaling in its convergence test. The MetalWorld midpoint pass now exposes the validated
body mass and gravity force; Matter does not yet use them.

The compensated reference-area contact response currently selects only an
unbacked stationary plane. A moving proxy otherwise uses the generic gap
calculation. Its dynamic form needs the relative plane position at the same
stage as the FEM candidate, with rounding accounted for, and endpoint
clearance must use both owners actual accepted trajectory. The current
fixed-plane endpoint scratch is not such a certificate.

Matter can reduce contact-sample moments, but the public MetalWorld result
surface inspected here does not expose a prismatic joint constraint moment.
That reduction alone cannot provide the independent owner-side constraint
reaction required by the preregistered torque gate. The eventual native
fixture must obtain and validate that reaction from the rigid owner, or fail
the gate; it must not infer both sides from one contact sample. These are
source findings, not completed numerical tests.

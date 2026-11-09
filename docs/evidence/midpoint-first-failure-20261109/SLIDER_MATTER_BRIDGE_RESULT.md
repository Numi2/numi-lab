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

The primal residual rebuild subtracts articulated mass action and adds
contact. It now has a guarded route for one half-step of the validated slider
owner gravity force at Newton iterates, private Armijo trials and final
certification. A missing or invalid owner mass or force rejects this route.
The route is still unreachable in a native coupled midpoint transaction
because the dynamic-plane and coupled-owner guards remain closed. Full-step
residual and impulse certification remain separate requirements.

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


## Unsupported precision and ownership requests fail closed

The generic MetalWorld adapter cannot represent compensated Matter
candidate-position outputs or borrowed-encoder and source-mass recording
requests. Previously it silently omitted those fields while forwarding the
request, so a caller requesting low components could mistake unwritten
storage for the owner result. The adapter now rejects such requests before
calling the owner. A geometry request with legacyHighOnly false is rejected
also; non-geometry operations may still carry that global flag when their
geometry outputs are canonically empty. This is containment, not support for
compensated moving-plane geometry.

The adapter test invokes the actual callback with a mock owner, checks each
unsupported field independently, verifies that no rejected request reaches
the owner, and checks forwarding of a supported request. The existing native
owner and coupled rollback tests also passed; see
metal-world-bridge-guard-001.log and metal-world-bridge-owner-ctest-001.log.
The moving-plane admission guard remains in place. None of these checks
establishes native two-owner contact, force, moment, work, or replay accuracy.

## Slider gravity residual preparation

The owner mechanical metadata is forwarded to Matter. For the restricted
one-degree-of-freedom midpoint owner, the shared rigid row receives h/2 times
the signed owner gravity force, consistent with its half-step velocity
increment; gravity is constant and contributes no tangent. The same value is
bound at all three rigid residual assembly sites. No moving-plane admission
rules have been relaxed.

The actual Metal residual operator passed 24 mass, gravity, timestep and
velocity cases, with maximum absolute difference 1.19e-11 N s from an
independently evaluated row. The rebuilt fixed-plane native transaction,
adapter guard and existing legacy two-owner late-failure/crossing rollback
checks passed. See slider-gravity-residual-001.log,
slider-gravity-residual-inputs-001.sha256,
slider-gravity-integrated-ctest-001.log and slider-gravity-legacy-ctest-001.log.
These checks do not exercise the currently guarded Matter-to-slider contact
path and do not establish its force, impulse, torque, work, or replay gates.

## Distinct endpoint planes in the clearance check

The existing reflected-endpoint clearance operator now accepts separate start
and endpoint plane offsets, while requiring a common normal. The admitted
fixed-plane runtime binds the same state at both ends. This makes the
clearance calculation usable for a translating plane only after the future
rigid owner supplies its actual initial and committed endpoint positions; it
does not supply those positions or open the dynamic contact guard.

The Metal check rejects an advancing plane that crosses a stationary node,
accepts a clear approach, rejects an initially unsafe receding plane, and
rejects a nonfinite endpoint. Existing fixed and tilted plane cases and a
257-node mixed batch also pass. The rebuilt fixed-plane native transaction,
midpoint gravity operator and dynamic admission guard pass. See
endpoint-moving-plane-001.log, endpoint-moving-plane-ctest-001.log and
endpoint-moving-plane-inputs-001.sha256. These are geometric operator and
fixed-plane transaction checks, not evidence of a two-way contact rollout.

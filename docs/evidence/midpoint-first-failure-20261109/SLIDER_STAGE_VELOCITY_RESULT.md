# Slider midpoint stage velocity and failure status

A native owner test first exposed that the borrowed midpoint candidate supplied
the correct stage position but a zero body velocity: one case returned 0
instead of 0.015095 m/s. That failed result is retained in
`slider-stage-velocity-001.log`. The explicitly admitted slider owner now
sets the private candidate body's world-z velocity from the same source
velocity and midpoint increment used for its position. Across eight accepted
cases the largest stage-velocity error was 1.75e-9 m/s.

A subsequent nonfinite-stage test found a separate acceptance defect: the
borrowed candidate reported an error through the contact-status arena, but
the free-motion commit only consulted the environment status. The test
observed a successful ordinary ABA endpoint despite the bad stage; this
failure is retained in `slider-stage-velocity-005.log`. The midpoint path
now promotes candidate failure into the transaction status before commit.
The same test then reported status 3 and restored q and v. The late-failure
same-context retry and eight analytic midpoint steps continued to pass.

Six scoped owner, moving-plane admission and legacy rollback CTests passed;
the successful native trace and hashes are
`slider-stage-velocity-008.json` and
`slider-stage-velocity-inputs.sha256`. The test builds now depend on the
actual Metal shader target so a stale library cannot silently satisfy them.

The new shader functions are optional for non-midpoint use and midpoint mode
fails closed if they are unavailable. An attempted old-library compatibility
check was inconclusive: the available November 8 library lacked an earlier
required late-failure reconciliation entry point and could not initialize.
Its failure is retained in `slider-stage-velocity-oldshader-001.log`; it is
not evidence of either compatibility or incompatibility of the new functions.
A subsequent targeted test rebuilt the previous MetalWorldContact shader
from commit `34522612c` alongside the current remaining shader objects. The
resulting library retained the earlier reconciliation entry point and lacked
both new midpoint functions. The legacy owner probe succeeded, and the
explicitly enabled midpoint owner refused to encode. The library and source
hashes and both outputs are recorded in
`slider-stage-velocity-optional-shader-001.json`.

This repairs a necessary owner contract only. The moving-plane Matter guard
remains closed; these tests do not qualify a two-owner contact transaction,
its impulses, moments, interface work or full-environment replay.


## Owner mass and gravity handoff

The explicitly admitted slider owner now reports the compiled dynamic body
mass and the world-z gravitational force it uses in midpoint publication.
They are read-only pass metadata, and ordinary owner calls report an invalid
marker and zero values. This supplies a single owner-derived source for a
future coupled residual; Matter does not yet consume it and its dynamic
admission guard remains closed.

The native owner probe passed 48 steps across two masses, three gravity
values, two timesteps, two initial velocities, and two loads. It compared the
reported mass and gravity force with the compiled model and checked the
full-step position and velocity and stage velocity. The worst errors were
7.08e-12 m, 2.32e-9 m/s and 1.97e-9 m/s, respectively. The ordinary candidate probe verified the absence of midpoint
metadata. The native trace and binary/shader hashes are in
`slider-gravity-owner-003.json`; the six owner, admission and existing
late-rollback CTests passed in `slider-gravity-owner-ctest-002.log` (and the expanded gravity test in
`slider-gravity-owner-ctest-004.log`). These tests still do not exercise Matter
contact or establish coupled impulse, torque, work, or restoration.

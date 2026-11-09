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

This repairs a necessary owner contract only. The moving-plane Matter guard
remains closed; these tests do not qualify a two-owner contact transaction,
its impulses, moments, interface work or full-environment replay.

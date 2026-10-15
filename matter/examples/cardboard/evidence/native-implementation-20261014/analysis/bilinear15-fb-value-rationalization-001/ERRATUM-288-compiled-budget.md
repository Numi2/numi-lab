# Erratum: run 288 compiled local iteration budget

The original `failure-comparison.json` is retained unchanged. Its field
`compiled_local_newton_iterations_288: 32` is incorrect: it copied the
requested CLI value into a field that describes the compiled material.

The source-bound native observation is
`analysis/native-tet1528-projection-diagnostic-086-overlay-001/failed-policy-288-observation.json`
(SHA-256
`0cac3e4f69337c881e281302428e890c50d842df4e0dd3c726be8b3eb7fa223a`). It
records `requested_command_local_iterations: 32` and
`compiled_local_newton_iterations: 16`, with accurate-policy status 5 at
environment 0 / tetrahedron 1528, zero accepted steps, and no FGMRES iterations.
The native run manifest SHA-256 is
`90f52f600acf399383723a249bfdda1ffba1cada06f6a235511635bea6bc84c9`; its
observations CSV SHA-256 is
`0be594282577aa60c313aabb1f31601d0ce5c4321c48b39f318efb5338c71ccd`.

Therefore, the run tested a compiled local limit of 16. It did not test a
compiled limit of 32. The requested CLI setting remains evidence of user input
only and must not be reported as the active material budget.

The pointwise input used in native294 is captured run283, not the exact failed
run288 trial candidate. The numerical replay below establishes a cancellation
mechanism on the captured state; it does not retroactively identify run288's
exact candidate.

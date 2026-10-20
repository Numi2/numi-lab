# Vascular row 23 diagnostic 026

Same repaired EE physics and inputs as integrated 025; request 21 x125us.
Only diagnostic change: existing NM_VASCULAR_TRACE_ROOT=20, which captures
the existing bounded vascular arena snapshots across microticks/iterations.
No new physical parameters, acceptance tolerances or source changes.

Require exact first20 accepted CSV rows vs 025, same row23 failure and
three completed microsteps. Resolve the actual compiled equation identity,
candidate state, Newton direction, residual/tolerance and trial merit before
choosing a numerical correction. No claim of completed step21 or 32-step pass.

# Repeated standalone paired-rod state and rejected-step replay

Objective: test whether the unchanged `8e01095ff` research kernel and host can
carry paired position information through successive actual standalone Metal
rod calls, while preserving caller state and published output when a GPU rod
step rejects. This is host-side state continuation, not persistent GPU owner
state, MetalWorld, contact, general 3D mechanics, or a production change.

Cases: eight nodes, x axis, offsets 0.25 and 100 m, fixed, moving and internal
hard anchors, 32 accepted 1 ms steps per case, two complete executions each.
Moving anchors follow deterministic small sinusoidal target positions; target
velocities are exact per-step finite differences. Run a clean trajectory and
an otherwise identical replay with a deliberately reversed endpoint at step
16, which must reach a dispatched GPU failure without publication. Require
exact preservation of all input state and the prior published output, then
retry the correct step and compare all physical results and statuses through
step 31 with the clean trajectory. No alternate candidate or numerical gate
will be tuned from results.

Independent reference: separately assemble and solve the exact dense FP64
implicit axial equations, propagating its own reference state and same anchor
schedule. For every accepted step require position <= 1 micrometre, velocity
<= 0.001 m/s, reaction error <= max(1 micronewton, 1%), work error <= 1e-8 J,
local momentum residual <= 0.001 m/s using independent native full-energy
finite differences, with the 5 and 10 nm derivatives agreeing <= 1e-4 m/s.
Record all step outputs, rejected diagnostics, hashes and repeated runs. The
paired research code is not enabled or promoted by this study; other rod
orientations, physical calibration, dynamic contact, coupled rollback,
surgical workflows and speed remain unqualified.

# Native midpoint stage-precision repair: bounded scope

The retained first-failure trace and independent same-equations directional
check identify a concrete loss of every vertical update in the FP32 midpoint
stage. An offline unrounded correction reduces that state below the existing
full-step limit; it does not prove contact or later-step convergence. Work on
this isolated branch, and keep both endpoint and stage compensation disabled
by default and fail-closed outside the existing stateless one-environment
FEM-fixed-plane midpoint fixture.

A valid native candidate needs a private high/low midpoint stage velocity.
Initialize it for each new working solve; accumulate every accepted Newton
correction in that representation; use the same represented velocity for
candidate displacement, material and contact evaluation, line-search trial
geometry, mechanical residual, reflection and full-step physical certificate.
Carry the accepted endpoint low word through checkpoints, restore, fingerprints
and physical digests as previously documented. Scratch stage state must never
become accepted state on rejection, and cannot leak into the next transaction.
Continue to use only the borrowed GPU command buffer. Do not bypass clearance,
change the physical contact law, loosen the physical tolerance, or increase
iteration budgets. All unhandled modes must reject rather than consume only
one limb.

Admission requires the full 10 ms n=8 force-bearing rollout, independent
full-step momentum and kinematics, positive clearance, unchanged reference
force-bin, peak, total and cumulative-impulse gates, a near-contact start,
whole-state rejected-step restoration and deterministic continuation. Compare
mode-disabled behavior and the earlier n=2/n=4 physical trajectories. A single
stage, successful GPU kernel, or noncontact diagnostic is not a completion or
performance result. The real two-way rigid-body test and matched-cost timing
remain separate requirements after the control fixture qualifies.

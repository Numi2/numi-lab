# Non-aliased mode-5 linearized wave check

Mode 8 gave a falsified n=8 P2 acceleration prediction because its midpoint
samples coincide with the P1 interpolant. After seeing that failure, select
mode 5 as a separate non-aliased case, not as a replacement for it. The
same exact linearized, contact-free, laterally constrained standing-wave
problem applies. P1 and P2 interpolants differ for mode 5 at n=8 and n=16.

## Prediction registered before execution

For both n=8 and n=16, compare P2 consistent-mass acceleration error with
both P1 consistent and lumped mass errors, each mass-normalized against the
exact continuum initial acceleration. Predict P2 error will be smaller than
both P1 variants on each mesh. Compute the same Rayleigh-frequency diagnostic,
and require relative SPD mass-solve residual <=1e-9. Retain a failure without
reinterpreting the mode-8 prediction.

Run sequential n=8/16 assemblies in one process, with a 180-second timeout,
one BLAS thread and no GPU. This is an exact linearized initial-state
comparison. It does not establish nonlinear impact force, complete transient
response, spatial qualification or the unchanged 0.25% force/impulse gates.

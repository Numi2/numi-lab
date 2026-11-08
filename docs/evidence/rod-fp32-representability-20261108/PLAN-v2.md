# Follow-up: compensated Metal force evaluation

The prior exhaustive world-space search returned 0.0770653 m/s; the tested
single-FP32 local-coordinate subset returned 0.00929077 m/s, both above the
0.001 m/s gate. The reference and independent finite-difference checks passed.
This contradicts the simple-local-coordinate candidate for that subset, but
does not prove all possible local-coordinate candidates fail. It motivates a
new prediction: existing Numi compensated-scalar arithmetic, using two FP32
values per position and material parameter, can evaluate the same straight
axial momentum equations on Metal within 0.0001 m/s of an independent FP64
calculation and below the existing 0.001 m/s momentum gate. Rounding positions
back to a single world-space FP32 should fail at least the retained offset
fixture. No tolerance changes.

Use a separately compiled test kernel that calls Numi's existing compensated
geometry functions, with compiler reassociation disabled. It reads a static
snapshot only and computes one axial free-node residual; it does not integrate,
commit state, alter production, or implement another simulation. Inputs are
split host FP64 values, including source and target positions, rest length,
stiffness, velocity and timestep/mass. A matched naive arm discards all low
parts. Compare each Metal result with an independent FP64 evaluation of the
exact split input values. First replicate the six free-node rows of the retained
8-node case. Then test newly generated straight analytic snapshot rows at
unused coordinate offsets 0.4 m, 1.25 m and 100 m, sizes 3, 6 and 9, with
fixed end supports and a prescribed symmetric perturbation. Repeat the full
input twice; require bitwise identical GPU outputs and all values finite. The
CPU solves the fixed-end implicit spring equations to construct the snapshots;
verify its momentum equations before using them. These are not independent
physical specimens, and they do not test coupled owners or multi-step behavior.

A failed build, missing output, arithmetic disagreement, native error, or
changed input/runtime hash invalidates the relevant result; keep raw evidence.
Stop after these declared cases. Success supports testing an owned, persistent
compensated state representation; it cannot establish such a representation or
a qualified surgical solver. Existing CCD and AL experimental flags stay off.

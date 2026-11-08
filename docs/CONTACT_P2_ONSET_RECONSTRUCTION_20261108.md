# P2 exact-freefall onset reconstruction

Before further expensive spatial refinement, test whether the impact onset
can be reproduced from an analytically known state. The fixed P2 n=8
geometry and physical law are unchanged. Before activation there is no
contact, deformation or lateral motion, so uniform displacement
u_z=-g*t²/2 and velocity v_z=-g*t are an exact solution of the semidiscrete
consistent-mass equations. At t=4.5 ms the bottom gap is 100.67375 µm,
outside the 100 µm activation band. Retained n=2/4/8 trajectories agree
with this displacement within 7e-19 m and report zero contact force/impulse.

## Prediction and gates registered before execution

Initialize n=8 at this exact t=4.5 ms state; integrate to 5.0 ms with the
same P2 law, order-18 surface quadrature, max steps 16/8 µs, rtol 1e-8/
1e-10, displacement atol 1e-12/1e-13 m, velocity atol 1000 times greater.
Sample at 0.1 ms and independently evaluate Duffy-48/64 surface force and
gap. The first full 1 ms bin (4–5 ms) is the onset-segment impulse divided
by 1 ms because the earlier force is identically zero.

Require coarse/fine first-bin agreement and fine versus retained n=8
full-trajectory first-bin agreement within 0.25% of the retained n=8 peak
1 ms force bin. Require sampled order-18 vs Duffy-64 force difference <=0.25%
and Duffy-48 vs 64 <=0.05% of that same peak, positive sampled gaps and
Jacobians, global momentum balance <=0.25% of retained full impulse,
total-energy drift <=0.25% of free-fall kinetic energy at 4.5 ms, and
contact-work balance <=0.25% of max sampled contact energy or work
(floor 1e-12 J). Reject nonpositive order-18 stage gaps/Jacobians. These
are segment-validation checks, not the full spatial force or impulse gates.

Run only the n=8 pair, sequentially, with 300/600-second process-group
timeouts, a 4 GiB sampled-RSS stop, 1 Hz RSS and before/after host load
capture, one BLAS thread and no GPU. Stop on failure and retain outputs.
A successful shortened n=8 segment permits planning, not automatically
launching, a separately coordinated n=16 segment. Neither segment can
establish full 10 ms spatial convergence or clearance between quadrature
points/evaluations. The unchanged full-force and impulse 0.25% gates remain
unmet.

## Result

Both n=8 runs completed: 28.4 s coarse and 48.6 s fine, with sampled peak
RSS 599,408 and 564,096 KiB. The reconstructed fine first bin was
0.02702469123 N versus 0.02702469141 N in the retained full trajectory,
a difference of 2.09e-9 of retained peak bin. Coarse versus fine difference
was 4.44e-8. Both satisfy the registered 0.25% checks.

Minimum sampled gap was 95.16 µm; sampled Jacobian, Duffy quadrature,
momentum, energy and contact-work checks all passed. This validates the
shortened-start method for the retained n=8 first bin. It does not qualify
the full trajectory, spatial convergence, between-sample clearance or a
future n=16 segment.

[Evidence](evidence/contact-p2-onset-reconstruction-20261108/) retains
input hashes, resource samples, outputs and gate calculations. No GPU or
n=16 trajectory ran.

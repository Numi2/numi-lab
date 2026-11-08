# P2 n=16 onset-only spatial diagnostic

The exact-freefall n=8 segment reproduces its retained full-trajectory first
bin within 2.09e-9 of peak and passes the declared checks. Use that
independently checked shortcut to evaluate n=16 over 4.5–5.0 ms only, with
the same geometry, material, mass, continuum contact energy, order-18
surface quadrature and freefall state. Prior to 4.5 ms contact is inactive.

## Prediction and gates registered before execution

Run coarse/fine DOP853 segments at 16/8 µs maximum steps, rtol 1e-8/1e-10,
displacement atol 1e-12/1e-13 m, velocity atol 1000 times greater. The
onset-segment impulse divided by 1 ms is the original 4–5 ms force bin.
Require coarse/fine bin difference <=0.25% of the retained n=8 peak bin,
and apply the earlier exact-freefall segment's Duffy-48/64 quadrature,
sampled gap/Jacobian, momentum, total-energy and contact-work checks with
the same normalizations and thresholds. Reject nonpositive order-18
quadrature gaps or volume Jacobians at ODE evaluations.

After these checks, compare n=16 and n=8 first-bin forces against a 0.25%
diagnostic threshold normalized by the known n=8 peak bin. This is
intentionally a *partial-trajectory diagnostic*, not the original complete
10 ms spatial gate: n=16's full-trajectory peak, impulse and later bins are
unknown. A pass or failure cannot by itself prove or disprove the complete
spatial gate; no outcome is assumed. Between-point or between-step clearance
is not certified.

Run the n=16 coarse/fine pair sequentially with 900/1,800-second
process-group wall limits, 4 GiB sampled-RSS stop and 1 Hz RSS monitoring.
Stop on execution failure; retain partial output, host load and memory,
source and input hashes. The n=16 setup-only probe previously sampled
1.54 GiB RSS; trajectory and Duffy-reference memory may be larger. n=8
shortened runs took 28/49 s; 8× element scaling suggests several minutes
per n=16 run but sparse-factor cost and host load make that uncertain.
Single-thread BLAS; no GPU; coordinate host use before starting. Full
trajectory, convergence, production, replay and physical-material claims
remain open.

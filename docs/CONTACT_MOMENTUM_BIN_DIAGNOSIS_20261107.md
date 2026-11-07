# Momentum-derived force-bin diagnostic

The retained trajectories expose final total momentum but not momenta at
individual 1 ms bin boundaries. A rerun is required to calculate this
observable. Retain the original study and its unchanged 0.25% force-bin
and impulse gates.

## Prediction registered before rerun

Rerun the same nodal-contact n=16 and n=32 fixed-law trajectories, at the
previously temporally qualified 1 µs step, 10 ms duration. At each 1 ms
boundary compute mass-weighted total z momentum directly from node
velocities. Independently of pressure reporting, infer average external normal
force in the bin as [p(t+1ms)-p(t)+m*g*1ms]/1ms. Compare that observable
with the retained RK4-integrated contact-pressure bins, normalized by the
retained peak reference bin. Also check that newly integrated pressure bins
reproduce the retained outputs and clearance stays positive. Each must agree
within 0.25%; this is a diagnostic accounting threshold, not a substitute
for the unchanged mesh qualification gate. If they disagree, the proposed
pressure-reporting/accounting defect is supported; agreement rules out that
particular explanation at this resolution, but cannot rule out a shared
physical-model or contact discretization error.

Compare the momentum-inferred n=16 and n=32 bins with the same original
normalization. If pressure accounting explains the retained 12.43% spatial
difference, the corresponding momentum difference would be below 0.25%.
The expected outcome, by discrete momentum balance, is that this alternative
observable will also show a failure. This is an independent observable from
the same discrete trajectories, not an independent FEM reference and not
evidence that either mesh is accurate. Run sequentially, with a 180-second
timeout per process, preserving output and stopping on failure.

## Result

Both reruns completed (n=16 13.3 s; n=32 105.4 s), stayed positive in
clearance, and reproduced the original pressure bins to 1.35e-15 or less of
the respective retained peak bin. The momentum-derived bins independently
matched the retained pressure bins to 6.09e-14 at n=16 and 1.44e-12 at n=32
of their respective peak bin, satisfying the declared accounting check.

The momentum-derived n=16→32 maximum bin difference is **12.4253%** of
the n=32 peak, agreeing with the retained pressure-based comparison. The
0.25% spatial force gate therefore still fails. This outcome does not support
the hypothesis that the discrepancy is a pressure-bin reporting error. It
shows that total body momentum follows different trajectories at the two
mesh resolutions; it does not say which trajectory is accurate, and it
cannot distinguish physical wave dispersion from contact-discretization or
other shared modeling effects. Original total-impulse spatial error remains
0.628%, also above 0.25%. No production qualification follows.

[Evidence](evidence/contact-momentum-bins-20261107/) contains source and
binary hashes, exact arguments and runtime, ten full momentum-bin records
per mesh, and the reproducible comparison. No GPU was used.

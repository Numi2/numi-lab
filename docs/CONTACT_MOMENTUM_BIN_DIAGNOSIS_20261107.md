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

# Free-end trajectory qualification

Before the run, select previously generated cases 92101 (3 nodes, 0.5 ms), 92105 (5 nodes, 1 ms) and 92109 (8 nodes, 2 ms). Advance each for 10 accepted consecutive steps with gravity. Carry the Metal API output into the next Metal call, and independently advance the FP64 native-energy reference from its own previous accepted state; never use the Metal output to initialize the reference. Use the same fixed physics parameters and inertia-normalized FP64 search and final curvature criteria as FREE-3D-REFERENCE-PLAN.md.

At every step preserve the existing one-step errors: position <=1e-6 m, velocity <=0.001 m/s, twist <=1e-4 rad, twist rate <=0.001 rad/s; native local linear and angular residuals <=0.001 in velocity units, agreement of two native gradient evaluation scales <=0.0001, and a perturbed local-velocity negative control must fail. Require per-step total momentum balance <=1e-12 kg m/s and accumulated momentum relative to the initial state and gravity impulse <=1e-11 kg m/s. Retain every output and continue only after an accepted, passing step. Run the full set twice with identical records. Report failed or incomplete trajectories without changing gates.

This is an explicit research API, not MetalWorld transaction, contact, attachment, checkpoint, a material calibration or a surgical workflow.

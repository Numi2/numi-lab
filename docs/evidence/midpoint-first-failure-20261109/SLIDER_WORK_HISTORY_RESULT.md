# Independent slider contact-work histories

Four independent FP64 reference runs now include cumulative FEM-side,
slider-side, and relative contact work on the same 2 µs clock as their
retained trajectories. For both the initially separated and initially active
fixtures, the new outputs reproduce every historical field exactly after
removing the added history. The first attempt failed that strict comparison
because the newer primary output schema included an extra initial-gap field;
its output and receipt remain preserved. The repeat instrumented the
hash-matched historical source for that case.

The 0.5-to-0.25 µs refinement passed the registered 1% cumulative-work
criterion for all six histories. The largest normalized coarse/fine
difference was 2.52e-10; each output also passed the independently checked
identity between FEM work, slider work and relative work. Exact inputs,
commands, binary hashes, four outputs and analysis are in the
`slider-work-history-reference-002` records. This qualifies the temporal
work histories of these CPU references for this fixture, not a native run,
spatial convergence or physical material calibration.

The native analyzer now checks cumulative work for each owner and their sum
against the appropriate time-qualified history, using the registered 1%
history gate. It separately verifies each measured nodal impulse against the
unchanged pressure law with the existing impulse tolerance and checks that
the nodal impulses reproduce the total. This closes a blind spot in which
opposite impulse errors at zero nodal velocity could cancel and do no work.
Synthetic tests reject that case and a coherent work-history error while
preserving the local work calculation. They also retain the impulse, torque,
residual and clearance perturbation tests.

The two existing one-step instrument traces pass this narrower local
impulse/work check, as recorded in
`slider-work-history-native-one-step-001.json`. The 200 µm step still fails
the independent rigid momentum gate by 4.692 times its bound. No complete
native reacting-body rollout has passed, and work-history comparison cannot
replace exact whole-environment restoration and retry.

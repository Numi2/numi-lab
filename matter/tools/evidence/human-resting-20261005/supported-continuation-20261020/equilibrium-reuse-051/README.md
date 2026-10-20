# Supported equilibrium and exact-work reuse 051

The supported-pose caller now explicitly limits any secondary joint-stop
unloading step to its initial recruitment RMS plus 1e-12. The default caller
keeps the previous unloading policy. No physical force, tolerance, or numerical
acceptance gate was relaxed. The earlier global change failed the redundant
muscle regression and is retained in 050.

Static-support, nonzero monolithic field, and native material-frame checks pass
(3/3, 42.03 s). The material-frame check compares serial versus parallel tangent
cache state bitwise, including stateless/stateful materials and rollback.
The compiled source snapshot is compiled-source-inputs.json and source.patch.
Some later checkout file hashes in run provenance reflect concurrent source
edits, not this frozen binary; executable and library identities remain exact.

The 0.01-bound control's first accepted CSV row matches frozen048 in every field.
Its physical wall time is 16.657867917 s versus 050's 15.103505375 s; this trial
does not establish a performance improvement. The column-major cache scheduling
follow-up is an unmeasured hypothesis recorded separately.

With recruitment bound 1.0, the initial acceleration residual RMS is
2.23475045151e-5, maximum 8.17881326586e-5, root maximum 2.06386810671e-5;
three closed support rows supply 811.443483223 N in the initialization solve.
The bound is optimizer capacity, not a command setting every muscle to one.
The run accepts eight 125 us roots (1.000000047 ms), without root assistance,
in 64.485442750 physical seconds: 0.0000155074x real time.
This is short numerical continuation, not supported-rest equilibrium,
whole-body anatomical acceptance, or long physiological qualification.

The forced late reject/retry passes and exactly reproduces the first two
ordinary accepted CSV rows, all 77 fields (rollback-parity.json).
A single 1 ms cadence trial is rejected before any accepted step:
Matter status10, residual 0.00016296353715, inner residual 0.00491931149736.
The unchanged 24-Newton/64-inner budgets and existing gates stay in force.
No further cadence escalation is admitted. Failure state and clock remain at t0.

Each child retains the exact invocation, binary/library hashes, compact trace,
and diagnostic excerpts; no visual verification was performed.

# Untouched-seed evaluation result

All six native evaluations completed with no reported failed physics steps.
Every run had zero of eight clean-horizon environments. Height-or-tilt
termination counts were 83 and 82 for policy 1, 80 and 75 for policy 2, and
60 and 60 for policy 3, respectively on seeds 4081022 and 4081023. No run
met the preregistered skill screen. The prediction that at least five runs
would have no clean horizon was supported by all six.

The additional seeds were unused in the documented pilot, training and
selection, but the full historical seed inventory was not established.
The three policies use independent training seeds and were evaluated on the
same two seed values; the environments share one simulated host and are not
independent hardware replications. No real-robot, physically calibrated, or
time-to-skill claim follows. All task-owned GPU work was paused for each pair;
other CPU and system activity were not controlled.

Per-run native JSON, commands, process results, and input hashes are retained
here. The host archive named in archive.sha256 also contains the raw state
traces and stderr; it is not a standalone runtime environment. The same policy
and runtime hashes bind these outcomes to the earlier registered training.

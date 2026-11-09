# Accuracy-adjusted cost: qualification and execution guards

The same n=2, 10 ms reference-area fixture is required for both integrators.
The existing midpoint ladder selected 15.625 µs, the largest passing step
among its tested values. Its 0.08504% error is the 1 ms-binned force error;
the sampled-peak error is 0.2584%. This is not a spatially converged result.
The backward-Euler fine-control ladder is still running; no eligible control
or speed comparison is claimed by this record.

`tests/contact/run_accuracy_adjusted_cost.py` now checks the complete ladder
records and chooses the largest *tested and certified* timestep for each arm.
It refuses incomplete attempts, mismatched producer/shader/benchmark hashes,
changed reference data, mismatched physical inputs, or analyses that do not
reproduce from the retained traces. This is a provenance and eligibility guard,
not a substitute for spatial or physical validation.

After both arms qualify and recognized workloads have stopped, the runner
executes three complete 10 ms rollouts per arm with alternating pair order.
Each has no per-step snapshots and must match the qualified final physical
state and fingerprints. A completion-boundary snapshot lies outside the timed
rollout. The summary reports all six accepted-simulated-time rates and their
medians. The predeclared practical-benefit condition is at least 10% improvement
in median rate with every matched midpoint run faster. Failed or interrupted
runs remain evidence, and prevent a completed benefit result. Process-name
checks are not a complete detector of unrelated GPU activity.

The selection, provenance and summary tests cover modified native inputs,
incomplete trials, nonmonotonic passing ladders, the paired criterion and
incorrect order. They do not constitute a native timing result.

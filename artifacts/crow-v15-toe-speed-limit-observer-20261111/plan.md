# Toe speed limit diagnosis

Reproduce the second-seed 80-lane, 250-step flight start with the same body, frozen controller, dense contact host, initial seed and all runtime inputs. Enable the existing read-only accepted-state and final-substep contact observer on all lanes, and compare sampled q, v and policy actions against the 1,600-step run. Identify all speed-limit hits on lane 28 and the associated last-substep contacts; these are not full-substep impulses and do not establish the causal collision by themselves. Do not alter control, body, reset or solver.

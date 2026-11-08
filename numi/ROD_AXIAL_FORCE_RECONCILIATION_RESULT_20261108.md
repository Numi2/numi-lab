# Reaction and published-geometry reconciliation

The [pinned diagnostic](ROD_AXIAL_FORCE_RECONCILIATION_PLAN_20261108.md) recomputes support reactions from the published FP32 node positions, rounded constitutive data and endpoint momentum, and compares them with both the published reaction and the independent rounded-input FP64 discrete solution.

For the previously failing 128-node, 16 m case, the left support reaction was 0.10092590004 N. Recomputing from the published geometry gives 0.10092590830 N, differing by approximately 8.3e-9 N; the independent solution is 0.09844384303 N. The measured reaction therefore agrees closely with the spring force implied by the rounded published state, while the state-implied force misses the physical reference by 0.00248207 N. This directly supports the [spacing analysis](ROD_AXIAL_FLOAT_INPUT_RESULT_20261108.md) for this case. Emitting the reference force alone while retaining that published geometry would introduce an inconsistency.

The full nine-case output, failed original gate and identities are [retained](measurements/evidence/rod-force-reconciliation-20261108). This is not a general proof of force consistency or of an optimal solution. The 16 m case remains incorrectly accepted and the research candidate remains unpromoted.

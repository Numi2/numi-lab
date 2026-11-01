# Current fractional allocator trial 020

This increment adds complete implicit residual reachability to the shared-capacity
fractional allocator, keeps Dinic's sink-depth BFS cutoff, and emits a full terminal
cut witness only after exact admissible-pair closure is exhausted. The 3-way owner
merge is based on publication commit 125e3913; it retains that commit's semantic
support helper functions and uses the exact trial-020 allocator body.

Focused checks and the current CLI syntax check are recorded in
validation-receipt.json. The existing semantic-support check now expects the
trial-020 parent-K4 seed count of 9; it still checks exact child masses and recorded
flow replay, and its edge-budget case remains explicitly INCONCLUSIVE.

The actual full-graph attempt returned INFEASIBLE_FULL_GRAPH_CUT: 12.778598752468307
kg was sent against 12.998725017533202 kg minimum child capacity, leaving
0.2201262650649905 kg unmet. The earlier 018 attempt sent the same amount and was
also infeasible. Both executions report zero native steps and no physical admission.
Their elapsed times are retained as observations only; they are not a performance
benchmark or evidence of successful transfer.

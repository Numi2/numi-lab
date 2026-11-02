# Whole-body Metal work regression
Compare the retained production kernels with tiled node validation and force
audits, sparse anchor projection, GPU-gated working-set resets, deterministic
corner incidence, and local blood-particle compaction.

Run on SSH Numi using run.py with --matter-metallib, --respiration-metallib
and an external --output directory. Optional --native-library and --material
run the existing attachment, deformation, rejected-step rollback and replay
controls with the new paths forced on.

The suite checks tile boundaries and 318,974-node arrays, two environments,
invalid late rows, cancellation, inactive/failed reaction rows, 128-DOF sparse
projection, both working-set owners and every coupled direction block, ordered
tetrahedron incidence/mass including duplicate corners and high valence, and
blood capacity, tie order, multiple moving-COM merges, material incompatibility,
spin and extreme-coordinate fallback. The legacy validator's overwritten
duplicate-admission flag is corrected; invalid indices cannot be dereferenced.

Topology, projection and particle/history/ledger comparisons are bitwise.
Reordered force sums use the existing 1% plus 1e-6 absolute gate.
Three matched passes retain whole-command GPU timestamps without counter
instrumentation or encoder splitting. Small-workload overhead and failed
trials remain evidence. These controls do not establish integrated throughput,
anatomical qualification or clinical validity.

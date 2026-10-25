# Batched exact construction projector

The connected skin preparation driver previously scanned every node in Python and invoked the ball/halfspace projector separately for each contained-ball plane. Batch those rows with the existing exact analytic projector and enumerate only source/trust overlap rows. No material, force, time-integration or GPU runtime equations change.

Pass47 tests:24 randomized serial parity cases, empty/infeasible/duplicate constraints and19 retained construction tests including the5 actual overlapping source/trust rows. Three phase-alternated synthetic helper measurement pairs at13,518 nodes and1,417 planes: serial43.75–46.50ms; batched2.29–2.35ms; median19.10x helper speedup. All measured output Float32 coordinates are bitwise equal. This microbenchmark does not measure integrated repair runtime or physical throughput. Exact emitted geometry audits remain mandatory.

Frozen Run032 source function is retained as the parity reference. Existing uninstalled geometry proposals, physical ownership and shared checkout are preserved. No accepted native step.

# Cardboard regional tangent diagonal oracle

This record preserves the direct Metal kernel check completed on 2026-10-06.
It compares the experimental regional-FEM diagonal against independent
centered finite differences of force for the authored liner and medium laws.

The command was:

```sh
cmake --build build-cardboard --target numi-matter-cardboard-preconditioner-gpu-check -j 8
ctest --test-dir build-cardboard -R '^matter\.metal\.cardboard_preconditioner$' --output-on-failure
```

The target built successfully and the targeted CTest passed on Apple M4 in
0.64 seconds. The direct kernel dispatched with `physical_steps=0`; it compared
12 diagonal entries per material and reported:

| Material | Maximum relative error | Maximum absolute error |
| --- | ---: | ---: |
| `hajali2009_liner_hill_ideal` | `2.72786e-05` | `54.3262` |
| `hajali2009_medium_hill_ideal` | `2.69773e-05` | `41.7709` |

The check also requires the regional selector to alter the compiled-world
fingerprint and rejects adaptive, mixed-FEM, and reserved-topology fixtures.
The first run failed because the reserved-capacity fixture was erroneously
admitted. Compiler arenas are sized to reserved capacity, including dormant
slots, so vector-size comparisons did not test immutability. The guard now
compares capacities with active topology-node and active-tetrahedron counts;
the same negative fixture then rejected as intended. Both the initial failure
and successful CTest log are retained here.

This is an arithmetic/oracle check of the opt-in preconditioner diagonal. It
does not show faster convergence, a successful full FEM solve, or physical
validation of cardboard.

Artifacts:

- `ctest-lasttest-20261006-001.log`: successful targeted CTest output, copied
  before the next CTest run could replace `LastTest.log`.
- `failed-capacity-guard-001.log`: retained failed first attempt.
- `sha256sums-20261006.txt`: source, material, executable, and metallib hashes
  associated with this check.

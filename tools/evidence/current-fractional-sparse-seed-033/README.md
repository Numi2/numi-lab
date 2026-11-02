# Fractional allocator bounded sparse seed 033

This publication applies the bounded initial-seed policy to the current `main` source after streamed witness binding 032. The 032 commit remains the parent. The isolated 033 trial header (`1ecba67f…`) was based on a separate 030d snapshot and contains other support changes, so the publisher preserves the 032 app header and transplants only the 033 seed-reserve and seed-selection changes. The published header therefore has its own SHA recorded in `receipt.json`.

The initial graph keeps the four nearest admissible children per source parent and the receiver's original P1 preference groups. If no P1 group has an R-valid source, the seed includes the nearest source from the authenticated admissible exact groups. The complete admissible groups remain indexed for the implicit residual/cut expansion. This changes initial graph sparsity while preserving the admissibility and mass-flow checks.

## Focused controls

Run the pinned focused checks with:

```sh
tools/run_current_fractional_sparse_seed_033_checks.sh /tmp/current-fractional-sparse-seed-033
```

The run compiles and executes the fallback/budget control, a six-child mass/accounting control, and the required non-P1 implicit-route control against the published header. It also compiles the same non-P1 fixture against the 09caf055 baseline header and reruns the published 032 stream/gzip-reader controls.

The required-route fixture confirms the baseline starts with 12 seed edges and adds no implicit edge. The bounded seed starts with 10, then materializes one implicit edge. Both return positive B-to-far-child flow, six children at 1 kg and 1000 kg/m³, 6 kg total, and parent totals `{4,1,1}`. Repeated calls are deterministic. The route assignment changes, so the evidence makes no exact-flow-array parity or cost-optimality claim.

This is source-only synthetic validation. It does not run the full allocator, native solver, physical admission, or physical steps. The first runner attempt exited before compilation due to an unset empty Bash array; the runner was corrected and the recorded final strict run passed.

CMake configure was also attempted and stopped on three missing files already absent from the 09caf055 checkout: `numi/VERSION.in`, `numi/REVISION.in`, and `matter/tools/human_skin_fat_static_query.cmake`. No unrelated files were created to work around that. The three focused 033 targets compile directly using the runner's strict flags; CTest itself was not run.

# Tactile performance

This is a measured local result from 2026-07-30, not a projected product
claim.

## Host and build

- MacBook Air, Apple M4, 10-core GPU, 24 GB unified memory
- macOS 26.6 (25G5028f)
- Release CMake/Ninja build
- `MR_TACTILE_QUERY_METAL_ANALYTIC_BVH4`
- Metal ray-query support detected, but not selected
- debug-hit stream disabled
- depth, target-anchor tangent motion, velocity, and history enabled
- one analytical sphere target per sensor
- median/p95 from the reported measured iterations after explicit warm-up

`observe_ms` is the host convenience path: unified-memory input copy, command
encoding, commit, and completion wait. Native RL composition uses
`MetalTactileContext::encode`, which does not perform those host operations.
Diagnostic readback is timed separately and is not part of the RL path.

## Results

| Environments | Sensors/env | Atlas | Median observe ms | p95 ms | Sensor frames/s | Samples/s | Retained bytes | Bytes/env | Readback ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 2 | 16×16 | 1.439917 | 2.423287 | 355,576 | 91,027,469 | 16,577,072 | 64,754 | 2.207292 |
| 256 | 2 | 32×32 | 3.745083 | 5.569883 | 136,713 | 139,993,693 | 65,458,736 | 255,698 | 4.659375 |
| 256 | 2 | 64×64 | 4.426125 | 4.959658 | 115,677 | 473,812,195 | 260,985,392 | 1,019,474 | 11.598042 |
| 256 | 5 | 32×32 | 2.641125 | 3.184737 | 484,642 | 496,273,368 | 163,517,268 | 638,739 | 7.111041 |
| 2,048 | 1 | 41×41 | 6.550750 | 7.119892 | 312,636 | 525,541,045 | 428,502,816 | 209,229 | 20.065875 |

One 256-environment, two-sensor, 32×32 run with an update period of four
physics steps measured 1.470000 ms median and 1.860158 ms p95 over 20
iterations. Its reported 356,658,625 samples/s counts logical sample slots.
Only every fourth map performs geometry queries; retained-map and reduction
work still runs on skipped updates.

The p95 spread is materially larger than the medians on this fanless host.
These short runs are useful implementation checks, not thermal
characterization.

### Compound ownership lookup spot check

The current compound-surface benchmark writes one solver-contact contribution
per sensor, so reduction executes the immutable shape-to-sensor lookup. Two
back-to-back 256-environment, two-sensor, 32x32 runs with 5 warm-up and 20
measured iterations produced:

| Backings/sensor | Median observe ms | p95 ms | Retained bytes | Lookup bytes |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.5300625 | 3.70657085 | 65,475,280 | 12 |
| 4 | 1.5264795 | 5.79898895 | 65,475,972 | 36 |

The median difference is within the noise of this short run. The four-backing
case retained 692 additional bytes in total. The p95 values are reported
without interpretation because the host had just restarted and was still
settling background services.

## What is not measured here

- per-kernel geometry, reduction, and history counter samples;
- complex-mesh throughput sweeps;
- a full physics-step enabled/disabled delta with tactile encoded into the
  same command buffer;
- other Apple GPU generations;
- sustained thermal behavior;
- translator inference or policy-network cost.

These are left explicit rather than inferred from the aggregate numbers.

## Reproduce one point

```sh
./build/bin/metalrobo_tactile_benchmark \
  --environments 256 --sensors 2 --backings-per-sensor 4 \
  --width 32 --height 32 \
  --warmup 5 --iterations 15
```

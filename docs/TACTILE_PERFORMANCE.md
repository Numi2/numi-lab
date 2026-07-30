# Tactile performance

This is a measured local result, not a projected product claim.

## Host and build

- MacBook Air, Apple M4, 10-core GPU, 24 GB unified memory
- macOS 26.6 (25G5028f)
- Release CMake/Ninja build
- `MR_TACTILE_QUERY_METAL_ANALYTIC_BVH4`
- Metal ray-query support detected, but not selected
- debug-hit stream disabled
- one analytical sphere target per sensor
- median/p95 from the reported measured iterations after explicit warm-up

`observe_ms` is the host convenience path: unified-memory input copy, command
encoding, commit, and completion wait. Native RL composition uses
`MetalTactileContext::encode`, which does not perform those host operations.
Diagnostic readback is timed separately and is not part of the RL path.

## Results

| Environments | Sensors/env | Atlas | Median observe ms | p95 ms | Sensor frames/s | Samples/s | Retained bytes | Bytes/env | Readback ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 2 | 16×16 | 0.926042 | 4.027979 | 552,891 | 141,540,017 | 6,068,784 | 23,706 | 5.910750 |
| 256 | 2 | 32×32 | 1.079125 | 4.504630 | 474,458 | 485,845,477 | 23,493,168 | 91,770 | 1.958917 |
| 256 | 2 | 64×64 | 3.005584 | 6.546192 | 170,350 | 697,751,918 | 93,190,704 | 364,026 | 7.314125 |
| 256 | 5 | 32×32 | 1.482375 | 5.275592 | 863,479 | 884,202,715 | 58,612,564 | 228,955 | 4.197167 |
| 2,048 | 1 | 41×41 | 4.683292 | 7.710217 | 437,299 | 735,100,011 | 152,973,088 | 74,693 | 30.041666 |

One 256-environment, two-sensor, 32×32 run with an update period of four
physics steps measured 0.955771 ms median and 4.335329 ms p95 over 20
iterations. Its reported 548,549,810 samples/s counts logical sample slots.
Only every fourth map performs geometry queries; retained-map and reduction
work still runs on skipped updates.

The p95 spread is materially larger than the medians on this fanless host.
These short runs are useful implementation checks, not thermal
characterization.

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
  --environments 256 --sensors 2 --width 32 --height 32 \
  --warmup 5 --iterations 15
```

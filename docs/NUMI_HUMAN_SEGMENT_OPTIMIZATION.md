# Numi Human segment optimization evidence

## Result

The Human stand shader optimization preserves the audited mechanics at both
the 64-step (0.8 ms) and 512-step (6.4 ms) cap-8 horizons on a physical Apple
M4 Pro Mac mini. The production command-buffer cap is therefore set to eight
steps, and the optimized shader is eligible for promotion on this branch.

This evidence does **not** qualify standing, force convergence, clinical or
physiological validity, or a general performance envelope. Each timing value is
one observational sample per variant. A performance claim still requires
repeated interleaved runs with controlled load, thermal state, memory, energy,
and counter evidence.

## Change under test

The baseline was clean commit `61c14e0a72a8a3a31a4250020c17c68ae5566020`.
The candidate admitted one unstaged source delta only:
`src/metal/NumiHumanStand.metal`.

The shader change:

- reuses dead post-mass-assembly spatial scratch for equality derivative and
  target-velocity caches while `q` is unchanged;
- replaces work-only contact `J*v` rescans with the already-computed Delassus
  contractions, subtracting only diagonal regularization and retaining the
  deliberately nonsymmetric off-diagonal terms;
- preserves the production state-update order (normal, tangent 0, tangent 1,
  each in increasing DOF order); and
- skips a limit correction only when its incremental impulse is exactly zero.

The executable, input payloads, diagnostic driver patch, CMake configuration,
toolchain, physical machine, runtime environment, segment schedule, and Matter
metallib were identical across each pair. The audited MetalRobo metallib was
force-cleaned and rebuilt from the complete Ninja graph for each case.

## Physical M4 observations

| Horizon | Baseline wall | Optimized wall | Observed wall speedup | Baseline authoritative segments | Optimized authoritative segments | Observed segment speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 steps / 0.8 ms | 171.929078 s | 113.363924 s | 1.5166x | 123.468685 s | 65.408798 s | 1.8876x |
| 512 steps / 6.4 ms | 1035.158126 s | 570.847542 s | 1.8134x | 987.495698 s | 523.291378 s | 1.8871x |

Both comparisons passed:

- initial and terminal `q` and `v` are bitwise identical;
- all 319 non-timing, non-work mechanics fields are exact;
- all eight signed/absolute work metrics are within the paired FP32 arithmetic
  bounds;
- deterministic replay is bitwise;
- authoritative segment and native execution-stage schedules are exact; and
- the same binary, physical machine, toolchain, build configuration, inputs,
  and unrelated Matter metallib are retained.

For the sustained comparison, the largest work delta consumed about 0.19% of
its declared bound.

## Provenance

- Machine: Mac mini, Apple M4 Pro, 24 GB, macOS 26.6 (`25G72`).
- Human inputs: `f53c2e1053bc90e7ec44fd69a5b49f96b71b71ab`.
- Baseline shader SHA-256:
  `2a33f080920bc0b26fb1528d13f0fc969b5815e7ee82b9352d740063df9f1678`.
- Optimized shader SHA-256:
  `93ce40a8a0d7c6c6e3269c1b8e33dd1e02372f2d2e93856c6e4ae48469ac7532`.
- Baseline MetalRobo metallib SHA-256:
  `94a0c34d1fe4e940ee02e34fbdabc1b1c5dc59dc07595e015bb323658876cda6`.
- Optimized MetalRobo metallib SHA-256:
  `e0156b2dbcacea0a39653dee75722b9ade4e04e822e0184d10592f3bd6b8a093`.
- Physical-machine receipt SHA-256:
  `5ad2b8017e76f37668214d653119b5e55ba8793bd2af90079d8f62f502a0bad7`.
- Comparator script SHA-256:
  `4434e071ee4123221b73b2aa34ea1e86dfdae0b139eff5f8b86e7603620f6d5c`.

The immutable case receipts, comparison reports, and file hashes are under
[`docs/media/numi-human-segment-optimization-v1`](media/numi-human-segment-optimization-v1).

# American crow: native instrument checkpoint, 7 November 2026

This is an engineering checkpoint on macmini, not a completed ecological RL
environment or a promoted animal controller. Owning project: Numi Lab,
`agent/american-crow-rl-20261107`, based on crow revision `577c54ff`.
Execution used an Apple M4 Pro (16 GPU cores, 24 GiB RAM), macOS 26.6,
Xcode 26.6, CPython 3.13.14, MLX/MLX Metal 0.32.4, and NumPy 2.5.4.

## What actually ran

- Release builds of the native rollout, trainer, robot catalog and run compiler.
- Native exact replay and chunk-one versus chunk-eight fingerprint comparison.
  A real stochastic RNG defect was detected: the random key included the
  submission-local index. Both ordinary and residual sampling kernels now use
  accepted episode time. Original one-step random streams are preserved.
- A separate stochastic-PolicyPack control exercised both ordinary and
  residual policy execution across both chunk sizes, with exact fingerprints.
- Visual rollout scaling at 128, 512 and 1,024 parallel environments. The
  1,024-lane short probe measured approximately 49,042 environment steps/s;
  this used an untrained compiled policy and does not measure learned behavior.
- Explicit transfer of the retained 684-input unpromoted distilled actor to
  the current v9 and v10 contracts. V10 inserts 123 zero-connected input
  columns at index 84. The critic is fresh. These transfers do not restore a
  qualified checkpoint and do not constitute learning.
- Three v10 development seeds, 32 environments and 1,600 steps each: **zero
  first-attempt completions out of 96**, 93 first-episode physical failures,
  and zero failed physics steps. The controller was rejected. Qualification
  seeds and held-out layouts remain unused.
- V9 motor diagnosis on one development seed, 32 environments and 1,600 steps
  for each of seven bands. Isolated bands 0, 1, 2, 4, 7 and 9 had no physical
  termination. Continuous band 10 had four. These limited controls do not
  qualify all milestones or supported landing.
- Native/MLX learning: 131,072 samples over two updates, then 65,536 samples
  after learner checkpoint restore. Published revisions advanced 1 → 3 → 4.
  Whole-process throughput was approximately 10,450 and 22,510 steps/s;
  training-loop throughput was 11,255 and 24,133 steps/s. These are short
  engineering runs, not sustained performance or policy-quality results.
- CTest: first-episode recorder controls and native run compiler checks passed.
  Python: two first-episode admission tests and seven MLX worker tests passed.

## Evidence and boundaries

The `trials/` tree preserves prospective plans, raw JSON evidence, execution
receipts, failures, stderr and summaries. `payload-inventory.json` identifies
all original run files, including native rollout payloads, actor transfers and
learner checkpoints retained under `.numi/runs/american-crow/` on macmini.
Large payloads are retained there; this source checkpoint does not duplicate
them into Git history. No negative trial was overwritten or discarded.

The first-episode recorder permanently closes a lane on its first terminal
transition. Later retries cannot improve its count. Reaching the final
waypoint and then crashing remains a failure. Censored horizons are explicit;
waypoint proximity never establishes a supported landing.

Remaining work includes physically active head/beak mechanics, an audited
eye-mounted sensory contract, finite resources and accepted consumption,
physiological state, persistent world/RNG/sensory checkpoints, memory tasks,
social information and ecology. The legacy offset camera, hybrid body
wrenches and privileged route observations remain explicit limitations.
The overall environment goal remains open.

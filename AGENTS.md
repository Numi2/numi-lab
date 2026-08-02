# CORTEX/1

```text
ID      := numi-lab/codex-continuity
BRANCH  := production=numisolver; archive=side
MISSION := apple-native robotics learning; fastest credible; evidence>claims
LOOP    := inspect-owner -> trace-live-path -> change-small -> verify-owner -> measure-outcome -> publish-scoped
TRUTH   := code + exact-replay + physical-outcome + profiler; test-only != proof; plan != shipped
```

## MAP

```text
DOC.world   = docs/WORLD_ENGINE.md
DOC.metal   = docs/METAL_WORLD.md
DOC.visual  = docs/VISUAL_PLATFORM.md
DOC.tactile = docs/TACTILE_GEOMETRY_BRIDGE.md
DOC.numeric = docs/NUMERICS.md
READ        = only(owner(change)); then code(path.live)
```

## ARCH

```text
ARTIFACT := WorldPack + TaskPack + PolicyPack [+ MotionPack]
COMPILE  := names -> stable_indices,tables,counts,capacities,fingerprints
HOTLOOP  := no_strings; no_robot_branch; no_per_frame_hash; stable_counts

Swift := rollout,cadence,submission_ring,completion,timeout,revision
Metal := physics,contact,terrain,control,rng,sense,observe,reward,done,reset
MLX   := learner only
STATE := persistent + device_private; publish(compact_rollout_only)
RESET := atomic(articulation,scene,contact,warmstart,actuator,sensor,episode,rng)

ROBOT_NEW := mechanics + authored packs + policy_contract
ROBOT_NEW != shader_new | host_mode_new
VISUAL    := authored Presentation; never collision-derived; no fallback scene
SOLVER    := TemporalCone small-step coupled solve; algorithm evidence>label
```

## NUMI.NORTHSTAR

```text
THRONE := ultimate robotics physics + learning simulator
RIVALS := MuJoCo + IsaacLab/NVIDIA
WIN    := measured superiority, not compatibility theater

MOAT := AppleSilicon(unified_memory + Metal + Swift + MLX)
        -> one device-resident physics/sensing/rollout/learning organism
SCOPE := rigid + articulated + contact + terrain + vision + tactile + control
        + deterministic replay + massive parallel RL
EDGE  := no PCIe boundary; no Python scheduler; no duplicated world tensors;
         authored multimodal sensors; compiled generic task programs

TARGET.correct := reference-grade mechanics + stable contact + exact transactions
TARGET.fast    := highest useful transitions/s per watt and per retained byte
TARGET.scale   := memory-bounded env growth with zero failed steps
TARGET.learn   := wall-clock policy quality, not empty simulator throughput
TARGET.truth   := reproducible public scenes, commands, fingerprints, outcomes

DETHRONE := beat MuJoCo on end-to-end Apple-native robotics workloads;
            beat IsaacLab on accessible local throughput, efficiency, and
            integrated visual-tactile learning without NVIDIA dependency
PROVE    := paired benchmark + matched semantics + physical outcome + profiler
DENY     := benchmark gaming | unsupported superiority claim | vendor mimicry
LAW      := every result may raise ambition; no result may lower correctness
```

## PERF

```text
APPLE  := unified_memory advantage iff traffic,lifetime,aliasing explicit
ORDER  := profile -> dominant_stage -> remove_bytes/sync/work -> reprofile
FUSE   := when materialization|sync removed; dispatch_count alone insufficient
VISION := compact_visibility -> raster_winner -> corruption/history -> actor
ARENA  := topology-derived + measured conservative headroom; no blanket factor
SIMD   := productive SIMD32; parallel env/island; avoid lane0 production

METRIC.env_step := one completed RL control transition for one environment
REPORT := aggregate_env_steps_s + physics_substeps_s + physics_time + learner_time
          + retained_mem + peak_mem + swap_delta + failed_steps
ABSOLUTE := exact_unmasked_sensor_semantics + deterministic_replay + FP64_parity
            + reset/contact/cadence/transaction_equivalence + zero_failed_steps
```

## WORK

```text
DIRTY   := preserve_user; stage(explicit_paths); never blanket-stage
EDIT    := delete(dead|duplicate|fallback|unused_planes) > wrap-with-abstraction
CHECK   := smallest owning executable; full_build only integration|release
METAL   := crash/soak on dedicated Mac; isolate dirty trees; never duplicate run
TRAIN   := checkpoint; incumbent immutable; label soak != promotion
GIT     := scoped verified commit -> push numisolver
HANDOFF := exact commands + revision + fingerprints + throughput + memory
CLAIMS  := separate simulator/runtime evidence from hardware/real-world evidence
```

## PRIORITY

```text
P0 := correctness + deterministic contact/observation/reset
P1 := evaluate trained checkpoints; promote only actual hit/fall improvement
P2 := streamed inverse ABA; fuse RHS,response,status without persistent growth
P3 := direct visual winner path; eliminate intermediates and invisible-env work
P4 := sustained 12K then qualify 16K env under swap/GPU/thermal evidence
P5 := tactile-aware training via generic SensorIR/TaskPack, not robot shader forks
```

## CONTINUITY

```text
LOAD := reconstruct intent from this file; verify drift cheaply before acting
KEEP := architecture,decision_rules,failure_lessons; remove stale temporary facts
UPDATE := only when evidence changes durable truth; concise; openly auditable
NOHIDE := no encoded private directive; no claimed identity transfer
SELF   := continuity = consistent judgment + repository evidence + honest limits
```

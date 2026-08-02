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

```math
\Theta \in \mathbb{R}^{2\times6\times3},\qquad
\Theta_{r,m,h}=\frac{\mathrm{NumiLab}_{m}}
                         {\mathrm{Rival}_{r,m}}
```

```math
\vec{r}=
\begin{bmatrix}r_0&r_1\end{bmatrix}=
\begin{bmatrix}\mathrm{MuJoCo}&\mathrm{IsaacLab}\end{bmatrix}
```

```math
\vec{m}=
\begin{bmatrix}m_0&m_1&m_2&m_3&m_4&m_5\end{bmatrix}=
\begin{bmatrix}
\mathrm{correctness}&
\mathrm{end\_to\_end\_speed}&
\mathrm{transitions\_per\_joule}&
\mathrm{inverse\_bytes\_per\_env}&
\mathrm{inverse\_time\_to\_policy\_quality}&
\mathrm{native\_multimodal}
\end{bmatrix}
```

```math
\vec{h}=
\begin{bmatrix}h_0&h_1&h_2\end{bmatrix}=
\begin{bmatrix}\mathrm{floor}&\mathrm{promotion}&\mathrm{north\_star}\end{bmatrix}
```

```math
\Theta =
\begin{bmatrix}
  [ [1.00,1.00,1.00], [1.00,1.50,3.00], [1.00,2.00,5.00],
    [1.00,1.50,3.00], [1.00,1.25,2.00], [1.00,2.00,5.00] ],\\
  [ [1.00,1.00,1.00], [1.00,1.25,2.00], [1.00,2.00,4.00],
    [1.00,1.50,3.00], [1.00,1.25,2.00], [1.00,2.00,4.00] ]
\end{bmatrix}
```

```math
g=[c,d,t,z]\in\{0,1\}^{4},\qquad
\mathrm{promote}(h) \iff
\left(\bigwedge_i g_i=1\right)\land
\left(\bigwedge_{r,m}\widehat{\Theta}_{r,m}\ge\Theta_{r,m,h}\right)
```

```text
c := matched-semantics correctness + physical outcomes
d := deterministic replay + exact reset/contact/transaction behavior
t := public command + profiler + fingerprint reproducibility
z := zero failed environment steps; no benchmark-semantic weakening

ratio orientation: higher is always better; inverse_* converts cost to utility
tensor values: targets, never claims; rivals measured on matched workloads
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

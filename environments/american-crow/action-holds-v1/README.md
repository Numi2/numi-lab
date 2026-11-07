# Native held actions for cued foraging

Use this portable observation configuration with the v13 retinal-feeding native
rollout/trainer and a recurrent stochastic actor. It reuses the existing visual
assets. A new ABI50 task holds head yaw for256 accepted ticks and beak requests
for16 ticks, selecting first at decision64. Other actions and all previous-action
history remain neutral before64. Retina and GRU retain their existing cadence.
Ordinary episode reset restarts the action schedule, never the food/cue lifetime.

```
tools/numi crow-env prepare-cued-foraging --output NEW_OUTPUT --allocation-seed 2651109801 --memory-choice-decision-tick 64 --head-yaw-hold-ticks 256 --beak-hold-ticks 16
tools/numi crow-env action-holds-admission --output NEW_OUTPUT
```

Both hold periods are required together, in1..512, with an explicit decision
tick. Native TaskPack GATE0002 binds the schedule and exact task/action/observation
fingerprints. The608byte task header remains unchanged; old shaders reject50
before encoding. ABI47/48/49 and rollout7/8 writing keep their prior semantics.

The privileged typed outcome `policy_action_selection_mask` (`action-mask`,
neutral direction) records newly selected action dimensions. It enters neither
actor/critic sensing nor reward. PolicyRollout v9 makes selected-factor PPO
mandatory; old readers reject9. Native and Python readers reject metadata under
legacy tags, invalid masks and malformed schemas. Every native proposed latent
and full log probability is still recorded and checked against its generating
policy; PPO likelihood, entropy and KL use only selected action factors. Actor
loss averages decision samples; critic loss uses every valid transition. Full
recurrent BPTT preserves the cue-to-decision path. Existing previous-action
history has its original two-tick observation lag and carries held requests,
never the ignored proposals.

Development checks: full Release build;34 Python tests including independent
FP64 loss/gradient references, ignored-proposal invariance and critic-only no-
decision batches;15 malformed native task programs;21 invalid C configurations;
7 malformed native rollout controls; exact v9 and legacy v8 roundtrips; stale49
shader rejection. Native matching/miss port controls still consume40mg/0 and
remain partition-exact. A200tick checkpoint restores held requests/raw latents,
GRU, retina and food exactly through tick352 at chunks1/8, including new head
selection320. Continuous-vs-split training matches all learner/Adam/native state.

The eight-update32-lane development scout produced81920 accepted transitions,
with food intake in29/256 independent lifetimes and no rejected physics steps.
Per-update totals:120000,180000,80000,0,145000,140000,120000,45000 micrograms.
This fixes the observed zero-intake exploration failure; it does not establish
learning improvement or learned memory. End-to-end loop throughput was2308
steps/s for this small native renderer/physics/MLX configuration, not production
capacity. Replay diagnostics find only~3.2e-8 L2 head-decision gradient through
early cue images versus~.177 through recent images with the current randomly
initialized GRU. Memory initialization/retention needs further work before a
larger training claim. Physics, food stock/radius and intake-only reward unchanged.

Evidence, failures, exact runtimes and restoration receipts live in the Git
artifact archive. Raw and duplicate checkout payloads are removed only after
pushed Git recovery is verified. This remains an estimated supported-head
engineering task, not biologically calibrated free crow ecology.

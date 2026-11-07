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

## Explicit initial memory timescales

Fresh recurrent trainer initialization accepts
`--recurrent-memory-timescale-ticks 256` together with
`--initialize-policy ... --recurrent-hidden-size 64 --recurrent-encoder-sizes 96,48`.
The optional maximum must be an integer in2..4096 accepted control ticks. The
learner initializer also accepts0 to preserve legacy initialization exactly.
This is an initialization-only option: saved PolicyPack GRU parameters fully
determine resumed execution, and the flag is excluded from life-training option
identity just like encoder sizes. It cannot modify an imported/resumed actor.
No configuration/wire ABI or native gate implementation changes.

For H units, log-space tau from2 to the declared maximum, and set only the
update-gate input bias to `-log(expm1(1/tau))`. An isolated unforced cell then
has `z=exp(-1/tau)` and hidden decay `h[t]=exp(-t/tau)*h[0]`. Actual gate activity
and effective retention depend on inputs, recurrent couplings and learned
parameters; no fixed memory guarantee is imposed. All gates remain trainable.
Other weights, critic, RNG draws and legacy default parameter bytes are exact.

38 Python checks pass, including the isolated analytic decay/gradient test,
legacy byte exactness and selected-factor FP64 PPO controls. Eight malformed
Swift/worker options reject before simulation or artifact publication. A paired
replay of the same Git-restored native inputs and seeded actor changes only
update-gate input biases. Early-image head64 gradient L2 increases from
3.214976e-8 to0.007818736 (243197x); early-cue permutation changes the head64
mean by0.000688658, versus2.793968e-9 with legacy initialization. This admits a
usable optimization path, not learned cue memory. Native training, held-out
causal evaluation and broader lifelong tasks remain separate requirements.

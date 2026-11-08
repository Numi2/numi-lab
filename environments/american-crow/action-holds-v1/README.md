# Native held actions for cued foraging

Use this portable observation configuration with the v13 retinal-feeding native
rollout/trainer and a recurrent stochastic actor. It reuses the existing visual
assets. The current ABI54 task holds head yaw for256 accepted ticks and beak requests
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

## Native temporal-history failure found at64 lanes

The subsequent64-update64-lane campaign accepted1310720 transitions but showed
no intake improvement (52/512 first lifetimes versus47/512 last). Fresh trainer
continuations at chunks8/1 produced exact policies, rollouts and53 model/Adam
tensors, plus exact retinal/resource components, but world checkpoints differed
in12 data bytes and their checksum. Diagnosis found overlapping native temporal
history: both observe/complete kernels allocate two action arrays per lane but
compute lane base as `environment*actions`. One lane's position history overlaps
the next lane's velocity history, with nondeterministic writes exposed at the
32-lane SIMD boundary. Other world buffers and GRU state matched exactly.

This runtime does not qualify native future continuity or a complete learning
environment. Preserve its exact failures/runtime/checkpoints in Git. Correct and
version the shared native layout, add cross-lane controls, then repeat relevant
learning/recovery admissions. No learned memory success is demonstrated.

## Corrected native temporal history (ABI54)

The shared history repair supersedes faulty ABI47..50 with51..54. Both observe
and complete now stride over velocity and position arrays independently for each
lane. Task/observation fingerprints change, while physical action identity stays
unchanged. Old policies/checkpoints and mixed host/shader pairs reject before
accepted execution. No automatic weight or checkpoint migration is provided.

Native32/33/64/65-lane controls check13,968 history words against accepted joint
state and compare complete checkpoint/stream bytes across chunks1/8. The archived
faulty checkpoint fails1,970/2,304 words under an independent q/v oracle. Current
40mg/0 cue matching/miss controls and64-lane uninterrupted-vs-split PPO continuations
pass: all53 learner/Adam tensors/metadata, policy/rollout bytes and all three native
components match. Run `tools/numi crow-env temporal-history-admission --output NEW`.
This repairs native correctness; the flat learning campaign remains a failure,
and learned memory/causal generalization and wider lifelong behavior remain open.

## Frozen retinal initialization

`tools/numi crow-env calibrate-retinal-initialization --policy INITIAL_POLICY
--rollout NATIVE_PRECHOICE_ROLLOUT --output NEW_DIRECTORY` authors a fresh
revision1 PolicyPack. The calibration is64 ticks with at least8 native lanes,
no selected actions, reward, termination, physics failure or intervening reset.
At least1024 valid image frames are required. It uses only native RGB and validity;
food allocations, labels, outcomes and rewards never fit statistics. FP64 population
moments center only1152 RGB features at indices97..1248; a default0.05 standard
deviation floor caps gain at20. Proprioception, validity/age, every network/gate,
critic, action distribution and semantic contract remain byte exact. Reverting
only id and normalization on the published pack reproduces original source bytes.

Normalization is frozen in the PolicyPack before fresh execution, never changed
inside an existing native lifetime or optimizer checkpoint. Cameras, physical
observations, food, rewards and native semantics stay unchanged. The default trainer
initialization is unchanged; use the resulting `initial.policypack` as an explicit
fresh `--policy-pack` without `--initialize-policy`.

On3968 valid native frames, the bounded initializer improves early-image head-choice
separation26.8x on identical inputs; a memory cut at40 removes the separation.
Three64-lane320tick PPO collections admit native/MLX parity and exact fresh chunk1/8
continuation across all53 learner/Adam tensors, policies/rollouts and native state.
These are numerical/mechanism controls, not learned memory or ecological success.

The fixed-rate normalized64-update campaign accepts1,310,720 transitions without
reported physics failures, but does not demonstrate intake improvement. The
last KL is~1.25e-5 versus configured0.01, with only two full-batch Adam steps
per independent life. The stronger visual gradient alone is insufficient under
this update schedule. Preserve these outcomes; next assess bounded KL-adaptive
optimization through the existing explicit learning-rate override. No learned
memory claim follows from the normalization mechanism or this flat campaign.

## KL-adaptive optimization admission

The next16-update64-lane trial continues normalized policy65 and Adam state on
new native populations with the existing KL controller. It accepts327680
transitions with no reported physics rejections at3853 looptransitions/s.
Intake rises37/256 to56/256 across training blocks; this is not held-out evidence.
Per-update KL/clipping/gradient/rate diagnostics are now retained. Last averaged
KL0.033 and22% clipping require continued monitoring rather than a convergence
claim. On identical last-life inputs, source65 already has a0.5103 cue-dependent
head mean gap; adaptive81 has1.1406. Exchanging early RGB flips these gaps and a
memory cut at40 removes them, but these are offline algebraic diagnostics.

A scoped trainer guard permits explicit learning-rate initialization from saved
model/Adam state only for a new independent cued population. Exact life restore
still forbids overrides and restores adaptive rate/state unchanged. Final81
fresh chunk8/1 continuations match all53 model/Adam tensors and metadata,
policy/candidate/rollout bytes and world/retina/food snapshots. Evidence, startup
and harness failures, exact runtime and Git-executed recovery are retained in
`artifacts/crow-rl-adaptive-optimization-20261108`. Native held-out causal memory,
manipulation, social learning and lifelong ecological qualification remain open.

## Held-out native causal cue memory

The public `memory-causal-admission --policy PACK --output NEW` runs intact,
GRU-cleared40, absent and swapped cue arms on three unused allocation/action seeds.
For unchanged policy81,192 lifetimes/arm give correct-side counts157/80/82/25
and intake counts38/12/13/5. Actual byte-level controls preserve pre-choice physics,
postcue RGB and food allocation; memory clearing changes only the GRU while full
retina/food state stays exact. The cue controls alter native rendered visibility.
This supports causal learned memory in this fixed fixture, not reliable feeding
or general lifelong intelligence. Evidence/recovery: `artifacts/crow-rl-native-causal-memory-20261108`.

## Qualified fixed-fixture deployment97

The canonical `policies/deployment97.policypack` is the deterministic actor mean,
registered with SHA256 in the project profile. On three fresh allocation seeds,
it chooses the food side and consumes40000..45000ug in all192 tested allocations.
Clearing only GRU at40, removing the cue or swapping the rendered cue gives zero
intakes in every control. The actor, recurrent parameters, normalization and
semantic/action contracts match the Gaussian publication exactly. Gaussian
exploration gives49/192 intakes and173/192 correct choices on the same seed set.
The192 deterministic replicates contain only two initial head commands in a fixed
two-cue geometry; they do not establish192 independent geometric challenges or
natural foraging reliability. Evidence: `artifacts/crow-rl-deployment-mean-20261108`.

128-lane learning is admitted at5203 transitions/s and382185480B MLX peak with
zero swap. The changed population/minibatch/rate recipe prevents a pure scaling
comparison with earlier64-lane runs. Exact fresh chunk1/8 continuation matches all53
model/Adam tensors and metadata, policy/rollout bytes and world/retina/food state.
This closes the fixed-fixture native foraging/memory foundation. Persistent ecology,
solid-object manipulation and social learning remain the next implementation scope.

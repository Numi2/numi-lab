# Numi Crow v10 world-model navigation

V10 is the first Numi Crow development track that joins the qualified visual
flight actor to an authored obstacle/perch course, accepted-state RGB-D replay,
an MLX latent dynamics model, cross-entropy planning, and a deployable native
PolicyPack. The bird remains an estimated American-crow hybrid. The evidence
below is Apple-native simulation evidence, not measured animal or hardware
flight.

## Runtime boundary

- Task: `birdflow_american_crow_navigation_v10_world_model`
- Current development actor contract: 763 observations and 15 actions. It
  keeps temporal flight state at `[0, 84)`, the original thirteen route
  values at `[84, 97)`, the post-WP2 route/state adapter at `[97, 119)`, the
  WP1-to-WP2 route/state adapter at `[119, 141)`, the waypoint-two-only
  route/state adapter at `[141, 163)`, and the 600 visual/history values at
  `[163, 763)`.
- Task fingerprint `6195779659272326306` and observation fingerprint
  `1163663963628678811` keep every authored waypoint, reach radius, reward,
  and policy action unchanged while making the three adapter gates explicit.
  The post-WP2 adapter is zero through waypoint one; the inter-gate adapter is
  nonzero only while waypoint one is active; the final adapter is nonzero only
  while waypoint two is active. Older V10 PolicyPacks require an explicit
  zero-connected transfer because these semantics are fingerprinted.
- Sensor contract: four 16x9 instance-masked depth frames at history offsets
  0, 3, 8, and 18, plus 24 derived features
- V9/V10 training and rollout require an explicit
  `--visual-observation-config`. The runtime now fails closed when that config
  or its masked-depth sensor is absent; it cannot silently substitute 600
  zero-valued visual/history inputs.
- Course: two gate bodies, two slalom bodies, and a final perch, all ordinary
  static collision bodies in the Temporal Cone world
- Splits: `training`, `held-out-a`, and `held-out-b`; the same PolicyPack runs
  on every split. `development-reference` is a fourth invocation-only paired
  regression fixture and is never qualification evidence.
- Replay: canonical `numi.crow-replay.v1`, now including the full actor
  observation, course identity, scheduled-reset provenance, and RGB-D metadata

The course geometry is part of native physics. It is not a render-only prop.
Held-out splits transactionally change its default poses while retaining one
deployment ABI, allowing a single controller to be compared on unseen layouts.

### Turn-preview development status (29 August 2026)

Commit `ccfe463` adds a fingerprinted yaw-only preview at the first slalom
exit: position, reach radius, reward, action authority, and all five authored
waypoints are unchanged. A 3,964,928-sample, 32-lane PPO continuation reached
revision 122 with zero failed environment steps, but its deterministic
selection screen completed 0/8 lanes and did not reach waypoint one. A second
31-update protected pilot trained only actor columns 92 through 96 (the
five-way stage mode); it also completed 0/8 lanes with zero failed environment
steps. The earlier broad preview was separately rejected after it regressed
gate acquisition.

These artifacts are retained under the remote `numi-runs` workspace, but none
are candidates and no three-seed qualification was launched. Inspection of the
owning Metal RNG corrected the earlier fingerprint hypothesis: reset samples
are keyed by seed, environment, episode, control step, and channel, not by the
TaskPack fingerprint. The unstable comparison came from episode-local course
randomization and from confusing a one-repeat smoke with the earlier 20-repeat
development horizon.

`development-reference` now pins the five accepted course-body poses captured
from selection seed `2650817001`, environment 4, band 5, where the inherited
V33-r11 actor reached waypoint one. The invocation skips only the 15 course
position-offset operators. Root state, sensors, controllers, dynamics,
terminations, action authority, rewards, and all other authored randomization
remain live. The runtime labels this mode `paired-development-only`; the
qualification script still admits only the training and two held-out splits.
Use band 5 and identical seed/environment/horizon settings when comparing two
policies on this fixture. A pass here is a regression signal, not promotion.

## Model and deployment path

`numi crow navigation` exposes the complete path:

```sh
# Accepted native flight data. Collection disables scheduled resets by default.
numi crow navigation collect --milestone full-journey \
  --policy-pack /path/to/incumbent.policypack \
  --visual-observation-config \
    assets/crow_navigation_course/crow-navigation.sensor-fast.visual-observation.json \
  --envs 32 --steps 1600 --chunk 1 \
  --crow-replay-pack /path/to/training.crowreplay.json

NUMI_CROW_MLX_PYTHON=/path/to/python-with-mlx \
numi crow navigation model-train \
  --replay /path/to/training.crowreplay.json \
  --output /path/to/world-model

NUMI_CROW_MLX_PYTHON=/path/to/python-with-mlx \
numi crow navigation plan \
  --model /path/to/world-model \
  --replay /path/to/training.crowreplay.json \
  --maximum-action-delta 0.08 \
  --output /path/to/demonstrations.json

NUMI_CROW_MLX_PYTHON=/path/to/python-with-mlx \
numi crow navigation distill \
  --model /path/to/world-model \
  --demonstrations /path/to/demonstrations.json \
  --replay /path/to/training.crowreplay.json \
  --base-policy-pack /path/to/incumbent.policypack \
  --policy-pack /path/to/candidate.policypack \
  --output /path/to/student
```

The loader rejects mixed fingerprints, reset discontinuities, scheduled-reset
data, and ground-only datasets before training. Planning uses CEM inside an
explicit action trust region around the incumbent. Distillation initializes
the exact incumbent topology, retains all replayed incumbent actions, and
blends only a bounded planner correction. Model-predicted return is diagnostic;
it cannot promote a controller.

## 29 August 2026 five-waypoint development

### State feedback and isolated inter-gate update

The 719-input state-feedback parent from
`crow-v10-state-feedback-self-imitation-v94-20260829` scores cumulative
waypoint counts `[189, 117, 29, 12, 12]` across the fixed three development
seeds, 64 environments per seed, and 1,600 steps per environment. It has zero
failed environment steps. This improves the retained route-only residual but
still completes only 12/192 lanes.

The 741-input transfer inserts 22 zero-connected observations at actor index
119 without widening the inherited `[582, 326, 198]` hidden topology. On all
three development seeds, every per-environment maximum waypoint and route
progress value is exactly preserved before learning. Full-route comparisons
use chunk one, fixed difficulty band 10, and scheduled resets disabled; omitting
any of those controls is a different experiment.

`crow-v10-intergate-self-imitation-v97-20260829` uses 2,048,000 native samples
and success-filtered self-imitation from trajectories that actually reach
waypoint two. Only first-layer columns `[119, 141)` change: every other actor
weight, every actor bias, and the exploration parameter remain exact. Revision
57 preserves all 189 waypoint-one arrivals and raises waypoint-two arrivals
from 117 to 126, but it reduces full completion from 12 to 11. It is retained
as a bottleneck-positive diagnostic, not as the deployment parent.

A bounded scalar line search between the exact parent and revision 57
selects alpha 0.50. The resulting development parent scores
`[189, 118, 30, 12, 12]` with zero failed environment steps: it preserves all
12 full completions while adding one waypoint-two and one waypoint-three
arrival. A subsequent 1,638,400-sample WP3-success continuation changed only
five late-residual output rows, but its fresh deterministic screen did not
improve completion and it was rejected.

That selected parent completes 6.25% of development lanes, far below the 90%
promotion gate. It is transfer-positive source and training evidence, not a
reliable controller, held-out qualification, or promoted Crow policy. The
waypoint-two-specific follow-up below tests the next isolated segment; shared
late-head continuation did not solve the WP2-to-WP3 bottleneck.

### Waypoint-two-specific update

The 763-input transfer inserts another 22 zero-connected route/state values at
actor index 141. They are nonzero only while waypoint two is active. Before
learning, all three development seeds exactly preserve every per-environment
maximum waypoint and route-progress value from the 741-input alpha-0.50
parent.

`crow-v10-waypoint-two-self-imitation-v101-20260829` trains only first-layer
columns `[141, 163)` from 786,432 stage-two native samples. Episodes become
imitation evidence only after physically reaching waypoint three. Revision 45
raises established full completion from 12 to 13 but reduces waypoint-three
entries. A bounded adapter-amplitude search selects alpha 0.50 and restores
the upstream distribution: the retained 763-input parent scores
`[189, 118, 30, 13, 13]` with zero failed environment steps. Relative to the
741-input parent, WP1, WP2, and WP3 counts are identical while WP4 and WP5 each
increase by one.

This is a Pareto-positive development result, but 13/192 is only 6.77%
completion and remains far below reliability or promotion. The next bounded
milestone is a waypoint-three-specific adapter trained from real WP4 success;
no held-out qualification or new BirdFlow showcase render is justified yet.

### Waypoint-three capacity and rejected learners

The current task ABI adds a 22-value waypoint-three-only route/state copy at
actor indices `[163, 185)`. Components 79...100 are zero outside waypoint
three. Inserting those columns before the 600 visual values creates a
785-input actor while preserving every inherited observation and first-layer
column. The zero-connected transfer in
`crow-v10-waypoint-three-adapter-v103-20260829` reproduces every waypoint,
progress, and completion value on all 192 established lanes exactly.

A stage-three probe initially appeared ideal: all 64 lanes reached waypoint
five. That result instead exposed curriculum saturation. Its eleven accepted
reset rows came from an older 697-input parent, and the retained 763-input
parent already solves every sampled continuation. Two stage-three learners,
including a stronger distillation pass, changed only `[163, 185)` but changed
no full-route waypoint count on a fresh seed or the established selection.
They are rejected.

The follow-up trained on the real band-10 route rather than the saturated
reset pool. Eight updates supplied 819,200 native samples, but only 98...244
samples per update carried accepted waypoint-four teacher evidence. The
shallow adapter remained outcome-neutral. A bounded 2x/4x/8x dose screen
showed no gain and regressed at higher authority. The interpolation tool now
preserves every source-equal float exactly during extrapolation; without that
guard, roundoff in nominally unchanged carrier weights caused false upstream
route drift.

The dedicated residual experiment widens the inherited hidden topology from
`[582, 326, 198]` to `[626, 370, 242]`. Its 44-wide paired-sign path sees only
the waypoint-three inputs and has an exact-zero output projection. The MLX
initializer now preserves the stochastic source exploration head and critic
when widening this actor; an initial run was stopped immediately after that
preservation defect exposed an unintended 0.20 exploration scale. The fixed
pack retains 0.04 and again reproduces all 192 parent lanes exactly.

`crow-v10-waypoint-three-residual-full-route-v110-20260829` trains only the
isolated residual hidden blocks and output tail. Its final action correction
is material (0.0217 mean absolute and 0.0711 maximum on active samples), but
revisions 2, 3, 5, 7, and 9 all remain count-neutral on the fresh screen and
the densest established seed. A final residual dose screen is neutral at 2x
and 4x and loses one completion at 8x. No learned waypoint-three candidate is
retained.

The development result therefore remains `[189, 118, 30, 13, 13]`, or 6.77%
completion. The retained behavior is still the v102 alpha-0.50 parent,
expressed under the current 785-input ABI by the exact-zero v108 residual
transfer.

### Current-parent stage-three arrival pool

The native rollout CLI now accepts `--crow-navigation-arrival-pack`. With
one-step submissions it captures only the first waypoint-three crossing in
each environment: accepted `q` and `v`, course scene state, policy action,
actor observation, task outcomes, and eventual route high-water mark. The
payload is contract-bound and SHA-256 recorded under schema
`numi.crow-navigation-arrivals.v1`; it avoids both a host-authored state
approximation and a full-trajectory dump.

Three established-seed captures from the exact v108 transfer contain 30
current autonomous arrivals: 13 later reach waypoint four and 17 do not. Their
payload hashes are:

- `2650815802`: `4ead528919123dc3f26b13fbf1e1256864dfa1df358c8172c6802db37acae665`
- `2650815902`: `93d58b940afd8ac9d28609bf51ab9a530724f001e5f28cf41672553dce430db0`
- `2650816002`: `f9892ef2c70e6428127368dfefce4b599aac08ef7a6de1d259dede8a305c6f94`

`create_crow_navigation_arrival_pool.py` validates the shared native
contract, rejects scheduled-reset or malformed inputs, retains source
provenance in a manifest, and generates the compiled Metal constants. The
active pool uses every row and rebinds each reset with that row's actual
randomized incoming course yaw rather than the former single historical yaw.

This materially changes the curriculum test. On seed `2650820401`, the exact
parent changes from a saturated 64/64 terminal result to waypoint counts
`[64, 64, 64, 36, 32]`: 32 completions, four waypoint-four-only outcomes, and
28 waypoint-three failures, with zero failed environment steps. Direct
teacher authority raises waypoint-four reach to 47/64 and completion to
35/64, establishing limited recovery signal rather than assuming it.

The 786,432-sample v113 residual learner, still filtered for physical
waypoint-four success, remains full-route count-neutral on the fresh screen.
V114 admits teacher labels from all active waypoint-three states, but it also
fails locally: on fresh stage-three seed `2650820801`, the parent completes
35/64 while candidate revisions complete only 33...34/64. Both learners are
rejected.

Inspection of their rollout contract found that student-authority one stored
the student's executed action as its own teacher target while also disabling
the PPO ratio. The actor therefore received an identity imitation target. The
training-only navigation boundary now follows DAgger semantics: the student
still owns the physical trajectory, but the accepted-state expert action is
stored as the counterfactual label and physical episode outcome controls label
eligibility. Teacher authority also remains available through the fifth target
instead of ending as soon as waypoint four is crossed. Neither change adds an
autonomous supervisor.

Across three 64-lane current-pool seeds, the pre-change teacher reaches
waypoint four 146/192 times and completes 118/192, versus 123/192 and 105/192
for the parent. Giving it pitch-moment authority or extending it through the
terminal target was count-neutral within one completion and was rejected as a
controller change. The useful signal is the original wing-bank/yaw expert,
not a broader hand-authored controller.

V115 and v116 confirmed that repeated curriculum-reset high-water counts are
not a promotion screen. V116 improved a three-seed local stage-three total by
18 completions, but the exact full-route result regressed from
`[189, 118, 30, 13, 13]` to `[189, 118, 30, 13, 12]`. Exact full-route
screening fixes difficulty band 10 and disables scheduled resets; omitting
either is a different workload.

V117 and v118 then trained on authentic full-route histories rather than the
reset pool. After 1,228,800 success-filtered samples they remain exactly
count-neutral at `[189, 118, 30, 13, 13]`. V119 includes physically reached
waypoint-three failures. It leaves discrete counts unchanged, but moves every
waypoint-three failure on the coarse seed closer to the target by
`0.006...0.178 m` (mean `+0.079 m`). Removing its learned yaw row improves
that mean to `+0.093 m`, so the yaw row is rejected.

The wing-only v120 continuation finally crosses additional waypoint-four
boundaries. Revision 123 changes the three-seed result to
`[189, 118, 30, 15, 7]`: two more approach reaches but six fewer terminal
completions, with zero failed environment steps. It is rejected. This exposes
the next implementation target precisely: capture the eight new failed
waypoint-four arrivals and add isolated waypoint-four authority. More
waypoint-three dose cannot optimize the post-gate landing state after its
adapter becomes identically zero.

The native arrival capture therefore supports an explicit waypoint in
`1...4`; omitting it preserves waypoint three. A waypoint-four capture of
rejected v120 revision 123 under the exact established protocol contains the
15 physical approach crossings above: seven terminal completions and eight
failed continuations. Payload hashes are:

- `2650815802`: `51589caa0ec0a9ff6bd8e97da3982ef33508cacc7e938810d1ebe6c16e4cc986`
- `2650815902`: `0e90b31f7f42c91298779b082c4a9915f3adcb8525797d5b661d0f67cfe2551a`
- `2650816002`: `44bb102b6ec8f82a218a36a81c5df964983e0fbaeffe727f68df9d1784d695ed`

These rows are retained as training evidence, not as a promoted policy or a
terminal reset pool yet. The stage-three generator fails closed on a
non-stage-three payload so waypoint-four data cannot be silently compiled
under the wrong Metal constants.

### Exact terminal-approach capacity

The V10 actor now exposes another 22 route/state values at indices
`[185, 207)`. The current contract makes them identically zero except at
waypoint four, so terminal training cannot perturb the upstream WP3-to-WP4
arrival distribution. Vision history moves intact behind the insertion.
Exact transfer from rejected v120 revision 123 expands the actor from 785 to
807 inputs and the hidden topology from `[626, 370, 242]` to
`[670, 414, 286]`; sampled inherited outputs have maximum error `0.0` when
the new adapter is zero, and the source critic and exploration head remain
unchanged.

Two 614,400-sample authentic-history learners were screened. V121 activated
the adapter only after waypoint four and changed the lone waypoint-four
failure's progress by just `0.00017 m`, confirming that the filtered action
arrived too late. A historical V122 screen activated the isolated capacity at
waypoint three as well, but its coarse-seed checkpoints remained at four
completions and later revisions reduced waypoint-four reach from five to four.
Both learners are rejected.

### Measured waypoint-four curriculum and terminal objectives

The arrival-pool compiler now accepts explicit waypoint-three or waypoint-four
payloads and generates distinct fail-closed Metal constants. The waypoint-four
pool contains all 15 autonomous v120 crossings: seven later completions and
eight failed continuations. Curriculum resets restore each row's accepted
`q`, `v`, action history, command/phase, journey phase, root offset, and
incoming-course yaw instead of using the former single historical template.

On fresh stage-four seed `2650824401`, the unchanged parent produces 213
completion events over 32,768 environment steps. Direct accepted-state expert
authority produces 432 on the same workload, with zero failed environment
steps in either run. This is a measured corrective-action gap, not full-route
evidence.

V123 tests binary return-weighted self-imitation: sampled Gaussian policy
actions become regression targets only for physically completed routes, while
partial and failed routes retain zero navigation weight. After 614,400
authentic full-route samples, every screened revision remains exactly
`[62, 39, 10, 5, 4]` on the coarse seed. The objective is retained as an
opt-in learner mode but the candidate is rejected.

V124 applies accepted-state DAgger to 196,608 samples drawn from the exact
waypoint-four pool. Revision 7 improves a fresh local screen from 201 to 238
completion events (`+18.4%`) and raises eventual completion from 61/64 to
64/64 environments, with zero physics failures. When the residual was also
active at waypoint three, however, the authentic full route lost one WP4
arrival. Strict waypoint-four gating restores the upstream carrier exactly,
but the three established full-route seeds remain count-neutral at
`[189, 118, 30, 15, 7]`. A 2x residual dose is neutral on the coarse seed;
4x reduces completions from four to three and 8x reduces them to zero. V124 is
therefore rejected and no policy is promoted. The exact capacity and measured
reset distribution are retained for a future terminal objective, not treated
as reliable five-waypoint evidence.

An action-wise audit of V124's 2,155 qualified samples shows exactly where the
accepted-state expert differs from the student. Mean absolute corrections are
`0.167` on actions 0/1, `0.083` on actions 4/5, and `0.548` on action 13; all
other action rows differ by exactly zero. V125 therefore adds only action 13
to the four trained wing rows. Its fresh local screen improves from 202 to 247
completion events (`+22.3%`), but all three established full-route seeds are
again exactly count-neutral at `[189, 118, 30, 15, 7]`. Doses 2x and 4x are
neutral on the coarse seed and 8x removes all four completions.

Two outcome-driven alternatives also fail transfer. V126 applies PPO directly
to native stage-four return with fixed 0.08 exploration; its local completion
density stays at 210...211 events through revision 6 and the only checkpoint
with better reset-pool coverage is full-route neutral. V127 performs binary
return-weighted regression on sampled actions from physically completed exact
arrival episodes. Its best local revision moves from 226 to 232 events
(`+2.7%`) but is full-route neutral. Finally, V128 admits expert labels from
both successful and failed waypoint-four episodes. It moves a fresh local
screen from 216 to 226 events (`+4.6%`) and is also full-route neutral.

These results close the one-step terminal objective family for this cycle.
More epochs or residual dose are not justified by the measured transfer. The
next development objective is sequence-level terminal shooting or planning
from the captured failure arrivals; distillation should begin only after a
planned action sequence flips a failed native physical rollout.

### Retained route-residual candidate

The retained development candidate is revision 41 from
`crow-v10-route-residual-stage2-refine-v89-20260829`. It is not promoted. Its
actor widens the inherited hidden topology from `[512, 256, 128]` to
`[538, 282, 154]` with 26 fixed paired-sign features derived from the thirteen
late-route observations. The new output columns initialize at exact zero. PPO
freezes every inherited hidden weight, carrier output column, output bias, and
the exploration parameter; only residual columns for actions `0,1,4,5,13`
are trainable.

The late-route observations remain zero until waypoint two. This makes the
proven takeoff and two-gate controller the exact executed carrier before the
first slalom, then grants the residual head authority over the later turn. A
post-waypoint-one diagnostic reached all five waypoints in 3/64 full-route
lanes but reduced waypoint-two arrivals from 37/64 to 13/64. Moving the gate to
waypoint two preserved the counts at `[62, 37]` exactly and retained one full
completion on that seed.

The transfer-positive stage-two refinement was selected across three
deterministic full-route seeds, 64 environments per seed and 1,600 steps per
environment. The untouched carrier scored cumulative waypoint counts
`[189, 117, 2, 1, 0]`; the retained residual scored
`[189, 117, 26, 11, 11]`. Both had zero failed environment steps. Thus the
residual preserves every measured waypoint-one/two arrival, creates 24
additional waypoint-three entries, and produces 11 genuine full-route
completions. Its aggregate completion is only 5.7% (9.4% conditional on
reaching waypoint two), so this is a development breakthrough, not reliable
five-waypoint qualification.

A lower-rate continuation was stopped after its isolated screen regressed.
Its only plausible revisions scored `[189, 117, 22, 8, 8]` and
`[189, 117, 20, 8, 7]` on the same full-route seeds, both below the retained
parent. No later checkpoint was promoted.

### Real arrival distributions and rejected branches

Sixteen accepted waypoint-one arrivals and fifteen accepted waypoint-two
arrivals now seed the native curriculum with captured `q`, `v`, root offset,
course-frame yaw, accepted action history, command/phase state, and journey
phase. The older three-row waypoint-two pool produced an apparent 28/64
terminal completion gain but failed full-route transfer; it is retained only
as evidence of a lucky continuation distribution. Pure teacher, yaw-only
teacher, wing-bank residual teacher, high-noise PPO, and low-noise first-layer
PPO all failed their transfer gates and were stopped.

### Pre-residual experiments

The current development reference uses the authored 0.42 m waypoint reach
sphere, band 10, one-step submission chunks, 1,600 steps, and no scheduled
resets. Its strongest retained parent, revision 14, completes waypoints one and
two in all 32 lanes and waypoint three in 5/32 lanes on selection seed
`2650817001`; it completes neither waypoint four nor waypoint five. This is a
paired development result, not held-out qualification.

The continuation fixed four training-contract defects without easing the
autonomous task:

- V10 route extensions initialize at a zero physical/task mean instead of the
  masked-depth empty-pixel mean of one.
- The 697-input actor publishes an explicit stage one-hot. Transfer inserts its
  five zero-connected columns at `[92, 97)`, preserving inherited route and
  visual behavior. `--train-actor-observation-extension-only` with
  `--train-actor-observation-extension-count 5` can freeze the inherited actor
  and train only this adapter from fresh optimizer state while leaving the
  critic trainable.
- Curriculum stage three samples three nonterminal waypoint-three arrivals
  collected from the retained parent. Each reset restores accepted `q`, `v`,
  previous action and drive history, previous generalized joint velocity,
  navigation command, wing cyclic phase, and normalized journey phase. The
  first 84 actor inputs at reset match their replay source with maximum absolute
  errors of `0`, `0`, and `4.47e-8` across the three templates.
- Under the two-slalom preview fingerprint, curriculum stage two likewise
  samples three current-parent waypoint-two arrivals that continue
  autonomously to waypoint three. This replaces a stale single reset from an
  older low-exploration revision that terminated on contact in 64/64
  development-reference lanes. The first 97 non-visual actor inputs at reset
  match their replay successors with maximum absolute errors of `2.24e-8`,
  `2.98e-8`, and `5.96e-8` across the three templates.
- Curriculum stage four now restores the captured waypoint-four root offset,
  course-frame heading, prior action history, command, cyclic phase, and
  journey phase instead of combining a historical pose with zero temporal
  state. On the source seed's matching environment and episode, the first 84
  actor inputs match the accepted replay successor with maximum absolute error
  `5.96e-8`.
- V9/V10 visual execution now requires the authored sensor config. This closes
  a real evidence defect: recent route-parent training had omitted the config,
  so actor indices `[97, 697)` were identically zero despite the run being
  described as visual.

To reconnect vision without disturbing the retained controller, the MLX
worker can zero-connect an arbitrary actor-observation range while folding the
normalized raw-zero contribution into the first-layer bias. The transformed
revision-14 pack exactly preserved the paired 8-lane waypoint counts
`[8, 7, 1, 0, 0]` both with the old route-only input and with live masked-depth
input. A 131,072-sample live-vision continuation subsequently reached waypoint
three in 2/8 lanes at its best sparse screen, but its paired 16-lane result was
identical to the parent at `[16, 16, 1, 0, 0]`. It is rejected, not promoted.

Exact stage-three reset training was also bounded and stopped. A single-template
run and two three-template variants—route/stage columns and stage-one-hot
only—produced no waypoint-four completion. Their best sparse screen reached
waypoint three in 2/8 lanes, while the paired 32-lane route/stage candidate
regressed from 5/32 to 1/32 waypoint-three completions. These runs demonstrate
that replay-exact resets alone do not solve transfer back to the autonomous
route distribution.

The fingerprinted second-slalom preview was then screened on the same
development-reference seed. Explicitly transferring the retained parent gave
waypoint counts `[32, 32, 4, 0, 0]` over 32 full-route lanes. From the corrected
stage-two reset distribution the unchanged parent gave
`[64, 64, 64, 0, 0]` over 64 lanes and 512 steps, with zero failed environment
steps. A 786,432-sample route-plus-aerodynamic continuation immediately lost
waypoint three and was rejected. A separate 786,432-sample continuation that
trained only actor columns `[84, 97)` preserved `[64, 64, 64, 0, 0]` at every
saved checkpoint and increased maximum route progress from `1.3285 m` at
revision 3 to `1.4287 m` at revision 25, but still produced no waypoint-four
completion. It is retained as a bounded diagnostic, not promoted or extended.

Reverse-curriculum screening exposed that the former stage-four reset was an
artificial terminal-skill blocker: the unchanged parent scored
`[64, 64, 64, 64, 0]` and every sampled episode exited through the 2.5 m
ceiling. After restoring the accepted temporal state and course-relative root
offset, the same parent scores `[64, 64, 64, 64, 64]` over 64
development-reference lanes and 512 steps, with 5,440 native navigation
completions and zero failed environment steps. This establishes the isolated
waypoint-four-to-five segment; it does not establish autonomous arrival at
waypoint four or full-route completion.

The fixed promotion gate is three untouched randomized-course seeds with 32
environments and 1,600 steps per seed: at least 28/32 full five-waypoint
completions on every seed, at least 90% aggregate completion, and zero failed
environment steps. Selection seed `2650817001` is excluded from qualification.

No 29 August candidate is promoted and no three-seed qualification was
launched. The retained residual is the first neural candidate in this track to
produce repeated full-route completions while preserving its measured early
route carrier, but 11/192 remains far below the reliability gate. The next
training work must improve conditional waypoint-two-to-five completion on real
arrival distributions; simply extending a regressing continuation is not a
pre-registered development plan.

## Retained 28 August 2026 artifact

This section describes the historical 684-input artifact. Its hashes remain
valid for that contract, but it is not compatible with the current 697-input
development ABI and is not evidence of five-waypoint reliability.

Three training replays supplied 4,796 contiguous transitions. Across those
replays, 66.40% of frames were above 0.35 m and the maximum replayed root
height was 1.4332 m. The retained latent model uses 128 hidden units and 32
latent units. Its held-out next-observation MSE was 0.63084 in normalized
observation space; this is a model-fit statistic, not a physical accuracy
claim.

The retained planner evaluated 512 start states over a 20-step horizon with
256 candidates, 32 elites, five CEM iterations, and a maximum per-action
deviation of 0.08. The student blended 10% of that correction and deviated by
at most 0.006513 from the incumbent on planner states. It preserves the
incumbent `684-512-256-128-15` actor topology.

Native qualification compared incumbent and candidate across three seeds,
32 environments, 1,600 steps, and all three course layouts: 18 runs in total.
All 15 predeclared gates passed with no failed Metal environment step or
contract failure.

| Course | Forbidden contacts | Total terminations | Mean height | Mean tracking |
|---|---:|---:|---:|---:|
| Training | 58 -> 28 | 105 -> 96 | 0.6256 -> 0.6713 m | 0.7998 -> 0.7250 |
| Held-out A | 51 -> 30 | 96 -> 96 | 0.6752 -> 0.6912 m | 0.7359 -> 0.6892 |
| Held-out B | 39 -> 26 | 97 -> 96 | 0.6349 -> 0.6651 m | 0.7908 -> 0.7163 |

The tracking regression is retained explicitly; it stayed inside the
predeclared 0.10 absolute retention bound. The candidate is promoted for this
simulated navigation contract, not declared universally superior.

Retained hashes:

- latent model: `a1eca4fabe38d9355c3a15032120ef47f7c72094344779ee6e6c0321b59cab3e`
- deployment PolicyPack: `d4eef1080a77203fa0b4e86394dcd46a99a374df119690a6413ea00792b0f50f`
- qualification payload: `f07f74cf06f83b67041d4396dd44cbbef1d9e500f7f05b7803de55308d3780e5`

The retained bundle is in
`artifacts/crow-v10-world-model-20260828/`. Full per-run native evidence and
training replays remain in the ignored `.numi/runs` workspace; their hashes
are locked by the retained qualification artifact.

## Qualification

Run the fail-closed matrix with explicit artifacts:

```sh
NUMI_CROW_WORLD_MODEL_ROOT="$PWD" \
NUMI_CROW_WORLD_MODEL_BUILD="$PWD/build-crow-world-model" \
NUMI_CROW_WORLD_MODEL_BASELINE=/path/to/incumbent.policypack \
NUMI_CROW_WORLD_MODEL_CANDIDATE=/path/to/candidate.policypack \
NUMI_CROW_WORLD_MODEL_RUNS="$PWD/.numi/runs/crow-v10-qualification" \
./tools/crow_world_model_qualification.sh
```

This qualifies control and collision outcomes. High-quality feather rendering,
multi-angle presentation, and movie quality remain BirdFlow responsibilities
and require separate inspection of real rendered frames.

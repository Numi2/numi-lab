# Matter

Matter is Numi Lab's Apple-native variational solver for coupled FEM, MPM,
mixed fields, internal material variables, and every contact involving a Matter
continuum. The `coupled` branch has one GPU-resident nonlinear authority rather
than sequenced deformable, pressure, transport, contact, and rigid-response
solvers.

## Transactional neuron-culture learning inside physics

Numi Lab now includes a synthetic neuron-culture research platform: deterministic delayed
LIF networks, short-term depression, excitatory STDP, a virtual 60-electrode
MEA, and phase/tubulin neurite growth execute as one fingerprinted Apple-Metal
program. Neural and growth state remain unpublished until the embodied step is
accepted; rejected futures cannot become plasticity or memory.

```sh
numi neurons protocol --preset potter-switch-v1 --output run.ncrun.json
numi neurons view --live
```

Fresh authored weights and source preparation are deliberately separate:
`numi neurons simulate --mode potter-equilibrate-v1 --checkpoint-out
potter-equilibrium-seed-2056.ncstate` publishes a resumable accepted state for
later protocol runs. Promotion restores each seed's exact seven-hour state into
every mapping and ablation; STDP-off freezes weights without changing culture
identity or initial state.

The bundled reference preserves the Potter experiment's 1,000 neurons, 50,000
synapses, and 60 active electrodes. It is a simulation platform—not a living
culture, hardware MEA, experimentally calibrated biological twin, or clinical
model. Architecture, equations, commands, sources, and executable evidence are
in [`docs/NEURON_CULTURE.md`](docs/NEURON_CULTURE.md).

The same accepted-root protocol now carries synthetic culture state beside the
mesoscale Brain: ten NHCNT support rows prepare bounded virtual-MEA input;
proposal/ACK/apply keep it private; aggregate publication releases Brain,
Human/Matter, HumanIO, and culture as one visible root. The portable 2D animat
remains the learning benchmark. A fixed 3-network-seed × 5-CPS-set promotion matrix and
three ablations prevent showcase claims from outrunning measured advantage.
The platform and transactional embodiment are implemented; adaptive-learning
promotion remains open until that complete matrix passes its fixed statistical
gate.

Numi Human lends its live NHTENDON2/3 endpoint transfers, body kinematics,
Jacobians, and step status to Matter without a host readback or second command
buffer. `NumiHumanTendonFEMLoadAdapter` drives prescribed bone-following FEM
anchors and projects accepted fixed-node reactions back through each owning
body Jacobian before Human dynamics. Passive attachment-only execution injects
no tendon load and replaces no MyoSim force share; active tendon coupling also
applies the equal-and-opposite load-side traction and replaces the exact
declared anchor-endpoint share of MyoSim `J^T`. A post-validation phase commits
or rolls back the matching Matter transaction. Transfer-only execution still
leaves `J^T` untouched; production coupling replaces a share and never adds a
duplicate force owner. The passive multi-body qualification and its limits are
recorded in
[`docs/NUMI_HUMAN_PASSIVE_FEM_ATTACHMENTS.md`](docs/NUMI_HUMAN_PASSIVE_FEM_ATTACHMENTS.md).

The environment-wide Newton unknown is

\[
x=(v_{FEM},\pi,T,p_f,\phi,a,v_{MPM},\Delta v_{rigid}),
\]

and restarted matrix-free FGMRES is the sole linear convergence owner. IPC's
squared-distance logarithmic barrier contributes primal gradients and PSD
Hessian actions directly; per-node timestep ratios keep cross-rate contact the
gradient and Hessian of one environment action. No contact multipliers,
Delassus rows, or post-contact correction solve remain.
Matter ABI v21 explicitly identifies the tapered needle-tip capsule that may
drive tissue puncture; generic shaft or swage capsules cannot mutate topology.
Matter ABI v20 exposes only controls that own live work: Newton/FGMRES
budgets, bounded field-smoother passes, residual tolerances, and a minimum
contact-separation ratio relative to the authored contact slop. Both contact
line search and final Metal acceptance read that ratio. Relative correction is
finite certificate telemetry; the assembled residual is the convergence
authority.
The right preconditioner uses the componentwise diagonal of those same PSD IPC
blocks in its FEM and MPM fine and translation-mode mechanics approximations;
it does not assemble a second contact matrix or introduce another iteration
owner.
MetalWorld keeps sole ownership of ABA and generalized coordinates while its
borrowed coupled-candidate callback supplies kinematics, mass action,
inverse-mass preconditioning, and accepted publication.

Matter Language supports explicit state updates, authored implicit residuals,
specialized multiplicative von Mises and Drucker-Prager return maps, and
`average|max|sum` remesh-transfer policies. Active sparse MPM grid velocity is
part of the same Krylov vector as FEM and rigid increments. Its private Krylov
and basis arenas reserve only the cooked quadratic-support bound rather than
the full authored grid; APIC, deformation, and material state publish only
after global candidate acceptance.

Topology mutation is transactional. Cohesive separation, erosion, edge
split/collapse, 2–3/3–2 flips, and vertex smoothing rebuild derived incidence,
surface contact, and preconditioner structures on Metal. Cavity-wide material
transfer follows each state's average/max/sum policy, and the rebuilt nodal
state is corrected to its pre-remesh momentum and volume-integrated fields
before certification. Conservation and element-quality failures reject the
environment. Private arenas grow
geometrically only between completed submissions through a borrowed
command-buffer migration. Migration rebases every active tetrahedron and
cohesive-face reference into the expanded global arenas before rebuilding
incidence; shaders never allocate.

The hot path uses one borrowed MetalWorld command buffer, private authoritative
state, deterministic sparse ordering, SIMD32 reductions, and no internal
commit, wait, or CPU counter read.

## Curved surgical-needle tissue passage

The dual-PSM surgical probe now drives the authored 26 mm half-circle needle
about its actual 8.28 mm curvature centre at a 20 mm/s terminal speed. Matter
admits entry from accepted tapered-tip contact, then extends a connected
diameter-scale tract only when the sharp point reaches the live frontier. A
rotating tip follows its swept endpoint velocity rather than the finite
collider chord, and midpoint-tangent chords bound accumulated path drift.
The terminal taper and its adjacent widening segment both remain live tissue
contact geometry; the physically swaged 250 mm monofilament is carried by the
DER constraints rather than prescribed along the needle orbit.

On Apple M4, the source-corrected 0.82 mm porcine-jejunum coupon qualification
reached 207.061 um of clearance beyond the *deformed* distal surface after 937
base DER substeps in 234 Matter groups. It retained all 40,800 tetrahedra with
zero removed tissue mass, formed seven connected tract segments and six links,
bounded maximum orbit error to 7.587 um, retained a 0.997651 minimum
determinant, ended at 1.277 um hard-swage error, used at most five FGMRES
columns, and reported zero failed steps. The accepted continuation then carried
the needle to 1.0995 mm distal handling-segment clearance and held its dynamic
giver/receiver bridge for 25 steps. That bridge retained the same seven tracts
and all 40,800 tetrahedra, limited additional tissue motion to 8.382 nm, ended
at 10.23 um hard-swage error and a 0.671 mm/s live contact residual, and
reported zero failed steps. The tract remains a mass-conserving sub-element
contact discontinuity, not a constitutive crack surface or clinical
validation. Receiver acquisition, extraction, and robot-driven knot formation
remain separate physical qualification boundaries.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-curved-passage-only
```

The receiver-side frame qualification now respects the same handling rule
instead of reaching for the newly exposed tip. It advances the analytic needle
orbit by 1.8878 rad until shape 18, 13.975 mm of arc behind the point and one
segment inside the authored one-third-to-one-half handling interval, has 120 um
of full-capsule clearance beyond the distal tissue surface. A six-coordinate
damped IK solves jaw midpoint, rail tangent, and roll about the needle under
the source PSM limits. The selected 10 mm insert-normal approach retained
3.786 mm clearance above the catch pad, had no incidental needle or cross-arm
contacts, and closed to 8/8 jaw contacts with all 16 raw contacts in the safe
zone. This fast gate is kinematic and collision geometry only. Continuous
shaft/tissue contact through the remaining curved passage and dynamic receiver
extraction remain open and are not inferred from it.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --receiver-frame-ik-only
```

The post-puncture extraction fixture also has a live dynamic baseline before
the second instrument moves. It initializes the rotated steel needle and its
hard-swaged 250 mm PDO 3-0 DER strand at exact rest length, retains the giver's
normal 60 um insert preload without an artificial transport over-closure, and
holds the fully coupled island for 100 ms. On Apple M4 the deterministic
endpoint retained 12 safe-zone contacts distributed across both transverse
insert sides and both longitudinal rows (`0101`/`1111` patch masks), 89.4 um
seat drift, 3.49 um/s relative point slip, 8.71 mm/s maximum strand speed,
0.893 um hard-swage error, and a 0.577 mm/s live temporal-cone residual with
zero failed steps. This is a dynamic post-puncture giver/needle/thread fixture;
it does not infer continuous tissue contact, receiver acquisition, extraction,
or knot formation.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --receiver-extraction-giver-hold-only
```

From that accepted dynamic state, the distal receiver now follows a
branch-continuous six-coordinate frame trajectory to the exposed handling
segment with its jaws open. The 300 ms cubic move peaks at 50 mm/s and used
23.8% of the authored joint-speed envelope. Its Apple M4 qualification had
zero receiver/needle and cross-arm contact samples, retained the giver at
2.57 um seat drift and 56.1 um/s relative slip, reduced maximum strand speed
to 0.432 mm/s, ended at 0.872 um hard-swage error and a 0.658 mm/s live
temporal-cone residual, and reported zero failed steps. The accepted q/v,
needle, strand-node, and twist state is resumable. Receiver closure, load
exchange, tissue extraction, and robot-tied knot formation remain separate
qualification boundaries.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --receiver-extraction-approach-only \
  --state-output-dir /tmp/numi-distal-extraction
```

Receiver closure now establishes dynamic dual positive control before either
instrument changes load. A 240 ms cubic LND closure peaks at 0.154 rad/s,
ending at the gentle 15 um overlap while the giver retains its 60 um seat,
then holds both instruments for 200 ms. The Apple M4 endpoint covered the
giver insert footprint with `0101`/`1111` masks and the receiver with
`1111`/`1111`, retained 1.21 um giver seat drift and 3.60 um/s relative slip,
limited strand speed to 0.509 mm/s, ended at 1.06 um hard-swage error and a
0.529 mm/s live temporal-cone residual, and reported zero failed steps. This
qualifies simultaneous control only; it does not yet claim load transfer,
giver release, tissue extraction, or a knot.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --receiver-extraction-closure-only
```

The next resumable phase transfers the working preload without teleporting or
kinematically attaching the needle. Over 134 ms, the receiver advances from
the gentle 15 um overlap to its normal 60 um transport preload while the giver
backs off from 60 um to 15 um, followed by a 100 ms coupled settle. On Apple
M4, the receiver retained 8/8 safe-zone contacts with full `1111`/`1111`
coverage. The unloading giver retained 4/4 bilateral safety contacts with
`0101`/`0101` masks, spanning both transverse insert sides; unlike the new
load-bearing owner, it is not required to occupy both longitudinal rows. The
endpoint measured 10.2 um receiver seat drift, 6.49 um/s receiver relative
point speed, 0.448 um maximum DER edge error, 1.768 mm non-neighbour strand
clearance, 0.925 um hard-swage error, a 0.521 mm/s temporal-cone residual,
zero cone violation, and zero failed steps. This qualifies load exchange while
both instruments remain in contact; giver release, continuous tissue
extraction, and robot-tied knot formation remain separate boundaries.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --receiver-extraction-load-exchange-only \
  --resume-receiver-extraction-positive-control \
  build-coupled-dev/testing/suture/extraction-state/receiver-extraction-positive-control.tsv
```

Giver release is a separate receiver-ownership gate. The giver opens from its
15 um overlap to the calibrated 0.30 mm diametral clearance over 240 ms at a
0.154 rad/s peak jaw speed, then both arms hold position for 200 ms. On Apple
M4, the endpoint had zero giver/needle contacts while the receiver retained
8/8 safe-zone contacts and `1111`/`1111` insert masks. Receiver seat drift was
59.9 um, relative point speed 2.21 um/s, relative angular speed 1.86 mrad/s,
maximum strand speed 0.484 mm/s, maximum DER edge error 0.405 um, hard-swage
error 0.886 um, and the live temporal-cone residual 0.711 mm/s. Cone violation
and failed steps were both zero. This qualifies settled sole receiver control;
it does not yet claim tissue-side retraction or continuous tissue coupling.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --receiver-extraction-giver-release-only \
  --resume-receiver-extraction-load-exchange \
  build-coupled-dev/testing/suture/extraction-state/receiver-extraction-load-exchange.tsv
```

The released giver must withdraw before the receiver translates the curved
needle. A direct receiver move exposed a real sequencing failure at 100 ms:
the receiver still held 8/8 contacts, but the open giver recontacted one side
of the needle. The corrected phase retracts the giver 8 mm upward along its
trocar axis over 600 ms, then holds for 100 ms. Its Apple M4 endpoint measured
8.000 mm projected giver travel, zero planned giver/needle and cross-arm
contact samples, receiver `1111`/`1111` masks, 9.20 um receiver seat drift,
4.54 um/s relative slip, 0.561 mm/s maximum strand speed, 0.380 um maximum
DER edge error, 0.795 um hard-swage error, and a 0.472 mm/s live contact
residual. Cone violation and failed steps were zero. This qualifies instrument
clearance before extraction; it does not yet qualify the receiver translation
through continuously coupled tissue.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --receiver-extraction-giver-retreat-only \
  --resume-receiver-extraction-giver-release \
  build-coupled-dev/testing/suture/extraction-state/receiver-extraction-giver-release.tsv
```

From the cleared-giver state, the receiver now translates the loaded needle
6.5 mm over 500 ms in ten resident 50 ms chunks. Every chunk must retain zero
giver contact, bilateral four-quadrant receiver coverage, bounded reseating,
and a healthy DER strand before the next command is submitted. The receiver
uses the authored 75 um transport preload only during motion, then returns to
the qualified 60 um seat over 40 ms and holds for 100 ms. On Apple M4 the
needle followed 6.4994 mm total and 6.4994 mm along the commanded direction;
the endpoint retained zero giver contacts and receiver `1111`/`1111` masks,
12.5 um seat drift, 3.89 um/s relative slip, 0.719 mm/s maximum strand speed,
2.47 um maximum DER edge error, 3.35 um hard-swage error, and a 0.653 mm/s
live contact residual with zero cone violation or failed steps. The minimum
non-neighbour strand clearance was 49.9997 um at the authored 50 um
self-contact shell, inside its existing 0.1 um FP32 readback allowance. This
qualifies receiver-only dynamic carry in the neutral-zone fixture; continuous
deformable-tissue extraction and robot-tied knot formation remain separate
boundaries.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --receiver-extraction-retraction-only \
  --resume-receiver-extraction-giver-retreated \
  build-coupled-dev/testing/suture/extraction-state/receiver-extraction-giver-retreated.tsv
```

## Surgical-knot instrument protocol

The deterministic protocol gate derives the instrument envelope from the
source-pinned 8 mm Large Needle Driver and verifies the jaw-centre geometry of
a two-wrap first throw followed by five alternating square single throws
(`2=1=1=1=1=1`). The extra securing throw follows a [published PDS II knot
mechanics study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4167833/), while the
product instructions require flat square ties and note that monofilament may
need additional throws. That study used USP 0, so its count is a conservative
research protocol rather than a clinical prescription for this USP 3-0
strand. Every throw integrates signed winding from sampled motion, requires a
finite-clearance tail transfer through the bight gate, bounds jaw-centre speed,
and requires monotone opposing cinch motion. Same-handed, missing-wrap, and
missed-gate controls are rejected.
This is an instrument-trajectory authority only; it is not evidence that the
thread formed or retained a knot, that the path is reachable by the articulated
PSMs, or that a live tissue-coupled robot sequence executed.

The same probe now partitions the 250 mm DER rest material at two tract-owned
edges. Its source-aware setup targets a 19 mm intracorporeal short end with a
1 mm tolerance and retains at
least 180 mm of working arc for the double throw, then emits eight bounded
25 mm-or-shorter draw strokes while conserving the inter-tract stitch span.
This turns pull-through into an explicit regrasp plan rather than one
workspace-invalid long translation. The live opposing-bite root continuation
now instantiates the same plan from the two material edges measured in the
accepted tissue tracts; the strokes themselves are not yet live PSM evidence.

After the short tail is physically acquired, the first-throw preflight maps the
two-wrap protocol into the checkpoint's live deformed-tissue frame and solves
both full PSMs through staging, winding, bight transfer, and opposing cinch. It
checks the carried curved needle, distal instrument envelopes, cross-arm
geometry, and authored joint-speed limits at every sample. This command does
not advance physics or claim a formed knot; the subsequent loaded DER replay
owns that evidence.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-knot-first-throw-preflight-only \
  --resume-tissue-checkpoint tissue-thread-acquired \
  build-coupled-dev/testing/suture/tissue-thread-acquired.tsv
```

`--tissue-knot-first-throw-stage-only` crosses the next physical boundary but
stops before winding. It densely resamples the certified articulated join at a
1 ms coupled Matter step containing sixteen 62.5 us DER substeps, then moves
the short-tail LND and needle-owning LND into the collision-clear operative
plane with the needle and PDO carried only by live finite-patch contact. Every
bounded chunk requires both grasps, the hard swage, DER geometry, contact
residual, deformable solver certificate, active tetrahedra, and zero removed
mass to remain accepted. The two material edges occupying the puncture tracts
and their binding revision must remain exactly unchanged; staging is not
allowed to hide an unintended pull-through by rebinding proxies. Two
consecutive quiescent hold boundaries are required before publishing the v3
`tissue-knot-first-throw-staged` checkpoint. That checkpoint also records the
exact right-handed knot frame, full protocol content fingerprint/sample, and
the held DER material window; a later process must continue the same throw
rather than selecting a nearby strand edge or changed trajectory. The path is
implemented but is not reported as
live-qualified until an earned `tissue-thread-acquired` checkpoint has executed
it on Apple Metal. It does not yet form a wrap, bight transfer, cinch, or
retained knot.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-knot-first-throw-stage-only \
  --resume-tissue-checkpoint tissue-thread-acquired \
  build-coupled-dev/testing/suture/tissue-thread-acquired.tsv \
  --state-output-dir build-coupled-dev/testing/suture/first-throw-stage
```

`--tissue-knot-first-double-throw-only` consumes only that exact staged
checkpoint. It resumes after protocol sample zero, densely executes both
winding turns, the bight crossing, and the opposing cinch with the needle and
short tail carried by physical LND contacts. Every completion-bounded chunk
retains the same standing material edge and two tissue-tract proxy edges,
checks both live grasps, hard swage, DER stretch/self-clearance, full needle and
instrument clearance, accepted FEM determinant/residual certificates, all
tetrahedra, and zero removed tissue mass. Promotion additionally requires at
least two radius-correct, materially separated final strand contacts plus
nonzero Metal self-contact normal and tangential impulse under the Coulomb
bound. It publishes `tissue-knot-first-double-throw` only after a quiescent
hold. The command is implemented but awaits an earned 0.82 mm-tissue staged
checkpoint and Apple Metal replay; even a passing double throw is not the
complete `2=1=1=1=1=1` knot or its retention proof.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-knot-first-double-throw-only \
  --resume-tissue-checkpoint tissue-knot-first-throw-staged \
  build-coupled-dev/testing/suture/first-throw-stage/tissue-knot-first-throw-staged.tsv \
  --state-output-dir build-coupled-dev/testing/suture/first-double-throw
```

`--tissue-knot-next-square-throw-only` advances exactly one of the five
alternating securing throws. It accepts only the immediately preceding
published throw, verifies that checkpoint against the canonical prior terminal
sample and handedness, and uses a 128-sample smooth, speed-bounded recenter
before executing the next winding, bight transfer, and cinch. The same live
grasp, hard-swage, DER, FEM, collision, clearance, and quiescence gates remain
mandatory. Each promoted phase increases the required count of materially
separated, friction-loaded strand contacts from three through seven and writes
`tissue-knot-square-throw-1` through `-5` with the newly completed protocol
sample. Repeat the command with the latest output phase until `-5`. This path is
implemented and build-qualified but still awaits an earned live checkpoint and
sequential Apple Metal replay. The final `-5` phase proves completion of the
authored `2=1=1=1=1=1` instrument sequence only; opposing-load retention is a
separate required transaction.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-knot-next-square-throw-only \
  --resume-tissue-checkpoint tissue-knot-first-double-throw \
  build-coupled-dev/testing/suture/first-double-throw/tissue-knot-first-double-throw.tsv \
  --state-output-dir build-coupled-dev/testing/suture/square-throw-1
```

`--tissue-knot-retention-only` consumes only
`tissue-knot-square-throw-5`. Both LNDs keep their physically certified end
ownership while a 128-sample smooth trajectory pulls the standing tail and
working needle 0.5 mm in opposite knot-frame directions. Every 64 ms boundary
must preserve the target-only short-tail grasp, distributed needle grasp, hard
swage, seven materially separated knot contacts, live DER self-friction, both
puncture tracts, FEM topology/mass/certificates, and operative clearance. The
accepted hold must achieve at least 0.4 mm per end and 0.8 mm total projected
separation, while both the standing-jaw tangential reaction and hard-swage
reaction reach 0.05 N from the final physics-substep impulses. It publishes
`tissue-knot-load-retained` only after two quiescent loaded boundaries. The
threshold is a conservative executable simulation gate for the source-scaled
PDO model, not package-calibrated knot-pull strength or clinical evidence; the
mode remains pending an earned full-knot Apple Metal replay.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-knot-retention-only \
  --resume-tissue-checkpoint tissue-knot-square-throw-5 \
  build-coupled-dev/testing/suture/square-throw-5/tissue-knot-square-throw-5.tsv \
  --state-output-dir build-coupled-dev/testing/suture/knot-retained
```

```sh
./build-coupled-dev/bin/metalrobo_surgical_knot_protocol_probe
```

`--tissue-suture-pull-stroke-only` turns the first remaining material stroke
into live receiver motion. It accepts either the thread-root checkpoint or a
prior stroke checkpoint, identifies the first and opposing tracts from their
deformed bite locations, and retains one exact DER material edge in each while
the receiver translates the still-grasped needle away from the proximal wall.
Each stroke is capped at 25 mm and 20 mm/s. Acceptance measures source-material
progress rather than assuming commanded Cartesian travel: the working arc must
increase by the stroke length within one DER-edge quantization band, the
inter-tract stitch span must remain invariant, the needle and complete LND
envelope must move monotonically away from tissue, and Matter mass, topology,
channels, determinants, residuals, the hard swage, and the receiver grasp must
all survive. Repeat from `tissue-suture-pull-stroke` until the command publishes
`tissue-suture-pull-complete`. The path is implemented and fast-regression
compiled, but its long Apple-Metal execution is not yet qualified.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-suture-pull-stroke-only \
  --resume-tissue-checkpoint tissue-opposing-bite-thread-root \
  build-coupled-dev/testing/suture/tissue-opposing-bite-thread-root.tsv \
  --state-output-dir build-coupled-dev/testing/suture/pull-stroke-01
```

The separate PDO clamp fixture places the authored 0.20 mm 3-0 strand inside
four finite flat proximal medial Large Needle Driver insert patches while
preserving the qualified eight-patch distal needle groove. A 65-node local
specimen is aligned with the instrument axis while the open jaws begin at a
2 mm insertion standoff. The source PSM advances for 200 ms at no more than
15 mm/s, then closes for 120 ms below its 0.16 rad/s jaw limit to establish
15 um geometric preload before the far-end support is released and the strand
is pulled at 2 mm/s. Acceptance requires bilateral normal contact plus a
frictional response distinct from an otherwise identical zero-friction Metal
solve. The checked Apple M4 run first made bilateral contact at global step
244, held it for 37/40 loaded steps, cut absolute centre-node slip by 96%
relative to the control, replayed exactly, and reported zero failed steps. This
qualifies insertion-axis approach, jaw closure, and temporary retention;
six-axis targeting of the post-bite strand and a robot-formed throw remain
separate execution boundaries.

```sh
./build-coupled-dev/bin/metalrobo_surgical_thread_grasp_probe
```

`SurgicalThreadTargeting` resolves the next targeting boundary from an actual
DER centreline. It requires enough arc length on both sides of the proposed
grasp, a locally straight window spanning the real 1.4 mm LND thread-insert
patches, clearance from a triangle surface and every finite needle capsule,
and a right-handed rail/separation/approach frame. Its deterministic probe
displaces the preferred grasp around a blocking needle, reproduces the target
exactly, and rejects both an unsafe near-surface strand and malformed surface
topology.

```sh
./build-coupled-dev/bin/metalrobo_surgical_thread_targeting_probe
```

The operative `--tissue-thread-target-only` gate accepts only a v3
`tissue-suture-pull-complete` checkpoint. A needle-tip passage or thread-root
checkpoint is intentionally insufficient: the hard-swaged DER root and the
complete steel needle must first clear the live wall, then measured
source-material transfer must leave the planned short tail while one edge
remains in each accepted tract. The target gate restores the exact Matter and
MetalWorld reset, verifies that no draw remains, and bounds every candidate to
the DER arc beyond the live first-tract material edge. It prefers the middle of
the measured ~19 mm tail while retaining finite clearance from both the tract
and free end, builds the surface from live FEM nodes, then requires
velocity-limited giver PSM IK plus sampled cross-arm, support-pad, and needle
clearance on the open-jaw approach. This command does not advance physics or
claim thread contact, frictional retention, or robot-tied knot formation.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-thread-target-only \
  --resume-tissue-checkpoint tissue-suture-pull-complete \
  build-coupled-dev/testing/suture/tissue-suture-pull-complete.tsv
```

`--tissue-thread-acquisition-only` crosses the next boundary while keeping the
same required input checkpoint. The free giver follows the certified short-
tail patch frame to a 2 mm standoff, approaches open, closes the four dedicated PDO
patches to the independently calibrated 15 um preload, settles, then moves
0.5 mm away from tissue at a bounded 2 mm/s and holds the load. It publishes a
v3 approached checkpoint before closure and an acquired checkpoint only if the
selected material edge remains in bilateral patches with normal and tangential
impulse, the receiver still owns the needle, the swage and DER remain bounded,
the puncture channels are byte-unchanged, and all FEM mass/tetrahedra and solver
certificates survive. The thread-frame IK regression explicitly measures the
2.696 mm longitudinal offset between these proximal patches and the distal
needle groove. This live acquisition command is implemented but has not yet
been qualified from an earned completed-pull v3 checkpoint.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-thread-acquisition-only \
  --resume-tissue-checkpoint tissue-suture-pull-complete \
  build-coupled-dev/testing/suture/tissue-suture-pull-complete.tsv \
  --state-output-dir build-coupled-dev/testing/suture/thread-acquisition
```

## Loaded PDO knot-contact mechanics

The rod solver now resolves a bounded network of frictional thread/thread
contacts instead of applying each crossing once. Its Apple Metal path compacts
up to 64 load-bearing contacts, performs eight alternating projected
Gauss-Seidel sweeps with accumulated Coulomb-disk multipliers, and rejects an
overflow rather than silently dropping a crossing. The FP64 oracle uses the
same alternating accumulated-impulse update.

Rod ABI v11 also preserves the friction solve's completion evidence instead
of discarding its threadgroup accumulators: every accepted rod status reports
the friction-active pair count, maximum inferred normal impulse, maximum
accumulated tangential impulse, and maximum Coulomb-disk utilization. A
frictionless crossing must publish zeros; the matched frictional crossing must
publish nonzero normal and tangential impulses, utilization no greater than
one within FP32 allowance, and bit-exact replay. This gives live knot execution
an impulse authority independent of its final geometric contact certificate.

The deterministic Apple M4 fixture uses the authored 250 mm PDO 3-0
monofilament (0.20 mm diameter), 128 rod nodes, and a pre-tied five-crossing
core under 0.5 mm opposing endpoint displacement. Fourteen edge pairs occupy
the 50 um contact shell. The shared material-separated certificate recovered
all 14 with 2.346-39.267 um surface gaps, while straight-thread and
interpenetrating controls reject. Dynamic self-friction at the explicit
research value of 0.12 reduced RMS crossing slip from 3.182 to 1.605 mm/s on
Metal and from 5.966 to 4.170 mm/s in FP64 while the Metal fixture carried
0.483 mN endpoint preload and replayed exactly. The 2.565 mm/s Metal/FP64 slip
difference remains an open nonlinear-parity boundary. This fixture qualifies
loaded multi-contact mechanics only: it is pre-tied, is not a square or
surgeon's knot executed by the robots, and is not package-calibrated or
clinical strength evidence.

```sh
./build-coupled-dev/bin/metalrobo_surgical_knot_mechanics_probe
```

## dVRK GS21 suture pickup

![dVRK Large Needle Driver lifting a GS21 needle and blue suture](docs/media/numi-lab-dvrk-gs21-suture-pickup.png)

The source-grounded Classic PSM Large Needle Driver acquires a curved GS21
through distributed opposing tooth contact without a weld, then carries its
physically swaged 25-node, 180 mm discrete-elastic-rod thread. The deterministic
3030-step qualification retained the grasp for 2000 consecutive lift frames,
lifted the needle 8.043 mm and thread root 7.905 mm, ended at 4.26 um swage
error, reported zero penetration and a `2.01e-5` maximum KKT residual, and
byte-restored the articulated/rigid rollback checkpoint.

The image is a 1280x960 Apple M4 Metal reference render from the accepted
physics state. Its 7172-triangle presentation pack is bound to articulated and
rigid body poses; the 181.527 mm rendered thread centerline comes directly from
the accepted rod state. This is simulation evidence, not clinical validation.
The thread geometry and coupling are executable; its material constants are
research defaults rather than package- or specimen-calibrated values.

```sh
./build-coupled-dev/bin/metalrobo_supported_needle_pickup_probe \
  --state-output /tmp/numi-suture-pickup.state.tsv
./build-coupled-dev/bin/metalrobo_suture_visual_probe \
  --state /tmp/numi-suture-pickup.state.tsv \
  --output-dir /tmp/numi-suture-visual
```

Current Apple M4 qualification covers compiler/package invariants and live
Metal contact, multiphysics, sparse-MPM, articulated-coupling, and reference
render paths. These checks establish the recorded workloads; they are not a
blanket hardware or performance claim.

# Synthetic skin stitch handoff — 2026-09-30

Base revision: `298e5f8565d1cea297c705572493c8d29934d76e` (`coupled`).
Implementation lives in isolated worktree `/Users/home/numi-skin-suture-20260930`.
This checkpoint is source and native probe evidence, not a completed stitch.

The opt-in `--synthetic-skin` mode compiles a homogeneous 30 x 24 x 1.5 mm
FEM specimen with a 16 mm incision and separate lips at a 0.6 mm gap. Its
neo-Hookean constants are authored simulator values. The existing two-PSM
needle, swaged DER thread, Matter puncture/contact, and incision topology are
reused. The default porcine-jejunum mode remains selected without the flag.

Native M4 probe results from the build in `build-skin-wound`:

| Mode | Observed result |
| --- | --- |
| synthetic skin coupon | 1,328 FEM nodes, 5,184 tetrahedra, 36 lip pairs, exact 0.6 mm authored gap |
| tissue coupling | one accepted contact, 49.6 nm maximum tissue displacement, zero failed steps |
| tapered puncture | one accepted channel, 126 um tract, zero removed tissue mass, zero failed steps |
| channel advancement | one accepted channel, 200 um signed needle advance, zero failed steps |
| opposing-bite topology | 65,280 tetrahedra, 836 contact nodes, both bite targets resolved; no GPU dispatch in this mode |
| live suture entry | four accepted contacts, one channel, thread root error 0.139 um, zero failed steps |
| curved through-wall passage | 372 Matter groups / 1,489 DER substeps, 1.836 mm tip advance, 0.200 mm deformed distal clearance, 12 connected channels, all 65,280 tetrahedra retained, zero failed steps |
| compact-mesh curved passage | 372 Matter groups / 1,489 DER substeps, 0.200 mm deformed distal clearance, 12 connected channels, all 46,080 tetrahedra retained, zero removed mass, zero failed steps |

That long passage used the 34 x 40 x 8 baseline mesh. A 30 x 32 x 8 operative
mesh now retains the 1.5 mm wall and 0.6 mm wound gap while reducing the
tetrahedra to 46,080. Its local graded spacings are 0.144, 0.188, and
0.188 mm, all within the 0.2 mm strand activation reach. The compact mesh
passed topology, live suture entry, and grouped cadence checks; its entry
GPU time was 3.816 s versus 6.055 s for the baseline, and one grouped
cadence step took 3.739 s versus 5.608 s. These short matched timings do not
establish an end-to-end passage speedup or long-horizon correctness.

Four focused synthetic-skin CTests pass on the current build. The existing
porcine-jejunum coupling mode also passed after the initial skin integration.
The completed baseline passage binary SHA-256 was:
`1ff94c23cd21735ee3fad3975dd0a7c39ea01dc5890926683842c93b9da03bb2`.
Synthetic material SHA-256:
`379945f593f12396b44c7148026f0250344a86278e01b7b26cb5f6d823a51e35`.

A native `--tissue-suture-passage-only --synthetic-skin` run from that binary
completed successfully and is logged at
`build-skin-wound/synthetic-skin-passage.log`. Passage GPU time was 2,149.479 s;
its final accepted state hash was `0xf615a90f42e6990` and the log SHA-256 is
`620948112583777901597435fec7dc38ffbf498270c2c8da83b83920404ceee7`.
The separate
`--tissue-curved-passage-only` mode fails
at the post-entry cadence switch for both skin and jejunum because it omits
the rod-contact capability required by that switch. The suture-contact
cadence mode passes at multiplier 4. Source now also measures the center lip
gap after knot loading and requires a reduction for the skin variant. The
rebuilt topology test passes and measures nine central lip pairs at 0.6 mm;
the knot-loaded gap gate itself still awaits a complete native sequence.
The current compact-mesh probe binary SHA-256 is
`ecd5a49555186ad1089717f2faedfbaaacbe6e3e1fb475ecb9d007f3b87e7b20`.
A compact-mesh `--tissue-curved-pull-through-only --synthetic-skin` native run
from that binary passed the through-wall passage gate with 1,278.393 s GPU
time and state hash `0x15382e1005422770`. The fixed passage-prefix log is
`build-skin-wound/synthetic-skin-compact-passage.log` with SHA-256
`f1b6ad275c159becf46a9fa2ca0fd620911203968b19aa72b9473fcbdcae6b0e`.
The earlier full-mesh passage used
2,149.479 s GPU time; these are single-run observations, not a general
performance qualification. Maximum skin-node displacement in the compact
passage was only `2.91038305e-11 m`; it establishes a resolved puncture tract,
not a visibly deforming or cinched wound. The subsequent thread pull was
stopped after 5,040 base DER substeps and 480 Matter steps, at orbit angle
3.26982416 rad and thread-root distal clearance -0.620692481 mm. It had zero
sampled strand contacts and zero strand reaction, so pull-through remains
unqualified. The partial log is
`build-skin-wound/synthetic-skin-pull-through.log`, SHA-256
`e1a0629d4606a1d172fd04026a4fb5b588826e646365c5bca3baee633b95350d`.
The run's executable retains the SHA-256 above. A later rebuild had SHA-256
`1993cf25e00ff2c683c6a7ea8d2db36375d5646d7c68f0f027c41ac06748c76c`;
it adds live center-wound-gap mean and maximum measurements to pull-through
progress and final output. Those fields are absent from the stopped run, and
this rebuild alone is not pull-through qualification.
The original dynamic-needle grip reset hit both jaws but started only 10 um
from the skin and struck a premature puncture channel while relative motion
remained unqualified. The revised live mode stages the skin 0.5 mm from the
needle and includes that approach distance in its planned 235-step bite.
It holds the closed giver through 11 grouped Matter steps / 44 base DER
substeps. `--tissue-robot-first-bite-grip-only --synthetic-skin` now passes:
8/6 jaw contacts, complete 15/15 insert masks, 89.63 um seat drift against a
100 um limit, 1.249 mm/s relative point speed against 2 mm/s, and 0.524
rad/s relative angular speed against 0.6 rad/s. The accepted state has zero
puncture channels, all 46,080 tetrahedra, zero removed tissue mass, minimum
determinant 0.999999821, 4.449 um swage error, and 1.367 um maximum thread
edge error. The gripper is reset closed at the needle; no physical approach
or pickup has yet been demonstrated.

`--tissue-robot-first-bite-drive-only --synthetic-skin` now passes a further
12 grouped Matter steps / 192 base DER substeps / 12 ms using articulated
efforts and no kinematic needle target. The planned free-space tip advance is
52.45 um; the measured dynamic tip advance is 25.64 um. Final jaw contact is
8/6 with full 15/15 insert masks, 0.344 um seat drift, 0.0379 mm/s relative
point speed, zero puncture channels, all 46,080 tetrahedra, zero removed mass,
minimum determinant 0.999999821, 0.927 um swage error, 0.545 um thread-edge
error, and zero failed steps. Drive GPU time was 72.324 s. This proves a short
robot-driven free-space approach with an accepted grasp, not a through-wall
bite. The stable native test log is
`build-skin-wound/synthetic-skin-robot-approach.log`, SHA-256
`0ed74e6cef1d4a0a4b6d4af1ec08267c1556cce1397659e830c02e90b79d36a1`;
the executable SHA-256 is
`8da12e63bfe71135267cbd97bad5bd49438f078bf9cef481a00ed91965a58ad3`.
The same accepted run published both grip and approach checkpoints under
`build-skin-wound/skin-robot-checkpoints/`. The approach TSV has SHA-256
`800117ee5da4ff211d91a3c36191169a6cfdd223323fb8f33a5a30953ac5a060`;
its 30,316,752-byte live Matter snapshot has SHA-256
`f74e4fbf58262625e40a75a69c63fc632bc9f3a4423fefb1d05bab7308a98acc`.
The visual probe accepted that snapshot as `synthetic_skin_wound` and rendered
`build-skin-wound/skin-robot-visual/handoff-close.png` plus
`handoff-overview.png` at 1280 x 960. It reported 6,336 tissue triangles
from verified live FEM geometry; the overview PNG SHA-256 is
`a9397b38a9de6fe8143ad8420dcdb0f2e5e35080cc3c416fe48102f283dcb853`.
The current evidence JSON SHA-256 is
`8ba15009b9fbf9b972905d3eda469696e2b1888955f524d119e751b0725fd280`.
This is one rendered approach state, not a motion video or completed stitch.
Checkpoint publication now writes the actual `synthetic_skin_wound` material
identity and 30 x 24 x 1.5 mm specimen dimensions when `--synthetic-skin` is
selected; the visual probe accepts that identity only with a live v3 Matter
snapshot and verifies the dimensions before building a surface. The accepted
approach checkpoint above now exercises that path. Earlier handoff and
visual-probe binary SHA-256 values were
`22acd0b40cf42c03195af9a69ca841de000bca94383a7f820cb2f33ea64a907d`
and
`969e16939cb8697c65ee909d2dd0a1c9c13f375cf3552a5513412ac55edc080b`.
The current renderer binary SHA-256 is
`30fe482ee15d36ca0cd3710feb71a52e5fffaa93d5258d89e1e54ffee0532432`.
Its visual evidence JSON carries the verified tissue-model identity.

The accepted robot approach is now a resumable v3 checkpoint. This command
restores its complete Matter snapshot and MetalWorld reset state, then checks
the zero-channel pre-puncture skin invariants without advancing physics:

```sh
./build-skin-wound/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-checkpoint-restore-only --synthetic-skin \
  --resume-tissue-checkpoint tissue-robot-first-bite-approach \
  build-skin-wound/skin-robot-checkpoints/tissue-robot-first-bite-approach.tsv
```

It passed byte-exact Matter authority, all 46,080 tetrahedra, zero puncture
channels, zero removed mass, and positive minimum determinant in 0.65 s wall
time. The new `--tissue-robot-first-bite-continue-only` mode starts from that
checkpoint and drives the giver's joints without prescribing needle motion.
Its 16x diagnostic run advanced the dynamic needle tip 59.91 um during 12 ms
of simulated time, with 6/6 giver-jaw contacts, zero puncture channels, all
46,080 tetrahedra, zero removed mass, and zero failed steps. GPU time was
72.490 s; the log is `build-skin-wound/synthetic-skin-robot-continue.log`,
SHA-256 `0f6ad3bdcdcf2095eded0a1ee3fa35da38a539164e310cf52fdda0bfe75be79c`.
At 32x cadence from the **same input checkpoint**, 12 Matter groups / 384
base DER substeps represented 24 ms, with 119.87 um planned and 111.12 um
measured tip advance. It retained 6/6 jaw contacts, 3.02 um seat drift,
0.299 um swage error, 0.428 um maximum thread-edge error, zero puncture
channels, all 46,080 tetrahedra, zero removed mass, positive minimum
determinant, and zero failed steps. GPU time was 96.648 s, or 1.50x the 16x
run's simulated-time throughput. Wall time was 98.15 s. The log is
`build-skin-wound/synthetic-skin-robot-continue-32.log`, SHA-256
`47023ead087fc1fb94215c0622fd6d1a3cf2025cf1b6d3344178761c5e75a429`;
the executable SHA-256 is
`6d97bbd1c9b784e47560d229935dca81b3b0b039f3f4a60f97961b7d51f5be9d`.
Its output checkpoint is
`build-skin-wound/skin-robot-continue-32/tissue-robot-first-bite-approach.tsv`,
SHA-256 `4af105b63c7243fb4a536ed68a25e404ade3de0a9763b0496a3ce8ff19f163cc`;
the linked Matter snapshot has SHA-256
`1c100ab468f52978e5485923abd983f12566064fd9d7215d69431e9da0433278`.
That newer checkpoint also passed byte-exact restore at state step 621. This
is still free-space approach, not needle entry or a completed stitch.

A phase-local two-Newton trial at the same 32x cadence reproduced every
MetalWorld checkpoint TSV field from the cooked seven-Newton run except the
Matter archive reference. Its Matter residual was 7.8104e-7 against the
unchanged 5e-4 bound, and GPU time fell from 96.648 to 68.113 s for the same
24 ms and 111.12 um dynamic tip advance: a 1.42x gain on that matched
segment, or 2.13x simulated-time throughput relative to the earlier 16x
continuation. The trial log is
`build-skin-wound/synthetic-skin-robot-continue-32-newton2.log`, SHA-256
`1eb68446926aa2bcf30576293b5a5181aca59dc3add7bb980442c7a34d176463`.
One Newton pass also met broad physical bounds and took 60.992 s GPU but
changed the saved robot and thread state; it was rejected for this exact-state
free-space shortcut. Reducing the FGMRES budget from 32 to 16 with two Newton
passes reproduced the state but took 67.800 s GPU, only 0.313 s less than the
two-pass default. The production free-space continuation therefore retains
the cooked FGMRES budget and selects two Newton passes only if the live
tapered-tip contact-node separation exceeds the maximum planned capsule
sweep by at least 0.2 mm. No residual, contact, or topology threshold was
relaxed.

From the two-pass checkpoint at state step 621, another guarded 32x segment
passed and advanced the dynamic tip 113.09 um. The modeled minimum
tapered-tip-to-skin contact-node separation fell from 0.3768 to 0.2640 mm, while
8/6 giver-jaw contacts, all 46,080 tetrahedra, zero puncture channels, zero
removed mass, and zero failed steps remained intact. GPU time was 68.130 s.
The resulting v3 checkpoint passed byte-exact restore at state step 1005.
Its run log is `build-skin-wound/synthetic-skin-robot-continue-next.log`,
SHA-256 `c216f5e8beeb8e05d7950f60c6676d61d9a9c3e8d2f98ad24fc42dd891abf9e4`;
its TSV SHA-256 is
`327a055f0e514291efe19206ff86891cfcb566b249fc02ddd188bca9b00116bb`.
The linked Matter snapshot SHA-256 is
`2b0e41052eed983971dea2cc0da55adf3417cdf72d5482dd0b9fe15ba841acc6`.
The next identical shortcut was rejected before physics because its planned
sweep could enter the skin contact band. The free-space executable SHA-256 is
`cc09c970d2aea6387b45c324a49d7aa5ffb0c852e8948ce078e36a402096edf5`.
No free-space result proves a robot-driven puncture.

`--tissue-robot-first-bite-contact-approach-only` restores the step-1005
checkpoint, explicitly returns from the saved two-pass budget to the cooked
seven Newton passes, and performs another bounded 32x/24 ms articulated
approach. It passed: 113.78 um actual dynamic tip advance; tapered-tip
contact-node separation 0.2640 to 0.1515 mm; 8/6 giver-jaw contacts, full
15/15 insert masks, 0.614 um seat drift, 0.339 um swage error, 0.501 um
maximum thread-edge error, all 46,080 tetrahedra, zero puncture channels,
zero removed mass, positive determinant, 7.4556e-7 maximum Matter residual,
and zero failed steps. GPU time was 97.081 s. Wall time was 178.47 s on this
run; the larger wall/GPU gap is unassigned, so use GPU time only for the
solver comparison. The v3 output checkpoint restores byte-exact at state
step 1389. Its run log is
`build-skin-wound/synthetic-skin-robot-contact-approach.log`, SHA-256
`cbb8a401a770d1b1d7c31d5c5ba5a500580bea30d295d792b176efc07b152878`;
the TSV SHA-256 is
`f18bc4025bd23ba7702fa9a34a9e09d10f5b949d0218857148dafc4167e91796`;
the linked Matter snapshot SHA-256 is
`ffb2f600fec31f54d59caad22d0bea72978e67b512659ff2e167df152030aacc`.
The executable SHA-256 is
`c8152cc2131f10e3d8bd2f7c8ad04c6058bd5de788629094190939dc26867b91`.
Repeating the pre-contact bridge from step 1389 is rejected before physics:
its planned sweep could cross the 100 um contact guard. The next bounded modes
measure actual needle/skin contact, deformation, and reaction at the full
solver budget; this bridge alone proves a loaded robot approach to the
contact boundary.
The native visual probe also accepted the step-1389 checkpoint's
`synthetic_skin_wound` identity and live FEM snapshot, rendering 6,336 tissue
triangles in `build-skin-wound/skin-robot-contact-visual/handoff-close.png`
and `handoff-overview.png`. The overview PNG SHA-256 is
`e1a11cc1bcafe5c607156658bf7f6c2fd7a8c99623efd61fa31c5b2078b76688`;
the visual evidence JSON SHA-256 is
`01b2821a79455b0ebd76f2ae07f520ff4c0be494f7977e45cca8525fa5eeaa6f`.
This is a rendered pre-contact state, not a motion video or puncture proof.

`--tissue-robot-first-bite-entry-only` restored the step-1389 checkpoint and
advanced the robot-held dynamic needle through 16 full-seven-Newton Matter
groups. The tip moved 82.01 um, reducing the minimum modeled tip-to-skin-node
distance from 151.51 to 71.82 um. One tapered-tip Matter contact carried an
accepted 1.014e-8 N s normal impulse; the tissue reaction was 1.014e-8 N s
and the largest FEM-node displacement was 0.466 nm. All 46,080 tetrahedra
remained active, with zero removed mass, zero puncture channels, minimum
determinant 0.999997377, maximum residual 1.020e-6, and zero failed steps.
GPU time was 101.181 s for 16 ms of simulated motion. The state at step 1645
passed byte-exact v3 restore. The log SHA-256 is
`c476285348dad3bd7b156653a12a5de9f5b601df7d09345c7add87a5a07481f1`;
the checkpoint TSV SHA-256 is
`02fba6e790c89e409b9d30e0f0ef7161ad8f6949c53e483a86af965ac0bb89e0`.
The visual probe rendered its live 6,336-triangle FEM surface in
`build-skin-wound/skin-robot-entry-visual/handoff-close.png`, SHA-256
`1b5b1a7a914a6c2aae80e45519b3a4b0734cc9f24ef95e6137b21985de84f743`.
This is still a single still image, not a stitch video.

`--tissue-robot-first-bite-contact-advance-only` then restored step 1645
and passed an eight-group, 16x, seven-Newton continuation. The tip advanced
another 50.47 um to a 21.44 um modeled skin-node distance. Its one accepted
tip contact carried 6.992e-8 N s and tissue reaction 7.032e-8 N s; the
largest FEM-node displacement was 2.675 nm. All 46,080 tetrahedra remained,
zero channels and zero removed mass, minimum determinant 0.999983966,
maximum residual 2.913e-6, and zero failed steps. GPU time was 53.873 s for
8 ms of simulated motion. The step-1773 state passed byte-exact restore.
The log SHA-256 is
`c7a18e6df595002a4df112042b73d5e998da0b318b1c77a42263fa3a5d61588d`;
the TSV SHA-256 is
`186b72c0232e2fadcedfe3b288e1dfcadc2767c2b1e1a544e41d44a5bf76968c`.
The native puncture impulse gate is 5e-7 N s; this prefix did not earn a
puncture channel. A longer continuation from step 1645, and a short 8x
continuation from step 1773, both rolled back at the unchanged Matter contact
feasibility gate (`NM_STATUS_CONTACT_FAILURE`, code 6). Increasing the Newton
budget to nine did not repair that failure. Step 1773 is the last accepted
contact state that also satisfies the terminal grasp-speed limits.

The separate `--tissue-robot-first-bite-puncture-microstep-only` mode restores
step 1773, uses one 62.5 us Matter/DER group with the full seven-Newton
budget, and drives the PSM along the needle orbit at a 1 mm/s target speed.
The prior contact pair (skin node 8570, tapered-tip proxy 0) was 21.87 um
apart and had 3.63 mm/s inward admission velocity. The native microstep
created one connected puncture channel with 1.138e-6 N s accepted tip impulse
against the authored 5e-7 N s gate, 1.138e-6 N s tissue reaction, all 46,080
tetrahedra active, zero removed mass, minimum determinant 0.999984324,
maximum residual 2.588e-6, and zero failed steps. It was **not** a qualified
robot grip endpoint: the 0.149 um seat drift was small, but relative point
speed 2.455 mm/s exceeded the 2 mm/s limit and relative angular speed
1.439 rad/s exceeded the 0.6 rad/s limit. The phase is explicitly
`tissue-robot-first-bite-puncture-transient`, not a completed bite. Its
step-1774 checkpoint passed byte-exact restore. The log SHA-256 is
`493b5de3f8815bbef1aa16edb402e5e4231ee6a44711ca1a2b46e1de970ca370`;
the TSV SHA-256 is
`1dba07020b808babea41aa560bbbb527f99021c2e68267c38a7a8acad9d3d54c`.
The final rebuilt probe reproduced that TSV byte for byte; its executable
SHA-256 is
`c394ed8b8a8f521c55ca38e0ae6db8feef448f77b4da0277c78c1e166a4e49d5`.

The same mode resumed that exact transient state for four further 1x
microsteps. It retained one active channel, all tetrahedra, zero removed
mass, and zero failed steps through step 1778, which also restored byte
exactly. Relative point speed fell to 0.364 mm/s but angular slip rose to
3.230 rad/s; the terminal grasp gate remains unmet. The final sampled tip
contact and reaction were zero, so these four steps prove channel retention,
not a second puncture impulse. The continuation log SHA-256 is
`b9a7bda61aa7965942e77ac33eb22f2604ba2837434e2f47bdd9cbc69a50f415`;
the TSV SHA-256 is
`3b83ab70d2f790ca5eb610f5935a984b119de119c51e6157f374bad25d4daa8a`.
A four-microstep zero-command diagnostic from step 1778 still advanced the
dynamic tip 11.77 um under its carried momentum, with angular slip near
2.99 rad/s. It failed the acceptance gate and published no checkpoint; its
diagnostic log SHA-256 is
`8c5c0de56f02c0b7f83bca8e1677b782873c168d85d76e303f22c2d7a5830f15`.
Robot grip and braking must control that momentum before this transient
channel can become a qualified robot-driven bite.

For the 100x performance request, an opt-in
`--tissue-robot-first-bite-continue-fast-only` groups the same 24 ms
pre-contact interval into six 64x Matter steps instead of twelve 32x steps.
The named fast mode reproduced its first trial's checkpoint byte for byte
and passed byte-exact restore. Its GPU time was 58.031 s versus 68.113 s
for the 32x/two-Newton path, a 1.17x
improvement for this segment. It advanced the tip 112.00 um versus 111.12 um
and changed saved robot and thread fields, so it is a distinct experimental
trajectory, not an exact-state replacement for the 32x contact-checkpoint
lineage. The verified fast-mode log SHA-256 is
`010cd0dc83a7df418698abb19f7be10f26db210822f0a22d670774a877cbd580`;
the checkpoint TSV SHA-256 is
`f0dcaf19034d516cb98b859287a5ee17b8e6e76ca7975e2844fd73bf073883ba`.
The 100x target on this matched segment would require at most 0.681 s GPU
from the accepted 68.113 s baseline. Further cadence increases are not an
exact-state optimization, and the contact phase still needs its full solver
budget. The later performance checkpoint profiles the contact broadphase
and measures one algorithmic improvement. No full robot-driven stitch or
100x end-to-end speedup has been measured.

The `--tissue-robot-first-bite-ik-only --synthetic-skin` geometry probe found
a 250-step giver approach and 185-step 5 mm/s needle-orbit path at the
original 10 um clearance. The entry
jaw midpoint is 0.164 um from its needle seat; peak approach and bite joint
velocity ratios are 0.454 and 0.746 of their limits. An authored-surface
envelope scan of the 250 approach and 185 bite samples found at least 0.900
and 2.257 mm needle-driver clearance from the skin respectively, with no
unsafe sample. The planned tip travels 1.817 mm and clears the authored distal
surface by 0.181 mm. This proves reachable,
collision-free command geometry only. The live 0.5 mm staging clearance
extends the planned orbit to 235 steps and retains 0.175 mm planned distal
clearance; its through-wall physical result is still untested.

## Performance checkpoint

The stopped pull used the original 1x Matter cadence near the tract and took
about 24 s of GPU time for each eight 5 um microsteps. To bound a faster
schedule, `--tissue-suture-cadence-baseline-only`,
`--tissue-suture-cadence-only`, and
`--tissue-suture-cadence-fast-only` now run a matched post-entry one-step
probe at 1x, 4x, and 16x from the same authored reset. On the M4 the grouped
GPU times were 3,375.995, 3,739.432, and 5,193.212 ms for 62.5, 250, and
1,000 us of simulated time. That is 3.61x and 10.40x simulated-time
throughput relative to 1x. All three probes retained 46,080 tetrahedra, the
one accepted puncture channel, zero removed tissue mass, positive minimum
determinant, and zero failed steps. The strand did not contact tissue in the
post-entry benchmark step, so these timings do not qualify grouped
load-bearing pull-through or whole-stitch performance.

The candidate pull schedule now uses 16x away from contact at 80 mm/s, 4x
near contact at 80 mm/s, and 16x after slowing to 20 mm/s. Both contact
settings advance 20 um per Matter transaction, below the authored 100 um
contact band and 25 um swage positional allowance. The setting compiles;
the complete strand-contact and reaction gates still need native validation.
The 100x target is unmet. Reaching it for the same full-resolution 46,080-tet
coupled problem would require roughly 33.8 ms per base-equivalent step versus
the observed 3.38 s. The Metal System Trace at
`/tmp/numi-skin-entry.trace` captured one entry step, but shader timeline
was disabled, so it does not attribute per-kernel costs. The benchmark log is
`build-skin-wound/synthetic-skin-cadence-benchmark.log`, SHA-256
`6af9f06ac1385c744ef7077a8a302a45f7a59398cefe0a3b4e71e09d48415429`;
the executable SHA-256 is
`30352b18ef5fe43b51be296f6181b2c377a9788662789251442b7370a73bfc38`.

An encoder-boundary Metal timestamp probe on the M4 attributed about 372 ms
per deformable-candidate build (five Newton passes plus the final certificate)
in the original post-entry skin microstep. The old broadphase checked surface pairs
quadratically in one SIMD group. Sorting by each surface's lower x bound
lets the same AABB test stop once the next lower bound exceeds the left
upper bound. A 100-seed, 200-box randomized negative/positive-coordinate
check found the same overlapping pair sets as all-pairs enumeration. With
the x sweep, candidate building took about 134 ms per build. The
unprofiled 62.5 us native post-entry step fell from 3,375.995 to
1,927.866 ms GPU, or 1.75x throughput; every published physical field
matched the prior run, including one puncture channel, all 46,080 tets,
zero mass loss, and zero failed steps. Six focused surgical, mixed-FEM,
topology, cohesive, and puncture tests passed. The final run log SHA-256 is
e6510dcf37dd36867a3edd45cfa3aaf23494c9a1d7d0c9de97716d146f860e87;
the executable and Matter metallib SHA-256 values are
c394ed8b8a8f521c55ca38e0ae6db8feef448f77b4da0277c78c1e166a4e49d5
and f594c387b3d34a000d8ab80e3049d76e5bea650901d4c4a9012fb7d1c3d8bf1a.
That intermediate measured segment was still 57x slower than the 33.8 ms
target implied by the requested 100x speedup from the original baseline.

The next candidate keeps the x sweep and parallelizes candidate enumeration
across left surfaces. It counts eligible pairs per surface, computes
deterministic capacity-bounded offsets, then scatters in the x-sweep
left/right order. The temporary sort buffers serve as count and offset
scratch, so no new resident allocation is required. A 100-seed randomized
check with 200 boxes per seed, signed coordinates, and five capacity cases
matched the serial sweep's ordered pairs. The final unprofiled post-entry
GPU times at 1x, 4x, and 16x were 1,132.357, 1,494.502, and 2,952.470 ms
versus 3,375.995, 3,739.432, and 5,193.212 ms on the prior shader.
That is 2.98x, 2.50x, and 1.76x on the three matched cadence modes.
All published physical fields matched their respective prior runs,
including the puncture channel, 46,080 active tets, zero removed mass,
positive determinant, and zero failed steps. Six focused tests passed.
The final log SHA-256 values, ordered by 1x/4x/16x, are
849dbe70082ef23992324a889bc979040db759dd3afada44ba2ad7d3d05d0f45,
bb0c3ab76026c8fcfc4fb84d775bcb1ea52045d92d237b1a6d1b689dbed75018,
and 8c99e5a5f06a5feedc156c2a2b9c3b5575773fa0376f46ab647fd0b857b3b4bf.
The linked library and shader SHA-256 values are
dafd704ac5a81a7437666632560913002495bd9c7fa0c0fc57723efc83bfb671
and 7d981a2606f7dbc5dcc723b388cbbec7c507ebc995592e6d98135606e5cb5b08.
The final 1x segment is still about 33.5x slower than the 33.8 ms
100x target. These timings do not establish whole-stitch or robot-contact
speedup.

The changed shader/program fingerprint correctly rejects the earlier
robot tip-contact checkpoint. Its diagnostic restore log SHA-256 is
2434a35fafdb8c3dce0f48d574f7ddb6f8580f4e137fd060763045f295d2f5bf.
Keep the original binary for the old accepted checkpoint lineage and
regenerate a new lineage under this shader before resuming the robot bite;
no old state was silently upgraded.

The stronger 75 um jaw-preload trial from robot contact step 1645 also
remains unqualified. A 13 ms ramp at 1 mm/s preserved Matter's 46,080
tets and zero failures but drifted 212 um at the needle seat. A 6 ms
partial ramp at 5 mm/s reduced seat drift to 28.9 um but produced
20.0 mm/s contact-relative point speed and 2.11 rad/s angular slip,
above the 2 mm/s and 0.6 rad/s grasp limits. Neither published a
checkpoint; their log SHA-256 values are
f19e715d8ee5a9fc77345d942aa2b49f7ec8660a0f386acbc3f77fb88f2c0229
and 0cb969049397267ee58c6bda20d852f896a04f4dd028a41d5191246033b224bc.

Open work: physically approach and pick up the needle with the robot, drive
it through the first skin lip with measured tissue reaction, qualify thread
pull-through with this 1.5 mm wall, then the second lip bite, knot tightening,
gripper release, and measured unloaded wound-gap retention. Existing
jejunal checkpoints must not be reused as skin evidence; generate fresh
skin checkpoints and compare their material/world fingerprints. Capture
gap, thread tension, needle reaction, tissue strain, contact, work/energy,
and exact replay before claiming a complete interrupted stitch. No skin
calibration or clinical validity is established.

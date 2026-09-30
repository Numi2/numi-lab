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

The resumed six-step, 64x robot approach now reports the maximum MetalWorld
required-contact count, normal velocity residual, and cone violation across
every step. With the regenerated first-bite approach checkpoint (byte-exact
restore, TSV SHA-256
`80de3e32f766e99f3f39a4daee417fab38395cf265bf55dc6ebcb72fbd44472c`),
the M4 ran a matched 16- versus 32-velocity-iteration comparison over the
same 384 base substeps. The 16-iteration diagnostic took 39.516 s GPU versus
62.576 s at 32 iterations, a 1.58x segment speedup, but the maximum normal
residual rose from 0.002068 to 0.003482 m/s. Both exceed the existing
0.002 m/s terminal contact-residual bound when applied conservatively to
every step. Both retained all 46,080 tetrahedra and passed the grasp and rod
checks with zero failed transactions, yet their saved needle/thread fields
and tip travel differ. The lower-iteration trajectory is therefore an
unqualified diagnostic, not a replacement checkpoint or a 100x result.
The 16- and 32-iteration log SHA-256 values are
`73ae91710802adc7ffc3b42c7dc5c7e5dbc38e9e30af5e5a78b176553962cb86`
and `d612e6b15cdfef3315ad971c35af21a1fa5653f1b2023b8824818ea982b20cf5`;
the executable and Matter metallib SHA-256 values are
`03d74fa92f73af6e7ab8c8ca7f180c5f7e257bb3b7b0268738242a48cfe3c944`
and `7d981a2606f7dbc5dcc723b388cbbec7c507ebc995592e6d98135606e5cb5b08`.
Two body-membership cache variants in the Wave32 contact kernel preserved
the post-entry probe result but did not improve its 1x or 16x GPU timing;
both were reverted. The measured bottleneck remains the contact solve, so
the next optimization needs to reduce work per validated contact iteration
without relaxing residual or topology checks.

The Wave32 rod impulse update now assembles and solves in cohort-private
threadgroup memory while retaining the same device-resident factor, contact
order, 32-iteration schedule, and terminal residual. The matched six-step
moving-gripper probe took 48.855, 48.850, and 48.854 s GPU, versus
62.576 s before the change: about 1.28x faster for the same 24 ms interval.
Every reported physical field matched. More decisively, its saved first-bite
checkpoint TSV and 30 MB Matter snapshot were byte-identical to the prior
run (SHA-256 `eba8613a1dc9d645ddd3784725727abc7141731ae7dc706ac9fa6d5b7609fae2`
and `ea00a176ecfb6f3854d58a2f484147561937f6d31e93e736b36919a98745e9cd`).
The post-entry 16x/one-Newton step took 2.220 s GPU in two runs, versus
about 2.329 s with device-memory scratch, and retained its puncture channel,
all 46,080 tetrahedra, and 0.000868 m/s MetalWorld contact residual.
Four focused surgical tests passed. The robot trial log SHA-256 values are
`ebe214aa6eae22b85f25a22837c623509e4ff464240d9908a0be2632b3b392e8`,
`4fd92323d122f55dc06f2651a3cd4c5c8b2a8991c7539ba41b0a7b8d06081c42`,
and `16154bf092ca5fa1d25729c97f11af500e60bae05c9c78e72cae17a7114c89ab`;
the MetalRobo metallib SHA-256 is
`08531cb0270799f4e2826e27ce0e6fb9ba4eba1323344a9662a5ba382041295e`.
This is an exact-state performance gain on one robot-approach interval. Its
maximum contact residual remains 0.002068 m/s, above the conservative
per-step 0.002 m/s bound, and neither a complete stitch nor 100x
end-to-end speedup has been established.

A follow-on single-rod packet trial copied the immutable factor into
threadgroup memory as well. It preserved the one-step physical fields but
changed the 1x GPU time from 1,123.130 to 1,122.724 ms and the 16x time from
2,220.523 to 2,217.097 ms. That gain is too small to justify the extra
shared-memory occupancy, so the factor cache was reverted; the validated
cohort-private impulse workspace remains. The trial log SHA-256 values are
`14649739434ca545ced835736d1584606df9e8e1d1854a79804273b7148014c4`
and `9302b28441f37a9ede0387b5c84a3bbea66869be2d582e5b1b66b4cd28ae9228`.

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
do not silently upgrade its checkpoints.

The robot lineage was regenerated from the authored reset under the faster
shader. Each accepted checkpoint was resumed with its matching program
fingerprint. The initial approach and terminal puncture checkpoints both
passed byte-exact restore. The bounded phases retained the previously
published robot, tissue, contact, and solver fields, including 46,080 active
tetrahedra, zero removed mass, and zero failed steps. Single-run GPU times
for matched phases were:

| Phase | Earlier GPU s | New GPU s | Speedup |
| --- | ---: | ---: | ---: |
| First robot drive | 72.324 | 36.472 | 1.98x |
| Free approach, step 237 to 621 | 68.113 | 55.997 | 1.22x |
| Free approach, step 621 to 1005 | 68.130 | 54.529 | 1.25x |
| Full-solver approach, step 1005 to 1389 | 97.081 | 60.738 | 1.60x |
| First tip contact, step 1389 to 1645 | 101.181 | 53.013 | 1.91x |
| Contact advance, step 1645 to 1773 | 53.873 | 29.913 | 1.80x |
| One puncture microstep, step 1773 to 1774 | 5.240 | 2.232 | 2.35x |

The seven phase GPU times sum to 465.942 s earlier and 292.894 s now,
or 1.59x for this bounded robot trajectory. This sum omits setup,
checkpoint I/O, and later stitch phases; it is not an end-to-end stitch
benchmark.

The new logs in table order are `skin-robot-parallel-drive.log`,
`skin-robot-parallel-continue-1.log`, `skin-robot-parallel-continue-2.log`,
`skin-robot-parallel-contact-approach.log`, `skin-robot-parallel-entry.log`,
`skin-robot-parallel-contact-advance.log`, and
`skin-robot-parallel-puncture.log`, all in `build-skin-wound/`. Their SHA-256
values in order are
`4b72d2fef8570b4976fbada4e5f4f4967f77e3f87881d5bc6ea7aba39b4a17c1`,
`df409cb1ea8533a149961010fa9662132cd94ef06dfd3c732e5b03b5e8979b3e`,
`9a5a48a3a9494aa3b76b9b584ec6a592987d6691fa02ee81e7ba5b802030d93d`,
`b7609e52e891f0c07f4788bfea7f2a09836135ef5eb444d768da1d50cd9496b8`,
`fcf35f2bc6371697931a5a2b251408848e2fc610cb5487cd5395664c77c05a8f`,
`316632d286461ab0facec1fc19a77675275934ff5c18b0289d53297dedd2fbab`,
and `ee259c2031864740cc874dc17fb9059ec3b538137a598d55b60789957547d839`.
The new step-1774 checkpoint TSV and linked Matter snapshot SHA-256 values
are `4dbb7408e85bd415a1138a506f4cd3ae124e42c89c053eb5e0a47ce90300a94a`
and `575069f55dc6647269e640fc520dffc4e28c83f6039c3cdd979097e383db1b7d`.
The checkpoint can be verified without advancing physics:

```sh
./build-skin-wound/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-checkpoint-restore-only --synthetic-skin \
  --resume-tissue-checkpoint tissue-robot-first-bite-puncture-transient \
  build-skin-wound/skin-robot-parallel-puncture/tissue-robot-first-bite-puncture-transient.tsv
```

The executable, linked library, and Matter metallib SHA-256 values are
`c394ed8b8a8f521c55ca38e0ae6db8feef448f77b4da0277c78c1e166a4e49d5`,
`dafd704ac5a81a7437666632560913002495bd9c7fa0c0fc57723efc83bfb671`,
and `7d981a2606f7dbc5dcc723b388cbbec7c507ebc995592e6d98135606e5cb5b08`.
The new puncture checkpoint again has one connected channel and 1.138e-6
N s accepted tip impulse, but 2.455 mm/s relative point speed and 1.439
rad/s angular slip exceed the 2 mm/s and 0.6 rad/s grasp gates. It remains
`tissue-robot-first-bite-puncture-transient`. These bounded timings establish
neither a 100x speedup nor a completed robot-driven stitch.

A further one-step performance audit tested narrower broadphase changes and
then reverted all of them. A diagnostic readback on the post-entry 1x skin
step measured 33,944 deformable candidate pairs, zero active self-contact
rows, and 368,640 reserved candidate slots in one environment. The original
shader took 1,132.357 ms GPU on the matched step; the rebuilt original shader
later took 1,142.222 ms, showing the scale of single-run variation. A
parallel surface compaction trial took 1,130.238 ms; a two-bit stable radix
sort took 1,127.909 ms. Capping active-row compaction at candidate count took
1,118.304 ms. Adding candidate-count indirect narrowphase dispatch took
1,113.391 ms. Caching the first 16 eligible right surfaces per left in
otherwise unused sort scratch took 1,109.459 ms. Every trial retained the
published puncture channel, 46,080 tetrahedra, zero removed mass, positive
determinant, residual, thread bounds, and zero failed steps. The cumulative
candidate-bound result was only about 2% faster in this single-run comparison,
far below the 100x request. Its log SHA-256 is
`55ed3d37506ac13a936ae35cbdea49d2b99f4fa99bc3883f15ce6d0f11efa08f`;
the diagnostic candidate-count log SHA-256 is
`01591947fa05de49cdbe29ccc88e2f7c01da1c84f78b6a50d6cc0927e88295f4`.
The tested trial code was removed, and the restored metallib and linked
library hashes match the preceding accepted shader lineage. This bounds the
benefit of trimming unused capacity on this step. It does not distinguish
the first candidate sweep from the remaining coupled solver work; dispatch
boundary timestamp sampling was unavailable on this device. A larger gain
needs stage attribution and an algorithmic change to candidate generation
or the full-resolution solver, verified against the same physical gates.

A subsequent memory-layout audit also found no qualifying gain. Gathering
active primitives into sorted contiguous scratch before the pair sweep took
1,129.682 ms GPU on the same 1x step; its log SHA-256 is
`8d3de2fd9fd6630d63494409bf10a7f7369b750ed7bbfec363a36277e9fa6c51`.
Hoisting repeated node-lineage reads out of the 3-by-3 adjacency comparison
made the step slower at 1,333.008 ms GPU; its log SHA-256 is
`6fe8ef5a6babca791e003a1555a8621766905f994df3719d89b6880f2796b608`.
Both retained every published physical field, and both trials were removed.
The accepted metallib and linked-library SHA-256 values were restored to
`7d981a2606f7dbc5dcc723b388cbbec7c507ebc995592e6d98135606e5cb5b08`
and `dafd704ac5a81a7437666632560913002495bd9c7fa0c0fc57723efc83bfb671`.
These results reject sorted-record locality and manual lineage-load hoisting
as useful changes on this skin step. The remaining route must change the
amount of candidate work or the number and cost of full coupled passes,
while preserving contact admission, rollback, and final residual checks.

The next bounded attribution used the post-entry 1x step, which has zero
active deformable self-contact rows. Temporarily clearing the surface count
after the normal sort, after compaction but before radix sorting, and before
surface construction produced 1,122.485, 1,088.637, and 1,070.387 ms GPU,
respectively. All three retained the published physical fields. These are
single-run diagnostic ablations, not valid general contact implementations:
they place roughly 60 ms of this 1.13 s step in the surface/contact path and
show that more broadphase work cannot deliver 100x here. The diagnostic log
SHA-256 values, in the same order, are
`df7de681dbac8d084b853800619667f57610d1403fc5613f1dec601887a8c30b`,
`f985c425d8feb91c65da825ccbcdef9d13da42720e039d4ef77e7e8965944dd7`,
and `2bc1d7ea954c4cc15ebb6e6573687c5346d987f2beef45d8abefb22ca60121e0`.
All ablation code was removed; the restored metallib and library hashes match
the accepted values above. GPU dispatch-boundary counters were unsupported
on this M4, and the available Metal System Trace reported no shader intervals.

An opt-in `--tissue-suture-cadence-16x-one-newton-only --synthetic-skin`
probe reduces the post-entry step from five Newton passes to one while
retaining the required ten-column FGMRES cycle and the 16x cadence. On the
same rebuilt executable, the 1x/five-Newton reference took 1,135.494 ms
for 62.5 us, the 16x/five-Newton reference took 2,956.622 ms for 1 ms,
and the 16x/one-Newton probe took 2,324.739 ms for 1 ms. The latter is
145.296 ms per base-equivalent step: 7.81x the measured simulated-time
throughput of this 1x reference and 23.24x that of the original 3,375.995 ms
1x run. It is still 4.30x slower than the 33.760 ms base-equivalent target
for 100x over that original run. The one-Newton probe passed the one-step
tract, tet-count, determinant, swage, strand, and zero-failed-step checks;
its maximum residual was 2.508e-6 versus 2.425e-6 in the matched
16x/five-Newton run. The altered residual means this is a distinct
experimental trajectory, not an exact-state optimization. There was no
active tissue contact in the post-entry step, and no robot-driven stitch or
load-bearing pull-through was tested under this reduced budget.

The opt-in probe log SHA-256 is
`0f1995b25fbc904002b24de31d6876033e134580bd1532d31e448c6dd6a68450`;
the matched 1x and 16x reference log SHA-256 values are
`9d012188f84fbed6c726b701e7cc315f86e56db0fd9e164bae99315b65345560`
and `e056cab23ccc7ac07659c2bff798d397d9dc9d9f15e829ce06224e92c93773ff`.
The executable SHA-256 is
`b731ebd51eb3aa4b314b4c32caf49fcd73f8cbf6c4e146bd7532b7f62a07c0a0`.
The default solver budget and accepted shader lineage are unchanged.

One more diagnostic varied the FGMRES column budget after the accepted entry,
using a temporary runtime that allowed a partial restart cycle. At 16x with
one Newton pass, budgets of five, three, and one columns passed the same
post-entry one-step tissue gates with zero active Matter contact samples at
2,266.200, 2,215.667, and
2,167.439 ms GPU. Their maximum residuals were 2.409e-6, 1.740e-6, and
1.721e-6, respectively, compared with 2,324.739 ms and 2.508e-6 for
the retained ten-column probe. The apparent residual improvement is a
different accepted trajectory, not proof that fewer columns generalize to
load-bearing contact. Removing nine of ten columns saved only 6.8% of the
single-step GPU time, so the partial-cycle runtime and test hooks were
reverted. The diagnostic log SHA-256 values, in five/three/one order, are
`27ace68bd67b5516d506da33e269856f18ca0a8dbd01679bfa65e640877e068e`,
`44ac2191de215c57a7232529d21e8f87cf25271c918c09508f201316f7de818d`,
and `7ce9309d514da3a0fd9f3bd876b072add9410bd9446cd5b94963ba59316f1d29`.

At the same one-Newton/ten-column budget, diagnostic 1x, 4x, and 16x
steps took 501.612, 864.805, and 2,322.959 ms GPU. The differences are
about 121 ms for each additional encoded physics substep over both spans.
That repeated coupled-substep work, rather than FGMRES column count or
empty self-contact broadphase, is the next performance attribution target.
These grouped steps advance different simulated intervals; the slope is a
diagnostic estimate, not a whole-stitch speed claim. Their log SHA-256 values
are `57bdb2be36703b52a1be2d7d3e22b216cb4f691faf2076728cd2f1daf9ed3bc9`,
`db99b223ae6dca08d5790efe41e480c8dd5a1cefeafe1fd292216e613208e586`,
and `5ac95a760fbdeb14071a9e8844a90bae74af1949464dfb8fc2a2bc50d188b9ff`.
The restored library and executable SHA-256 values match the preceding
accepted `dafd704a...` and `b731ebd5...` binaries.

The M4 supports Metal timestamp counters at compute-stage boundaries, even
though it does not support dispatch-boundary counters. A temporary native
counter probe read shared samples after completion and split the 16x,
one-Newton post-entry step by encoded MetalWorld substep stage. For the first
15 substeps, median GPU times were 0.390 ms before the rod, 3.631 ms in
the rod stage, 116.928 ms in the contact stage, and 0.417 ms after contact.
The final substep measured 338.455 ms before the rod, including the coupled
Matter work, and 160.414 ms in contact, including its 16 extra final solver
iterations. The instrumented total was 2,330.238 ms versus 2,324.739 ms
for the earlier uninstrumented run, a 0.24% difference. Its log SHA-256 is
`418c497373e8544224bd21ae984f4e766206a45e578cdb4f34b18dd0cf174427`.

A post-entry-only diagnostic then reduced MetalWorld's 32 velocity iterations
plus 16 final iterations to 16 plus zero, and to one plus zero. The earlier
incomplete one-step checks passed at 1,592.598 and 943.778 ms GPU; median
contact-stage time across the first 15 substeps fell to 73.667 and
33.050 ms. The one-iteration result is 58.986 ms per base-equivalent step,
or 57.2x simulated-time throughput against the original 3,375.995 ms 1x
run, still 1.75x slower than its 100x target. This one-iteration timing
fails the terminal contact residual gate described below and is not a valid
physical speedup. These diagnostics changed the swage and strand fields and
tested only one post-entry step; they do not qualify the robot grip,
load-bearing contact, whole stitch, or a 100x end-to-end improvement. The
16- and one-iteration log SHA-256 values are
`b9204745c21808bf7507fe6464b724283df1c8ee36c066321f6a28c79d646c2f`
and `c4078aba976e143c2f1d8912f617851af6673d21c09d94b9c91316ef2558d434`.

The scene reserves 61,296 contact constraints, 20 island-work slots, and
1,917 tile-work slots. Suppressing the distributed spill branch in one
diagnostic run did not improve total GPU time; an exact-zero impulse-delta
early exit in the Wave32 loop also gave no measurable gain. Both trial
changes and all counter instrumentation were removed. The accepted linked
library and executable SHA-256 values were restored to
`dafd704ac5a81a7437666632560913002495bd9c7fa0c0fc57723efc83bfb671`
and `b731ebd51eb3aa4b314b4c32caf49fcd73f8cbf6c4e146bd7532b7f62a07c0a0`.
The next solver change needs to reduce the cost per Wave32 iteration while
retaining its final contact residual and rollback gates.

The native contact-status readback resolved the missing gate. This step has
143 active MetalWorld constraints despite zero active Matter contact samples.
At 48 contact iterations the terminal normal residual is 0.000868 m/s; at
16 iterations it is 0.001341 m/s, below the existing 0.002 m/s terminal
bound. One iteration leaves 0.193026 m/s and fails that bound by 96.5x.
The cadence probe now calls `requireTerminalResidual` and prints the contact
status, required constraint count, solver iterations, impulse delta, normal
residual, cone violation, and factor residual. Native 1x, 4x, 16x, and
16x/one-Newton modes all passed the added gate. A temporary one-iteration
negative control was rejected with `code=0 iterations=1
residuals=[0.000000,0.193026,0.000000,0.000000]`; its log SHA-256 is
`237a0fe389c6db098e6385c8ac3c0ca6d531c8ab27a77ded1bd75c777f78ddc8`.
The final probe executable SHA-256 is
`7a51915cf02c97dfcac1fb3356be3cff0e9558724a9f9a64384b62dba887e6be`;
the Matter-linked library retains the accepted `dafd704a...` hash. The
one-iteration shortcut is disqualified; the 16-iteration one-step result
still needs full robot-contact and exact-replay qualification.

An explicit iteration sweep on the rebuilt, on-chip-rod-workspace shader
now prints GPU time and contact residual before enforcing that gate, so
rejected settings retain timing evidence without being accepted. For the
same 16x/one-Newton post-entry step, one iteration took 932.093 ms GPU but
left 0.185153 m/s normal residual. Settings of four through seven
iterations also failed the 0.002 m/s bound; at seven, the residual was
0.002374 m/s and GPU time 1,174.770 ms. Eight iterations were the first
tested setting that passed: 0.001867 m/s and 1,213.736 ms GPU, repeated at
1,213.087 ms with the same physical fields. Sixteen and 32 iterations took
1,535.874 and 2,176.486 ms and left 0.001267 and 0.000744 m/s residual.
The eight-iteration step gives about 44.5x simulated-time throughput against
the original 1x/3,375.995 ms reference, but this is one post-entry step with
no active Matter tissue contact, not an accepted robot-contact or
load-bearing stitch trajectory. It is a distinct state and has only
0.000133 m/s margin below the terminal contact bound. Even the failing
one-iteration step takes longer than the 540.159 ms per-16x-step target for
100x throughput, so iteration cuts alone cannot meet that target in the
current pipeline.

Two temporary setup ablations were rejected. Replacing the nonrod relaxation
row sum with `1/activeRodConstraintCount` damping produced the same eight-iteration
fields and 1,213.057 ms GPU, so it offered no useful timing gain. Omitting
the exact rod self-response from preconditioner construction took
1,223.111 ms and worsened the eight-iteration residual to 0.003250 m/s.
Both shader changes were reverted; the accepted MetalRobo metallib SHA-256
is again `08531cb0270799f4e2826e27ce0e6fb9ba4eba1323344a9662a5ba382041295e`.
The diagnostic probe executable SHA-256 is
`0213685f9171a9a2cfe472a9bd2f7dc74b2fc81f16ad427f0f7f0c6b66bc84b5`.
The one-, seven-, eight-, and repeated eight-iteration log SHA-256 values are
`49e346f370c0bbec5acc5f79f7a1ae052c48e7df43ee0a72c5e0db1d3ccf4f51`,
`efee4db19a7cfe82c1b62b07667decf6f3e677a63f2b502e83058452b9b8ddce`,
`d00af45a3847405351d4c5c06c3ede9de850f47079c2731144af031472933f2f`,
and `bbcc185ce586dfc83fec6953c2c0acd9173621b49ad9093de811bb856252feec`;
the two ablation log SHA-256 values are
`782c60fce716a251c9bdcd99d4b879d2b20a0507cb7ce4b7e8c0287f8552e1d5`
and `58d6a637446774fc1b1a3b79c4ea913550d15bfeab4f39d08e9f9db8336db915`.
The focused static-equilibrium, checkpoint-restore, synthetic-skin puncture,
and live-cadence tests passed after the shader was restored.

The contact-capacity hypothesis was also checked. The default heterogeneous
world reserves 61,296 constraint blocks while this post-entry step uses 143.
An opt-in 512-block request was rejected by the heterogeneous compiler's
topology-envelope minimum. A temporary operational-capacity compiler
diagnostic then allowed that request, retaining device overflow checks and
the same eight-iteration probe. It produced the same reported contact,
tissue, needle, and strand fields, but took 1,226.105 ms GPU versus
1,214.993 ms with the default capacity. Both the CLI override and compiler
change were reverted; the original executable and shader hashes were
restored. The default, rejected-request, and operational-capacity trial log
SHA-256 values are
`542ece320d38a1b8afd30e6e4caf5e84938694f5fd25ac3bee43f1002bc0db53`,
`b7b4a8d067d6df2df0986afa2e57e63f5898d4e47ceb63c875a7bc83cf4660a2`,
and `f0edae35d223b5a90acbb06eb8ffecbfd2bfdf437b18f7c5a12abb7784f17be2`.

The stronger 75 um jaw-preload trial from robot contact step 1645 also
remains unqualified. A 13 ms ramp at 1 mm/s preserved Matter's 46,080
tets and zero failures but drifted 212 um at the needle seat. A 6 ms
partial ramp at 5 mm/s reduced seat drift to 28.9 um but produced
20.0 mm/s contact-relative point speed and 2.11 rad/s angular slip,
above the 2 mm/s and 0.6 rad/s grasp limits. Neither published a
checkpoint; their log SHA-256 values are
f19e715d8ee5a9fc77345d942aa2b49f7ec8660a0f386acbc3f77fb88f2c0229
and 0cb969049397267ee58c6bda20d852f896a04f4dd028a41d5191246033b224bc.

The next 100x optimization gate identified a serial transactional manifold
copy in every MetalWorld contact microstep. Temporary stage-boundary Metal
timestamp sampling of the first contact substep in three short submissions
measured that encoder at 17.34-17.65 ms. The kernel previously used one GPU
thread per environment to copy every manifold header and point in a loop.
It now dispatches one thread per manifold; thread zero also copies the scene
bodies and manifold count, and a zero-capacity manifold retains that copy.
The hybrid-CCD and standard contact paths both use the new dispatch. The
same sampled encoder fell to 0.083-0.086 ms. The profile instrumentation was
removed after the measurement; retained logs are
`build-skin-wound/contact-commit-profile-before.log` (SHA-256
`db0f83ee6170e67a7cbff95bf84bbf485f8d3f85423fac59dd9fc21abf3a4e28`)
and `build-skin-wound/contact-commit-profile-after.log` (SHA-256
`2ac121f323cc2329435501f564e45aebcfff13532828a4493882e501c4872774`).

With the full eight-sweep solve retained, the uninstrumented 16x/one-Newton
post-entry probe took 945.588 ms GPU versus the preceding 1,213.087 ms:
22.1% less time, or 1.283x speedup for that step. Reported tissue, needle,
strand, contact, and Matter fields matched, including 46,080 tetrahedra,
zero failed steps, and 0.001867 m/s normal residual below the 0.002 m/s
gate. Four focused surgical CTests passed. Against the original 1x
3,375.995 ms reference, this single-step simulated-time throughput is
57.1x; the 100x target requires at most 540.159 ms for a 16x step.
There were zero active Matter tissue contacts in this probe, and no complete
robot-driven skin stitch has been timed or qualified at this setting.
The final uninstrumented log is
`build-skin-wound/contact-commit-final.log` (SHA-256
`1c77a9b29173df426d319cfc4c84ac1d250a3ac7f6f3784affd2de21a6bd170b`).
The final Metal library and shader SHA-256 values are
`74091b497441b33c4fa56a38f615b9d4b19bcbba278fbee34f786372b640e458`
and `5e6f2cef7ab0a4dad63ce6647f2d0517952ef5e989af47e2f48a273de34f0e61`.

Reducing only the pre-contact generalized IR sweep to one, two, four, or
six iterations was rejected. Those trials took 838.853, 856.905,
886.804, and 913.313 ms GPU, but their respective normal residuals
were 0.002014, 0.002032, 0.002035, and 0.002036 m/s, all above the
0.002 m/s gate. The full eight-sweep solver remains authoritative. The
remaining repeated pre-contact and canonical IR solves each consumed about
10 ms per measured substep, while the final Matter substep has a separate
large cost; these are the next measured optimization targets.

Open work: physically approach and pick up the needle with the robot, drive
it through the first skin lip with measured tissue reaction, qualify thread
pull-through with this 1.5 mm wall, then the second lip bite, knot tightening,
gripper release, and measured unloaded wound-gap retention. Existing
jejunal checkpoints must not be reused as skin evidence; generate fresh
skin checkpoints and compare their material/world fingerprints. Capture
gap, thread tension, needle reaction, tissue strain, contact, work/energy,
and exact replay before claiming a complete interrupted stitch. No skin
calibration or clinical validity is established.

The next bounded performance pass profiled the final 16x Matter substep and
found repeated FEM constitutive bytecode evaluation in its FGMRES operator:
the first 46,080-element operator kernel took about 21.5 ms of a roughly
22 ms column operator. For the exact authored `neo_hookean(mu, lambda)`
energy, the compiler now records a source-qualified flag and the FEM
operator evaluates analytic stress and tangent with the live environment's
mu/lambda parameters. Other expressions, programmatic materials, stateful
materials, and dissipative materials retain generic bytecode evaluation.
The profiled first element kernel fell to about 0.096 ms. Temporary timing
instrumentation was removed; the stage logs are retained in
`build-skin-wound/matter-stage-before.log` and
`build-skin-wound/matter-stage-canonical-neo.log`.

The uninstrumented 16x/one-Newton post-entry step took 784.488 ms GPU,
versus 945.588 ms before this change. This is 17.0% less time for that step,
or about 68.9x simulated-time throughput against the original 1x/3,375.995
ms reference. The 100x threshold remains 540.159 ms per 16x step. The
step retained 46,080 active tetrahedra, zero failed steps, needle advance
19.965 um, 143 MetalWorld constraints, and 0.001867 m/s terminal normal
residual below its 0.002 m/s gate. The Matter maximum residual changed from
2.508e-6 to 1.835e-6 and FGMRES used six rather than seven columns, so
this is a numerically distinct run, not byte-exact replay. The exact-source,
programmatic-fallback, and altered-expression compiler checks passed,
along with static-equilibrium, checkpoint-restore, synthetic-puncture, and
cadence-transition CTests. The run log is
`build-skin-wound/canonical-neo-16x-one-newton.log` (SHA-256
`5bc93c8710e36e3c5b10e25c87a979e341fa3df71a3b75edfa36d51a83fd51dc`).
The Matter metallib SHA-256 is
`c0a0de242251f677dc6db658627c48fa19421b0a0dddf476d48e42159e94`.
This timing contains no active Matter tissue contacts and does not establish
a 100x whole-stitch speedup or robot-driven stitch qualification. The
remaining single-step gap is at least 244 ms; repeated MetalWorld contact
solves dominate the remaining time and require a separate bounded profile
that preserves the terminal residual gate.

The next bounded contact-solver change reuses each fixed generalized
articulation response `M^-1 J^T` across the eight ordered sweeps of a
single solve. It stores the response in that authored constraint's three
existing response-column slots; the Cholesky factorization and row Jacobian
are unchanged during the solve, while each sweep still reads the updated
velocity and applies impulses in canonical order. Both pre-contact and
post-contact generalized solves recompute their first-column response, so
the cache has no cross-solve lifetime. A cross-solve reuse trial saved only
about 6 ms and was removed; a rod-slot prefix trial saved no time and was
removed.

The final uninstrumented 16x/one-Newton post-entry step took 636.552 ms GPU,
versus 784.488 ms before response reuse and 945.588 ms before both changes.
That is about 84.9x simulated-time throughput against the original 1x/
3,375.995 ms reference. The 100x threshold remains 540.159 ms, about
96.4 ms lower than this run. The reported 46,080 tetrahedra, zero failed
steps, needle and thread geometry, Matter residual, 143 MetalWorld
constraints, and 0.001867 m/s terminal normal residual matched the
pre-cache run. Five compiler/surgical CTests and four further coupling,
robot-grip, tissue-entry, and topology CTests passed. The final log is
`build-skin-wound/generalized-response-cache-final.log` (SHA-256
`9a65e56e3af09e1803d821afd2707d3f1799888183f1b2b10d9c11ac8cd68b62`).
The MetalWorld metallib SHA-256 is
`ba4f62d64c6b4c232b2ba8b39e8c10f268c506a5c3055febf7ed4f4546283406`.
This is still a single post-entry step with zero active Matter tissue
contacts; no full robot-driven stitch, load-bearing contact timing, or
100x end-to-end performance is qualified by it.

A shared Metal stage-boundary timestamp pass on the same 16x/one-Newton
step isolated the next solver cost without changing the reported physical
fields. Across the 16 measured contact substeps, the pre-contact generalized
IR solve used 88.653 ms, the post-contact generalized solve used 88.320 ms,
and the Wave32 temporal cone solve used 115.475 ms. Their medians per
substep were 5.534, 5.511, and 7.195 ms respectively. Together these
stages account for 292.448 ms of the 636.264 ms instrumented step. The
case contains 20 authored blocks, 28 articulation velocity coordinates,
and 128 rod nodes; 14 blocks have only generalized endpoints and six
include rod attachments. The M4 supports these timestamps at compute-stage
boundaries, not individual dispatch boundaries. Profiling hooks were removed
after capture. The retained shared-stage and Wave32 logs are
`build-skin-wound/metalworld-shared-counter-profile.log` (SHA-256
`ac4cf4d929738b6260ddcf1f612914de36fe8e68165367eefb405469b4154840`)
and `build-skin-wound/metalworld-wave-profile.log` (SHA-256
`64f6efd1cbb4b254ba6b7f30f363b0568fb9fe75cf103f90e4fe2008ab8915de`).

Three isolated shortcuts were rejected. Skipping generalized response updates
only for exactly zero impulse deltas took 633.187 ms; specializing the 14
pure generalized rows took 635.608 ms. Both preserved the reported physics
but were within the run-to-run timing spread of the 636.552 ms reference,
so neither was retained. Cutting only the post-contact generalized solve
to one sweep reduced the step to 607.476 ms but left a 0.293471 m/s
terminal normal residual, far above the 0.002 m/s acceptance bound. Its
failure proves that post-contact ordered sweeps are still required for this
state. All trial code was removed. The respective trial log SHA-256 values
are `517eaf48f5b75b87ef69461bf86db3fcb8a7e2cf5a80b74b798a7aea83253a0d`,
`ed24a2ed08681842ef10441e086ed07a9c593d330a4658005124b5c585645533`,
and `ee60a5d1666761fe94c99785336d1baf826e0bc51c258f673f7933d16e1650b5`.
The 100x target remains open: the next implementation must reduce the cost
of valid ordered rod/contact response work, not remove required sweeps.
After restoring the accepted shader and host code, the uninstrumented
post-entry step passed at 631.779 ms GPU with the same reported physical
fields and 0.001867 m/s normal residual. Five focused static-equilibrium,
checkpoint, synthetic-puncture, cadence, and compiler CTests passed. The
restored run is `build-skin-wound/generalized-profile-restored-final.log`
(SHA-256 `85bb7f4d715104a75aa0ec52ad8dd2ee2656c4ad0c185c5ae330542448dae830`).

The next accepted solver change keeps all eight ordered generalized and
Wave32 sweeps. The pre-contact solve now writes its factorized rod response
directly into the per-attachment retained arena slot; the post-contact
solve reuses that fixed rod and articulation response from the same
substep. The rod factor and authored row Jacobians remain unchanged between
the two solves, while relative velocities, impulses, and the final residual
are still evaluated against the updated state. A separate GPU kernel then
prepares the six independent rod attachment responses in parallel before
the pre-contact ordered sweep. Each attachment owns a disjoint arena slot;
invalid factors remain rejected by the serial solver's existing checks.

On the 16x/one-Newton synthetic-skin post-entry step, the intermediate
cross-solve rod reuse took 588.127 ms GPU, in-place response construction
took 579.866 ms, and cross-solve articulation reuse took 569.006 ms. The
parallel rod preparation reached 537.256 ms, with three subsequent repeats
at 536.133, 535.260, and 535.252 ms. After tightening the parallel
kernel's factor and endpoint bounds checks, that uninstrumented step
took 535.650 ms GPU. Its 23 reported non-timing fields match the restored
631.779 ms run exactly: 46,080 active tetrahedra, zero failed steps,
143 MetalWorld constraints, 19.965 um needle advance, Matter residual
1.83548e-6, and 0.001867 m/s terminal normal residual below the 0.002 m/s
gate. The intermediate run log is
`build-skin-wound/generalized-parallel-rod-qualified.log` (SHA-256
`6180d5503c8032ce903934c59872437dd2bea32754b52f97e9aee9bd1dcf7d72`),
and that intermediate MetalWorld metallib SHA-256 is
`f5453092afc48410bd8a8282f37795f3f69c4c4a5dcb44441b17f7dbe60c5408`.

The final solver also retains each row's effective-mass denominator in its
unused response-column slot across the ordered sweeps and pre/post pair.
The row's relative velocity, projected impulse, and residual still use the
latest candidate state. The denominator-cache trial took 534.294 ms GPU;
three repeats took 531.715, 530.833, and 531.778 ms. All 23 reported
non-timing fields again matched the restored reference, including the
0.001867 m/s terminal contact residual. The retained final repeat log is
`build-skin-wound/generalized-denominator-repeat-3.log` (SHA-256
`cc46debef620a12e638aaaef8d5c08e5fbdb28871d70cdb082b9843a959a2d6f`),
and its MetalWorld metallib SHA-256 is
`8059ec81644b9216d80f3a3461ce6738881ffd0fc3cb887951211773777fb199`.
Eleven focused surgical, robot-grip/drive, checkpoint, topology, cadence,
and Matter compiler CTests passed on this solver version.
The final source-matched rebuild passed the same native step at 531.363 ms;
`build-skin-wound/generalized-denominator-final-built.log` has SHA-256
`39ae8e932d1cb7ab345eea2e5cd43ab9f9d13d17c55b0cd98e4249eb02bf78ff`.

The final fast step is 101.6x simulated-time throughput against the
original 1x/3,375.995 ms baseline for this post-entry state; the stated
100x threshold was 540.159 ms per 16x step. A fresh 1x step on the
optimized build took 669.419 ms with the default 32 plus 16 contact sweeps,
so the fast setting is about 20.1x throughput relative to today's 1x
setting. These comparisons do not establish a 100x full-stitch speedup:
the fast probe contains zero active Matter tissue contacts and does not
execute a complete robot-driven first bite, second bite, or knot. The
fresh 1x log is `build-skin-wound/generalized-parallel-rod-1x-baseline.log`
(SHA-256 `ff9c12df76cf42af3a8565b59e09c3b44df7ed388ee391db803a48009ecc6ee4`).
The retained robot first-bite drive CTest accepted 12 commanded steps and
192 base DER substeps over 0.012 s of simulated time with zero failed steps,
but took 20,538.506 ms GPU and had no active puncture channel. Its
robot-contact output is preserved with the 11-test run in
`build-skin-wound/generalized-denominator-ctest-11.log` (SHA-256
`518a2b51e4da9d0f78bb021a4bb0af9bf910b762f931c2cd50fee4422bd64a1c`).
The narrow post-entry cadence throughput result therefore cannot be used
as a performance claim for robot approach, load-bearing first bite,
pull-through, opposing bite, or wound retention.

### Current-binary robot continuation check

The older retained step-1773 robot-contact checkpoints do not restore with
the current device program: restore rejects their program fingerprint. A
fresh current-binary 12-step robot drive produced the step-237
`build-skin-wound/skin-robot-current-drive/tissue-robot-first-bite-approach.tsv`
checkpoint (SHA-256
`68f1c0805ad5ad1cfc913eaa70c11478f3cdc254e09ca268d2cce19be9d5ba4b`).
Its exact restore passed with 46,080 active tetrahedra, zero puncture
channels, and zero removed mass. The drive took 20,587.067 ms GPU; its log
is `build-skin-wound/skin-robot-current-drive.log` (SHA-256
`11a4650cf9b36e4a082c18865196a740ed0b010b36dc34af68ba2b996379a79b`).
The restore log is `build-skin-wound/skin-robot-current-drive-restore.log`
(SHA-256
`1e0db3cbad1f57bf3e93bd3655e13b5b19bc49386af81361b311c54256256d22`).

A 12-step, 384-base-substep continuation from that checkpoint retained the
robot grasp and rod, advanced the needle tip 110.96 um, and ended 377.22 um
from the nearest tissue node. It had no tissue contact or puncture. The
MetalWorld normal residual peaked at 0.003875 m/s, above the 0.002 m/s
terminal-contact threshold used elsewhere in the probe; the peak was step 0,
with steps 2 and 3 also above that threshold.
The terminal step's residual was 0.001337 m/s. The per-step diagnostic is
`build-skin-wound/skin-robot-current-residual-steps.log` (SHA-256
`c127eea6ace6392937692ddb9564120d475a2b8bbd47a251512c30db505f4156`).
The diagnostic now prints the peak step and all per-control-step residuals
on the existing robot approach result line. The continuation's 24,677.235 ms
GPU time does not qualify as a 100x result.

Doubling the velocity sweeps to 64 plus 16 cost 39,416.182 ms GPU and still
peaked at 0.003590 m/s. A separate one-step checkpoint hold replayed
byte-exactly, preserved all tetrahedra and zero puncture channels, and
produced a step-253 checkpoint. Continuing from it moved the peak to step 2
but left it at 0.003843 m/s, with 24,640.847 ms GPU time. The hold and
follow-on logs are `build-skin-wound/skin-robot-current-held.log` (SHA-256
`0d8461b9e192e8779625fb1abb518194d21c48d04ff280efd458adebc41c088e`)
and `build-skin-wound/skin-robot-current-held-continue32.log` (SHA-256
`679bd54727dc0a4770ba0f21b59028131c3c52a109ebb907a67f97f20b404feb`).
Neither extra sweeps nor a single held step cures the early contact
transient; do not promote either continuation checkpoint into the stitch.

### Fresh robot-to-skin contact and puncture timing

Using the current device program and the step-237 checkpoint above, a fresh
robot-held sequence reached skin contact and then one transient puncture.
Every resumed phase reported byte-exact checkpoint restore and zero failed
transactions. The measured GPU times below exclude scene setup, file I/O,
and the initial drive's 1,099.540 ms grip-settle GPU time.

| Phase | Simulated interval | GPU time | Final tip-to-node separation | Tissue result |
| --- | ---: | ---: | ---: | --- |
| Initial robot drive | 12 ms | 20,587.067 ms | 485.63 um | no contact |
| First 64x approach | 24 ms | 22,543.695 ms | 375.98 um | no contact |
| Second 64x approach | 24 ms | 22,609.168 ms | 262.82 um | no contact |
| Full-Newton guarded approach | 24 ms | 31,038.832 ms | 150.78 um | no contact |
| Full-Newton entry | 16 ms | 28,737.594 ms | 71.57 um | 1.055e-8 Ns tissue reaction |
| Contact advance | 8 ms | 14,944.283 ms | 21.76 um | 7.009e-8 Ns tissue reaction |
| Single puncture microstep | 62.5 us | 1,456.255 ms | 20.64 um | one transient channel, 1.124e-6 Ns reaction |

The seven phase timings total 141,916.894 ms GPU for 108.0625 ms of
simulated motion, without an exact matched old-binary baseline for this
whole sequence. They therefore establish neither a 100x whole-robot
speedup nor real-time operation. The fast 64x first approach saved only
about 2.13 s against the current 32x 24 ms continuation, and its first
two control-step normal residuals were 0.002267 and 0.002289 m/s, above
the 0.002 m/s terminal-contact threshold used elsewhere. The later contact
advance ended at 0.002969 m/s. A successful command exit alone is not a
qualified contact state.

At the entry checkpoint (`build-skin-wound/skin-robot-current-entry/`
`tissue-robot-first-bite-contact.tsv`, SHA-256
`85f7e25d29ba5df308221cd5f0a2344bcc6a9382cabf02ac8ed45783966b2522`),
the robot grasp and strand checks passed, 46,080 tetrahedra remained active,
and the tip and tissue impulses were positive. The entry checkpoint restored
byte-exactly at state step 1645; the resulting contact-advance checkpoint at
step 1773 has TSV SHA-256
`6a972adff440ab67f229343bb54811493b1a5035f58c5eecdbacbb57d5ce5029`.
The single microstep from state step 1773 earned one puncture channel, but
point slip was 2.466 mm/s against the 2 mm/s grasp limit and angular slip
was 1.441 rad/s against 0.6 rad/s. Its MetalWorld normal residual was
0.006427 m/s. The step-1774 transient TSV has SHA-256
`81d86ad1c5a617a2fb38a72ebc21cffdb7aca49a4bfe7a1e2af7778fdcb0d42b`;
its byte-exact restore passed with one active channel and all 46,080
tetrahedra (`build-skin-wound/skin-robot-current-puncture-restore.log`,
SHA-256 `42eb55fa88050dd81f3a61fcc1c7d7563b936a4df4180051cd9294a7bc7b02dd`).
The state remains a transient puncture checkpoint, not a robot-driven bite
eligible for pull-through.

Raising that single step to 64 velocity sweeps lowered point slip to
0.518 mm/s but left angular slip at 1.031 rad/s and residual at 0.003886
m/s. At the maximum permitted 128 sweeps, angular slip was still 0.767
rad/s, residual 0.003237 m/s, and one jaw's insert coverage fell to
patch mask 5 from 15. A one-step held-contact attempt from state step
1773 rolled back at its last substep and published no checkpoint. A
checkpoint-velocity feedforward trial preserved the puncture but did not
improve slip, residual, or the matched approach timing; its code was
reverted. These tests point to controlled braking and robust jaw contact
as the next physical blocker, alongside the measured coupled-solver cost.

Retained log SHA-256 values, in phase order after the initial drive, are
`e692350f3c4877de91f8399b4d6c1108e452a672a58c0e4130f9b45a761eac58`,
`392090284660206bf9a4f3a6f1382b45a2b9f156e25d2208f37a92ed06dde4fa`,
`3881d16e92b022ac06d26f73b3d5e60793291e63c3761b1d2cf2022644fd5ee2`,
`9801a63dccc24d0def433b92971899eb38e9b761a248e2a275d6b90d0782d2a1`,
`3afd663e7bebb58daafca59c90424560525a1c778ba94645cc291ae074542931`,
and `3ef76719b4a4d1158365e4982328feb924cb007ee4ab2cb16485cdc861cb2622`.
The 64-sweep, 128-sweep, and rejected hold logs have SHA-256 values
`e602240bcee77aab1d85d071e057001ca4ea988a3d6b471e809407e9e0743508`,
`2ead8d3ec51cc59fd13c721c4cd2caa5d23405a76c9a38e4a21f3208c7d529ec`,
and `a084b313b882a8e2c468c19084dc93aa3bbdf7a7a34efef3f55e5c9d186eac1e`.

### Robot contact braking trial

An opt-in `--robot-contact-brake` for the synthetic-skin contact-advance
mode now uses fourteen robot-driven 1 ms waypoints. It ramps the needle
orbit target from 5 to 1 mm/s and requires the planned capsule sweep to
stop at least 20 um before the nearest tissue node. The original eight-step
constant-speed contact advance is unchanged. From the same step-1645 entry
checkpoint, the final brake advanced the tip 53.23 um and ended at 19.03 um
separation, close to the original 21.76 um. Its needle point speed fell
from the original contact advance's 4.934 mm/s to 0.709 mm/s, about 7.0x
lower. The bilateral grasp, strand, and 46,080-tet checks passed, with
7.982e-8 Ns measured tissue reaction and zero puncture channels. Its last
contact residual was 0.000519 m/s, though the maximum across its fourteen
steps was 0.003594 m/s. The 14 ms motion cost 28,550.816 ms GPU; this is
a physical-control diagnostic, not a performance speedup.

The braked contact checkpoint restored byte-exactly at state step 1869.
Its TSV SHA-256 is
`4b605822a49b3ea7330461cc1ecb0d97dab72533b8479f63352fe567ae3fedf2`;
the native run and restore logs have SHA-256 values
`66e71fb0d3751fe8fbae8dbcb384183bead0fdaa8a8fe29d084207ad09854aa7`
and `8e4196168afedc2f452ed34d89e561009e5b871cbf35600126bc298845c3c9e2`.
The three focused robot IK, grip, and drive CTests passed after this code
change.

One 62.5 us puncture microstep from the braked checkpoint again earned
one channel and retained all tetrahedra. Point slip fell from 2.466 to
0.985 mm/s, below the 2 mm/s grasp bound, but angular slip remained
1.440 rad/s against the 0.6 rad/s bound. Its normal residual was 0.005933
m/s. This step-1870 transient checkpoint passed byte-exact restore; its
TSV SHA-256 is
`b2a0777cc528cfddb990ba45db7c993ed116dada8b314a5a48c681a99d6b0a1e`.
The microstep and restore logs have SHA-256 values
`84856fb01798837af05b2c9bd5393625ca65c7a60a172534ec64dcd2be70cdb7`
and `2dc8fedd24cd868c66631b953b0b693ed70272c1ebf8d3b3f8cb35aca1e99881`.
The lower entry speed reduces translational slip but does not qualify a
robot-held puncture or the rest of the stitch. The next physical change
must retain the needle's orientation under the puncture impulse without
weakening the measured contact and topology checks.

A paired grip diagnostic now prints relative needle-tangent spin alongside
total relative angular speed on robot continuation results. On the same
native puncture microstep, the original constant-speed checkpoint had
1.221 rad/s tangent spin within 1.441 rad/s total relative angular motion.
The braked checkpoint had 1.030 rad/s tangent spin within 1.440 rad/s total.
The implied components perpendicular to the needle tangent are about
0.766 and 1.007 rad/s, respectively. Slowing translation reduced point
slip but traded some tangent spin for tilt; it did not reduce total
orientation loss. The two diagnostic logs have SHA-256 values
`db2694dfb05e5aaeedf1672dc56cea39c6693c3f965e24f277b31653ebc210c2`
and `11e07743234ec3f6e7d416b338ce795f868896e5000df72a408036c716b523fd`.

Two IK-compensated jaw-preload experiments from the same step-1645 contact
checkpoint were rejected. Ramping the calibrated rail engagement from
60 to 75 um over the fourteen braking steps gave full 8/8 jaw contacts,
but moved the needle seat 220 um, above the 100 um grasp bound, and had
a 0.0473 m/s maximum contact residual. Even a 60-to-65 um ramp moved the
seat 202 um and peaked at 0.1842 m/s residual. Neither produced an
accepted checkpoint, and all preload-control code was removed. Their
diagnostic logs have SHA-256 values
`04aa92f6119ee431d5c84772a74d33eae0babc890440508fe8f01f3e2c5a71a6`
and `13fec6706e197524f43d802edeb696d8013af1de4283f1a1ab8e70d92fa389c0`.
The remaining failure calls for better rotational contact authority or
contact-solver response, not an unverified clamp-force increase.

### Guarded one-Newton robot free approach

The opt-in `--robot-fast-one-newton` is accepted only with
`--tissue-robot-first-bite-continue-fast-only --synthetic-skin`.
It changes the first contact-free 64x robot continuation from two Matter
Newton passes to one; the existing live tip-to-tissue clearance guard and
all grasp, strand, tissue-certificate, and checkpoint checks remain in
force. Near skin contact the option was rejected before advancing physics,
and the full seven-pass solver was used for the subsequent guarded approach
and entry. Its rejection log SHA-256 is
`3d4c2f428a8c0c77e83ff6ef65f5edacb0cc56a02ef7718a258360893c11b9db`.

From the same current-binary step-237 robot approach checkpoint, two
one-Newton runs took 21,710.762 and 21,703.030 ms GPU for the same 24 ms
of simulated motion. The final source-matched rebuild took 21,711.049 ms.
All three produced byte-identical checkpoint TSVs (SHA-256
`4e4e177c2c17da25599a3c4d5ed03c86c50d9656330c3d2633000ff2ca2e9b00`).
The one-Newton checkpoint restored byte-exactly at step 621. A paired
two-Newton run on the same build took 24,896.143 ms, while an earlier
two-Newton run took 22,543.695 ms; the measured one-pass gain is thus
about 0.84-3.19 s, or 1.04-1.15x for this segment, across these runs.
The trajectories are distinct, not bitwise-equivalent: the one-Newton
run's Matter maximum residual was 5.285e-7 versus 7.560e-7 in the earlier
two-pass run, and its MetalWorld maximum normal residual was 0.002307
versus 0.002289 m/s. Both runs retained all 46,080 tetrahedra, zero
puncture channels, and qualified terminal grasp and rod checks. Neither
maximum across steps meets a conservative 0.002 m/s per-step screen.

From the one-Newton checkpoint, the normal two-Newton second 64x approach
still passed grasp, rod, and Matter checks at 262.65 um final tip-to-node
separation. The full seven-Newton guarded approach reached 150.62 um, and
the full-solver entry reached 71.38 um with positive accepted tip impulse
and 1.022e-8 Ns tissue reaction. This verifies an alternate contact-free
prefix leading to live skin contact, not a qualified puncture, whole
stitch, or 100x whole-path speedup. The first fast segment's source-bound
logs, repeated run, paired two-pass run, and exact restore have SHA-256
values `feda82bfcf5c8f2da3ba93cb2ef5a74d61e1c7d2bb34b7bc2f6d076f65be6a46`,
`e435d8d01373c2b634a1b6e2fc257e6279d4fb603784d74ada83c6b43d7dab1b`,
`d36ac541cd7d3ee7332ee525c324fde63de971d0659a9436e3962653adc4e1c3`,
and `ab9ce31d566d637783fa6c4ce83b8cc8a0de81465d1a955e671009f608f24f39`.
The final rebuilt run log has SHA-256
`dcfc796b1ec4da990a22151002523c45c69cc6f2fd7cda5b878b93f3f7d008c6`.
The next 64x, guarded approach, and entry logs have SHA-256 values
`dd6a759bf846cfe3794e8cd5e9e9a229e971386e94681b3f00c145d9c13f281f`,
`671311f1f3a0eb91790af038bde679d6fb3d01932db736e29aaec2110530f0f3`,
and `50129346acf41be7a8ef2a8458438020547ba25d5d1d16d70198402ba30738db`.

The next generalized-solver trial avoided rebuilding unused local Cholesky
arrays and row right-hand sides during cached ordered sweeps. On the same
16x/one-Newton post-entry step with eight velocity iterations, the baseline
took 527.748 ms GPU, the two trial runs took 529.691 and 528.362 ms, and
the restored shader took 533.694 ms. All reported non-timing physical
fields matched, including 46,080 tetrahedra, needle advance, Matter
residual, 143 constraints, and 0.001867 m/s terminal contact residual.
There is no measured speed gain; the trial was removed. The restored
`MetalRobo.metallib` SHA-256 is again
`8059ec81644b9216d80f3a3461ce6738881ffd0fc3cb887951211773777fb199`.
The baseline, two trial, and restored log SHA-256 values are
`ad2bedcbed1ebe06b68a2f45897b597b12354605d85646896ddbd7a208383404`,
`956e6f702766693a8eb1c4b8db90e395c6e83d1c3d9f6a7ab54f74d4c161c54c`,
`84c219d4c9201d3bba9ac2d30f2225e02794a068f7f6962e962684d34944822b`,
and `9cdecbf0ac3d95da3ac9ce9c8d7db0c72269cc436aaefc9cb3b5f6ddd7d6c124`.
The remaining generalized-stage hypothesis is the ordered rod response
update over the 128-node strand, not local array clearing; its cost still
needs a bounded GPU measurement before an algorithmic rewrite.

A deliberately invalid rod-update ablation did not isolate that cost:
omitting the per-row rod velocity update changed the trajectory, raised
the normal residual to 0.041837 m/s, and was rejected by the live
0.002 m/s gate. Its 2,101.449 ms GPU time cannot be compared as an
isolated rod-kernel saving because the coupled state and downstream work
changed. The diagnostic log SHA-256 is
`00f4a52a4560ae00a4361f640a8607874ff0c2f6797ea71b554638020caaf439`.
The ablation was removed and the accepted metallib hash above restored.
The next attribution must time the ordered rod stage without changing its
impulses or downstream state.

A fresh, temporary Metal stage-boundary timestamp pass on the accepted
shader measured all 18 encoded pre-contact generalized, Wave32, and
post-contact generalized solves in the same 16x/one-Newton post-entry
probe. Their median encoder times were 2.312, 7.198, and 1.582 ms,
respectively; across the 18 encodes, Wave32 used 132.360 ms and the two
generalized stages together used 71.824 ms. The instrumented grouped step
took 535.668 ms GPU and retained every reported physical field, including
the 0.001867 m/s terminal normal residual. These are stage-boundary GPU
times, not a decomposition of individual Wave32 functions or a full-stitch
performance measure. The retained profile log SHA-256 is
`23a2e64a8f60e99405d961f487b1596bafa2e13eb149a19a472feaeebf744abc`.

A Wave32 trial cached each scene body's static island-membership decision
for the first 64 bodies across velocity sweeps, with the original scan for
higher indices. It preserved all reported physical fields but increased
median Wave32 encoder time to 8.056 ms and grouped GPU time to 548.541 ms.
The trial and timestamp hooks were removed. Two uninstrumented runs on the
restored source took 534.969 and 533.337 ms GPU, retained all 46,080
tetrahedra and the same residual, and restored the accepted metallib
SHA-256 `8059ec81644b9216d80f3a3461ce6738881ffd0fc3cb887951211773777fb199`.
The trial and two restored log SHA-256 values are
`502782784103f4fbbc2ed247bd569afc3f1f925aaeda0fecafa8ba64ef9d6449`,
`dda7d411be09db0897cf0f7e50b71101ad197778f4c13aff1ad30dd6b2a8e592`,
and `90542e6e33ab4d92162aecf7caf7c99801fa4eba9b42e85e19bc363a93b70ae8`.
The 533.337 ms step is 101.3x simulated-time throughput against the
original 1x/3,375.995 ms reference for this state; it is not a 100x
whole-stitch result. Further Wave32 work needs a measured change to its
factorized rod response or repeated contact update, with the ordered
impulses and residual certificate retained.

The Wave32 factorized-rod impulse path now stops its preliminary island
participation scan at the first valid rod contact. Tile construction writes
local constraint indices in ascending order, so that contact is already the
minimum failure key; the subsequent scan-ordered impulse assembly, factor
solve, and residual evaluation are unchanged. Four short post-entry runs
with this change took 527.799, 531.390, 533.355, and 531.776 ms GPU;
four nearby restored-source runs took 538.781, 537.659, 539.172, and
538.727 ms. The medians are 531.583 versus 538.754 ms, a 1.35% gain for
this grouped step. All eight runs reported identical non-timing fields,
including all 46,080 tetrahedra, the puncture channel, zero failed steps,
143 MetalWorld constraints, and 0.001867 m/s terminal normal residual.
The candidate metallib SHA-256 is
`a8e3386ae682f49e3b87f9324fa31918d26a2b732636e26536f78cf8a27bee33`.
The four candidate log SHA-256 values are
`6ad890b667aab1b845a04b837c59ea2e30df5e3f75cd122ce394ec541071d721`,
`a39783c84d51b0aefe6f1042d1661535422b9a00262cedfb13c46d1c318b223f`,
`b697bdc87187c54fa2eae182d53f0e1be8fdea7dfb5d08d42b250ea7eeb8151f`,
and `13d16054e4a49f9bce593d70309be96d063a3fe02b47cfdfb6f71a439551bb97`;
the four restored logs have SHA-256 values
`b94b750f007c0b8bb9c1a8210521c5cbf9ff67ebb24408ed16799910f9c21`,
`8983fdf3d1b33248a297020d261588d9ea32138b1c19056da98feff256e871a9`,
`f9c7cc5120fb1f978ef27e50741bbbe22c78ae6ee986cac0a2c44b52a4233e7e`,
and `74cb5f3bd927e195843d983a4a39b2ae5dc8f9c51dde17076ca9971ecf0fcd13`.
The three focused tissue-entry, opposing-bite-topology, and cadence CTests
passed (log SHA-256
`fea44682a2b20146b5923ed15462be278b9c0d72cbb559925be5afe71908b000`).
A separate default handoff CTest was stopped after two minutes without a
result; it does not qualify this change for the full handoff or whole stitch.
The curved-passage CTest was also stopped after 99 seconds without a result,
before the pull-through CTest started; neither is qualified by this pass.
The earlier single-pass assembly trial overlapped an independent Human GPU
run, so its 546-549 ms observations are not used for performance selection.

The guarded one-Newton free-space option was next applied to a **second**
64x/24 ms robot-held approach from the byte-exact step-621 checkpoint,
using the current shader above. The live 376.0 um tip-to-node clearance
minus the planned 120.0 um capsule sweep exceeded its 200 um guard. The
one-pass segment took 21,336.410 ms GPU versus 22,138.597 ms for a
current-build two-pass run from the same checkpoint, a 1.038x segment
throughput gain. Both passed terminal grasp and rod checks, retained all
46,080 tetrahedra, had zero failed steps and no puncture channels, and
ended at 262.650 um clearance. Their trajectories are distinct: Matter's
maximum residual was 4.753e-7 versus 7.315e-7, and MetalWorld's maximum
normal residual was 0.002301 versus 0.002302 m/s. Both peak contact
residuals remain above the conservative 0.002 m/s per-step screen, so this
is not a qualified robot-contact speedup. The one-pass log and checkpoint
TSV SHA-256 values are
`5a7b49b71677962e93618f49ad94f70de777c58c97dd1a59fc0483d1b24c3ce8`
and `1d31d451b1bb01cf984f6d6bec90dd149d38e35c4f9841d34bb4a9f160a8fb25`;
the paired two-pass values are
`05588e088fe4ce1bd7f9945060ca874f6309e3dded256fb8a5980e3a5c7185ee`
and `365464c679830cb2c09181a1c05f4461adabb733197271ddda87d8651d4b8e19`.

The one-pass step-1005 checkpoint restored byte-exactly (restore log
SHA-256 `d738a4697a99a642fa12cd5d6e628114097819243e83138e08fd6e250fed8325`).
Normal full-Newton guarded approach and entry then advanced the robot-held needle to 150.617
and 71.357 um clearance. Entry produced one positive accepted tip impulse
of 1.021e-8 Ns and 1.023e-8 Ns measured tissue reaction, with accepted
grasp and rod, all tetrahedra, and zero failed steps. The step-1645 contact
checkpoint also restored byte-exactly. The guarded approach and entry took
30,599.504 and 28,389.122 ms GPU; their maximum MetalWorld residuals were
0.004474 and 0.003385 m/s, still above the per-step screen. The continuation
logs and final checkpoint TSV have SHA-256 values
`5bb6acaae919f13d3a9545369007ed77488c39d8561f9bdf21b2901358c981b8`,
`922eb903f224d224b94c34d19ee4ef29e05cb012bc01c391718e9c5e73d4661d`,
and `cd3204562385a963fbee3440589965203f288a4ff624d274e2345a509a33c02f`.
The contact restore log SHA-256 is
`328038ee6dd8152bef054337b26244e43375e9bc94e843d784a8b9dc07dc936a`.
A proposed third one-Newton free-space segment from step 1005 was rejected
before physics because its planned tip sweep could enter the skin contact
band (log SHA-256
`ad4501cd6060d8d34a8faac908e14b4471616c81248e85d0b6c5c3e83de82ca7`).
This extends a source-bound robot-held approach to live contact; it does
not establish a qualified puncture, complete stitch, or 100x whole-path gain.

The step-1645 contact checkpoint above was continued with the guarded
14-step/224-DER-substep brake. The robot-held tip advanced 53.242 um to
18.722 um skin-node clearance at 0.704 mm/s needle speed. Terminal jaw
contacts were 4/6, seat drift was 0.542 um, point slip was 0.00837 mm/s,
and angular slip was 0.03293 rad/s. The grasp and rod checks passed; the
tip had positive accepted impulse (7.989e-8 Ns), tissue reaction was
8.047e-8 Ns, all 46,080 tetrahedra remained active, and no puncture
channel opened. GPU time was 25,713.051 ms. The maximum MetalWorld normal
residual was 0.003267 m/s, above the conservative 0.002 m/s screen. The
step-1869 checkpoint restored byte-exactly. The brake log and checkpoint
SHA-256 values are
`20bc8cc6dea26a27ada235e598bad45d95823197e8291ec3c28e0beee8c92802`
and `72d6a7e852ee21ab76bef3f479607da77348697e8632add886364c656fda8b55`.

One subsequent 62.5 us puncture microstep created one accepted channel
with 1.279e-6 Ns accepted tip impulse and 1.279e-6 Ns tissue reaction,
without a failed step, tetrahedron loss, or rod-check failure. It took
1,455.984 ms GPU. Point slip was 1.004 mm/s, under the 2 mm/s grasp
bound, but angular slip rose to 1.446 rad/s, above the 0.6 rad/s bound;
1.041 rad/s of that motion was about the needle tangent. Both jaws still
touched the handling region (8/7 contacts), but maximum jaw-friction
utilization was 0.999986. The maximum MetalWorld normal residual was
0.005921 m/s, also above the 0.002 m/s screen. This is a physically
accepted **transient**, not a qualified robot-held first bite. Its
step-1870 checkpoint restored byte-exactly. The microstep log, checkpoint,
and restore log SHA-256 values are
`c505dab2572762f3fd8e1651750e907b4cacd0b38c661f9f270406cc0a54e9ec`,
`0786d7c58e0d39659ebfc500d3dc5d26044859b6169a167a2b913a616967ea05`,
and `01d53bbaa06591a0f5c4cec9e7490f8dd2da4bf957a02258525ab5fa30994dce`.
This second, one-Newton-derived lineage reproduces the earlier rotational
failure after braking; increasing free-space cadence has not qualified
puncture or 100x whole-stitch performance.

The puncture replay now reports each giver jaw's solved impulse and torque
about the needle handling-segment tangent. At the unchanged 32+16 contact
sweeps, the jaws carried 5.362e-5/5.380e-5 Ns normal impulse and
1.925e-5/2.255e-5 Ns tangential impulse. Their tangent-axis torque impulses
opposed one another at -5.520e-9/+6.559e-9 Nms, leaving +1.039e-9 Nms
net. The authored materials have zero torsional impulse capacity; this
diagnostic describes the final solved contact wrench, not the entire
puncture-time torque history. Its diagnostic replay log SHA-256 is
`8f6699bcddd8459379cf26a6e700068dd74b5b78f00f1546e8a6a7c48acfe3e0`.

Two **diagnostic-only** material pilots were rejected. A 0.20 mm effective
torsional contact length, equal to the authored insert radius, created
finite torsional capacity but left all reported motion and contact fields
identical to baseline (log SHA-256
`dd66cfb92b858ce7c8554e37c8a0d5613cd7c4bb8e9b9237f2a55a0517ab5157`).
Increasing effective insert/needle static and dynamic friction by 50 percent
raised angular slip to 10.983 rad/s, point slip to 56.694 mm/s, and normal
residual to 0.047749 m/s (log SHA-256
`d2b9d9ce0a181f8eccb89df4cf9d10e260ffb7ad446ac05301c0527abc95d319`).
Neither pilot wrote a checkpoint, and both material changes were removed.

Increasing the existing MetalWorld contact sweeps improved convergence on
the same restored step without changing the material:

| Velocity + final sweeps | Angular slip (rad/s) | Normal residual (m/s) | GPU (ms) |
| --- | ---: | ---: | ---: |
| 32 + 16 | 1.446 | 0.005921 | 1,448.289 |
| 64 + 32 | 1.216 | 0.000413 | 1,500.206 |
| 128 + 64 | 0.877 | 0.000160 | 1,619.558 |
| 128 + 128 | 0.735 | 0.000133 | 1,691.112 |

The 128+128 mode is the current MetalWorld sweep limit. It passes the
0.002 m/s contact-residual screen but still fails the 0.6 rad/s rotational
grasp gate and uses 17 percent more GPU time for this microstep. All four
runs retained one puncture channel, all 46,080 tetrahedra, zero removed
mass, and zero failed steps. The 64+32 and 128+64 replay log SHA-256
values are
`43bc0b5f23d18d728f97b4af3a115c3bfc71cd4d4a18312e1c3ffc6237e6767f`,
and `1691d174779866dcc763eb8056466d4ff175c556dd32d214d537370aedc0d304`.
After removal of the diagnostic material pilots, the source-matched
128+128 replay reproduced the same physical fields and wrote a step-1870
transient checkpoint that restored byte-exactly. Its replay log,
checkpoint TSV, and restore log SHA-256 values are
`249bab95620099abeef2420f5a67bbb62d467400055642611a16a160cf902348`,
`7a166f47f0ae422c25d3b00c1c9bc545b11c42e516f581a6e72ef1cca452766`,
and `01d53bbaa06591a0f5c4cec9e7490f8dd2da4bf957a02258525ab5fa30994dce`.
These are one-step convergence diagnostics, not a faster whole-stitch result.

### Restored rigid-contact state at the first skin puncture

The step-1869 braked checkpoint stored the Matter, rod, scene-body, and
articulation states but discarded MetalWorld's persistent rigid-contact
manifolds. The next process therefore started its jaw/needle contact solve
cold. `MetalWorldBatch` can now accept a complete, capacity-packed initial
manifold cache for a non-resident contact submission; it validates the active
headers and points before upload. The synthetic-skin probe writes a versioned
sidecar alongside the accepted braked checkpoint, bound to the exact TSV
contents and step, and can restore it for the one-step puncture replay. The
sidecar stores only 16 active manifolds and expands into zero-initialized GPU
capacity on read: 5,204 bytes versus the prototype's 4,914,956 bytes. A
one-byte mutation was rejected by the content-hash check before advancing
physics.

The new brake pass reproduced the same step-1869 TSV byte for byte (SHA-256
`72d6a7e852ee21ab76bef3f479607da77348697e8632add886364c656fda8b55`).
Its compact cache SHA-256 is
`3f5772902434d40544fd2192a21f2154ad1982ba2c174cfdd3880633999d0e9b`.
With the same binary and checkpoint, the paired 62.5 us puncture results
were:

| Initial rigid contacts | Warm jaw contacts | Angular slip (rad/s) | Point slip (m/s) | Max normal residual (m/s) | GPU (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold | 0/0 | 1.446157 | 0.001003959 | 0.005921038 | 1437.874 |
| Restored cache | 4/6 | 0.089184 | 0.000073354 | 0.000943873 | 1447.136 |

The warm run passed the existing 0.6 rad/s angular-grasp and 0.002 m/s
normal-residual screens, maintained both jaw contacts, and earned one
puncture channel with 1.2785e-6 Ns accepted tip impulse and 1.2785e-6 Ns
tissue reaction. All 46,080 tetrahedra remained active, with zero removed
mass and zero failed steps. Its step-1870 Matter checkpoint restored
byte-exactly. The accepted tip advance was only 0.01599 um in this single
microstep, so this qualifies a bounded robot-held puncture step, not a
complete bite, pull-through, or knot. The cold replay remained a physically
accepted but unqualified grasp transient. Warm GPU time was 0.6 percent
higher on this pair, so the requested **100x whole-stitch runtime improvement
is not demonstrated**; the sidecar size reduction is a storage improvement,
not a runtime speedup.

The brake, warm puncture, cold paired replay, exact restore, and corrupted
archive rejection logs have SHA-256 values
`a481401b657dba63a6a02f917d1532473a6a13a3aa57c45ff5b3aebfef5fb74d`,
`723e0a308f978bbad6aa90dd94ed88a3e22fcb3779f8f328f875e89a2332bef9`,
`74d9d6001a5e159aa14d3fb93e8d8486e67720064f4bd789248f207c691ac2c1`,
`10e0133baa27278cc69095ed776c0d714adf3698e6f4e01b7cd86a2d84760c0f`,
and `cd4cde005c6502e570d0c1d810205a775e91dbca679fda2c3624de7023db130c`.
The accepted puncture checkpoint SHA-256 is
`513f41761ef3cd42fd9c62a0b8f6542a4e9987a57abd3e281674ce0ddb6e9faf`.

### Warm robot puncture continuation and grouped cadence

The accepted step-1870 puncture now writes its own compact rigid-contact
sidecar, so a subsequent process can restore the same jaw manifolds. A
four-microstep continuation from that checkpoint retained 4/6 warm-started
jaw contacts, the accepted puncture channel, all 46,080 tetrahedra, zero
removed mass, zero failed steps, a qualified grasp and rod, and a maximum
0.001118 m/s MetalWorld normal residual. It advanced the held tip another
0.1283 um over 0.25 ms of modeled time at 3,748.626 ms GPU. No new tip
contact or tissue impulse occurred during those four steps, and the needle
remained 18.431 um from the closest tissue node; this is sustained early
puncture-state control, not passage through the skin. Its step-1874 Matter
checkpoint restored byte-exactly. The accepted puncture sidecar, four-step
continuation log, checkpoint, and restore log SHA-256 values are
`e48012d195acc63cf2e48ad4f9d62128c2bef5a49f7c05300bf29d558909ca02`,
`c608157928a7125b537a85c54a541c88931af421adb8ddfadaea2605e143ec77`,
`04c3b7817ea5370295c99e8950c40462cfc6d65ec2c4d724762901f76313a14c`,
and `62cfa0f22f4bfa707f8871129574389525e3caa0c8b73fe46cc48e34906b7391`.

On that same starting checkpoint, 16+8 contact sweeps reduced GPU time to
3,663.473 ms (2.3 percent) while retaining the grasp and a 0.001685 m/s
maximum normal residual below the 0.002 m/s screen. An 8+4 trial took
3,557.612 ms but its 0.002505 m/s residual failed that screen. The probe
now classifies a puncture with a failed residual screen as a transient even
when its jaw-kinematics gate passes; the 8+4 checkpoint must not be treated
as qualified. The two trial log SHA-256 values are
`8ecbe28619f68f899f5cda84a129b082104ed02b591db7a094809b96cde161f9`
and `0546956fc0b7c29c056ae97468349241ae4cde019a363acad8f1282abcbdee5a`.

An opt-in `--robot-puncture-cadence` groups 2, 4, 8, or 16 base microsteps
into one full-Newton coupled control step, restricted to an accepted
puncture checkpoint with its exact rigid-contact sidecar. From step 1870,
the grouped-4 trial covered the same 0.25 ms as the four separate base
steps at 1,143.572 ms GPU, a 3.28x paired throughput gain. It retained the
same 4/6 jaw contacts, qualified grasp and rod, one channel, 46,080 tets,
zero failed steps, and a 0.000458 m/s maximum normal residual. Tip advance
was 0.1266 um versus 0.1283 um in the four-step reference. The grouped-8
trial covered 0.5 ms at 1,332.906 ms GPU, retaining the same physical gates
and a 0.000452 m/s maximum normal residual. Its step-1878 Matter checkpoint
restored byte-exactly. These are two bounded continuations, not a measured
100x full stitch. Their run-log SHA-256 values are
`58f5e45c1bc04848671eca90d2701e824d35efdb8f2c4a15ff8ea3d2623ee6f8`
and `160cf4966eb4f31134ff74cc9e193afac1a44d31e5a68305acb40bf975d84bee`;
the grouped-8 checkpoint and restore-log values are
`fd3575cf5e6ac569e6ca8c36accc813d8ff8261b3198ff7f2265483a534a788f`
and `3069e45bb9b726899543a295258c2e3dbd4c64fd12e25301276af600c39490c5`.

The grouped-8 continuation also passed with an opt-in one-Newton Matter
budget. On the same checkpoint and 0.5 ms horizon, GPU time fell from
1,332.906 to 676.968 ms (1.97x), while the 4/6 warm jaw contacts,
qualified grasp and rod, one active puncture channel, all 46,080 tets,
zero failed steps, and 0.000452 m/s maximum contact residual remained.
The Matter maximum residual was 2.328e-6 versus 1.626e-6 with seven
Newton passes; both were accepted by the live solver, and the resulting
step-1878 checkpoint restored byte-exactly. The tip advanced 0.2732 um,
but there was no new tip impulse or tissue reaction in this continuation.
This run and the four-base-microstep reference imply about 11.1x
simulated-time throughput for this short post-puncture segment, with a
different numerical trajectory and twice the modeled horizon. It is not
a full-stitch speedup or evidence of passage through the skin. The run,
checkpoint, and restore-log SHA-256 values are
`73985f080f1dd25fd9017f3696fe8c3dd25e268839fc661d85300db490399d41`,
`c5aacc99195dcb017456e44ab8edaf942add5817d6aea27fcc52f93a47153319`,
and `5846a2dfb783dd3c12d78503a2fa15a1552b8a57d46560f3d4f023e919b5b86d`.

At grouped cadence 16, the same step-1870 starting state advanced 1 ms.
Seven Newton passes took 1,745.470 ms GPU; the opt-in single pass took
1,099.454 ms GPU (1.59x faster). Both retained one active channel, all
46,080 tets, zero removed mass and failed steps, a qualified robot grasp
and rod, and a 0.000417 m/s maximum normal residual. The one-pass step
advanced the tip 0.7839 um to 17.314 um modeled node clearance, with no
new tip impulse or reaction. Its Matter residual was 2.329e-6 versus
1.627e-6 with seven passes; its step-1886 checkpoint restored byte-exactly.
Compared with four separate base microsteps, the one-pass grouped-16
setting has about 13.6x simulated-time throughput for this **early**
post-puncture state. It has not been tested at the next load-bearing skin
contact or over a complete stitch. The one-pass run, checkpoint, restore,
and seven-pass paired-run SHA-256 values are
`ae54404af1c552f13e536c66a70d272a49d509d0ddfa3802af0c9e1bf44f1f2d`,
`362b2493b852c010ea37d42cdaa69a86133f9fb6ec8237af4929f9a68a058433`,
`75d08f1321e86efad2cbace395c49aedf9322e83924a5447fc4b8d551570e33c`,
and `7fd0ba1b63b39669ceccf2833ab482a427a3a4460eacdf27584a63951f207287`.
The next performance gate was a source-matched continuation from step 1886
toward renewed tip loading; the executed result is recorded below. The
13.6x segment result cannot be
extrapolated across penetration, thread pull-through, the opposing bite,
or wound closure; those stages still need executed trajectories and timings.

### Continued held needle toward the skin-node contact band

An opt-in `--robot-puncture-grouped-steps` (1-8 control steps) now permits
bounded multi-millisecond continuation from an accepted puncture checkpoint
with its exact rigid-contact sidecar. The first four-step, 4 ms run from
step 1886 preserved the grasp, 46,080 tetrahedra, one channel, zero failed
steps, and a 0.000434 m/s maximum normal residual, but the probe rejected
the checkpoint because there was **zero new tissue displacement**. That
condition is required when opening a channel; an already accepted channel
can persist during motion with no newly measured tissue load.
The guard now accepts this precise no-new-load case only if the restored
active channel count persists. The pre-fix diagnostic log SHA-256 is
`eba0fbf85da540ead80600665dadc0fc33d75cc10aa8599b2ccc633a4cdb9419`.

The source-matched replay passed and produced a step-1950 checkpoint. Four
further 4 ms, one-Newton, cadence-16 continuations reached 20 ms after
step 1886. All five runs retained the grasp, rod, active channel, tetrahedra,
zero removed mass, and zero failed steps. Their final modeled needle-to-node
clearance was 5.571 um, down from 17.314 um at step 1886. Each 4 ms group
took about 4.20 s GPU. Two one-millisecond seven-Newton steps then reached
5.059 and 5.078 um clearance. A subsequent four-millisecond one-Newton
group reached 3.208 um at step 2302; its checkpoint restored byte-exactly.
Across these 26 ms, the eight submissions took 28.698 s GPU in total. Their
maximum reported normal residual was 0.000509 m/s, below the 0.002 m/s
screen. The initial short four-base-microstep reference cost 3.749 s for
0.25 ms; its state and horizon differ, so it is not a matched 26 ms speedup
denominator. No continuation reported a **new** tip impulse, tissue
reaction, or tissue displacement. The channel remained active, but this
does not demonstrate further tissue penetration, through-wall passage,
thread pull-through, or a 100x whole-stitch result. The next physics step
must inspect channel/needle geometry and renewed load before treating the
sub-5-um node clearance as tissue engagement.

The five accepted 4 ms logs have SHA-256 values
`d367d0a446014ab75ee65256cb09da7764ba47e3ded99a861b1c7ea513be730f`,
`f222a16f5ce0336b7c8cf166fe12e3c47a91f5962f396351d1ed26c0febfffbd`,
`926b741c640208e8aa217215391ac0f41f65cc7b71874f8c2b3aa221aa760cf5`,
`690ed5454966af835c2e73804174bfcaf8313d4c5093f2cab0fd1597b54c831a`,
and `650fbe286823e51d77394e1d4c69fc1b27a7eb806ac498910380d6d26b26b487`.
The two seven-Newton and final one-Newton logs have SHA-256 values
`9856f084a71ac3c2be1f93d5ed22bcf591961abae7a468bb5940ae3f2e913201`,
`4cba2b9af3537d401fe5541dee27bb1ab5b51c11987792cd3f7c1dba40ddcc57`,
and `8e649b0cb6877ebc57464c37edd4c982ef5a8ba32718cb2d183525b9cc0a4ec4`.
The final checkpoint and restore log SHA-256 values are
`45d05c176e921f9696bb78bf445274b1c335726dcd7787b6d4a690bc0ae220a8`
and `9af1b876b4d1c3837fa894e9b9d8f8049c0e1a7da03a9c436d1c7b59932e973e`.

### Channel geometry and bounded faster needle motion

A read-only `robot_puncture_channel_geometry` line now accompanies exact
puncture-checkpoint restore. At accepted step 1870, the physics-triggered
channel had a 350 um radius and 63 um half-length; the needle tip was
62.985 um before its proximal end and 188.985 um from its distal end. At
step 2302, the tip was still 42.764 um before the proximal end and
168.764 um from the distal end. The contact shader admits the matching
needle proxy when its node sweep lies within the finite channel plus
87.5 um of axial padding (`0.25 * radius`). Together, the restored geometry
and shader rule explain why the decreasing nearest-node separation did not
produce renewed tip load; this is a source-based inference, not a measured
through-wall passage. The two restore-log SHA-256 values are
`061df8f63380e25d83465ee837f6e14cf56c0d7983a67a09fdb1f4c51b36fec3`
and `e3ba275218a1cf12b3076eff9a133efac56097c8bb47f48f531e4e096e9ee4a4`.

An opt-in `--robot-puncture-speed-mmps` accepts 1-5 mm/s only for a
grouped continuation from an accepted puncture checkpoint and its rigid
contact cache. The default remains 1 mm/s. This changes the commanded
physical needle motion; it is **not** a compute-speed optimization. Starting
from the same step-2302 checkpoint, a one-millisecond 2 mm/s pilot passed
with 0.395 um actual forward tip advance. The 5 mm/s pilot was rejected:
its tip moved 0.867 um backward despite 4.996 um planned forward motion.
The pilot-log SHA-256 values are
`c4150f6e1b0036bbec10ba07eba3e3deeda5ae71cc6827120b870cad325ad3fa`
and `db8aab284eb990cc2cf77a818ba88c37bcc01dcf1b765aac88a2df1e8cb536a9`.

From the accepted 2 mm/s pilot, a four-millisecond group advanced the
tip 5.887 um, and an eight-millisecond group advanced it 13.619 um. Both
retained the qualified grasp and rod, one active channel, all 46,080
tetrahedra, zero failed steps, and normal residuals below 0.002 m/s.
Neither produced a new tip impulse, tissue reaction, or tissue
displacement. The latter run reached step 2510, cost 8,441.346 ms GPU for
8 ms modeled time, and had -2.491 um nearest-node separation. A negative
nearest-node separation inside the channel exemption is not itself a
collision or tissue passage. The step-2510 checkpoint restored byte-exactly;
the needle tip remained 22.825 um before the channel's proximal end and
148.825 um from its distal end. The four- and eight-millisecond run-log
SHA-256 values are
`3965e4db28c032d8c58087e35d71a2376efd955de312bad365f2409704f2ddf7`
and `526d2c09de2939fade0c28ebf8c6791c47da73eb6ccacc0872210e276aff59a2`.
The final checkpoint and restore-log SHA-256 values are
`37d2b8cc81c71875671fa2b298052060756c04de4efc1a3cadc816c49777a03a`
and `89b4a38e85066fddb0cc3914af13a6b4096ba66482ea672482fb80988887efbd`.
The cadence-16 held-needle segment is roughly 1.05 s GPU per modeled
millisecond. Renewed tip loading at the distal frontier, through-wall
passage, thread pull-through, opposing bite, and knot remain open execution
gates.

The grouped cadence option now also permits 32 and 64 base substeps per
submission, within MetalWorld's 64-substep supported limit. A cadence-128
diagnostic stopped before physics advanced because it exceeded that limit;
its log SHA-256 is
`f3e12da088297f493a228fda470f04d5664bff6533f37c4ddd86c67c1dd7f5b5`.
For a matched eight-millisecond continuation from step 2382 at 2 mm/s and
one Newton iteration, cadence 16 with eight submissions took 8,441.346 ms
GPU; cadence 64 with two submissions took 7,142.716 ms GPU (1.18x faster).
The cadence-64 result retained a qualified grasp and rod, one active channel,
all 46,080 tetrahedra, zero failed steps, and a 0.000809 m/s maximum normal
residual. Its tip advanced 13.181 um versus 13.619 um with cadence 16.
Neither run reported new tip impulse, tissue reaction, or displacement;
neither proves another load-bearing puncture event. The cadence-64 step-2510
checkpoint restored byte-exactly, with the tip still 149.261 um from the
channel's distal end. Its run, checkpoint, and restore-log SHA-256 values
are `13a0aacaab9456e1a0fe232439e926b4783f13a0fdc7c3cd20c40ff77e21b4fa`,
`eb5766a944e2a2f2b17ce9d6a6750868033416cf0ffc0bf9506e263776707f0b`,
and `8661090ac52dfb3af67ddc49f250edf2a7f080b01a2d86e159d518899d4048af`.
At 0.893 s GPU per modeled millisecond, this accepted segment remains far
from real time or a measured 100x whole-stitch speedup. Cadence grouping
alone is unlikely to close that gap; further gains require profiling and
reducing the work of each FEM/contact substep without losing the load-bearing
physics gates.

### Native substep trace and contact-iteration trial

An Xcode Metal System Trace of a source-matched 1 ms, cadence-16,
2 mm/s continuation from step 2382 recorded 1,085.642 ms GPU. The exported
GPU intervals account for about 1,083.172 ms. One combined MetalWorld/Matter
substep encoder took 264.505 ms; most other substep encoders took about
51.8-52.4 ms. The trace had Shader Timeline disabled, so these are encoder
durations, **not** measured per-kernel or per-solver-stage timings. The
accepted run retained 7/6 jaw contacts, qualified grasp and rod, one
channel, and all 46,080 tetrahedra. Its log SHA-256 is
`0d3153e60552606258b7f15a5193f930af04dc6e52a212f8176fcc31cc0aa783`;
the retained trace is
`build-skin-wound/skin-robot-profile-cadence16.trace`.

The probe already exposes the coupled rigid-contact iteration counts. On a
matched 8 ms continuation from step 2382 at cadence 16, speed 2 mm/s, and
one Matter Newton pass, reducing the temporal-cone budget from 32/16 to
16/8 iterations reduced GPU time from 8,441.346 to 5,944.478 ms (1.42x).
The candidate retained the qualified grasp and rod, one active channel,
all tetrahedra, zero failed steps, and a 0.000791 m/s maximum normal
residual, but jaw contacts changed from 6/6 to 4/6. It produced no new tip
impulse or tissue reaction, and its step-2510 checkpoint restored
byte-exactly. This is a **no-new-load segment result**, not a puncture or
whole-stitch qualification. Its run, checkpoint, and restore-log SHA-256
values are `7a8455c678c6ce167ba8b3b510941b38f8b32a60dc24a0923af334ad8aa655e4`,
`46f5880ec4f40ae3c7a59ab30937e0943319793babcc71921a3120bf5bb0b48b`,
and `4e3a237dd03356ba9d11cff41e98d514af995492ec759f57d55157216678da2f`.

The next lower 8/4 iteration budget took 4,750.555 ms GPU, but its
maximum normal residual reached 0.002208 m/s, above the 0.002 m/s screen.
The probe therefore archived only a **transient, unqualified** checkpoint;
the process exit status alone must not be read as an accepted puncture
continuation. Its run-log SHA-256 is
`d6211d78890a47d84d15add42e17e993aef4b788f3830c43a16d4690f52344c5`.
The intermediate 12/6 budget passed at 5,418.047 ms GPU (1.56x versus the
matched 32/16 baseline), with 4/6 jaw contacts, qualified grasp and rod,
one channel, all tetrahedra, zero failed steps, and a 0.001139 m/s maximum
normal residual. Its accepted checkpoint restored byte-exactly. The run,
checkpoint, and restore-log SHA-256 values are
`bfa994c0f8812e987788464278b72b9aa2948dba61f482bbaf64518f75772400`,
`f81de730c8539f73830ebd9cce9045c0dfb2cf3c0ac65c49482cc838acee3357`,
and `cdf447ab0b5190096e3ab0ad325e0ff7ee8f85e18508b6104205cc8c9dd5b44d`.

Combining the accepted 12/6 budget with cadence 64 on the same step-2382
start and 8 ms horizon took 4,204.425 ms GPU, versus 8,441.346 ms for
cadence 16 and 32/16 iterations: a measured 2.01x gain on this segment.
It retained 8/6 jaw contacts, qualified grasp and rod, one active channel,
all 46,080 tetrahedra, zero failed steps, and a 0.000818 m/s maximum normal
residual. The tip advanced 13.832 um with no newly measured tip impulse,
tissue reaction, or displacement. Its step-2510 checkpoint restored
byte-exactly, still 148.608 um from the channel's distal end. The run,
checkpoint, and restore-log SHA-256 values are
`61dcedc39f5100e95dc0b6f02d407c44649a35ed0924120091aaac9ca794cb4f`,
`7352c923a7d474f26581bedd53e1e60412fb4838c7cbb242914d7732109cb24f`,
and `c06af20a65d8c192cf13bc24ce0481dd7ff2378c14d7c9311362760994689067`.
This accepted result is still limited to an unloaded channel transit, and
it is far short of a 100x whole-stitch speedup. The next performance work
needs measured kernel-level attribution and load-bearing penetration,
thread, opposing-bite, and knot runs before promoting an iteration budget.

### Contact-node admission and finite channel-chain guard

The contact shader checks the **continuum node** and its predicted position
against the union of same-needle finite channels. Tip position alone is not
the admission test. A read-only audit of the accepted step-2510 checkpoint
found 11 of 638 wound-field contact nodes currently inside its sole channel.
The closest modeled node to the tip capsule (node 8570, -0.853 um capsule
separation) is among them. The checkpoint restored byte-exactly; this audit
does not reproduce the shader's predicted-position test or establish a new
contact event. Its log SHA-256 is
`8e95f8f04e5d58682eac263f09756f0cebe6b2f8011466b8ba1440ab8209ec16`.

The robot probe's earlier `channels <= 1` guard would reject any native
extension of the initial finite tract. The continuation guard now allows
additional segments only when the original active segments remain byte
identical, every segment keeps the physics-triggered tip source, and every
new segment connects downstream through a finite, forward-aligned channel
chain. New-segment count is bounded by the executed physics substeps; the
entry transaction still admits only its first segment. A restored existing
chain may grow without claiming newly measured tissue displacement. The
executed extension and its limits are recorded below.

### Robot-held extension of the first finite channel

The accepted step-2510 checkpoint had a -0.853 um nearest-node capsule
separation, with that node currently admitted by the channel. The probe's
old positive-start-separation preflight rejected any continuation from that
state before physics. The amended preflight permits this exact restored,
currently admitted overlap for puncture continuation only; first entry and
free-space approach still require positive clearance. A 24 ms, cadence-64,
12/6-iteration continuation passed from step 2510 to step 2894, advancing
the held tip 41.599 um with qualified grasp and rod, one channel, all
46,080 tetrahedra, and a 0.001813 m/s maximum normal residual. The
checkpoint restored byte-exactly. The run and checkpoint SHA-256 values
are `369cf6baf168f95ef372d4d22ccbee2ec909df48396bc4f97`
and `f52f93e629f277a8d9881e74648ebedc16bf1c064a9e6f826fd9d73173dd5afd`.

Larger commanded groups were not sustainable near the first channel
frontier. From step 2510, a 32 ms fast group reached 6.635 mm/s maximum
normal residual and 194 um jaw-seat drift; from step 2894, a 16 ms fast
group reached 2.041 mm/s and 191 um drift. Neither was accepted. A later
4 ms step that first created a second channel under full solver budgets
also lost the grasp (177 um drift); its chain geometry was valid, but its
checkpoint was rejected. These diagnostic run-log SHA-256 values are
`003ec138915b03be88966e664f93776ebd253d9769340c3e5019833f1900e3a7`,
`6f80fb9773031fe52692888d594476de3ac395fefc83ffa0428a46e45a73bd37`,
and `7a7907e2e7e85f9de1bf0f1223d3be94bd635e1163cdeec75daa836399c5431e`.

The accepted control boundary used separate 1 ms, cadence-16 steps at
1 mm/s with 32/16 contact iterations and seven Matter Newton passes. At
step 4334 the robot-held tip triggered **one new connected channel segment**:
8/7 jaw contacts, 0.137 um seat drift, qualified grasp and rod, two active
physics-triggered segments from tip proxy 0, all 46,080 tetrahedra,
zero failed steps, and a 0.000578 m/s maximum normal residual. The new
segment passed the unchanged-source, forward-connected chain guard. Its
Matter checkpoint restored byte-exactly. One further accepted 1 ms step
reached step 4350 with both segments intact and a 0.000984 m/s maximum
normal residual; that checkpoint also restored byte-exactly. The growth
run/checkpoint/restore SHA-256 values are
`5a70429b2fc68eb813c3193ae6aad0991e7cc3db2012269aa2cc846f3db153ea`,
`0adb6f8a08247d9cc938fb0d9b838bf8e21b7d89b4fa80265d7bbfbf7e4f1321`,
and `8face8704091cd1d1a16682322ce49cf5b45e6098bad86609781b2acdae185cc`.
The post-growth run/checkpoint/restore values are
`3d61abbeaf5ae9e8fa2444815e8046508b5612a0adaac174bda092bc3a7782d9`,
`b29cf09676d9defa9d05f66584d71173477c837a8c6134e23b21179d3bc4b59c`,
and `f029df4f0165eaf6c1d25d71afdbd12edb245a37354073f88c93f70e307bfb84`.
Neither accepted step reported a new tip impulse, tissue reaction, or tissue
displacement. This is a native finite-tract extension, not through-wall
passage or a load-bearing performance qualification. The growth step took
2,134.205 ms GPU for 1 ms modeled time; 100x whole-stitch speed remains open.

### Second-channel continuation: accepted contact budget and its boundary

Starting from the exact step-4350, two-channel checkpoint above, a matched
4 ms, cadence-64, 1 mm/s continuation with seven Matter Newton passes and
the full 32/16 rigid-contact iterations passed at 4,370.834 ms GPU. Changing
only the rigid-contact iterations to 12/6 passed at 2,841.675 ms GPU,
**1.54x faster for this one step**. Both runs retained a qualified robot
grasp and rod, two connected physics-triggered channels, all 46,080 tets,
zero failed steps, and maximum normal residuals of 0.000572 and 0.001468
m/s respectively. The 12/6 checkpoint at step 4414 restored byte-exactly.
Neither step reported new tip impulse, tissue reaction, or displacement.

The cheaper budget is not valid for an unguarded continuation. The next
4 ms step from its step-4414 checkpoint, again at 12/6, lost the grasp when
tip contact appeared: seat drift reached 121.299 um and normal residual
0.181469 m/s. The probe rejected that state. Repeating that same step with
32/16 passed at 4,341.611 ms GPU, qualified grasp and rod, two connected
channels, all tets, and 0.000615 m/s normal residual. Its step-4478
checkpoint restored byte-exactly. An 8/4 first-step trial failed the grasp
and 0.002 m/s residual screens; a grouped 8 ms first-step trial using one
Matter Newton pass and 12/6 contact iterations lost the grasp. Both
checkpoints were transient and are not continuation evidence.

For a same-start, same-commanded-8-ms comparison, two full-budget 4 ms
steps from step 4350 took 4,370.834 + 4,349.271 = 8,720.105 ms GPU.
The accepted 12/6 then 32/16 schedule took 2,841.675 + 4,341.611 =
7,183.286 ms GPU, a **1.21x selected-schedule gain**. Both trajectories
ended at step 4478 with the robot grasp and rod qualified, two connected
channels, all tets, and zero failed steps; both final checkpoints restored
byte-exactly. The schedules lead to numerically different valid states,
so this is a matched-command performance comparison, not identical-state
replay. The selection was retrospective: failed trial GPU time is excluded
and this does not establish an automatic safe schedule or a wall-clock
speedup for online search. The second segment still has zero authored FEM
contact nodes in its volume at step 4478, and neither accepted route
reported new tissue reaction. Through-wall passage, thread pull-through,
opposing bite, knot, and a 100x whole-stitch improvement remain open.

The retained logs are `build-skin-wound/skin-robot-second-channel-{full4ms,
12x6-4ms,8x4-4ms,fast8ms,12x6-next4ms,full-next4ms,
full-next4ms-baseline}.log`; accepted checkpoints and restore logs live in
the corresponding directories. The reduced first-step, rejected reduced
next-step, accepted full next-step, and full baseline next-step log SHA-256
values are `0cf94ebc74c07597a4717f0e010e06a3f34ef2b164ae3d2be0da669c349343f3`,
`c87dca5523e54cfd5efed3c1045a3e7f65e5f6ff87e3361ff9f39fa882d4785e`,
`b24945152e2b479b2bdc4e2cab5da570d336d7fe100c3e877d7af31ec352b1de`,
and `46b05504b0a6042ed4b1d025782343ae894fb23750785516f8827425df3c60be` respectively. The two accepted step-4478 checkpoint
SHA-256 values are `8838ebfdaa5617d0590a6d3adf62b5c378a1ceeea3b4b38e0102c2bb96086e13`
and `ab0abb71706b33efcb9e21e915115ee91e71bdf54d4b12b125e9aa7770ccacef`.

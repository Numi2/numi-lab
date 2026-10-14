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

### One-Newton continuation and rigid-substep motion guard

On a single rebuilt probe binary (SHA-256
`6a7d627644419907500edecd0384df1b3b68ebe469f23571821faedf97523a2e`),
the same step-4350 checkpoint and 1 mm/s command were advanced for two
separate 4 ms control steps. The 64-rigid-substep, seven-Newton, 32/16
contact baseline took 4,314.781 + 4,311.415 = **8,626.196 ms GPU**.
One Newton pass and 12/6 contact iterations on the first step, followed
by one Newton pass and 32/16 contact iterations on the second, took
2,104.737 + 3,600.555 = **5,705.292 ms GPU**, a **1.51x selected-schedule
gain**. Both paths passed their per-step grip, rod, contact-residual,
46,080-tet, and connected-two-channel gates. The faster path's terminal
normal residuals were 0.001468 and 0.000601 m/s; its step-4478 Matter
checkpoint restored byte-exactly. Both paths still reported zero new tip
impulse, tissue reaction, and tissue displacement, so this gain covers
held-needle channel transit, not load-bearing penetration or a whole
stitch. This is a retrospective schedule comparison; failed selection
trials and process startup are excluded from GPU time.

The probe now exposes `--robot-puncture-rigid-substeps` independently of
the Matter cadence for bounded diagnostic trials. At the same 4 ms Matter
step, reducing rigid substeps from 64 to 32 took 2,679.040 ms GPU but
advanced the needle tip 40.216 um against 3.997 um planned. The previous
100 um overshoot allowance incorrectly accepted this as a continuation.
For puncture microsteps, the guard now caps advancement at twice planned
advance plus 1 um; the 32-substep run is rejected, while the accepted
64-substep speedup above still passes. This rules out the apparent
*unscaled-cache* substep speedup at this state. Four focused static-equilibrium, checkpoint,
synthetic-puncture, and cadence-transition CTests passed after rebuilding.
The 100x whole-stitch target remains open.

The rebuilt full-baseline run logs have SHA-256
`4d2f39ad746a48f5ba30617ce2da49f8e90818344bc26948ce322eb55ce918a6`
and `d2020d50219f2522f79f6748a045cb317f28e4003858f556024c07366398c8ef`.
The rebuilt selected-schedule logs have SHA-256
`87aac20e2cd2d26f5d8a254978c4c06dd7cd33a8c6497c659e5d40444210538b`
and `c43ca7156ac947320a22783af8da346c2b6e24dfe602f7ba029b52709aadf0ba`.
Its final checkpoint and restore log are
`ae7458de2988fd1a818d19f36b35e844c97572840a51ec6420ed95938ca9ce8c`
and `ac08fc1eb0a1a8fd4c5749e366a5a1fcb41d9e4daf48c20c1e10a7341b122a0a`.
The rejected 32-substep log is
`build-skin-wound/skin-robot-second-channel-rigid32-strict4ms.log`
(SHA-256 `790b8434f2dd266545010b259eb9fe735118d24a3c2d1a0809865d14c3d63b13`).

### Timestep-aware contact-cache continuation

The rejected 32-rigid-substep run resumed contact impulses produced at
62.5 us per rigid substep, then applied them at 125 us per substep.
Discarding the warm start made the tip jump 92.260 um and lost the grasp.
Scaling the cached normal, tangential, and rolling impulses by the ratio
of new to old rigid timestep restored the commanded motion and qualified
grasp. The cache archive now records this rigid timestep multiplier in a
hashed `NUMIMF03` header; legacy `NUMIMF02` archives remain readable as
multiplier one. Transitioning from multiplier one to two scales impulses
by two; continuing at two leaves them unchanged; returning to one scales
them by one half. A copy with only its multiplier altered was rejected by
the archive content hash before advancing physics.

On one rebuilt binary (SHA-256
`52c9d2683bc379fc2a07bb6db316167e19962eafff9aadefd13ba259a7e76e0c`),
a matched **single-submission 8 ms** test from the exact step-4350
checkpoint took 8,592.792 ms GPU at 64 rigid substeps per 4 ms, seven
Matter Newton passes, and 32/16 contact iterations. With 32 rigid
substeps per 4 ms, one Newton pass, and 12/6 contact iterations, it took
2,313.979 ms GPU: **3.71x faster** on this accepted held-needle segment.
Both preserved the robot grasp and rod, two connected physics-triggered
channels, all 46,080 tets, zero failed steps, and byte-exact final
checkpoint restore. The faster path's maximum contact normal residual
was 0.001770 m/s, below the 0.002 m/s screen; its needle tip advanced
9.928 um versus 7.994 um planned. Both paths reported zero new tip impulse,
tissue reaction, and displacement. This is an unloaded channel-transit
comparison, not a load-bearing puncture or whole-stitch speedup.

The same 12/6 fast setting failed the next 8 ms block with a 0.004275 m/s
peak contact residual; its exit-zero transient checkpoint is not accepted.
Even 32/16 iterations missed at 0.002295 m/s on the first 4 ms step.
Raising the contact budget to 64/32 for that step passed at 0.000932 m/s;
the following 4 ms step passed at 32/16 and 0.000613 m/s. The resulting
**16 ms accepted continuation** has two connected channels, all tets,
qualified grasp and rod, and a byte-exact step-4478 restore. Its total
GPU time from step 4350 is 2,313.979 + 3,137.991 + 1,932.728 =
7,384.698 ms. This longer route has no source-matched 16 ms full-budget
comparison and no new tissue reaction, so no 16 ms speedup or through-wall
claim follows. A 16-rigid-substep single step also passed its terminal
screen but ended beyond the next puncture step's skin-entry band; it is
not a validated continuing schedule. Four focused CTests passed on the
MF03 binary. The 100x whole-stitch target remains open.

The paired 8 ms baseline and fast logs have SHA-256
`23944ed6e5f538e2f5e016b4d1beeb4b83281329b9986e8fc83d65ee2e9183b9`
and `89d6df509e7219e359814e9a76d50fce1bcf96827af9370c0d3c4ebb55ae5e47`.
The fast 8 ms checkpoint and restore log are
`f8b09838d9bd7b06d9521e585205abd06ba8665e07b3a7f36dd6e062fe4e188c`
and `147611cda5a5a5af4e65c0f8ae94876748babc0012218f1290580172cb3acdee`.
The rejected second fast block, accepted 64/32 step, and accepted next
32/16 step logs are `090ce23ee22a119998003e2faf535655b033f07138e9f94eaf46d543a11f88fe`,
`dd5745d499d098056eeb11af79c11781ce00f0c65313350b170617fa0fc1525d`,
and `6c70bf4df5e7d1263736ec3cca92bb006824d210a008000bd3497c20196c0d2d`.
The final step-4478 checkpoint and restore log are
`5f8d90668478be39075f2c02ba5803b7ece1b0eb78be811ccd01fb7b0a187a88`
and `e4b09ce68ff4f3dd1b6499966b2423a1d824c73bca2018212eaf2670b97dc04e`.

### Guarded transit into the second channel

The MF03 checkpoint chain continued from step 4478 through 4510, 4542,
and 4574 with three accepted 4 ms, 32-rigid-substep, one-Newton,
32/16-contact steps. At step 4574 the tip was 3.987 um from the first
channel's distal end and 6.776 um before the second channel's proximal
end. Two full-budget 4 ms steps then crossed those boundaries: at step
4702 the tip was 1.011 um inside the second channel. Another accepted
4 ms, 32-rigid-substep, one-Newton, 12/6-contact step reached step 4734,
with the tip 4.785 um inside. Every step retained a qualified grasp and
rod, both connected channels, all 46,080 tets, zero failed steps, and a
normal residual below 0.002 m/s; each published checkpoint restored
byte-exactly. The nine submissions from step 4350 through 4734 advanced
40 ms of modeled time in 23,038.042 ms GPU. This is an accepted selected
route, **not** a matched 40 ms speedup measurement or an automatic
scheduler. It is still about 576 GPU milliseconds per modeled millisecond.

At the new second-channel interior state step 4702, a matched 4 ms
full-budget continuation took 4,323.404 ms GPU. The accepted fast
continuation took 1,177.197 ms GPU, a **3.67x local gain** with the same
1 mm/s command. The faster step's 0.001636 m/s normal residual passed its
screen; the full step reached 0.000914 m/s. Neither produced new tip
impulse, tissue reaction, or tissue displacement. The second channel
still contained zero authored FEM contact nodes at the fast step-4734
restore. This verifies transit into an existing physics-triggered segment,
not load-bearing advancement, through-wall passage, or 100x whole-stitch
performance.

The two full-budget frontier/entry logs have SHA-256
`41c1b9a26af0b2b959d80f71794597e7da0d42529091cd2eaca5858e96b0eaf9`
and `4e0626d4df8ef0e8ca195f8cba889994b0ba50a075f9abc59fcdf45f5122ae4c`.
The matched second-channel fast/full logs are
`98b5b55eb50ce20b185aa217ecc5f283091664c9f852fa5be68bc189dd0a5016`
and `4a2cdfe68574b7c24cb83cbee1c60950afcb31fcce4b7918c41dbdd21523fde2`.
The final fast checkpoint and restore log are
`03f9011d824756cceb3a74db152d3eb0d4d9446e8ca569d1e75fedf88e91922d`
and `7607fba5aba5fcf17a5904eb28112678b071d7e0f1f30b5c5be69ecd2169be46`.

### Matched 16 ms second-channel transit schedule

One further 4 ms step from the accepted step-4734 checkpoint used 32 rigid
substeps, one Matter Newton pass, and 32/16 contact iterations. It passed
with 8/6 jaw contacts, a 0.000678 m/s maximum normal residual, two
connected channels, all 46,080 tetrahedra, and a byte-exact step-4766
restore. Its GPU time was 1,940.641 ms; the tip was 8.753 um inside the
second channel's proximal end, with zero authored FEM contact nodes in
that channel. An 8 ms 12/6 trial from step 4734 and a 4 ms 12/6 trial
from step 4766 missed the 0.002 m/s residual bound (0.002695 and
0.002179 m/s respectively). Their exit-zero transient checkpoints were
not promoted.

From that same accepted step-4766 state, the matched 16 ms baseline used
32 rigid substeps per 4 ms, one Newton pass, and 32/16 contact
iterations throughout. Its first two 4 ms steps took 1,991.931 and
1,972.586 ms GPU; the next grouped 8 ms took 3,797.108 ms. The total
was **7,761.625 ms GPU**. A selected schedule used 16 rigid substeps
per 4 ms, 16/8 contact iterations for the first grouped 8 ms, and
26/13 contact iterations for the next grouped 8 ms. Those submissions
took 1,516.404 and 1,896.179 ms GPU, totaling **3,412.583 ms**, or
**2.27x faster** over the matched modeled interval. The reduced first
block passed with a 0.001414 m/s maximum normal residual. Repeating its
16/8 contact budget in the next block missed at 0.002967 m/s, and
24/12 still missed at 0.002535 m/s; 26/13 passed at 0.001141 m/s.
An eight-rigid-substep trial failed the grasp and residual gates and moved
the tip backward. These rejected paths are not performance results.

Both 16 ms paths retained a qualified robot grasp and rod, two connected
physics-triggered channels, all 46,080 tets, and zero failed steps. The
selected path's final step-4830 checkpoint and the baseline's step-4894
checkpoint restored byte-exactly. Their tips ended 23.225 and 22.606 um
inside the second channel, respectively; both still had zero authored FEM
contact nodes there and reported zero new tip impulse, tissue reaction,
or tissue displacement. The different rigid substep counts make the
state-step numbers different despite the same 16 ms modeled horizon.
This is a guarded, unloaded channel-transit gain, not a 100x whole-stitch
speedup or a load-bearing second bite.

The accepted step-4766 run/checkpoint/cache/restore SHA-256 values are
`273b7a56d0a300ea02e3acdff330bb4a65fa6b634df397cd9d8579dad4d83b2b`,
`187c8224bf0cfafe23a47f2234cd2453b75ba93b984bbb4bd397a8fcf688deea`,
`f61aac7ba5436284c2284a896d1821361c21f96160096c6869b162b905d74cf9`,
and `cfab210d0bb0bffbc27294a4606e4b3a8f01c7d9ca75e749bc5d7017466ba2a8`.
The selected first/second block logs have SHA-256
`e4317b214735819880592083224a7ef05d0de1a45a08fae8340e6f3d4ddf3bb0`
and `4bd1c73d5895e69806f4b9804e4718a3601101874e2daf5af9e8073a55694cf4`.
The selected final checkpoint/cache/restore values are
`17cbf005262ed349d593c8d5b9bc592b324d00e5a8899c920ad5dd5e8429ae97`,
`d459b5e5805bf1d0a6f1996ecd7ed55a6153e455d00bda3a8b8f84dfb978f853`,
and `cc0fd7d8425b41b42337993b7ca80d8c1127ab5c8b778dde3aa026b70ae1eef3`.
The baseline final-block log is
`91f734ed055c0337598bae015e9f666010ccbfe7cc750eca258b1196c24a6e6d`;
the rejected repeated 16/8, 24/12, and eight-substep logs are
`c596251728b5f57cc8ecc55b22db238710d1d3633a15927e226477650e62298b`,
`5f22a2122946c9b9c68b03bd96fbc7fa4e175168f04bc16137723d6929c3685c`,
and `aec27c21c26e9c4c6ecc74c70ab73c55623f16e77b5f7a5df4b82c24ddbecc89`.

### Current second-channel GPU attribution and timestep boundary

An opt-in, temporary Metal stage-boundary timestamp pass on the accepted
step-4766 checkpoint measured a 4 ms, 16-rigid-substep, one-Newton,
16/8-contact continuation. Its 565 sampled encoders left the run at
779.915 ms GPU with qualified grasp and rod, two connected channels,
all 46,080 tets, and a 0.001414 m/s maximum normal residual. The 16
Wave32 temporal-cone encoders used 243.289 ms in total (14.723 ms
median); pre-contact and post-contact generalized encoders used 66.556
and 52.599 ms. These three disjoint encoder families account for about
46% of this one command buffer's GPU time. They do not attribute the
remaining work or establish a load-bearing stitch speedup. A broader
1,815-encoder sample perturbed the timing and its summed intervals
exceeded the command-buffer duration, so it was not used as a cost
decomposition. All timestamp hooks were removed after measurement. The
broader trace-log SHA-256 is
`7aee39d502e09c12cd907812d1541a15d26d5b12cdded07709d993549b26bd93`.

A bounded Wave32 trial copied the one-rod factor into 14.8 KiB of
threadgroup memory for reuse across contact iterations. It preserved
every reported non-timing field, but two unprofiled runs took 770.252
and 747.465 ms GPU versus 772.220 and 753.580 ms on the restored
shader. The approximately 0.5% median difference was within this
pair's timing variation and did not justify the extra threadgroup
memory. The trial was reverted. The restored executable and MetalRobo
metallib SHA-256 values are
`52c9d2683bc379fc2a07bb6db316167e19962eafff9aadefd13ba259a7e76e0c`
and `a8e3386ae682f49e3b87f9324fa31918d26a2b732636e26536f78cf8a27bee33`.
A fresh uninstrumented run on those artifacts again passed all local
physics gates with identical reported non-timing fields (log SHA-256
`27656efa9603e4f892b1861f35096eef4a4d8e7407ceaa3ebb64febf3c0dbf3c`).

Reducing to eight rigid substeps while retaining 32/16 contact
iterations gave a 0.001129 m/s contact residual but failed the grasp
angular-speed gate at 0.663 rad/s (limit 0.6 rad/s). Increasing to
40/20 iterations passed at 0.001637 m/s residual, qualified grasp and
rod, and restored byte-exactly. Yet its 4 ms step took 753.342 ms GPU
and advanced the tip only 2.329 um, versus 3.003 um in the nearby
accepted 16-substep/16/8 step at comparable GPU cost. Eight substeps
therefore did not improve this path's throughput or tip progress; no
longer eight-substep route was qualified. The original 16-substep
schedule remains selected. Neither setting measured new tip impulse,
tissue reaction, or displacement. The 100x whole-stitch target remains
open; reducing Wave32 alone cannot close the measured gap.

The focused stage-profile, factor-trial pair, restored-control pair,
eight-substep 32/16 rejection, eight-substep 40/20 acceptance, and
byte-exact restore logs have SHA-256
`e5249fa6d50952530588108c026d00e564a15016134bf2869fce6ba638caf924`,
`11cc46df5cd3772e659d226eec5d24b88fa57d998873226bbbb757c86abaab26` /
`c0b1b0f44a06080c1e6ea150e02573b3295b91441cc4924449a9227427c230a8`,
`e3f5bb179bc767bada0d6987ceda67a1a117a1508e23f0c42d6939b61c9b27f6` /
`9da6e10244a8a5362981ac9fb0e77fea4a9b7392ded281dcde242e5129295277`,
`7db3b3712e76e67cd8b1a231d891def108e158b0aa326a5106b64ab79e63be34`,
`9c765b22489ca9ded61414b5670fcd78029cbc394335502117bc911a3b85154d`,
and `f0aaf785163d1d1636822df89c6fbc6897d2213835ffdf938eadd14c4adcc0fd`.

### Smaller contact budget at the second-channel transit

The same accepted step-4766 checkpoint was continued with the unchanged
solver for two grouped 8 ms blocks at 16 rigid substeps per 4 ms, one
Newton pass, and 12/6 contact iterations. Both blocks passed the robot
grasp and rod gates, retained two connected physics-triggered channels
and all 46,080 tets, had zero failed steps, and restored byte-exactly.
Their maximum normal residuals were 0.001899 and 0.001888 m/s against
the 0.002 m/s limit. GPU times were 1,345.585 and 1,367.397 ms, totaling
**2,712.981 ms for 16 ms modeled time**. This is **2.86x faster** than
the matched 7,761.625 ms full-budget continuation and 20.5% less GPU
time than the previously selected 3,412.583 ms schedule. The tip advanced
6.554 and 8.384 um in the respective blocks. At final state step 4830,
it was 23.546 um inside the second channel; that channel still had zero
authored FEM contact nodes. No new tip impulse, tissue reaction, or
tissue displacement was measured. This is therefore an unloaded
channel-transit result, not a load-bearing second bite or a 100x
whole-stitch speedup.

A third consecutive 8 ms block at 12/6 exceeded the residual limit
(0.002205 m/s). Repeating that block from the same exact step-4830
checkpoint at 16/8 passed with a 0.001324 m/s residual, qualified
grasp and rod, two channels and 46,080 tets, and a byte-exact state-step
4862 restore. It took 1,565.251 ms GPU and advanced the tip 7.883 um.
The accepted three-block route totals 4,278.232 ms GPU over 24 ms
modeled time. There is no matched 24 ms full-budget run, so no 24 ms
speedup is inferred.

Two bounded shader changes were rejected. Removing redundant-looking
Wave32 body-membership tests preserved the local physics gates but was
slower in two matched 4 ms pairs: 820.154/793.529 ms GPU versus
779.217/749.085 ms for the unchanged shader. Doubling rod-row
relaxation let 10/5 contact iterations pass the first grouped 8 ms
(0.001191 m/s residual, 1,274.996 ms GPU), but the next 8 ms failed
at 0.002772 m/s. The unchanged shader already failed the first 10/5
block at 0.004099 m/s. The stronger relaxation also gave no speed
advantage at 12/6. Both shader edits were removed, and the restored
metallib/executable SHA-256 values are
`a8e3386ae682f49e3b87f9324fa31918d26a2b732636e26536f78cf8a27bee33`
and `52c9d2683bc379fc2a07bb6db316167e19962eafff9aadefd13ba259a7e76e0c`.

The accepted 12/6 block logs have SHA-256
`53613aec3f89e35ca931a5080a11041606129aebc3e438b1fe1eb578621e18fb`
and `48e2508fe4e27728b55eaee1d1dd5fe30437a91683dbd24ba00c95dbe60f01bf`.
The final 16 ms checkpoint/cache/restore are
`233d1421f83b9c82853f66f1baa1fe1c3f995165d3821105cba16ac5bdb66148`,
`b352da3cbefaecaca4c159d33327aba6dd0204ed3be131c653beeaef31eec9bd`,
and `7da2a8121b8d5d73e8326d6fb7a15ae7f0bbf2cfdea7aa1856f1945417a86f37`.
The accepted 16/8 third-block log and checkpoint/cache/restore are
`cef188e51b420efdf538045f547a998c47ee8785c89b0a3f8ae26e6de8477047`,
`00b73a6f300b4cb2941dfe4e045f58bc25b373f35d7f638b6732222ad0d73767`,
`fe5c16fc64b5affdedacef4e30ab0c5a787c5079381c15156a6e997f315211f0`,
and `bbcfbd9d0635db4c9acad7f9e3bfc13270c6795778666fc8cdd1a8009c4c23e6`.
The rejected 12/6 third-block, relaxed 10/5 first/second block, and
unchanged 10/5 first-block log SHA-256 values are
`ab0ac9a545fee9957937cda9973a944cbb048bd9e68bc0a68113a03300a94568`,
`3dfe6c518636334ece1d1caea153bb527299d5fc7e9c115c9f944b51ea1e8f58`,
`60c1f95582882a881579a22ba6a25103d7f05b499963f4dc360e1ca933b120e6`,
and `6aab102bbeb40acba8649e60c28f76d9d908f2b5477bfac1cb3b24e5016e2212`.
The unchanged/candidate Wave32 membership timing-pair logs have SHA-256
`0d88213205031c41bc773baaaea54a5e43972ec4ea3a4d2f7c6cbfee97cc2443` /
`f20690e9954fececa201a12d9de7bfcd8fe94a0246b3cc1897df04c45c31455f`
and `17647d6955bce27ca0ae85d5ad2d6355d62caea09b840a1527d78cf609c6673d` /
`d6ee172c2a30c01f3701373395d280e2e9705a2cc89ca415eb2c8326a8648a80`.

### Residual-gated contact early-out (rejected)

An opt-in Wave32 experiment checked the normal complementarity residual
after each pair of contact sweeps and would stop below 0.0015 m/s, under
the 0.002 m/s acceptance limit. From the exact step-4766 checkpoint, the
two grouped 8 ms continuation blocks passed with residuals 0.001899 and
0.001814 m/s, retained both channels and all 46,080 tetrahedra, and had
zero failed steps. The reported solver counts were **18/18** in both
blocks, equal to the full 12 + 6 configured sweeps, so the early-out did
not activate.

The retained adaptive run reported 2,630.993 ms GPU. A same-checkpoint
run with the feature disabled reported 1,395.866 ms with the same 18/18
sweep counts, residuals, and byte-identical final Matter checkpoint and
contact cache. The previously accepted fixed-solver pair took 2,712.981
ms. These timings vary too widely to attribute a gain, and the solver
counts show no saved sweeps; the shader and API edits were reverted.

The adaptive log is
`build-skin-wound/skin-robot-adapt-12x6-8msgroup-verified.log`
(SHA-256 `1f43a51d1815f96a4963ff984d50fbb49e1f6cd5551f99508bfe94993ce1e207`);
the fixed control log is
`build-skin-wound/skin-robot-fixed-12x6-8msgroup-control.log`
(SHA-256 `d7d8a4482d77ba44f978dee118f80e6e7414ea5c0338096ceb3dc1e9c9e829cb`).
The identical final Matter checkpoint and contact cache hashes are
`1556c454ebb997d937d25cd405da9ed51832d7ed818ff6c6592083f69aca17f0`
and `3d1e24903cf731a02200e085158f50dc290df6f32aa07aa9078ea56f61860b6a`.
The temporary instrumented probe and metallib hashes were
`dfe054aa2bc4ca0b825ad86140538a44fb40432aa24e79df23fd12e23406c107`
and `eb92ef77512b875d95a4511d383ecd575415d924b1d9f031114000766d2a2e86`;
the tracked source is back to the unchanged fixed-sweep implementation.

### Rigid-substep reduction (one-step diagnostic, no qualified speedup)

The 16x one-Newton probe now accepts an opt-in
`--tissue-suture-rigid-substeps 1|2|4|8|16` override. It changes only the
rigid substep count for this diagnostic mode; production defaults are
unchanged. Fresh authored-reset runs used a 1 ms grouped post-entry step,
one Newton iteration, ten Matter FGMRES columns, and zero final contact
iterations. Every listed run retained one channel, all 46,080 tetrahedra,
positive minimum determinant 0.999997377, the expected 19.965 um needle
advance, and zero failed steps. These runs have zero active Matter contact
samples at the measured boundary, while MetalWorld reports 143 required
rigid-contact constraints; they do not exercise a load-bearing needle
pull-through or a whole stitch.

| Rigid substeps / velocity iterations | GPU ms (repeat values) | Terminal normal residual (m/s) |
| --- | ---: | ---: |
| 16 / 8 | 1,016.659; 1,581.532 | 0.001867 |
| 8 / 8 | 1,726.933; 2,520.965 | 0.001954 |
| 16 / 16 | 1,670.787; 1,105.554 | 0.001267 |
| 4 / 16 | 1,388.451; 1,127.525 | 0.001389 |
| 2 / 32 | 1,904.422 | 0.001341 |
| 4 / 12 | 3,330.965 | 0.001281 |

The 4/16 candidate passes the 0.002 m/s contact bound with more margin than
8/8 and uses fewer rigid contact sweeps than 16/16. Its two timings have a
1,388.451 ms versus 1,670.787 ms favorable comparison, then a 1,127.525 ms
versus 1,105.554 ms unfavorable comparison. The median of the two runs is
about 9% below the 16/16 median, but the paired repeat reverses direction;
the timing spread is too large to claim a repeatable win. Against 16/8, the
4/16 median is only about 3% lower and raises the Matter residual from
1.835e-6 to 2.487e-6 with seven rather than six FGMRES iterations. The 8/8
candidate stays only 0.000046 m/s below the contact limit and was slower in
both samples. The tested 4/8 point failed at 0.003350 m/s. No substep setting
met the 540.159 ms target reproducibly for 100x simulated-time throughput
over the original 3,375.995 ms 1x reference. A single earlier 526 ms sample
fell below that threshold, but it came from the noisy overlapping sample
series below and was not repeatable.

A separate Numi Matter workload was observed during the earlier 526--2,396 ms
control sample series, so those samples are excluded from comparison. After it
exited, no Numi-specific workload or macOS thermal/performance warning was
observed, but these native GPU timings still varied from 1,017 to 3,331 ms.
The current evidence supports a contact-accuracy boundary, not a performance
qualification. Keep 4/16 as a probe candidate only; do not change the default
cadence or claim a 100x result.

The 16/8 controls are
`build-skin-wound/skin-perf16-clear-control-1.log`
(SHA-256 `b5d59dc23799e616cf06f60532237a1b76fd5ae6897c6924430699b40c5d9560`)
and `build-skin-wound/skin-perf16-clear-control-2.log`
(`7bd8fe7db22568e6418fff0ea956f2e207562140a32ca40fbed0393db9e2085c`).
The 8/8 runs are `skin-perf16-rigid8-clear-1.log`
(`b3df9d6d2182a3b9393d6014ba8734a1555c128694c11b66321e002a945f28ab`)
and `skin-perf16-rigid8-clear-2.log`
(`d2ca34af3ca0c317f08aae490a5bae0165e630213eb5e5da837cb81dad6b8712`).
The 16/16 controls are `skin-perf16-v16-clear-control-1.log`
(`ce06ad52a3f0c7448f4f45dca199f2e085209064bc29429e4db00fce42bb849f`)
and `skin-perf16-v16-clear-control-2.log`
(`1aa4f90a13f95d4d7d0358be42e955212abf5cc5d7bd0310cd435e9332dcfd18`).
The 4/16 runs are `skin-perf16-rigid4-v16-clear-1.log`
(`ae17253595abfe0d39446f3a6b4a9e470f0bec79b4d14356f4c860387b6153e4`)
and `skin-perf16-rigid4-v16-clear-2.log`
(`104601ee6f3cf97ff5bb658d3bff617fa2eef9404f503a242d0ad12049dc5e40`).
The remaining logs are `skin-perf16-rigid2-v32-clear-1.log`
(`09fd8424855e71628153deedea008d7ac852ca3f15214ead5c7f98b6b883a16b`) and
`skin-perf16-rigid4-v12-clear-1.log`
(`a68cf59dd824a6d31e1005deb02b93321bfa17fdb13f20f2801ee9c37d6c91fd`).
The probe, linked library, and Metal library hashes are
`80e78470d6a5d1b8311bbe2607bdf2889b3b568876fa279462cc5c86039317f3`,
`109d520ff1764af9efe36ca043db649d59407bf02c8046deeec794394c698b39`,
and `a8e3386ae682f49e3b87f9324fa31918d26a2b732636e26536f78cf8a27bee33`.

A temporary one-sweep rod-contact override also produced the same published
contact residual, Matter fields, swage error, and strand edge error as the
four-sweep control, while its GPU time was 3,156.368 ms versus 1,016.659 ms.
The reset probe reports zero suture-strand contacts, so this step does not
exercise the rod-contact sweep being varied. The override was removed and
the timing is not evidence for or against that solver optimization. Its log
SHA-256 is `ec18cd0f8a4807d3195384f2c0f4937e34e5d05266eb2fe1f8b52cb95d876f8c`;
the temporary probe hash is
`5863fa9f15bc7b6617d5f43ab85a40d2c60547feaf58440c14e443b17d90666e`.
The next iteration-count experiment needs an accepted load-bearing checkpoint
with active needle/strand contact.

### SSH Mac mini loaded-checkpoint continuation (diagnostic only)

The user-requested heavy-run host was `macmini`, an M4 Pro Mac mini running
macOS 26.6, with 56 GB free on its working volume at the start. A Release build from the exact transferred
working-tree sources completed there. The remote source snapshot commit is
`ec57a577d44b18017145e6fba92aad025db2a7a3`; its provenance record is
`build-skin-wound/mini-perf/source/PERFORMANCE_RUN_PROVENANCE.txt`.
The source probe hash is
`4a878adcd5a56e46cbbe1040d1978f1a31d47c5f340153c8a368450460854783`, and
the executable hash is
`80e78470d6a5d1b8311bbe2607bdf2889b3b568876fa279462cc5c86039317f3`.
The exact local Matter and MetalRobo metallibs were copied to the mini because
the different Xcode versions otherwise produced a different device-program
fingerprint and prevented restoring the accepted input checkpoint. The
step-1870 checkpoint hash is
`513f41761ef3cd42fd9c62a0b8f6542a4e9987a57abd3e281674ce0ddb6e9faf`, its
Matter snapshot hash is
`a5a0fcf0e9f347998152e3919ecea0ef25ddc744716400c3f7297cc6f6e371a4`, and its
contact-cache hash is
`e48012d195acc63cf2e48ad4f9d62128c2bef5a49f7c05300bf29d558909ca02`.

Each candidate restored that same accepted state and ran eight grouped 1 ms
continuation steps at one Newton iteration. The settings varied rigid
substeps, velocity sweeps, and final contact sweeps. Every passing row below
retained the channel, all 46,080 active tetrahedra, qualified jaw grasp and
rod attachment, zero failed steps, and a maximum normal residual below the
existing 0.002 m/s screen. The probe reported **zero new puncture-tip
contacts, zero accepted needle impulse, zero tissue reaction impulse, and no
new puncture channels** in every row. It therefore measures continuation
cost and contact/grasp stability at this checkpoint, not needle penetration
or tissue response.

| Rigid / velocity / final sweeps | GPU ms (repeat values) | Actual needle advance | Max normal residual | Result |
| --- | ---: | ---: | ---: | --- |
| 16 / 32 / 16 | 7,588.945; 7,580.929 | 6.718 um | 0.000417 m/s | Pass |
| 8 / 32 / 16 | 4,509.249 | 7.109 um | 0.001126 m/s | Pass |
| 4 / 32 / 16 | 2,963.626; 2,973.747 | 3.300 um | 0.001475 m/s | Pass |
| 4 / 24 / 12 | 2,656.587; 2,655.880 | 2.789 um | 0.001706 m/s | Pass, narrow residual margin |
| 2 / 32 / 16 | 2,199.182 | -0.173 um | 0.001848 m/s | Rejected: resumed approach lost skin/thread authority |
| 1 / 32 / 16 | 1,822.041 | -115.139 um | 0.006845 m/s | Rejected: grasp and residual gates failed |
| 4 / 16 / 8 | 2,344.808 | 1.137 um | 0.002924 m/s | Rejected: residual gate failed |

The matched 4/32/16 pair averaged 2,968.687 ms, about 2.55x less GPU time
than the 16/32/16 pair average of 7,584.937 ms. Its measured needle advance
was 3.300 um instead of 6.718 um, despite the same 7.994 um planned advance.
Reducing the sweep counts again to 4/24/12 lowered the mean to 2,656.233 ms,
but further changed the trajectory and left only 0.000294 m/s of margin to
the residual screen. These are measured configuration tradeoffs, not an
equivalent-trajectory speedup. Keep the 4/24/12 and 4/32/16 settings as
opt-in probes; do not promote them to defaults or use them to claim 100x
performance, a completed stitch, or tissue-load qualification.

The uninstrumented comparison logs and their SHA-256 values are listed below;
their paths are relative to `build-skin-wound/mini-perf/runs/`:

- 16/32/16: `robot-puncture-c16-rs16-g8/run.log`
  (`a8c2f7e4732697fabbac59526c7e3d0cf28b971a929ac1ff83ddbce315963815`)
  and `robot-puncture-c16-rs16-g8-r2/run.log`
  (`af27c53884c89051f4972ff0e1774756fff3c8819899ef2d8718d2d930f868d3`).
- 8/32/16: `robot-puncture-c16-rs8-g8-r1/run.log`
  (`be9f62313aadd496f687bd63e8d98784916c7609a1483efbc2db5629021cf0e9`).
- 4/32/16: `robot-puncture-c16-rs4-g8-r1/run.log`
  (`8d07acd278b4b5f3e08cefbe028f4f5baa7f5a45978ffa1fb9f5120968148b4f`)
  and `robot-puncture-c16-rs4-g8-r2/run.log`
  (`8a41fbadce469cc221e6eb79d627e0b710dfed0b41d131f53219c4276195007e`).
- 4/24/12: `robot-puncture-c16-rs4-v24-f12-g8-r1/run.log`
  (`aa7f01725727b19134850f3dfabd7b1749964159b87de55402c817b7b87e8473`)
  and `robot-puncture-c16-rs4-v24-f12-g8-r2/run.log`
  (`b9bc33b677c77882a3fd64b840bfd2a1cab85aabde12b474623a99cb9fb985f4`).
- Rejected 2/32/16, 1/32/16, and 4/16/8 logs:
  `robot-puncture-c16-rs2-g8-r1/run.log`
  (`f31fccb44e327d409affde56900c093026bb2f20105bb26bb87118e95138dac9`),
  `robot-puncture-c16-rs1-g8-r1/run.log`
  (`4e2e0dd1fc16320dc3f336e3176c4386d221a763a139f5f1b5ce9dce0e708201`),
  and `robot-puncture-c16-rs4-v16-f8-g8-r1/run.log`
  (`d6df0fcd58861eeff32f84a3a195c84066bb952f419bb7c59f9f4663f72d50fc`).

The Metal System Trace is retained at
`build-skin-wound/mini-perf/profiles/robot-puncture-c16-rs4-v24-f12-g8.trace`,
with Xcode exports beside it. It shows one persistent MetalRobo command buffer
with 69 encoders and about 1.27 ms of CPU encoder-record time against roughly
2.69 s reported GPU time for the instrumented run. The profiler changed the
timing, so that sample is excluded from the benchmark table. The trace shows
the persistent GPU graph dominates this workload, but does not resolve time
by individual shader; the selected GPU-counter profile was unsupported on
this device. A safe kernel-level optimization needs finer supported shader
attribution and a contact-active load-bearing input state. No kernel or
production solver default was changed in this iteration.

### Cached first-bite Newton-budget sweep on `macmini` (diagnostic)

The earlier cadence sweep began after the puncture and had no new needle/tissue
reaction. To measure a loaded event, I restored the cached pre-impact state at
step 1869, including its 16 rigid-contact manifolds, and replayed the first
needle/tissue contact on the user's M4 Pro Mac mini. The input checkpoint TSV,
Matter snapshot, and converted rigid-contact cache SHA-256 values are
`72d6a7e852ee21ab76bef3f479607da77348697e8632add886364c656fda8b55`,
`3bf1b55112844eebcb7631e53c38a38fd3cb479418db4c58b2a56e384d48c433`, and
`3f5772902434d40544fd2192a21f2154ad1982ba2c174cfdd3880633999d0e9b`.
The restore reported byte-exact authority before physics advanced. The source
SHA-256 for this diagnostic build is
`7fe3bd0264c5a7fcfeb0f3ed8cf09389bf684160dd2fd87625768d0ab2490662`; the
Release executable SHA-256 is
`bd22c15ca88cf96d145f0e4ec5569b2a2c60ea22af180219e5d97edf2c0cfaea`.

An opt-in `--robot-puncture-newton-budget 1..7` probe now allows bounded
Newton-count sweeps only from a cached synthetic-skin contact step or cached
grouped puncture continuation. Contact cadence can start from the cached
pre-puncture state for this experiment. The normal solver default remains
seven Newton iterations; these flags do not change production settings.

| One-step simulated interval | Newton budget | GPU ms | Result |
| --- | ---: | ---: | --- |
| 62.5 us | 1 | 434.408 | Pass; 1 channel, loaded needle/tissue impulse, zero failed steps |
| 62.5 us | 2 | 669.698; 653.531; 654.926 | Pass; same published rigid/strand fields as budgets 3–7 |
| 62.5 us | 3 | 813.527 | Pass |
| 62.5 us | 4 | 964.574 | Pass |
| 62.5 us | 7 | 1,163.871 | Pass; default-budget control |
| 125 us | 1 | 477.165 | Pass; one sample |
| 125 us | 2 | 701.909; 699.182 | Pass; repeated |
| 125 us | 7 | 1,209.639; 1,212.690 | Pass; repeated default-budget control |

At the 125 us interval, the two-Newton mean was 700.545 ms versus 1,211.164
ms for seven Newtons: **1.73x faster and 42.16% less GPU time** on this cached
puncture step. Both runs formed one new channel, retained all 46,080 tissue
tetrahedra, qualified the grasp and rod, and reported zero failed steps. The
needle's accepted impulse was `6.41064332e-7 N s`, tissue reaction was
`6.41137332e-7 N s`, tip advance was `92.14765 nm`, and maximum sampled tissue
displacement was `0.9313 nm`. The maximum normal residual was
`0.000776365 m/s`, under the unchanged `0.002 m/s` screen; the minimum
determinant remained positive at `0.999993503`. Two Newtons and seven Newtons
matched on all printed contact measurements and all non-archive-reference TSV
fields. Each saved checkpoint independently restored with `authority_byte_exact=yes`;
their Matter archive references differ, so this does not claim byte-identical
internal FEM snapshots.

One Newton was faster still, but its 125 us sample changed the published tissue
displacement and determinant from the two/seven-Newton result. It is a useful
lower-bound experiment, not the conservative candidate. No Newton budget in
this sweep was promoted to the default.

The contact event did not survive larger single-step intervals: at 250 us
(four rigid substeps) and 1 ms (16 rigid substeps), the probe recorded needle
impulse and tissue reaction but created no accepted puncture channel, so both
runs failed the existing gate. One rigid substep at 125 us also failed the
overall approach gate. An eight-step 1 ms continuation at two rigid substeps
did create the channel and retained the contact residual and grasp, but all
three budgets (1, 2, and 7 Newtons) ended with only 13/15 insert masks and were
rejected. Their GPU times were 2,406.868, 3,314.458, and 6,491.807 ms. This
shows a real cost reduction on that rejected segment, not a qualified
continuation speedup.

The accepted 125 us logs and output TSVs are under
`build-skin-wound/mini-perf/runs/robot-puncture-contact-125us-newton-*`; the
two/seven-Newton TSVs, excluding the run-specific `matter_snapshot` reference,
both hash to
`be4370c85c6ecb400160368d1a26a49fa5ae377b479f36e4b01a469d6ac2e9d8`.
The budget-2 run log hashes are
`5946d37cf3052df7f0cd09eeb9e88e67c7d15c425a0d8c32d6c9e409f98b5825` and
`2bdc25f91cec357d278bed415f50bcf735eda311f9d3c8ee9d5ab2d56f494aca`; the
budget-7 log hashes are
`3bd9538e958eeac0d121ec3de1746f8625d11fbd50894e787d9a5b89c996a782` and
`c2225207b37ef921a59c0f93c61281028dcc2e625a58518900bd7f9623896db9`.
The one-Newton 125 us log hash is
`f0b6c558d0a933b9af2be0db8811edfeba0fa431719c9cab9a69c0a30146b992`.
The rejected 250 us, 1 ms, one-rigid-substep, and grouped-1 ms logs are retained
beside these runs with their measured failure states.

The contact-active slice demonstrates a repeated **1.73x** gain with matching
logged contact measurements. A separate contact-free follow-through from the
accepted puncture state reaches **7.17x** on the same 8 ms horizon, but advances
the needle only 23.3% as far as its seven-Newton reference and produces no new
tissue contact. The result is a performance bound for this diagnostic probe,
not a completed-stitch or 100x result. Through-wall passage, thread pull-through,
and a completed stitch remain unqualified.

### Eight-millisecond loaded-checkpoint continuation on `macmini`

For a longer-horizon comparison, each run restored the same accepted step-1871
checkpoint, produced by the seven-Newton 125 us first-bite run with 16 rigid
substeps and 8 velocity/final sweeps. The checkpoint restored with
`authority_byte_exact=yes`, one active puncture channel, and all 46,080 tissue
tetrahedra. The input TSV, Matter snapshot, and rigid-contact cache SHA-256
values were `1d925bb43b091a768af5ffd6ad651c412c5e80ac2cefc2f8e5b78ee8e209744f`,
`fd878a7ad9cf19129e490dffa4703febbaf03886bfdb883354a1156d8b6fb631`, and
`c9aca406b5cac9c0400b3fd47979d9e583f756d890468bad309fa145b04a5935`. Every
run covered 8 ms, retained all tissue tetrahedra, and began from that same
checkpoint. Cadence is the grouped timestep multiplier; `64 x 2` means two
grouped steps at cadence 64. Rigid substeps and velocity sweeps are shown
separately because they change the simulated trajectory.

| Configuration | Velocity/final sweeps | GPU ms | Needle advance | Probe gate |
| --- | ---: | ---: | ---: | --- |
| 16 x 8, 7 Newton, 16 rigid substeps | 32/16 | 11,516.991 | 6.494 um | Pass; baseline |
| 16 x 8, 2 Newton, 16 rigid substeps | 32/16 | 8,402.258 | 6.494 um | Pass |
| 16 x 8, 1 Newton, 16 rigid substeps | 32/16 | 7,600.428 | 6.494 um | Pass |
| 64 x 2, 1 Newton, 64 rigid substeps | 32/16 | 6,562.304; 6,555.034 | 6.535 um | Pass; repeated |
| 64 x 2, 7 Newton, 64 rigid substeps | 32/16 | 7,511.298 | 6.536 um | Pass |
| 64 x 2, 1 Newton, 32 rigid substeps | 32/16 | 3,469.710 | 5.832 um | Pass; one sample |
| 64 x 2, 1 Newton, 16 rigid substeps | 32/16 | 1,926.168; 1,925.300 | 2.324 um | Pass; repeated |
| 64 x 2, 1 Newton, 16 rigid substeps | 24/12 | 1,641.491; 1,641.368 | 1.515 um | Pass; repeated |

Among the default-cap configurations in this table, the fastest repeated
setting is **7.02x lower GPU time** than the seven-Newton baseline (1,641.430 ms mean), with positive needle advance,
qualified grasp and rod, all insert masks, zero failed steps, and contact
residual `0.000775996 m/s` under the `0.002 m/s` screen. This is not a
trajectory-matched 7x improvement: the needle advances 1.515 um versus 6.494
um in the baseline, the follow-through creates no new tissue contact, and peak
tissue displacement is only `7.28e-12 m`. At 16 rigid substeps and the default
32/16 sweeps, the repeated 5.98x setting advances 2.324 um. With 64 rigid
substeps (128 total rigid substeps across the two groups), one Newton instead
matches the baseline needle advance within 0.7% and gives a repeated 1.76x
reduction, while the tissue state still differs at the nanometer scale.

The faster cases below 16 rigid substeps did not preserve forward needle
motion, so their shorter times are rejected:

| Configuration | GPU ms | Needle advance | Failure |
| --- | ---: | ---: | --- |
| 64 x 2, 1 Newton, 8 rigid substeps, 32/16 sweeps | 1,160.586 | -0.652 um | Reversed needle; continuation gate failed |
| 64 x 2, 1 Newton, 8 rigid substeps, 24/12 sweeps | 1,015.098 | -1.742 um | Reversed needle, grasp unqualified, residual 0.001984 m/s |
| 64 x 2, 1 Newton, 4 rigid substeps, 32/16 sweeps | 777.572 | -145.742 um | Lost grasp and thread authority |

Reducing velocity/final sweeps to 16/8 at 16 rigid substeps also failed: the
needle reversed by 1.030 um. These rejected runs retained the 8 ms horizon and
checkpoint; they are not included in speedup claims. Repeated passing log
hashes are `d0e2b1cad98feeca3fd32b2da991ce03c10649f83263f2aa1119ea3e3151d068`
and `c032bf62492d06a06a6490ba54fa848ef1fea7d8e42d8baf1b1b2cf8fea07e3e` for
the 16 x 8 seven-Newton baseline; `a5eef29d0daf4690d8f70c7ca60ced997f52f16cc9305a510955563f2076c6ae`
and `1bece41ff0c55938f0b2a692b8632a25d24dc624040d9788cd073f8e26e67f6e` for
the 64 x 2, 64-substep one-Newton pair; `b08b19b1ed7fb7859eb6156c6b82279e0a6b0e49259e648ef1a52998597f8ad4`
and `54b9fb4c6b6d4dc0b660b75d4620ed82a9ad0c5e7dbcf3cf2680f9524ae37bdd` for
the 64 x 2, 16-substep 32/16 pair; and
`bb7ead87bf9f0f2129ff9dfbca106c81ce808dc520d83927ebefd17360691f28` and
`5e6fbda50cd4bffff6feea4550748bfe6dd584ee43bd6c20407efdba9b169068` for the
64 x 2, 16-substep 24/12 pair. All logs, including failed boundaries, are kept
under `build-skin-wound/mini-perf/runs/robot-puncture-followthrough-8ms-*`.

### Partial FGMRES cap sweep on the same checkpoint

The accepted puncture continuation cooks a 32-iteration FGMRES budget (restart
width 16). The default-cap control actually used three Krylov iterations. A
runtime/API override now allows a final partial restart cycle; its default
remains unchanged. The one-tetra runtime probe exercised an 8-iteration partial
cycle, reported a healthy solver status, and restored the budget from its state
snapshot. The cached suture continuation also passed with caps of 8, 2, and 1.

| FGMRES cap | Actual iterations | GPU ms | Needle advance | Probe gate |
| ---: | ---: | ---: | ---: | --- |
| 32 (default) | 3 | 1,640.473 | 1.515 um | Pass; one control sample |
| 8 | Not logged | 1,633.473 | 1.515 um | Pass; one sample |
| 2 | 2 | 1,610.220 | 1.538 um | Pass; one sample |
| 1 | 1 | 1,608.185; 1,606.014 | 1.511 um | Pass; repeated |

The one-iteration cap averages 1,607.099 ms, a **2.1%** reduction from the
repeated 24/12-sweep, 32-cap mean of 1,641.430 ms. It retains positive needle
advance, qualified grasp and rod, all insert masks, zero failed steps, and a
`0.000775995 m/s` maximum normal residual. Its tissue displacement is zero at
the reported precision and the continuation has no new tissue contact. The
same 8 ms result is **7.17x faster by reported GPU time** than the 16 x 8,
seven-Newton control, but advances the needle only 23.3% as far. Reducing the
Krylov cap cannot explain or close the remaining performance gap: the default
solve already stopped after three iterations and the cap change saved only
about 2%. No default solver budget was changed.

The default-cap run log SHA-256 is
`44763ee3ecc5c763d9159ab780e0738b17be144f3a687d1e141b1710e892dc02`; cap-2 is
`1059942649b76fda0afcf14a51026fea58d840bfe77d58524e05387cbcea177a`; cap-1
repeats are `d2d299e2b6ab362795b13b2b8ee9ad8430d8cace0b7f85a9207d2e590e7c3d85`
and `7e79f3b9134fe254808c5622c3b00fdc912438f7737ae367baa2f7e615079e4b`.

### Adaptive far-field mesh A/B on `macmini` (diagnostic)

An opt-in nonuniform grid preserves the original 30 x 24 x 1.5 mm specimen,
16 mm incision, 0.60 mm lip gap, eight thickness layers, and all three
measured bite-local spacings. It keeps the 1 mm longitudinal and 0.75 mm
transverse coordinates around the bites, then coarsens the outer field to
1.75 x 1.5 mm cells. The synthetic-skin suture mesh falls from 30 x 32 x 8
(46,080 tetrahedra) to 24 x 24 x 8 (27,648 tetrahedra), 40% fewer elements.
The run still reports local spacing `0.143714, 0.188063, 0.187500 mm`;
minimum rest tetrahedron volume remains `8.44602121e-13 m^3`. The legacy
uniform path remains the default; `--synthetic-skin-adaptive-mesh` is accepted
only on a synthetic-skin suture-contact diagnostic.

The Release build and three matched repeats per case ran over SSH on the M4
Pro Mac mini (macOS 26.6). The synthetic-skin wound coupon CTest passed, and
both mesh layouts compiled into the native contact world and completed the
bounded runs with zero failed steps.

| Diagnostic GPU time | Uniform 30x32x8 median | Adaptive 24x24x8 median | Change |
| --- | ---: | ---: | ---: |
| First tissue-entry transaction, 62.5 us | 900.390 ms | 537.196 ms | 1.676x faster; 40.3% lower |
| One-Newton cadence continuation, 16x grouping | 888.815 ms | 824.404 ms | 1.078x faster; 7.2% lower |

The entry diagnostic retained the same three needle-tip contacts and identical
`9.51956882e-7` reported impulse in both layouts, with 10 maximum FGMRES
iterations and zero failed steps. The cadence diagnostic advanced the tip
`19.965 um` in both cases with the same `0.000868109 m/s` contact residual,
143 constraints, and zero failed steps. The smaller mesh needed 9 Matter
FGMRES iterations instead of 6. Neither diagnostic had any suture-strand
contact; the cadence step also had no active contact. These measurements show
that mesh reduction speeds the entry cost, but contributes little to a later
contact-free grouped step. They do not qualify a loaded stitch or whole
passage.

A separate full `--tissue-suture-passage-only` baseline was advancing, not
hung, but took 88.1 seconds to reach 96/372 steps (three 32-step chunks cost
21.7 seconds of GPU time each). I interrupted it before completion and did not
run the adaptive variant. It provides no passage-speedup result. This long
horizon is the dominant next profiling target; the 100x objective remains
unmet.

The six repeated entry logs are `suture-entry-baseline` (`d27a7156ed2c2ac34ced59762f9b71d45e0ccd1d43f1f5df329cc36a07e3d6f5`), `suture-entry-baseline-r2` (`33b533efb93b8176da0542e8e31b0c035b5dcee30b4acb09bd9edd7bf41bd280`), `suture-entry-baseline-r3` (`52502ed2bcaede5ae1f6c437112483468bea00fd802b6bbd40292e77cf87a595`), `suture-entry-adaptive` (`b75f4ba617e470bcbfb3658b977e66d311fc50b85353474e23a953932c024dfa`), `suture-entry-adaptive-r2` (`0f0898e0dc16662894e881b8d1d43044b4d76ed12e8dd987bf606b6256965bcd`), and `suture-entry-adaptive-r3` (`b4ac1680697aace866633159919001b69f923db5fddba765b0ef5506a6e4aa51`).
The cadence logs are `suture-cadence-baseline` (`e3c819bd0977d9ef74c272fd5838a274ebde1367889e438d9b8a90c254391679`), `suture-cadence-baseline-r2` (`afbabcb7a140814d02e0faa7717c760600f427e1b6e09e9a46efc49457749c2c`), `suture-cadence-baseline-r3` (`bb35db264bbd90cde995b8eb8a1236c3bb510bea443391c48fbd46004fad9215`), `suture-cadence-adaptive` (`da582a8b5c4903b947b024570f4d1aac81c50bb8e8274a81eb21ff76cbb8240c`), `suture-cadence-adaptive-r2` (`6d27cd862c124477cb1db819115ac1317b7e5cfafbf5b328dac8d925140d2f1f`), and `suture-cadence-adaptive-r3` (`a939a15b29580cb8b22e090a5c153592b3cb0c5906fb97b04dc20d2955c7989a`). The interrupted passage log is `suture-passage-baseline` (`93ceeda533d51339349e3faabb509014e79ae626155ac200dd9e3b27db09e7bf`). All paths are under `build-skin-wound/mini-perf/runs/`.

No default mesh or solver budget changed. The optimization is a measured
1.676x entry-stage gain, not a 100x simulation speedup; full through-wall
passage, strand loading, pull-through, and stitch closure remain unqualified.

### SIMD-parallel FEM backtracking on `macmini`

The mixed-FEM nonlinear line search used one GPU thread to scan the full
tetrahedron array for each trial alpha, up to eight times per Newton step. It
now assigns one 32-lane SIMD group to each environment/object, stripes the
tetrahedra across lanes, and reduces minimum determinant, element validity,
and the mixed-volume Armijo sums before considering the same ordered alpha
trials. It adds no device buffers. Since SIMD summation changes reduction
order, the acceptance condition is unchanged, and the measured accepted state
and all other printed fields are compared explicitly below.

The Release build and six candidate repeats ran over SSH on the Apple M4 Pro
Mac mini (macOS 26.6). Three prior adaptive-mesh repeats are the serial
controls. Each candidate run had zero failed steps, and its non-timing output
matches the serial controls byte for byte after removing wall/GPU timing lines.

| Adaptive-mesh diagnostic | Serial median GPU time | SIMD-parallel median | Reduction |
| --- | ---: | ---: | ---: |
| First tissue-entry transaction | 537.196 ms | 428.949 ms | 20.1%; 1.25x faster |
| 16x one-Newton cadence | 824.404 ms | 802.155 ms | 2.7%; 1.03x faster |

The entry case retains the same `0xb44a4607c90cc4e1` accepted-state hash,
three needle-tip contacts, `9.51956882e-7 N s` puncture impulse, one puncture
channel, zero strand contacts, positive minimum determinant `0.999992192`,
maximum Matter residual `3.06571724e-6`, and zero failed steps. The cadence
case retains `19.965 um` tip advance, `0.000868109 m/s` normal residual, 143
required rigid constraints, and zero failed steps; that measured boundary has
no active contacts, so its 2.7% gain is the relevant throughput check rather
than loaded-suture evidence. No individual-kernel GPU timing was captured.

Two focused CTests passed: `metalrobo.surgical.synthetic_skin_wound_coupon`
and `metalrobo.surgical.live_suture_tissue_entry`. The compiled metallib SHA-256
is `e3e3078d319685aca09f225b03b69032c20e99692de597d4dbf4bbe67efc19b8`;
the linked library is
`3e437c43729f28ede38b11d50b9c9a4cad9bb3d6a1d6b59899e40a9e035bc899`.
The final source hashes are `e9e8b41e6eae338acc63f87995b27e3c1757b16be5aa9f0218d36f90bddf6d09`
for `matter/src/metal/fem.metalinc` and
`1e7a217875d8f69a9aae3a6cb4a52e3d6430f367edc968a0a41d1c6f806d9844` for
`matter/src/runtime.mm`.

The three SIMD-parallel entry logs are `line-search-parallel-entry-r1-r3`
(`e2216660af7453964180e6d162323af3f7da8d51c0473629e68cfca001cbbbca`,
`317f924a74119848f3afaa0eeb13f74a3822d204bf58b6a0aead0c6f13f04615`,
`7558e33d2246c92e842bd9c7047b72a1e4cca3d5daee275e41d0a74a28c9d871`).
The cadence logs are `line-search-parallel-cadence-r1-r3`
(`2aa8311ad6e86d976899d738560aa1b0a2a551bea38b24b182ad7b6027fa7f15`,
`74cfeccec287667998a36434cabfaae302a5b43f353b6bebb7bb29194d81c6c7`,
`e3952babb21af7bd235573cc18149cc33f1d0229c9005ff6c6ec8afa8a37f09d`).
Their serial-control triads are the adaptive entry/cadence logs listed above;
a fresh serial entry control is also retained as `line-search-serial-control-20261002`
with log hash `1502b7c65093b43e34d243e08026b2728ebd9f8862a41788e561ce55141edbf0`.
All paths are under `build-skin-wound/mini-perf/runs/`.

A per-lane early skip after finding an invalid element was tested and reverted:
its entry median was `428.433 ms`, indistinguishable from the accepted SIMD
version, and its cadence median was slightly slower (`802.428 ms`). A cached
125 us loaded-contact checkpoint from the previous metallib was also tried;
restore rejected it before physics with “Matter snapshot does not match the
initialized device program.” No loaded-contact speed claim uses that attempt.
The accepted entry optimization is a measured short-step gain, not a 100x
whole-stitch result; the 100x target remains open.
The next useful gate is a source-matched loaded-contact run with a checkpoint
created under this metallib, followed by attribution of repeated FEM/FGMRES
work during the long deformable continuation.

### Balanced local mesh comparison on `macmini`

An opt-in `--synthetic-skin-balanced-mesh` keeps the 1 mm incision grid,
0.75 mm bite/seam grid, and eight through-thickness layers, while coarsening
only the remote in-plane field. It preserves the 30 x 24 x 1.5 mm specimen,
16 mm incision, 0.60 mm lip gap, local bite spacings
`0.143714, 0.188063, 0.187500 mm`, and minimum rest-tet volume
`8.44602121e-13 m^3`. The mesh has 19,200 tetrahedra, 58.3% fewer than the
30 x 32 x 8 uniform layout. Three Release repeats per configuration ran on
the M4 Pro Mac mini with the same probe and Metal library.

| Diagnostic | Uniform 30x32x8 | Balanced 20x20x8 | Reduction |
| --- | ---: | ---: | ---: |
| First tissue-entry transaction | 712.904 ms median | 317.758 ms median | 55.4%; 2.24x faster |
| 16x one-Newton cadence | 802.155 ms reference | 778.356 ms median | 3.0% lower |

All entry repeats retain three needle-tip contacts and exactly match the
uniform run's `9.51956882e-7 N s` impulse, `9.50467630e-7 N s` axial impulse,
one puncture channel, and zero failed steps. The accepted state hash differs
because node and element counts differ; maximum tissue displacement and the
Matter residual also differ slightly. The cadence repeats advance the tip
`19.9654 um`, retain 143 required constraints and the `0.000868109 m/s`
terminal normal residual, with zero failed steps. This cadence boundary has no
active contact and supports only a post-entry throughput comparison.

A more aggressive 14 x 16 x 8 candidate used 10,752 tetrahedra and ran at a
212.307 ms entry median. Its contact count changed from three to one and its
tip impulse rose 16.4%, so it is excluded from the selected mesh result. The
balanced layout remains opt-in for other synthetic-skin modes. Entry and
contact-free cadence alone do not establish passage performance. The matched
through-wall results below qualify only this bounded synthetic-skin passage
probe, not loaded pull-through or stitch closure. The 100x target remains open.

The probe source SHA-256 is
`d0a57acd024491d9640c6bbbf4ac02756184bdc29092152127fbb010aafc2d29`; the
Release probe, linked library, and Metal library hashes are
`1601b7fb1624c7bdcd6a37f98f92bd83d3bbf4d7bfdd837f5894d22fb90f3108`,
`3e437c43729f28ede38b11d50b9c9a4cad9bb3d6a1d6b59899e40a9e035bc899`, and
`e3e3078d319685aca09f225b03b69032c20e99692de597d4dbf4bbe67efc19b8`.
Focused CTests `metalrobo.surgical.synthetic_skin_wound_coupon` and
`metalrobo.surgical.live_suture_tissue_entry` passed 2/2; the retained
`LastTest.log` SHA-256 is
`102b2f88cddb68322971fe62a6661ae3b623c696cc0fe7f86d5f16e0c0550c59`.

Run logs are under `build-skin-wound/mini-perf/runs/`. In uniform-entry,
balanced-entry, compact-entry, and balanced-cadence repeat order, their
SHA-256 values are:

- `uniform-simd-entry-r1-r3`: `33efcb50db2f956c2951d068f6f2f1f77575346551dbcf7c80f5d9ccb2e95174`, `95905441fd91b735ad4999d75d3fc76982f36931a565952f6a10aa2da24a385a`, `a16add035d0c8514fe3268602e91862e58c4593021eaafdeb6bdad7e4276d8a`
- `balanced-mesh-entry-r1-r3`: `dc4697cd49e951e3fcf5bd9757cf0f33a8fae241b837eeede25867e3ef483748`, `a2c0b0f515fca7b02dc256f7e44d4d99526feb6e9e994dd9ebca1bec8a60c1f4`, `063e9da2946d1556776dc443d7e4a3746ccd0f5677ab2f872975bb6a68bc0527`
- `compact-mesh-entry-r1-r3`: `b0e86b116bc12c5d7f8a0d3cfac5fdfd65298f8fc456c12d3b7a03dc65345792`, `84537c97067026904ce54c5219d7a47d83a81c3bd11b5a83c721728a6f56dde3`, `1b0019d2ed9cbd1ba6af23febbf4c830c637b0abb9b460a255aea9a6dae2523a`
- `balanced-mesh-cadence-r1-r3`: `29f4bb57074e03e60aeb59ab65803264393f1f9c93c28fe241428b14b9aa7235`, `35e7bcf541cb4a5f47d5251162dfd4fbfb9fc83ad32f77fa779f9202a6c6af3e`, `c9657badea3b7ae688f42693b2600ce292338e7ee0db352605b96e57a83c7e97`.

### Full-passage cadence comparison on `macmini`

The uniform 30 x 32 x 8 control completed the 372-group through-wall passage in
195,019.115 ms GPU time, with 1.836 mm tip advance, 0.200 mm distal clearance,
12 linked puncture channels, no removed tissue mass, and zero failed steps.
The balanced 20 x 20 x 8 mesh completed the same path twice at the existing
4x Matter cadence: 119,676.026 and 119,599.047 ms. Both runs produced the same
`0x806897f2818fc838` state hash and all printed passage metrics matched. This
is a repeatable 1.63x gain over the uniform control.

For the explicitly selected balanced synthetic-skin through-wall mode, the 8x
cadence groups eight base DER substeps into each 500 us Matter transaction.
Other suture phases retain their existing cadence. It passed the same
through-wall gate three times in 188 groups at 94,643.794, 94,673.184, and
94,774.957 ms GPU time, a repeatable 1.26x gain over balanced 4x cadence and
2.06x over the uniform control. All three runs produced the same
`0xc42ce05051f76556` state hash, 12 linked channels with 1.512 mm total tract
length, 0.2193 mm distal clearance, 1.8553 mm tip advance, 0.019984 m/s
terminal speed, positive minimum determinant, zero removed mass, and zero
failed steps. The 8x speed error is 16.1 um/s, within the probe's 20 um/s
terminal-speed tolerance. This remains synthetic research-default evidence;
it does not qualify tissue calibration or a full stitch.

| Layout and cadence | Passage GPU time | Groups / base DER substeps | Outcome |
| --- | ---: | ---: | --- |
| Uniform 30x32x8, 4x | 195,019.115 ms | 372 / 1,489 | Passed |
| Balanced 20x20x8, 4x | 119,676.026; 119,599.047 ms | 372 / 1,489 | Passed twice; same state hash |
| Balanced 20x20x8, 8x | 94,643.794; 94,673.184; 94,774.957 ms | 188 / 1,505 | Passed three times; same state hash |
| Balanced 20x20x8, 16x | No accepted timing; 43,436.460 ms chunk GPU / 44.53 s wall for 8/0 diagnostic | 96 / 1,537 planned | Rejected: 0.019968 m/s exit speed and 0.003564 m/s final contact residual miss their bounds |
| Balanced 20x20x8, 12x | No physics run | — | Runtime cadence API rejected this multiplier before advancing state |

An 8x first-entry transaction retains the same three tip contacts and exact
normal and axial impulses as the uniform and balanced 4x entry repeats. A
Metal System Trace of that entry attributed about 312 ms to the persistent
batched MetalWorld graph and less than 1 ms to explicit state readback. The
short trace did not expose shader timing: Metal GPU Counters reported that its
counter profile is unsupported on the M4 Pro, and Shader Timeline remained
disabled. The trace points to the combined GPU graph as the next optimization
target but does not identify an individual kernel. The 32/16 reference path is
2.06x faster than the uniform passage; the bounded 8/0 candidate below reaches
3.58x with elevated intermediate contact residuals. The 100x objective remains
open.

The completed passage logs are retained under
`build-skin-wound/mini-perf/runs/`; trace artifacts and focused test output are
under `build-skin-wound/mini-perf/traces/` and
`build-skin-wound/mini-perf/tests/final/`.

### Reduced contact-solver budget candidate — 2026-10-02

On `macmini`, the balanced 20 x 20 x 8 passage at 8x Matter cadence was run
twice with eight main velocity sweeps and no final sweeps, compared with the
32/16 reference budget. GPU times were 54,322.204 and 54,479.574 ms (54,400.889
ms mean), versus the reference median of 94,673.184 ms: 1.74x faster, or 42.5%
less GPU time. Both candidate runs produced the same state hash
`0xea5f16a2aa2a9537` and passed the existing through-wall checks: 12 connected
tract channels, 11 links, all 19,200 tetrahedra retained, zero removed tissue
mass, 0.2193 mm distal clearance, 1.8553 mm tip advance, 19.983936 mm/s exit
speed within tolerance, and zero failed steps. The hash differs from the
32/16 reference, so this is not a byte-identical solver substitution.

Per-chunk contact residuals are now logged without turning the intermediate
chunk samples into new passage acceptance gates. The repeated 8/0 run reported
normal residuals of 0.002443, 0.003115, 0.002382, 0.007623, 0.011440, and
0.001879 m/s. The final chunk is within the existing 0.002 m/s terminal bound,
but earlier chunks exceed it; treat this as a bounded performance candidate,
not a general solver default or qualification for the loaded robot sequence.
The stricter per-chunk check also rejected the 32/16 reference at its first
chunk (0.003045 m/s), so it was removed while the per-chunk diagnostics remain.
The 100x objective remains open.

A 16x cadence trial with the same 8/0 budget took 43,436.460 ms across its
three passage chunks but failed the 20 um/s exit-speed tolerance and the
2 mm/s terminal contact-residual bound. Its geometry, tract, tissue, and rod
checks otherwise passed. The faster measured time is therefore rejected; the
8x candidate remains the best result from this sweep.
The rejected run log is
`/Users/n/numi-skin-suture-perf-100x/runs/balanced-mesh-suture-passage-16x-8i0f-20261002/run.log`
with SHA-256
`50f19156c1c4244ea6f6e2cc9a6199f2120d4717e1b95f0ae92f44269f44f5f6`.

### Channel-only puncture fast path — 2026-10-02

The Mac mini kernel profile showed that the entry transaction spent 30.451%
of sampled GPU timestamp ticks in topology certification, 8.712% in topology
conservation, 9.298% in node-mass rebuild, and 7.408% in incidence scatter.
The detector used inside Newton iterations emits only physics-triggered
cylinder punctures; these append embedded channels without changing FEM nodes,
fields, tetrahedra, mass, or node incidence. The runtime now skips redundant
incidence rebuilds for that detector-only path. Conservation and topology
certification return early only when node, tetrahedron, and cohesive-face
counts and accounting are unchanged, and the topology-generation increase
matches the added channel count. Authored remeshing and any other topology
change still take the full rebuild, conservation, and certification path.

Five focused Mac mini CTests passed: topology mutation, millimetre-scale
topology conservation, topology rollback, synthetic-skin tapered puncture,
and opposing-bite topology. Three balanced 20 x 20 x 8 entry runs took
189.550, 189.744, and 189.366 ms GPU (189.550 ms median), versus the saved
317.758 ms baseline median, a 1.68x entry speedup. All three retained one
puncture channel, all 19,200 tetrahedra, zero removed tissue mass, and zero
failed steps. Their accepted-state hash was `0x8af09b89480505a8`; it differs
from the earlier `0x6b47347dfaadfbe3` because the channel-only path no longer
applies checkpoint-relative corrections to FEM state that the puncture did
not mutate.

The matched 8x, 8/0 through-wall passage took 47,035.138 and 47,105.111 ms GPU
(47,070.124 ms mean), 13.5% less time than the earlier 54,400.889 ms 8/0 mean.
Both runs produced hash `0xde5b1f892ae854b6`, retained all 19,200 tetrahedra
with zero removed tissue mass, built 12 connected channels and 11 links, and
passed with 1.8553 mm tip advance, 0.2193 mm distal clearance, 19.983936 mm/s
exit speed, 1.879 mm/s terminal normal residual, and zero failed steps. The
hash differs from the earlier 8/0 candidate, so this is an outcome-matched
performance candidate rather than byte-identical replay. Against the saved
32/16 reference median of 94,673.184 ms, the combined 8/0 and topology
fast-path result is 2.01x faster. Intermediate chunk residuals still exceed
2 mm/s; only the established terminal residual is an acceptance bound.

Lower budgets were measured and rejected against that terminal bound:
4/0 took 40,660.226 ms but ended at 3.242 mm/s, 6/0 took 43,821.403 ms and
ended at 2.483 mm/s, and 7/0 took 45,407.297 ms and ended at 4.673 mm/s.
Their through-wall geometry and exit-speed checks passed, but the 8/0 setting
remains the fastest passing solver budget. The 100x objective remains open.

The source-matched run, CTest, and rejected-budget logs are retained at
`/Users/n/numi-skin-suture-perf-100x/runs/topology-fastpath-passage-8x-8i0f-20261002/`,
`.../topology-fastpath-passage-8x-4i0f-20261002/`,
`.../topology-fastpath-passage-8x-6i0f-20261002/`, and
`.../topology-fastpath-passage-8x-7i0f-20261002/`. The passing run log SHA-256
is `83bfa18a6e0be741c49ce291ef4d43b990b4fffd8cf3056121b207e29471849e`; the
five-test CTest log SHA-256 is
`d2cbcde67472220326874e461ec9d4bbeff9bde95e9d61758c7d905e1c2825e1`.

### Exact contact-sort reuse — 2026-10-02

The contact surface sorter now checks whether its previous ordered primitive
indices are still an exact permutation of the current active set and remain
monotonic by the current transformed lower-x key, including stable index order
for ties. It reuses that order only when all checks pass; any inactive old
primitive, changed active count, or ordering inversion falls back to the full
eight-pass radix sort. This makes slowly moving or stationary deformable
surfaces avoid repeated radix work without using a displacement tolerance or
loosening the sweep bound.

Nine Mac mini CTests passed after this change: the five topology/skin tests
above plus the synthetic-skin wound coupon, shared rigid coupled contact,
articulated contact-boundary recovery, and post-commit contact guard. Three
entry repeats took 177.999, 177.872, and 178.621 ms GPU (177.999 ms median),
with the same accepted-state hash `0x8af09b89480505a8` and zero failed steps.
That is a further 1.065x over the topology-only fast path and 1.79x over the
saved entry baseline median.

The matched 8x, 8/0 passage then repeated at 43,829.072 and 43,837.594 ms GPU
(43,833.333 ms mean). Both runs retained the earlier passage result exactly,
including state hash `0xde5b1f892ae854b6`, 12 connected puncture channels,
11 links, all 19,200 tetrahedra, zero removed tissue mass, 1.8553 mm tip
advance, 0.2193 mm distal clearance, 19.983936 mm/s exit speed, and a
1.879 mm/s terminal normal residual. This is a further 1.074x over the
topology-only fast path and 1.24x over the earlier 8/0 passage mean; combined
with the 32/16 reference it is 2.16x faster. The 100x objective remains open.

The sorter and topology run set is retained at
`/Users/n/numi-skin-suture-perf-100x/runs/contact-sort-reuse-passage-8x-8i0f-20261002/`.
The passing full-passage logs have SHA-256 values
`f9f719ab9c5cfcbf4cbeb7a4697cdd1ac44bb419e7fe3bbc2a4f312e4f7da141` and
`d21f2f38bc60202588f0e43c6c19a205752016852ba8c42b542b25b7a335393d`; the
nine-test CTest log is
`14c6a3712eee7938e699b686bedb431a936a805fd5052171bed15b1a3b837cde`.
The source hashes are runtime `c3411d5c891bb2d63a3be9be4a63e8e5ed00325de6a0b982e9b5991beadb1837`,
topology shader `ddddcc3a07b30c007731e4bae0a8200efc284fd0c8aab5ff2f21646a51ff8db6`,
and contact shader `bdcee89ad4b51e308c9e40de31fb9d83efc4dcc552ec0bc4619ea6a5404667ea`.
The matching Metal library SHA-256 is
`859e81cd0fdeb4820926f285d27c48960a96d1a73ed9f4bc796f3847ff1ce633`.

Reproduce the bounded 8/0 candidate explicitly with:

```sh
./build/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-suture-passage-only --synthetic-skin \
  --synthetic-skin-balanced-mesh \
  --velocity-iterations 8 --final-velocity-iterations 0
```

The repeated run is
`/Users/n/numi-skin-suture-perf-100x/runs/balanced-mesh-suture-passage-8i0f-residual-diagnostic-r2-20261002/run.log`
on the mini; log SHA-256:
`99fdbce396f5bad6f8f4a7d24e4b218519d71302399aabb51a7885c68834ba77`.
Its probe source SHA-256 is
`6f3fdd1d744e79ddc012345cf84ec65de788a19d46d0d163355406c3ec2dff67` and binary
SHA-256 is
`76b85f943e1d9ace669fc177a78ae4429096541bcfadbfbc85088ddfad69e021`.
Focused synthetic-skin coupon and live-entry CTests passed 2/2; their retained
`LastTest.log` SHA-256 is
`c9bed8811d0856c685c3f79ad3b8f3a9b356db47bb2b22ed8780643a9a6e58e1`.

### Compact local mesh through-wall result — 2026-10-02

The opt-in `--synthetic-skin-compact-mesh` layout is now available for the full
synthetic through-wall passage diagnostic. For matched-horizon comparison it
uses the same 8x Matter cadence as the balanced layout; the normal default
mesh, passage cadence, and 8/0 contact budget are unchanged. The compact
14 x 16 x 8 mesh retains the full specimen extents and eight thickness
layers, with 10,752 tetrahedra versus 19,200 in the balanced 20 x 20 x 8
layout (44% fewer).

Two complete Mac mini runs passed the established terminal contact residual
bound and produced identical state hash `0xe04decedd9fb41ac`. They each ran
188 groups / 1,505 base DER substeps with an 8/0 contact budget. GPU times
were 38,206.760 and 38,131.589 ms (38,169.174 ms mean; 0.20% repeat spread),
versus 43,833.333 ms for the balanced 8x/8/0 repeats: 1.148x faster and
12.92% less GPU time. The state hash differs from the balanced mesh because
the discretization changed.

Both runs formed 12 linked puncture channels with 1.512 mm total tract length,
retained all 10,752 tetrahedra and zero removed tissue mass, advanced the tip
1.8553 mm, reached 0.2193 mm distal clearance and 19.983936 mm/s exit speed,
and ended at a 1.879475 mm/s normal contact residual under the 2.0 mm/s bound.
Minimum determinant was `0.999999762`; failed steps were zero. Intermediate
chunk residuals reached 7.623 and 11.440 mm/s. They remain diagnostics under
the existing terminal-only residual criterion.

A compact 4x/8/0 passage diagnostic took 46,149.257 ms and completed the
probe's geometry checks, but its terminal normal contact residual was
8.104 mm/s, above the established 2.0 mm/s bound; it is rejected. The accepted
candidate is specifically the repeated 8x/8/0 result above.

A compact 16x/8/0 trial completed its 96 planned groups in a summed 34.408 s
of chunk GPU time, but the terminal contact residual was 3.564 mm/s and the
exit speed was 19.968 mm/s. Both endpoint bounds fail, so this faster point is
rejected; 8x remains the fastest accepted compact setting. Its run-log
SHA-256 is
`b49121ab06af79770c47adf5d1bc2dfabebf474fb320efeedc198c8a94f1688a`,
with provenance SHA-256
`d2c4a6bc6926c43eb4d0710d083c1f557e50d0e141951381a19ce6b9fc1dc0c2`.
The 16x probe source was
`385e2f248f04b32f61d8712eb287943f759da8bfe097b112bf23cf59a491a6cb` and its
binary was
`721f14a06b3ba297e14f94cc0d5965fcab0abd5bdc2f61532764718637031e99`.

The passing logs are
`/Users/n/numi-skin-suture-perf-100x/runs/compact-mesh-passage-8x-8i0f-equal-cadence-20261002/run.log`
and
`/Users/n/numi-skin-suture-perf-100x/runs/compact-mesh-passage-8x-8i0f-repeat2-20261002/run.log`,
with SHA-256 values
`2d3bda846f323a3dd95e1bdbba65e5f464d41b4d1ed8e14f35766722755fa215` and
`f425c3df0e0211115f24a8d20736b8354e1c987490be52d156ab4c8786128276`.
Their shared provenance-file SHA-256 is
`3947fdb53fb92ac771a9e2b5b4f01b32763262c7cf1348c474d58aa385f82d0a`.
The rejected 4x log is
`/Users/n/numi-skin-suture-perf-100x/runs/compact-mesh-passage-4x-8i0f-residual-reject-20261002/run.log`
with SHA-256
`bbad7ab9d0050618533fe7f297b9c38d5bfb02d1479f5734a17998daeeda5bb1`.

The nine focused topology, contact, rollback, and synthetic-skin CTests passed
again; the retained log SHA-256 is
`caa19730c2f60a5017aaa615d84c3892cbf477b02a35584f4c93f2189625a289`.
The probe-source SHA-256 is
`0a476b54a5e345edf1eb5f4f179151652830059a88feaf6305210c0d11c11a89`; the
probe binary is
`4c941553a077b03703a97abac3673c95edf584c67b7d1b643d784aade7aad1d5`, and the
Metal library remains
`859e81cd0fdeb4820926f285d27c48960a96d1a73ed9f4bc796f3847ff1ce633`.
The speedup is a repeatable 1.148x over the balanced 8x/8/0 passage, not a
100x result; the performance objective remains open.

### One-Newton compact passage budget — 2026-10-02

The opt-in `--synthetic-skin-newton-iterations 1` override applies after the
accepted entry and only to the following compact through-wall passage. It
retains the 8x Matter cadence, 8/0 contact-solver budget, mesh, and all
acceptance tolerances while reducing the passage Newton budget from five to
one. Two complete Mac mini runs passed the existing gates and produced the
same state hash `0xb9b2ef613090d2f0`. Each formed 12 linked puncture channels,
retained all 10,752 tetrahedra with zero removed tissue mass, advanced the tip
1.855291 mm, reached 0.219291 mm distal clearance and 19.983936 mm/s terminal
speed, and ended with 1.879475 mm/s terminal contact residual. Minimum
determinant was `0.999999762`; failed steps were zero.

GPU times were 33,513.402 and 33,583.710 ms (33,548.556 ms mean; 0.210%
repeat spread), 1.138x faster and 12.106% less GPU time than the repeated
five-Newton compact baseline. The state hash differs from the five-Newton
result, so this is a distinct accepted simulated trajectory rather than an
exact replay of that baseline. The override remains opt-in.

The exact invocation is:

```sh
./build/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-suture-passage-only --synthetic-skin \
  --synthetic-skin-compact-mesh --synthetic-skin-newton-iterations 1 \
  --velocity-iterations 8 --final-velocity-iterations 0
```

The repeated logs are
`/Users/n/numi-skin-suture-perf-100x/runs/compact-mesh-passage-8x-8i0f-newton1-r1-20261002/run.log`
and
`/Users/n/numi-skin-suture-perf-100x/runs/compact-mesh-passage-8x-8i0f-newton1-r2-20261002/run.log`;
their SHA-256 values are
`05288796d658078f58a48ce9f609b9a7fd73d6af7c8576ed44f2a2e93f5758dd` and
`3a8640e5ce6e8d1d32747062e10d29fa4ab1a5be2db644d01e07434f016faa62`.
Both use probe source SHA-256
`4c7f4cc24bf0ddd0bba61a7a4a0ebd1b79f85d89d88a70f52295fe76de2aa701` and
probe binary SHA-256
`6d451f99d5df8ef97d031a1b60d7e71af3eb1a3fa128d5b215cfe68afb52cbd0`.

A 16x/8/0 trial with the same one-Newton budget completed 96 groups in
32,056.402 ms of summed chunk GPU time, but ended with 3.564439 mm/s contact
residual and 19.968 mm/s terminal tip speed, outside both endpoint bounds.
It is rejected; 8x remains the fastest accepted compact cadence. Its log
SHA-256 is
`ca692998f48cb2ab5e0a39259f31fe0a7dfd0ef2d8df1064abb0dd51dbf68536`.

A one-pass field-smoother diagnostic passed twice with the same five-Newton
state hash and a 37,762.186 ms mean, only 1.011x faster than the compact
baseline. A one-column FGMRES runtime override passed once at 37,294.981 ms
(1.023x), with a different state hash. Combining those overrides with one
Newton produced a single 33,290.439 ms run, 0.8% faster than the repeated
one-Newton mean; that marginal, unreplicated difference is not treated as an
additional performance qualification.

The focused topology, contact, rollback, and synthetic-skin CTests passed
9/9 after the Newton-budget and solver-source changes. The retained CTest log SHA-256 is
`bd5036a6b7d268103e4ec52e2295ad8f1f6381932ab942a92f24798235a5bdc7`.
The one-Newton speedup is repeatable but remains far below the 100x objective.

Reproduce the compact candidate with:

```sh
./build/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-suture-passage-only --synthetic-skin \
  --synthetic-skin-compact-mesh \
  --velocity-iterations 8 --final-velocity-iterations 0
```

### Earlier 14 x 14 reduced-mesh draft and terminal residual gate — 2026-10-02

The first accepted `--synthetic-skin-reduced-far-field-mesh` layout changed
only the far-field transverse spacing relative to the compact mesh. It retained the
longitudinal incision endpoints at ±8 mm, end supports at ±11.5 mm, local
bite-spacing anchors, the full specimen, and all eight thickness layers. The
resulting 14 x 14 x 8 grid has 9,408 tetrahedra, 12.5% fewer than the compact
14 x 16 x 8 grid. Its measured local spacings are 0.1675, 0.1880625, and
0.1875 mm, each below the 0.200 mm contact-band limit.

Two runs on the Mac mini, using the same final source and binary, passed the
8x/8/0 passage budget with one Newton iteration and produced identical state
hash `0x1a37cc2b6298b0d9`. Both formed 12 linked channels, retained all 9,408
tetrahedra with zero removed mass, advanced the tip 1.855291 mm, reached
0.219291 mm distal clearance and 19.983936 mm/s terminal speed, and ended at
1.879475 mm/s contact residual with zero cone violation. Minimum determinant
was `0.999999762`; failed steps were zero. GPU times were 33,092.342 and
33,190.078 ms (33,141.210 ms mean; 0.295% spread). Against the repeated compact
one-Newton mean of 33,548.556 ms, this is 1.012x faster and 1.214% less GPU
time. The mesh creates a distinct state from the compact run. This historical
14 x 14 result is superseded by the 14 x 12 layout below and does not approach
the 100x target.

The final probe now checks its endpoint contact certificate against the
existing 2.0 mm/s velocity-residual and `1e-5` cone-violation limits. Before
this change, the passage loop explicitly disabled those endpoint bounds, so
low-budget trials could print `tissue_curved_through_wall_passage=ok` while
missing the performance-study residual limit. The gated 8/0 case passed; a
4/0 rerun exited with failure at a 3.242 mm/s terminal residual. This is an
acceptance-check correction, not a relaxed tolerance.

The command used for this historical run was:

```sh
./build/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-suture-passage-only --synthetic-skin \
  --synthetic-skin-reduced-far-field-mesh \
  --synthetic-skin-newton-iterations 1 \
  --velocity-iterations 8 --final-velocity-iterations 0
```

The final probe source SHA-256 is
`bc697d404585eb67076b877464cb34c0484b0dc2a2519b79b1015205c19beee6`; the
binary SHA-256 is
`043c2042564347fb350b4e50d4ad5d720e6f52453bbb028b455b22f9e38ff354`. The
passing logs are
`/Users/n/numi-skin-suture-perf-100x/runs/reduced-far-field-mesh-passage-8x-8i0f-newton1-terminal-gate-r12-20261002/run.log`
and
`/Users/n/numi-skin-suture-perf-100x/runs/reduced-far-field-mesh-passage-8x-8i0f-newton1-terminal-gate-r14-20261002/run.log`,
with SHA-256 values
`70bd1190e9969ab3cce8813699c4f50a8991ac990b3d334ad08f033b9c4ec107` and
`dbe957ff094d10806e73fcb3603b5bd6e22f954ed6f69b9c7059661b73ae4ce6`. The
4/0 gate-rejection log SHA-256 is
`b55a1c47d96820fd0bd29b5ebc5228b395119530f6866df874bdf9a76a483777`.

Two earlier grid drafts were rejected before GPU work: one exceeded contact
spacing at 0.251 mm (log SHA-256
`17b619f5b17f9128598225136393aad004405e15d6518c21d19ca0bf9711caa9`); the
next omitted the ±8 mm incision endpoints (log SHA-256
`0063de711b9c7018832531002cb4db05091dd3aaa87f073b5c03572e8cc9ed8d`). The
accepted layout retains those required nodes and coarsens only the transverse
far field.

Exploratory reduced-mesh contact budgets of 4/0, 5/0, 6/0, 7/0, and 4/4
completed their simulated horizons but ended at 3.242, 3.023, 2.483, 4.673,
and 7.780 mm/s, respectively. All exceed the 2.0 mm/s endpoint bound and are
rejected despite their lower GPU times. Their run-log SHA-256 values are
`1f37c7ffff351a3625262a937a9d5ac7de2990e8e57d808edea2ad9281da8166`,
`155e40cf5f5554a39bb295244413d14faf2b9b1fe237bac6395bc8ccad0f00a`,
`39ce6be80dfdb656220b605ad905ed1d054410daa5973dab2867cdedcb17469d`,
`a7f43edc2fd75a9cac0a65f25b6544f9713902e93ee61a85adf8ab4e5ad5babc`, and
`912022e59c448b30a63e8cebb60bd793ff7b5b3fde8f6ab3330159428da1d35c`. A
single one-pass smoother run retained the accepted state hash at 33,002.687
ms (run-log SHA-256
`e5f491a5cdd4f5591f1830fddeff3f0531c979d6b44d3b5ae245fe1c9865b126`), but was
not repeated and is not counted as an additional speedup. The
100x performance objective remains open.

### Metal System Trace of the earlier 14 x 14 passage — 2026-10-02

A diagnostic Metal System Trace was captured on the final-gated 14 x 14 binary
and completed that reduced-mesh passage with state hash
`0x1a37cc2b6298b0d9`. Profiling overhead raised its measured GPU time to
33,137.102 ms, so it is not used in the speed comparison. The exported GPU
intervals total 33,165.562 ms: eight coupled Matter/MetalWorld intervals use
33,129.263 ms (99.89%), while rigid-only work uses 32.735 ms and other GPU
intervals use 3.564 ms. These coupled labels span the persistent graph from
contact broadphase and island construction through the temporal-cone solve,
rod integration, and Matter publication.

The trace exposes command-buffer intervals but no per-dispatch durations or
kernel names. Its GPU execution-point table contains timestamps and submission
IDs only, and encoder-list durations measure encoding rather than GPU work.
The capture therefore locates the cost in the coupled graph but cannot
separate broadphase, contact assembly, solver, FEM, and rod costs. The saved
`.trace` package is 473 MB and took about 4.5 minutes to finalize; its files
are under
`/Users/n/numi-skin-suture-perf-100x/profiles/suture-passage-final-gate-20261002/`.
The run log SHA-256 is
`e117dd78abe6a6be72b2d6c725dbbccdf609d9be6d9de4619c0207be830a9229`; the GPU
interval export SHA-256 is
`1bcb89ad8cff05a94149f73ae493779fd47f7bebf05ef17975f563e6fbdecbec`. The
100x objective remains open because this profile does not resolve an
individual kernel bottleneck to target. This profile predates the 14 x 12 baseline and the local-preserving
14 x 10 candidate below.

### Previous 14 x 12 far-field passage — 2026-10-02

The reduced-far-field flag previously selected a 14 x 12 x 8 grid with
8,064 tetrahedra, 25% fewer than the compact 14 x 16 x 8 mesh. It is retained
here as the comparison baseline for the local-preserving 14 x 10 layout below. Incision endpoints,
end supports, bite-spacing anchors, specimen extent, and all eight thickness
layers remain. Local spacings are 0.1675, 0.1880625, and 0.1875 mm, all within
the 0.200 mm contact-band limit.

Two source- and binary-matched Mac mini runs passed the strict endpoint
certificate with one Newton iteration and an 8x/8/0 contact budget. They
produced the same state hash `0xa1914183f7c8a37d`, formed 12 linked channels,
retained all 8,064 tetrahedra with no removed tissue mass, advanced the tip
1.855291 mm, reached 0.219291 mm distal clearance and 19.983936 mm/s terminal
speed, and ended at 1.879475 mm/s residual with zero cone violation. Minimum
determinant was `0.999997377`; failed steps were zero. GPU times were
32,590.066 and 32,504.370 ms (32,547.218 ms mean; 0.263% spread), 1.031x over
the 33,548.556 ms compact one-Newton mean. This is a repeatable but small
gain; the 100x target remains open.

Reproduce on the Mac mini with:

```sh
./build/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-suture-passage-only --synthetic-skin \
  --synthetic-skin-reduced-far-field-mesh \
  --synthetic-skin-newton-iterations 1 \
  --velocity-iterations 8 --final-velocity-iterations 0
```

The passing logs are
`/Users/n/numi-skin-suture-perf-100x/runs/reduced-far-field-y12-passage-8x-8i0f-newton1-r15-20261002/run.log`
and
`/Users/n/numi-skin-suture-perf-100x/runs/reduced-far-field-y12-passage-8x-8i0f-newton1-r16-20261002/run.log`,
with SHA-256 values
`9be51aba577c4c6b5bc514bdbc645dd59bd80dfcc23e27f9aef381948292049d` and
`93a54a9e63c6a26bbb11975cbbd42c861e4cf79d2b8405011916760c427f2d14`.
The current probe source and rebuilt Mac mini binary SHA-256 values are
`83906be7b0a14f11892015dc76d825af2fef59d324b29ca3ceec747afc8f00aa` and
`6fd38b453b5464d2ed4a1a9366cec234e23a2821d17db3e12986d4eda77a3bc0`.

A previous 14 x 10 x 8 node map passed geometric passage once at 32,215.217
ms but is rejected: it shifted entry impulse to `1.05196e-5 N s` (about 9.5x)
and raised minimum contact-node mass from `3.23e-9 kg` to `3.07e-8 kg`. It
coarsened the local bite map and is distinct from the local-preserving 14 x 10
layout qualified below. Its run log SHA-256 is
`81ed6f33d6ba5f8845a12504e8e3f83ec245b8578fb28efc59c5fd6a94ae1a2a`; source
and binary SHA-256 values were
`c4e99a9cc4025eeb3cf74ece18127002456797e9b76d2946a5863636afa591e5` and
`c0189ae5dfbfde9b85db362d3bcd75a0f91852506cba5ee020f5e3e1e0b99606`.

### Wave32 normal-row Cauchy bound — rejected (2026-10-02)

The exact absolute-normal row sum in `mrWorldWave32SolvePacket` scans every
source contact for every target row. A candidate replaced that scan with a
Cauchy–Schwarz upper bound formed from the square roots of rigid normal
diagonals and one cohort reduction per island. The baseline and candidate
were built from the same isolated Mac mini source snapshot at HEAD
`8ce44aca14f850dc05259c93d5d723e255ae3237`, including the same dirty
application and Matter sources; only `MetalWorldContact.metal` differed.
The baseline shader source SHA-256 is
`9dc66e39aa8f98c67d6a0e1e2a4503aaf42fcd1dcfce569480e7e78d81cfb25c`, and
the candidate is `33fb5458d22ce5092fdae1cdd686c6d4ebd47fdb8e2f41c9383218bbefa9c119`.
The corresponding `MetalRobo.metallib` hashes are
`b27c17089eebc0af865ba4d47198d84e6a609b284202a674480bbf6ba175d997` and
`c73acb779b6e2e5b1168e5f48ded8fde55a0b226616c6d152a23b1c418600258`.

Three matched 16x/one-Newton, 8/0 cadence probes gave median GPU times of
376.224 ms for control and 375.569 ms for the candidate, a 0.17% difference.
Each retained 27,648 tetrahedra, one puncture channel, 143 MetalWorld
constraints, the same 0.00186694 m/s normal residual, and zero failed steps.
This short probe had zero active tissue contacts. The six logs are under
`/Users/n/numi-skin-suture-perf-100x/runs/goal-wave32-{control,candidate}[-r2|-r3]-20261002/`;
their SHA-256 values, in control then candidate order, are
`bd8c8ab428da2be3c110c51c696ef7fe6c2cbbdb30063ee795fa3dc10d0d4463`,
`affd921fa2043be596f3a627e34ca188fd01159fc3c8d792573ef7eb2449ac2f`,
`c9c5149cc45e6a754379264f3e21c686a30b8afca7c1fda622b420e82aefc745`,
`c2f3c81ab3e8f4ca5baa4d408fc2978cdc373052a9f98bfbe498bd5ccb0945dc`,
`501904c1964a51555366f13bb68506a2a202076a3398676b08c9019233e542ee`, and
`b92d0e77f6eccc7752e138990b2b53882e629cc0ecde95b71db3d5926170c08c`.

One full 14 x 12 x 8, 8x/8/0 passage run per build also passed with the same
state hash `0xa1914183f7c8a37d`, all 8,064 tetrahedra retained, 12 linked
channels, 1.879475 mm/s terminal contact residual, and zero failed steps.
Control took 32,590.250 ms GPU and candidate 32,528.102 ms, a 0.19%
difference below the 0.263% spread of the prior source-matched repeat pair.
The control and candidate passage log SHA-256 values are
`6c72fbeb362210685897374be698dfb621a8b271adaf26d7225d52879cdce556` and
`390c4b741a1a5edb294fc643d9282edc1235e3c44bbe379a30747fc7d1901d7d`.
Because the candidate did not produce a repeatable throughput gain, its
shader change was reverted. The full-stitch 100x objective remains open.


### Local-preserving 14 x 10 far-field passage — 2026-10-02

The revised `--synthetic-skin-reduced-far-field-mesh` layout keeps the full
14-cell longitudinal map, all incision and end-support nodes, the full eight
thickness layers, and every transverse node within the two bite neighborhoods.
It removes only the distant transverse nodes at +/-7.5 mm and joins the
remaining far field to the unchanged +/-3.75 mm local nodes. This yields a
14 x 10 x 8 mesh with 6,720 tetrahedra (16.7% fewer than the prior 14 x 12 x
8; 37.5% fewer than the compact 14 x 16 x 8). The local spacings remain
0.1675, 0.1880625, and 0.1875 mm. The one-step entry diagnostic matched the
14 x 12 entry impulse at `1.1079010846515303e-6 N s`, the minimum contact-node
mass at `3.2320513586923359e-9 kg`, and the same accepted entry state; all
6,720 tetrahedra remained and removed tissue mass was zero. The entry log SHA-256
is `f7077d11d77d3cc83dadae288ac83b69068b5f4179d45e52a39d8ff8ce88a730`.

Two full 8x/8/0, one-Newton passages passed with the same state hash
`0x8623d3dfc8df0407`. Both retained all 6,720 tetrahedra and zero tissue mass,
formed 12 linked channels, advanced the tip 1.855291 mm, reached 0.219291 mm
distal clearance and 19.983936 mm/s terminal speed, and ended at 1.879475 mm/s
contact residual with zero cone violation. Minimum determinant was
`0.999997377`; maximum matter residual was `8.32791685e-8`; failed steps were
zero. GPU times were 32,175.837 and 32,208.171 ms (32,192.004 ms mean; 0.100%
spread), 1.011x over the prior 14 x 12 mean of 32,547.218 ms. The state hash
differs from 14 x 12 because the meshes differ. This is a repeatable but small
gain; the 100x target remains open.

Reproduce on the Mac mini with:

```sh
./build/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-suture-passage-only --synthetic-skin \
  --synthetic-skin-reduced-far-field-mesh \
  --synthetic-skin-newton-iterations 1 \
  --velocity-iterations 8 --final-velocity-iterations 0
```

The passing logs are
`/Users/n/numi-skin-suture-perf-100x/runs/reduced-far-field-y10-local-bite-map-passage-r2-20261002/run.log`
and
`/Users/n/numi-skin-suture-perf-100x/runs/reduced-far-field-y10-local-bite-map-passage-r3-20261002/run.log`,
with SHA-256 values
`06a316958bf65a2c7c93b68d2e039e7049baaccd84be8af676ad8dad580840c7` and
`9a11875efc7e36cbde84bb62dc08a94e160083d0cfcd1bd009bcca73d0f689be`. The entry
log, source, runtime, rebuilt probe binary, and Matter metallib SHA-256 values
are `f7077d11d77d3cc83dadae288ac83b69068b5f4179d45e52a39d8ff8ce88a730`,
`4ad9dc85d21dbeb815fdcfe8e8c6c265a6e7040b1f554bb14caab8d084fd48c2`,
`b94b6da18e8b705b1cd0e96391263a28fd5d83acf5c39539722cccd2faccd23d`,
`80064a5a6fe1661140fc1e33fd1dfe65b684f9916889d83c3926639c4ce95d2d`, and
`859e81cd0fdeb4820926f285d27c48960a96d1a73ed9f4bc796f3847ff1ce633`.

### Other measured dispatch and cadence trials — 2026-10-02

A runtime guard omits sparse MPM setup, scans, and indirect kernels when the
compiled world has zero MPM particles, blocks, and active-node capacity. The
full passage fixture reported all three capacities as zero. The MPM freefall
CTest passed in 158.39 s, preserving the MPM-enabled path. On the 14 x 12
passage, the guarded candidate measured 32,660.668 ms versus 32,595.741 ms for
control (+0.199%); this is within the prior 0.263% run spread and is not a
demonstrated speedup. Its Matter metallib matched control at
`859e81cd0fdeb4820926f285d27c48960a96d1a73ed9f4bc796f3847ff1ce633`.

A four-substep local-skin cadence doubled the passage to 372 groups from 188
while leaving total DER substeps nearly unchanged (1,488 versus 1,504). The
GPU chunks summed to 35,188.280 ms and the endpoint was rejected: terminal
contact residual was 8.103992 mm/s (limit 2.0 mm/s) and distal clearance was
0.199994 mm (limit 0.200 mm). A 12 x 10 longitudinal/transverse mesh trial was
also rejected before GPU work because it removed a required incision end
support. Neither trial is included in the performance result above.

### Canonical neo-Hookean stress fast path — small gain (2026-10-02)

For the compiler-marked exact canonical neo-Hookean energy with its canonical
parameter layout, no internal material state, and no viscous stress program,
the FEM stress evaluator now computes the closed-form first Piola stress
directly. Other material programs retain the existing evaluator. This removes
nine generated scalar-program evaluations per element only for the exact
guarded material case.

A fresh 8x/8/0, one-Newton control passage took 32,210.914 ms GPU. Two
source-matched candidate repeats took 32,080.680 and 32,062.371 ms (32,071.525
ms mean, 0.43% faster than control). After syncing the final probe source to
the Mac mini and rebuilding, a third full passage took 32,000.441 ms GPU. All
three candidate runs produced state hash 0xd178249698f5b1e4. The rebuilt-source
run passed the full endpoint gates: 188 groups and 1,505 DER substeps; all
6,720 tetrahedra retained; 12 channels and 11 links; zero removed mass;
1.855291 mm tip advance; 0.219291 mm distal clearance; 19.983936 mm/s terminal
speed; 1.879475 mm/s terminal contact residual; zero cone violation; minimum
determinant 0.999997377; maximum Matter residual 8.32470874e-8; and zero failed
steps. The third run is a rebuild confirmation, not an additional matched
control pair.

Reducing the synthetic-skin field smoother from three passes to one also
passed with the same state hash and endpoint metrics, at 32,040.779 ms GPU.
That single run does not establish a speedup. The 100x target remains unmet.

Control and two-repeat fast-path log SHA-256 values are
9fa7f0585ff6a61a3e1f659ece32a63c37e35f92d54ebdd802e5a879de894ae9,
ea5f15ac42521c30b4e295ac248acb2766b5851952e39f648c36a63d7d258b75, and
b50eb016c4b2c53943f09f9a24fb76fcdbfd2785a07ce672a3282fedfe151896. The
rebuilt-source passage and one-pass smoother logs hash to
0923a1672b35a763de72f36f84e91fb33778d6bc4ce50c4fccb8e5d015f519fe and
50756fc2b7be9f9a08b6b9fa86f0aae11e1f048f8ba91e399d39990b9cc7c440.
Their run directories are under
/Users/n/numi-skin-suture-perf-100x/runs/ on the Mac mini. The exact current
probe source, stress shader, probe binary, and Matter metallib SHA-256 values
are 4ad9dc85d21dbeb815fdcfe8e8c6c265a6e7040b1f554bb14caab8d084fd48c2,
07e5ccfc371fb3239d720f848ac4410a16003b2e517cecf6b0b5ae7874ac6e36,
80064a5a6fe1661140fc1e33fd1dfe65b684f9916889d83c3926639c4ce95d2d, and
4b24ba6845caab23d3b51cbc5e16b061e532f0ad0c5a622588f48b16412d2762.

A four-layer through-thickness trial was rejected by the existing entry-mesh
contact-resolution gate before GPU submission: 0.375 mm layer spacing exceeds
the 0.200 mm strand-contact reach. The qualified eight-layer resolution is
therefore the current minimum for this contact map; no full passage was spent
on the rejected mesh. The entry log SHA-256 is
3f631d76c3301578eef919419252cd7e27bb93f4057cb0a9f3436154c53a01bb.

The Mac mini exposes a timestamp counter set but the Metal driver rejects
dispatch-boundary counter sampling through the tested legacy compute/blit
path. A prior temporary stage-boundary pass did collect per-encoder timestamp
ticks for the entry transaction, which exposed topology certification,
conservation, mass rebuild, and scatter costs; that instrumentation is not in
the current executable. The passage System Trace provides command-buffer
intervals, not individual dispatch durations. Treat the old entry breakdown as
diagnostic evidence for that entry workload, not as an attribution of the
through-wall passage. Further performance trials should use a retained,
source-matched stage-boundary profile instead of broad parameter sweeps.

### Fresh 16x one-Newton post-entry cadence control — 2026-10-03

Three source- and binary-matched Mac mini runs used
`--tissue-suture-cadence-16x-one-newton-only --synthetic-skin
--velocity-iterations 8 --final-velocity-iterations 0`. GPU times were
410.074917, 410.010625, and 409.968375 ms (410.017972 ms mean, 0.026% spread).
Against the original 1x/3,375.995 ms reference, this is 131.7x simulated-time
throughput for this post-entry cadence state, below the 540.159 ms per-step
100x threshold by 130.141 ms.

All three runs reported the same physical values: a 16x grouped timestep of
1 ms; 19.965429 um needle advance against 20.000001 um expected; one active
puncture channel; all 46,080 tetrahedra retained; zero removed tissue mass;
minimum determinant 0.999997377; Matter residual 2.46819468e-6; nine Matter
FGMRES iterations; 143 rigid constraints; eight MetalWorld solver iterations;
and 1.86694006 mm/s terminal normal residual, under the 2.0 mm/s bound. The
cone violation was 5.96e-8 and failed steps were zero. Matter reported zero
active tissue contacts, so this is a post-entry cadence result, not a
contact-loaded passage or full-stitch performance qualification. The full
through-wall passage remains about 32 seconds; the 100x end-to-end goal remains
open.

The logs are
`/Users/n/numi-skin-suture-perf-100x/runs/goal-fgmres-readfusion-control-8i0f-20261003/run-{1,2,3}.log`
on the Mac mini. Their SHA-256 values are
`b2d09d02c8ed604e127586e69cb4797cca16fa625f12e29d0322957beed57cfe`,
`2b45d0f9083166aa85abc2c77e5e5be8a0f2c4439f7cba5b641eb3b59970d179`, and
`1aac4b10b78f578bdbaa56913b3ef8e550b6779ed1b93f9997cd5b203eb6ee3a`. The
probe source, FGMRES source, canonical stress source, executable, and Matter
metallib hashes are `4ad9dc85d21dbeb815fdcfe8e8c6c265a6e7040b1f554bb14caab8d084fd48c2`,
`f26bf768162d3bf439159c13b8d3e674e67988a52e707be08c7def619f73fd32`,
`07e5ccfc371fb3239d720f848ac4410a16003b2e517cecf6b0b5ae7874ac6e36`,
`80064a5a6fe1661140fc1e33fd1dfe65b684f9916889d83c3926639c4ce95d2d`, and
`4b24ba6845caab23d3b51cbc5e16b061e532f0ad0c5a622588f48b16412d2762`.

### Generalized rod-attachment prefix cache (rejected) — 2026-10-03

An exploratory shader trial replaced repeated scans of earlier authored
constraints with translation and twist attachment prefix counters in the
ordered generalized solve. It compiled and completed the full 14 x 10 x 8
reduced-far-field passage without changing the endpoint: 188 groups, 1,505 DER
substeps, 6,720 active tetrahedra, 12 connected puncture channels, 0.219291 mm
distal clearance, 1.879475 mm/s terminal contact residual, and zero failed
steps. The state hash remained `0xd178249698f5b1e4`. Matter reported zero
active tissue contacts and only `4.65661287e-10 m` maximum tissue displacement,
so this is passage timing rather than a load-bearing deformable-skin benchmark.

The matched control took 75,437.4895 and 75,451.1827 ms GPU (75,444.3361 ms
mean); the prefix candidate took 73,056.7417 and 80,240.4299 ms (76,648.5858
ms mean, 1.60% slower). The candidate's spread is too large to support a
small performance claim, so the change was rejected. This pair was
exploratory, not preregistered. The probe and shared library were identical
across arms; the control and candidate shader sources hash to
`8e41245251bc812a7d7af21565262556f93d7a841b354e5a21801c3f3b95acd6` and
`30fa1607ee8287bd92249ab5736c0db07e9b6ed3806fe098fcb53b96e79e82a3`.
The probe hash was
`e9b0522d45d4da83613d4a31a3924aef7a91e242e7b4bbe57c04561355033c6a` and the
shared-library hash was
`5a297c6f580b4f0a41a4079de18fb1500fb6c637de76e4134374f654483ea2e6`.

On the Mac mini, the current control pair is more than twice as slow as the
previous 31,674 ms matched runs, although the new control repeats agree within
0.02%. The host reported no thermal or performance warning, but its GPU was
already at 14% utilization in an idle snapshot. Treat cross-session timing as
confounded until a source-bound profile explains the shift. The exact control
source and metallib were restored in the isolated run tree; the experimental
source, rebuilt metallib, four full-passage logs, and build receipts remain in
`/Users/n/numi-skin-suture-perf-100x/runs/goal-wave32-distributed64-20261003/`.
The prefix metallib reconstructed from its retained source hashes to
`b857990be7abf1c427e215f41b12893640caf3d608689276f81b5ce914340984`; the
restored control metallib hashes to
`fa9c48a4cb93f6846735aaf27913d19a0188cdf3f7bdfca9207a4bc62470877f`.
Neither this solver micro-change nor the 100x whole-stitch target is accepted
by these results.

### Canonical stress fast path in an unused FEM helper — rejected

A follow-up tried the canonical neo-Hookean stress/tangent fast path in
`fusedFEMOperatorAtNode`. Source inspection found no call sites for that helper.
Although the intermediate AIR files differed, control and candidate linked to
the same `NumiMatter.metallib` SHA-256
`4b24ba6845caab23d3b51cbc5e16b061e532f0ad0c5a622588f48b16412d2762`, so the
candidate did not change the executable GPU program. I reverted the source
change and count this as no performance result. The source and build artifacts
are retained under
`/Users/n/numi-skin-suture-perf-100x/runs/goal-fem-force-fastpath-20261003/`.

### FGMRES cap on first synthetic-skin entry — bounded diagnostic

The existing `--synthetic-skin-fgmres-iterations N` override now also applies
to `--tissue-suture-entry-only --synthetic-skin`; it remains opt-in, and the
default FGMRES budget is unchanged. On the Mac mini, five counterbalanced
control/candidate pairs compared the default ten-iteration solve with a
one-iteration cap on the same 30 x 32 x 8 synthetic-skin first-entry event.
The order alternated between control-then-candidate and candidate-then-control.
Mean GPU time fell from 713.785 ms to 433.522 ms, a 1.65x event speedup and
39.3% lower mean time. The candidate was faster in all five pairs; median
paired speedup was 1.61x.

All five candidate runs formed the same three active needle-tip contacts and
matched the control's normal and axial contact impulse fields exactly. Each
retained the puncture channel and all 46,080 tetrahedra, removed no tissue
mass, and had zero failed steps. The candidate used one FGMRES iteration,
reported a positive minimum determinant of `0.999999821` and Matter residual
`2.84217117e-6`. Its accepted-state hash differs from the default result
(`0x958ea835321e8fdd` versus `0xa3ddb432216e32de`); maximum sampled tissue
displacement changed from `0.931 nm` to zero. This is a bounded first-contact
step with sub-nanometer tissue motion, not a completed stitch or a
load-bearing deformable-performance qualification.

The focused `metalrobo.surgical.live_suture_tissue_entry` CTest passed. The
counterbalanced run logs, exact source/build hashes, and summary are retained
under
`/Users/n/numi-skin-suture-perf-100x/runs/goal-entry-fgmres-cap-20261003/`.
The full through-wall passage and end-to-end suturing task remain far from
100x; no production solver default changed.

### Lower temporal-cone iteration budgets — rejected — 2026-10-03

The retained full-passage profile localized about 9.5 seconds of the earlier
32-second run to 1,505 Wave32 temporal-cone stages (8 iterations per stage,
about 6.37 ms median). Matter post-commit work was about 73 ms across the
passage, so reducing Matter solve cadence is not a high-impact target. The
Metal System Trace still groups the rest of the persistent world graph into a
coarse interval and does not identify the remaining per-kernel cost.

On the Mac mini, the 14 x 10 x 8 reduced-far-field probe was rerun from the
same control executable and linked libraries used by the retained 8-iteration
control (`probe` SHA-256
`e9b0522d45d4da83613d4a31a3924aef7a91e242e7b4bbe57c04561355033c6a`,
`MetalWorldContact.metal` SHA-256
`8e41245251bc812a7d7af21565262556f93d7a841b354e5a21801c3f3b95acd6`).
That control completed in 75.45 s GPU time. One iteration completed the same
188 passage groups in 19.92 s wall time; four iterations took 25.57 s. Both
reached the geometric through-wall tract checks, retained all 6,720
tetrahedra, removed no tissue mass, and had zero active tissue contacts, but
both were rejected by the terminal solver gates. The one-iteration terminal
velocity residual was `0.161066681 m/s`; four iterations produced
`0.00324214832 m/s`, above the `0.002 m/s` limit. Both ended at
`0.019968 m/s` tangent speed, 32 um/s below the 20 mm/s target and outside its
20 um/s tolerance.

Two hybrid budgets, four base iterations plus four or eight extra iterations
on each transaction's final substep, also failed: terminal residuals were
`0.00777965831` and `0.00657229219 m/s`, with the same terminal-speed miss.
The runs took 26.38 s and 27.11 s wall time. These faster measurements are
diagnostic only: the cases fail physical acceptance, do not establish a
counterbalanced speedup, and show effectively zero skin deformation. No
solver default or production behavior changed.

The four logs and source/library manifests are retained under
`/Users/n/numi-skin-suture-perf-100x/runs/goal-wave32-velocity{1,4}-20261003/`
and `.../goal-wave32-velocity4-final{4,8}-20261003/`. The next high-value
work is a source-bound profile inside the remaining persistent world graph,
then a convergence-preserving reduction in dependent DER substeps; further
fixed iteration-budget reductions do not meet the existing endpoint gates.

### Contact-kernel stage profile and parallel rod publication trial — 2026-10-03

An opt-in Metal timestamp probe sampled 2,085 named contact-kernel encoders
across the first passage substeps, covering 38 kernel labels. The five largest
sampled medians were 1.899 ms for the pre-contact typed scalar IR sweep,
1.509 ms for transactional rod publication, 1.396 ms for parallel authored
rod responses, 1.184 ms for 32-contact tile construction, and 1.043 ms for the
canonical generalized ConstraintIR solve. The probe run used four velocity
iterations and took 63 seconds with instrumentation enabled; it failed the
existing solver gates and is profile data only, not a performance result.

The rod-publication kernel visibly copied every rod node and edge in a single
GPU thread. An experimental rewrite dispatched one thread per node or edge
while preserving the environment's transactional commit/rollback selection.
The focused `metalrobo.surgical.live_suture_tissue_entry` CTest passed. A
source-isolated full-passage control/candidate pair both passed all gates and
produced the same state hash, `0xd178249698f5b1e4`. Control GPU time was
31,652.475 ms; candidate time was 31,631.908 ms, a 0.065% difference that
does not establish an improvement. The case still had zero active tissue
contacts and only `4.65661287e-10 m` maximum skin displacement. I reverted
the kernel rewrite and retained no production change.

The first attempted pair was discarded because the executable's absolute
runpath loaded the candidate shared library for both arms. The accepted pair
used per-arm `DYLD_LIBRARY_PATH` directories and matching metallibs; loader
output confirmed each selected its own `libmetalrobo.dylib`. Source, binary,
library, metallib, CTest, and passage logs are retained under
`/Users/n/numi-skin-suture-perf-100x/runs/goal-rod-commit-parallel-20261003/`.
The profile artifacts are under
`/Users/n/numi-skin-suture-perf-100x/runs/goal-contact-kernel-stage-profile-20261003/`.
The 100x whole-stitch target remains open; the remaining major kernels need a
convergence-preserving algorithmic reduction, and the present passage remains
a free-needle diagnostic rather than load-bearing deformable performance.

### Wave32 articulated Jacobian projection reuse — diagnostic only — 2026-10-03

The Wave32 block setup now reuses each target row's articulated Jacobian
projection while forming the effective 3 x 3 response. A first pair of
retained runs under `goal-wave32-jacobian-projection-cache-20261003` is
excluded: its source snapshot had the candidate shader, but the saved build's
`MetalRobo.metallib` was the pre-change image. A fresh control and candidate
build were made from the same source tree, differing only in
`MetalWorldContact.metal` (control SHA-256
`0623eb63aedead318cd13a69119b3763305580988e17eb08684d0b843bc22c50`,
candidate SHA-256
`8e41245251bc812a7d7af21565262556f93d7a841b354e5a21801c3f3b95acd6`).
The resulting MetalRobo metallibs differ (`90cd7e9d...` control,
`fa9c48a4...` candidate), while both arms use the same probe binary and
NumiMatter metallib. Loader output confirmed each arm loaded its own shared
library and shader image. The full control/candidate library hashes are
`65e40cbe83f92a2719b9d7054f0cc3351ec91d9cb5f150b59879c20c574fea74` and
`0701c6b7eb93eda3aabdc49f579bbfa60b1284e42ebbc93654fdc2685598cbc8`; the
metallib hashes are
`90cd7e9d90ac50620c618c118ea09a477e3819641b1ce51f51a94820f1857b99` and
`fa9c48a4cb93f6846735aaf27913d19a0188cdf3f7bdfca9207a4bc62470877f`.
Both runs used probe SHA-256
`dde2d504125e5291905493504d6289ca16585bdc9d62219a49d4c3cd67a62ebb` and
the same NumiMatter metallib SHA-256
`4b24ba6845caab23d3b51cbc5e16b061e532f0ad0c5a622588f48b16412d2762`.

Two counterbalanced pairs ran the same reduced-far-field synthetic-skin
passage at eight velocity iterations and zero final iterations. Both arms
passed all 188 groups with the same state hash
`0xa44b6385e7e6aae3`, 1,505 DER substeps, a 1.879475 mm/s terminal residual,
and zero failed steps. GPU passage time was 34,724.858 ms control versus
34,492.471 ms candidate in the first pair, then 84,357.472 ms control versus
83,104.976 ms candidate in the reversed pair: nominal candidate reductions
of 0.67% and 1.48% (1.25% on the paired means). Wall time did not consistently
improve (35.15 s versus 35.52 s, then 85.13 s versus 83.51 s), and absolute
passage time varied 2.4x between pairs. This does not establish a dependable
speedup.

Both runs still had zero active tissue contacts and only `1.16415322e-10 m`
maximum tissue displacement, so this remains a free-needle diagnostic. The
shader change is retained only as an experimental candidate; no 100x claim,
load-bearing skin result, or production qualification follows from it. The
32/16 run in `control-r1.log` is also excluded because it omitted the 8/0
iteration overrides and failed the terminal residual gate. Source, binaries,
matching metallibs, build logs, and all run logs are retained under
`/Users/n/numi-skin-suture-perf-100x/runs/goal-wave32-jacobian-projection-paired-20261003/`.

### Runtime rod-attachment prefix metadata — rejected — 2026-10-03

The scalar generalized solve repeatedly scanned immutable authored blocks to
find rod-response cache slots. A candidate counted translation and twist
attachment prefixes once while seeding the per-step runtime IR copy, then read
those slots in the ordered solve. Canonical ConstraintIR, its zeroed reserved
words, and the solver order were unchanged. The source change was confined to
`MetalWorldContact.metal` (control SHA-256
`8e41245251bc812a7d7af21565262556f93d7a841b354e5a21801c3f3b95acd6`,
candidate SHA-256
`c0d60c45d894612353890518df108e457e8e8919d9e7504443e91dccf6600c07`).
Fresh control and candidate metallibs differed, and loader output confirmed
the candidate arm loaded its own library.

One candidate run at eight velocity iterations and zero final iterations
passed all 188 through-wall groups and matched the prior control state hash
`0xa44b6385e7e6aae3`, with zero failed steps and 1.879475 mm/s terminal
residual. Its GPU time was 34,447.615 ms and wall time 36.01 s. The nearest
same-source control run took 34,492.471 ms GPU and 35.52 s wall: a 0.13% GPU
difference with slower wall time. The other exact control run took
83,104.976 ms GPU, showing the same 2.4x run-to-run spread seen in the prior
pair. This is not a dependable speedup. I removed the prefix-metadata shader
change and restored the pre-experiment source; the candidate remains in the
run artifacts for reference.

The passage still had zero active tissue contacts and only
`1.16415322e-10 m` maximum tissue displacement. It does not qualify
load-bearing deformable performance or move the 100x whole-stitch goal. The
source, distinct builds, loaded libraries, passage log, and checksums are under
`/Users/n/numi-skin-suture-perf-100x/runs/goal-generalized-attachment-prefix-metadata-20261003/`.

### Residual-guarded generalized PGS early exit — measured candidate — 2026-10-03

The ordered generalized ConstraintIR solve stops after a complete sweep only
when its measured maximum row residual is at or below 1 mm/s. The configured
velocity-iteration budget remains the ceiling; sweeps continue while the
residual exceeds this stricter cap. This preserves the ordered updates and
residual gate while avoiding unused sweeps after convergence.

On the Mac mini, two counterbalanced 8/0 reduced-far-field synthetic-skin
passages ran from the same transferred source snapshot. The control took
34,626.715 and 34,533.417 ms GPU; the candidate took 32,039.591 and
32,113.171 ms. The paired mean fell from 34,580.066 to 32,076.381 ms, a
repeatable 7.24% reduction (1.078x). Both control runs passed all 188 groups
and produced state hash `0xa44b6385e7e6aae3`; both candidate runs passed and
produced `0x66f27c3575af300c`. Endpoint tract geometry, 12 connected channels,
all 6,720 tetrahedra, zero removed mass, tip advance, speed, and terminal
residual remained within the same acceptance gates. The differing state hashes
mean this is not a bitwise solver substitution.

The full passage had zero active tissue contacts and only `1.164e-10 m`
maximum tissue displacement, so its timing is a free-needle passage diagnostic,
not load-bearing deformable-performance evidence. The focused
`metalrobo.surgical.live_suture_tissue_entry` CTest passed for both builds. An
additional matched synthetic-skin entry run retained the same three tip
contacts and normal/axial impulses, all 46,080 tetrahedra, zero removed mass,
the same Matter residual and minimum determinant, and sub-nanometer tissue
motion; its accepted state hash differed. Its single GPU timing was 581.893 ms
control versus 550.150 ms candidate, so the 5.45% entry reduction is diagnostic
only. Neither result establishes a 100x whole-stitch speedup or a load-bearing
stitch.

The control and candidate sources differ only in `MetalWorldContact.metal`,
hashes `8e41245251bc812a7d7af21565262556f93d7a841b354e5a21801c3f3b95acd6`
and `34d892b57bcc4ab967a9ef106b66b839ae0ce4dbcea2a4c6d7e3eb2281acf4f2`.
The transferred probe source SHA-256 is
`2ad8bf3c80e5ff0a0457bc6f570cf1e147470196bc5b84cac29ad23aff3e6d79`; the
control/candidate probe binary hashes are
`2025835e5b715efa78b444d735104e9bea8377e323ede1e45b7cc7325c41736c` and
`06e8d49cc352b001f7c3fab0baea66ccfb37cd0543fdf6f74915356c32cdd63e`.
Runs, builds, and CTest logs are retained under
`/Users/n/numi-skin-suture-perf-100x/runs/goal-generalized-residual-early-exit-20261003/`.
The 100x whole-task target remains open.

The candidate load check confirmed dyld selected the candidate build's own
`libmetalrobo.dylib`. Control/candidate library SHA-256 values are
`4264f0637cc330df129a801792d812de9e57309b41140928167f4a0a5a08e7dc` and
`e91c306a8d3b4273a0ae5323446fc8f7eeb1dbe8630fe01f65e404e9efd2fe12`; their
`MetalRobo.metallib` hashes are
`fa9c48a4cb93f6846735aaf27913d19a0188cdf3f7bdfca9207a4bc62470877f` and
`cf283a89ff984b96f8d33b4cc6d78d3dfc8c3a3e3a2812f49ed1a17f5df9e7c0`. Both
arms use the same `NumiMatter.metallib`, SHA-256
`4b24ba6845caab23d3b51cbc5e16b061e532f0ad0c5a622588f48b16412d2762`.
The four passage log hashes are `bb46a415e5f2d3ce0315f28a89aa97fdadd1d4aa781616ddc5cd89f4f9bc6cf8`,
`4a60a558092b48727ac8bcc42388cc5885e9374d9ab733df301ff9f7cafc71d4`,
`cb8c23cbeff3b9c6a5c05472f5bccbb2d3df70b3d042fe9adc763c232ffa22b0`, and
`772599ac67b6d534296c48714fadfc1217609fbdb07fc71ae6cd4805b1f53408` in
control-1, control-2, candidate-1, candidate-2 order.

### Bounded cadence escalation screen — rejected above 8x — 2026-10-03

I extended the external coupled-cadence setter and snapshot validation to
accept bounded integer multipliers, then screened 9x and 12x passage grouping
against the passing 8x configuration with the same reduced-far-field mesh,
one Matter Newton iteration, eight velocity iterations, and zero final
iterations. The focused `matter.runtime.suture_proxy_window` test passed in
both isolated builds. The setter and snapshot round-trip accept 3x, but the
full passage acceptance gates remain authoritative.

The isolated 8x control passed 188 groups and retained all 6,720 tetrahedra,
12 connected channels, zero removed tissue mass, and zero failed steps. It
still had zero active tissue contacts and only `4.66e-10 m` maximum tissue
displacement. One 9x screen completed 168 groups, 12 channels/11 links, and
retained the tissue mass, but failed the terminal contact-velocity bound at
`0.002651 m/s` against the existing `0.002 m/s` limit. Its completed chunk
times summed to 70,806 ms versus 70,979 ms for the 8x control, only a 0.24%
single-run difference before accounting for the failed gate. A 12x screen
completed 127 groups and also retained the tract and tissue, but its terminal
contact-velocity residual was `0.025280 m/s`, over 12 times the same limit.
Its completed chunk times summed to 64,014 ms, but the failed physical gate
rejects that result. The default remains 8x; cadence coarsening has not shown
a qualified end-to-end gain.

The first 9x attempt was stopped before simulation because the probe compared
two independently rounded float timesteps for exact equality. A local
arithmetic check showed a one-ULP difference (`0x1.26e97ap-11` versus
`0x1.26e978p-11`, relative `1.03e-7`). The check now permits at most one
adjacent float while still requiring exact multiplier/divisor identity. The
rerun passed this configuration check and reached the existing physical
gate; no physics threshold changed.

The current 8x control took 70,979 ms GPU time, while an earlier 8x candidate
run took 32,040 ms with the same `MetalRobo.metallib` hash. This 2.2x
cross-session difference makes absolute timing comparisons unreliable; the
9x screen also showed essentially no same-session improvement. Use matched
same-session runs for future performance claims. Raw logs, source/build
snapshots, loaded-library evidence, and preregistration plans are retained
under
`/Users/n/numi-skin-suture-perf-100x/runs/goal-integer-cadence-coarsening-20261003/`.
The 100x whole-task target remains open.

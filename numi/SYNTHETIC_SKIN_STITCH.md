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

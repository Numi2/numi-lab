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
sweep could enter the skin contact band. The current executable SHA-256 is
`cc09c970d2aea6387b45c324a49d7aa5ffb0c852e8948ce078e36a402096edf5`.
The next solver path must use full contact authority near the skin; none of
these free-space results proves a robot-driven puncture.

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

Open work: physically approach and pick up the needle with the robot, drive
it through the first skin lip with measured tissue reaction, qualify thread
pull-through with this 1.5 mm wall, then the second lip bite, knot tightening,
gripper release, and measured unloaded wound-gap retention. Existing
jejunal checkpoints must not be reused as skin evidence; generate fresh
skin checkpoints and compare their material/world fingerprints. Capture
gap, thread tension, needle reaction, tissue strain, contact, work/energy,
and exact replay before claiming a complete interrupted stitch. No skin
calibration or clinical validity is established.

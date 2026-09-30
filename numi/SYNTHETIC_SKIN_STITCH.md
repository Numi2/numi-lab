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
The first grip-probe rebuild, which also gates authored-surface clearance along the
giver approach and first bite, has SHA-256
`d7639e6f81e18c01b8f4b2e092892a444a3bbd27536ad774f2ca42a7edd187fb`.
It adds `--tissue-robot-first-bite-grip-only --synthetic-skin`: a one-transaction
dynamic-needle, closed-giver grasp probe with bilateral contact, insert
coverage, swage, rod, and Matter gates. It compiles and its independent
geometry tests pass. After the long pull stopped, the native grip mode reached
a free dynamic needle and bilateral 8/8 jaw contacts with full 15/15 insert
patch masks. It failed its loaded-grasp gate: 25.33 um seat drift was within
the 100 um bound, but needle/jaw relative point speed was 0.4053 m/s against
the 0.002 m/s bound. This is contact evidence, not a qualified grasp or bite.
The latest rebuild, which additionally rejects premature puncture or tissue
mass loss at grip reset, has SHA-256
`18f7be3f39eeee80fae72006c3b184b9a33e31e434f570cc05133c384c515d85`.
The subsequent `--tissue-robot-first-bite-drive-only --synthetic-skin` mode
holds that dynamic grip and sends 48 articulated effort commands over the
first 3 ms of the needle orbit, without a kinematic scene-body target. Its
matching CPU trajectory preflight passes: planned tip advance is 12.646 um,
and the limiting joint velocity is 0.750 of its bound. It will require live
bilateral needle contact, positive actual tip advance, accepted Matter state,
all FEM tetrahedra, zero removed mass, and a qualified swage/DER strand. This
mode compiles and its CPU preflight passes, but its native Metal test has not
run because the initial grip gate fails. The earlier probe binary SHA-256 was:
`e35b24f39a40eea9691f34a8a62215be821e59a2cd54ed79627abaf497346d9e`.
Checkpoint publication now writes the actual `synthetic_skin_wound` material
identity and 30 x 24 x 1.5 mm specimen dimensions when `--synthetic-skin` is
selected; the visual probe accepts that identity only with a live v3 Matter
snapshot and verifies the dimensions before building a surface. Both targets
build, but no skin checkpoint has yet been rendered. Earlier handoff and
current visual-probe binary SHA-256 values were
`22acd0b40cf42c03195af9a69ca841de000bca94383a7f820cb2f33ea64a907d`
and
`969e16939cb8697c65ee909d2dd0a1c9c13f375cf3552a5513412ac55edc080b`.
The visual evidence manifest now carries the verified tissue-model identity.

The `--tissue-robot-first-bite-ik-only --synthetic-skin` geometry probe found
a 250-step giver approach and 185-step 5 mm/s needle-orbit path. The entry
jaw midpoint is 0.164 um from its needle seat; peak approach and bite joint
velocity ratios are 0.454 and 0.746 of their limits. An authored-surface
envelope scan of the 250 approach and 185 bite samples found at least 0.900
and 2.257 mm needle-driver clearance from the skin respectively, with no
unsafe sample. The planned tip travels 1.817 mm and clears the authored distal
surface by 0.181 mm. This proves reachable,
collision-free command geometry only. Contact, load transfer, tissue forces,
and arm-driven puncture still require a native run with a dynamic needle.

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

Open work: qualify thread pull-through with this 1.5 mm wall, then the second
lip bite, robot-driven manipulation, knot tightening,
gripper release, and measured unloaded wound-gap retention. Existing
jejunal checkpoints must not be reused as skin evidence; generate fresh
skin checkpoints and compare their material/world fingerprints. Capture
gap, thread tension, needle reaction, tissue strain, contact, work/energy,
and exact replay before claiming a complete interrupted stitch. No skin
calibration or clinical validity is established.

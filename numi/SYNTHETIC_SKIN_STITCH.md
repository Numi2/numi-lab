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
not a visibly deforming or cinched wound. The same process is now in thread
pull-through,
with output at `build-skin-wound/synthetic-skin-pull-through.log`. Inspect the
live process and log before launching another Metal run. Its first two pull
chunks had no strand contact and therefore do not qualify pull-through.
The run's executable retains the SHA-256 above. A later rebuild has SHA-256
`1993cf25e00ff2c683c6a7ea8d2db36375d5646d7c68f0f027c41ac06748c76c`;
it adds live center-wound-gap mean and maximum measurements to pull-through
progress and final output. Those new fields will not appear in the already
running process, and this rebuild alone is not pull-through qualification.
The first grip-probe rebuild, which also gates authored-surface clearance along the
giver approach and first bite, has SHA-256
`d7639e6f81e18c01b8f4b2e092892a444a3bbd27536ad774f2ca42a7edd187fb`.
It adds `--tissue-robot-first-bite-grip-only --synthetic-skin`: a one-transaction
dynamic-needle, closed-giver grasp probe with bilateral contact, insert
coverage, swage, rod, and Matter gates. It compiles and its independent
geometry tests pass, but the native grip mode has not run while the pull-
through process owns the GPU. It does not yet drive the full bite.
The latest rebuild, which additionally rejects premature puncture or tissue
mass loss at grip reset, has SHA-256
`18f7be3f39eeee80fae72006c3b184b9a33e31e434f570cc05133c384c515d85`.

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

Open work: qualify thread pull-through with this 1.5 mm wall, then the second
lip bite, robot-driven manipulation, knot tightening,
gripper release, and measured unloaded wound-gap retention. Existing
jejunal checkpoints must not be reused as skin evidence; generate fresh
skin checkpoints and compare their material/world fingerprints. Capture
gap, thread tension, needle reaction, tissue strain, contact, work/energy,
and exact replay before claiming a complete interrupted stitch. No skin
calibration or clinical validity is established.

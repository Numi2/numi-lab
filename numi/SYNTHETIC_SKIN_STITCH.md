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

The focused CTest coupon and tapered-puncture cases pass. The existing
porcine-jejunum coupling mode also passes after this change. Binary SHA-256:
`1ff94c23cd21735ee3fad3975dd0a7c39ea01dc5890926683842c93b9da03bb2`.
Synthetic material SHA-256:
`379945f593f12396b44c7148026f0250344a86278e01b7b26cb5f6d823a51e35`.

A native `--tissue-suture-passage-only --synthetic-skin` run started from that
binary and is logged at `build-skin-wound/synthetic-skin-passage.log`. Check the
actual process and log before restarting it; the probe buffers progress until
its GPU chunk returns. The separate `--tissue-curved-passage-only` mode fails
at the post-entry cadence switch for both skin and jejunum because it omits
the rod-contact capability required by that switch. The suture-contact
cadence mode passes at multiplier 4. Source now also measures the center lip
gap after knot loading and requires a reduction for the skin variant. The
rebuilt topology test passes and measures nine central lip pairs at 0.6 mm;
the knot-loaded gap gate itself still awaits a complete native sequence.
The rebuilt binary's SHA-256 is
`903de3529b78adab1256c7b1300bdfc1a3e4a5d746f5a2a2bc00e128e3dc01f7`;
the already running passage process still owns the earlier binary inode.

The `--tissue-robot-first-bite-ik-only --synthetic-skin` geometry probe found
a 250-step giver approach and 185-step 5 mm/s needle-orbit path. The entry
jaw midpoint is 0.164 um from its needle seat; peak approach and bite joint
velocity ratios are 0.454 and 0.746 of their limits. This proves reachable
command geometry only. Contact, load transfer, tissue forces, and arm-driven
puncture still require a native run with a dynamic needle.

Open work: qualify the long curved passage and pull-through with this 1.5 mm
wall, then the second lip bite, robot-driven manipulation, knot tightening,
gripper release, and measured unloaded wound-gap retention. Existing
jejunal checkpoints must not be reused as skin evidence; generate fresh
skin checkpoints and compare their material/world fingerprints. Capture
gap, thread tension, needle reaction, tissue strain, contact, work/energy,
and exact replay before claiming a complete interrupted stitch. No skin
calibration or clinical validity is established.

# Numi Human all-DoF nonvisual audit v1

The whole-body support certificate now has an opt-in
`--whole-body-all-residuals` mode. It records all 128 generalized coordinates,
not only the twelve largest accelerations, and includes the eight strongest
source-muscle contributions at each coordinate. The companion parser joins
the native indices to the pinned MyoSim source manifest, so every record has a
source coordinate, child body, muscle name, activation, tendon force, moment
arm, and generalized-force contribution.

The lower-stop experiment is independently opt-in through
`--fifth-mcp-lower-stop-counterfactual`; routine support certificates no longer
pay for or silently include that negative control.

## Current measured state

The current passive-aware NHMYO2 fit is documented in
`NUMI_HUMAN_PASSIVE_FIT_V4.md`, and its recruitment convergence qualification
is documented in `NUMI_HUMAN_RECRUITMENT_CONVERGENCE_V5.md`. The Apple M4 Pro
960-sweep rerun closes body weight to `2.33e-9` relative error and replays
bitwise, but internal balance remains false:

- normalized residual RMS: `0.980016256505`;
- 34 coordinates exceed `1 rad/s2`, two exceed `10 rad/s2`, and none exceed
  `100 rad/s2`;
- the largest errors are bilateral wrist flexion at `20.09/16.53 rad/s2`;
- the next largest errors are left third-MCP flexion (`-9.24`), thumb CMC
  flexion (`-7.04`), right third-MCP flexion (`-6.73`), left CMC abduction
  (`6.37`), and right fourth-MCP flexion (`6.22`); and
- all lower-body residuals remain in the complete 128-coordinate report rather
  than being omitted by a hand-only certificate.

The bilateral fifth-ray force decomposition remains nearly mirrored. Adding
the activation-zero fit gate reduces fifth-MCP abduction from
`62.30/60.34 rad/s2` to `1.76/1.25 rad/s2`. `UI_UB5` passive tension falls
from `2.23 N` to `0.213 N` against a source oracle of `0.0746 N`; LU_RB5,
RI5, and FDS5 reproduce their zero source passive force to numerical precision.
The remaining fifth-MCP error is no longer a dominant whole-body blocker.

The source-derived ADM candidate was tested because it is absent from MyoSim,
but its straight pisiform-to-proximal-phalanx line has the wrong generalized
force sign and was rejected. The bilateral lower joint stops also reduce the
scalar objective but remain nonanatomical and dynamically unbalanced. The next
mechanics owner is therefore a source-resolved intrinsic/extensor/flexor
force-sharing constraint (including pulley/hood behavior). The dominant live
scope is now source-resolved wrist and third-MCP force sharing, followed by the
measured shoulder and lower-body residual families. Arbitrary torques, sign
flips, and rest-at-stop fixes are excluded.

The current machine-readable evidence and exact pinned source-name map are in
`docs/media/numi-human-recruitment-convergence-v5/`; earlier all-DoF states
remain archived in their versioned media directories.

## Boundary

This is a static unilateral support and source-muscle recruitment audit. It is
not internal equilibrium, dynamic contact, sustained standing, or deformable
tissue qualification.

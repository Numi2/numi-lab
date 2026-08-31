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

The final adaptive NHMYO2 fit is documented in
`NUMI_HUMAN_COMPLIANT_FIT_V3.md`. The Apple M4 Pro rerun closes body weight to
`2.17e-9` relative error and replays bitwise, but internal balance remains
false:

- normalized residual RMS: `2.93226681485`;
- 50 coordinates exceed `1 rad/s2`, seven exceed `10 rad/s2`, and none exceed
  `100 rad/s2`;
- the largest errors are the bilateral fifth-MCP ab/adduction DoFs at
  `-62.30` and `-60.34 rad/s2`;
- the next largest errors are bilateral wrist flexion (`54.28`, `51.47`),
  bilateral third-MCP flexion (`-19.30`, `-18.61`), and left shoulder rotation
  (`15.36`); and
- all lower-body residuals remain in the complete 128-coordinate report rather
  than being omitted by a hand-only certificate.

The bilateral fifth-ray force decomposition remains nearly mirrored, but the
false lumbrical preload has been removed. At the final optimum FDP5 carries
about `9 N`, EDM about `5.8 N`, UI_UB5 retains `2.23 N` passive tension, and
the fifth lumbrical carries only `0.01567 N` passive tension instead of
`17.4896 N`. The remaining fifth-MCP error is therefore a force-sharing and
route-constraint problem, not a bone-rendering problem or a reason to restore
the rejected lumbrical preload.

The source-derived ADM candidate was tested because it is absent from MyoSim,
but its straight pisiform-to-proximal-phalanx line has the wrong generalized
force sign and was rejected. The bilateral lower joint stops also reduce the
scalar objective but remain nonanatomical and dynamically unbalanced. The next
mechanics owner is therefore a source-resolved intrinsic/extensor/flexor
force-sharing constraint (including pulley/hood behavior), followed by the
measured wrist and lower-body residual families. Arbitrary torques, sign flips,
and rest-at-stop fixes are excluded.

Machine-readable evidence and the exact pinned source-name map are in
`docs/media/numi-human-whole-body-all-dof-v1/`.

## Boundary

This is a static unilateral support and source-muscle recruitment audit. It is
not internal equilibrium, dynamic contact, sustained standing, or deformable
tissue qualification.

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

The Apple M4 Pro run closes body weight to `2.01e-9` relative error and replays
bitwise, but internal balance remains false:

- normalized residual RMS: `3.79966295338`;
- 56 coordinates exceed `1 rad/s2`, 13 exceed `10 rad/s2`, and two exceed
  `100 rad/s2`;
- the two >100 coordinates are the bilateral fifth-MCP ab/adduction DoFs at
  `-147.46` and `-146.13 rad/s2`;
- the next largest errors are bilateral wrist flexion (`74.70`, `68.92`),
  bilateral third-MCP flexion (`-38.37`, `-33.51`), and bilateral fifth-ray
  flexion/abduction coupling; and
- lower-body blockers remain visible rather than hidden by hand errors:
  right hip flexion/adduction and left knee appear in the top seventeen.

The bilateral fifth-ray force decomposition is nearly mirrored. FDP5 carries
about 51 N with a 2.558 mm transverse moment arm, the lumbrical has about 17.49
N passive tension with the opposing 6.494 mm moment arm, and EDM is saturated
at 59.06 N while contributing another same-sign transverse moment. This is a
force-sharing/recruitment and route-constraint problem, not a bone-rendering
problem.

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


# Numi Human ADM feasibility v1

The native FP64 probe consumes the source compiler's `NHADM1` payload and the
same pinned `NHRIGID2` body/joint model used by the whole-body equilibrium
certificate. For each hand it:

1. evaluates the registered pisiform and fifth-proximal-phalanx endpoints with
   analytic articulated point Jacobians;
2. applies a unit internal tensile pair and projects it to generalized force;
3. checks the target fifth-MCP moment arm against an FP64 central-difference
   route-length oracle;
4. calculates the tensile force required to cancel the measured neutral-pose
   residual; and
5. checks bilateral parity, rigid-root force/moment closure, force-capacity
   sensitivity, and bitwise replay.

## Result: candidate rejected

The source-inferred straight ADM routes are mechanically coherent but have the
wrong sign for this residual:

| Metric | Right | Left |
|---|---:|---:|
| Route length | 68.14 mm | 74.07 mm |
| Fifth-MCP moment arm | -7.446 mm | -8.031 mm |
| Force algebraically required | -9.95 N | -9.07 N |

A muscle-tendon can pull but cannot generate the negative tensile magnitudes
shown above. Adding this ADM route would reinforce the existing negative
generalized residual. It is therefore not admitted into the live Hill-type
muscle set.

This is not a numerical ambiguity: the maximum analytic-versus-central-
difference moment-arm error is 1.96e-11 m, bilateral absolute moment-arm
difference is 7.29%, root force residual is zero, root moment residual is
5.07e-16 Nm, and replay is bitwise.

The next correction must come from anatomically valid antagonist recruitment,
passive tissue/slack-length calibration, or a better registered multi-segment
intrinsic-hand route. Sign-flipping ADM, applying an arbitrary torque, or
holding both joints at a hard stop is prohibited.

Evidence: `docs/media/numi-human-adm-feasibility-v1/`.

## Boundary

This is a source-inferred CPU FP64 feasibility result. It does not qualify a
live Hill-type actuator, whole-body static equilibrium, dynamic contact, or a
subject-specific route.


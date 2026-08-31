# Numi Human upper passive joint-tissue preflight

Status: **preflight only**. This milestone adds a sourced, conservative static
passive-force channel to the whole-body equilibrium oracle. It does not claim a
complete hand, ligament, cartilage, fascia, or sustained-motion model.

## Why this exists

The source MyoSim full-body payload supplies upper-extremity muscle paths and
muscle-tendon parameters, but its neutral wrist and finger coordinates have no
passive coordinate stiffness. Muscle-only recruitment consequently plateaued
at an acceleration-normalized internal residual of `21.3456930846`, even after
exact nonlinear-force checkpoints, coupled pose/recruitment search, and 960
activation sweeps. That plateau is a source-mechanics gap, not a rendering
problem.

The compiler now accepts explicit rows of

`tau_target = -K(target, source) * (q_source - q_rest)`

and includes this generalized force exactly once in support, recruitment,
residual, pose search, replay, and diagnostics. Existing callers pass no rows
and retain the previous behavior.

## Sourced v1 parameters

The wrist uses the all-subject mean stiffness matrix reported for passive wrist
flexion/extension and radial/ulnar deviation:

`K = [[1.28, -0.18], [-0.18, 1.74]] N m/rad`

Source: *Characterization of the passive stiffness of the human wrist and
forearm* (Pando et al.), open primary article:
<https://pmc.ncbi.nlm.nih.gov/articles/PMC3424077/>.

The four non-thumb rays use the published middle-finger linearized stiffnesses
as an explicit fallback where digit-specific source data are unavailable:

| Coordinate | Stiffness (N m/rad) |
|---|---:|
| MCP flexion | 0.054261 |
| MCP ab/adduction | 0.1779 |
| PIP flexion | 0.0231 |
| DIP flexion | 0.0037206 |

Source: *A biomechanical model of the human finger for studying the effects of
external loading on finger stiffness* (primary open article):
<https://pmc.ncbi.nlm.nih.gov/articles/PMC10869888/>.

The bilateral MyoSim DoF mapping is explicit: right wrist deviation/flexion
`40/41`, left `78/79`; right ray bases `46, 50, 54, 58`, with left offset `38`.
The rest coordinate is zero because this is the source model's neutral pose.
There are 40 coupling rows total. The same middle-finger values on index,
ring, and little rays are a documented inference, not digit-specific evidence.
Thumb passive mechanics remain absent.

## Apple M4 Pro nonvisual result

The executable whole-body support certificate used pinned MyoSim full-body
mechanics, NHEQ1 joint constraints, NHCNT1 bilateral plantar contacts, 65
activation samples, 240 activation sweeps, and 12 coupled pose sweeps.

| Metric | No passive rows | Sourced upper v1 |
|---|---:|---:|
| Internal normalized residual RMS | 21.3456930846 | 3.79966295338 |
| Reduction | - | 82.1994% |
| Root acceleration residual | 4.97491334401 | 4.93896846021 |
| Active position limits | 11 | 3 |
| Accepted pose steps | 4 | 12 |
| Weight/support error | 2.1842e-9 | 2.0084e-9 |
| Replay | bitwise | bitwise |
| Internal balance | false | false |

All 10 unilateral plantar witnesses remained active and nonnegative. The
97.1319506911 kg body received 952.864475124 N support against
952.864477038 N expected weight; maximum floating-root force residual was
1.9137e-6 N-equivalent. The passive run took 97.06 s on Apple M4 Pro.

## Remaining anatomical blocker

The dominant errors are symmetric fifth-ray MCP ab/adduction DoFs 59 and 97:
`-0.07405` and `-0.07285 N m`, producing approximately `-147` and `-146
rad/s^2` in their small inertias. On the right, FDP5 contributes `-0.13029 N m`,
the passive lumbrical contributes `+0.11358 N m`, and extensor digiti minimi is
already saturated at activation 1.0 while adding `-0.04749 N m`. The left side
has the mirrored pattern.

This points to the source extensor-hood/interosseous route and force-sharing
representation. The next correction must begin with a bilateral moment-arm and
attachment audit against open anatomical evidence. Adding another arbitrary
spring would hide the error without repairing tendon-to-bone mechanics.

## Evidence boundary

- This is a linearized conservative static preflight, not live ligament,
  capsule, cartilage, fascia, or tendon material dynamics.
- It is not a deformable solve and contains no damping or hysteresis.
- It does not establish internal static balance, dynamic stability, sustained
  standing, or loaded hand contact.
- The ARMS hand/wrist model is useful as a literature validation reference, but
  its downloadable model license contains noncommercial restrictions; no ARMS
  model data or code are incorporated here.
- Promotion requires source-resolved extensor-hood/interosseous mechanics,
  thumb passive mechanics, one-step force transfer, and sustained loaded-motion
  evidence.

Machine-readable evidence:
[`docs/media/numi-human-upper-passive-joint-v1/m4-pro.json`](media/numi-human-upper-passive-joint-v1/m4-pro.json).

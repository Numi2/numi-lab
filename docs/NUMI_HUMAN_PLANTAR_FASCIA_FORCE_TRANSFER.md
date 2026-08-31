# Numi Human bilateral plantar-fascia force transfer

The Human now has a fail-closed, nonvisual preflight for the reduced plantar
fascia windlass law. It resolves five rays on each foot from exact named
BodyParts3D 4.0 surface patches: the calcaneal plantar origin, the matching
metatarsal-head pulley, and the proximal-phalanx plantar insertion. All five
rays on a side use the single authored source MTP coordinate; the certificate
does not invent independent toe articulation.

For a routed band with calcaneal point `c`, metatarsal-head point `w`, distal
point `d`, pulley radius `r`, and MTP angle `theta`, the reduced length and
tension are

```text
L = |c - w| + |d - w| + r |theta|
T = max(0, k (L - L0)).
```

The published five-ray rest-length pattern is
`0.151, 0.149, 0.148, 0.140, 0.131 m`; the reduced stiffness allocation is
`60, 50, 50, 20, 20 N/mm` (aggregate `200 N/mm`). The aggregate approximates
the published intact-fascia mean of `203.7 +/- 50.5 N/mm`; the per-ray split is
a reduced modeling allocation, not five independently measured cadaveric
properties. Each tension is applied as one three-point force system at the
calcaneus, metatarsal pulley, and proximal phalanx. Exact point Jacobians map
that system once through `J^T`, so the preflight has one passive rigid-force
authority and no hidden plantar actuator.

On Apple M4 Pro, `0.1 rad` of prescribed MTP dorsiflexion engaged all ten
rays. Total tension was `285.054 N` right and `288.049 N` left. Maximum force
closure residual was `8.53e-6 N`, maximum world-origin moment residual was
`7.25e-7 N m`, and the force-on versus force-off articulated response differed
by at least `1.56e-6 rad` in configuration and `0.0156 rad/s` in velocity.
Replay was bitwise. The machine-readable receipt is
[`bilateral-m4-pro.json`](media/numi-human-plantar-fascia-v1/bilateral-m4-pro.json).

## Reproduction

```sh
metalrobo_numilab_human_myosim_visual_probe \
  myosim-fullbody-core-reference.nhrigid \
  myosim-fullbody-muscle-reference.nhmyo \
  bodyparts3d-myosim-major-bones.nhbones \
  /tmp/unused-visual-output \
  --muscle-step-seconds 0.0001 \
  --support-contact-payload myosim-fullbody-support-contact.nhcnt \
  --joint-equality-payload myosim-fullbody-joint-equalities.nheq \
  --bilateral-plantar-fascia-certificate
```

The command exits before camera setup or rendering.

## Sources

- Sikidar and Kalyanasundaram, *An open-source OpenSim ankle-foot
  musculoskeletal model for assessment of strains and forces in dense
  connective tissues*, DOI `10.1016/j.cmpb.2022.106994` (five-ray topology
  and source rest-length pattern).
- Kitaoka et al., *Material properties of the plantar aponeurosis*, DOI
  `10.1177/107110079401501007` (intact-fascia structural stiffness and failure
  measurements).

## Evidence boundary

This is a tendon force-transfer law and a kinematic/one-step response
preflight. It is not a deformable plantar-fascia FEM solve and is not promoted
as a live NumanX continuum owner. The present law is linear after engagement;
it does not qualify the measured nonlinear toe region, rate dependence,
failure, subject-specific material calibration, loaded foot-ground contact,
sustained gait, or clinical validity. A future FEM promotion must publish its
prepared state and reactions through the owning Human/NumanX transaction with
rollback and replay; a kinematic target shortcut is not admissible evidence.

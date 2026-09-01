# Numi Human bilateral plantar-fascia mechanics

The Human owns five passive plantar-aponeurosis rays per foot in the live
borrowed-command-buffer Human/Matter transaction. Exact BodyParts3D 4.0
surface patches provide the calcaneal origin, matching metatarsal-head pulley,
and proximal-phalanx insertion. Each foot retains the one authored OpenSim MTP
coordinate shared by all five rays; this model adds no independent toe joints
and no plantar actuator.

## Owning force law

For geometric segment length `Lg`, pulley radius `r`, signed MTP winding
`theta`, neutral route length `L0`, and stretch `lambda`, the route is

```text
L      = Lg + r |theta|
lambda = L / L0
T      = 0, lambda <= 1
```

Above slack, the source Natali-type hyperelastic relation is

```text
sigma = mu (lambda^2 - 1/lambda)
      + k/(2 alpha) [exp(alpha (lambda^2 - 1)) - 1] lambda^2
A     = A0 lambda^(-2 nu)
T     = sigma A
```

with `mu=14.449 MPa`, `k=254.02 MPa`, `alpha=10.397`, and `nu=0.4`.
The public model's `70 mm^2` aggregate reference area is allocated across the
five source rays as `21, 17.5, 17.5, 7, 7 mm^2`. This proportional ray split
is a reduced modeling assumption. It is not five independently measured
subject-specific areas.

Three point forces act at the origin, pulley, and insertion. The explicit arc
derivative contributes an equal-and-opposite pure couple of magnitude
`T r sign(theta)` to the pulley and toe bodies. All point loads and torques are
mapped once through exact body `J^T`; force and world-origin moment closure are
audited before the enclosing Human transaction can be accepted. A CPU FP64
oracle independently evaluates tension, strain, and Simpson-integrated strain
energy. Routes reject non-unit axes/quaternions, nonfinite parameters, and
strain above the declared limit.

## Deformable representation

Each side also contains a `220`-node, `250`-tetrahedron Matter FEM shell. Its
soft isotropic matrix is only a transverse shape regularizer. It does not own
axial collagen stiffness and therefore cannot carry compression as a false
plantar-fascia force. The matrix rest scaffold is prepared at the bounded
qualification pose while the routed bands retain neutral slack lengths; this
avoids an artificial Dirichlet impact and does not erase windlass preload.

The M4 Pro eight-step gate at `0.1 rad` dorsiflexion engaged all ten rays. It
measured total routed tension of `165.931 N` right and `169.827 N` left,
stored energy of `0.111643 J` and `0.119569 J`, force closure below `9.84e-7 N`,
and moment closure below `5.34e-7 N m`. FEM determinants remained within
`0.97871...1.00046` right and `0.95557...1.00993` left; maximum anchor error
was `10.6 um`. The live force-on versus source-only response differed by
`1.00e-4`/`1.07e-4` in configuration and `0.2219`/`0.2387` in velocity.
Rejection rollback and replay are exact.

The machine-readable receipt is
[`bilateral-m4-pro-live.json`](media/numi-human-plantar-fascia-v2/bilateral-m4-pro-live.json).

## Reproduction

```sh
metalrobo_numilab_human_myosim_visual_probe \
  myosim-fullbody-core-reference.nhrigid \
  myosim-fullbody-muscle-reference.nhmyo \
  bodyparts3d-myosim-major-bones.nhbones \
  /tmp/unused-visual-output \
  --tendon-payload numi-human-tendon-attachments.nhtendon \
  --muscle-step-seconds 0.0001 \
  --muscle-step-count 8 \
  --support-contact-payload myosim-fullbody-support-contact.nhcnt \
  --joint-equality-payload myosim-fullbody-joint-equalities.nheq \
  --bilateral-plantar-fascia-certificate
```

The command is nonvisual and exits before camera setup or rendering.

## Sources

- D'Hondt et al., open four-segment dynamic-foot model and implementation,
  including the Natali relation and `70 mm^2` reference area:
  <https://github.com/Lars-DHondt-KUL/3dpredictsim/tree/four-segment_foot_model>
- Natali et al., nonlinear and time-dependent plantar-aponeurosis mechanics:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC3950543/>
- Kitaoka et al., intact-fascia structural stiffness and failure measurements:
  <https://pubmed.ncbi.nlm.nih.gov/7834064/>
- Stecco et al., cadaveric five-bundle anatomy:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC3879302/>

## Evidence boundary

This qualifies a bounded bilateral nonlinear windlass owner, exact tendon-to-
bone load transfer, a neutral deformable shape matrix, rollback, and replay on
Apple M4 Pro. It does not qualify subject-specific calibration, viscoelastic
rate dependence, damage or rupture, loaded foot-ground contact, sustained
walking, clinical predictions, or independent toe articulation.

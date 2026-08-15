# Matter

Matter is Numi Lab's Apple-native variational solver for coupled FEM, MPM,
mixed fields, internal material variables, and every contact involving a Matter
continuum. The `coupled` branch has one GPU-resident nonlinear authority rather
than sequenced deformable, pressure, transport, contact, and rigid-response
solvers.

The environment-wide Newton unknown is

\[
x=(v_{FEM},\pi,T,p_f,\phi,a,v_{MPM},\Delta v_{rigid}),
\]

and restarted matrix-free FGMRES is the sole linear convergence owner. IPC's
squared-distance logarithmic barrier contributes primal gradients and PSD
Hessian actions directly; per-node timestep ratios keep cross-rate contact the
gradient and Hessian of one environment action. No contact multipliers,
Delassus rows, or post-contact correction solve remain.
Matter ABI v21 explicitly identifies the tapered needle-tip capsule that may
drive tissue puncture; generic shaft or swage capsules cannot mutate topology.
Matter ABI v20 exposes only controls that own live work: Newton/FGMRES
budgets, bounded field-smoother passes, residual tolerances, and a minimum
contact-separation ratio relative to the authored contact slop. Both contact
line search and final Metal acceptance read that ratio. Relative correction is
finite certificate telemetry; the assembled residual is the convergence
authority.
The right preconditioner uses the componentwise diagonal of those same PSD IPC
blocks in its FEM and MPM fine and translation-mode mechanics approximations;
it does not assemble a second contact matrix or introduce another iteration
owner.
MetalWorld keeps sole ownership of ABA and generalized coordinates while its
borrowed coupled-candidate callback supplies kinematics, mass action,
inverse-mass preconditioning, and accepted publication.

Matter Language supports explicit state updates, authored implicit residuals,
specialized multiplicative von Mises and Drucker-Prager return maps, and
`average|max|sum` remesh-transfer policies. Active sparse MPM grid velocity is
part of the same Krylov vector as FEM and rigid increments. Its private Krylov
and basis arenas reserve only the cooked quadratic-support bound rather than
the full authored grid; APIC, deformation, and material state publish only
after global candidate acceptance.

Topology mutation is transactional. Cohesive separation, erosion, edge
split/collapse, 2–3/3–2 flips, and vertex smoothing rebuild derived incidence,
surface contact, and preconditioner structures on Metal. Cavity-wide material
transfer follows each state's average/max/sum policy, and the rebuilt nodal
state is corrected to its pre-remesh momentum and volume-integrated fields
before certification. Conservation and element-quality failures reject the
environment. Private arenas grow
geometrically only between completed submissions through a borrowed
command-buffer migration. Migration rebases every active tetrahedron and
cohesive-face reference into the expanded global arenas before rebuilding
incidence; shaders never allocate.

The hot path uses one borrowed MetalWorld command buffer, private authoritative
state, deterministic sparse ordering, SIMD32 reductions, and no internal
commit, wait, or CPU counter read.

## Curved surgical-needle tissue passage

The dual-PSM surgical probe now drives the authored 26 mm half-circle needle
about its actual 8.28 mm curvature centre at a 20 mm/s terminal speed. Matter
admits entry from accepted tapered-tip contact, then extends a connected
diameter-scale tract only when the sharp point reaches the live frontier. A
rotating tip follows its swept endpoint velocity rather than the finite
collider chord, and midpoint-tangent chords bound accumulated path drift.
The terminal taper and its adjacent widening segment both remain live tissue
contact geometry; the physically swaged 250 mm monofilament is carried by the
DER constraints rather than prescribed along the needle orbit.

On Apple M4, the current 0.77 mm porcine-jejunum coupon qualification reached
101.5 um of clearance beyond the *deformed* distal surface after 916 62.5 us
microsteps. It retained all 3,456 tetrahedra with zero removed tissue mass,
formed eight connected tract segments with 5.22 um maximum orbit error and
2.51 um radial-error range, retained a 0.99774 minimum determinant, ended at
6.85 um hard-swage error, and reported zero failed steps. The tract is a
mass-conserving sub-element contact discontinuity, not yet a constitutive crack
surface or clinical validation. Receiver extraction and knot formation remain
separate physical qualification boundaries.

```sh
./build-coupled-dev/bin/metalrobo_dual_psm_suture_handoff_probe \
  --tissue-curved-passage-only
```

## dVRK GS21 suture pickup

![dVRK Large Needle Driver lifting a GS21 needle and blue suture](docs/media/numi-lab-dvrk-gs21-suture-pickup.png)

The source-grounded Classic PSM Large Needle Driver acquires a curved GS21
through distributed opposing tooth contact without a weld, then carries its
physically swaged 25-node, 180 mm discrete-elastic-rod thread. The deterministic
3030-step qualification retained the grasp for 2000 consecutive lift frames,
lifted the needle 8.043 mm and thread root 7.905 mm, ended at 4.26 um swage
error, reported zero penetration and a `2.01e-5` maximum KKT residual, and
byte-restored the articulated/rigid rollback checkpoint.

The image is a 1280x960 Apple M4 Metal reference render from the accepted
physics state. Its 7172-triangle presentation pack is bound to articulated and
rigid body poses; the 181.527 mm rendered thread centerline comes directly from
the accepted rod state. This is simulation evidence, not clinical validation.
The thread geometry and coupling are executable; its material constants are
research defaults rather than package- or specimen-calibrated values.

```sh
./build-coupled-dev/bin/metalrobo_supported_needle_pickup_probe \
  --state-output /tmp/numi-suture-pickup.state.tsv
./build-coupled-dev/bin/metalrobo_suture_visual_probe \
  --state /tmp/numi-suture-pickup.state.tsv \
  --output-dir /tmp/numi-suture-visual
```

Current Apple M4 qualification covers compiler/package invariants and live
Metal contact, multiphysics, sparse-MPM, articulated-coupling, and reference
render paths. These checks establish the recorded workloads; they are not a
blanket hardware or performance claim.

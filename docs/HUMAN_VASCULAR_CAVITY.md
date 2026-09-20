# Deforming vascular cavities

Matter can bind one absolute-volume hydraulic compartment to the closed inner
boundary of an immutable FEM wall. The compartment pressure becomes an
independent unknown in the existing Newton–FGMRES system. The same accepted
transaction owns wall motion, hydraulic volume, pressure, flow, transported
amounts, and exact physical time.

## Equations and units

For each bound compartment, the ordinary conservative volume row remains
`V - V_accepted + dt * sum(signed Q) = 0`. An additional pressure row enforces
`V - V_boundary(x) = 0`. Connections use the independent absolute pressure in
Pa; volume is in m³, flow in m³/s, and transported amount in mol. Bound
compartments have no additional compliance, reference-volume offset, or
elastance waveform: these would duplicate the wall response.

The oriented triangular boundary defines its signed volume. A fixed accepted
vertex supplies the evaluation origin. Wall force is `(p - p_external) * G`,
where `G = (grad V(x_accepted) + 4 grad V((x_accepted+x)/2) + grad V(x))/6`.
Volume is cubic in the positions, so this discrete gradient satisfies
`G · (x-x_accepted) = V(x)-V(x_accepted)` in exact arithmetic. This is an
interface work identity; it does not remove the time-integration error or
dissipation of the rest of the model.

The matrix-free operator differentiates both pressure and geometry, including
the curvature of this discrete gradient. FEM directions are velocity
increments, so their geometric direction is `dt * delta_velocity`. Existing
Human attachment lifting and transpose scattering apply once to the wall
force and its derivative. All borrowed external FEM loads are retained in a
derived force buffer. The geometry constraint participates in line search and
final residual certification.

A bound wall uses every hydraulic microtick. Compilation sets its immutable
scheduler base, active, and requested exponents to the world maximum; package
and snapshot admission check that invariant. A separate mechanical stepping
loop is unnecessary.

Immutable worlds without mutation commands skip topology transactions and
mass rebuilding. Their cooked nodal masses and inverse masses remain exact;
mutable worlds and explicit mutation requests retain the existing validation
and transaction path.

## Authoring and admission

Use `WorldSource::vascular.cavities` with
`VascularPressureLaw::deformingCavity`. Each `VascularCavitySource` identifies
one compartment, one FEM object, initial absolute pressure, pressure scale,
normalized geometry tolerance, source and mechanical identities, and stable
oriented `materialWall` faces using object-local node indices. The public
example is `matter/tools/vascular_cavity_fixture.hpp`.

The compiler requires:

- Exactly one binding for each deforming compartment, using absolute volume.
- Real exposed tetrahedral faces, oriented from the lumen into the wall.
- A connected, closed, consistently oriented vertex-manifold surface with
  positive enclosed volume and unique owned faces.
- An initial hydraulic volume matching the actual cooked geometry within both
  the declared tolerance and an independent FP32 relative bound.
- Immutable, nonadaptive FEM ownership and finite representable pressure and
  transmural pressure.

An outer solid boundary cannot be relabeled as a hollow lumen: the required
orientation and positive enclosed volume reject that construction. Artificial
partition interfaces are rejected. Provenance identities bind the authored
inputs but do not prove anatomical correctness, embeddedness, or calibration.

ABI 29 and package 14 serialize the boundary tables, incidence, and appended
pressure rows. Snapshot 7 and accepted-state proof manifest 7 cover the new
state layout. Reset, checkpoint, replay, and rollback use the same vascular
state arena. Snapshot restore also rejects inconsistent hydraulic volume and
FEM cavity geometry; signed pressure remains valid when finite.

## Qualification and remaining anatomy work

`numi-matter-vascular-cavity-check` exercises a synthetic hollow shell with a
real absent central cell: 64 nodes, 156 tetrahedra, 12 inner faces, eight free
inner nodes, and 56 fixed exterior nodes. Its direct Metal-kernel checks and
accepted-step checks separate operator correctness, pressure work,
conservation, wall deformation, and transaction behavior. The fully fixed wall
is a control. The source fixture declares its material constants and carries
no anatomical calibration claim.

This contract is currently available through native C++ WorldSource authoring.
Human physiology JSON v1–v3 remain their existing lumped-network formats.
Anatomical use still requires an admitted wall mesh, a consistent cavity/wall
registration, physical port/valve boundary ownership, and measured material
and loading calibration. The previous RA/RV cavity partition alternatives do
not select an anatomical wall or define a physical septum.

Hydraulic volume and transported moles are not three-dimensional blood mass,
inertia, or momentum. The implementation does not add blood mass to the FEM
wall or subtract it from an anatomical rigid donor. That requires explicit
source mass inclusion, conservative donor accounting, and spatial mass and
momentum transfer. This increment does not qualify whole-heart mechanics,
standing, walking, or biological accuracy.

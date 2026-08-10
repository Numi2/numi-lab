# Matter

Matter is Numi Lab's Apple-native solver for coupled rigid, articulated, and
continuum physics. The `coupled` branch develops one GPU-resident nonlinear
authority rather than sequencing independent deformable, pressure, transport,
and contact solvers.

For mixed FEM, one Newton step solves over

\[
x=(v,\pi,T,p_f,\phi,a,\lambda),
\]

where `v` is nodal velocity, `π` is mechanical pressure, `T` is temperature,
`p_f` is pore pressure, `φ` is electric potential, `a` is activation, and `λ`
is contact impulse. Matrix-free FGMRES applies the generalized KKT operator;
specialized PCG and sparse contact kernels act only as block
preconditioners. Contact uses the full sparse Delassus response

\[
W=JM^{-1}J^\mathsf{T},
\]

including off-diagonal coupling between contacts that share a rigid body or
articulation. FEM contact remains in the primal KKT blocks so its inverse mass
is not counted twice.

All work is encoded on MetalWorld's borrowed command buffer. The hot path does
not create a second queue, commit internally, wait, or read solver state back
to the CPU. A fixed 16-vector Krylov basis is restarted to honor the larger
linear-iteration budget without retaining a 48-vector arena.

Topology mutation is a deterministic active-set change inside the same
transaction. Fracture, cutting, puncture, or deactivation updates candidate
topology and restarts Newton. Mechanical state, pressure, transported fields,
contact warm starts, topology, constitutive history, scheduler state, and
learned weights commit together or roll back together.

This branch is active solver development. Bounded live-Metal probes are the
qualification workloads; unsupported application replicas are not retained as
solver evidence.

Build and run the owning probe:

```sh
cmake --build build-coupled-dev --target metalrobo_matter_physics_probe
./build-coupled-dev/bin/metalrobo_matter_physics_probe --multiphysics
./build-coupled-dev/bin/metalrobo_matter_physics_probe --coupled-contact
```

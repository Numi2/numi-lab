# Native midpoint transaction: fixed regularized contact

Before native trials, implement an isolated opt-in native transaction for the
existing single-environment stateless elastic, frictionless reference-FEM /
immutable plane fixture. The prior AL and directional CCD options stay off.
The public timestep remains h. The Newton working configuration uses h/2 and
the same reference-area energy coefficient, material, plane and clearance as
the comparison. Preflight must reject unsupported owners and mutable inputs.

Retain accepted state while working. Assemble at the midpoint, validate the
reflected endpoint and element Jacobians, and certify x1-x0=h*vm and
M(v1-v0)=h*fm in full-step physical units. A half-stage residual must be
converted consistently and must not be replaced by a backward-Euler residual
at the endpoint. Publish full physical contact impulse while preserving its
curvature and force-law meaning. Check accepted/rejected history and rollback.

Native gates: unchanged-law backward Euler control, freefall, contact onset,
active force, endpoint crossing rejection, invalid-element rejection, late
owner rejection and exact replay from nonzero contact. Compare nodal states,
impulses, work and independently reconstructed FP64 midpoint residual against
the fixed-mesh CPU reference at matching h. Use unchanged 0.25% force/impulse
gates for the 10 ms fixed-mesh reference, then separate mesh refinement. Pin
revision, binary, shader and inputs; retain failures. A completed scoped step
does not qualify spatial convergence, full Matter coupling or performance.

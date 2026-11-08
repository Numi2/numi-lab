# Midpoint Metal endpoint candidate: predeclared scope

Implement a device-only, research-stage kernel that accepts separate accepted
and midpoint reference-mode FEM arrays and writes a third endpoint array, plus
per-node result codes. It is not called by the runtime and must not expose an
integrator option. It reflects displacement and velocity and checks finite
values and fixed-node consistency. Use strict, outward-rounded interval
arithmetic to prove both accepted and reflected positions exceed a declared
plane-clearance floor. An uncertified, nonfinite, malformed or crossing node
must produce a failure code and leave its output slot untouched. No accepted
state, status or history buffer is writable through this kernel.

Before running on Apple M6, exercise successful free and fixed nodes, a
positive-gap midpoint whose endpoint crosses the plane, an initially unsafe
point, finite tangential motion, a non-axis-aligned plane, cancellation at a
large coordinate origin, exact-floor uncertainty, nonfinite/overflow inputs,
and mismatched node metadata. Compare valid endpoint displacements and
velocities with independent FP64 evaluation of the supplied Float32 inputs.
Check the full accepted/midpoint input arrays for byte preservation and output
sentinels for every rejection. Pin source and shader. Retain unexpected
failures.

This is an intermediate endpoint mechanism, not a runtime or physical-contact
qualification. Material/Jacobian checks, the midpoint solve, physical residual,
impulse/history conversion, identity and atomic commit are still required.

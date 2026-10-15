# Narrow whole-sweep mortar witness exemption (source 091 + overlay 002)

This package corrects the paired-boundary witness rule from overlay 002. The exemption is now limited to feature queries whose four (vertex-triangle) or four (edge-edge) endpoint displacements are finite and exactly equal in FP32. That condition makes the relative feature configuration invariant over the complete linearly interpolated sweep; it is a sufficient, deliberately narrow whole-sweep condition. The existing paired-barycentric-boundary and non-approaching checks remain required. Any relative deformation, sliding, unequal rounding, or unsupported motion stays contact-eligible, so this may retain false-positive seam contacts rather than hide later collisions.

## Why overlay 002 was not conservative

The manifested source is `build-source/matter/src/metal/contact.metalinc` with SHA `10b6e5ed205a5616cf5cde0896779122cd8bc920b716c8918ee50e6b20a6ac48`. In `considerMortarVertexTriangle` and `considerMortarEdgeEdge`, after CCD returns its earliest witness, it computes relative displacement at that one witness and suppresses the entire feature query when `dot(relativeDisplacement, normal)` is nonnegative within epsilon. The CCD query is not restarted after the witness. A changing triangle normal can therefore permit a later inward crossing after the first, locally non-approaching seam hit.

A CPU counterexample is recorded in `cpu_boundary_contract_check.json` and uses the solver's linear endpoint interpolation exactly. A=(0,0,0), C=(0,1,0), B(0)=(1,0,0), B(1)=(cos(10°),0,sin(10°)), and P(t)=(0.8t,0,0.08t); therefore B(t)=(1+t(cos(10°)-1),0,t sin(10°)). At t=0, P=A and the initial +Z normal dot point displacement is +0.08. The later edge crossing is at

`t = 0.08 / (0.8 sin(10°) + 0.08(1-cos(10°))) = 0.5708824738476331`.

At that time the edge parameter is 0.4607016386514907 (strictly interior), the point-edge cross residual is zero, and the relative normal speed is -0.0802994110, an inward crossing. Thus overlay 002's local test can discard the initial endpoint witness and miss a later crossing even under linear nodal trajectories.

## Bounded evidence

`cpu_boundary_contract_check.py` checks the analytic counterexample, exact common-translation eligibility, changed-motion rejection, and the prior barycentric/shared-edge/4-patch controls. It is a CPU predicate check only. The sufficient whole-sweep condition is exact equality of all four feature-vertex endpoint displacements, which makes their relative positions invariant for the entire linear sweep. Unequal or unresolved motion remains eligible. The patch applies cleanly to frozen source 091; no Metal compilation or GPU execution was performed for this 003 correction. Existing overlay-002 native/CPU artifacts remain immutable and are not evidence that this corrected patch ran natively.

The four-patch capacity and shipping multiplicity audit from overlay 002 are unchanged by this correction. The recorded maximum multiplicity of four describes only that exact source-091 300×200×200 mm, nxPerPitch=8, one-thickness-slice geometry. It is not proof of runtime admission or full shipping-box execution.

## Apply

Apply `mortar-union-four-patch-003.patch` to the frozen source 091 snapshot. The patch includes the overlay-002 changes plus the narrower whole-sweep condition. `manifest.json` binds the source snapshot, exact patch and source files, CPU report, and preserved overlay-002 record.

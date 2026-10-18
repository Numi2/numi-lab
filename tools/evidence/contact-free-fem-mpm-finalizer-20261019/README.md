# Contact-free FEM/MPM deferred finalization regression

This source-level increment extends the existing accepted-step finalizer to immutable, nonadaptive, contact-disabled FEM objects, and permits those objects in the same Runtime as contact-disabled MPM and the vascular graph. It reuses the existing FEM, MPM, topology/scheduler, and vascular checkpoint/rollback kernels. Late rejection also clears the publication bit on each per-object solver certificate. The objectless vascular-only path has no logical per-object certificate entries; the runtime retains a zeroed one-slot allocation sentinel.

The scope deliberately excludes rigid bodies, Human attachments/coupling, active contact, adaptive and mixed/multiphysics solvers, topology mutation, reset, identification, learned updates, and parameter-update transactions. The fixture has no contacts or attachments.

The Objective-C++ fixture uses a synthetic fixed-base tetrahedron and a synthetic damage/history material. It is a rollback-protocol regression, not a tissue-material or physiological validation. In combined mode it adds one active contact-free MPM particle to the same WorldSource. The late gate rejects after 256 microsteps; the fixture byte-compares accepted FEM/MPM/vascular authorities, requires the FEM node and material history to change on accepted execution, and checks same-Runtime retry against a restored clean acceptance. A separate vascular-only mode confirms the objectless sentinel remains unpublished.

## Native receipts

- receipts/native-fem-mpm.json and .log: exact final-source mixed FEM+MPM+vascular run on Apple M6, 256 microsteps per attempt. Rejection status 9, accepted retry status 0, exact rollback/retry parity; accepted node delta 1.65877e-3 m, per-tet material-history delta 5.66092e-6, and active particle-state delta 2.45243e-3 in the encoded state.
- receipts/native-vascular-only.json and .log: objectless vascular reject/retry run. Snapshot arrays contain one inert padded sentinel where the logical count is zero; the solver-certificate sentinel remains validity.w=0.
- receipts/native-build.json: source, runtime library, fixture, material, Metal source, and tested/published public-header hashes. The published header differs from the tested header only in the API documentation comment; no declaration or compiled behavior changed.

## Rebuild

On the same macOS/Apple-Silicon source host, first build a libmetalrobo.dylib containing this branch's matter/src/runtime.mm against the current 207 engine closure and use the pinned Matter Metal source identified in the build receipt. Then compile this fixture as Objective-C++23 against the matching Matter public headers/runtime library, defining NUMI_MATTER_METALLIB to that exact Metal source. Set NUMI_FEM_MPM_NETWORK and NUMI_FEM_MPM_MATERIAL to the inputs in the receipt. Set NUMI_FEM_MPM_COMBINED=1 for the mixed case or NUMI_FEM_MPM_VASCULAR_ONLY=1 for the objectless case; leave both unset for FEM-only. Dispatch through the host's exclusive native-run wrapper.

# Native surface attachment runtime check, SSH Numi, 2026-10-23

The production force owner was published in 38660e86. This follow-up tests it
through the actual Metal Runtime and Human adapter, beyond initialization.

All nine release runner stages passed on Apple M6. One finite barycentric spring
applies 0.02 N initially to one tetrahedron with a fixed articulated body target.
Two accepted 0.1 ms steps deform its free node by 5.99179e-6 m. A forced rejected
attempt leaves accepted FEM nodes, Runtime cursor, and accepted contact history
bitwise unchanged. Restoring the initial physical and adapter snapshots and
replaying both accepted steps reproduces physical nodes and adapter authority
bitwise. The Runtime cursor denotes the accepted frame index, not frame count.

CPU work-transpose and all node-pair tangent finite differences, actual Metal
force/energy/wrench/projection, invalid state rejection, optional source spans,
malformed source validation and fingerprint authority tests also pass.

The retained development failures are harness errors: missing Metal import and
wrong restore result field; unspecified body Jacobian point offset; and an
incorrect test assertion about the accepted-frame cursor. They were corrected
in the fixture without changing production physics. The successful final run
receipt binds the tested source and binaries; older harness logs are not final
source measurements. Compiler warnings are retained without suppression.

This is two isolated mechanical steps (0.2 ms), with a fixed body fixture. It is
not a full-body joint-motion run, moving-bone test, physiology measurement,
performance qualification, or approval of the pending skin mesh. The spring is
finite; this test adds no hidden wrist lock to the Human scene. Full joint skin
transfer, anatomical site correspondence, global bone contact selection, and
integrated motion qualification remain open.

Reproduce using the command and matching fresh library/metallib in the runner
receipt. Rebuild callers with the new C++ layout. Retained profile-039 binaries
remain untouched and must not be mixed with these headers.

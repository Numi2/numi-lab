# Coupled FEM / free BOX operator checker

This standalone checker applies the source-079 production FEM and rigid-contact operator kernels to a one-tetrahedron / one-free-BOX fixture and compares their assembled action with an independent FP64 directional derivative of the complete residual. It includes FEM-only, free-rigid translation, free-rigid rotation, and mixed FEM/rigid directions. The residual includes FEM inertia and internal force, rigid translational and full SO(3) inertia, and equal-and-opposite contact impulse and COM torque. Contact uses the production frictionless BOX inertial-barrier path; the FP64 contact reference comes from the source-077 angular checker, whose helper bytes are also present unchanged in source 079.

The fixture is diagnostic only: it performs zero FEM time steps and does not simulate or qualify cardboard folding, assembly, material calibration, or a full box. The source-authored Neo-Hookean constants are density 1000 kg/m^3, mu 30000 Pa, and lambda 50000 Pa. Thresholds are strict: the FP64 derivative refinement must resolve to 1e-6 relative, and native-vs-FP64 action must be within 1e-3 in the complete scaled norm and separately in both FEM and rigid blocks. A branch-crossing or unresolved reference is inconclusive.

## Binding and build

The production compiler/runtime libraries, shared ABI, and metallib are bound to `/Users/home/cardboard-implementation-20261014/snapshots/source-runtime-diagnostic-return-repair-079` and `/Users/home/numi-cardboard-20261006/build-cardboard-provenance-079`. The independent FP64 BOX/contact helper is included from `/Users/home/cardboard-implementation-20261014/snapshots/source-stateless-compensated-and-validation-admission-077/matter/tools/cardboard_free_box_angular_native_check.mm`. Full hashes and command arguments are in [manifest.json](manifest.json).

Host compilation and linking passed with `-Wall -Wextra -Wpedantic -Werror`. No Metal command was executed for this checker in this package; the parent task owns the native run. The executable embeds the absolute source-079 metallib path at compile time and takes no command-line arguments. Its intended invocation is:

```sh
/Users/home/cardboard-implementation-20261014/analysis/coupled-fem-free-box-operator-077/build/cardboard_coupled_free_box_operator_check
```


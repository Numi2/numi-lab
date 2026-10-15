# Source-370 material-point checker budget-binding correction (112-001)

Overlay 112 corrects one admission bug in 111. The immutable source-370 package is expected to contain a 16-update descriptor, while the checker privately compiles the same medium source and exact parameter words at both 16 and 32 updates. The old comparison required both private descriptors to equal the package budget, so CPU admission rejected the intentional 32-update diagnostic compile. The corrected comparison now separately asserts the expected private budget (16 or 32), the immutable package budget (16), and equality of all other material descriptor fields, parameter words, and initial-state words. The 16/32 fingerprints must remain distinct.

Overlay 111 and its failed admission condition are preserved unchanged at `../exact-source370-point-replay-overlay-111-001/`. Overlay 112 applies only to its checker source file; it does not alter the production solver, shader, CMake target, candidate data, or physical acceptance criteria.

## Bindings

- Parent checker: overlay 111 `source-overlay.json`, SHA-256 `7598760eef248a2ef09c095a765238843827f2128b5cd8279ec4f4ef303f3de0`.
- Parent checker source SHA-256: `2eea53940a37be0b59dd8ddb595b7c3c6e1038d699a62921129edd8aabcf2e2b`.
- Corrected checker source SHA-256: `023806c32a018fe4c136d6a83cd1b1c3e838c8df29e856e3ff3f080ce10a54da`.
- The exact370 package, candidate nodes, accepted state, manifests, medium asset, and source107/source108 bindings remain identical to overlay 111.

## Validation

- Host Objective-C++ syntax-only compilation passed with `-Wall -Wextra -Wpedantic -Werror`.
- Patch dry-run against the exact 111 checker passed.
- CPU admission and native dispatch have not been run by this overlay author. Root should apply 111 plus this correction, build the target, and run `--cpu-admission` before any point dispatch.
- No GPU work was performed. The 32-update private point evaluation remains a diagnostic recompile, not a source-370 full-trajectory replay or qualification.

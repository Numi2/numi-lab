# Per-cell skin/fat mass reaches the FEM solver

The unpaired-volume owner can now transfer a verified parent-mass ledger into child-cell masses. The compiler uses these masses for nodal inertia instead of replacing them with uniform density times reference volume. Material parameters remain unchanged. The owner checks material/donor closure, local density bounds and existing ancestry requirements; split source-donor versus node-frame ownership is still rejected because the current debit consumer cannot represent it.

Source identities and global per-cell mass metadata are retained in compiled fingerprints and optional v20 packages. Legacy empty-ledger worlds retain v19 bytes. GPU structures are unchanged, but host C++ structures grow: rebuild producer, runtime and application together before using this path.

The focused compiler and owner-to-compiler tests pass on SSH Numi. The explicit-mass fixture verifies nodal mass and inverse mass, unchanged materials/topology, invalid inputs, package round-trip and tampering. The legacy v19 fixture is byte-identical to the base writer; the old reader rejects v20 with a recook diagnostic.

This is an executable owner/compiler increment. The actual connected-body candidate has not supplied an admitted conservative reference/mass transfer, has not been activated in the native application, and has accepted no physical steps. Geometry, supported rest, motion, physiology and performance remain separate qualification work.

The canonical nine-file patch is applied to main through an isolated index. The shared checkout, normal HEAD and normal index are preserved. The isolated harness CMake is retained as test evidence only; it does not replace the canonical Matter CMakeLists.txt.

The supplemental regression covers legacy FEM before and after an explicit-mass object, a dormant capacity row, and aligned object identities/global mass rows. All three focused test executions pass. The complete canonical Matter CMake file also generates and registers the new test; this is not a full runtime/application build.

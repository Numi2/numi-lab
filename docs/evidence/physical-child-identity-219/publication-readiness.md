# Publication readiness: base commit 957b85f9cc57cefe2dc182e49e1e90426e23737

## Exact code delta

- A apps/NumiHumanPhysicalSkinChildIntegration.hpp (439 lines): opt-in identity validation, existing-sidecar retention, F32-capture/F64-live exact representation checks and fingerprints.
- M apps/NumiHumanPhysicalSkinChildTetGen.hpp (5-line hunk): retain caller-initialized immutable sidecars while importer populates empty child tables.
- M apps/numilab_human_myosim_visual_probe.mm (66 added lines): guarded environment opt-in, explicit input paths, exact identity import/validation and pass receipt.

The source delta is 3 files, 508 insertions and 2 deletions. The app file alone is exactly 66 added lines against the specified base. The staged evidence adds a compact directory at docs/evidence/physical-child-identity-219/; it contains logs, the harness source, commands, limits and hashes, with no duplicate source tree, native binary, or large input CSVs.

## Checks and runtime evidence

- Fresh numi-human-native build passed with -Werror; final incremental build reports no work to do.
- Worktree and staged-against-base diff checks pass; a fresh external index replayed the code patch against the base byte-for-byte.
- Standalone importer checks pass with actual 7,233-node/28,632-cell parent identity input, exact F32 capture/current and F64 reference semantics, preserved sidecars, and negative node/tet/material/label/partial-path controls.
- Native preparation control 003 passed identity import, existing unpaired remap, and identity installation. It advanced zero physical time and does not qualify changed skin or motion.
- newmesh=false; changed-child O/I/D and node/cell sidecar maps described in scope-and-limits.md are the remaining integration input path.

## Preservation

The real worktree remains at f75955f08d6edf0f85027eac7614017ac9fbc996 with its preexisting dirty state. Before/after status snapshots and real-index SHA-256 files match. Staging exists only in publication-957b85-staging.index and publication-957b85.patch. No commit or push was made.

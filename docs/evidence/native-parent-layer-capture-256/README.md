# Native parent O/I/D source capture 256

This capture adds a source-only snapshot of the existing paired parent: per-node O/I/D world and donor-local coordinates, pinned source identity, skin/fat thickness columns, and the 14 frozen deep attachment tuples. It does not change the physical mesh, mass, simulation state, or runtime setup.

- Parent commit: `87761b99241c2ed3190396d3910ddf74204a29fc`.
- Comparison app source at `b3df9cf` and `87761` is byte-identical.
- Added app source SHA-256: `d7592178d2f997c3854c5efbce9bae819306b17196d7add2ed2f24d19bce60cd`.
- Build: `cmake --build build-native --target numi-human-native -j1`.
- Native object SHA-256: `2d30ff08df380686e0d7c7435e87cf8e6c82c6ac8ed75ba019ad733f8c3acfdd`.
- Executable SHA-256: `c1d912d269547b604a22bb34ec5e47ce81f3e9303b6b213960b631f6123152e7`.
- Capture-only result: exit `0`, physical steps `0`, runtime created `false`.
- Rows: 7,233 total; 2,411 each for O, I, and D; 7,233 distinct compiled parent nodes; 14 frozen D attachment rows.
- Source identities: 2,332 O direct vertices and 79 O triangle embeddings, retaining the pinned payload and layer-coordinate hashes.

The local run directory is `/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-parent-layer-capture-256/run-001`. `receipt.json` contains CSV output hashes. The CSV payloads are retained there and are not duplicated in this evidence folder. Ten pre-existing parent capture outputs were byte-identical to `native-parent-weights-243/run-001`; the per-file comparisons are in the receipt.

This is a parent registration capture only. It does not admit changed child geometry, mass transfer, or physical motion.

# Free-box angular checker reconciliation (074)

This external checker overlay is based byte-for-byte on the sealed 073 checker. It repairs the contact fixture and failure reporting that caused the later native run to stop at `face_pos` without useful sample data. It is not a native result and does not alter production physics.

The fixture now gives the dynamic BOX proxy `adaptiveObjectIndex = NM_INVALID_INDEX`; with the previous zero-initialized value, object 0 was incorrectly treated as the proxy's adaptive owner. The authored BOX point velocity now approaches the outward contact normal, and the host asserts the relative normal velocity is less than -0.49 m/s before calling production. If status, valid-sample identity, or smooth-branch markers are absent, the checker prints status, diagnostics, raw sample fields, FEM node state, projected rigid state, and authored local point; it records failure and continues independent contact cases. The inertia sample uses the same diagnostic behavior and remains fail-closed.

The 073 path oracle, `A = -dR` convention, branch-stability requirements, and FP64 full inertia-chain check are retained. The FP64 derivative and production-versus-FP64 thresholds remain 1e-6 and 1e-3. FP32 finite differences remain separately quantization-sensitive and can be inconclusive. No threshold was relaxed.

`verify_overlay.py` checks fixture invariants and base binding. Objective-C++ syntax-only compilation passed against source-071 headers with `-Wall -Wextra -Wpedantic -Werror`. No GPU execution was performed. Root must build and run this exact overlay before any native claim.

`cmake-cli-option.patch` is a separate test-only patch against frozen 072 `matter/CMakeLists.txt`. It registers `matter.cli.cardboard_graph_option`, running `numi-matter-cardboard-probe --help` and requiring `regional-galerkin-schwarz` in output. The upcoming 073 probe source must be restored from 071 (with its authorized export patch); the test ensures that CLI option does not silently disappear. This test runs no native simulation.

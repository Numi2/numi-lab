# Supported verification 022

This compact package retains the completed CPU preparation and geometry verification receipts. It includes the preparation provenance, input identities, build log, four fixture compile/run pairs, and the compact-boundary and Float32 geometry receipts with their run logs. Test executables, native app binaries, metallibs, and large raw geometry tables were not copied.

Repository base revision and current HEAD both equal c927871bafc1194f8ef314fc0687e4d123d11dd8. All 15 source-file SHA-256 values recorded in preparation-provenance.json still match the current files; no post-receipt source hash change was found among those paths.

The input identity check passes: current NumiHumanWholeBodySkinReduction.hpp SHA-256 matches the recorded header identity (60735d7457e2a8737d60e20cf155aa2fda4f97478072ddef44346bbdd903fe68), and the coordinate SHA-256 matches the compact-boundary receipt (8a82e721cc3800fe2d7293a927ed4d4084e89e4ff260dfc6c44a344bd5b80b2e). All 13 retained artifacts match their hashes in the preparation provenance.

Four CPU fixture compile/run pairs are retained: registered ligament catalog, mixed-donor accepted FEM skin transfer, whole-body skin layers, and precomputed skin/fat layers. Their compile logs are retained even when empty; their run logs contain the result summaries.

Geometry receipts: compact-boundary overall_ok=True, source open edges=45, exposed boundary count=9418; Float32 overall_float32_geometry_ok=1, with zero reported self/crossing pairs and zero nonpositive/degenerate skin/fat tetrahedra. These are CPU geometry checks, not native evolution or physical qualification.

Build log records the application and shader build, including compiler warnings. The preparation provenance records executable SHA-256 83bf98bee369a57a9dd19e74a6aea343cc767486e189d0ff6a2aa6c8982e5b2e; no binary was retained, so this package does not claim a current executable-path rehash.

All retained file hashes are listed in SHA256SUMS. source-hash-verification.json records the per-file current-versus-receipt comparison and packaged-output hash verification.

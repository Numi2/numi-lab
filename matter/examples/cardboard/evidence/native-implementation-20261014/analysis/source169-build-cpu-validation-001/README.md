# Source169 build and CPU validation

This package records a read-only audit of the frozen `source-warm-cgls-refinement-169` snapshot and its isolated build at `build-cardboard-provenance-169`. It does not edit source169 or the root/source168 tree.

The frozen snapshot contains 804 SHA-256-bound source files. The frozen manifest and the build record have identical source maps; the audit recomputes every file hash from the snapshot. The build record is successful and names the probe and `numi-matter-cardboard-warm-start-check` targets. All five declared artifacts (probe, checker executable, and three metallibs) and the two build metadata files match the hashes in `source-build.json`.

CPU checks were rerun and retained here:

- 14 CTests selected with `-L cpu -R '^matter\\.analysis\\.'`; all passed.
- 2 additional Python-only CTests, `matter.analysis.cardboard_resolution_study` and `matter.analysis.cardboard_convergence`; both passed.
- 11 cases in `test_cardboard_solver_environment`; all passed.

The CMake test `matter.metal.cardboard_warm_start_kernel_contract` is registered against `numi-matter-cardboard-warm-start-check` with labels `cardboard;matter;metal;solver;synthetic`, a 60-second timeout, and `RUN_SERIAL TRUE` (source CMakeLists.txt lines 725–728; generated CTest registration also checked). It was deliberately excluded from execution because the test loads a Metal device, dispatches the checker kernels, and is not a CPU test. The checker binary and its dedicated metallib were hash-verified as build artifacts only.

These results validate source/build integrity and the selected host-side CPU tests. They do not establish native kernel behavior, feature-off parity, Matter trajectory behavior, or solver qualification. The frozen source scope itself states that the warm-start refinement is default-off and that no native trajectory or feature-off parity qualification is established. The historical earlier 28-test console-only run is not replaced or retroactively counted as this receipt.

Run `python3 audit.py` to recheck frozen-source hashes, build artifacts, CMake registration, and the recorded CPU receipts. The exact invocation and outputs are in the `*-receipt.json` and `*.log` files.

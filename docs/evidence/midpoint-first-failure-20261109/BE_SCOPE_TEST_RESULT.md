# Precision scope and long-instrument checks

The first material-scope test failed because it changed a compiled material without recomputing the world fingerprint. Initialization correctly rejected the inconsistent world before reaching the precision-mode admission guard. The test now recomputes the canonical fingerprint, and confirms that a corotated material is rejected by the intended precision guard; the scoped Neo-Hookean fixture still initializes. Both the scope test and the stepping/restore regression pass.

The 005 frozen native binary, which adds that guard and per-step temporary-object lifetime management, reproduced all 64 retained BE 002 short-run physical and trace fields exactly after omitting wall/GPU timing. The result, source and binary hashes are retained under the `be-precision-005` prefix. This is an instrument parity test, not a full fine-timestep or contact qualification.


The complete rebuilt contact suite reported 17 of 17 passed in be-precision-005-ctest.log. After CTest had completed, the surrounding shell wrapper exited nonzero because it attempted to assign a read-only shell variable. The log contains the complete pass summary; the wrapper error did not alter the tests.

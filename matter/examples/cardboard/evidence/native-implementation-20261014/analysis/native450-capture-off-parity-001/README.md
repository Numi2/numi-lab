# Native 450/452 CGLS capture-observer parity

This sealed audit compares the complete source142 native run with the bounded CGLS observer enabled (450) against the same run with only the two observer environment variables removed (452). The CPU auditor reports exact equality for all 35 output files, stdout, observations, 64/64 Armijo records, 4096/4096 true-residual records, and the status5 payload. Remaining stderr is byte-identical after only run-path normalization and removal of capture-marker/timing lines. Source/runtime hashes, compiled package, world fingerprint, probe arguments, and prediction environment match.

Both executions have terminal exit code 2 without timeout and end with one accepted step followed by a status-10 rejection. Thus the result establishes observer-history parity for this exact pair, not a complete solve, solver qualification, or physical validation. The CGLS direction’s acceptance remains governed by ordinary Armijo; it does not inherit the original Newton certificate.

The separately sealed reference-binding addendum is `../native450-reference-fingerprint-addendum-001/`. The analyzer source, CPU tests, preflight, and tool manifest are in `../native450-capture-off-parity-tool-001/`.

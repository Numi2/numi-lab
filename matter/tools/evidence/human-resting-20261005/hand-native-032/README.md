# Native rigid-digit reduction and long release diagnostic

All execution and exact geometry tests ran on the SSH Mac mini, Apple M4 Pro.
`manifest.json`, invocation records and source diffs retain exact inputs,
binaries, configuration and recording hashes. The unretimed native recordings
remain at the absolute Mini paths in the manifest. Compressed files are
lossless copies, with both source and retained hashes recorded.

The explicit rigid-digit mode completed six simulated seconds in 192.362
wall seconds (0.03119 times real time), retaining all source muscle/tendon
state, body masses and free wrists. All forty selected q values remained
exactly at reference and their final velocities were zero. Rejection/retry
preserved accepted history and matched uninterrupted replay. Complete exact
outer-skin audits at accepted steps 639, 2783 and 2999 found zero forbidden
crossings. Those three states do not qualify an entire continuous run.

The prior contact-driven release run completed 320 simulated seconds in
3,112.067 wall seconds (0.10283 times real time), with 59 breaths and 373
complete cardiac filling/ejection cycles. It remains anatomically failed:
the final full-skin audit found 1,601 hand-involved and eight radius-region
crossing pairs. Late lateral center-of-mass drift was about 0.0347 mm/s.
Known lung/rib and ventricular-wall defects remain in both runs. Neither
run is final physiological, anatomical or clinical qualification.

`audit-resting-whole-skin-032.py` failed to replace the prior run path and
therefore audited the old 031 states at 639 and 2783 before failing at absent
step 2999. Those outputs are retained under 031 only. The corrected 033
wrapper explicitly selects the run and validated all three actual 032
accepted receipts. The original successful interpretation was withdrawn.

The 32-step integrated profile attributes about 19.07 ms per step to the
91-row equality factorization and 24.52 ms to the constraint finish. These
are short profiling samples, not a final throughput qualification.

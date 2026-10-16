# Publication 030 independent byte audit

This audit independently reads the publication binding, current and historical summaries, prior/current NUL pathspecs, all 43 newly listed external case directories and their repository payload copies, plus the source173 frozen manifest and original host build metadata. It does not modify those inputs or the Git index.

The audit passed: 2,278 payload rows were verified (2,202 raw copies and 76 gzip copies), the exact historical 1,524-record summary prefix and prior pathspec byte prefix remain intact, the expanded pathspec contains 10,656 unique entries, and all 804 source173 hashes match both the frozen snapshot and current checkout. All seven source173 build artifacts matched their recorded hashes. The publication keeps the coupon/full-cycle and physical-assembly claims unqualified; 476/477 native results are excluded.

The audit verifies payload bytes and source/build provenance. It does not verify Git staging or remote commit state, does not establish physical qualification, and does not treat textual references to excluded run IDs as included run results.

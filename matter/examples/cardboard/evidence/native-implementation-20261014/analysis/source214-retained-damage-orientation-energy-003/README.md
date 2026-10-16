# Source 214-003 reproduction and provenance correction

This package corrects the build-lineage ambiguity in 214-002 without changing its tested C++ source. The 214-002 manifest named canonical `source-composed-strict-periodic-freebox-210` as its base, while its receipt compiled from `source-composed-strict-periodic-freebox-210-buildfix-003`. The buildfix-003 snapshot's checker TU was verified byte-for-byte against the canonical base hash (`5d54910a…44d6f`), but that receipt still did not prove compilation of the modified candidate TU.

A new full source snapshot was made with APFS CoW from buildfix-003; exactly `matter/tools/cardboard_retained_matrix_damage_check.cpp` was replaced with the tested 214-002 candidate. Its 819-file frozen map differs at that one path. The child copy's inherited empty `build-cardboard-native-worker/worker.lock` was absent from the parent's frozen map and had no open handle; only that unbound copied lock file was removed before freezing the child. The parent tree was not changed.

A fresh CMake build from the new snapshot rebuilt the supporting compiler library and targeted checker. `reproduction-commands.json` binds configure/build/run argv, compiler and CMake versions, the generated compile database entry naming the actual candidate TU, the exact link command, source map, built object/library/executable hashes, logs, and passing test output. This is a synthetic CPU implementation regression only: no GPU execution, physical FEM transaction, or calibration claim.

Use the `configure.argv`, `build.argv`, and `test.argv` arrays from `reproduction-commands.json`. Do not substitute the canonical base or buildfix-003 checker TU for the candidate path. 214-002 remains unmodified.

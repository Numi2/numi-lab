# Final integrated study readiness v017

v017 fixes the direct-native recorder's source-pin return-contract crash. The frozen package is the versioned set of scripts, tests, revision record, and test logs in this directory. It keeps the 310-second-per-arm protocol and the existing half-drive intervention window at 60-100 seconds. No physical study registration or completed paired result is included.

The exact v016 failure is retained here as actual-v016-recorder-failure.stderr plus its execution record and original launch arguments. The pinned v016 recorder assigned two names to source_pin_files, while the pinned preparation owner returned four values. The v017 recorder consumes that four-value contract. post-package-status-v017.json binds the v016 and v017 attempts and their runtime metadata by SHA-256.

The first v017 test run used the preparation environment's Python 3.13 and failed a strict macOS SDK platform identity check. That log remains alongside the final test log. The corrected actual-scene plan-only regression invokes the pinned Xcode Python 3.9 and passes with 51 tests plus 19 subtests. revision-017.json reports the package-freeze checks; the separate post-package record reports the later native attempt.

After v017 was frozen, the recorder launched the actual 20-second identity probe. The run was stopped after an independent exact native-pose scan found three row307 self intersections at accepted step 0. The attempt ran for 68.27 seconds, confirmed the expected runtime image and unchanged source/assets, but ended by SIGINT with no completed identity validation. It observed no intervention response; the configured intervention lies outside the 20-second horizon. This is not a passed probe or an accepted scene.

The v016 publication is preserved verbatim in its original path. README-v016-publication.md and SHA256SUMS-v016.txt retain its publication-level summary and checksum list. This v017 evidence directory is a scoped publication, not a self-contained copy of all external owner sources or assets; consult the absolute paths and hashes in its revision and status records.

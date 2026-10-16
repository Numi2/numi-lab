# Current status erratum

The earlier `README.md` was written before native command 004 launched. Command 003 was refused before native launch because it selected the source154 executable with source157 runtime artifacts. Command 004 was later launched at 00:54 UTC and is an active native468 run. Its solver case and launch guard are deliberately excluded from the recovery index. The index records the prepared command packages and their small input aliases only; it does not archive an active run or imply a completed result.

The source157 executable and metallibs are not copied into this recovery-index package. Their source-build artifact SHA-256 values are recorded in `external-recovery-index.json`; command004's active case remains outside this index. Rebuilding from the retained source157 snapshot is required before a future replay if those build outputs are unavailable.

The original README, recovery index, and build receipt are preserved byte-for-byte. This addendum records status after command004 launch.

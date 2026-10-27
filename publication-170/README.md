# P170 connected-surface construction evidence

P170 is a static construction checkpoint on top of P169 commit ff83a29f52cc23e6a51839af0b1e818f20103d3f (manifest SHA-256 c35db7825bcc1586eee18e456f2309297719fb7a3373398e0cf980f3ec95d4cf). The package keeps the tested 602-v2 and 580 construction lineage, frozen 619 run-004 evidence, the v7 serialization fix and its tests, plus recovery records for the recent P168/P169 publications.

## Construction results

- 602-v2 records the 1043 surface checkpoint; 580 run-005 records 1002, with zero contained points and zero self pairs.
- The 580 run-006 attempt ended on ENOSPC before it wrote a candidate. That failure is retained.
- The recovered-space 580 run-007 candidate passed the exact full construction check at 878 surface pairs, zero contained points, zero self pairs, 185 banks and 237 anchors. Candidate SHA-256: 699727ec76fa4d7c30448e42382a5b3052e8a5c8029ffcd85b58e592557b0faf.
- 619 run-004 retains its strict 901 checkpoint and all failed/partial evidence. Its coordinator stopped while serializing a PosixPath after finalization. The later v7 source changes that output to a string; the combined v6/v7 suites passed 22 tests. The package includes a recovery receipt, not a claim that v7 reran the geometry.
- Large run-004 fit JSON reports are represented by lossless zstd archives with a fresh restoration proof; original source reports remain present.

These are construction-only results. Native admission is false and accepted physical steps are zero. The 45 pre-existing head boundary edges remain open; skin/fat physical transfer, cell positivity, motion, and native qualification are not complete.

## Recovery provenance

P168 and P169 fit-report archives were freshly downloaded from GitHub and restored before their plaintext source reports were retired; the remote recovery receipts are included. The 18 obsolete alternate Git index cache files were streamed to a host backup and all 18 were freshly restored from it on numi before retirement. Their tar.zst backup, manifest, restore, and execution receipts are included. The P169 push receipt links to the immutable publication.

The manifest lists the exact payload paths, sizes, POSIX modes, and SHA-256 hashes. P170 publication state is recorded by its separate push receipt.

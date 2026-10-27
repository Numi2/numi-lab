# P161: offline construction evidence through 1,595 surface pairs

P161 packages a construction-only checkpoint with **1,595 surface pairs, 0 contained point/bank pairs, 0 self pairs, 185 complete banks, and 237 typed anchors**. Candidate SHA-256: `adafeec98c87ff305846fd9d4c1963bed399d7cf1eeb33adffc977bfa8ac67bb`. Its exact full-check report is under `evidence/surface-clearance-546/scalar-retention-590/raw003-paired-001/regional-552/final/report.json`.

The earlier P160 checkpoint was 1,697/0/0. The completed 584 run improved it to 1,601/0/0 in round 002. Its round-003 fit remains partial and nonaccepted; all three rounds, including partial files and failure/telemetry records, are retained. The six 41 MB fit logs/reports are stored as lossless zstd payloads under `evidence/compression-preservation/584-run-001`; the receipt pins exact original paths, SHA-256, byte counts, mode and mtime. Originals remain in the source tree. Restore any file before workflows that require its plaintext path with the recorded helper command.

Other branches are kept with their actual outcomes:
- 588 raw paired branch: 1,695/0/0, a separate unselected checked proposal.
- 585 edge-only pilot output: 1,710/8/0. The 1,697 number is the input branch count, not the pilot output; its 8 contained pairs keep it unselected.
- 589 v1 produced eight certified outward target endpoints only. No candidate geometry run or full candidate check is represented by those endpoint records.
- 590 retained 582/global branch: 1,619/2/0 (negative); 552 regional retention branch: 1,595/0/0. The evidence preserves the earlier 590 serialization failure and unknown LP telemetry. The result is attributed to the regional-retention workflow; this package does not attribute the gain to the alpha range 0.99718 to 1.0.

The read-only 588 orientation audit and 586 v2 unit tests are included. The historical source-orientation dependency is copied byte-for-byte from the local P095 package member verified against its P095 manifest; the provenance record names the older alias and actual member path without claiming a server-path verification. Focused test receipts are under `test-receipts/`:
{
  "586v2": {
    "returncode": 0,
    "seconds": 0.09897720790468156,
    "log_path": "test-receipts/586v2/pytest.log",
    "log_sha256": "a4ce3c4d2d126cf54794b588fe0568dcf174b80e68913316f73e0da7b9f1c64b"
  },
  "588damping": {
    "returncode": 0,
    "seconds": 0.09422104200348258,
    "log_path": "test-receipts/588damping/pytest.log",
    "log_sha256": "0e96d7ca7406320840ee2e704cbada03e2fe4b6d6aea1d667b4f685b07e7e24d"
  },
  "589v1": {
    "returncode": 0,
    "seconds": 0.08980133407749236,
    "log_path": "test-receipts/589v1/pytest.log",
    "log_sha256": "ef0e7ef49e71fa999842be0cb8d22ec1058d3a384ed5c0a2ec6efb8f434975ac"
  },
  "590v1": {
    "returncode": 0,
    "seconds": 0.21694358414970338,
    "log_path": "test-receipts/590v1/pytest.log",
    "log_sha256": "dd16cbdbb2cff39a7ff41edae14420833d1d368adba1f7523723a58e363ea28d"
  }
}

The prior P160 source-pin index and exact P160 manifest/commit reference are under `evidence/source-dependencies/`. Newly recovered P160 text mirrors and the cache-cleanup inventory receipt are under `evidence/recovery/`.

This remains static construction evidence. Native admission is false, physical admission is false, accepted physical steps are zero, and the zero-surface goal is not achieved. P161 is staged for review only; no commit or push was made by this builder.

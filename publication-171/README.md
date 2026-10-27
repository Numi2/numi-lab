# P171 connected-surface construction evidence

P171 extends the verified P170 prefix at commit `c1390f7f33f4bd90e5cf2306f137666e62eb5089` (manifest SHA-256 `e6a9572750d02b438e776d04a3cd20b5cb3a6fc9d1a685ee8fa791452261d0a4`). It records two completed rounds of the stopped v7 run-005, the tested v8 source and unit suite, one extra-cut diagnostic, the second obsolete-index-cache recovery archive, and P170 publication/recovery receipts.

## Frozen construction evidence

- v7 run-005 round 001 has a strict full construction check at 875 surface pairs, zero contained points, zero self pairs, 185 banks and 237 anchors. Its candidate SHA-256 is `0c455299a2809c678892623ffa9268a200976f8ba33279773e7f4ac322305fbe`.
- Round 002 has a strict check at 796 surface pairs, zero contained points and zero self pairs. Its candidate SHA-256 is `1a79cf4051e61e2c92124958b90098e6e6bd509f0424118fa85cad5bce69fd5a`. These are trial checkpoints, not promoted main state.
- The outer run stopped during round 003 after the disk guard reported 182,734,848 available bytes against a 262,144,000-byte minimum. Its execution receipt records return code 1 after 772.83 seconds. The partial/stopped batch and all earlier failed candidates remain in the package.
- The 630 cut-003 attempt repeated candidate SHA-256 `fee2e3fb255155feaeec0bf73ffe4421f588617fd40354eca136fc9ad07f2d67` and retained 15 self pairs. It did not produce a passing construction step.
- The tested v8 source and test are included with the root test launch, execution and log: 17 tests passed. P171 does not claim a completed v8 geometry run; active run-006 and its outputs are excluded.

Six fit reports larger than 15 MiB are preserved as lossless zstd payloads. Each archive was restored into a fresh temporary directory and checked against the original SHA-256, byte count, mode and modification time. The plaintext source reports were still present at packaging time; no source report was deleted for this package. Restore them with the helper and the receipt under `evidence/compression-preservation/run-005-and-630/`.

The second alternate-index-cache record contains seven archived cache files and a fresh filesystem recovery proof. P170 commit/remote verification receipts and its independent three-report retirement proof are also included.

The known 45 pre-existing head boundary edges and two loops remain; P171 does not claim whole-skin closure. All included results are construction-only: `native_admission` is false and `accepted_physical_steps` is zero. No physical, anatomical, or native qualification is claimed. `publication-manifest.json` lists every package member except itself with exact path, size, mode and SHA-256.

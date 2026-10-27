# P162: offline construction evidence through 1,577 surface pairs

P162 records a **static construction-only** result. The best exact full-check construction in this frozen scope is **1,577 surface pairs, 0 contained point/bank pairs, 0 self pairs, 185 complete banks, and 237 typed anchors**. Candidate SHA-256: `05b499db61f22492f4ee4611951bdb4e02e196a3175205dda862bc08befc7729`. The matching finalization and full-check reports are both retained. The state records `candidate_selected: false`, `native_admission: false`, and `accepted_physical_steps: 0`.

The previous published construction checkpoint remains P161 at 1,595 surface pairs. P162 does not mark any candidate as physically selected. The zero-surface goal remains open.

## Included evidence

- P589 source-outward target generation retains both the earlier source-outward run and the v2 depth-maximizing target proposal, per-run records, and test sources. Its local orientation provenance is pinned to the exact P095 audit snapshot at `e43908188f134182635ac36a9a1602abad1918df374a04f3f7cc9eb1812d7497`; the package records a local source path and does not infer remote presence from that path.
- P590 v2 retains the original target and alpha, regional 552 output, full 545 report, recovery receipt, owner/test sources, and run launch/execution. P591 retains the mixed-target selection, emitted proposal, regional retention, full-check outputs, and the original failure evidence.
- P592 area-blocker arithmetic and topology search records remain, including the failed/incomplete search candidate and logs. P593 preparation artifacts retain depth, guard refresh, surface targets, and exact source input receipts. P594 retains the crossing diagnosis. P596 retains the connected topology checkpoint, lineage, search, and full-check evidence.
- P597 preserves v1, v2, and v4 target proposals, source snapshots, recovery verification, and all focused test receipts. The included P601 comparison aligns 1,585 pair IDs and bitwise checks face rows, barycentrics, inside samples, bank identifiers, and raw nearest directions. It distinguishes proposal cosines from directions measured at emitted F32 endpoints. V4's nearest-angular method retains nine source-only fold exceptions; 1,576 proposals satisfy both 0.05 proposal caps, while 898 emitted F32 endpoint directions meet both caps.
- P598 retains the completed run002 proposal and 1,580/0/0 retained-regions checkpoint after regional retention, compatibility-v4 reports, and all run inputs. Its 41 MB fit-stage log and 41 MB fit report are stored as lossless zstd archives with original SHA-256, byte count, mode, and mtime. The P161 restore helper was run against the fresh P162 archives into a temporary directory; both restored content hashes, sizes, modes, and mtimes matched before packaging.
- P599 retains the 1,577/0/0 component topology proposal, full exact check, finalizer, lineage, search, tests, and failed or unselected branch evidence.
- `source-pin-index.json` records discoverable Python source hashes and whether each source is in P162, an exact P161 member, or still present at its original path. `prior-publications.json` pins the P161 parent chain. `evidence/prior-publications/publication-095/source-path-orientation-audit-112.json` is an exact local evidence snapshot.

## Recovery of the two compressed P598 files

From an empty restore directory, run the included helper as follows, using the archive directory shown below as `--compressed-root`:

```sh
python restore_compressed_text_mirrors.py \
  --receipt compression-receipt.json \
  --destination-root /path/to/empty/restore \
  --compressed-root payloads
```

The receipt gives both exact original P598 paths, their original hashes and metadata, and the corresponding compressed archive hashes. The two original plaintext files remain at their research paths until the remote P162 manifest and archive blobs are independently verified and a fresh consumer check confirms they are unused.

## Limits

No geometry solver or native physics job was run by the publisher. P162 does not establish skin/fat transfer, tissue motion, physical or native admission, accepted physical steps, or completed anatomy. All construction and test evidence is offline and must remain separate from physical qualification.

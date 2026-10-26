# Offline connected skin construction checkpoint

This package snapshots completed offline geometry work through **hard-joint-fit-016**. It retains the completed trial directories, their inputs, launch and execution receipts, reports, witnesses, failed attempts, and source snapshots. `topology-017`, later numbered directories, the active `topology-builder-018.py`, the unrun broad preparation script, and Python/pytest caches are outside this snapshot.

## Recorded construction counts

These are construction diagnostics from the named outputs. The rows compare different branches and objectives; they are not physical qualification results.

| Checkpoint | Remaining contained points | Reported surface crossings | Exact self pairs | Note |
|---|---:|---:|---:|---|
| Initial state | 251 | 2,895 | 0 | starting candidate |
| round-002 | 170 | 2,832 | 0 | connected point-priority batch |
| round-003 | 160 | 2,831 | 0 | continuation |
| round-004 retry state | 150 | 2,827 | 0 | best surface branch in this series |
| endpoint-batch-004 state-002 | 19 | 2,862 | 0 | 31 more surface crossings than round-003 |
| greedy-endpoints-008 state-12 | 12 | 2,877 | 0 | alternative endpoint branch |
| local-patch-009 state-6 | 6 | 2,889 | 0 | alternative local patch |
| spatial-patch-012 state-5 | 5 | 2,861 | 0 | alternative spatial patch |
| connected-cones-014 selected state | 3 | 2,863 | 0 | selected point-count branch; candidate SHA-256 below |

Selected candidate: `connected-cones-014/15-270-spatial/candidate.npz`, SHA-256 `215fa7da9bc9c3fa2a189b853343282e21403ef97c1d691a2a28cab36563ea82`. Its retained point summary reports 3 remaining contained points; the retained surface report covers all 185 banks. The report has `native_admission=false` and zero accepted physical steps.

The three remaining witnesses are nodes 18495, 19702, and 20182 on body 43 against bone 16. The read-only ancestry diagnosis is retained at `diagnosis/remaining-three-topology-ancestry-001.json`; it reports their source-map support, nearest typed-anchor relation, and local-motion receipts. It did not run a solver or a whole-geometry audit.

## Provenance and limits

The construction used 237 typed anchors with unchanged verified CSR lineage; the actual source cap is 12 mm, the minimum-area ratio is 0.1, and the edge-gradient cap is 0.5. The 45 pre-existing boundary edges remain; this does not certify a closed skin surface. Failed 191-self-pair and 50-self-pair fits, the hard-point-007 infeasibility, and the hard-joint-fit-016 infeasibility remain in the snapshot with their reports and witnesses.

This is offline construction evidence only. It does not establish completed anatomy, physical skin behavior, finite-element admission, native execution, or visual quality. No visual inspection, new whole-geometry audit, native run, or GPU geometry job was performed for this publication. Existing exact point and surface reports are preserved as recorded evidence and retain their original qualification fields.

For `full-point-005/retry-fit/report.json`, the reported runner SHA-256 was `c2d94f6428acd54cd9c3ad0f22136499b109baa9eb26081b6b50791a54b3b729`. The matching historical runner source was reconstructed in the publication snapshot by removing the later active-adapter branch from the current source; the resulting bytes match that recorded hash exactly. The reconstruction and external 485/490 wrappers are recorded in the manifest.

## Lightweight tests

`/Users/n/numi-human-prep-venv-20261005/bin/python -m pytest -q test_point_batch545.py test_direct_depth545.py test_connected_patch545.py test_project_patch545.py test_component_active545.py` passed **8 tests**. The UTC timestamps, source hashes, output, and shared-checkout state receipts are in `tests.json` and `tests.log`.

`publication-manifest.json` contains SHA-256 and byte size for every included payload file. Six large stdout JSON mirrors were omitted only after checking that each parsed JSON payload is an exact field-for-field subset of its retained complete structured report; their original path, full SHA-256, byte size, matching report hash, and process-slot prefix are recorded in the omission list.

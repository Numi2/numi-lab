# Connected skin/fat construction and exact bone audit

The maintained entry point is `matter/tools/human_skin_fat_bone_audit.py`. It accepts an explicitly hashed candidate, typed node IDs/topology, frozen bank export and optional matching native query. It rejects a stale node export and reports exact route witnesses; `--all-tet-edges` enables the broader edge check. The canonical focused suite has 11 passing cases on SSH Numi (Apple M6). No full candidate edge scan is claimed here.

Payload copies are frozen execution evidence, not additional maintained simulators. The corrected bank-local audit source is preserved separately from older failed reports. Older pre-fix source bytes with hashes 1e0575/e4dfe2 are unavailable; their report hashes alone do not establish source recoverability.

Candidate 022 passes its 20 local exit constraints but retains 16 route/bone intersections and 17 inside node/bank pairs. Native query 023 intentionally stops before physical simulation. Its exact emitted nodes match candidate 022; the initial-node CSV is pre-override baseline and is rejected for candidate matching. Candidate 024 retains positive cells but makes no repair progress: 48 of 85 construction rows fail. A phase-I LP isolates a conflicting deep-column nearest-exit construction row. These are retained failures, not anatomy acceptance.

No reference/mass transfer, physical activation, accepted timestep, native visual confirmation or throughput qualification is claimed. The inherited 061 bank archive and 063 baseline-depth evidence remain required dependencies; the identical 146 MB triangle export is not duplicated.

Run the canonical test with the existing Numi Python environment:
`python -m pytest -q matter/tools/test_human_skin_fat_bone_audit.py`.

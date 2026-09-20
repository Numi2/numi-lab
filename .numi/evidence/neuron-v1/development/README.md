# Development observations

These compact JSON records are retained negative engineering observations from
pre-canonical Potter protocol iterations. They are **not** `.ncrun.json`
manifests: those early binaries did not record complete repository, metallib,
accepted-state, checkpoint, or runtime identity. Their filenames deliberately
use `.observation.json` so tooling cannot mistake them for promotable evidence.

They document parameter and schedule exploration only. The final promotion
gate consumes a separately generated, fully identified 3-seed × 5-mapping
qualification record and never uses these values to relax its fixed threshold.

`protocol-v6-partial-equilibrium-map0-negative.observation.json` is the first
same-checkpoint comparison after exact 200--400 ms triangular RBS timing and
the corrected ablation definitions. Adaptive PTS trailed PTS-off by 1.67
percentage points, so the partial state was not promoted or used to justify
expanding the qualification matrix.

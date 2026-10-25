# Coupled layer trials 128

Only frozen runs 002 and 003 are retained. Both are construction-only on a partial hybrid outer surface. Neither ran TetGen or created material/mass ancestry, and neither is a native or physical qualification.

Exact Float32 self-pair counts (O, I, D, skin shell, fat shell): run 002 = 0, 144, 1,790, 372, 2,460; run 003 = 0, 0, 55, 428, 274. Thus run 002 fails across all non-outer layers/shells; run 003 still fails D and both shells. Run 003 uses baseline offsets and makes no active inner/deep reconstruction.

The common hybrid O input is copied and the 115 candidate / capacity-field / accepted coarse-face data are pinned to the publication parent. Run 003's exact builder is included. Run 002's execution manifest names source hash 14df3352… but that exact script snapshot is unavailable; the included builder is the run-003 source and is not claimed as an exact run-002 reproducer.

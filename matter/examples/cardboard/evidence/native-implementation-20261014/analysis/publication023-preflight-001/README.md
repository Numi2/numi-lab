# Publication 023 preflight record

This directory preserves the exact source-139 build metadata and preparation, build, retention, and audit commands needed to interpret the selected cases. The metadata reports a successful six-target build against the sealed source-coupon-cache-admission-139 snapshot. The cache-admission integration states that the source-139/cache-on full attempted history matches source-133/cache-off history and that both stop at Newton 48; the cache test is not a completed coupon qualification.

The repeated-Cauchy coupon 438 is retained as an input-seal preflight failure (`execution_started=false`, duplicate external input basename). The later 442 run is retained separately: its 64-Newton attempt fails, and no full-cycle convergence or assembly result is claimed. The older 442 adjudication 001 remains present as superseded evidence; adjudication 002 is the current interpretation.

The prepared case list and inventory hashes are in `publication023-cases-prepared.json` beside this directory. This preflight bundle deliberately excludes build trees and source snapshots; the case archive includes the retained source snapshot bundle and its hash inventory.

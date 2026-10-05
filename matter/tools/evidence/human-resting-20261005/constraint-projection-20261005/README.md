# Native source-limit projection regression

This evidence covers the scalar source-position limit row used by the existing
Metal standing solver. When a q coordinate was outside the existing 16-FP32-
epsilon deadband, the row corrected only 20% of the remaining error per step.
Over 30 simulated seconds this left three independent coordinates just beyond
that deadband. The fix now aims the same unilateral generalized-impulse row at
the next representable q value toward the legal side of the deadband. The
deadband, source bounds, impulse owner, and existing 4 rad/s correction cap are
unchanged; there is no post-step clamp or separate force.

The focused CPU regression passes 83 checks on the Mac mini. It covers both
lower and upper limits, including the FP32 integrated-q value for one-ULP
overshoots. The `numi-human-native` target then built successfully with CMake;
two existing unused-variable warnings in `MujocoMuscleReference.metal` were
reported.

The fixed native run is
`native-constraint-projection-30s-nextafter-001`. Its immutable invocation
SHA-256 is
`7df38a44548be100a97bc60a1a796f029d23be1cf5161ec95ee5eacbe9c2c4da`.
It accepted all 15,000 steps at 2 ms (30.0000014 simulated seconds) on the
24 GB Apple M4 Pro Mac mini. Native horizon wall time was 299.852 s, or RTF
0.10005. The in-run rejection/replay check passed, physiology and body clocks
matched, and root assistance was false. All six runtime metallibs, the native
executable, consumed assets, constraint header, and its Metal include stayed
unchanged across the run; exact hashes are in `invocation.json` and
`execution.json`.

The full source-revision field in this immutable invocation is stale: it was
inherited from an earlier run, and the Mac mini source mirror has no Git
metadata. Do not use that field or the inherited `source_file_sha256` map as a
complete revision claim. `source-revision-reconciliation.json` records this
discrepancy. The fixed helper, its Metal include, focused test, native
executable, dylib, all six metallibs, and consumed input assets are identified
by exact hashes in the pre/post-run receipts.

The terminal audit checked all 118 enabled scalar source ranges and all 51
NHEQ equalities against the retained native q. No coordinate exceeded the
existing 16-epsilon band. Thirty-six coordinates were outside their exact
source interval by more than 1e-9, but all remained within that pre-existing
numerical band; the maximum raw interval error was 1.803e-6. The maximum NHEQ
residual was 4.37e-10. Thus this run demonstrates the numerical band and
equality checks, not mathematically exact source-range containment or clinical
validation.

The 470-sample full-skin audit had no nonfinite vertices and no vertex more
than 1 mm below the bed. Maximum functional-volume relative error was
1.086e-6. The minimum skin/bed gap remained -0.111 mm, the known initial
intersection limitation. The continuous 30-second native recording is retained
on the Mini at
`/Users/n/numi-human-resting-evidence-20261005/native-constraint-projection-30s-nextafter-001/native-viewer.mov`
with SHA-256 `c36ac403d771fee6acfaa5d6d7982955e53fccc5d78a53c43cd00fe19e52f0fe`.

The earlier `native-constraint-projection-30s-001` run, before the one-ULP
inward target, is retained as a failed/inconclusive attempt: the native log
shows all 15,000 steps completed but two coordinates were still one FP32 ULP
beyond the deadband. Its post-run receipt writer raised a `KeyError` while
assembling hashes and lost the child exit code; `receipt-writer-failure.txt`
preserves that error and `execution.json` leaves the return code unavailable.
That exploratory result is not promoted as a passing run.

Compact receipts for the 1-second preliminary run, the inconclusive run, and
the fixed 30-second run are in their respective subdirectories. Large logs,
CSV traces, state packs, and native recordings remain on the Mac mini at the
paths recorded in each invocation.

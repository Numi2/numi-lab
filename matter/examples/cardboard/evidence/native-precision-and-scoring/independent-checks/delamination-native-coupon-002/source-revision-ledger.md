# Native Mode-I coupon source and failure ledger

This ledger accompanies the passing receipt in this directory. It preserves
failure outputs as copied captures and identifies the limits of the historical
source binding. The older evidence tree at
`/Users/home/cardboard-evidence-20261006` was not modified.

## Failed native attempts retained

The original stdout captures are copied byte-for-byte into `prior-failures/`:

- `delamination-native-debug-001.log`: the source-seeded positive-opening state
  failed the first native admission gate near the `F33` validity boundary.
- `delamination-native-debug-002.log`: validity and state projection passed,
  but the generated stress path failed. The then-current energy expression
  produced an FP32 division denominator below Matter's guard.
- `attempt-001.stdout.log`: the first source-traction comparison failed at
  `392977.47 Pa` versus `393030 Pa`; the seed stretch is represented in FP32.
- `attempt-003.stdout.log`: the native algorithmic tangent comparison failed.
  That capture did not retain enough numerical detail to reconstruct its
  source revision or precise error.
- `attempt-004.stdout.log`: the native tangent differed by relative error
  `0.411088` at loading step 64 (`-840116 Pa` versus `-1426555.300381 Pa`).

These are real failed checks; they remain visible alongside the pass. Their
log files do not contain source hashes, and the corresponding earlier `.nmatter`
source snapshots were not retained in the original evidence capture. Therefore
those failures cannot be bound to a recoverable exact source revision. No old
source revision has been reconstructed or fabricated.

## Passing source and executable binding

`source-final/` is a copy of the exact final material asset, material metadata,
CPU source oracle, native Metal kernel/checker, plot renderer, and Robertsson
architecture assessment used for the closure package. `build-artifacts/`
contains copies of the native checker executable and its specialized Metal
library. `SHA256SUMS` hashes every file in this bundle except itself. The
passing test output identifies the compiled material by fingerprint
`5570626462714653447` and reports zero physical FEM steps.

The concise and full CTest outputs are `ctest.log` and `LastTest.log`; the
native measurements and scope are summarized in `receipt.md`. The source-oracle
CSV and SVG are analytical reference trajectories, not native solver data.

# Retained native experiment evidence

Host: ssh numi, Apple M6 Mac mini 16 GB, 7 November 2026.
Both experimental implementations remain disabled. See the owning report:
[FEM–plane AL](../../FEM_PLANE_AL_EXPERIMENT_20261107.md).

`source-identity.json` pins the base commit, current source and native artifacts.
`final-tracked.patch.gz` and `final-source-files.tar.gz` retain the final code.
Older candidate patches/instruments bind their own observations where captured;
they must not be relabelled as the final candidate. Native binaries are hashed,
not checked in. Rebuild on an Apple Metal host with the report commands.

Sealed plans precede their respective revisions. Attempts, failures and raw
per-step or per-sweep observations are retained. JSON is authoritative;
`.stderr.gz`/`.log.gz` retain diagnostic streams. A runner or geometry-test pass
is not a physical-accuracy or production pass. No qualified throughput ratio
is available for a failed/incomplete physical comparison.

`candidate001` retains the lagged multiplier update. `candidate002` retains
corrected working-state updates before complete instrument checks. `final`
retains the loose-tolerance series (including nonzero-dual replay); `tight`
is the final 1e-9 fixture tolerance with qualified local momentum accounting,
all rollouts incomplete. The primary reference and force gates failed.
Near-contact starts and the initial upward-motion calibration are retained.
The initial calibration lacks exact source/binary hashes; do not use current
source hashes to fill that gap. Compiled source-physics fingerprints are
recorded per run; the plans did not archive independent package bytes.
Late near-start timings in candidate002 overlapped a CCD test and are excluded
from timing interpretation. Final and tight series had no competing GPU trial.

`calibration002-al.json` is intentionally empty: initialization rejected the
unsupported mixed-FEM model. Its attempt note and stderr retain the failure.

# Retained native experiment evidence

Host: ssh numi, Apple M6 Mac mini 16 GB, 7 November 2026.
Both experimental implementations remain disabled. See the owning report:
[Interval-local CCD](../../CCD_INTERVAL_LOCAL_EXPERIMENT_20261107.md).

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

`candidate001` retains the zero-horizon integration error. `candidate002`
retains the pre-fallback-fix results, including its accepted rotating swept hit.
The root `scale*.json`, `world-steps.json` and analyses are candidate003.
`world-spherical-inertia-calibration.json` is an earlier invalid inertia
calibration, not the final thin-box experiment. Its instrument is retained.
Root full-step data uses correct box inertia, includes the unused 177 rad/s
rotation, and identifies accepted unresponded swept hits in the legacy arms.

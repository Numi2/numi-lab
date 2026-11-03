# Build/startup and measurement cleanup — 2026-11-03

Production BUILD_TESTING=OFF now excludes executables ending in -check from the default all target in the owning root and Matter directories. All check targets remain explicitly buildable; BUILD_TESTING=ON retains ordinary test builds. The existing production graph changed from 703 commands with 126 check targets to 439 commands with no check-target compilation. Testing-enabled configuration retains 134 check targets.

The two tiled scheduler pipelines compile only when requested. Disabled-path cavity integration checks and the enabled full Human prefix pass. No correctness check or physical gate was removed.

Historical source snapshots in solver-finalization now use six canonical archives and relative links instead of 18 independent archives. Every decompressed archive was verified identical; original gzip bytes were checked against immutable commit fc8ca33230895b1aeb0c831964e0eaa5403206f9 and remain recoverable there. The snapshot manifest records both identities. The current tree stores 20.3 MiB less archive data; Git history was not rewritten.

The benchmark runner defaults full endpoint capture off and permits explicit capture for numerical comparison. Historical run plans remain unchanged. Quiet trial 065 accepted the same four 15.625us initialization steps, with bitwise physiology CSV and accepted-geometry parity against 063. Stdout fell from approximately 833 MB to tens of KB; accepted-loop wall time changed from 33.5443 s to 27.5349 s. Mean GPU time remains approximately 6.35 s; no physics-kernel gain is attributed to dump removal.

Full native checker and moving-wall cavity/rollback/replay checks passed. Build/testing configuration evidence is retained here. This is still a tiny initialization prefix, not supported rest, motion qualification or 0.1x performance.

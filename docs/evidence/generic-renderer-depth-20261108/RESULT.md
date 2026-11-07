# Result: actual Franka renderer depth and edge reference

The pinned native process ran 2026-11-07 23:49:10.159078–23:49:10.657765 UTC, exited 0, and passed input preflight/postflight hashes. It rendered the retained Franka Gaussian scene at 160×120 for four environments. This is a correctness comparison, not a timing result.

The ordinary scene had 3,708 depth-valid pixels and 178 geometry pixels without depth; the 100% depth-dropout scene had zero depth-valid pixels and 3,886 geometry pixels without depth. With edge flicker disabled, both original and candidate outputs matched the independent CPU reference on both scenes. The real renderer clears invalid depth to a very large value, so this observation does not reproduce the original synthetic far=5/noise failure.

With edge flicker probability one, the CPU reference calls for 805 border pixels adjacent to valid depth in the ordinary scene. The candidate matched every reference value; the original differed on exactly those 805 pixels. Both arms matched the all-far reference for the dropout scene. This demonstrates a change in observation behavior on the uncached larger-image path. It does not decide whether filling sensor-invalid border pixels is the desired product contract; that edge behavior must be explicitly resolved before integration. The reference implements the preregistered neighbor-fill rule and is not an external camera calibration.

Exact renderer depth and validity and all output planes are retained in planes.bin.gz, with layout in PLAN.md and hashes in trace.sha256. This test does not cover direct-winner rendering, a live task/policy, mixed physics failures, additional sensor profiles, or observation fingerprints. The test binary's historical owner library is pinned; its extra research hooks were not installed. Product source remains unchanged.


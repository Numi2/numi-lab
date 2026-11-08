# Repeated paired-position standalone rod and rejection replay

**Research-only result, not production qualification.** The unchanged paired host and Metal candidate from main commit `8e01095ff` were exercised through actual standalone native rod calls on Apple M6, macOS 27.0 (26A425). Host-side double state was split into FP32 high/low parts on every call and reconstructed on each return. This checks host-owned state continuation; it does not create or test a persistent GPU or MetalWorld state owner.

The predeclared matrix had six cases: eight-node straight x-axis rods at 0.25 and 100 m offsets, with fixed, moving, or internal hard anchors. Each execution ran a 32-step clean path and a second 32-step path, with a reversed endpoint injected at step 16 of the latter. Each case was independently executed twice. A second full series strengthened the instrument to compare each accepted result, including state, GPU status, and every supported reaction field. The first series and its less-complete output comparison are retained separately.

In the strengthened series, all **768 accepted calls** passed the position, velocity, force, work, local-momentum and gradient-agreement gates against the independently propagated dense FP64 reference. The 12 deliberately bad steps were dispatched, returned GPU status 4, did not publish, and preserved both the input and previous complete output. On retry, every subsequent complete result matched the clean arm; the six full case pairs repeated identically. No failed step advanced the caller-owned state in this instrument. This is a same-process rejection/retry test, not serialized checkpoint restoration or device-error recovery.

Maximum observed errors for either arm over the second series:

| Measure | Maximum | Gate |
| --- | ---: | ---: |
| Position | 9.24e-13 m | 1e-6 m |
| Velocity | 9.39e-9 m/s | 0.001 m/s |
| Reaction force | 1.17e-8 N | max(1e-6 N, 1% of reference) |
| Anchor work per step | 1.23e-13 J | 1e-8 J |
| Free-node local momentum residual | 4.51e-6 m/s | 0.001 m/s |
| Difference of 5/10 nm energy gradients | 4.09e-7 m/s | 1e-4 m/s |
| Dense FP64 reference free-row residual | 9.90e-12 N | 1e-8 N |

The full rod energy gradients are independent CPU finite differences of the same model, not physical material measurements or an external solver. The reference propagates its own state over all 32 steps with the same prescribed anchor schedule; it is not reset to the GPU result at each step.

`PLAN.md` records the pre-trial gates and the instrument addendum. The first `analyze.py` was written after its GPU series completed, so its hash is recorded as an analysis hash, not a pretrial hash. The second probe and analyzer hashes were recorded before its series. `pretrial-v2-sha256.txt` binds the unchanged prior candidate host and metallib as well as the new probe. The executable and metallib are not included; source, commands, logs, analysis and per-execution outputs are retained. Raw `o*.jsonl` files in each series are compressed with `gzip -n`; decompress copies into the corresponding directory to rerun the analyzers. The analyzer requires NumPy. Times in the runner are process walltimes only, not a GPU benchmark.

This does not resolve the production solver false acceptance, the oblique fallback failure, transverse/bending/twist dynamics, persistent device state, coupled contacts, surgical loading, material calibration, or actual performance. Do not enable the candidate. The next engineering step is to put paired positions under an authoritative persistent state owner with atomic failure restoration, then address a full vector force and tangent with the same independent local checks before coupled use.

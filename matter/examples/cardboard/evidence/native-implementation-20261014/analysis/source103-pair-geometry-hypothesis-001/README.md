# Pair 92/127 CPU geometry hypothesis

This is a read-only, CPU-only analysis of the exact pair-specific endpoints prepared from center capture 359 and private trial 23 of capture 358. Inputs are bound through `report.json` to source snapshot 103, both capture payloads, package 003's prepared manifest, and its 4,320-byte native input binary.

The selected EE(0,0) case starts at a closest-segment distance of about 10.061393 µm and finishes at about 9.999283 µm. The source's FP32 effective band is about 10.001907 µm after its 8-epsilon coordinate padding. Independent double-precision geometry over the promoted captured float32 endpoint words crosses the effective band near the end of the linear path and is within nominal thickness by the final endpoint. With the source's absolute endpoint speed bound, the independent conservative-advance recurrence first reaches the effective band on iteration index 27 (28 evaluations); source 103 allows 17 evaluations (indices 0–16). The bound is 1.82 times the exact-real relative endpoint-velocity bound in this case.

This supports a *hypothesis* that EE(0,0) exhausts while approaching a real near-thickness event. It is not the source-103 Metal result: package 003 has been compiled and CPU-verified, but the corrected native diagnostic has not yet been dispatched. No production behavior was changed. A certified hit would still mean that the candidate enters the configured contact band; it would not make the path collision-free.

The narrowest conservative follow-up, conditional on the native pair replay confirming exhaustion, is a bounded continuation of the same advancement recurrence only for calls that exhaust the original budget. Keep the current speed bound, thickness padding, per-step formula, and separating-plane checks unchanged; continue from the existing `time` for a small fixed tail and fail closed if it still cannot certify. This can turn an unresolved status into a detected hit or a certified miss, but cannot by itself make an actual hit acceptable. Do not replace the speed bound with the smaller relative-motion bound until its FP32 rounding/error enclosure is proved. Do not increase the band, discard a witness, or turn uncertainty into a miss.

Reproduce with:

```sh
python3 analyze.py
```

The CPU calculation uses an independent closest-segment implementation in float64. Its CA trace is not a formal certificate and is not bitwise equivalent to Metal. `report.json` is the machine-readable result; `manifest.json` binds the report and script hashes.

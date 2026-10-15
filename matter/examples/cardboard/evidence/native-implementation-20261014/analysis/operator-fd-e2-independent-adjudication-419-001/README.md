# Source125 run 419: independent operator and finite-difference adjudication

This CPU-only report binds the rich Newton48 capture from run419, the 422 fresh production operator replay, the exact 424 native endpoint sweep, and independent FP64 response evaluations. It is an adjudication of diagnostic layers; it does not qualify the failed coupon trajectory or any material calibration.

All 1,728 center material projections pass the captured residual/correction gates. The independently assembled FP64 action differs from the native fresh action by 5.2065e-9 Ns, or 2.8227e-5 relative. The production fresh action equals the captured action exactly; the cached-versus-fresh difference is 2.7817e-8 Ns. The mass term agrees within 1.59e-12 Ns, locating the center action difference in the stiffness contribution. This is small relative to the total action and does not explain the unresolved native finite-difference comparison.

For independent FP64 endpoints at epsilon 2^-22, every projection passes and local liner/medium tangent errors are 3.66e-8 and 1.07e-7. The assembled secant still differs from the center tangent by 2.2462e-5 relative. At still smaller steps the assembled secant error grows, which is consistent with cancellation in the small total action; this is a measured step trend, not a universal noise floor. At epsilon 1/64, the endpoint path is large and has lambda-activity crossings; one liner endpoint projection fails, so a complete FP64 assembled secant is unavailable.

The native 424 endpoint replay is bitwise exact against run419 at each retained endpoint, but no tested epsilon resolves the native centered secant against the captured A direction. The best admitted native result is 1.0948 relative error at epsilon 2^-14, still a failure. At epsilon 1/64 the error is 17,199.84. At epsilon 1/32 both side statuses are 5 and the secant is inadmissible. Those status codes are stage/environment status, not per-tetrahedron local failure flags.

The separate first-status5 capture contains one private Armijo internal-force record at Newton26 for tet1464. Its determinant is 0.995947, while its reported maximum stress is 1.17918528 GPa against a 1.0 GPa validity bound. This is one captured rejected/private evaluation; it is not a classification of every row in the later finite-difference side arrays.

Unavailable endpoint metrics remain JSON `null` with an explicit unavailable status in the preserved small-step report. The adjudication script rejects nonfinite JSON constants and serializes its output with `allow_nan=False`.

Reproduce the adjudication with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 adjudicate_419.py
```

The script invokes the already-built source125 status5 decoder on the run419 payload, verifies the input report bindings and exact native endpoint parity, and writes `report.json`. It performs no native/GPU execution and does not modify the source run or its reports.

# Root 2, microtick 3 contact Armijo diagnostic

The corrected replay selects the actual terminal failure: root 2, zero-based microtick 3, solver iteration 9, contact slot 2624. The earlier root-2/microtick-2 capture is the preceding successful microtick; the scheduler increments `completedMicrosteps` only on success. Keep that earlier selector miss as historical context, not a terminal explanation.

The selected record is an unmapped vertex-triangle contact: identity `[19,19,0,1]`, source `[15478,16093,2624,1]`, nodes `[10165,10162,10167,9917]`, weights `[1,-0.461901963,-0.538098037,0]`. Failure is status 6 (barrier Armijo rejected). All 12 trial rows passed CCD certification at full fraction, initial-touch/nonclosing gate, and finite-energy checks.

At the failing row, base energy is `6.34614352e-8`, directional derivative `-8.35808159e-11`, and base separation `8.17600176e-6 m`. Trial energy rises at every tested alpha. For alpha 1 through 1/16, energy increases by `1.3870434e-9` through `3.3317420e-9`. At alpha 1/32 through 1/2048, FP32 endpoint quantization leaves only node 10165 x changed by `-3.7e-9 m`; the same rounded feature endpoints and energy repeat. The plateau energy `6.35281623e-8` remains `6.67271e-11` above the base and is rejected by the strict Armijo RHS. This is a discrete potential-ascent / derivative-consistency failure with a quantized small-step plateau, not a CCD failure or accepted no-motion step.

The replay reproduced the same failure tuple and byte-identical accepted CSV prefix as native run 037 (two accepted root rows). It is diagnostic evidence only; it does not establish a general contact correction or sustained support. Exact executable SHA-256: `37d11b2888dcd4413cc22052f284d26b3f36bce971989c4112c1a2f524e82867`; metallib SHA-256: `dcde161fd4ca8bc6f0020cf43164493b45b934214c1c1a45d791fb8036220469`; source base: `878da43f57389bfd715f8db953f4512b197fb518`.

Machine-readable details are in `summary.json`, `selected-traces.json`, and `selected-traces.tsv`. The original trace and full replay artifacts remain in the build-validation run directory recorded by the summary.

# Exact production VTCCD adjudication: Newton 9 trials 21 and 22

This package verifies the inline, read-only record of the exact `vertexTriangleCCD` calls selected at private Armijo Newton 9. It joins each helper record to its existing full-trial capture and compares the instrumented execution against its matching uninstrumented capture. It does not qualify the full-contact fixture; both complete fixture executions still fail.

The records are from `full-contact-trial21-vtccd-call-405` and `full-contact-trial22-vtccd-call-406`. Their matching baselines are `full-contact-trial21-capture-398` and `full-contact-trial22-capture-399`. All four captures are strict JSON, pass payload and per-array SHA-256/extent checks, and bind source physics fingerprint `915f50b10edc2e46`. The two instrumented captures add one 336-byte array, `actual_vertex_triangle_ccd_call`; all 129 original arrays have identical hashes to their respective baseline. All 32 production Armijo rows, transaction JSON records, and normalized control logs match. Each before/after diagnostic snapshot differs only in bytes 12–19, the serialized device-program fingerprint; all remaining bytes match. The run receipts report normal fixture failure (exit 1), no timeout, and unchanged source inputs.

The selected candidate is environment 0, slot 173, surface pair 36/123, direction 1, vertex 2. Production calls the right primitive's third node (32) against the left triangle nodes (13, 10, 14). The exact helper arguments show a 9.99999975 µm authored thickness and 99.9999975 µs timestep. For trial 21, Metal returned `hit=false`, `certified=false`, `toi=0.9422037005`, and separation 10.001938790 µm. For trial 22 it returned `hit=true`, `certified=true`, `toi=0.9419823289`, and separation 10.001822375 µm.

The source's effective distance threshold is `thickness + 8*kEpsilon*max(thickness, coordinateScale, 1e-12)`, 10.001906958 µm for these inputs when each source float operation is rounded to binary32. Thus the two returned separations straddle the threshold by +0.03183 nm and −0.08458 nm. A separate binary64 closest-point sweep uses the exact captured binary32 endpoints, linearly interpolated in binary64. It finds the path crosses the same effective threshold at `t≈0.9423439985`, with a final gap 3.5843 nm inside the threshold. The independent calculation is a diagnostic, not a replacement for the Metal result.

Trial 21 therefore records an **uncertified query**, not a certified separation or miss. In the frozen helper, `certified` starts false, the loop permits 16 conservative-advancement iterations, and exhaustion returns false without certifying. Trial 22 returns a hit slightly before the independent binary64 crossing; the gap between the Metal separation and that independent distance is about 0.107 nm, on the scale of a float32 ulp around millimetre coordinates. The nearby trial endpoints differ by only a handful of representable low-coordinate bits while their z paths are the same. The full capture confirms the narrowphase status linkage: trial 21 writes status code 6 at slot 173 with diagnostics `[36,123,2,4]`, while trial 22 clears the private narrowphase status and the active count rises from 123 to 129. The correct reading is a real near-threshold swept contact whose outcome is highly sensitive to float32 interpolation and the fixed iteration budget. This evidence does not justify relaxing thickness or accepting an uncertified ray.

The independent sweep uses Ericson point/triangle closest-point regions in binary64 on the captured endpoints, and bisects the monotone distance to the source-defined effective threshold. It reports the continuous-path estimate separately from the actual production helper fields.

## Reproduction

Run:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/trial21-22-exact-vtccd-adjudication-001/audit.py
```

The script rejects invalid JSON, bad payload/array hashes or extents, unexpected array additions, mismatched common arrays, changed Armijo rows, unexpected transaction differences, and a different selected helper tuple. Its report is `report.json`.

Source references are in the frozen source-122 snapshot:

- `matter/src/metal/contact.metalinc:3234` begins `vertexTriangleCCD`; lines 3248–3265 set the certification state and effective thickness, 3270–3318 perform the 16-step conservative-advancement loop, and 3322 returns false after exhaustion.
- `matter/src/metal/contact.metalinc:3950–4018` select vertex/triangle orientation, write the exact-call record, and fail closed when `certified` is false.
- `matter/src/metal/common.metalinc:80–86` defines the distance roundoff added to authored thickness.
- `matter/include/numi/matter/shared.h:622–637` defines the 336-byte observer record.

No tolerances, physical laws, production status gates, or GPU acceptance behavior are changed by this report.

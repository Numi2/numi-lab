# Exact captured VTCCD budget replay (diagnostic overlay 001)

This overlay adds one standalone Metal replay target against frozen source122. It consumes the exact 336-byte `actual_vertex_triangle_ccd_call` records from retained native trial21 and trial22 captures. It runs the same vertex/triangle CCD implementation at compile-time iteration budgets 16, 32, 64, and 128, and exports each conservative-advancement query as `(time_fraction, gap_to_effective_thickness_m, separation_m, flags)`. `time_fraction` is the CCD sweep parameter (not wall-clock duration); `flags` bit 0 is certified and bit 1 is hit. A final uncertified return at exactly the requested budget is reported as budget exhaustion; the trace is not interpreted as a miss.

The physical entry point remains `vertexTriangleCCD(...)`, with its original call signature and fixed 16-step default. It delegates to `vertexTriangleCCDBudgetCore<16u, false>`; this specialization does not write diagnostic records. No thickness, roundoff, step-size, feasibility, or status threshold changes. The native checker requires the 16-step replay to match the exact captured production result bits for both input captures. Higher-budget outcomes are descriptive and are not assumed to certify a miss or improve physics.

This is a compile-and-replay candidate only. The Metal translation unit compiled and linked successfully, and the Objective-C++ checker passed strict syntax validation. The Metal compile emitted two existing warnings in source122 `fgmres.metalinc` about threadgroup variables initialized only by lane 0; this overlay does not touch those lines. No GPU replay was run in this package. Root should run the standalone target later against the retained capture files and preserve any failure or unchanged uncertified outcome.

## Files and source binding

Base snapshot: `/Users/home/cardboard-implementation-20261014/snapshots/source-native-tet858-vtccd-traces-122`.

- `overlay/matter/src/metal/contact.metalinc` is generated deterministically from source122's `matter/src/metal/contact.metalinc` SHA-256 `ea5a9caf6f47588258aeeb5e8be57168826acfb94a566ecbfea7e647ad50fb5c` by `generate_overlay.py`.
- `overlay/matter/tools/cardboard_vtccd_iteration_budget_check.mm` validates capture/payload/array hashes and exact selection, dispatches the new replay kernel, checks bitwise budget16 parity, and prints all trace rows as JSON.
- `overlay/matter/CMakeLists.txt` adds the standalone checker executable, but intentionally does not register a CTest because it requires retained capture paths.
- `validate_overlay.py` is the CPU-only source and input contract check.

Run CPU source/capture validation:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/private-vtccd-budget-replay-overlay-001/validate_overlay.py
```

After applying the overlay and building the target, root can run:

```sh
numi-matter-cardboard-vtccd-budget-replay-check \
  /Users/home/cardboard-implementation-20261014/native/full-contact-trial21-vtccd-call-405/trial-capture.coupled.json \
  /Users/home/cardboard-implementation-20261014/native/full-contact-trial22-vtccd-call-406/trial-capture.coupled.json
```

The 405/406 source and capture adjudication remains separately sealed in `trial21-22-exact-vtccd-adjudication-001`. Both source runs failed the complete fixture; the overlay does not change that verdict.

# Private mortar CCD diagnostics overlay on source 095

This external overlay is a source-bound follow-up to native private-mortar run 314. It carries the exact shared-edge proof from overlay 005, makes paired reference-map endpoint identity exact in all three barycentric coordinates, adds diagnostic partition-stage codes, and repairs/extends two fixture geometries. It is not a production qualification and has not been run on the GPU.

The paired-boundary map now requires exact FP32 equality of each matching endpoint on both mapped parent faces. The CPU guard tests cover the original shared-edge tolerance sliver, strict opposite-side geometry, same-side overlap, positive-area overlap, collinear uncertainty, one-ULP mismatch on either mapped side, and a 1e-6 near-map mismatch. Existing clipping and overlap thresholds are unchanged.

The CCD witness-location predicate still has its existing 8e-6 barycentric roundoff tolerance. A new native private-sweep negative case has exactly matching reference maps, a 4 nm in-plane parent offset on a 4 mm face, zero endpoint displacement on every node, and an initial cross-patch overlap. Its expected result is contact failure. This deliberately tests whether the witness tolerance can hide an inadmissible initial contact despite common zero translation. The expected result is not yet measured.

The partial-patch covered-crossing control begins with 0.2 mm separation and leaves at least 60 um clearance over the uncovered complement, versus 10 um collision thickness. The added uncovered-complement crossing begins with the same positive gap and crosses at the uncovered zero-first-barycentric vertex; the active-patch edge retains at least 18 um clearance, so the crossing is isolated to the uncovered complement. The arithmetic check is fixture analysis only, not a native CCD result.

Private partition failures now report stage code in diagnostic component z: 1–13 identify left-partition validation stages, 101–113 identify right-partition validation stages, and 200 means both partitions built but the sweep itself did not certify. This is diagnostic-only; failure status and geometric thresholds are unchanged. It should help distinguish the unresolved four-patch failure from a sweep failure on the next GPU run.

Build/check commands:

```sh
cmake --build build-host --target numi-matter-cardboard-full-contact-armijo-check --clean-first -j 4
clang++ -std=c++20 -O2 -ffp-contract=fast -Wall -Wextra -Werror \
  -I build-source/matter/include mortar_shared_edge_guard_check.cpp \
  -o mortar_shared_edge_guard_check
./mortar_shared_edge_guard_check
/Users/home/.pyenv/versions/3.10.12/bin/python3.10 fixture_geometry_check.py
```

The host target and Metal library compiled successfully. The CPU guard check passes 14 checks. No GPU/native execution was performed for this overlay. Native run 314 remains a failure: 9/14 expected gates passed and five expected-accept cases rejected. In particular, the partial-covered crossing and 3/4-patch cases remain unqualified; the four-patch partition rejection is not resolved by this report.

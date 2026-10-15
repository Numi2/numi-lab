# Exact-dyadic audit of the remaining source FP32 overlap rejects

This is a CPU-only diagnostic follow-up to overlay 005. It checks every candidate side record that failed 005's exact-shared-edge predicate. The records were emitted from the canonical source-091 shipping geometry after the source's existing FP32 clipping test found an overlap above the unchanged `1e-10f` floor while its long-double comparison did not.

The captured map vertices are the exact binary32 barycentric coordinates consumed by the partitioner. The audit reparses each hexadecimal float as binary32, then runs two independent triangle-intersection calculations:

- exact Sutherland–Hodgman clipping with Python rational arithmetic on each input float's exact IEEE-754 dyadic value;
- strict double-precision clipping on those same promoted FP32 coordinates, with no geometric epsilon and `math.fsum` for final polygon area.

The original threshold is compared as its exact binary32 value (`14411519/144115188075855872`, approximately `1.000000013351432e-10`). It is not changed.

## Result

The extractor produced 635 side records for 497 unique stable-ID patch pairs. Exact rational clipping found exact zero intersection area for all 635 sides. Strict double clipping returned zero for all 635 and agreed with the exact area-floor decision for every record. Therefore, for these source maps, the 497 unproven rejects are tolerance-induced clipping artifacts, not positive-area intersections.

Five controls agree between exact rational and strict double clipping: exact adjacent edge with zero overlap; same-side triangles sharing an edge with positive overlap; positive overlap without shared endpoints; a one-ULP inward endpoint perturbation whose positive intersection is `1/33554434` and exceeds the floor; and a one-ULP outward gap. These controls show the full-double calculation preserves the existing area rejection decision for positive cases and does not convert the near-edge overlap into a safe exact-edge exemption.

This is evidence for a **CPU overlap-test** replacement using full-double clipping on canonical FP32 inputs. It does not validate the existing Metal partition builder: its clipping and complement construction still use FP32 plus a `1e-7f` clip tolerance. No Metal/native execution was done. Do not use this result alone to admit the 497 cases in runtime contact; the GPU partition path needs its own geometry-safe implementation and native controls.

## Reproduction

The source-091 geometry extractor was copied to `shipping_mortar_partition_candidates.cpp` with only an added diagnostic output for unproven source-sliver sides. It was compiled against the source-005 build tree using `-O2 -ffp-contract=fast`, then run to create `candidates.log`. The candidate log includes all exact hexadecimal map coordinates. Run:

```sh
python3 verify_exact_vs_double.py
```

The script writes `comparison-report.json` and exits nonzero on any candidate floor-class disagreement or any failed control. `exact-dyadic-results.json` is the earlier full numerator/denominator dump; the final comparison report is the concise authoritative summary with every candidate row and all controls.

No production files were changed and no GPU job was run.

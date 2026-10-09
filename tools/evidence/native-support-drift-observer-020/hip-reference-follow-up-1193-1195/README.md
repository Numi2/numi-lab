# Hip-capsule reference sensitivity follow-up (1193–1195)

This follow-up records a nominal 1.0 run and two prespecified 0.5 and 2.0 scale sensitivities, all using the same 255 immutable assets, 40 s duration, 0.002 s step, capture schedule, contact iterations, and activation cap. It retains the 1193 region-pair counterfactual diagnosis and the full final-step exact audit summaries, run declarations, executions, native metadata, audit scripts, and external hash pins. MRVPack captures and native logs are not copied; their paths and SHA-256 values are pinned in external-hash-pins.json.

## Result

All three scales failed the exact final-pose geometry gate at accepted step 20,000 (40.0000019 s). With 859 target surfaces checked using the exact Float32-lattice triangle predicate, target crossings and unallowed skin self-intersections remained:

| Hip-capsule scale | Target-crossing pairs | Skin self-intersection pairs | Degenerate skin faces | Invalid target triangles/surfaces |
|---:|---:|---:|---:|---:|
| 0.5 | 2,817 | 340 | 0 | 0 / 0 |
| 1.0 | 3,073 | 575 | 0 | 0 / 0 |
| 2.0 | 3,776 | 691 | 0 | 0 / 0 |

Pair coverage completed and all captured inputs remained unchanged. The lower scale reduced the raw counts, but it did not pass; the higher scale increased them. No scale was selected or adopted.

Each arm logged 2,500 accepted observer samples; 2,490 were outside the neutral ab/ad flexion fit interval [-12°, 0°]. These are observer samples at 8-step intervals (0.016 s), not all 20,000 physical steps or empirical fit observations. Abduction/adduction is not parameterized. The excursions limit how strongly the nominal fit can support those states.

The 1193 region diagnostic is read-only. Its disabled counterfactual is 20 s only, so it does not isolate a 40 s causal effect. Its support inspection describes 32 NHCNT point witnesses on a single plane, with no femur or patella support points; it does not represent a broad hip/thigh contour. The nominal 1193 run has audits at steps 0, 9,983, and 20,000; the 0.5 and 2.0 variants are audited at step 20,000 only. No interpolation between captures is claimed.

## Evidence map

- [Final-step counts and fit-domain diagnostics](final-step-geometry-summaries.json)
- [1193 region-pair counterfactual diagnostic](diagnostics/region-pair-counterfactual-diagnostic.json)
- [Step-20,000 scale comparison](diagnostics/scale-sensitivity-step20000-comparison.json)
- [Nominal 1.0 run declaration, execution, metadata, audit declaration/summary/result, and audit script](runs/nominal-1p0/)
- [0.5 scale run evidence](runs/scale-0p5/)
- [2.0 scale run evidence](runs/scale-2p0/)
- [Shared native runner](scripts/run.py)
- [External source, run, capture, receipt, and log pins](external-hash-pins.json)

The top-level bundle SHA256SUMS covers these added files. The external pin manifest records 316 referenced files; all 316 were present and SHA-256 verified when this follow-up was assembled.

# Connected skin and fat repair

This increment continues one connected 7,360-node, 32,796-cell skin/fat construction with the same mechanical reference, material identities, conservative mass ledger, and 61,950-vertex detailed skin embedding. It does not install new anatomy or accept native physical steps.

## Implemented changes

Use actual compiled reference coordinates to derive O-D construction directions, checking every generated point against its declared parent barycentric ancestry. Select free outer-boundary regions from current crossing witnesses while keeping inner skin/fat coupled. Generate crease and cell-gradient constraints from failed proposals within one shared LP budget. Full Float32 cell, layer, coarse/detailed surface, crease and exact detailed self-intersection checks remain mandatory.

Cell-gradient caps are measured in the current emitted geometry. Per-cell bounds account for the current quality and volume margins, allowing safer regions to advance while protecting thin cells. Tests cover reference-frame covariance, typed identities, mapped connectivity, thin cells, random bounded gradients, exhausted budgets and failed-refinement telemetry. All 144 Python tests and 11 subtests pass. Material laws, Metal runtime, source mass ownership and the prepared native receiver are unchanged.

## Corrected direction fixture

The previously published source-field-audit-001 fixture used legacy O_world_m/D_world_m coordinates from a different frame. Its helper was internally consistent but that fixture was not valid for the active compiled reference. Field 002 derives directions from the actual reference and verifies generated ancestry. All 7,360 directions differ from 001, with a maximum angular difference of 89.36 degrees. The old fixture remains published failed evidence. No run used it to construct the new ray constraints.

## Measured static progress

These counts are exact edge/bone groups and uniquely inside mechanical nodes against 185 banks, not triangle-pair totals or certified penetration depths.

| Trial | Final crossing groups | Inside nodes | Outcome |
|---|---:|---:|---|
| Initial connected mesh | 2,498 | 140 | Fixed comparison origin |
| 099 | 2,498 | 140 | Initial LP time limit, no move |
| 100 | 2,421 | 136 | Four passing steps |
| 101 | 2,312 | 149 | Three steps; count-score stop retained |
| 102 | 2,311 | 148 | Three steps; count-score stop retained |
| 103 | 1,246 | 65 | Eight passing steps, 550.36 seconds |
| 105 | 1,029 | 46 | Three passing steps, then midpoint construction stop |

Run 103 continues on actual same-witness clearance improvement even when diagnostic counts temporarily rise. It never bypasses geometric gates. Run 105 starts from its exact final candidate rather than resetting displacement history.

The final passing 105 candidate is SHA-256 402af57dc071411ae21658c54729c8cc680c5893da573e879c598d0797b9656e. All 32,796 cells have positive volume and pass the 1e-5 construction-quality floor. Minimum quality is 2.06418e-5; J ranges 0.381641 to 2.438782. Maximum accumulated displacement is 11.4519 mm. The complete detailed surface has zero unexplained self-intersections and zero violations of the configured 15-degree crease-change guard. That guard is a construction choice, not an anatomical reference range or proof of realistic deformation.

105 stopped before its fourth solve because nine span midpoints alias edge endpoints after Float32 rounding. The saved queries have zero expected/actual parity disagreements. The failed stop and its witnesses are retained; resolving endpoint aliases is a subsequent increment, not silently rewritten evidence.

## Remaining admission work

There are still 1,029 crossing groups and 46 inside nodes. Edge/node screening does not establish complete closed-solid clearance, neighboring-organ clearance or anatomical validity. The eight retained motion poses and native 16/60-second runs remain unqualified. Existing head-height skin boundary loops are not reclassified as closed.

The prepared receiver already fits the sparse map (at most three entries per active row), but the native loader, arbitrary skin/fat ObjectSource, explicit boundary/contact maps and legacy layer-index assumptions still need integration. No runtime physics, controller history, physiology, visual acceptance or real-time performance claim is made. All work ran on SSH Numi (Apple M6); no images were opened.

## Evidence and reproduction

852 retained artifacts include failed trials, executed owners, drivers, source hashes, corrected reference fixture, query identities and compact candidate arrays. Large reused inputs are independently checked against their previously published bytes. Existing public artifact formats are retained.

Run:
    python3 verify_retained_artifacts.py --canonical-root /path/to/numi-lab --git-root /path/to/numi-lab
    cd /path/to/numi-lab/matter/tools
    python3 -m pytest -q test_human_skin_fat*.py

Run drivers retain exact SSH paths and input hashes. The dirty native checkout and index remain untouched; publication uses an alternate index.

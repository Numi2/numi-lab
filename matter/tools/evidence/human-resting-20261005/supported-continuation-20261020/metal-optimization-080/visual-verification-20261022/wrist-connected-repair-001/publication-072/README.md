# Connected skin/fat repair: coupled constraints and retained trials

This increment extends the existing connected authoring owner. It does not install a mesh, advance physical state, change a constitutive law, or qualify the human scene.

The LP path now uses the same fixed-origin detailed-skin mapping, projected-area constraints, support connectivity and exact Float32 area/orientation backtracking as the sparse least-squares path. The accepted-transfer reader verifies the existing 075 source export and typed row identities. A reusable SAT separator builder composes triangle-pair inequalities through those same mechanical displacement weights. Its rows can be passed as hard lower-linear constraints; they do not receive clearance slack.

All 47 canonical Python regressions pass. New integrated tests demonstrate (a) a positive coarse tetrahedron can still fold the mapped detailed skin and must backtrack, and (b) an infeasible contact request cannot soften a hard mapped self-separation row. The SAT builder preserves the exact Float32 weights in Float64 authoring arithmetic. Its rows are not a collision certificate. No native shader parity is claimed.

## Measured static outcomes on SSH Numi

All comparisons below use the same 9320-node 037 baseline unless explicitly identified otherwise. Baseline: 8618 edge/bank crossing groups, 1028 inside nodes and 1030 inside node/bank pairs.

| Trial | Crossing groups | Inside nodes | Detailed self-intersections | Cell result |
| --- | ---: | ---: | ---: | --- |
| 076, one connected sparse iteration | 8474 | 998 | 0 | All emitted cells pass |
| 077, four active-set iterations | 8431 | 995 | 3 | Cell and orientation gates pass; actual self-collision fails |
| 080/082, hard LP and 25 fine-skin separator rows | 8571 | 1013 | 0 | All emitted cells pass; accepted fraction 1/32 |

The exact detailed-surface audit processes all 108987 faces and does not exempt adjacent faces when their intersection extends beyond the common indexed feature. The 080 and 082 candidates are byte-identical. Unrounded LP-direction replay shows fractions 1 through 1/8 fail signed cell quality and 1/16 fails J; 1/32 passes the checked gates. Hard separator and thickness rows remain satisfied throughout. The result is not a completed anatomy repair.

A more accurate sparse normal solve reduced its linear normal residual by about ten orders of magnitude on the initial identical system, but improved the geometric result only modestly and left large constraints unresolved. It remains an experimental retained backend, not the canonical solver default. A separate source-surface topology branch reduced 126 nonpositive cells to 114 through internal reconnections; a subsequent connected untangle left 81 and introduced two outer self-intersections. Neither was admitted.

## Whole connected volume reconstruction

The complete native skin/fat PLC includes 9734 exterior facets, including 180 side-strip closure facets, and 4777 shared material-interface facets. Those closure strips do not establish that the detailed whole-body epidermis is closed.

Trial 083 preserves all original vertices but TetGen exits 200 under -YY because boundary recovery inserted points. Six of its output cells also fall below the quality floor. Its preserved output is diagnostic only. Source inspection corrected a construction assumption: -S0 does not impose a global zero-point cap, and the -S budget is initialized after mandatory boundary recovery. Trial 084 therefore permits conforming subdivision and independently counts generated nodes.

The bounded 084 run exits 0 in 0.187 seconds with 7361 nodes and 32801 cells. All 7233 original coordinates are exact; 125 added points are on source PLC segments and three are interior, with per-material barycentric ancestry. Every source facet has a consistently oriented child subdivision, maximum area-ratio error 1.67e-14. Initial emitted reference/current cells are all positive, with no q-floor failures (minimum q 5.012e-4). This is a baseline remesh construction result. Transfer of the actual repaired 037 deformation, conservative source-mass mapping, clearance and native motion qualification remain separate gates.

## Frames, ownership and limitations

The 075 coarse/source export is in the AP initial frame. Its direct comparison with an unrelated posed world capture was invalid and is retained as superseded evidence. The corrected source-ID comparison has 2.241 micrometres maximum difference without fitting a transform. Direct P1 replacement of the established additive mapping increased affected folds and was rejected.

Current detailed anatomy, materials, source identities, reference mass ownership and native simulation remain unchanged. Positive cell determinants, mapped-face orientation, self-intersection clearance, bone clearance, mass conservation, motion qualification and native physical acceptance are separate gates. No candidate here has passed all of them. Remaining bone crossings and native motion/physiology qualification are explicit open work.

Execution and authoring were on SSH Numi. There were no visual inspections or physical steps. At the last check only about 3.1 GiB was free; the declared 12 GiB floor for native qualification is unmet. No sole source assets or unverified evidence were removed.

## Reproduction and integrity

Use the retained input pins and run-specific commands in evidence/. The source-artifacts manifest verifies stored and uncompressed bytes, including failed trials. The canonical manifest binds the exact tested owner files.

```sh
python3 verify_retained_artifacts.py --canonical-root /path/to/numi-lab
python3 -m unittest discover -s /path/to/numi-lab/matter/tools -p 'test_human_skin_fat*.py' -v
```

The static authoring uses the existing NumPy/SciPy environment; it does not provide CPU per-step physics. The existing Metal runtime remains the simulation owner.

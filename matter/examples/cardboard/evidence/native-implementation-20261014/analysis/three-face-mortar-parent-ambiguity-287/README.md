# Three-face mortar parent-pair diagnosis (run 287)

This CPU audit binds to the retained run 287 source and activation trace. It reproduces the geometric identity of the failing cohesive contact pair and traces the two native uses of the parent-pair lookup helper. The primary machine-readable result is `analysis.json`; `manifest.json` binds the audit, source files, run receipt, and trace by SHA-256.

The first three-face activation attempt ends before FGMRES. Its contact status is code 6 at candidate slot 78, with diagnostics `[16, 99, 0, 1]`; private validation reports the same pair and branch 5. Under the fixture’s stable primitive encoding `primitive = 4*tet + opposite`, these primitive IDs map to parent tetrahedra 4 and 24. Independent reconstruction of the split pad shows the exact shared interface nodes are `(9, 10, 13)`. The fixture constructs three subtriangles by copying one `CohesiveMortarPatchSource` and changing only each subtriangle’s barycentric map and stable identifier, so all three patches belong to the same parent tetrahedron pair.

The lookup at `contact.metalinc:1348-1395` scans active mixed-mode face references for a parent primitive pair. It returns failure at the second match (`if (found) return false`) while its result type can carry only one `NMCohesiveFaceGPU`. Both the private trial validator (`1808-1816`) and contact candidate generation (`2913-2921`) call this helper and convert the ambiguity to `NM_STATUS_CONTACT_FAILURE`. Therefore this failure is a parent-pair multiplicity limitation; the recorded evidence does not identify bad geometry, a tiled-page storage defect, a material-law failure, or FGMRES failure.

There is already a separate helper, `contactFeatureCoveredFromMortarParent` (`1862-1907`), which scans matching references and returns true if any active patch covers a point. That gives point-feature suppression union-like behavior. It does not resolve the missing swept-complement behavior: both paths through `activeMixedCohesiveFaceForPrimitivePair` build `NMMortarContactPartition` from only one face (`1818-1826`, `2924-2932`), and `buildMortarContactPartition` holds at most 10 subtriangles (`1200-1345`). Simply deleting the duplicate guard or selecting one face would leave complement CCD inconsistent with the union of active mortar patches.

## Interim correction boundary

The current runtime contract can be made fail-fast by validating that a parent primitive pair has no more than one authored mixed-mode mortar face. In parallel, the tiled-storage parity fixture can exercise three pages with three distinct, adjacent parent pairs instead of three patches on one pair. That restores the intended page-boundary test without pretending to implement union coverage. This is an interim contract; it does not support multiple mortar patches on one parent pair.

Real union support needs the contact lookup to return a bounded reference span/list rather than one face, duplicate-parent validation to distinguish legal disjoint patches from overlapping or degenerate patch maps, a deterministic barycentric union partition for both primitive sides (with an explicit maximum patch/cell capacity and fail-closed overflow), and the same union semantics in trial validation, candidate generation, and feature suppression. Cohesive face forces and histories must remain attached to each original face ID; union coverage is only for deciding which ordinary surface-contact regions are covered. Dormant patches are omitted from the active union. Uncovered complements remain ordinary contact candidates, while covered-covered regions are suppressed. Overlap between active patch interiors or inconsistent side maps must reject rather than double-cover or silently drop contact. The partition overflow bound needs validation and tests before any wider patch count is admitted.

## Reproduction

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/three-face-mortar-parent-ambiguity-287/analyze_parent_pair_ambiguity.py
```

The audit is source/trace analysis only. It does not execute Metal, validate a repaired overlay, or qualify physical folding/adhesion.

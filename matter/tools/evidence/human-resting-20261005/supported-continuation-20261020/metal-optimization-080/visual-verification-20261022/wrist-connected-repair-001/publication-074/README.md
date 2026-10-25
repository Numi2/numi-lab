# Connected detailed-skin and fat clearance

This increment extends the existing connected authoring owner; it does not install a candidate anatomy or run native physics.

The LP now applies true-area and normal-cosine constraints to detailed skin through the existing displacement map. Sparse detailed-face derivatives compose with W, and only faces with nonzero support in the free mechanical domain receive linearized rows. Empty-support regions produce no constraints. Full-face emitted Float32 area/orientation and exact whole-surface self-intersection checks remain mandatory. Material laws, mass ownership and the Metal physical solver are unchanged.

All 101 Python tests and 11 subtests pass. New checks cover random-direction chain-rule derivatives, mapped normal-cosine bounds, support filtering and empty-support cases. The existing prepared-domain C++ owner is unchanged from 18df90e2.

## Mass precision clarification

The local precision repair in 18df90e2 changes 54 geometric overlap values and retains its failed predecessor. Only the subsequent parent-local mass partition leaves its corrected input CSR bitwise unchanged. The raw geometric coverage gate and conservative mass-ledger gate remain separate. See publication-073 for the corrected CSR, local factors, strict source-ledger checks and separate spatial-moment measurements.

## Measurements and limitations

Run 091 starts from the frozen 089 fourth candidate: 2,121 edge/bone groups and 129 bone-inside nodes on 7,360 mechanical nodes and 32,796 cells. All comparisons use that same topology, fixed reference, 185 bone banks and detailed map. Numeric results below are static asset-authoring outcomes, not contact forces, native accepted steps or throughput.

Run 091 completed four iterations in 122.64 seconds. Crossing groups fell to 1,836; inside nodes ended at 129 after intermediate counts of 132. All four candidates have zero invalid/subthreshold cells and zero detailed self-intersections. The final minimum quality is 1.91997e-5, J spans 0.411889 to 2.47943, and the maximum accumulated displacement is 4.4384 mm. These bounds are measurements, not evidence of native equilibrium. Candidate SHA-256: 0675413e628d157cf48fc8a00508ed25b718126c40881fc3e5c03cdb069bb903.

The head-layer audit in publication-073 remains unresolved. Outer skin nodes are outside the head banks, while deep fat nodes account for most head inclusion witnesses. Global edge-count improvement does not establish regional anatomical clearance. No thickness, density, cell-quality or collision gate is weakened.

A matched four-pass nearest-exit experiment (093) reached 1,751 edge/bone groups and 91 inside nodes in 129.45 seconds; all cells remained valid and all detailed self-intersection checks were zero. Its candidate hash is 387477869e6c19925f4ba96549c83cbb013578fe5eb4d2e6544c70b36309d1d5. It is diagnostic, not admitted: a source-relative directional audit found 544 of 1,028 eligible head edge planes opposing the O-to-D outward column direction (15 generated-support planes were excluded from that diagnostic). Nearest exit alone does not select the connected exterior side.

The regional distortion audit (094) also found hand crease risk in 091: maximum signed-dihedral change was 76.54 degrees. An opt-in construction safeguard (096) now bounds periodic signed-dihedral changes relative to the immutable detailed origin, adds analytic sparse hard rows through W, and checks the actual emitted Float32 surface during backtracking. The configured trial bound is 15 degrees; this is a declared construction safeguard, not a literature-derived anatomical threshold. Its whole-surface check reproduces 094: 091 has 1,205 bound violations and 093 has 755. Neither candidate is promoted merely because its crossing count improved.

The bone-clearance authoring owner also now provides exact first-exit selection along an explicit source-layer direction. It checks the complete bank bounds, all ray-hit classifications and odd inside-to-outside parity, retains rational hit ordering and rejects ambiguous witnesses. No nearest-exit shortcut is used by that API. The direction comes from source-input row O-D vectors, not global identity labels. Actual prolongation covers all 7,360 nodes, including 127 generated nodes and 183 parent choices, with maximum parent-direction disagreement 4.61e-15; raw barycentric roundoff is retained and reported under the explicit 1e-10 bound. This validates the construction field, not anatomical outwardness or a repaired mesh. The subsequent combined construction is qualified separately.

The reproducible layer-owner audit (095) confirms varying inferred construction fields. For 75 selected head rows, authored field and authored rest-coordinate I-D medians both equal 9.9006 mm; posed/current emitted spacing has median 10.2565 mm. That difference alone does not establish a faulty thickness default or registration. A reference update requires regenerated geometry and conservative accounting, rather than weakening current gates.

The current native application still requires a trusted refined-input loader and compiler/callsite regeneration before it can use this candidate. No native shader parity, motion qualification, anatomy acceptance or physiology change is claimed. Existing runtime sources, unrelated dirty changes and the working back source remain preserved.

## Reproduction

Use the pinned run drivers and source manifests in evidence/. They import the same canonical authoring owners; copies are retained execution evidence. Major reused inputs are already published and independently hash-verified in publication-073.

```sh
python3 verify_retained_artifacts.py --canonical-root /path/to/numi-lab --git-root /path/to/numi-lab
cd /path/to/numi-lab/matter/tools
python3 -m pytest -q test_human_skin_fat*.py
```

All execution was on SSH Numi (Apple M6), with no visual inspection. The Metal runtime remains the per-step physics owner.

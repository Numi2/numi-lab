# Connected skin/fat construction and static query
This increment changes offline mesh authoring and source verification only. No anatomy candidate, physical timestep, constitutive reference reset, or performance claim is admitted.

The authoring owner now retains a fixed construction origin, cumulative displacement objective, connected P1 displacement-gradient control, exact Float32 cell/Jacobian/quality, skin/fat thickness, boundary area and normal checks. It offers a sparse regularized least-squares direction and bounded semismooth inequality fit; these produce directions, not feasibility certificates. Mandatory exact backtracking and unchanged outside vertices remain authoritative. The default legacy LP path remains available.

The static query executable reuses the exact existing D37 containment owner. Query-row IDs are explicit and distinct from native global node IDs. Its baseline and refined results match the retained native-owner query byte for byte. The edge audit accepts an explicit query-row domain without inventing global anatomical IDs.

## Measurements
Both whole-volume HiGHS methods reached 90-second limits without a usable incumbent. Failed trials and as-run sources are retained.

The sparse run 002 completed four construction updates in 33.4 seconds. These are offline authoring timings, not simulation throughput. Run 004 accumulated incompatible inequality equalities and stalled; its failed source is retained. Run 005 corrects that approach with currently violated constraints and a monotone penalized merit line search.

On the same 9,320-node / 37,935-cell refined topology, run 005 ends at 7,763 edge/bone groups and 873 contained nodes, versus 8,618 groups and 1,028 contained nodes in its refined baseline. It has no nonpositive cells or quality values below 1e-5, J=0.357781..2.031048, boundary area ratios 0.643054..1.948580 and minimum normal cosine 0.872015. Maximum cumulative movement is 12.5782 mm. These results are insufficient for admission; further unconstrained drift near quality limits is not the next repair method.

Refinement adds query nodes on existing crossing chords, so these raw counts must not be compared with the older unrefined candidate's 437 groups. Positive local cells do not prove global injectivity, skin clearance, valid detailed transfer, anatomy or native dynamics.

## Checks and limitations
34 focused Python checks pass, including analytic gradients, rigid/affine displacement units, finite domains, anchored nodes, legacy parity, exact boundary gates, local inequality merit and query identity handling. The static query CMake target builds with warning checks and byte-matches baseline/refined output.

Full surface-solid intersection, detailed 294 skin transfer, mass/source ownership integration and native physical qualification remain pending. No image was opened. Runtime FEM, articulated, circulation, physiology and accepted-state owners are unchanged.

# Numi Human P188: connected hand patch diagnostics

Construction diagnostics on SSH Numi continuing P187 commit 0e7a8b97c1e89657bfa82743c414610edd3a01fd. The best complete static audit still has 653 comparable parent skin/bone intersection identities, 125 sampled inside node/bank pairs, zero skin self-intersections, and 45 skin boundary edges. The candidate remains unadmitted.

Bank 109/114's 187 refined crossing pairs across 111 skin faces lie in two distinct edge-connected patches after boundary expansion and one unambiguous four-face hole fill. The two face sets contain 239 and 38 faces, with simple degree-two boundary loops of 49 and 14 edges. Exact rational target-bone edge checks found no boundary hits or inside boundary nodes. Their minimum float64 edge distances are 0.04600094 and 0.22048631 mm; no anatomical thickness threshold was imposed. Stored neighboring-bank crossing pairs did not touch the boundaries. No geometry was changed.

For 43/16 the previously saved alpha-0.5 connected fit was identified as an already screened failure: 985 refined/653 parent pairs unchanged, while its fixed-reference minimum area ratio was 0.1000096 against a 0.10 floor. It was not rerun or admitted.

These local boundary checks do not prove clearance of patch interiors, skin/fat material validity, physical transfer, moving poses or native acceptance. Source and mixed-reference attribution are retained in P182/P183.

The 115/118 diagnostic now reconstructs 46 rational segment pairs as a single 46-edge cycle. A local face-wise topology prototype preserves the 25-edge outer boundary, parent-face area coverage and shared contour incidence, with rational ancestry and single shared Float32 endpoints. It is a patch-only artifact, with no bone-clearance or self-intersection checks, no integrated mesh and no physical admission. Its shared endpoints represent the intersection contour after Float32 rounding; they are a construction intermediate and do not constitute a cleared surface.

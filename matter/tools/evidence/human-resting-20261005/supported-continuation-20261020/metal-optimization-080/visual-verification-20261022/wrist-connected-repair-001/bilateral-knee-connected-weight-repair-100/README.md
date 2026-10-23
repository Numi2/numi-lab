# Bilateral knee transition binding repair

Candidate 100 extends candidate 091 using the existing NHSKIN5 reference rebind owner. It changes 68 inferred skin-weight rows through same-side femur/tibia exchange; the neutral surface changes by at most 13.87 nm. Detailed topology, bindings, stored normals and every unedited vertex record remain exact. This is an inferred reference assembly, not measured attachment data.

A dense SLSQP attempt failed its inequality constraints and is retained in fit-report.json. The replacement uses sparse, triangle-local Jacobian coloring with bounded trust-region least squares. The 186-variable, 558-face fit took 1.79 seconds on SSH numi. It reached its 100-evaluation limit (not an optimizer convergence claim), but its returned vector satisfies the independently checked constraints. Maximum absolute weight exchange is 0.0877357. Exact replay of the saved vector regenerates the emitted NHSKIN byte-for-byte.

The emitted Float32 source passes full bilateral knee regional stretch and area checks at 0.25, 0.5, 1.0 and 1.5 radians. At 1.5 radians, minimum stretch is 0.354999315, maximum stretch 2.38993216 and minimum area ratio 0.27437175. Candidate 091 had 28 regional failures there; candidate 100 has none. These thresholds (minimum stretch 0.35, maximum 2.4, area 0.25) are engineering diagnostics, not physiological limits. Forefoot results remain unchanged.

Exact full-surface probes across the 28 retained poses have no new triangle-pair witnesses versus 091. Existing shoulder/elbow failures and all 155 deep two-radian limb crossing pairs remain. The two-radian knee pose also still fails regional strain checks. This does not establish full knee range of motion, physical contact correctness or complete anatomy.

Source-only status: source normals/donor mappings, physical tissue/mass and accepted-state behavior require candidate-specific regeneration and native checks in D35. No source report here admits this as physical skin. The physical layers and simulation state are not changed by this source preparation.

Payload SHA-256: 27368e4bb9a0f59f41e67241a61b17de515a6ffb651b36dce6d47a81902869f3. See ../forefoot-connected-weight-repair-091/ATTRIBUTION.md for upstream BodyParts3D/MyoSim credits and licenses. Candidate 100 adds only the described inferred binding/reference-coordinate changes; retain those credits with this derivative.

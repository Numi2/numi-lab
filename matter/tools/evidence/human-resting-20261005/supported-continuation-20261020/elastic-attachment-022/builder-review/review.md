# Elastic skin attachment builder review

Read-only source review of the preparation-022 elastic skin attachment builder on 2026-10-20. No source edits, builds, native runs, GPU work, or visual inspection. This is an implementation review, not physical or biological qualification.

## Findings

The builder creates 20 elastic attachment samples from the frozen whole-body skin reduction. The ordered source-vertex/body-site identities in `apps/NumiHumanWholeBodySkinReduction.hpp:12-33` match the reconstructed original-20 set; all 20 remain within current confidence and geometric bounds. Current minimum confidence is 0.8524638414 at (skin local vertex 790, source vertex 13555, donor site 20). Current minimum incident altitude is 0.0002354949856 m at (local 692, source 8467, donor site 139). These are numeric donor-site IDs; no anatomical class is inferred here.

The builder uses SI units consistently. Matter parsing converts MPa inputs to Pa (`matter/src/language_impl.cpp:220-222,468-479`). The effective adipose shear modulus is 0.000551724138 MPa = 551.724 Pa; the allowed scale is 0.5ºw^~)Þw1.5. For each sample, stiffness is based on ºw^~)ÞtA/h (N/m), with area in m² and height in m; the reciprocal-modulus term closes strain over the same reference thickness. The scale is finite and bounded.

Reference-frame handling is coherent: donor transforms are checked as proper orthonormal transforms in `apps/NumiHumanSkinRigidPlacement.hpp:52-64`; donor-frame composition is in :29-35. The target point and normal are expressed through the mapped body-local reference coordinates. Rebase updates the elastic sample point together with its hard-anchor rows (`apps/numilab_human_myosim_visual_probe.mm:21108-21147)).

Force ownership is explicit. The shared ABI documents elastic-attachment sample semantics (`matter/include/numi/matter/numi_human_shared.h:131-152)). The Metal path applies force to the FEM node and the equal/opposite wrench to the body (`matter/src/metal/numi_human.metalinc:1006-1010,1042-1050)). Samples enter `fasciaBodyContactSamples` and bind once in either the standalone initialization path (`apps/numilab_human_myosim_visual_probe.mm:21231-21237`) or the integrated tendon/FEM path (:30011-30020); the integrated branch asserts the expected count (:30040-30042). Validation rejects duplicate contacted nodes and nodes also covered by active hard anchors (`matter/src/numi_human.mm:812-843)).

The builder adds no mass rows or mass values; this is a force-only sample ABI. Existing donor mass partition and assembly checks remain in place (`apps/numilab_human_myosim_visual_probe.mm:21149-21155,30061-30065`). One narrow audit limitation: the local builder check only requires positive node mass, rather than bitwise comparing it with baseline. The inspected builder does not modify mass, and the assembly partition checks provide the broader conservation guard.

## Review boundary

Source evidence supports unit/frame consistency, one force path, frozen identities, finite bounded stiffness, and unchanged mass construction. It does not establish measured tissue response or physical realism. GPU fixture verification is owned by the separate verifier.

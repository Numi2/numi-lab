# Registered passive-tissue integration evidence

This candidate connects an opt-in registered aortic wall to the existing CVSim21 graph and adds opt-in passive FEM owners for a skin/fat patch and a registered prostate surface. These are bounded owner-execution checks with explicit reference-pose and material assumptions. They do not qualify clinical circulation, tissue response, or human physiology.

## Validation

- The frozen 006 app compiled and linked against the pinned 233 Matter runtime. Binary SHA-256: 6dfe478188d64850e518ed5174fa564799d5662819babf9f3ed330f92ed2e798. Runtime library SHA-256: 71372f858367c5945eed34e7334f184f210f06c7cd21e6ec48cb340a4aebd60a.
- The CPU regression for translated-star centroid, volume/mass closure, and node/face ancestry passed; see cpu/centroid-239.json.
- The skin/fat owner is a localized registered BodyParts3D patch: 64 source faces, 50 vertices, 150 shared FEM nodes, and 192 skin plus 192 fat tetrahedra. It is not full-body skin.
- The prostate owner is one passive proxy surface: 223 source vertices, 442 faces, 224 FEM nodes, and 442 tetrahedra, with a 0.0118535515 kg donor debit. Its material is an effective passive proxy, not a prostate-specific fit.
- The vascular addition is one closed, capped aortic reservoir slice bound to existing CVSim21 compartment row 7. It is not a patent through-flow artery or a complete vascular tree.
- Three separate native runs accepted one 0.25 ms step each: wall only, prostate only, and joint skin/fat + wall + prostate. Blood MPM was disabled. Logs and invocations are under native/native-run-009/.
- A two-step 0.5 ms joint clean control and a late rejection at step 1 followed by same-context retry both completed. Rollback checks passed and the accepted resting-coupled.csv outputs were byte-identical. Receipts are under native/native-run-010-joint-rollback/.
- Visual verification was not performed.

## Limits and next measurement

These runs establish short-horizon integration and rollback behavior only. They do not establish long-horizon stability, real-time performance, or biological validity. Exact source inputs are pinned in source-pins-010.json.

The 006 joint one-step profile reported: resting_native_profile step=1 command_ms=69571.923125 gpu_ms=69570.9089583 host_copy_ms=0.288833 full_state_readback=1
The corresponding final render profile reported: resting_render_profile step=0 vertices=2920390 triangles=5239445 geometry_export=0 encode_wall_ms=0.160833296832 command_wall_ms=68.6368750175 gpu_ms=21.7009999906 audit_export_wall_ms=0.0646666740067 total_render_wall_ms=68.8763333601
The long command time is in physics command-buffer GPU work; separately reported render GPU time is small. Existing Matter stage timing samples only root 100, so this evidence does not identify the dominant solver kernel. Start optimization with a separately pinned profiling build and one measured owner path. Do not change mesh resolution, solver budgets, tolerances, or physical assumptions until the profile attributes the cost.

Skin/fat and prostate owners are opt-in. No nerve FEM owner was qualified. Blood MPM remains disabled in these validations. The generated flattened respiration Metal source is retained under generated/ as an exact build snapshot, not a canonical source edit.

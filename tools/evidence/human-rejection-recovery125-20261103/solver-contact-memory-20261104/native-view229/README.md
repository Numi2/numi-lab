# Accepted Human native render, 2026-11-04

A fresh full-scene run on SSH Numi / Apple M6 accepted 32 steps of 15.625 microseconds, reaching 0.500000024 ms. Body and physiology clocks matched; all ten exported CSVs were byte-identical to retained control226. The simulation loop took 229.663281541 s (2.177100407e-6 real-time factor); launch including preparation took 436.223511934 s. This is a release transient, not settled rest or sustained functional qualification. Zero complete breaths occurred.

Trials227 and228 stopped at the 12 GiB disk-reserve guard before a physical step. They remain in the original evidence workspace. An idle 2.8 GiB ANE compiler allocation was released before retry229; this recovery is not a physics optimization measurement.

The existing Metal renderer presents captured accepted GPU vertices. Four derived ocular instances remained articulated in the exported pack. The preview now resolves these using the matching existing accepted-body receipt, verifies pack content identity, and frames transformed source geometry for supine inspection. Default BodyParts3D semantic admission remains unchanged; Human coverage is explicit. Native three-angle rendering and four invalid-mode/receipt rejection cases passed.

skin-anterior.png and skin-oblique.png select resting Skin inspection. all-layers-anterior.png displays internal inspection surfaces too. The red areas are red anatomical meshes, not blood leakage: particle blood and bleeding were disabled in this configuration, while reduced circulation remained active. Visibility selection does not repair or certify skin/internal intersections. No vertex coordinates, material laws, forces, or physical acceptance gates were changed.

Remaining gaps:
- Audit the red internal surfaces against outer skin at the exact accepted pose; the all-layer image alone cannot classify penetrations versus inspection overlays. Finish wrist/hand/joint shape and motion checks.
- Export inspection masks and all derived instance poses faithfully so generic replay reproduces the native viewer without a preset.
- Establish a supported, settled operating point. This run was explicitly unbalanced release initialization.
- Resolve active linear-solver convergence/conditioning, repeated assembly and contact-certification costs, plus startup working-set pressure, to reach meaningful duration.
- Qualify repeated mechanical breathing and heartbeats, circulation/gas feedback, rejected-step rollback and the full baseline/intervention with this physical skin/fat configuration. The short render run establishes none of those longer outcomes.

Reproduce the render with the pinned preview executable:

    metalrobo_bodyparts3d_visual_probe STEP32.mrvpack OUTPUT --dimension 1024 --source-semantic-coverage --supine-view --accepted-receipt STEP32.receipt.json --human-resting-skin

The launch plan and build manifest pin original paths, binary, libraries, shaders, source and inputs. Full MRVs/receipts remain in native-scene-view-229; compact traces and image/binary hashes are retained here. No new simulation owner or artifact format was added.

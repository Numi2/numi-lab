# Opt-in contoured-bed support: implementation and evidence

This publication adds an opt-in, finite fixed-world heightfield support surface to the native resting scene. The scene query supplies each selected contact's plane to the existing contact solve. The default flat-bed path remains byte-identical in the retained disabled comparison; the contoured path is not an anatomy-qualified or clinically validated resting model.

## Source and focused checks

The source files in this commit are pinned individually in `source-pins.json` and correspond to the retained `/Users/n/numi-human-contoured-bed-build-1196` build. The final focused CTest run passed all five relevant tests: `numi_human_support_precision`, `numi_human_per_contact_plane_metal`, `numi_human_static_support`, `human.metal.resting_bed_query`, and `human.resting_bed_manifest_admission`. The legacy visual-probe compatibility build also passed. Logs are retained in `build-tests/`; no binaries or build outputs are included.

The flat-bed disabled reference, 1197 attempt 002, completed 10,000 accepted steps (20 simulated seconds). Its four CSV traces match the prior 1191 reference byte-for-byte. A first parity report is retained and says whole MRV packs differ because metadata differs. The corrected `accepted-state-geometry-parity-v2.json` comparison checks the non-metadata pack sections byte-for-byte and permits only the explicitly bound identity strings; it passes. The first disabled startup attempt is retained: Metal required explicit specialization of the optional bed-audit function even when the feature was disabled.

## Opt-in 20-second smoke and anatomy result

The contoured run, 1198 attempt 002, completed 10,000 accepted steps at 2 ms and 72 kg, with zero root assistance. The contact-trace audit passed its declared checks over 40,000 rows (32 contacts at each of 1,250 accepted eight-step endpoints). It found no negative normal impulses above its threshold and no friction-cone violations above its threshold. Maximum sampled endpoint coupled-contact penetration was 49.791 µm. These are endpoint samples; the CSV does not contain all eight substep impulses and does not establish full-step impulse closure.

The mesh/capture audit verified the 5 mm heightfield's 120,321 Float32 node positions and 239,200 triangle indices against the declared runtime lattice, and verified the non-bed initial scene records against the source capture. The recording review reads all compressed image samples and decodes ten selected frames; it does not decode every image frame.

The separate exact terminal skin-target audit failed: at accepted step 10,000 it found 3,727 NHSKIN-to-target triangle intersections across 26 of 859 target surfaces, with zero NHSKIN self-intersections and zero invalid target triangles. This was a terminal-only scan; the declared eight-capture anatomy scan was not completed. The run is therefore retained as an implementation/contact smoke only. The contoured bed is **not adopted or qualified**, and no long-horizon or physiological claim is made.

## Authored bed provenance

The 5 mm bed is an authored fixed-world surface derived from one accepted source-pose NHSKIN capture. It is not a measured mattress, participant-specific surface, pressure map, or clinically validated contour. The source scene manifest declares BodyParts3D CC BY 4.0 and MyoSim Apache 2.0 provenance for its anatomical inputs. The full bed manifest, height arrays, source skin, contact payload, anatomy, native movie, and MRV packs remain at their retained absolute paths and are hash-pinned in `external-artifacts.json`; no large runtime asset or capture is copied into this repository bundle.

## Profile comparison and limits

`profiles/1197-vs-1198/profile-comparison.json` and its replay script compare the retained flat and contoured 20-second profiles. Native GPU time averaged 130.316 ms versus 176.776 ms per profiled eight-step segment (+46.459 ms, +35.7%); integrated body wall time increased by 57.260 s over 20 simulated seconds. Regular render GPU time changed by 0.206 ms per frame (+1.6%). The logs do not time bed-query/contact-plane kernels separately. The comparison is two retained attempts, not a repeated controlled benchmark; their native executable hashes differ, although they share the base revision and loaded MetalRobo and shader pins.

The failed native startup attempts and audit-driver startup attempts are enumerated in `attempt-history.json`; their logs and diagnostics are preserved under `native/`. The external index rehashes 390 referenced paths. Two historical mutable paths now hold later versions: the 1197 run's old executable hash is resolved by its retained binary snapshot, and the old source-header hash is resolved by the retained 1197 runtime-source snapshot. Both discrepancies remain explicit in the index.

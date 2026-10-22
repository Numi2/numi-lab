# Native arm/hand actuation increment, 2026-10-23

This increment retains the existing articulated Metal/MyoSim/compliant-tendon owners. It adds paired accepted-state hand/wrist telemetry, correct normalized tendon-tension units, and a named initial configuration for selected-tendon tests. The old unqualified render-only radial wrist edit is disabled by default and available only with its explicit diagnostic environment switch.

The initial-reference preset seeds q36 and q74 to 0.1 rad once. It adds no ongoing joint constraint, root assistance or force. Every hand/wrist coordinate remains free except the model's authored joint limits. Admission rejects history, continuum-owner and conflicting pose/certificate combinations.

The left APL origin force-map envelope is regenerated through the existing compiler with amplification <=2.2. The original endpoint, body, source route and bone identity are unchanged; no attachment point is migrated. The original rejected envelope and failure receipts remain on SSH Numi.

## Executed evidence

- Isolated scoped native build: source SHA 0b4347a7ec00381bc8baa1f02cc83a09260e2dfe2f4de0048de1d202ee3cc8c5, binary SHA 6596e3f1b9a11708924443cfe4ad4c144a3fb90060c0122db4333f38f6cf5b0c. See scoped-native-build-manifest-003.json for dependency, shader and linked-library identities. Shared unrelated dirty work was not included in the published source patch.
- Final scoped build: paired one-step finger-flexion transaction accepted all 44 hand/wrist coordinates, bitwise replay, finite values, authored joint limits within the native owner's existing Float32 slop, and nonzero source force/torque/joint response. 30.2867 s native wall time includes startup, loading and serialization; it is not isolated solver throughput.
- Earlier exact working-source binary 245f9676...: 80 accepted 0.125 ms steps (10 ms total) each for bilateral finger flexion and extension, bitwise replay. Both actual endpoint poses pass exact Float32 self-intersection, surrounding-skin and bone-intersection checks on candidate060, both sides. Separate representative winding screens classify the 27 checked distal bone members per side inside the skin at all eight retained static poses.
- Separate one-step tests exercise wrist flexion/extension, thumb direct tendons, thumb opposition and opposing intrinsic finger-muscle cohorts. These are force-transfer tests, not grasp validation.
- The default-q initialization produces real thumb/thigh crossings; its failure is retained and is not an accepted resting anatomy result. Raw strict zero-limit failures are retained separately from the audit using the solver's existing 16*FLT_EPSILON bound; no joint state was clamped for a report.
- Initial 800-step trials with a 60 s watchdog timed out without an endpoint. They are retained as timeouts, not numerical success/failure. A later 800-step flexion run completed in 67.2269 s but its helper rejected zero-force channels; the raw result is retained pending slack/constitutive diagnosis. The matched extension was explicitly interrupted for resource coordination and is not a physics failure. Skin/fat integration remains incomplete.

## Reproduce

Use run_hand_actuation_cohort.py --binary <native-binary> --binary-sha256 <hash> --source-manifest <build-manifest> --cohort bilateral_finger_flexion --initial-reference-posture --skin-payload <pinned-060-NHSKIN> --skin-sha256 e30bd35e4c7cee48433c50cafb0987f23987a95d377034aecc017aea28d28ae1 --step-count 80 --timestep-seconds 0.000125 --execute --output <new-output-directory>.

The helper pins existing source payloads and records exact commands, hashes, compiled-source identity and accepted-state telemetry. Asset locations in receipts are SSH Numi paths; no new artifact ABI is introduced.

## Remaining limits

The rebuilt detailed hand source has not yet been admitted with regenerated physical skin/fat layers. The existing coarse layers inherit over-thick palm fields into fingers and one inferred capitate anchor is invalid; those failures are retained and being repaired. The 066 crease candidate is also undergoing full geometry checks. These native core tests do not qualify whole-body FEM/contact, grasping, continuous-path skin clearance, the 16/60-second integrated scene or the full coupled physiology deliverable. No clinical validation or completed anatomy claim is made.

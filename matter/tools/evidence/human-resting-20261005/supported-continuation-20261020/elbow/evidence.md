# Elbow continuation evidence

This package records the bounded elbow increment at the frozen source state. It is evidence for an inferred regional RCL proxy in the existing registered rigid model; it does not establish subject-specific ligament anatomy or material properties.

## Frozen inputs

- Repository: /Users/numi/numi-human-aortic-contact-opt-20261019, HEAD a143f2a0.
- Registered ligament header SHA-256: a1d5adb7a6fc601b0e97dcebe1bbb2d36010bdcefe935bf424943d9b56528d25.
- Native invocation: /Users/numi/numi-human-aortic-contact-opt-20261019/build-integrated-validation-20261019/supported-skin-ligaments-20261020/native-supported-ligaments26-009/invocation.json, SHA-256 d66117177c274e736753d51ff7644f6a33b42dd98dd2e95760da7398d97dfeb5.
- Source rigid SHA-256: 2c78cb4150b97cea6e8169dad9e8f5dd59af667b247e07b56e48e857435560e4.
- Source bones SHA-256: 43a31ef4771e2a383ae2c7caab1b994f0c28076c2a43501860cd1ba3fe7708c0 (ABI 3, 185 records).
- Source myotendon SHA-256: e5bb8a8168706bb3b23cf42b1d849659569ba2e6027bcc85729c3110fdbb8c4b.
- Native pose overrides: q36=0.1 and q74=0.1; the audit replays exact source joint FK and checks anchors against all 185 registered bone meshes.

## CPU fixture and geometry audit

cpu_fixture_command_output.txt preserves the final focused fixture command and its exact output. It passed with 28 catalog rows and a 42,000 N/m synthetic stiffness check.

actual_pose_catalog_audit.json is the exact-pose source-geometry audit. The JSON file SHA-256 is 1e346c4202051a1578de3cf16e9f77deda1a7cfd5f91b617ba09887d1635a037; its embedded canonical audit digest is 3958a73dc185f2d5c00df28666deb83d44d227026a80472008891a8feeedfe36. It records 28/28 admitted rows, the exact invocation/header/binary/source hashes, per-endpoint projections and normals, float32 body-local reconstruction, and segment checks against all 185 registered meshes. Surface meshes are open, so triangle-clearance does not prove closed-solid containment. This audit qualifies source geometry admission only, not live GPU force behavior.

## Admitted elbow row

The admitted bilateral RCL row uses the registered elbow frame Humerus->Ulna (joint 41 right, 91 left), but its second attachment is on the independently articulated radius. The implementation therefore resolves each endpoint through its own body transform; it does not treat the joint-frame child as the second ligament endpoint.

- Right source bodies/stable IDs: humerus body 41 / ID 12 to radius body 43 / ID 16.
- Left source bodies/stable IDs: humerus body 91 / ID 13 to radius body 93 / ID 17.
- Region: lateral epicondylar origin and radial-head insertion proxies. The source does not contain measured RCL enthesis patches; no subject-specific attachment claim is made.
- Source route landmarks for the bounded regional search: ECRB route nodes 763/1215, sites 682/1103; source myotendon payload hash above.
- Registered ulna-radius joints are 42 right and 92 left. No separate annular ligament owner is instantiated; the RCL is represented by a single inferred line, with no added mass.

## Excluded candidate groups and bounded search

bounded_candidate_search.py is the exact helper used for region-constrained endpoint-patch search. The bounded candidate neighborhoods were centered on registered muscle route landmarks: MCL anterior FCU to brachialis; MCL posterior pronator teres to triceps long head; LUCL ECRB to supinator. Search settings were endpoint triangle-centroid radius 12 mm, up to 100 candidates per endpoint, candidate line lengths 5-60 mm, and non-inward interpolated surface-normal cosine >= -1e-4. Candidate paths were checked against all 185 registered meshes using AABB culling plus scale-relative segment/triangle tests.

Final bounded counts:
- MCL anterior: candidate endpoint counts L 100/68 and R 100/69; 154/96 non-inward, length-compatible pairs; 0 clear paths.
- MCL posterior: L 100/46 and R 100/72; 0 non-inward, length-compatible pairs; 0 clear paths.
- LUCL: L 100/13 and R 100/25; 80/417 non-inward, length-compatible pairs; 0 clear paths.

These finite candidate neighborhoods do not show that the anatomical ligaments are absent. They show that the current straight-line, registered-surface owner did not admit these candidates under the declared regions and clearance rule. They remain excluded until a source-supported wrapped-path/route owner can represent them without fake anchors.

## Stiffness prior

The RCL tangent uses three-bundle aggregate stiffness: 15.5 N/mm per bundle x 3 = 46.5 N/mm = 46,500 N/m. The model's 10 mm^2 area is a normalization used to encode that tangent in the existing stress law, not a measured subject-specific CSA. The coefficient is a literature prior, not an individual measurement.

Primary source: Rahman et al. (2014), Table 3, Development and validation of a computational multibody model of the elbow joint, DOI 10.12989/aba.2014.1.3.169, publisher article https://www.techno-press.com/papers/358, publisher PDF https://www.techno-press.org/download2.php?journal=aba&num=3&ordernum=3&volume=1. Table 3 reports RCL bundle linear-region stiffness alpha = 15.5 N/mm (three bundles); the implemented aggregate is the three-bundle sum. This is an elbow-model parameter reference, not a subject-specific tensile measurement.

## Test scope

The focused CPU fixture passed. The exact-pose catalog audit passed 28/28 source-geometry rows. This package does not claim that the new 28-row owner has completed a native GPU replay; the parent owns integrated build and native replay.

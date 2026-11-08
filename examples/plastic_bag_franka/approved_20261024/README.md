# Approved thin plastic bag and Franka snapshot

This is the preserved research solver and scene accepted on 24 October 2026.
The approved solver is an immutable example; later empty-bag work has a separate source directory.

## Native run

- 8 seconds, 240 Hz, all 1,921 saved states.
- Folded and side-welded LDPE pouch: 200 × 230 mm, 75 µm film, 6.520 g; 1,081 dynamic nodes, zero pinned nodes.
- Ten finite cylinder coins: 30.61 mm diameter, 2.15 mm thickness, 11.34 g each.
- Franka FK drives finite compliant rubber pads, with normal and frictional contact on the film. The film is not attached to selected vertices.
- Air drop, approach and corner pickup, 0.323 m measured corner lift, release, second table impact.
- Metal integrates the dynamics. Blender renders the recorded native states without advancing physics.

The complete archive includes the native binary and metallib, source and historical segment sources, inputs, original checkpoints, complete trajectories, assets, audits and both videos.

[Download the preservation release](https://github.com/Numi2/numi-lab/releases/tag/plastic-bag-franka-approved-2026-10-24).

Archive SHA-256: `2e7e4456002bc77c4b67855bc6a4d89003f51b89e88bf43f0850a85c99f022ad`. Its internal `MANIFEST.json` contains hashes and sizes for every payload file.

## Evidence and limits

See [delivery receipt](evidence/delivery_receipt.json), [assessment](evidence/assessment.json), [source identities](evidence/combined_identity_sha256.txt) and [checkpoint lineage](evidence/lineage.json). Historical source segments were joined through exact checkpoints; the final source is not claimed to regenerate earlier segments bitwise. The archive is the replay and continuation boundary.

The measured final maximum coin speed is 0.002073 m/s. Saved states pass the recorded table, film, pair contact, energy and robot housing clearance criteria. These numerical checks do not establish real-world material accuracy. Material parameters remain uncalibrated, aerodynamic drag is absent, and robot joints follow a prescribed path without motor or whole-body collision dynamics. Robot mesh clearance was checked at saved states, not with continuous whole-robot collision detection.

The source retains reciprocal MPR/EPA retry for unresolved thin-face queries. The unsuccessful cap-feature probe is preserved as unadopted evidence. No failed result has been substituted for the approved trajectory.

## Build and replay

`build_only.sh` records the exact original remote build location and commands. To rebuild elsewhere, update its `SCENE_DIR` and compiled `PLASTIC_BAG_METALLIB` path to the new absolute location. Compile with Apple command line tools and Metal on Apple silicon. Execute from the directory containing `initial_state.json`, `robot.json`, and `grip.csv`. Preserve the supplied checkpoint inputs when continuing a segment.

Rendering helpers use Blender and the pinned Franka assets in the release archive. Their recorded absolute input paths identify the original experiment workspace. They are research scripts, not the Numi Lab production renderer or portable command-line interface.

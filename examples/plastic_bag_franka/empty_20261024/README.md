# Empty plastic bag: drop, corner pickup, high lift and release

This scene keeps the approved 75 micrometre film, density, membrane and bending laws, contact/friction equations, solver tolerances and active film iterations. Active coin count is zero. No film nodes are fixed or attached to a selected robot node. Actual finite fingertip normal and friction contacts lift the opening corner.

The requested eight second trajectory has 1,921 saved native states at 240 Hz. The robot raises its contact centre by 0.50 m, moves it 0.10 m outboard and turns the wrist after closing so the naturally unloaded sheet clears the stock finger housing. Robot poses are forward kinematics from constrained joint positions. The actual film corner height, contact alignment and full saved-state mesh clearance must pass in the completed delivery receipt.

## Continuation lineage

- `first_drop`: steps 0–240, empty free drop, parked robot.
- `prefix`: exact accepted states 0–720, using the tilted approach and finite corner pinch from the fixed-tilt attempt.
- `combined`: resumes the exact step 720 checkpoint with the new future wrist path; no state interpolation, render-engine dynamics or loaded-shape forcing.
- Rejected top-down approach and fixed-tilt lift outputs are retained separately. They have stock housing intersections and are not successful demonstrations.

## Performance scope

Only coin-only GPU dispatches are omitted for active coin count zero. Film self contact, membrane, hinge, static table contact and all table-friction rounds remain active. The retained-empty-work comparison must match state, final checkpoint and accepted budgets byte for byte. Report actual paired timings without extrapolating to coin-loaded scenes or full-scene throughput.

## Limits

Material parameters are uncalibrated. Air drag is absent, matching the approved comparison. Robot joints are prescribed; this is not coupled motor dynamics. Full robot housing clearance is audited at each saved state, not by continuous whole-robot collision detection. Video rendering presents saved native states and adds no physics.

The native summary retains a historical `release_event` text string describing a loaded 0.32 m lift. `scene_plan.json`, actual coin count, GPU checkpoint, recorded path and measured corner displacement describe this empty 0.50 m scene; the legacy string is not its measurement.


## Replay and restoration

The release archive includes the exact native executable, Metal library, source, path inputs, checkpoint segments, robot FK source/headers, presentation assets and a per-file SHA-256 manifest. The solver executable was built on macmini (Apple M4 Pro). Its embedded Metal library path is recorded in `build_only.sh`; keep that path when restoring the original binary, or rebuild in a new directory with the path adjusted.

To continue the preserved accepted state, place `solver/grip.csv`, `solver/robot.json` and `solver/initial_state.json` in the working directory, restore `evidence/prefix/checkpoint_720.bin` under `prefix/`, and run the supplied executable with:

```sh
./build/numi_classic_bag_drop --empty-bag --output combined --resume prefix/checkpoint_720.bin --steps 1920 --checkpoint-every 120
```

The preserved replay contains the complete first drop and accepted states 0–720 plus the exact continuation 720–1920. Join at the shared checkpoint state; do not duplicate step 720. Full from-start reintegration is a separate reproducibility experiment, not an additional completed measurement in this delivery.

Presentation helpers retain absolute asset/output paths from this task. Restore those paths or edit their path constants when rendering elsewhere. They consume saved native states; they do not run Blender physics. `--retain-empty-work` selects the otherwise identical empty-work comparison used for the paired timing measurements.

240 Hz describes the simulation step, not real-time throughput. The completed 720–1920 continuation simulated five seconds in 1,757.28 wall-clock seconds on Apple M4 Pro; raw GPU and wall timing records are preserved. The paired optimization result is scoped to the measured eight-interval windows.

The repository example lists the active empty-scene helpers only. The release archive conservatively retains inherited historical helpers and snapshots; the entry points described here govern this delivery.

[Independent GitHub archive download and restoration verification](evidence/github_download_verification.json) confirms all 531 payload hashes. The approved ten-coin example and video files remain unchanged.

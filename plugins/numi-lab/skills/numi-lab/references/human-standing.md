# Numi Human standing and embodied execution

When the user asks for Human to stand, move, or function as a complete body,
start the existing native whole-body run as soon as its inputs and device are
resolved. Bound discovery; a stalled `numi context` is a reason to inspect the
owner directly, not to repeat inventory. Verify the current source and binary
because installed overlays and retained builds can belong to different revisions.

Work against the first observed physical failure in that run. Prioritize
muscle activation and fibre/tendon dynamics, skeletal articulation, joint
constraints, gravity and unilateral foot contact. Keep one accepted physical
state, and bind visualization and control to that state. Never substitute a
robot demonstrator, static anatomy image, separate Brain simulation, hidden
root wrench, direct joint torque, or artificial pose lock for muscle-driven
Human standing. Explicit assistance is a diagnostic with its own label.

Use short execution prefixes to locate a failure, then extend the corrected
run toward at least 10 simulated seconds of unassisted loaded standing. Report
accepted simulated time separately from wall time, muscle/tendon and support
loads, falls or rejected steps, and the limiting failure. A 1:1 anatomical
claim needs separate source and calibration evidence; a standing result alone
does not establish it. If throughput prevents useful-duration execution,
profile and repair the dominant native path while retaining physical checks.

For sustained runs, `numi human stand ... --execute` omits mandatory replay
and retained per-step traces while preserving native physical admission.
`--muscle-feedback <length-gain> <velocity-gain-seconds>` observes committed
fibre state; `--muscle-path-feedback` observes accepted joint q/v through a
fixed source-path calibration. Both are explicit experimental excitation
controllers. Compare them with the same excitation-only baseline and do not
present them as an integrated NumiBrain run.

For the source whole-body standing scene, explicitly request
`--execute --steps 10000 --timestep 0.001 --muscle-path-feedback 10 1` and
supply the matching bone and muscle-surface payloads. The wrapper retains
64 contact iterations and the current-state source passive joint law.
The command's default short horizon is not a ten-second standing run;
judge the requested horizon from the actual accepted trajectory and outcome.

For the integrated Brain/Human path, run the published
`.numi/commands/human-brain-standing` on the Apple-native owner with the
matching Brain dynamic library. Its launcher defaults to four one-millisecond
steps and accepts an explicit longer horizon up to ten simulated seconds; it
renders the accepted skeleton and muscle surfaces. Check its run
manifest, `human_brain_joint_commit`, and `human_standing_progress` records.
The original default Brain program relayed source-prepared tonic excitation
after the first receptor frame, with zero excitation on step one because the
initial receptor validity was empty. The updated native source seeds only
initial path length and velocity from the prepared MyoSim result; remeasure
the first motor command on the named binary before claiming that fix executes.
The original short reproducible integration is not sustained
Brain-controlled standing or evidence of recovery from a perturbation. For
that claim, run a source-bound feedback program and compare matched
controller-on/off
physical trajectories under the same timed push before extending to at least
10 simulated seconds without root assistance.

For a short energy diagnosis, add `--endpoint-energy` to `--execute`.
It reports mass/inertia kinetic energy and available work at accepted native
endpoints. Keep solver-iterate correction work separate; missing source-limit,
projection and internal muscle energy terms do not establish energy closure.

Run only checks needed to validate the change and its actual physical outcome.
Do not turn a standing request into broad library tests, new evidence schemas,
ownership abstractions, documentation campaigns, or repeated millisecond-only
qualification. Finish with the runnable scene and measured outcome, or the
specific unresolved execution blocker and retained run, never test counts as
the standing result.

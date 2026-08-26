# Numi Crow Journey v7

`birdflow_american_crow_journey_v7` is the Apple-native training task for one
command-conditioned American-crow hybrid actor. It is separate from the older
standing-to-flight qualification task and has distinct task, observation, and
action fingerprints.

The task has 15 normalized actions, 84 actor/critic observations, and eleven
difficulty bands:

0. supported stand
1. walk or hop
2. takeoff
3. airborne straight-cruise stabilization
4. ground takeoff followed by straight cruise
5. left constant-curvature turn
6. right constant-curvature turn
7. approach
8. touchdown
9. landed hold
10. full 32-second journey

Band 4 is the first deployment gate. At 50 Hz it provides one second of
supported stand, up to four seconds for takeoff, and eight seconds of cruise in
a 650-step episode. A candidate must have no failed environment steps or
physical-boundary terminations, reach 0.55 m root height, achieve at least 0.65
mean tracking, remain below 0.35 mean tilt, and remain below 0.80 maximum tilt.
The held-out selector evaluates every earlier band to prevent skill regression.
`--advance-candidate` cannot bypass this selection for the journey task.

Journey v3 added a bounded pitch-moment action because the articulated tail and
pronation lanes could not arrest the measured landing attitude before wing
fold. The selected v3 actor cleared 95 of 96 full-journey trials, but one
randomized approach crossed the tilt boundary. A v4 soft pitch envelope failed
7 of 640 repeated journeys. V5 closed the pitch loop throughout approach but
still failed the original randomized landing. V6 coupled the teacher action
vector after the 21-second approach boundary, reducing but not eliminating the
rare failures. These results are retained as rejected evidence.

V7 makes the final authority boundary explicit. Before 18 seconds, the neural
actor has full authority. From 18 seconds through landed hold, accepted pitch
continuously activates the native teacher action vector from 0.16 through 0.22
rad. The network retains full authority at or below the lower boundary and the
supervisor owns the coupled recovery vector at or above the upper boundary.
The pitch lane additionally closes a proportional-and-rate loop over the
existing body-moment actuator from 21 seconds onward. Both paths use accepted
state on Metal; neither resets pose, injects lift, replays motion, or relaxes a
physical gate. The authority change is fingerprinted under a new task ID, so
older PolicyPacks cannot be silently evaluated as v7.

## Qualified deployment

The final candidate was selected under
`birdflow_crow_journey_absolute_protected_contract` with no blocking milestone
regressions. Its deployed PolicyPack SHA-256 is
`650e6ca7b14cb4351474586eef45b4959d40c1808174f607589935b18aefd0e9`.

The held-out matrix ran all eleven bands at seeds `2660001001`, `2660001002`,
and `2660001003`, with 32 parallel environments and 1,600 control steps per
run. Across 33 runs and 1,689,600 environment control steps it recorded zero
failed environment steps and zero non-timeout terminations. The minimum
run-level mean tracking score was `0.6748238`, the largest run-level mean tilt
was `0.0973423 rad`, the global maximum tilt was `0.2829840 rad`, and the
largest root height was `1.7136120 m`.

This evidence qualifies the hierarchical v7 simulation contract. It does not
convert the policy into a pure neural controller, a measured-crow model, or a
hardware result.

The native Metal teacher is a bounded training and safety carrier, not a
deployable neural policy by itself. `--birdflow-journey-student-authority`
blends a student into that carrier for scheduled authority transfer; teacher
labels remain unblended. Deployment evidence discloses the v7 state-triggered
approach supervisor in `action_carrier`.

Use the stable entry point rather than invoking task binaries directly:

```sh
numi crow journey train --milestone takeoff-cruise [training arguments]
numi crow journey evaluate --milestone takeoff-cruise --policy-pack POLICY
numi crow journey train --milestone turn-left [training arguments]
numi crow journey train --milestone touchdown [training arguments]
numi crow journey train --milestone full-journey --policy-pack POLICY
numi crow journey evaluate --milestone full-journey --policy-pack POLICY
```

Journey milestones own their difficulty band; raw band overrides are rejected
by the crow command. High-quality BirdFlow plumage capture is downstream of
autonomous held-out qualification. Numi's capture command is only a native
physics-debug view and must not be presented as BirdFlow visual-quality proof.

All results concern a simulated, estimated hybrid. They are not measured
American-crow flight and do not constitute hardware-flight evidence.

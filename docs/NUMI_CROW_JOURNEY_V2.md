# Numi Crow Journey v2

`birdflow_american_crow_journey_v2` is the Apple-native training task for one
command-conditioned American-crow hybrid actor. It is separate from the older
standing-to-flight qualification task and has distinct task, observation, and
action fingerprints.

The task has 14 normalized actions, 83 actor/critic observations, and six
difficulty bands:

0. supported stand
1. walk or hop
2. takeoff
3. eight seconds of airborne straight-cruise stabilization
4. ground takeoff followed by eight seconds of straight cruise
5. full journey, including turns, approach, and supported landing

Band 4 is the first deployment gate. At 50 Hz it provides one second of
supported stand, up to four seconds for takeoff, and eight seconds of cruise in
a 650-step episode. A candidate must have no failed environment steps or
physical-boundary terminations, reach 0.55 m root height, achieve at least 0.65
mean tracking, remain below 0.35 mean tilt, and remain below 0.80 maximum tilt.
The held-out selector also evaluates band 3 to prevent flight-skill regression.
`--advance-candidate` cannot bypass this selection for the journey task.

The native Metal teacher is a bounded training carrier, not a deployable neural
policy. `--birdflow-journey-student-authority` blends a student into the carrier
for scheduled authority transfer; teacher labels remain unblended. At authority
one, the student owns the action and its rollout is eligible for PPO attribution.

Use the stable entry point rather than invoking task binaries directly:

```sh
numi crow journey train --milestone takeoff-cruise [training arguments]
numi crow journey evaluate --milestone takeoff-cruise --policy-pack POLICY
numi crow journey train --milestone full-journey --policy-pack POLICY
numi crow journey evaluate --milestone full-journey --policy-pack POLICY
```

Journey milestones own their difficulty band; raw band overrides are rejected
by the crow command. High-quality BirdFlow plumage capture is downstream of
autonomous held-out qualification. Numi's capture command is only a native
physics-debug view and must not be presented as BirdFlow visual-quality proof.

All results concern a simulated, estimated hybrid. They are not measured
American-crow flight and do not constitute hardware-flight evidence.

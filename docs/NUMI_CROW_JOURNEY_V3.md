# Numi Crow Journey v3

`birdflow_american_crow_journey_v3` is the Apple-native training task for one
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

Journey v3 adds a bounded pitch-moment action because the articulated tail and
pronation lanes could not arrest the measured landing attitude before wing
fold. The action is part of the fingerprinted neural contract; there is no
hidden touchdown assist and non-foot contact remains a terminating failure.
Because this changes both action and observation widths, a v2 PolicyPack is not
silently imported. V3 begins from teacher distillation, while every candidate
and rejected checkpoint remains retained evidence.

The native Metal teacher is a bounded training carrier, not a deployable neural
policy. `--birdflow-journey-student-authority` blends a student into the carrier
for scheduled authority transfer; teacher labels remain unblended. At authority
one, the student owns the action and its rollout is eligible for PPO attribution.

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

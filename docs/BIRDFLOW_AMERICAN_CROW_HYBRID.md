# BirdFlow American-crow standing-to-flight hybrid

`birdflow_american_crow_estimated_hybrid` imports the locked BirdFlowMetal
American-crow visual profile into Numi Lab as a trainable *estimated hybrid*.
It is deliberately distinct from the Deetjen-derived dove hybrid and from the
BirdFlow renderer's native crow showcase.

The active import lock is
[`assets/birdflow/american-crow-numi-hybrid-v1.json`](../assets/birdflow/american-crow-numi-hybrid-v1.json).
It binds this package to BirdFlow's selected 0.45 kg, 0.91 m wingspan, 0.174 m
tail chord, 57 mm tarsus, and 4.6 Hz presentation-wingbeat estimates. A
separate [`v2` timing candidate](../assets/birdflow/american-crow-numi-hybrid-v2.json)
used the 6.4 Hz maximal-takeoff frequency reported by
[Jackson and Dial (2011)](https://journals.biologists.com/jeb/article/214/3/452/33507/Scaling-of-mechanical-power-output-during-burst).
It exceeded the pre-registered height envelope in the M4 Pro guard, so it is
retained as negative evidence and not activated. Both locks remain estimated
hybrids, not specimen measurements.

## Native task

The robot begins with both sole collision proxies on the ground. Its ten actor
lanes control bilateral flapping amplitude, tail pitch, body yaw moment, and
bilateral hip/knee/ankle position. Metal evaluates the resolved wing, tail,
airframe, leg, contact, and action state in one control transaction:

1. supported standing and leg push-off;
2. lift-off from physical sole contact;
3. forward-flight speed tracking; and
4. a bounded figure-eight flight reference.

The two wings use a device-resident eight-station blade-element closure and
write per-body wrenches before articulated ABA. This means wing loads depend
on the current accepted pose and velocity; the task never injects a replayed
takeoff force. The closure is an authored simulator parameter chosen to make
the imported 4.6 Hz estimate trainable. It is not a crow aerodynamic,
energetic, or CFD result.

Run it through the workspace capability:

```sh
numi crow train --envs 256 --steps 128 --updates 8 --chunk 8
numi crow evaluate --policy-pack .numi/runs/crow-standing-flight/deployment.policypack \
  --envs 64 --steps 512 --repeats 2 --chunk 8
```

Each training run retains its candidate, immutable incumbent, matched held-out
selection, policy fingerprints, rollout artifact, and physical outcome vector.
Promotion is conditional on that matched comparison. For BirdFlow flight tasks
the selector disables the generic 257-step reset stressor, runs the full
5,000-step task horizon on a held-out seed, and keeps a deterministic state
trace. To preserve a practical GPU submission cadence, that trace records
every eighth 20 ms control step (160 ms); its header and JSON evidence record
the stride. This downsampled trace is a replay/audit record, not a visual
flight movie or a substitute for an all-step diagnostic trace.

At the terminal standing-to-flight band, a candidate must have zero failed
environment steps, no increase in physical terminations, positive staged
progress, and at least the authored 0.70 forward-tracking score before the
protected deployment artifact can advance. A launch, a local lift-off metric,
a positive reward, a PNG, or a GIF does not establish controlled flight;
none establishes measured American-crow flight.

## Universal journey demonstrator

`birdflow_american_crow_journey_showcase_v1` is a separate visual-demonstrator
task. Its single 14-lane actor observes a native normalized journey phase and
must execute stand, walk/hop, launch, figure-eight cruise/turn, approach, and a
supported landing hold over one deterministic 32-second band-4 episode. It
does not use the standing-to-flight task's carrier or residual policy, and it
cannot replace that task's protected deployment artifact.

The authored Numi scene binds the BirdFlow surface to the airframe, both wings,
and tail. Simple thigh, shank, and foot meshes are explicitly estimated
presentation geometry matching the hybrid collision proxies; they are not
measured anatomy. The native renderer can drive both a live inspector and
deterministic 50 fps MP4/GIF export:

```sh
numi crow journey train --envs 256 --steps 128 --updates 8 --chunk 8 \
  --minimum-difficulty-band 0 --maximum-difficulty-band 4
numi crow journey evaluate --policy-pack PATH --envs 8 --steps 1600 \
  --minimum-difficulty-band 4 --maximum-difficulty-band 4 --no-scheduled-resets
numi crow journey window --policy-pack PATH
numi crow journey capture --policy-pack PATH
```

Window and capture deliberately require an explicit PolicyPack. A movie is
showcase evidence only after the matched held-out selector accepts the same
artifact; simulation remains an estimated-hybrid result rather than measured
American-crow locomotion or flight.

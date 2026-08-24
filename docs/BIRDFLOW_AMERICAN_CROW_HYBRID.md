# BirdFlow American-crow standing-to-flight hybrid

`birdflow_american_crow_estimated_hybrid` imports the locked BirdFlowMetal
American-crow visual profile into Numi Lab as a trainable *estimated hybrid*.
It is deliberately distinct from the Deetjen-derived dove hybrid and from the
BirdFlow renderer's native crow showcase.

The active import lock is
[`assets/birdflow/american-crow-numi-hybrid-v2.json`](../assets/birdflow/american-crow-numi-hybrid-v2.json).
It preserves the BirdFlow visual lock and its selected 0.45 kg, 0.91 m
wingspan, 0.174 m tail chord, and 57 mm tarsus estimates. It replaces only the
4.6 Hz presentation clock with a 6.4 Hz American-crow maximal-takeoff timing
anchor from [Jackson and Dial (2011)](https://journals.biologists.com/jeb/article/214/3/452/33507/Scaling-of-mechanical-power-output-during-burst).
The source records vertical escape takeoff, not a forward-flight trajectory,
inertia, or aerodynamic calibration. The original
[`v1` lock](../assets/birdflow/american-crow-numi-hybrid-v1.json) remains
unchanged, and the active package remains an estimated hybrid rather than a
specimen measurement.

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

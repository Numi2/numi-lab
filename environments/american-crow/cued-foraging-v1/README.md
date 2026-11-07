# Portable engineering cued foraging environment

The standard native rollout/trainer accepts prepared/retinal-observation.json.
Pack paths are relative to that file. The additive cued_foraging configuration
binds shared native world, sensor, finite food and lifetime ownership. Prepare a
fresh output with tools/numi crow-env prepare-cued-foraging --output NEW_DIRECTORY.
The native compiler authors assets without advancing simulation.

calibrated-ports.bin contains two exact float4 positions/radii (32 bytes),
recovered and SHA-verified from the native calibration Git archive identified in
calibration-provenance.json. One of two15mm ports contains45mg food per lifetime;
two20mg bites are possible. This is a conserved nutritional tracer at supported
engineering intake volumes, not solid food manipulation or calibrated biology.
The rendered indicator disappears at lifetime tick32; intake starts at192.
Physics/control is50Hz; estimated parallel pinhole retinal sampling is15Hz with
two-control-tick40ms delay. Allocation seed is hidden from actor observations.
Ordinary resets neither refill nor reroll food nor restart cue time.

The initial eight-lane H64 sequence PPO integration and fresh-process native/
optimizer continuations pass; initial food reward is zero. Learned memory,
confound-controlled choice, social learning, manipulation, ecology and production
performance remain unqualified. See docs/CROW_RL_ENVIRONMENT.md and the profile.

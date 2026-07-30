# MetalRobo developer contract

This repository has one current architecture. Treat old numbered design notes
as historical context, not as implementation choices.

## Start here

Read only the subsystem document relevant to the change:

- world authoring and packs: `docs/WORLD_ENGINE.md`
- Metal execution: `docs/METAL_WORLD.md`
- visual presentation: `docs/VISUAL_PLATFORM.md`
- tactile sensing: `docs/TACTILE_GEOMETRY_BRIDGE.md`
- numerical rules: `docs/NUMERICS.md`

Do not begin by reading the detailed capability ledger or by running every
executable in the repository.

## Architecture rules

- Visual Presentation V3 and its sectioned V2 asset/environment pack wire
  formats are the only authored presentation path. Never derive a visual
  scene from collision geometry, add a legacy presentation adapter, or
  synthesize a fallback pack hash.
- Existing perception-contract V1 labels are current wire-format names. They
  do not authorize renderer fallback behavior.
- Version numbers belong only on persisted wire formats or real ABI
  boundaries. Do not add version suffixes to ordinary types, examples,
  algorithms, or filenames.
- Use one current world-pack format. If an unreleased feature changes the
  current format, extend it in place instead of creating an intermediate
  compatibility generation.
- Fingerprints belong at artifact, cache, replay, or policy-observation
  boundaries. Do not add per-frame hashes or duplicate the same fingerprint
  across adjacent metadata files.
- Hot loops should consume stable counts, capacities, and existing state.
  Add per-sample or per-contact branches only when they are necessary and
  measured.
- A subsystem gets one focused correctness executable. Examples demonstrate
  product flow; benchmarks measure performance. Do not turn either into
  another regression suite.

## Verification

Build and run the smallest check that owns the changed behavior. Run a full
build only at an integration or release boundary. For tactile work:

```sh
cmake --build build --target metalrobo_tactile_check
./build/bin/metalrobo_tactile_check
```

The Franka flow is an example, not a second test suite:

```sh
cmake --build build --target metalrobo_franka_tactile_example
./build/bin/metalrobo_franka_tactile_example
```

Keep benchmark commands out of correctness workflows unless the change can
plausibly affect throughput or retained memory.

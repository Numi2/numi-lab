# Source189 production-helper replay package

This package is prepared for root-owned native replay 511. It does not launch a GPU job.

The input directory is /Users/n/cardboard-implementation-20261014-host-macmini/analysis/contact-armijo-energy-replay-499-001/case-496-inputs-v4. Its v4 manifest SHA-256 is b6c2e1872afe5f6e6685a57525fe91e604b20bf6ab5aa0fcb1acd546e80dea0f. The custom replay host and metallib are contact_barrier_trial_replay_source189_host and contact_barrier_trial_replay_source189.metallib; their hashes are 23f392404f9a58a221c3a505a944c21c72ec93cb960155ddfe90c885d932031c and 136bd6c922b42670852e3a6e9fbc07b2758feb702767c2704dd6bd00839ad209. The kernel invokes source189 production femTrialNodeAtAlpha, deformableBarrierEnergyAt, and certifyDeformableNewtonSweepToEndpoint. Output is 8,064 bytes (24 records x 21 float4), with no Matter transactions.

Important provenance limit: source-shared-contact-trial-189-buildfix002/frozen-source.json is stale for all three changed files. Its 814-entry map still binds the source188 versions. The actual three files and the compiled library's .d dependency closure are hashed in the manifest. Root produced source-shared-contact-trial-189-buildfix003, whose refreshed 814-entry source map verified 814/814 against disk. The runner requires a refreshed source root and rebinding of every source-tree dependency before dispatch; it does not accept the stale 002 map.

Run only after root verifies the source closure and confirms the native slot is free:

    python3 /Users/n/cardboard-implementation-20261014-host-macmini/analysis/source189-shared-contact-trial-002/run_source189_helper_replay.py \
      --source-root /Users/n/cardboard-implementation-20261014-host-macmini/snapshots/source-shared-contact-trial-189-buildfix003 \
      --output-dir /Users/n/cardboard-implementation-20261014-host-macmini/native/source189-source500-helper-replay-511

The output directory must be a new direct child of native/. The runner validates source-map hashes, the compiled metallib/host/dependency file, and all v4 inputs; it writes fresh logs, a preflight record and a receipt. The custom Metal library was compiled against source189-002 actual source bytes, not a regenerated source map; the three changed file hashes match source189-003. No full-trajectory, Armijo acceptance, or physical qualification is claimed here.

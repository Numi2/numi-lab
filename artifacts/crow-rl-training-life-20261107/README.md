# Coupled life and optimizer training checkpoint evidence

V13 training now resumes the same native life with its exact next policy and MLX
model/Adam state. An exclusive directory bundles policy, optimizer, native life,
generating rollout, unselected deterministic candidate and a manifest. Payloads,
manifest and directory entries are fsynced before atomic SHA-bound COMMITTED
publication. Partial directories are retained and inadmissible. This exercises an
interrupted-directory model, not forced process-crash or power-loss qualification.

Trial05:16 lanes,37 ticks/update, two epochs, uneven minibatches97. Four reference
updates equal one two-update split plus two independent two-update continuations
at chunks1/8. At every shared optimizer boundary all rollout/policy/candidate
bytes, model/Adam tensors+metadata and three native components match bit for bit.
Accepted clocks37,74,111,148 and revisions1→3→5 have no restore warmup. Lane12
consumed20,000 micrograms before interruption and retains it with conserved food.
Nine declared-cause controls reject uncommitted/bad-marker/missing-or-corrupt
optimizer/changed-configuration/mixed-policy/wrong-worker/wrong-scheduler and
input-output-overwrite attempts. No qualification seeds or promotion are used.

Attempts01/02 retain planning and boolean-parser failures before context creation.
Attempt03 retains premature legacy telemetry rejection before resident execution.
Telemetry now publishes after accepted updates rather than adding a fake step.
04 passes original comparisons/seven negatives.05 adds exact clock/food checks,
two negatives and an imported learner-code closure identity. Both successful
binary/Python-source versions independently resume their saved original bundles
and reproduce both future optimizer boundaries using retained runtime copies.
The exact installed Python/MLX/NumPy distributions are still required and checked.
Earlier failed runtimes have source deltas/hashes, not every historical binary.

Checkpoint publication averages16.1–16.5 ms per stage in05 (small warm-cache
instrument, not sustained scaling). Twelve native and26 Python checks pass.
Ordinary learner-only training reproduces its prior optimizer checkpoint hashes;
frozen feeding ordinary/residual fingerprints remain2255877855914022940 and
3082487550463979285. No learning algorithm or native transition changed here.

All plans, failed stages, update-boundary bundles, generating rollouts, candidate
policies, source patches, raw tests, receipts and matching-runtime replays are
preserved. payload-inventory.json records original and archived sizes/SHA-256.
Decompress *.gz to the name without .gz and verify original_sha256 before use.
Native life and safetensors/rollout binary payloads are losslessly compressed.
Invalid negative-input directories deliberately remain invalid and separate.
Runtime-receipts.json locates native binaries/shaders and complete Python-source
copies on macmini. Use the matching saved lib in DYLD_LIBRARY_PATH and saved
python directory via --python-root. Do not mix binaries, worker code or installed
MLX/NumPy versions with old checkpoints. See retained replay plans for exact CLI.

This admits coupled persistence for the present supported development fixture.
The controller remains an MLP without trainable persistent memory. Biological
fidelity, solid-food manipulation, physiology, social learning, robust locomotion,
production performance and held-out ecological qualification remain open.

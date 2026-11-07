# Retentive GRU initialization and native temporal-history race

Fresh optional trainable log-spaced2..256tick GRU update-gate biases improve the
early-cue replay gradient243197x while all other seeded parameters remain exact.
38 tests pass, including analytic unforced decay, legacy parameter exactness and
selected-factor FP64 PPO checks. Eight bad Swift/worker options reject. A fixed
64-update64-lane native campaign accepts1310720 transitions with zero reported
physics rejections at3864 loopsteps/s. Intake stays flat (52/512 first,47/512 last).
There is no learning-improvement or learned-memory qualification.

Fresh1-update continuations at chunks8/1 match policy/candidate/rollout bytes and
all53 model/Adam tensors/metadata, plus retinal and food state. World checkpoint
comparison FAILS:12 data bytes and32 checksum bytes differ. The12 data bytes are
three float32 history words at the SIMD boundary. Native observe/complete use
`environment*actions` as the base of a buffer allocated for TWO action arrays per
lane. A lane position history overlaps the next lane velocity history. Other
world buffers and GRU state are exact. This faulty runtime does not qualify exact
future execution. Version/fix the shared native layout and add adversarial
cross-lane tests before further learning. Preserve the failing runs and library.

Git preserves all raw evidence, failures, exact runtime, source blobs and final
checkpoint. restore_runtime.py recovers them by Git hashes. verify_recovery.py
loads all53 actual model/Adam tensors and admits the native life checkpoint on
the adjacent restored library without advancing any physical step. This proves
recovery of the faulty runtime and data, NOT repeatable future simulation.
All raw files are additionally verified against Git before cleanup; bulk checkout
payloads can be omitted after pushing recovery receipts. Matching OS/device and
MLX/NumPy required. No new simulation retry is used to hide the stride failure.

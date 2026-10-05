# Integrated accepted-cursor rollback probe

This probe exercises Matter's accepted control cursor together with the coupled Human transaction. After an accepted predecessor it forces the respiration dispatch gate to reject a candidate and checks that body `q/v/root/Myo`, Matter circulation and clocks, and respiration/Brain history remain at the last accepted state. It retries the same candidate and compares the resulting state with uninterrupted replay. A second case accepts two steps, rejects step 2, and verifies the remaining same-command-buffer suffix is inert.

The retained native evidence is on the Mac mini at `/Users/n/numi-human-resting-evidence-20261005/transaction-probe-005/` (`invocation.json`, `run-metadata.json`, `native.log`, and `resting-coupled.csv`). The probe result in `native.log` is:

```text
resting_integrated_rejection=pass rejected_after_accepted_predecessor=true body_q_v_root_myo_unchanged=true circulation_and_clock_unchanged=true respiration_brain_history_unchanged=true retry_matches_uninterrupted_replay=true same_command_buffer_reject=pass accepted_prefix=2 rejected_step=2 inert_suffix=1
```

The run used the uncommitted source snapshot based on repository revision `59d2aa5f534175080dd28582a58b45255ad869fc`. The cursor-source hashes used by the run are:

| File | SHA-256 |
| --- | --- |
| `matter/include/numi/matter/matter.hpp` | `65f9a768ef2b3f272a25e7d52d22e642944261701edf93a5f4497da552d95d99` |
| `matter/src/runtime.mm` | `b8558155c8dcc251273e3b30a1788a1e644db96de97f326a1fb0ca60e2b5411a` |
| `matter/src/metal/identification_scheduler.metalinc` | `de1fecafc3f2debe662990828e1379c181a595b3858e3b3c4948b5743882abe7` |

The invocation and metadata retain the complete launch arguments, environment, model/asset hashes, binary and metallib hashes, and device. It ran on `ns-Mac-mini.local` (arm64, macOS 26.6) for 64 × 1 ms simulated seconds; the integrated body horizon took 37.79 s wall time (0.00169× real time). Root assistance was disabled. The CSV and terminal-state output are retained for diagnosis. This is transaction/replay evidence only; it does not establish a stable resting pose, physiological validity, five-minute endurance, or native-viewer presentation acceptance.

Evidence file hashes:

- `native.log`: `9e0cdb683aeb1e90169fe22f649180361a71aaee65039a82b17c089b0cca747f`
- `invocation.json`: `b9d32c93604576667472c3b3381d0b18b0f8b5984a5dcfba401a6f55f05cda7e`
- `run-metadata.json`: `cbfe7be389098efc81fdafcd48bb43252f5d8760be12e24975bf18772a00636b`
- `resting-coupled.csv`: `87d0594cbe0bdd57bb9525f89f48e0156906a240cf98267a7e7e060168f2cc7e`

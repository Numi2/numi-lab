# Native candidate and transaction-state parity: runs 291 and 295

This package compares the raw selected Newton candidate and transaction snapshots from two retained failed native runs. It does not compare whole runs or certify convergence.

Inputs:

- Run 291: `native/strict-json-full-contact-arena-newton12-291`, source snapshot `source-strict-coupled-capture-json-088`, receipt exit 1, timed out false, inputs unchanged.
- Run 295: `native/trace-slot-production-contact-parity-295`, source snapshot `source-direct-projection-diagnostic-089`, receipt exit 1, timed out false, inputs unchanged.
- Both candidate captures are root 1201, Newton iteration 12, environment 0, and use the coupled raw-candidate capture schema. Both are the same physical failure, not a successful transaction.

Findings:

- Candidate capture payloads are byte-identical: 10,787,124 bytes, SHA-256 `ef7e5ff74370cf84d7c36fb647b3e2735add4695308b312807ccbbbc0c65c1aa`.
- Candidate manifests identify the same source-physics fingerprint `915f50b10edc2e46`; the device-program fingerprint changed from `d61f84737f5c7664` to `2fecbe5cf49ff5ed`. The only manifest differences are that fingerprint and the run-local payload path.
- All six paired serialized transaction snapshots are 904,359 bytes. Each differs at exactly byte offsets 12–19, the serialized device-program fingerprint. The source-physics fingerprint bytes at 4–11 match. All three transaction JSON files are identical.
- The snapshot header contract is confirmed from `matter/tools/cardboard_shipping_snapshot.hpp`: `writeSnapshotPayload` writes the message string, then `writeScalarFields`; with the empty message, the first 4 bytes are its length, bytes 4–11 are the `uint64_t sourcePhysicsFingerprint`, and bytes 12–19 are the `uint64_t deviceProgramFingerprint`. The raw serialized source fingerprint is `0xb8682516183db1dc` in both runs. The program words decode to `0xd61f84737f5c7664` and `0x2fecbe5cf49ff5ed`.
- The root-1200 diagnostic log is byte-identical. The root-1201 diagnostic log becomes byte-identical after replacing its expected run-local candidate-manifest path with a placeholder; no physics fields differ in that log.
- The initial inline comparison attempt treated binary snapshots as UTF-8 and aborted. That was a comparator/instrument error, not a simulation result. This report and script compare raw bytes.

Interpretation: the candidate and persistent physical-state payloads match across these runs, except that every snapshot embeds the expected device-program fingerprint change. This is not a diagnostics-on/off pair under one sealed executable: the source/build snapshots differ. It is not whole-container identity, and both runs retain failed physical verdicts.

Reproduce:

```sh
python3 compare_native_capture_parity.py \
  /Users/home/cardboard-implementation-20261014/native/strict-json-full-contact-arena-newton12-291 \
  /Users/home/cardboard-implementation-20261014/native/trace-slot-production-contact-parity-295 \
  --output report.json
```

The output hashes and per-file checks are recorded in `report.json`.

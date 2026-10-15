# Native 419 production-history parity audit

This read-only audit compares the rich operator-FD run 419 with production-history controls 409 and 415. All three runs exited 2 after one accepted control step, reported `status=failed`, and set `physical_validation=false`; no complete coupon cycle is qualified.

The same-source 415/419 output trees are byte-identical across all 35 files. They also have identical stdout/result JSON, all 64 residual-Armijo rows, all 4096 true-residual rows, and 4164 stderr lines after filtering the declared diagnostic and timing records. The source-123 control 409 and 419 have the same 33 of 35 output files byte-for-byte; the only differences are the three material source-path fields in the output manifest and the device-program fingerprint in the rejected-conditioning manifest. Both comparisons preserve full file hashes in `audit.json`.

The status-5 payload is byte-identical between 415 and 419 (`0f3ae7cd…`); only the manifest-relative payload path differs. It retains the source-125 observer’s known bad alpha value of zero, while the Newton-26 Armijo record says the rejected alpha-1 trial is followed by accepted alpha `0.0078125`. The alpha-lane correction package is separate and was not used in 419.

Run 419 adds the selected root-1/Newton-48 operator-FD capture. Its JSONL validates as 5606 records: one header, one integrity record, 1764 node records, and 3840 tetrahedron records (two environments, 882 nodes and 1920 tets each). The header binds the capture to Newton 48, the source/device fingerprints, timestep, finite-difference epsilons, and the cached algorithmic tangent; the final integrity row reports `production_inputs_bitwise_unchanged=true`. This audit verifies capture integrity and observational parity, but it does not adjudicate the captured derivative or identify the terminal Newton-48 cause.

Reproduce with `python3 audit_419.py`. The audit is CPU-only and does not change any run or frozen source.

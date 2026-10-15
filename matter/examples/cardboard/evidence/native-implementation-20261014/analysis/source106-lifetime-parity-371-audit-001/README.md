# Source106 callback-lifetime audit: runs 362, 369, and 371

## Verdict

`FAIL_UNEXPECTED_STATUS`. The callback metadata fix itself is verified for this capture: run369 reports `rhs_preserved_before_fgmres=false` even though its RHS buffer is byte-identical to runs362 and371; run371 reports `true`, matching the checked/captured RHS bytes. However, source106 run371 is not fully parity-clean: its rejected private trial at Newton iteration7 reports a different contact failure status than runs362/369. The status is retained without normalization.

All three attempts exited1. This audit does not establish a converged physical simulation or successful contact solve.

## Capture and state comparison

All three selected center captures are root1201/Newton7, contain145 arrays and11,670,948 payload bytes, and have identical raw payload SHA-256 `965d38af46af2bbc311cb1534193b8892581cb4d4548936dfc8d7ba9db0cc2e4`. In particular, `full_rhs_before_fgmres` is identical. RHS metadata values are362=true, 369=false, 371=true; no RHS flag was masked.

Three transaction JSON documents match exactly in every pair. Six before/after transaction binaries match outside only the serialized device-program fingerprint at byte offsets12–19; the fingerprints were checked against each run's center manifest. Source105 and source106 have the same Metal program fingerprint because the source106 delta is in runtime callback ownership and its new context header.

The 32 Armijo records match between362 and369. Run371 differs in only one tested-record field set, at Newton7 `trial_status`: runs362/369 report code6/failing_index283/diagnostics `[92,127,0,0]`; run371 reports code6/failing_index124/diagnostics `[24,127,1,0]`. The base health, selected alpha, rejected-trial outcome, other Armijo fields, Newton0–6, and Newton8–31 match. The same difference is mirrored in `fixture-free-body-control-1201.log`. Since this is a rejected private trial status, it still blocks the requested exact status/Armijo parity and remains unexplained.

## Source lineage and protocol

The audit validates the sealed source104→105 comparison lineage and the source105→106 callback overlay's exact two-file delta (`runtime.mm` and `coupled_capture_manifest_context.hpp`), including all frozen file hashes, build receipts, runtime artifact hashes, prediction inputs, and the normalized physical invocation/environment. The 371 attempt did not time out and its source inputs remained unchanged.

The source106 callback overlay copies the RHS-ready and related manifest values into an owned context before installing the asynchronous completion handler; it does not hard-code the RHS bit. The observed false→true correction with byte-identical RHS supports that narrow lifetime repair. The separate Newton7 status change means overall default-path parity remains failed and should not be described as a physics pass.

## Reproduction

Run the strict auditor with the configured Python3.10 interpreter:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/source106-lifetime-parity-371-audit-001/audit.py
```

Exit status1 is expected for this preserved `FAIL_UNEXPECTED_STATUS` result. The audit writes `report.json` before returning that status. The report contract tests pass three checks and assert that the RHS flag and the single status mismatch remain visible.

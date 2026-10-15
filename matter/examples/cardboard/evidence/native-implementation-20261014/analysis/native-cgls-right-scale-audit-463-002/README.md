# Native block3 CGLS audit: run 463

This package audits the source149 opt-in right-scaled CGLS direction captured at root 1, Newton iteration 48. The same-run operator capture did not run: source runtime explicitly skipped it because the compiled run has two environments. The audit therefore reuses operator capture 419 only after checking the strict physical-state chain 463 → 455 → 419.

The state join passes for all 17 required arrays and every other shared sidecar array. Execution fingerprints differ and are kept distinct. The 463→455 physical state/world join is exact; 455→419 is the retained exact source-input join, including its explicitly counted signed-zero-only tetrahedron serialization differences and zero other word mismatches. The resulting action comparison is descriptive across captures/builds and is not a numerical tolerance or noise-floor certification.

At Newton 48, environment 0 is selected and environment 1 is quiet. The CGLS state is `(status=2, completed=64, cap=64, selected=1)`: the source status is iteration-cap exhaustion, not convergence. Environment 1 is inactive and its vectors, field rows, mask, and right scale are zero. The native `q`, `r`, and `s` actions agree closely in absolute terms with rebuilt actions from the joined 419 operator; relative L2 differences are recorded in `report.json` but are descriptive only.

Comparison against completed runs 459, 453, and 455 shows the same four attempted rows. The accepted bent step 0, held reference rows, and held-reference step 1 row are exact. Stable physical/result artifacts are byte-identical. At bent step 1, 463 changes the attempted solver history (48 versus 44 FGMRES iterations) and certificate values, but status remains 10 and `step_accepted=0`. This is a failed prefix with one accepted control step, not a completed coupon, convergence, or physical qualification. Rejected-conditioning snapshots also differ and are listed in the structured report.

## Reproduce

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 -m py_compile \
  audit_chain.py audit_v1_parser.py test_audit_capture.py test_lineage.py
/Users/home/.pyenv/versions/3.10.12/bin/python3 test_audit_capture.py
/Users/home/.pyenv/versions/3.10.12/bin/python3 test_lineage.py
/Users/home/.pyenv/versions/3.10.12/bin/python3 audit_chain.py \
  --case /Users/home/cardboard-implementation-20261014/solver/cgls-block-right-primary48-463 \
  --receipt /Users/home/cardboard-implementation-20261014/solver/cgls-block-right-primary48-463/receipt.json \
  --command /Users/home/cardboard-implementation-20261014/analysis/native462-463-right-scale-command-package-001/native-463-command.json \
  --source-snapshot /Users/home/cardboard-implementation-20261014/snapshots/source-cgls-block-right-preconditioned-149 \
  --expected-environment 0 \
  --output report.json
```

The exact input hashes, selected-state lineage, attempted-history comparison, vector-action statistics, and limitations are in `report.json`. The audit consumes retained native data only; it performs no GPU work and changes no source or native run.

# Source-bound adjudication: two-stage coupon run 409

This package audits the already-finished native attempt in `solver/two-stage-certified-coupon-409`. It is read-only: it hashes the run inputs and outputs, parses the sealed prediction/receipt, checks the source/build closure, and adjudicates the Armijo and true-residual traces. It does not rerun the solver or alter the run.

Run it with the captured Python 3 interpreter:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/two-stage-coupon409-adjudication-001/audit_coupon409.py \
  --output /tmp/coupon409-audit.json
```

The output destination must be new. For a read-only check without writing a file, omit `--output`. The script fails closed on source, artifact, CSV, trace-count, status, or rollback mismatch.

The frozen source is `snapshots/source-two-stage-material-merit-123`; its `frozen-source.json` SHA-256 is `9a7f6bcbb6110dbd2a82f898b110ae0dd9c7290ab838155808c999a69679597e`. The source-build manifest verifies all 783 snapshot files, 11 build artifacts and two build-metadata files. The probe executable SHA-256 is `5cdf14b437eded9f6cc6c1aea402943f33a4bfa7069c207ce7d70788eded4f0c`; the `NumiMatter.metallib` SHA-256 is `14e64d56961421ce2007512c72ce2e8c48ba6c4c2ef85302a42af7538da33db3`. Compiler system headers and the OS/driver are not archived by the source-build record.

## Finding

The command planned 27 controls across loading, hold, unload, relax and release. It recorded only loading rows for macro steps 0 and 1. Bent control 0 was accepted; bent control 1 failed with status 10 after 64 Newton records. The two held-reference rows were certified no-ops. No hold, unload, relaxation or release control ran. The receipt completed normally with exit code 2 (not timeout), and the result says `status=failed`, `accepted_steps=1`, and `physical_validation=false`.

For the failed bent step, Newton 48 (zero-based) is the first line-search failure. Newtons 0–47 each selected an Armijo trial inside the uncommitted step transaction. At Newton 48, the pretrial status is clear (code 0), no trial was selected after 24 line-search candidates, and the final recorded candidate status is also clear (code 0). The status written by the solver is code 10 at failing index 48; its diagnostic records base raw residual `1.84443750e-4`, last tested residual `1.84443837e-4`, last tested alpha `1.1920929e-7`, and candidate count 24. The last raw residual is `8.7e-11` above the base (`4.72e-7` relative), but the log does not preserve the individual merit/status of all 24 candidates, so this small difference alone does not explain every rejected trial.

The inner FGMRES at Newton 48 is not the terminal failure: it returned solver/operator status 0 in three active restart cycles with 44 iterations (16+16+12). Its final fresh true residual was `2.50240851e-6` against right-hand-side norm `1.84443808e-4` (ratio `0.0135673`), and its convergence marker was set. This is an inner forcing result, not an outer nonlinear equilibrium certificate. Newtons 49–63 are gated by the already-latched Newton-48 code 10; the environment-0 slots in 960 residual-cycle event records carry solver status 10/operator status 4 and `true_l2=-1`; the same event records still carry the healthy held-reference environment. They are not additional failed nonlinear solves.

No recorded status at the failure identifies an invalid local material projection: the failed Newton's pretrial and last recorded trial statuses are both clear, and its three active FGMRES cycles report operator status 0. However, the Armijo trace stores only the last candidate status, not per-tetrahedron projection results for every one of the 24 candidates. This evidence therefore cannot certify that all candidate projections were valid or identify which merit/feasibility gate rejected each trial.

The outer step rollback is exact in the exported geometry: `accepted_step_000001_bent.obj` and `rejected_step_diagnostic_000002_bent.obj` have identical SHA-256 `42fbb543657063c3c3df7eed3ec4d07dc820f5f2ab887e2d3ceb6c3bfd68a522`. The held-reference meshes at both steps are also byte-identical. The failed-step material-state export reports status 10 and a cleared raw-accepted certificate flag. The provisional Newton updates in the rejected step were not committed as a physical control.

## Suggested bounded capture for a diagnostic rerun

Keep the same source-123 executable, metallib, physical options and full command from `prediction.json`; add a fresh operator capture path and select control step 1, Newton 48:

```text
NM_FEM_OPERATOR_FD_SELECTOR=1:48
NM_FEM_OPERATOR_FD_CAPTURE_PATH=<fresh unique JSONL path>
NM_FEM_RESIDUAL_ARMIJO_ROOT=all
NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT=1
NM_FEM_TRUE_RESIDUAL_ROOT=1
NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR=48
NM_FEM_TRUE_RESIDUAL_CERTIFY=1
NM_FEM_TRUE_RESIDUAL_RESTART=0
```

The selected-Newton operator capture can expose the base center and per-tetrahedron material projection at the first failing iteration; the true-residual selector narrows its cycle record to that Newton. Diagnostic-on/off parity remains necessary. These selectors do not convert the recorded failure into a pass, and they still do not enumerate every line-search candidate's per-tet projection. This run disables self-contact and tooling, so contact CCD tracing is not the relevant observer.

## Bound files

`audit.json` contains the machine-readable adjudication and exact command argv/environment. `manifest.json` binds this package and the source-409 evidence hashes. `audit_coupon409.py` is the verifier. The raw run remains in its original directory; this package does not duplicate the 4.4 MB stderr trace or the rejected-conditioning buffers.

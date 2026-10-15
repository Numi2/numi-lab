# CGLS active-mask predicate repair (source 142 candidate)

This candidate changes one comparison in `nm_fgmres_base_cgls_measure_initial`: the constrained-row validation now checks `mask.xyz == 0` before requiring `r` and `s` to be zero. Active rows (`mask.xyz == 1`) may carry nonzero CGLS residual and transpose residual. This matches the existing `form_alpha` validation and leaves the error bit, thresholds, recurrence, and dispatch unchanged. The repair is duplicated in the package's mirrored helper source and is otherwise byte-for-byte limited to this predicate.

Native446 established the pre-fix transition on the synthetic fixture: after `initialize_vectors`, env0 is `[status=1, iterations=0, cap=32, selected=1]`, errors 0, with nonzero r/s norms; after `measure_initial`, it is `[5,0,32,1]`, errors 16, before any iteration. The stage-diagnostic checker is separately sealed at `/Users/home/cardboard-implementation-20261014/analysis/native-base-cgls-diagnostic-443-001`; this repair package does not alter that checker. Native447 remains parent-owned and has not been run here.

## Validation

- Six existing CPU recurrence/admission tests pass under `/Users/home/.pyenv/versions/3.10.12/bin/python3`.
- `native/compile_only.sh` passes: Metal compile/link plus strict Objective-C++ syntax check of the original synthetic checker.
- No GPU/native execution or production solver qualification is claimed.

The exact one-line diff from source141 is in `core-v1/helper-mask-repair.patch`; `source-overlay.json` binds both mirrored helper copies and the compiled metallib. Preserve source141 and the native443/446 failures as historical evidence.

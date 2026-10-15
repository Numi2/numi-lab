# Same-Newton conditioning candidates for native capture 304

This CPU-only replay uses the exact rich native operator and right-hand side from source091 capture 304, root 2 / Newton iteration 63 / environment 0. It binds the operator capture SHA-256 `609ebcc9751f7ba5a5d42154f1969d065bd779ea44e901c80bb3c925152089ab`, operator report SHA-256 `9d55e082c17fa246a23a98e96ca17832fff8814d017f0bf69eaf1d2196408f6d`, and same-step regional manifest SHA-256 `75fd9d878fdd22c01929f7bb231046480d0daec9cefd77318d4703aff09ce8b00` (manifest hash checked against report.json; use the exact value embedded in report.json for canonical binding).

`run.py` compares bounded restart-16 continuations and a same-Newton coarse correction of the captured native correction. The coarse correction is `x_aug = x_native + P (P^T A P)^-1 P^T (b - A x_native)`, using the exact captured effective P and A, then checks `P^T(b-A x_aug)`. A second exploratory candidate projects the reconstructed graph-patch correction into the left A-orthogonal complement of P before adding it. No A, RHS, physical model, tolerance, or acceptance criterion changes.

The offline graph replay reconstructs patch matrices from the captured A and factors them in FP32. Krylov/MGS is a CPU reimplementation, not a native Metal preconditioner replay. Results diagnose candidate utility only; they do not qualify a solver change. The captured state is a rejected Newton transaction, not an accepted physical checkpoint.

The exact A/RHS offline restart-16 residuals from zero are approximately 0.952249 at 32 steps, 0.952224 at 64, 0.952224 at 128, and 0.952224 at 256. Continuing from the native 32-step correction gives approximately 0.952265 at cumulative 64, 128, and 256 steps. Thus more restart cycles do not address the failure in this approximate replay. A same-Newton exact RAP residual correction moves the captured native iterate from 0.952291 to 0.908699 relative residual, with `||P^T r||` reduced from `2.10e-5` to `5.94e-18`; subsequent approximate graph iterations plateau near 0.908356. This is useful but far from the required solve tolerance.

RAP is full rank 26 with reported 2-norm condition about 2047.8 and native LU validity in both environments. Thirty-three deterministic Rayleigh probes of the symmetric part were positive (minimum 0.0877), but this does not prove positive definiteness. The offline residual plateau is vastly above a solved residual and no measured numerical noise floor was established. Native tangent-action mismatch and CPU/native arithmetic differences remain limitations.

`report.json` SHA-256: `87de72e03802a2f3ac97d48152a1c5afb6148bca2d521cf9f9f484b32c7a274b`.
`run.py` SHA-256: `51fd3a13bbe0cadd78ef5d65d2011a86e83b981aba7581d3f188bda5e2905d89`.

Earlier immutable report `../regional-preconditioner-binding-304-002/report.json` remains preserved. No source, runtime, CMake, or native GPU output was modified.

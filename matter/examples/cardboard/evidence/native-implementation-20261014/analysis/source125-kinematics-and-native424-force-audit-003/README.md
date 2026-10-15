# Source 125 kinematics and native 424 force audit

This is a bounded CPU-only adjudication of source 125 control 1, Newton 48, environment 0, using the exact accepted center from rich capture 419 and native endpoint node/residual/force words from sweep 424 at epsilon `2^-14` (`6.103515625e-5`). It does not make a production, calibration, or physical-qualification claim.

The source-transcribed frame, compensated deformation high/low construction, and ordinary velocity-gradient path reproduce the center material inputs bitwise for all 1,920 tetrahedra (including 1,728 paper rows). All 1,728 center paper projections pass the independent FP64 return map. The minus endpoint passes all 1,728 paper rows. The plus endpoint fails only medium tet 1691: the FP64 return map stalls at iteration 1 with row norm 90.81492362. That row remains null in `per-paper-tet.jsonl.gz`; the complete independent assembled centered secant is unavailable.

On the common 1,919-tet subset, native and independent centered endpoint secants agree to relative `0.0014823`. Each differs from the center tangent on the same subset by about `0.2405`. The observed plastic-multiplier active/inactive predicate does not change on comparable rows, but this is only one predicate and not a full constitutive branch signature. The realized perturbation is not exactly the nominal correction: realized `dF` differs from `dt * captured H` by relative `3.433e-6`, and realized `dv` differs from the nominal correction by `1.054e-6`. Thus finite-step curvature or unobserved branch behavior remains possible; the results do not isolate pure FP32 quantization as the cause.

Native force/residual bookkeeping agrees internally: native force formula versus captured residual delta is relative `1.58e-5` on plus and `1.65e-5` on minus. On valid independent projection rows, native versus independent force-increment slopes differ by relative `1.65e-5` (plus) and `1.84e-5` (minus). The native full centered residual secant still differs from the independent center A action by relative `1.0948`. The failed tet's native contribution is retained separately; it is not imputed into the independent secant.

The exact numerical metrics and all per-paper-tet projection outcomes are in `report.json` and compressed `per-paper-tet.jsonl.gz`. `partial-failed-002-audit.py` preserves the prior script byte-for-byte; the preceding attempt had failed while writing due to a full filesystem. The compact rerun used strict JSON serialization (`allow_nan=False`) and produced less than 1 MiB total package output. Its command was:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 /Users/home/cardboard-implementation-20261014/analysis/source125-kinematics-and-native424-force-audit-003/audit.py
```

The package manifest binds the script, report, compressed rows, source files, capture, native sweep receipt/report, and independent reference helpers. The failed 002 attempt remains unchanged.

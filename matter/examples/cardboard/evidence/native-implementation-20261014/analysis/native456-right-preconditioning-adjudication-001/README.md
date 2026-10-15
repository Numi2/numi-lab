# Native 456/457 residual adjudication

CPU-only independent analysis of native residual replay packages.

Run `python3 adjudicate.py` to verify capture manifests and receipts, check input payload equivalence, and regenerate `summary.json` and `residuals.csv`. Run `python3 seal_report.py` afterward to regenerate this report, source bindings, and the local artifact manifest. Both scripts read immutable 456/457 capture artifacts; neither launches a native/GPU command.

`residuals.csv` contains all 300 observations. `summary.json` records per-run summaries and input identity. `source_bindings.json` identifies the exact native helper, source snapshot, compiled world, 146 direction studies, and every payload. `report.md` states the bounded result and its limits.

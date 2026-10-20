# Native contact precision 025 — prior contact failure cleared, later rejection remains

Execution exited 1 after 377.096 s with 20 accepted coupled CSV rows through 2.5 ms. The previous contact failure signature (matter code 6, object 9, index 1975, 2 microsteps) cleared in the targeted root-20, microtick-2, contact-slot-1975 trace: all 24 contact captures have top-level diagnostic reason code 0 (`none`), positive lane bounds before and after (both exactly 1), and full-fraction CCD certification. The 17 stored alpha trials are finite, CCD-certified and accepted; no stored trial has reason code 6.

The run then stopped on a different failure: matter code 10, global object sentinel 4294967295, failure index 23, 3 microsteps. Its first diagnostic value is positive (0.00048828125); the complete failure record is in `failure.json`. The 32-step horizon did not complete.

`resting-coupled.csv` contains all 20 accepted rows. `stderr.log` retains the full 76 KB run diagnostics; `selected-contact-trace.stderr.log` and `targeted-contact-analysis.json` isolate the requested contact witness. `selected_stdout.log` retains run setup and step markers while excluding bulk node/tet records. Exact executable, metallib, suite-library and source hashes are in `source-binary-binding.json`; the 13 invocation file assets are bound by the unchanged hash/size manifest in `input-assets-reference.json`.

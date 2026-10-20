# Native contact Armijo diagnostic 024

This is a failed 21-step native run retained as a numerical diagnostic. It advanced through 20 accepted coupled CSV rows, then exited with code 1 after the captured failure at root 20: matter code 6, object 9, local failing index 1975, microsteps 2. The parser selected the `after_deformable_limit` event with diagnostic code 6 (`contact_barrier_armijo_failed`).

The stderr file retains 24 trace-capture records; the analyzer selected 14 matching contact records and 12 attached CCD trial rows. All 12 trial rows had finite barrier values, were CCD-certified, and were rejected; the analyzer also confirms exact byte parity for the first 20 coupled CSV data rows against native 022. This does not establish a mechanics pass or sustained support.

Wall time: 376.210 s. The run remains bound to source base `baeb29f7d468a23d4a152848b0ca9953accd8abb`, executable SHA-256 `83bf98bee369a57a9dd19e74a6aea343cc767486e189d0ff6a2aa6c8982e5b2e`, and the exact source files listed in `provenance.json`. The 13 invocation file assets match the input hashes in native 022 and verification 023; see `input-assets-reference.json`.

`stderr.log` retains all 14 trace records. `selected_stdout.log` retains selected setup, ownership, transfer, and step-timing lines with original stdout line numbers. The large run pack and bulk raw stdout arrays are not copied.

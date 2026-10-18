# Shared-tissue rollback and FEM terminal audit

Build 015 and native run 005 passed on the whole application. The test accepted two body steps at 0.25 ms each, injected a late GPU rejection at step 1, verified state restoration and a same-context retry, and passed the terminal FEM audit. The exact invocation, build/source pins, selected native log records, and receipts are included.

The two accepted coupled CSV rows match the 009 baseline byte-for-byte. That comparison covers only those CSV rows, not full raw-state equivalence. The run covers 0.5 ms; it does not qualify long-horizon whole-body behavior or visual output. Runs 001–004 and their failures are retained in compact form.

# GPU timing profile 031

Two sequential two-step, 125-microsecond instrumentation runs exited 0. Both profiles' first two accepted CSV rows match native-030 byte-for-byte, including each other. This verifies only those captured physical rows before interpreting timing output.

The sparse profile selects all kernels at root 1 with a 256-dispatch interval. It retained 630 elapsed-time samples and 224 dispatch-count records (339,056 summed reported dispatches). The contact profile selects sort-surface-primitives, compact-deformable and scan-deformable-node-incidence at every dispatch for root 1. It retained 316 elapsed-time samples and six dispatch-count records (600 summed reported dispatches). Full stderr logs and parsed timing/count CSVs are retained by profile.

No explicit dropped/unavailable-sample counter appears in either log. That absence cannot be treated as zero loss. The root filter covers root 1 only; the CSVs have two accepted rows but there are no timing samples for root 2. The selected kernel sets and sampling intervals differ. Wall times (~27.9s and ~27.8s) are setup-dominated two-step replays and are not used to claim speedup or rank whole-scene cost. No supported-motion, steady-runtime or physiological qualification is implied.

Each profile keeps its own invocation and pre-run provenance. Their executable, Metal library and recorded source-file hashes match the native-030 source-67 baseline; the provenance is retained verbatim. The actual row-by-row parity result and hashes are in row-parity.json.

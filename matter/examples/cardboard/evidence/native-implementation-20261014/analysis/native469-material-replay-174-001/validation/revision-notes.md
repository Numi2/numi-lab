# Pre-seal validation corrections

These were package-authoring defects, not native material findings. The first validation-driver attempt had a Python keyword-argument syntax error before running a target. The next orchestration attempt completed strict syntax, object compile, and link, then stopped before export because it tried to hash the not-yet-created export output. The complete rerun below passed.

A transient copy of the source169 helper had one unnecessary `#include <optional>` line (SHA-256 `0058956fd78d83a80f1293f74e6b3cad5792c09092e7c73018b208f36cc466ef`). The exact one-line difference is retained in `superseded-helper-optional.patch`; `<optional>` now belongs in the wrapper and the packaged helper is byte-identical to source169 (SHA-256 `865a7399791390c3a1d27f9e8d70bb0aa9a5f8e52ebec14cd350a0c3fe258b81`).

The first summary-writer draft omitted a comma between `epsilon_sweep_rows` and `selected_mp_failure_roots`. The final source fixes the JSON delimiter. This source-level correction was caught before native execution; no native output was produced by the defective draft.

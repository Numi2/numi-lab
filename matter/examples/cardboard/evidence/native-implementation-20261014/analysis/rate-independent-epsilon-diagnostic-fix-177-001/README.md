# Rate-independent epsilon comparison repair

The source469 replay instrument now contracts its independent dP/dF with the exact realized deformation perturbation. The old average with dt times the realized rate perturbation introduced a direction error when FP32 rate rounded differently. Admission rejects compiled RATE opcodes, preventing this F-only reference from silently covering rate-dependent laws. Requested and realized rate deltas remain in every output row. The mathematical diagnostic uses a new v2 schema; the original v1 outputs remain unchanged.

The companion CPU analysis reuses the closed174 native outputs, independently projects all576 generated paper endpoints, and separates reference secant truncation from native endpoint stress error. Nine strict reference failures remain visible; no universal noise floor or whole material qualification follows. No physical solver or tolerance changes.

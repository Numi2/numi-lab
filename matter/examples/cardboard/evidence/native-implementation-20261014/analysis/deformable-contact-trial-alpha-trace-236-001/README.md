# Source236 selected per-alpha deformable contact trace

This is a read-only diagnostic overlay based on source233. It adds a bounded 32-entry record for one selected deformable contact at a selected control step. The trace stores the exact tested endpoints and production CCD/barrier-gate outputs; it does not change acceptance thresholds or solver state.

The source-bound Release build compiled the full-contact checker and probe. Thirteen solver-environment contract tests pass. No GPU run was performed. Observer-on/off physical-history parity remains a required next check, and this package does not establish physical qualification.

See source236-overlay.json for complete base and candidate file hashes, patch hash, build artifacts, preserved malformed-serializer incident, and limits.

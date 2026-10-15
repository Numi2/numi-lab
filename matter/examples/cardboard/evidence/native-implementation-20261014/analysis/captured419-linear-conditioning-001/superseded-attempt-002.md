# Superseded attempt 002

This invocation corrected the destructive SVD workspace bug, but used `-W error` with macOS Accelerate-backed dense vector/matrix products. Accelerate emitted divide-by-zero/overflow/invalid floating-point warnings despite finite returned modal products; the script stopped before writing an authoritative report. The final run uses explicit `einsum` for the small modal projections and the original CSR matrix for residual/operator matvecs, and runs with warnings treated as errors. No outputs from this attempt are used in the report.

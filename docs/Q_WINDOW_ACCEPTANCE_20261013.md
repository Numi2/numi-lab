# Accepted-Q audit window reconciliation, 2026-10-13

The published accepted-step parser and cadence helper retain the original inclusive Q window and segmented diagnostics. Current COM sampling includes each Q row and its available predecessor, so accepted displacement has an independently observable preceding state.

On the current native M4 source at c86f37ac45a7683a0630146e7ba8e52afdabf036, matched four-step runs with the same pinned constructor state compare the default control with Q auditing restricted to steps 2–3. Terminal physical STATE bytes and the shared final COM/momentum row are identical. The control emits COM step 4; the candidate emits COM steps 1–4 and Q rows only at steps 2–3. All before/preprojection/accepted Float32 q[129]/v[128] FNV fingerprints independently verify, and accepted step 2 equals before step 3. The current CPU parser/cadence regression target also passes.

See `Q_WINDOW_NATIVE_VERIFICATION_20261013.json` for the source, invocation and evidence identities. These checks qualify bounded diagnostic behavior, not performance or physiological capability. The old COM-only callback rollback and earlier parser/publication workspaces are superseded. Their exact staged/unstaged changes and original HEADs remain recoverable through the reconciliation archive and recovery refs.

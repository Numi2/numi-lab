# Exclusive derivative-sweep output

This overlay supersedes the unexecuted checker writer in frozen092. It reserves
each output with `O_EXCL` before dispatch, retains partial output on failure,
handles interrupted and partial writes, caps each output at 16 MiB, and records
the compiled local projection budgets. Existing files are never truncated.
The original001 overlay and frozen092 remain unchanged.

The sweep retains eight captured centers, nine basis directions, six signed
perturbation sizes, three center repeats and the same accepted history per
evaluation. No production shader, physical operator or acceptance rule changes.
Repeated centers measure same-batch repeatability, not an arithmetic noise floor.
Native branch-aware and independent accuracy checks remain separate.

# Native 467 capacity interruption audit

Native 467 was stopped by its launch-capacity guard after free space crossed the configured 512 MiB floor. The guard signalled only native child process group 31314. The source-bound runner receipt records exit -15 with `timed_out=false`; this was neither the runner’s 14,400-second timeout nor an ordinary nonlinear-solver result.

Control 0 completed and produced one accepted step for each environment. Control 1 reached encode end but has no completion record or output transaction. Keep the result classified as incomplete. The complete 8.7 MB stderr log was left in place and is bound by SHA-256 in `report.json`; only a short lifecycle excerpt is copied here.

The report binds the case prediction/receipt, output manifest/observations, launch command, and both capacity-guard records. No process signal, retry, or GPU action was issued by this audit.

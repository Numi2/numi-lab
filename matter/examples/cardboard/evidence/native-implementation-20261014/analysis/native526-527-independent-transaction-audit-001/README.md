# Native 526 versus 527 audit

CPU-only verification of source198 run receipts, captured output inventories, baseline/aliased parity, transaction snapshots, Armijo traces, and the source-level status path. No native rerun.

526 reaches the intended postCommit negative control: all 32 preDynamics Armijo records have solver status 0; five corrections are selected and 27 records are certified no-ops. The third transaction then reports status 1 / INVALID_DISPATCH at rigid proxy index 0 with diagnostic -3. The fixture mutates punch motion type to dynamic only for the postCommit endpoint. Runtime invokes prescribed-motion validation there, and the kernel rejects that non-kinematic prescribed body. Accepted state and cursor are rolled back.

527 fails earlier within the Newton18 line search: pretrial status is healthy, but the line search returns status 10 / NONLINEAR_SOLVER_FAILURE at failing index18; Newton19 onward is blocked. PostCommit validation skips unhealthy environments, so the injected fault is masked. Rollback is demonstrated, but not postCommit-fault coverage.

The regional option selects the source198 regional contact preconditioner, a conditioning path; fixture/material/timestep/acceptance settings are otherwise shared. Iteration counts and accepted trajectories differ. No matched same-state matrix action was captured, so this report does not claim direct numeric operator parity. Both receipts explicitly say physical validation is false.

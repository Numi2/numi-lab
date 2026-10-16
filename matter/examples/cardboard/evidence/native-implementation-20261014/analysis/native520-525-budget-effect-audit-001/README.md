# Native 520/525 solver-budget audit

This read-only audit binds the native 520 rejection and native 525 extended-budget run to the frozen source195 snapshot, source file, material, executable, metallib, capture and receipts. It makes no source changes and launches no native work.

The source CLI's extended-budget switch changes four solver budgets together: Newton 16 to 32, FGMRES 64 to 256, restart 10 to 16, and line-search 8 to 24. The residual tolerance remains 1e-7. The native 525 loaded control is accepted after its logged residual reaches 9.98805945e-8 at Newton24, followed by certified no-ops. The next control still exhausts line search at Newton18, then enters status10 at Newton19 after 256 FGMRES iterations. 525 therefore does not establish a complete successful fixture trajectory. It also selects scalar diagonal preconditioning, not regional contact preconditioning.

The result is evidence for the combined extended-budget setting on this fixture, not a restart-only causal result. The synthetic fixture and short native execution do not qualify shipping, material calibration, or physical contact.

Re-run the read-only evidence extraction on the host with:
python3 /Users/n/cardboard-implementation-20261014-host-macmini/analysis/native520-525-budget-effect-audit-001/audit.py

# Coupled FEM tangent-block screen 066

Question: does the existing optional 3x3 coupled-FEM tangent-block Jacobi preconditioner improve native Human mechanical-root throughput while preserving the accepted physical state?

Pre-registration: 2026-10-20, before screen-066 outcomes. Use the exact native executable, source revision, full Human composition, root pose, timestep, materials, contacts, masses, Newton/FGMRES budgets, convergence tolerances, and environment from frozen audit input 062. Only change `NUMI_HUMAN_RESTING_COUPLED_TANGENT_BLOCK` (unset for control, `1` for treatment). Each run advances one authoritative root at 125 microseconds, from identical initial state. Run ABBA: control-1, block-1, block-2, control-2. No visual inspection.

Prediction: the block may lower time if improved conditioning offsets its additional per-node work; require both paired treatment/control RTF ratios >=1.15 for promotion. Any rejected root, changed mass/contact/continuum certificate, or root assistance invalidates the screen. Compare declared terminal physical metrics with max(1e-7 SI units, 1% of the larger magnitude) absolute tolerance. This is a one-root exploratory screen; it cannot establish long-run stability or full coupled-physiology performance.

Stop on first rejected run or failed native certificate. Preserve all logs, command arguments, environment, source and binary identities, process wall, GPU timings, and comparison results. No FGMRES/Newton budget reduction is allowed.

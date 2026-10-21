# Three matched combined optimization pairs

SSH Numi, frozen build202; alternating order; Metal API validation disabled for timings. Enabled together: NM_FEM_PARALLEL_CONSTITUTIVE_BACKTRACKING, NM_FGMRES_FLAT_COLUMN_ICB, NM_MATTER_CONTACT_ZERO_STEP_GEOMETRY_REUSE. Fresh-RHS preflight remained disabled in both cohorts.

All three comparisons match all 17 endpoint measurements and all 68,335 FEM records byte-for-byte. Median post-first-root body throughput ratio is 1.133553 (13.36% higher throughput, 11.78% less time); median complete-process ratio is 1.046781. Ratios are measured jointly, not multiplied individual estimates. Exact spans, process times and source/binary/asset invocations are in the archive and JSON.

Each process simulates only 0.375 ms during startup. This does not qualify supported rest, sustained throughput or native presentation, and does not meet the 0.1x requirement. Initialization is still expensive; later physical qualification remains blocked by a coupled nonlinear/conservation failure. No visual verification was performed.

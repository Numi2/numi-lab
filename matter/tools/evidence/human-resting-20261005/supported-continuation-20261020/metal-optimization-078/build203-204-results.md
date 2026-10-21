# Builds 203–204: fresh-RHS preconditioner isolation
All execution on SSH Numi; no visual verification. 0.1x remains unmet.
Build203 gates all eleven cache producers together and requires the outer completion latch; final Newton setup remains full. The 58 focused and 73,527 full native checks pass. The three-root integrated candidate still fails the 1% gate (1.44% acceleration, 2.52% maximum determinant); remains disabled.
Build203 one-root on/control outputs match all 17 endpoint metrics and 68,335 FEM records. Divergence first appears at the following root.
Build204 adds explicitly opt-in diagnostic modes: mode1 executes the fresh measurement with direct full setup; mode2 uses indirect full setup. Both match control through three roots, all 17 endpoint metrics, all reported active iteration counts, and full FEM records. This excludes the fresh measurement and full indirect dispatch shape as causes; skipped cache production remains under investigation.
Metal API validation found a missing buffer5 in the multi-environment fixture invocation; fixed before the successful 64-case focused and 73,527-case full native run. The rejected attempts are retained.
These are 0.375ms startup checks, not sustained, supported-rest, timestep, presentation, or integrated performance qualification. No physical acceptance gate changed.

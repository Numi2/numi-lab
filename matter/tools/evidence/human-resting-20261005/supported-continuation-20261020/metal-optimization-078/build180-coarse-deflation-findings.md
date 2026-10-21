# Coupled coarse factor repair

Execution: SSH Numi, Apple M6. No visual verification. Build180 source closures and Metal API validation are retained. This is not 0.1x qualification.

The captured 32-mode RAP had exactly two zero rows/columns (modes6,26). Both normalized basis vectors were nonzero; these coarse directions were unsupported by this linearization. No active-set state is inferred from the capture. The remaining30-mode matrix is full rank: raw condition number approximately223000, equilibrated approximately1680; CPU Float32 emulation minimum pivot0.00422 versus floor3.58e-6. The earlier mode31 null claim was incorrect: cavity pressure is supported.

The optional equilibrated coarse LU now deflates exact unsupported rows or columns to closure, zeros their coefficient scales and retains all supported modes. This changes only the preconditioner, never physical equations or residual certification. Synthetic asymmetric zero-column and exact-null cases pass alongside rank/near-rank rejection:168 native GPU checks with Metal API validation.

The full-body one-root comparison now reports96/96 valid factors and1782 nonzero corrections, versus no valid factors before this repair. All17 endpoint comparisons pass1% (maximum0.9153%). However body time worsens7.003618s to12.708843s, used columns1587 to1782, active Newton begins34 to71, and the candidate last Newton norm1.1442e-5 misses its1e-5 target. It is not promoted. A passing endpoint comparison alone is insufficient.

The next targeted change is residual-corrected composition with existing local preconditioners using the retained operator images, rather than repeating an additive coarse experiment. Completed exact-zero certification work and optic graph patches retaining the full skin basis are independent implementation tracks.

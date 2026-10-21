# Source-M triangular solve increment

The physical source-mass query service distributes forward/back substitution dot products across SIMD groups for eligible n>=64 systems; scalar fallback remains. It retains the existing Cholesky factor, diagonal equilibration, force units and transaction factor lifetime.

Three alternating native paired measurements against the prior parallel-factor implementation give median reused-solve speedups 5.93673x at128DOFs and10.20832x at256DOFs. The32DOF fallback remains unchanged. Additional factor speedups are1.16944x/1.16342x. Known-solution/original-matrix normalized residual and forward-error checks cover eight decades of generalized-coordinate scaling; the full production MetalRobo.metallib passes below7e-7. All eight native source-M, mixed-unit, staged force, rollback, nonfinite rejection and ABI fixtures pass.

The metadata archive retains all three pairs, production-metallib test logs, transaction fixtures, and frozen141 hashes. These are local source-M service gains, not integrated Human throughput. The minimum0.1x remains unmet. Actual implicit Human force production is a separate unfinished path.

The original metadata archive filename filter omitted the promised standalone test and paired-kernel sources. The additional build141-sourceM-simd-sources.tar.gz contains those exact files, independently reopened and hash-verified; the original archive is retained. Shader compilation used Metal4.0/O3/fno-fast-math, with the existing source-M scale checker linked against Foundation/Metal.

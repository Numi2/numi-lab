# Source101 operator-path split addendum 005

This addendum supplements audit-004 without changing its sealed files or the native run. It binds source101 runtime/operator dispatches, the exact 360 capture, the existing FP64 cached-matrix reconstruction, and cycle-6 true-residual metrics.

Arnoldi FGMRES uses the cached per-tetrahedron 15-state tangent in this run (`executionBudgets.w == 1`, confirmed by the manifest and capture header). The fresh true-residual path and selected operator-FD action instead pass `useCache=0`. The diagnostic true-residual replacement option was off, so it measured fresh `b-Ax` but did not replace the Arnoldi recurrence or alter its convergence latch.

The operator-FD capture applies the selected `femSolution` correction using the fresh path and separately snapshots the cached local tangent. The existing offline assembler forms a cached-tangent reconstruction of `A*delta`; its action differs from the native fresh action by 6.322865e-6 Ns (0.80697%, max component 1.933147e-6 Ns). This is not a same-vector native cached-vs-fresh pair: offline FP64 assembly and native FP32 action arithmetic are also part of the difference, and no noise floor was measured.

For the same correction, the cached-tangent reconstructed residual norm is 2.158307524e-4 Ns; the fresh native true-residual norm is 2.157817650e-4 Ns. Those norms differ by 4.898736e-8 Ns (0.0227%), while each is about 22.1 times the cycle-6 Arnoldi estimate (9.7661823e-6 Ns). This shows the operator-path difference alone does not account for the scalar norm gap, but does not prove recurrence loss: a second native useCache=1 matvec and the fresh residual vector were not captured.

All 1,920 environment-0 tets have valid finite projection/tangent-cache status; this does not test accuracy. A per-tet center material/tangent `<=1e-3` reference gate was not run. The assembled action mismatch is larger than `1e-3` if treated as a generic action gate, but it is not a local tangent gate and has no noise-floor qualification. The independent derivative `<=1e-6` gate is unavailable: both FD epsilon-pair plus/minus endpoint evaluations returned status 5, so no valid assembled FD comparison exists.

No solver change or physical qualification follows from this addendum. The native transaction later rejected Newton23 after 24 line-search backtracks. See the machine-readable bindings and source line references in `operator-path-split-005.json`.

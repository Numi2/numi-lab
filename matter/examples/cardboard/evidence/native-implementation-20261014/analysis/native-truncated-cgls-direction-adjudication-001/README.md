# Native CGLS direction replay 440 adjudication

This CPU-only audit binds native replay 440 to the source125 captured operator, the CGLS direction package, and the earlier native424 center. It checks every recorded native artifact, the compiled helper/source build, exact CSR reconstruction and the per-profile direction payloads. Run it with:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/native-truncated-cgls-direction-adjudication-001/audit.py
```

It verified 278 native output files, 24 CGLS package files, four compiled-helper files, ten source125 build artifacts, and the native424 file inventory. The replay reports ten successful preflight/native commands, zero physical transactions, unchanged prepared inputs, all 15 residual-sweep status/history records per profile successful, and exact candidate-node/residual/force center bytes matching native424. The exact source125 unsymmetric CSR reconstructed from the retained operator capture has 2,538 free DOFs and 71,046 nonzeros.

At alpha 1, native residual ratios to the center were 0.313115 (FP64 CGLS32), 0.262344 (FP64 CGLS64), 0.268066 (FP64 CGLS128), 0.411093 (emulated-FP32 CGLS16), and 0.269338 (emulated-FP32 CGLS128). For every profile alpha 1 was the best tested positive alpha; all tested smaller positive alphas monotonically worsened the native residual. The exact linear CSR ratios at alpha 1 were 0.299390, 0.230056, 0.152613, 0.407697, and 0.160119, respectively. In particular, the 128-iteration FP64 direction has a better linear prediction than the 64-iteration direction but a worse nonlinear replay residual.

Cached and fresh native A*p actions are compared against A*p formed from the exact CSR and exported FP32 payload. See `report.json` for per-profile absolute/relative action errors, the full signed-alpha nonlinear residual curve, and comparisons with b-alpha*A*p. The tested positive-alpha set is 1, 1/2, ..., 1/32; alpha=2 is kept as a diagnostic sample but excluded from the best-alpha selection.

These are offline candidate evaluations only. No direction was installed into the solver, and the original Newton certificate does not transfer. The residual replay is not a CCD/Armijo acceptance, physical transaction, full assembly, or convergence qualification. The observed alpha 1 improvement only motivates a later controlled native implementation/replay. CGLS is a least-squares method on the captured operator; it must not be assumed stable or qualified merely because the offline linear residual decreases. A future native experiment should compare bounded CGLS against a transpose-consistent LSQR/LSMR path, preserve the exact physical operator, run production nonlinear trial checks and CCD/rollback, and retain the original terminal acceptance/certificate gates.

## Binding hashes

- Native 440 receipt: see the three `external_bindings.native440_*_sha256` fields in `manifest.json`.
- CGLS package manifest SHA-256: `37d9bfcf6706f38d08f5cfc89a2f60c0df74b9f3da35e78ada43df4c82fa6b4e`.
- Native helper compiled-program SHA-256: `b2f37a4eaaf922eb8ed8c5cbc721ab8ef174c230b56bc02f522b44ca5b3aea1f`.
- Source125 build manifest SHA-256: `35b821184ca36ad57c529bc0e33e4f20519996def1c312f8ac0f8028ecadb8e6`.
- Exact replay report SHA-256: see `manifest.json` (`files.report.json`).

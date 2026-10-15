# Native 436 regularized-direction adjudication

This CPU-only package audits the four source125 native residual replays in analysis/native-regularized-direction-replay-436. It verifies the replay files and helper binary/source against their manifests, rebuilds the exact captured CSR identity from the source125 capture, recomputes each exported FP32 direction's CSR action and linear residual prediction, and compares all four replay centers to native424.

Run with:

    python3 /Users/home/cardboard-implementation-20261014/analysis/native-regularized-direction-adjudication-001/audit.py

The audit checks every byte listed in the native436 manifest (223 files), all source125 build artifacts (11), the four entries in the native helper's compiled-program.json map, proposal inputs, and native424's 84 manifested files. The native executable is analysis/native-residual-direction-replay-419-001/build/native_residual_direction, SHA-256 cc872337366821d0752665ea0feb913f38c1ff7cd921696c395c9861fabccbec; its source native_residual_direction.mm SHA-256 is b00923d80b7897d4939c090ecbdbecdf48b63726ea188719ebca328675f76310. compiled-program.json SHA-256 is b2f37a4eaaf922eb8ed8c5cbc721ab8ef174c230b56bc02f522b44ca5b3aea1f. The source125 build manifest SHA-256 is 35b821184ca36ad57c529bc0e33e4f20519996def1c312f8ac0f8028ecadb8e6, and the compiled package SHA-256 is cafbf36052bbfa72e8be49fdba8b23e900469c9f15136a14a4ec1e4c0b5ccf7c.

## Findings

All four profiles passed preflight and native replay with exit code 0. The receipt records zero physical transactions and unchanged prepared inputs. Each profile emitted 15 signed residual evaluations: center plus positive and negative directions at 1, 1/2, 1/4, 1/8, 1/16, 1/32, and an additional diagnostic 2 step. Every recorded status pair is [0,0]; every record says accepted history is unchanged.

The center candidate-node, residual, and internal-force outputs are byte-identical across the four profiles and to native424. The center free residual is also exactly equal to the captured RHS in the audited FP64 comparison (L2 = 1.8444380407784322e-4 Ns).

For each exported direction, the native cached and fresh operator actions agree with the exact unsymmetric source125 CSR action to relative L2 error between 5.56e-8 and 6.09e-7. This is a captured-operator replay consistency result.

At alpha=1, the native nonlinear free-residual norm and the exact CSR linear prediction were:

| Direction | Native nonlinear residual (Ns) | Native ratio to center | CSR linear prediction (Ns) | CSR ratio to captured RHS |
|---|---:|---:|---:|---:|
| Raw Cauchy | 1.60052278e-4 | 0.867756 | 1.60055366e-4 | 0.867773 |
| Tikhonov 1e-2 sigma ratio | 4.23796319e-5 | 0.229770 | 3.95745441e-5 | 0.214562 |
| Tikhonov 1e-3 sigma ratio | 5.08658842e-5 | 0.275780 | 1.21817512e-5 | 0.066046 |
| Tikhonov 1e-4 sigma ratio | 7.13754899e-5 | 0.386977 | 5.98135996e-6 | 0.032429 |

The native residual norm decreases monotonically over the sampled positive backtracking values for every profile; the minimum among 1 through 1/32 is at alpha=1 for all four. All sampled negative-direction residuals increase from the center. The extra alpha=2 point is reported but excluded from the backtracking-best comparison.

The native nonlinear and exact linear curves are close for raw Cauchy and Tikhonov 1e-2. They diverge substantially at full step for Tikhonov 1e-3 and 1e-4. Thus the linear ratios are not reliable full-step nonlinear predictions for the smaller regularization values, even though their actual sampled nonlinear residuals still decrease.

## Interpretation and limits

These runs directly re-evaluate the production mechanical residual at prescribed alternative corrections around one frozen source125 candidate/history. They do not execute a physical transaction, line-search acceptance, CCD, or terminal solve certification. The original Newton certificate does not transfer to these directions; no tested alpha is called accepted. The result is evidence that these four samples reduce the measured residual at this one center, not that any is a valid or robust solver direction.

LSQR/LSMR is a reasonable candidate generator to investigate for this captured unsymmetric least-squares problem because it avoids explicitly forming A-transpose-A; CGLS would apply the normal equations and square the conditioning. The current evidence does not qualify either method as a production globalization path. An online use would need a source-bound, consistent A/A-transpose pair, bounded iterations and failure handling, then the ordinary nonlinear residual, geometry/contact feasibility, Armijo acceptance, rollback, and unchanged final certificate on the actual state. This replay supplies only A*p actions and nonlinear residual samples for prescribed vectors; it does not supply an accepted solve or test a CGLS/LSQR implementation.

No physical, assembly, material calibration, or solver-convergence qualification is claimed.

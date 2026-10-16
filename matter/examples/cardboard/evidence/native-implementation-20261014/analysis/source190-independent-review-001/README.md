# Independent source190 review

CPU/static source review only; this is not native or physical qualification.

The audit binds the source190 package, patch, frozen source maps, build artifacts, and retained CPU/build checks. Source190's CGLS experiment is bounded to the exact 1e-3 damping setting, a single-Newton fallback, and a 32/64 iteration budget. The changed direction is checked against the unchanged physical operator and then uses the existing nonlinear Armijo, acceptance, and rollback path. The prior FGMRES certificate is only a prerequisite and does not certify the changed direction.

One inherited issue blocks native use: the layout adapter writes selected[environment] from both the mechanical and field halves of its 2*E*N dispatch. The stores are equal, but concurrent non-atomic writes are a data race. The source190 and source188 copies have identical hashes, so this is not a source190 regression. A later overlay should restrict that store to the mechanical block and test one writer per environment across multiple E values while preserving all other row-layout behavior.

No native Metal execution or full coupon qualification is claimed. Exact hashes and checks are in report.json.

# Builds 214–219: coupled initialization and solver work

The 0.1× end-to-end target remains unmet. These are startup mechanics diagnostics, not a settled supported checkpoint or five-minute physiological qualification. No visual verification was performed.

## Implemented and verified
- First-cycle aggregate solver ICB scheduling (opt-in), paired with the ordinary path: all 17 reported endpoints match exactly. Body spans changed from 6.7809/4.2589/3.3931 s to 6.5501/4.2033/3.2571 s in one matched startup pair. Process startup includes compilation and is not a steady-state gain.
- Empty padded regional tangent projection tiles now write their owned zero partials without loading or reducing inactive work. Empty MPM worlds omit MPM pipelines and dispatches only after a complete zero-capacity/no-owner proof.
- Aggregate ICB telemetry distinguishes selected command spans from executed child kernels.
- Pressure-load continuation uses typed ABI-47 microstep state; only vascular traction and its tangent are ramped. Circulation and blood mass are unchanged. Default-scale integrated comparison is exact.
- Optional exact blood-mass mechanical derivative includes both the current-volume and acceleration derivatives. Finite-difference fixtures pass. Worlds with no registered mass owner treat this derivative option as an explicit fingerprinted no-op; malformed bindings still fail.
- Optional compensated cavity geometry and semismooth zero-flow transport now work with ordinary bidirectional and directional edges. Full cavity fixtures pass with compensation, persistent low words, semismooth transport and blood-mass tangent enabled together. Positivity, conservation, replay and rollback gates remain active.
- Refinement selection now attributes discrepancies per anatomical region, with deterministic source-bound selection and three passing CPU tests. It is a construction tool, not an accuracy certificate.

Build219 native FGMRES kernel contract: 77,465 checks pass. Default and compensated full vascular cavity checks pass. Thirty-four source contracts and three refinement tests pass.

## Measured failures retained
- Fresh-RHS cache-producer skipping still fails the existing 1% comparison: isolating producer families and disabling cross-Newton reuse did not resolve history dependence. It remains disabled. The terminal observer in build219 did not emit captures because its admission predicate was over-restrictive; absence of output is not evidence of cache parity.
- The regional815 candidate reduces the global mechanical coordinates from 41,796 to 12,468, including 1,884 coarse coordinates for 31,212 selected DOFs. It fails: attachment-force error is 4.13%, and selected shell strain discrepancy is 195.4%. Source identities, topology, rest positions and physical mass match. No qualified reduced-model speedup is claimed.
- Three-root timestep diagnostics with compensated cavity and semismooth transport accept 0.125, 0.25 and 0.5 ms. Their measured body-only factors are approximately 0.0000105×, 0.0000159× and 0.0000317×. These short release/initialization diagnostics do not qualify those timesteps at supported periodic rest.
- A 1 ms step fails mechanical convergence at step zero. Existing residual-adaptive forcing variants and an eight-step vascular pressure ramp also fail at step zero. The 2 ms case was not run after the failed gate.
- Full initial rigid support balances the articulated mass, but deformable tissue load transfer and its supported initialization remain unresolved. No hidden support, force deletion or acceptance-tolerance relaxation was introduced.

## Evidence and resources
The accompanying archive and receipt retain exact frozen dirty-source closures for builds214–219, invocation/device/configuration receipts, failed and passing trial logs, comparisons and the build219 native checker binaries. Archive members were independently reopened and SHA-256 verified.

Hash-verified APFS clone replacement of byte-identical completed outputs recovered approximately 15.4 GiB (df: 11.37 to 26.73 GiB available), preserving all file bytes. The exact per-file receipt is retained. No source or unique evidence was removed.

Further work remains on whole-stage GPU command suppression, supported load feedback and an accuracy-qualified reduced coordinate space. None of the measured gains establishes 0.1×.

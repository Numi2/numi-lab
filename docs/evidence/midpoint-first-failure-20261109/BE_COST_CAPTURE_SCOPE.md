# Scope of the matched native cost comparison

The matched cost harness runs complete accepted native transactions without taking per-step host diagnostic snapshots. It takes one completion snapshot after the timed interval to verify final physical state. Its output records `per_step_snapshots: 0`.

Both the compensated midpoint arm and the compensated backward-Euler control are currently admitted only with `RuntimeConfiguration::captureDiagnostics` enabled. The timing harness therefore records `capture_diagnostics: true` for both arms. This is a matched comparison of those instrumented runtime configurations, not a measurement of a diagnostics-disabled production build; enabling capture can change device work. The independent accuracy runs, timestep selection, exact final-state checks, unchanged law and physical error gates remain required.

Until the complete alternating timing pairs, trace audits and device-timestep physical-time correction pass, there is no performance conclusion. Even then, this n=2 fixture is not spatially converged.

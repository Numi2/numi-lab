# Accepted body mass and center of mass

The existing Metal presentation pass now reduces the committed articulated body positions and inverse masses into four diagnostic floats. They share the surface-audit output buffer and clock. The trace records body center of mass and represented mass on every displayed accepted frame; this adds no full-body host readback, physical update, or controller history.

The M4 Pro native scene completed 3,000 steps / 6.000000285 simulated seconds in 61.603131 seconds (0.097398× real time). All physiological/contact trace bytes, previous surface-trace values, and terminal body q/v values match the passive-viscera run exactly. Rejection and retry replay passed. Mass is 72 kg throughout. The last GPU COM differs from the existing terminal source-mass calculation by at most 19.6 nm.

During these first six seconds the COM moved [-2.900, +1.771, -15.249] mm. This records initial settling; it is not evidence that the later trajectory is stationary. The complete trace will distinguish initial settling from continued drift in the final long run. No threshold has been adjusted to turn displacement into acceptance.

The existing study parser admits complete, finite, constant-mass COM observations and rejects partial columns, nonfinite positions, zero mass, and changing mass. All 17 focused parser tests passed on the mini. Earlier traces without these new optional diagnostic columns remain readable.

`invocation.json` and `source.diff.gz` bind the frozen native source and runtime, while `parser-admission.json` identifies the subsequent analysis script. The movie and exact accepted render packs remain on the mini at the identities in `retained-large-artifacts.json`. Known cardiac-wall and organ-interface defects remain outside this numerical diagnostic admission.

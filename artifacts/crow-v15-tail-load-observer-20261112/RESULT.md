# Tail-inclusive bank-feedback force audit

The added tail observer exactly reproduces all 72 sparse q/v/action samples and all 9,600 prior wing/root wrench and source-state records in the matching bank-feedback run. Forward kinematics reproduces torso, both wing and tail center-of-mass positions at one matching replay state within 7.42e-8 m. This is a single alignment check, not full validation of every reconstructed position.

In the observed 1 s windows, mean world-axis tail yaw moment about the torso ranges from -0.002284 to +0.000765 N m; mean combined wing, torso and tail kernel yaw moment ranges from -0.013586 to +0.012871 N m. These are moments from this modeled aerodynamic kernel, not total external moments or evidence that a particular correction will produce a turn. The prior bank-feedback test made large sideways translations with little heading change; this measurement does not isolate their cause or qualify flight. The original omission of the tail from the previous wing/root totals remains explicitly documented in those artifacts.

The initial parity script compared against the constant-bank shader rather than the bank-feedback shader and failed, as retained in analysis-wrong-control.log. The corrected comparison uses the matching shader and its full input hashes; no data or simulation was changed.

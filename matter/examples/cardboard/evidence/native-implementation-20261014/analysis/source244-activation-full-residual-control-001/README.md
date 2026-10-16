Source244: opt-in full-residual mortar activation and body evidence

This overlay is based on immutable source241-004. It keeps the legacy default path: activation selector and sparse-private settings are cleared for the setup transaction. The new full-contact activation option is accepted only with native sparse-free-body parity and diagnostic-state output; it selects the existing full-contact residual Armijo implementation for the activation control only and requires the selected-path trace marker.

For each dense/sparse setup attempt, the diagnostic writes exact host scene input bytes (the array copied to MetalWorldBatch.initialSceneBodies), kinematic-target bytes, MetalWorldResult.finalSceneBodies if returned, runner scene state after completion, and returned q/v arrays. JSON records strides, counts, hashes, statuses, Matter before/after snapshot hashes, accepted-step cursor, and exact comparisons. The result scene is a published MetalWorld readback, not a direct raw MTLBuffer capture; the metadata labels that distinction. A rejected activation still records the output and host non-commit state before the test reports failure.

No solver kernel, physical operator, acceptance criterion, or tolerance changes. This is instrumentation plus an explicit path selector, not a physical qualification. No GPU test has been run for this overlay.

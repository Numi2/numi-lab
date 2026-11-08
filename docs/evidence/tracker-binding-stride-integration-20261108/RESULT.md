# Integrated object tracker path and stride verification

The integrated merge `33fe48f8a814561e37bb88ce60c0ef2c8640c85b` includes main `419130e08` and the prior integrated release work. On Apple M6, the renderer, full Metal shader, native visual probe, depth-validity test, two fingerprint migration checks, and task-program check were rebuilt. All five CTests passed. The native probe checks independent selected-pixel identity, exact normalized masked-depth output, compact history and actor observations for two environments, a nonzero output offset, and the graph-only no-binding control. The previous fixture failure, unchanged baseline, and eligibility-only stride failure remain in the main evidence.

This checks the affected integrated contracts and one selected scene. It does not establish physical-step rollback, deployment-valid perception, held-out task skill or performance. The old compact and extended policy fixtures are synthetic zero policies, not trained policies.

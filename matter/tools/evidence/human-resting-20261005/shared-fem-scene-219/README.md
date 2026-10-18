# Shared Human FEM scene: scoped source and evidence

This staged increment contains the 226 Matter microstep-count change, the matching 222 runtime finalizer, deferred-audit regression evidence from 225, the 009 headless shared FEM/fascia/CVSim run, and the retained 001–008 failure history. It is based on `9cc6e4e91ca41ef8c29261741171b230c486c589`.

The 009 run accepted two 0.25 ms steps (0.5 ms total), conserved the reported whole-body mass exactly, passed the terminal eight-strip fascia audit, and recorded positive finite FEM Jacobians for the inferred bilateral ACL references. It is a short mechanics integration slice; it does not establish whole-body qualification, a full breath or heartbeat, or deformable-organ/skin/fat/nerve contact. No visual verification was performed.

The 225 Apple M6 PASS is a separate deferred-audit adapter regression. The prior 001–008 failures are retained in `failure-history.json`; 007 is a resource-budget SIGTERM and 008 is the stale expected-microstep-count rejection. The 227 multirate regression passed on Apple M6: the unchanged 225 suite passed first, then the configured two-microtick step accepted, the expected-count mismatch rejected with bitwise rollback, and counts 0, 3, and 512 rejected. Exact 227 artifacts are under `deferred-audit-227/`. This focused adapter/runtime result does not expand the shared-scene qualification.

`evidence-manifest.json` pins the production source files, loaded binaries/libraries, native receipts, CSV, shader inputs, and test sources. `shader-inputs/` keeps the exact assembled shader inputs used by 225 and 226; the tracked production MSL source remains `matter/src/metal/numi_human.metalinc`.

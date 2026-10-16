# Source 252 free-body scene observer

This overlay extends the source244 activation observer to the subsequent free-body controls. For each attempted control it binds the control number and diagnostic attempt index to the exact input runner scene, kinematic targets, wrench vector, post-step host scene, available MetalWorldResult.finalSceneBodies vector, trace bytes, and pre/post Matter snapshot files and hashes. It records the runner accepted-step count and the Matter control/substep cursor.

The observer runs only under the existing --full-contact-activation-residual-armijo option, which requires native sparse free-body parity and a fresh diagnostic output directory. It captures controls 1200, 1201, and 1202 after the existing 1199 activation. A step that throws before returning a StepResult records the host and Matter state available after the existing catch/rollback path; published scene data is marked unknown. An empty returned scene vector is distinguished from an absent StepResult. Raw transient Metal buffers remain explicitly unknown because StepResult does not expose them.

The patch changes only matter/tools/cardboard_full_contact_armijo_check.mm. It does not change production kernels, test inputs, loading, control order, physical state updates, solver settings, or acceptance. The new --free-body-control-scene-evidence-contract mode exercises returned nonempty scene, returned empty scene, and missing-StepResult serialization using temporary CPU-only files and performs zero physical transactions.

Native 1200–1202 execution was not run by this package. The observer provides evidence for the next authorized replay; it does not make claims about acceptance, rollback, or physical behavior in those controls.

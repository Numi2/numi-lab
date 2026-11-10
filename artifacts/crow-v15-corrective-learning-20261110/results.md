# Student-visited corrective actor

The supervised development update used three physically recovered routing prefixes: the initial J recovery, late student-state J recovery and student-state I recovery. Each passed its recorded first-episode physical and pre-takeover parity checks; failed episodes and terminal landing segments were excluded. The result used 1311 training samples and 329 in-trajectory diagnostic samples from these correlated trajectories. The validation-command errors are recorded in training-results.json, but the diagnostic does not establish performance on new starts or layouts.

The frozen-flight controller, current v15 body and terminal landing actor were not retrained. A critic scaffold was used only to instantiate the generic supervised learner; it did not enter the supervised loss. The exported deterministic actor matched the trained actor's sampled outputs exactly. Its teacher-off development evaluation is a separate required experiment.

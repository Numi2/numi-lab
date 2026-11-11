# Pre-grip balance gain trial: native rollback

With the same eight perturbed development starts and uniform toe-release setup, this candidate increased pre-release/return fore-aft hip feedback gain from 0.5 to 0.8 and velocity damping term from 0.2 to 0.6. The release/return feedback, partial recenter, toe friction, physical body and native solver were unchanged.

The run exited with failure after 93 accepted logged endpoints per lane (zero-based steps 0-92). On attempted control step 93, the native solver reported status 11 in lane 1, substep 15, and rolled back the control step. The full simulator diagnostic and partial traces are retained. Logged endpoint positions and velocities remained finite, but this is not a complete-substep validity certificate. `partial-analysis.json` summarizes the accepted prefix. The candidate diverged from the reference by step 3, consistent with its pre-grip intervention.

The required release, final grip and reliability gates were not observed; the prediction is untested and this candidate is rejected for promotion. The diagnostic does not establish whether the control change, contact geometry, native solver or another interaction caused the rollback. No physical tolerances or event budgets were relaxed. The `analyze-high` scripts require a complete 550-step run and are retained as the originally prepared full-run analyses, not executed on this truncated output.

Source, patch, exact probe binary, input hashes, build log, compressed partial native state and contact outputs, and failure result are archived. No autonomous landing or biological conclusion follows.

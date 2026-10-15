# Coupled private Armijo trial capture overlay 002

This external observational overlay is based on frozen source 092 and supersedes overlay 001 without modifying it. It fixes two capture-only defects: intermediate residual stages were appended for every line-search trial in a selected Newton iteration, and the pre-rebuild context omitted the already formed private free-rigid state/candidate. No solver kernel, acceptance rule, tolerance, contact law, or 32 MiB readback limit changes.

The residual-stage and measurement appenders now require the selected `trialIndex`; other trial candidates do not contribute arrays. The exact present-array maxima for one selected capture are 71 (`pre`), 109 (`post`), and 136 (`both`), under the unchanged 192-entry metadata limit. Counts are: pre context 71; post context 80 when shared material/source state is included or 36 without it; five residual snapshots at five arrays each; and four measurement arrays. The manifest retains the actual array count and unchanged hard limits. The 32 MiB payload bound remains unchanged.

Both contexts now include `rigid_states_trial` and `rigid_candidate_trial`. The pre context occurs after `nm_fem_armijo_prepare_free_rigid_candidate` and `nm_project_primal_free_rigid_candidate`, so these arrays preserve the private free-body candidate before internal force/contact rebuilding. The post context includes the same arrays for direct comparison. They are required when the free-rigid residual path is enabled and otherwise remain optional.

The selector remains `NM_FEM_COUPLED_TRIAL_CAPTURE_SELECTOR=root:newton:trial:phase` and the output stem remains `NM_FEM_COUPLED_TRIAL_CAPTURE_PATH`. This is a diagnostic replay input, not a checkpoint or physical qualification. No GPU run is included.

`check_trial_capture_overlay.py` checks selected-trial gating, derived array accounting, rigid-array placement, and the unchanged byte limit against the source-bound overlay. The Objective-C++ syntax check uses the frozen-092 runtime flags.

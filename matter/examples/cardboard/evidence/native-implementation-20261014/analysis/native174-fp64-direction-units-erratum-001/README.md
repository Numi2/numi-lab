# Direction-unit erratum for DD derivative qualification

This erratum supersedes only the directional-input interpretation in the unfinished run-006/run-007 sweep and run-011 profile. The source capture generator at `bilinear-capture-layer-audit-source166-003/candidate/matter/tools/cardboard_bilinear_capture_layers.py` reads `F_direction_local_per_s` at line 171, computes `analytic = C @ direction` at line 206, then stores that action as `reference_directional_action` at line 227. The qualifier incorrectly treated the stored stress-rate action (Pa/s) as a deformation-rate input.

Run-006 produced no scientific sweep output and was interrupted. Run-007 was interrupted before its first directional checkpoint. Its completed center/side projections, all nine center tangent columns, and 17 MP-root tangent comparisons are independent of that unused direction array and remain separate valid CPU reference evidence. The 32-row run-011 profile used the wrong input; it returned a projection but failed branch classification on nonfinite states, and its timing is not representative of the intended perturbation.

The profile shows the exact DD implementation work on that invalid input; it does not justify changing or optimizing the equations. For subsequent endpoint analysis, use the exact source003 realized F endpoints, or source-bound `F_direction_local_per_s`, and keep Fdot/rate deltas separate because this constitutive law is rate independent. No tolerance has changed.

All failed and interrupted artifacts remain in place and are hash-bound in `report.json`. The three focused mixed-branch/derivative tests passed; they validate the reference functions, not this invalid directional experiment.

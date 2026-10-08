# Accepted support-slip and momentum observer

The optional full q-integration audit now records five whole-body linear-momentum stages and four contact-point tangential velocities per accepted step. Contact velocities use the actual pre-step GPU point Jacobian, including the weighted skin support callback. Source contact body IDs identify the support region; they do not identify the dynamically selected skin witness.

These are read-only diagnostics. CPU kinematics interprets retained GPU states only when the full audit is enabled; it does not calculate or feed the physical update. The normal execution path is unchanged. Full auditing has observation overhead and is not a performance mode.

## Native verification

Run 876 completed 10,000 accepted 2 ms steps on the SSH Mac mini, with the owner receipt reporting exit 0 and no changed source files. The native terminal reports 20.000000950 simulated seconds. Build and configuration logs, the exact launch driver, source/binary pin, invocation and asset hashes are retained here.

Against both completed native runs 838 and 878:

- Coupled physiology (60 columns, 10,000 rows), surface audit and COM momentum CSVs are byte-identical.
- The original 19 q-integration columns match over all 10,000 rows.
- The original 21 support-impulse columns match over all 320,000 rows.
- All 15 appended momentum fields and 12 appended contact-velocity fields are finite. Accepted momentum agrees with the prior COM observer within 1.8e-15 kg m/s.

The first build failure and earlier diagnostic attempts remain under `/Users/n/numi-human-resting-evidence-20261005/native-support-drift-diagnostic-876`. `analysis-inputs.json` binds the unchanged source files, binary and raw outputs. Its original analysis remains retained at the path recorded there.

## Interpretation limits

The velocities are the pre-step contact Jacobian applied to each stage's velocity, not accepted-endpoint skin slip. Stage momentum comparisons are partial checks; they do not establish total force or energy closure. Small post-settling drift remains under investigation. This fixture uses the retained 004 skin candidate, whose shared-atlas binding defect is independently known; successful observer parity does not qualify that anatomy.

The 32-versus-64 iteration comparison mentioned in the retained analysis is a separate predeclared numerical diagnostic. It is not a registered Numi Science study. Study 867 is the separately registered paired respiratory intervention.

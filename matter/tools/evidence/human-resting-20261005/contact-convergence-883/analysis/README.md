# Predeclared contact-convergence diagnostic: 32 vs 64 iterations

This report compares completed 883 (32 contact iterations) with completed 876 (64 iterations), using the exact metrics and code retained beside this file. The timestamped prelaunch declaration is SHA-256 `506d66604bf273db258265c96aca556458b8c2d1ded56a5839dee5e98034d045`. It predicted that if iteration-limited creep drives motion, reducing 64 to 32 would increase accepted pre-step-J tangent slip, post-init COM displacement/trend, and constrained momentum-minus-contact-impulse residual.

The 883 candidate is a timestamped predeclared 32-iteration diagnostic against the existing 876 64-iteration observer baseline. Neither run was registered through `numi science`; 867 is the registered science study. This diagnostic pair is not clinical or physiological qualification.

Both arms have runner exit 0, 10,000 accepted steps at dt 0.002 s, root assistance false, identical source/binary and asset hashes, and the same launch arguments except the declared iteration count and per-arm output/movie destinations. The native terminal records confirm the accepted counts; the owner wrapper receipts report no source changes.

## Results

- Post-init COM displacement (samples strictly inside 10–20 s): 64=0.122076 mm, 32=0.360930 mm; ratio 32/64=2.9566. Componentwise trends (mm/min), 64=[-0.23556397866894466, 0.7023193223358709, 0.3790542407582597], 32=[-1.8880068806656867, -1.1960634749824026, 0.034957848096885465].
- Accepted tangential speed on active contacts using pre-step weighted-contact J, p95: 64=2.5413334e-05 m/s, 32=0.00024732341 m/s; ratio=9.7320. This is not endpoint selected-skin slip.
- Constrained-stage ΔP−summed support impulse residual norm, p95: 64=6.8090966e-06 N·s, 32=7.4180547e-06 N·s; ratio=1.0894.
- Gravity-stage residual p95: 64=7.0660843e-06 N·s, 32=7.2934577e-06 N·s. q-advance momentum mean/max: 64=7.5542618e-09/1.3872193e-07, 32=5.1180899e-08/1.3605919e-07 kg·m/s. Final-acceptance maxima are near this scale and detailed in `analysis.json`.
- The selected PaCO2/PaO2/airflow/lung-volume/tidal/breath/ejection/gas-budget/blood-continuity/respiratory-volume fields are identical row-by-row. The full coupled CSV is not byte-identical because only mechanical/contact diagnostics differ: min_contact_gap_m, peak_penetration_m, normal_impulse_ns, pre_projection_contact_residual_m_s, pre_projection_limit_residual_generalized_s, pre_projection_equality_residual_generalized_s, post_projection_contact_residual_m_s, post_projection_limit_residual_generalized_s, post_projection_equality_residual_generalized_s, equality_position_projection_max_generalized, equality_velocity_projection_max_generalized_s, contact_sweeps. Per-field details are retained in `analysis.json`; no post-hoc pass boundary is used.

## Interpretation

The predeclared direction held for all three target quantities at 32 iterations: COM displacement was 2.96× the 64-iteration value, accepted pre-step-J tangent-speed p95 was 9.73×, and constrained-stage residual p95 was 1.09×. This supports a convergence-sensitive contribution to the measured motion in this 20 s diagnostic. The smaller residual ratio does not by itself explain the larger slip/drift ratio. Gravity residuals and final-acceptance momentum changes remained on the same small scale; q-advance mean increased while its maximum stayed similar. The physiology trace fields examined were unchanged. These diagnostics still do not identify a unique cause: slip uses the pre-step Jacobian, force/energy closure is partial, and articulated settling can move COM without equal root displacement.

883 is a timestamped predeclared diagnostic, not a `numi science` registered study. 882 was a separate timestamped predeclared 128-iteration diagnostic rejected before native launch; its declaration SHA-256 is `f29c606f9c65553db1c4d3b2e64c0d22d078c67e701489f1d522739843f96fc7`, and its failure receipt is retained without outcome data. The registered science result is 867.

## Reproduction

Run the exact retained script on the Mac mini:

`python3 /Users/n/numi-human-resting-evidence-20261005/native-contact-convergence-883/comparison-876-883/compare_diagnostics.py`

The script reads only 876/883 outputs plus the retained timestamped 883 and 882 declarations/failure receipt, then writes the JSON, Markdown, and manifest in this new comparison directory. It does not alter source, binary, or native run outputs.

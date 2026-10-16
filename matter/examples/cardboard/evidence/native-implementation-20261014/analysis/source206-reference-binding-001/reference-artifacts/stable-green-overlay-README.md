# Stable Green-strain reference overlay (019)

This isolated overlay changes only the cardboard_bilinear_fp64_reference.strain function:

- old: (F.T @ F - I) / 2
- candidate: with D = F - I, (D + D.T + D.T @ D) / 2

The expressions are algebraically identical. The candidate avoids subtracting a near-unit matrix after forming F.T @ F; it does not change material parameters, state equations, branch conditions, or projection/derivative gates.

The exact base reference in frozen source205 has SHA-256 a910339573822bf32c7ea895c295a0a51b8b5a99d901b1e3ab1c788c094f6995. The candidate has SHA-256 8064ba89f0c4b10413937cbf2a5886a9f7de0e4b02358988294620ebae51788f. reference.patch is the one-function patch against matter/tools/cardboard_bilinear_fp64_reference.py.

## Focused regression

candidate/test_cardboard_stable_green_strain_reference.py replays tet222, local deformation-gradient basis column6, from exact captured FP64 F, accepted history, and liner parameters. For each of powers33 and34 it projects both endpoints from the same accepted history, checks the unchanged 2e-12 scaled-residual and 2e-11 dynamic-correction limits, requires the center branch at both endpoints, and checks the centered FD against the strict implicit tangent at 1e-6 relative L2.

The stable source passes both powers: 3.8658183e-7 and 8.9219471e-8. The old source fails the unchanged gate at both: 4.9339229e-6 and 4.9115683e-6. Stable candidate run: 2 tests pass. Old-source negative control: 1 expected test failure at the FD gate; preserved in legacy-control.stdout.txt.

Run in the source tree with its matter/tools directory on PYTHONPATH:
python3 -B -m unittest -v test_cardboard_stable_green_strain_reference

## Broader CPU evidence

The stable implementation was evaluated at all 1,728 paper centers and nine basis directions over the prior powers10–24 sweep: 15,529/15,552 pairs had adjacent strict, same-branch FD matches; the remaining 23 are retained and rerun over powers24–40. The extended run resolves all 23, including tet222 columns2,5,6,7,8; no gate changed. Full per-endpoint data remains at analysis/native531-stable-green-strain-reference-017-001/extended23-v2/report.json.

Arithmetic invariants (100-digit Decimal strain check, 64 random objectivity/rigid-rotation/energy checks, and a recovered-frame audit) are recorded at analysis/native531-stable-green-strain-invariants-018-003/report.json. The frame archive is a separately recovered, source-bound reconstruction—not direct frame-row output in capture531. Its z=0 frame probe reports up to107.01Pa stress from non-orthogonality; this is not a coupon rest-state measurement and remains a separate possible native rounding issue.

The 018-001/002 invariant attempts and the first extended23 JSON-serialization attempt are retained. 018-001/002 used an invalid scalar-indexed strain check; those invariant summaries are not evidence. The first extended sweep failed to serialize a NumPy branch vector; the v2 sweep is the complete run.

## Limits

This is CPU-only reference evidence. The full 9-basis self-consistency sweep plus the 23-pair fine sweep closes the stated FP64 reference finite-difference checks, but does not measure native FP32 absolute noise, qualify native stress/tangent action, validate a full solver trajectory, or calibrate material inputs.

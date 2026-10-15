# Newton 12 candidate geometry/material audit

This is a diagnostic-only, source-bound analysis of native capture 291. The full native run failed at the subsequent Newton solve; this report does not qualify convergence and does not identify a production defect.

The capture binds to source physics fingerprint `915f50b10edc2e46` and device-program fingerprint `d61f84737f5c7664`. It records the accepted center candidate and Newton correction, but not post-trial per-term residuals or rebuilt contact state. The Python replay uses the captured FP32 words and explores the source candidate map with fused FP32 operations; it is a representability study, not a native trial.

## Findings

At root 1201, Newton iteration 12 accepted alpha `2^-22` (`2.38418579e-7`): residual went from `3.43603097e-6` to `3.43603074e-6`, a `2.3e-13` decrease. The next iteration failed. In a fused FP32 emulation of `v += alpha*correction; u = accepted_u + dt*v; x = rest+u`, alpha zero reproduces all captured candidate `v/u/x` words. At the selected alpha, 96/108 velocity words, 53/108 displacement words, and 21/108 position words change; max position shift is `1.7053e-13 m`.

All 36 nodes carry reference displacement mode 1. The tet uses reference-displacement gradients, and the material has `NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC` (flag 16), with no material history state. Source [fem.metalinc](../../snapshots/source-strict-coupled-capture-json-088/matter/src/metal/fem.metalinc#L181) builds deformation from the separate displacement `u`, then uses the compensated reference-deformation helper when the material/tet flags permit it. The displacement candidate itself is still updated in FP32. Reconstructing `F` from the captured FP32 `u` words and inverse-rest matrices gives nonzero changes in 47/48 tets, max Frobenius `|ΔF|=5.57e-10`; this is not a native stress/residual delta.

For 99 direct self-contact records (one mapped alias excluded), a CPU FP64 closest-triangle replay over the predicted endpoint positions sees no feature-branch changes and max gap change `1.21e-16 m`. The swept-start reconstruction sees max `7.04e-18 m`. But captured records match endpoint geometry to about `5.22e-12 m`, while they differ from reconstructed swept-start gaps by about `7.77e-8 m`; all 99 record TOIs are zero. Source [contact.metalinc](../../snapshots/source-strict-coupled-capture-json-088/matter/src/metal/contact.metalinc#L2799) describes `finish=positionAndMass` and `start=finish-dt*velocity`. That inconsistency is unresolved and must be tied to the precise contact producer phase/generation before the records are used to attribute trial residuals.

## Causal limit and next measurement

Both the material input and the contact endpoint are discretized, but the captured evidence cannot assign the tiny accepted residual decrease or the later plateau to either one. It also lacks separated inertial, material, cohesive, rigid-contact, and self-contact residuals after the selected trial. No production change follows from this audit. The next useful capture is the selected alpha before and after private-trial contact rebuild, with the exact candidate `u/x/v`, projected state/stress/element force, rigid pose/contact samples, complete self-contact records plus generation tags, and residual components under unchanged accepted history.

`alpha12_measure2.py` reproduces the FP32 candidate-map emulation and CPU geometry/F summaries. Its source and capture hashes are listed in `report.json`.

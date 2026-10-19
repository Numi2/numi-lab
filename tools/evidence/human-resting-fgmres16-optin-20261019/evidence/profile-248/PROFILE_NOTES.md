# Profile 248 notes

This isolated diagnostic run completed one 0.25 ms joint vessel, skin-fat, and prostate step on the accepted profile-248 binary. Its native log reports status 0, 256 completed microticks, 1,922 unknowns, Newton budget 12, FGMRES budget 64, restart 16 (four cycles), and a maximum of 16 FGMRES iterations in the last microtick. It logged 12 Newton entry records for each of 256 microticks. The status structure exposes the FGMRES maximum for the last microtick; this is not a per-Newton count.

The one-step `resting-coupled.csv` SHA-256 is `7e5d177ae7cbfe5c7c470bcfa8814e394f469678d1cd62289bedbb6a94b0a800`, matching the prior profile-006 joint run. This establishes output parity for the single accepted step, not a longer-horizon or performance qualification. Visual verification was disabled.

The timing probe successfully sampled 32 FGMRES encoders (sum 694.701 ms); another 3,040 sample-buffer allocations failed with `Cannot allocate sample buffer`. Timing coverage is incomplete and does not localize the remaining solve time. Do not infer that the sampled subtotal explains total GPU time or extrapolate performance from it.

## Skin contact label correction

The historical line `contact_surface=skin_outer_only` in `native-run-002/skin-fat-only/native.log` and `native-run-004/skinonly/native.log` is an obsolete label. The actual registered conforming FEM source in `profile-248/app-profile/source/apps/NumiHumanCoarseTissuePreparation.hpp` classifies contact on the exposed outer skin, the exposed deep-fat boundary, and only paired skin-fat interface vertices on open patch sidewalls. The shared internal interface remains ineligible for external contact, so there is no duplicate coincident interface contact surface. The source predicate is authoritative; the old log field must not be read as describing outer-skin-only contact.

## Profile 244 plan correction

`profile-244-stale-fields-correction.json` records the stale top-level app object and timing-header references in the frozen profile-244 run plan alongside the actual paths and hashes from its nested diagnostic build receipt. The original plan and invocation remain unchanged.

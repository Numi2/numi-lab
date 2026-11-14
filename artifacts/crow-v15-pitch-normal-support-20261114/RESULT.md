# Pitch correction during vertical-only contact: failed

The lane 0 control reproduced the previous control's logged state, command and contact events over all 450 intervals. Lane 1's first logged leg-target and state difference occurred at zero-based interval 305, when the alternate support condition changed the active correction. All 900 states, 28,800 contact substeps and 28,800 toe-velocity substeps were present without observer errors; no non-foot impulse was recorded before the falls.

Both lanes achieved the same longest bilateral grip streak of 11 intervals after return, far below the 100-interval terminal requirement. The control fell at one-based interval 370 and the candidate at 371. Both reached the 12 rad/s toe-speed limit before falling; the candidate had 111 threshold substeps versus 13 for control. Both had a longest qualified release interval of 43. The candidate is rejected; its one-interval fall delay does not indicate a qualified grip.

This deterministic constructed release/return comparison is not walking or landing qualification. Full raw traces are preserved in Git.

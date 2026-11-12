# Landing pitch and automatic release timing

Two previously retained full, teacher-off runs on the used development seed were reanalyzed. The first used the rejected early-wing candidate; the second used the downstroke candidate. Sequence values are decoded from the recorded controller status using the corresponding experimental feedback source. This is an observational timing audit, not an intervention.

All 11 physically failed episodes across the two runs crossed 0.1 rad absolute pitch before an intentional toe-opening request (sequence 150–229). Six of the 11 recorded failures never reached a toe-opening request before termination. In the five with an opening request, absolute pitch was already approximately 0.282–0.728 rad when the request first appeared. Counting the other three nonfailed, timed-out downstroke episodes separately, one also requested release after it had already pitched approximately 0.359 rad. Pitch began before the weight-shift portion in most, but not all, cases.

The controller runs a one-shot release after acquiring grip; its request is not proof of actual toe opening, slipping, or the cause of a fall. The paired endpoint pose and phase measurements may be offset within one control interval. This evidence rules out automatic toe opening as the necessary initiating event for these observed failures; it does not prove release is harmless. Stabilization and a controlled, separately measured release/regrip remain required. These are development layouts and seeds, not held-out validation.

Reproduce with `python3 artifacts/crow-v15-release-pitch-timing-audit-20261112/analyze.py`. The newer trace is read directly from hash-verified concatenated compressed parts without creating a second copy.

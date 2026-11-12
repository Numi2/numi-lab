# Stronger bank feedback: rejected

All 48 pre-intervention sampled states, velocities and actions exactly matched the retained same-lane baseline. Contact coverage was complete and no contact overflow or solver substep error was logged.

For bank limits 0.12, 0.18 and 0.24 rad, the final heading errors remained 1.220/1.312, 1.279/1.329 and 1.315/1.390 rad in the positive/negative direction. The 0.24-rad positive-direction lane physically failed at interval 1302 after two recorded ground-contact substeps; the first-failure collider pair was not identified by this observer, and the reported final 1.315-rad error is its last available sample at interval 1300. The strongest negative and the 0.18-rad positive lanes also exceeded the predeclared sampled height range. No lane pair achieved the predeclared 15% reduction in both directions.

Reject all three settings. Height and roll are sparse samples and do not establish continuous limits. No statement about the exact contact that caused failure, biological aerodynamics, flight qualification, autonomous route completion or landing follows from this screen. The controller used direct body moments before 9 seconds.

## Contact timing correction

The two ground-contact substeps counted in the first episode occurred at intervals 1215 and 1258, before the terminal interval 1302; the original observer did not retain their body pairs. Contacts at trace interval 1303 lie beyond the recorded first-episode boundary and are not attributed to the failed episode. A follow-up observer must include the earlier interval and independently verify replay parity. These observations do not establish whether either contact caused the later termination.

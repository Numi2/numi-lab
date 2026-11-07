# Corrected delayed-action depth result

The preregistered native probe completed on Apple M6 and matched all 3,072
actor observations at Float32 publication precision: 768 each for delay
components 0, 1, 2 and 3, across eight environments and eight steps. The
maximum printed difference was zero. Every step reported successful physics
and no task termination. One environment was reset at step 5 while the
other seven continued independently.

For the reset environment, depths 0, 1 and 2 each produced zero until the
first appropriate post-reset action became available, and then produced that
action. At depth 3 the eight-step run ends before the first nonzero post-reset
action would appear at step 9. The unaffected environments did exercise
nonzero depth-3 history; the reset environment's post-reset nonzero depth-3
publication remains unqualified.

The customized task's world, task, observation and action fingerprints are
retained for every depth in stderr; this task has a different history
configuration from the retained PolicyPack. The command, 3,072 sample rows,
exit, exact binary identities and UTC timestamps are retained here. The GPU
run lasted approximately 0.520 seconds, within the 120-second limit.

These results concern actor-frame history on this historical-owner fixture.
They do not qualify actor-current, critic, rejected-physics sensor
transactions, a full current-checkout native build, or policy performance.

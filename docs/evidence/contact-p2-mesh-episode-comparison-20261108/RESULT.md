# Fixed-law contact: mesh comparison of sampled episodes

This post-hoc comparison reads the audited, temporally qualified fine n8 and
n16 P2 traces at their identical saved 0.1 ms times. Both first show nonzero
force at 4.6 ms and remain active at the final 10 ms sample. Those samples do
not establish exact onset, release or the absence of shorter episodes.

At the 4.8 ms sample, the n8 force is 0.000989 N and the n16 force is
0.106497 N, the largest sampled absolute difference. The source columns and
timestamps match; 18-, 48- and 64-point force evaluations agree closely at
that sample within each run. Later 1 ms bins also differ, with the largest
bin discrepancy in the final bin: 4.9085% of the n16 peak bin. Integrated
impulse differs by 0.66949%. The unchanged 0.25% spatial force and impulse
gates remain failed. No time alignment or smoothing was applied.

These two meshes show different sampled force histories; their disagreement
alone does not identify the cause, establish a convergence rate or qualify
force response. Stored contact potential, contact work, gap and sampled
volume Jacobian are retained as context. The measurements use a fixed
regularized CPU continuum model, not physical material calibration or native
production replay. No new simulation or performance measurement was run.

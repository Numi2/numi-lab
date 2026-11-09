# Reacting-slider coupled transaction: retained failure diagnosis

This is a diagnostic result, not qualification. The normal build still rejects
the slider option; both older CCD and augmented-Lagrangian options remain off.
The diagnostic-only build has an additional fingerprint and explicit opt-in.

A true Matter / MetalWorld one-axis dynamic body accepted its first initially
active contact step. In separate forced-late-failure tests before contact and
after a nonzero contact, the environment reported failure category 11,
restored Matter physical state and rigid q/v byte-for-byte, and reproduced an
uninterrupted retry exactly. These short checks do not qualify a 10 ms rollout.

The four preregistered attempts all rejected with Matter status 10 before the
requested duration. These are the complete accepted-prefix counts, not full
rollout results:

| Initial gap | Timestep | Accepted | Accepted duration |
|---|---:|---:|---:|
| 200 µm | 15.625 µs | 576 / 640 | 9.000 ms |
| 95 µm | 15.625 µs | 503 / 640 | 7.859 ms |
| 200 µm | 7.8125 µs | 676 / 1280 | 5.281 ms |
| 95 µm | 7.8125 µs | 748 / 1280 | 5.844 ms |

For the first row, an exact physical checkpoint of the accepted state before
rejection was saved, round-trip checked, restored, and reproduced the same
status and numeric certificate. `nm_fem_midpoint_certify_slider_postcommit`
reports the position error 1.0362491247029038e-11 m against limit
9.01076002007839e-12 m; momentum error 4.647263043139338e-12 N s is below
its 3.512695342933192e-11 N s certificate limit. Therefore the generic status
10 is, for this specific failure, a post-commit slider position-certificate
failure, not evidence that Newton itself stalled. No conclusion is drawn about
what caused the other three failures without equivalent captures.

The separate preregistered per-step contact impulse-balance gate fails in the
accepted prefixes even though the independently captured contact body wrench
and FEM contact impulse balance. `slider-float-resolution-001.json` uses the
stored initial velocity of each step and measured wrench to calculate the
nearest possible binary32 endpoint velocity. It shows that 442/576, 379/503,
557/676 and 586/748 accepted steps respectively cannot satisfy the original
per-step momentum gate even with that nearest velocity. This is an explicit
representation limit for the measured trajectory, not permission to loosen
the gate or a diagnosis of every numerical error.

`slider-primary-failure-checkpoint-001` was stopped by a storage guard before
failure and has no checkpoint; it is retained as incomplete evidence. The
successful failure capture is `slider-primary-failure-checkpoint-002` and
`slider-primary-failure-replay-001`. All execution manifests bind the tested
binary and shader hashes. The short `slider-checkpoint-save-001` and
`slider-checkpoint-restore-001` outputs match exactly.

Midpoint clearance and contact potential/work are recorded per accepted step.
No full force, peak, cumulative-impulse, work, spatial accuracy or performance
qualification follows from these incomplete trajectories. A future repair must
represent and restore the rigid endpoint and contact geometry accurately
without weakening the equation or acceptance gates, then rerun the registered
full tests. The independent n=128 spatial and fine backward-Euler runs are
separate and remain in progress at the time this note was written.
EOF'
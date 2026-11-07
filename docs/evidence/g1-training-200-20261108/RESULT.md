# Three-seed outcome

All three fresh 200-update trials completed with no reported failed physics
steps, and all three candidates received positive matched selection scores.
The preregistered exploratory prediction is supported on those conditions.
Nevertheless, every candidate failed the initial skill screen: zero of eight
environments completed a clean horizon for each, and no candidate achieved
one metre of final progress in every environment. The candidate height-or-tilt
termination event counts were 85, 81 and 59; each matched incumbent had 93.
Those are repeated events, not percentages. The relative selector advanced
all three policies; none is qualified for deployment by these results.

The three training seeds share one selector seed, which was used to select
policies and is not an untouched evaluation set. These results do not establish
held-out robustness, physical accuracy or time to skill. Trials ran with
exclusive task-owned GPU scheduling, while other CPU activity and macOS
background activity were not controlled. Wall times were 71.45, 71.81 and
74.41 seconds including training and selection; these are not a benchmark
against an accuracy-matched system.

Complete training directories, policy and optimizer files, raw state traces,
logs, commands, hashes and host snapshots are retained in the measured-host
archive identified by archive.sha256. The compact repository records are not
the full archive or a standalone runtime environment.

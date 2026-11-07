# Release accepted-step batching exploration

A preregistered M6 Release-build study compared one and 32 environments in the
same existing Franka host-sinusoidal task, with 100 control steps after 20 warmup
steps. Five fixed seeds were paired with alternating arm order. All ten trials
passed the declared accounting and nonfinite-output gates. The sealed analysis
and revision verified successfully.

Median accepted environment steps per measured wall second were 328.74 for one
environment and 8,510.07 for 32. The corresponding medians of aggregate
accepted simulated seconds per wall second were 5.48 and 141.83. Those totals
sum time across environments; they do not describe a single trajectory.
Median measured wall times were 0.3042 and 0.3760 seconds and median reported
GPU durations were 291.34 and 360.62 ms. The runtime reported 1,088,458 and
31,989,754 retained buffer bytes, not process-wide peak memory.

The timing interval excludes host action generation; initialization and warmup
are reported separately. No rendering, learning, continuum, force or motion
accuracy requirement, comparison engine, or time-to-skill assessment was
included. Seeds describe repeated conditions on one machine, not independent
hardware replication. There is no optimization speedup or physically qualified
performance claim. The broad preregistered prediction was supported only for
this workload and observed conditions.

Other task-owned CPU builds and GPU runs were paused during these trials.
Host process snapshots immediately before and after are retained with the
results. macOS background services were active, with Spotlight metadata
processes visible after the run. These snapshots do not establish exclusive
system GPU ownership, continuous host idleness, or thermal control. The earlier
unoptimized study had separate documented background Git indexing and must not
be relabelled as this run.

The measured source artifact is bound by the registration (benchmark.cpp
SHA-256 8f991018465c5ce9f119c79569b62f27647b4063a53e0e718af554dabb3da939);
the measured executable SHA-256 is
c09db54d1abf4459b4c5ff51336ae9255d61a92867451b97d7860b2fa360801b.
The full executable, library, shader, input, registration, trial and revision
snapshots were archived and integrity-verified at
/Users/numi/numi-lab-benchmark-release-science-v2-20261107/archive-final.
The earlier analysis-only archive is retained separately. Compact trial outputs,
receipts, host snapshots, analysis and revision are in
[release results](evidence/benchmark-release-results-20261107/), and the
preregistered plan and source artifacts are in
[release plan](evidence/benchmark-release-plan-v2-20261107/). The compact
repository copy is not the complete portable notebook.

A useful next comparison requires independently specified physical force and
motion references and fixed error gates, then measurement of accepted simulated
time including rejected work. GPU occupancy and bandwidth remain unmeasured:
the host has Metal tools but not xctrace.

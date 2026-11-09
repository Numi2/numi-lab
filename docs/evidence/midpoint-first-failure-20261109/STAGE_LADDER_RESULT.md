# Eight-division midpoint ladder

With the same physical law, temporally qualified n=8 reference and fixed
accuracy and replay gates, 7.8125 microseconds is the largest passing step in
the registered ladder. Its 1 ms-binned force error was 0.07169%, sampled peak
force error 0.14450%, total impulse error 0.007717%, and maximum cumulative
impulse error 0.04027%. All 1,280 physical steps were accepted and replay and
rollback checks passed. The peak is sampled every 7.8125 microseconds; the
binned result is not a peak measurement.

At 15.625 microseconds the rollout completed but binned force error was
0.29033%, above the 0.25% limit, while the peak error was 0.84330%. At 31.25
and 62.5 microseconds force accuracy also failed; the 125 microsecond attempt
rejected after 55 accepted steps with restoration. The complete traces,
analyses, input fingerprints and failures are retained. These are instrumented
runs, not performance measurements. The n=8 mesh is not spatially qualified;
there is no matched-accuracy backward Euler timing or demonstrated speedup.

# Surface-sort candidate 2

The native run exited 0 after two requested 125 microsecond steps; its first two data rows are byte-identical to baseline 036. The candidate production sorter fixture is byte-identical to the retained legacy fixture, and the 112-case geometry checker passed.

Across 100 root-1 samples, surface-sort median fell from 10.0364995 ms to 7.282354 ms and summed sampled cost fell from 1.003526872 s to 0.728182577 s (27.44% reduction in this kernel). Compact medians were 3.004625 ms baseline and 3.0044375 ms candidate; incidence-scan medians were 0.631500 ms and 0.631542 ms. This is a kernel-specific measurement, not a whole-body speedup or sustained-runtime qualification.

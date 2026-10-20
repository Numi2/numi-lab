# Native surface-sort baseline trace 036

The run exited 1 after 510.234285542 s. It accepted 24 rows through 3 ms of a requested 3.125 ms horizon (25 x 125 microseconds), then reported the same step-24 matter failure as 035 (code 6, object 19, index 2624, one microstep, diagnostics [3.296739237e-08, -2.180359893e-09, 0.25, 0]). Its complete 24-row CSV is byte-identical to 035.

The retained 100-sample root-1 contact profile measures the stable surface sorter at median 10.0364995 ms and sum 1.003526872 s. Compact and incidence-scan medians were 3.004625 ms and 0.631500 ms. Microtick-1 trace capture missed all 24 requested captures (active_counts=0); this package does not analyze the failure mechanism. These are sampled kernel costs, not a whole-body speedup. Exact provenance, reverified input hashes, CSV, surface audit, and selected timing samples are retained. The large output pack and diagnostic JSON files are hash-identified but not copied.

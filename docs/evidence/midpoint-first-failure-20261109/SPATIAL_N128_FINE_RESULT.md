# 128-division time refinement and spatial comparison

The frozen independent FP64 reference completed the full 10 ms n=128 fixture
at h=0.25 µs. Its execution record binds the executable hash and limits; the
separate area validator checked all node weights, total area, clearance,
force samples and impulse accounting. Re-running
`spatial_n128_retry2_analysis.py` reproduces the retained analysis exactly.

Comparing h=0.5 µs to h=0.25 µs at n=128 passes all time gates. The largest
relative errors are 7.70e-10 for 1 ms-binned force, 2.69e-9 for sampled peak,
2.54e-10 for total impulse, and 3.15e-10 for cumulative impulse.

The historical n=64 versus n=128 comparison fails spatial qualification:
1 ms-binned force differs by 1.7274%, sampled peak by 20.6927%, total impulse
by 0.1618%, and cumulative impulse by 0.5037%. Both historical references
pass their separate time gates. The historical n=64 execution records do not
bind the executable hash directly; the separately queued hash-bound repeat is
needed to remove that provenance limitation. The 1 ms-binned force measure
and sampled peak are distinct quantities. The result is a reference-fixture
comparison, not a spatially converged native Matter result or calibrated
physical material.

`n128-force-curves-001.json` retains the curves examination separately from the
refinement gates. At common 2 µs sample times the largest instantaneous force
difference is 66.51% of the n=128 step-sampled peak. The largest binned-force
difference occurs in the 4–5 ms bin (0.02386 N at n=64, 0.02542 N at n=128).
The largest cumulative-impulse difference occurs at 5.078 ms. The peak times
in that record use the 2 µs trace and are not claimed to be the times of the
finer 0.25 µs step-sampled peaks. These are comparisons of the retained
numerical trajectories; they do not by themselves identify a physical cause.

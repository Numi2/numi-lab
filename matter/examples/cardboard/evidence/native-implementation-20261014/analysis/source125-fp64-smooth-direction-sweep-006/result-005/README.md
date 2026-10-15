# Rowwise epsilon interval addendum for source125 result-004

This postprocessor binds and re-reads the immutable `result-004` report, manifest, and compressed rowwise NPZ. It reports every maximal contiguous interval on the fixed epsilon grid where both endpoint projections pass, the recorded branch predicates match center, and the centered local relative error is at most 1e-6.

The descriptive representative is the midpoint of the longest passing interval, with ties resolved to the earliest start (largest epsilon); no point is selected by minimizing error. All intervals, including disjoint intervals and grid-edge intervals, remain in the JSON. A window touching power14 or power30 is bounded by an untested edge.

Rows with no adjacent nonincreasing error pair are explicitly listed; passing under the threshold alone is not called a resolved truncation-to-roundoff plateau. The separate assembled 1% action result at 2^-22 is inherited from result-004 and remains a different scope. Native FP32 and physical qualification remain open.

Reproduce with:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 /Users/home/cardboard-implementation-20261014/analysis/source125-fp64-smooth-direction-sweep-006/intervals.py --output /Users/home/cardboard-implementation-20261014/analysis/source125-fp64-smooth-direction-sweep-006/result-005
```

# Independent review of the source101 Newton-23 selector-5 capture

This CPU-only review leaves the sealed `source101-regional361-actual-newton23-006` package and native runs unchanged. It addresses the earlier 001–005 reader failures by independently reconstructing selector-5's current graph layout from the exact compiled `.nmatterpack` FEM node mask and tetrahedron connectivity, then comparing all six generated maps against the native capture.

The review also checks same-generation center-state bytes, run 360/361 input and output parity, the captured cached-tangent `A P` action, pivoted `RAP`, and per-patch LU reconstruction. It records each norm denominator and explicitly has no numeric threshold gate. The native attempts remain rejected (exit 2); this does not qualify solver convergence, constitutive accuracy, or physical behavior.

The review script was executed once with this package directory empty. It refuses to overwrite `report.json` or `manifest.json`. To reproduce, copy `review.py` into a new numbered empty review directory, change its `OUT` constant to that directory, then run it in the retained Apple Silicon Python environment with NumPy and SciPy.

`report.json` contains results. `manifest.json` binds the script, report, exact source101 files, compiled package, capture payload/manifest, 360 center capture, audited sparse operator and the preserved failed 001–005 attempts by SHA-256.

This revision additionally binds each of the 37 ordinary outputs and the 789 staged input files, records the same-selector comparison scope, and verifies compiled-package/runtime fingerprints.

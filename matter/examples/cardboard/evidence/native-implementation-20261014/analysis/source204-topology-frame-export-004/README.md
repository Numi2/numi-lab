# Source204 diagnostic topology export

This isolated CPU-only overlay extends the cached-operator topology NPZ with material-frame rows, direct-capture provenance, active flags, and quaternion provenance. It fails closed on inconsistent frame-array extents and checks that directly captured rows remain exact float32 values.

The focused test set passed 29 tests. One unrelated source-bound-runner integration test was excluded because it collided with the active native worker lock in the retained full-suite attempt003; no test was altered. Rerun all 30 tests after the worker exits before any broader integration claim.

No physics, operator, tolerance, or native execution change is included.

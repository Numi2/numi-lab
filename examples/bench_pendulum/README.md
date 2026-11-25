# Bench Pendulum

This fixture exercises the same fixed-base URDF workflow available to
user-authored experiments in the Mac app. The comparison maps the URDF
mass, inertia, center of mass, joint axis, limits, effort, and damping onto
one MuJoCo hinge. The trace records post-integration angle at each 20 ms
control tick. Each engine receives the same normalized command held across
the same number of substeps. The arm clears the floor throughout this test.

Install the Python package and MuJoCo 4.0.1 in an Apple silicon environment,
then run the comparison against an extracted Numi Lab app.

```sh
python -m pip install -e ./python mujoco==4.0.1
python examples/bench_pendulum/compare.py --app "/path/to/Numi Lab.app" --output ./pendulum-results
```

The report includes exact native trace replay, angle error across three
substep settings, native retained buffer bytes, and throughput from one to
512 environments. Native collection time and end-to-end process wall time
are reported separately. MuJoCo uses the native CPU rollout and returns the
full physics state. This fixture covers free articulation. Compare contact,
complex robots, GPU learning workloads, and energy independently.

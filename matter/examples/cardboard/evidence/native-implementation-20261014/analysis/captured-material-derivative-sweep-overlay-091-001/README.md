# Controlled material derivative sweep

This checker-only overlay adds `--native --derivative-sweep` to the existing
source-bound material replay executable. It reads and hashes the unchanged
3,456-point capture, then explicitly authors controlled directions around
eight retained centers: the six prior rejected liner points and medium points
1634 and 1638. No capture header or source identity is fabricated.

Each policy dispatches 960 independent material evaluations: three original
center repeats, nine row-major basis actions, and positive/negative samples at
six perturbation sizes for each basis. Each evaluation starts from the same
captured accepted material history. It records actual high/low deformation,
rate, direction, parameter identity and all 65 output words. Deformation and
rate perturbations retain the FEM relationship `delta Fdot = delta F / dt`;
the actual rounded words remain available for a realized-direction audit.

Repeated centers observe same-batch nondeterminism only. They do not establish
an absolute arithmetic noise floor. Perturbed sides can cross constitutive
branches or fail projection; no such row is treated as a smooth derivative.
The instrument returns diagnostic completion, never a numerical qualification.
Independent branch-aware, componentwise and near-zero comparisons remain
required. This adds no solver, material, shader, tolerance or history changes.

The source091-based Objective-C++ syntax check passes. The existing CPU
admission path also constructs the sweep and checks its exact bounded count;
that runtime check requires a separately built executable. No native run is
included in this package's initial manifest.

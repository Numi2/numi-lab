# Free-BOX operator-sign and branch-gate checker overlay (070)

This checker-only revision is based on the sealed source-069 approach-fixture overlay. It changes no production runtime, shader, solver tolerance, contact law, or acceptance threshold.

## Why this revision exists

Production assembles the free-rigid residual with contact impulse added and inertial momentum subtracted. The FGMRES matrix action is the negative residual derivative, `A = -dR`. The earlier checker compared its production action to `+FD(R)`, causing a relative discrepancy near 2 despite matching magnitudes. This revision compares against `-FD(R)` and records the convention explicitly for both contact and inertia.

The fixture uses approaching contacts. Its differentiable barrier branch is the selected maximum of inertial and impact stiffness, not a non-closing-only state. The checker now classifies that max branch with a relative margin from the nonsmooth crossover and requires the baseline, plus, and minus samples to keep the same contact feature and same stiffness branch. Ties/crossover samples remain inconclusive; no threshold or physics gate was relaxed.

## Binding and validation

- Source snapshot: source-069, SHA-256 `8fb4d0ed380412c1d46a3ebbb2b3817cbbfe1ce39a124796a41f88c483086d61`.
- Base checker overlay: `free-box-angular-sample-approach-fixture-069-overlay`, checker SHA-256 `4e0d3f1f59ec844fea14b89ec725d67208d07ab80293f460443906b1a94aa230`.
- New checker SHA-256: see `source-overlay.json`.
- Objective-C++ syntax check against frozen source-069 headers passed with `-Wall -Wextra -Wpedantic -Werror`.
- No Metal rebuild or GPU execution was performed for this overlay. Native case 219 remains bound to the base checker hash and does not validate this correction. A fresh source-bound native run is required.

The `checker-residual-sign.patch` is the exact delta from the base checker. Previously sealed source-069 overlays and native-219 inputs are left unchanged.

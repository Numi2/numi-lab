# Metal pinned-rod reaction witness

The [committed Metal witness](measurements/metal_rod_pinned_probe.mm), run on ssh numi, recreated the same effective two-node axial fixture used in the failed CPU experiment, with two distinct hard endpoints, zero gravity and damping, 1 ms timestep and the 128-iteration limit. The Metal owner reported host success, GPU status 0 and published the result, but the right support reaction was `0.00017894202028401196 N` against the independently calculated end-of-step elastic force `0.089379112368462382 N`: 99.799794% relative error. The 1% force gate failed.

The prestrained state is explicitly constructed from a restricted affine fit to one 8-0 Nylon specimen; it is not an earned surgical state or a measured stress-free length. This witness establishes an inaccurate published force for this exact two-endpoint fixture, not the error of the complete surgical trajectory. It does not measure contact work or replay.

The compiled program, owner dylib, Metal library, and source hashes plus exact result are [retained](measurements/evidence/metal-pinned-20261108). The shader records projection corrections but resets that maximum before the convergence test, which then examines end-of-sweep changes; cancellation is a plausible shared mechanism with the CPU diagnosis. A separate forced-sweep CPU diagnostic supported that mechanism there, but no forced-sweep Metal diagnostic has been run. The CPU-only exact pinned solution in `codex/rod-pinned-direct-20261108` is not a Metal fix.

The required next step is an independently certified Metal force/work and restoration repair that also supports ordinary rod motion and coupled surgical interactions. Do not infer surgical force accuracy from accepted-step or grasp counts.

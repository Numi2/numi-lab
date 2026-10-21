# Build178: anatomically restricted optic experiment

SSH Numi, Metal API validation enabled, no visual inspection. Compiler and runtime keep every skin row in the complete row map with exactly zero reduced support; fine skin coordinates remain identity coordinates. The CPU basis and source-label checks pass, including B/B-transpose virtual-work and projected mass identities, source hashes/layout separation and fail-closed ownership.

The optic-only full-coordinate preconditioner passes native process acceptance but **fails the unchanged 1% endpoint comparison**: anchor reaction 5.5892%, acceleration 1.8357%. Body cost is 7.025682209 to 7.078335583 seconds. Used columns increase 1587 to 1802; active Newton begins increase 34 to 70. The last Newton norm remains 2.50034554e-5 against the reported 1e-5 target. This is not a passing optimization and is not promoted. Removing the skin coarse modes degrades this system; the next full-model conditioning candidate retains them and changes only optic local patch coverage.

The optic-only hybrid-coordinate experiment also completes a native probe, but its last Newton target is not reached. Its matched force/strain comparison is retained separately. Process acceptance must not be promoted into convergence, model fidelity or resting physiology qualification.

The optic-only hybrid reaction-history relative L1 error is **7.3740%**, so this narrower reduced model also fails the 1% force gate. Preserving fine skin alone is insufficient.

Three-root 16-versus-32-node follow-up reverses the initial single-root improvement: final acceleration differs 4.1059%, max determinant 1.2035%, and the post-first body interval is 8.512516 versus 8.276194 seconds. This candidate is not promoted. The control post-first interval is only 250 simulated microseconds and measures about 0.000030207x mechanics throughput, not settled integrated performance.

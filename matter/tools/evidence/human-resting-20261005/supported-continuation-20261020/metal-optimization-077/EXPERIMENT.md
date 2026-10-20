# Tangent selection audit and material work census

Question: does the current parallel cache select a different tangent law for columns 1-8, and which current Human materials can use projection/factor reuse?

Source baseline: 245a7610. Current evaluateTangentVectorWithCandidate already owns canonical selection and precision fallback; evaluateAlgorithmicStressTangent's stateless branch calls that same helper. The alleged shortcut bypass does not occur at this revision. Preserve one helper and add explicit regression coverage rather than another selector.

Verification: use existing native cached/direct action checks for all three canonical source materials, both tagged and forced bytecode paths, with nine affine and six fixed-seed random nodal directions under two environments and rotated material frames. Preserve existing thresholds and serial-cache bitwise check.

Instrumentation: opt-in CPU upload-time material census only. Report emitted program instruction counts and active source tetrahedra, not measured shader instructions or time. It cannot apportion the earlier 8.06 GPU seconds, projection iterations, factorization counts or spills. Verify actual Human world reports whether any stateful finite-difference fallback can run. Default simulation and acceptance algorithms remain unchanged.

No prediction of throughput gain is made for these audit/regression changes. Keep shared-factor and canonical switches off by default, retain coupon-only diagnostic admission, and preserve completed-Newton latches.

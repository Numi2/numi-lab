# Cadence 044: rejected exploratory 1 ms step

The preregistered 1 ms experiment failed its stability prediction. Eight root steps were requested; no root step was accepted. The run exited 1 in 24.136170999961905 wall seconds. One diagnostic replay requested one root and reproduced the failure receipt byte-for-byte in 24.130515416967683 seconds.

The accepted baseline remains the published 125 microsecond configuration. No production source, material, nonlinear tolerance, contact tolerance, or iteration budget was changed for 044. This result does not establish an admissible faster cadence.

The replay refutes the initial alpha synchronization hypothesis in DIAGNOSTIC.md. The shared line search used alpha 0.25 once and 1.0 thereafter, but the 24 Newton iterations never latched convergence. The final traced assembled residual was 9.786016016732901e-5 against the unchanged 1e-5 target. The subsequent mixed certificate rejected the candidate with residual 1.372531260130927e-4 and correction 3.7423690082505345e-4. The compact trajectory and exact producer are retained in replay-root0-044/analysis.

No accepted body, tissue, vascular, controller or presentation time advanced. This is negative numerical evidence, not physiological or whole-body anatomical qualification. Improving nonlinear convergence would require a separately tested solver change; this failure is not permission to relax the certificate.

The child packages retain source/binary/input identities, invocation, failure receipt, compact traces and hashes. Large scene packs remain in their completed runtime directories and may share an inode with the identical retained canonical pack.

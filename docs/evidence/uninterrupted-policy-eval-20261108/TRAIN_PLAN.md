# Integrated non-G1 train and selection check

Run one minimal MLX training update on the retained synthetic Franka pack, two environments, eight training steps, and a two-environment held-out selector. Require successful completion, candidate and incumbent evaluation identities containing `--no-periodic-resets`, two initial host resets and no host periodic resets during the requested 256-step evaluation, and unchanged fixture hashes. Preserve all outputs and failure. This verifies the integrated command and selected state contract, not learned skill or measured physical performance.

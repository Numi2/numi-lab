# Uninterrupted policy evaluation

The native two-environment horizon check observed a periodic host reset at step 220 in environment 1, even though the selector requests one 256-step held-out horizon. This biases task metrics and clean-horizon reporting. Keep default rollout scheduling unchanged, but add an option that disables only periodic host resets and retains the first authored reset. Make policy selection request that option.

Before evaluation, require the default two-environment results at 221, 256 and 257 steps to remain clean rates 0.5, 0.5 and 0, with timeout counts 0, 1 and 1. With periodic resets disabled, require clean rates 1, 1 and 0, timeout counts 0, 2 and 2, and exactly two initial host resets, using chunks 1 and 8. Check both zero-action and same-runtime compiled-policy paths; keep all authored failures active. Verify selector arguments and the existing non-G1 policy tests. No performance, policy-skill, or physical-contact claim follows.

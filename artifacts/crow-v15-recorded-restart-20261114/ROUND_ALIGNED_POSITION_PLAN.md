# Aligned branch small fore-aft offset test

At x=17.770 m, +0.2 rad branch yaw produced 166 stable gripping intervals and then forward tipping. Test a +0.002 m x offset on that exact yawed round collider, with no balance correction and otherwise identical toe targets, state, policy, observer and 400-interval horizon. This probes whether the center of support was slightly behind the bird; it is a single local development test with a relocated branch and forced landing phase, not autonomous generalization.

Count only the first episode. Report both feet's all-substep support/opposition, root tilt and speed, nonfoot/ground contact, and first termination. Require no reset/termination and 250 final consecutive intervals satisfying the existing gates. Preserve failure.

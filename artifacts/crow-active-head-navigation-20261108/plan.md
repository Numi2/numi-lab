# Active-head navigation admission, 2026-11-08

Test the newly separated v15 native course contract without loading an old
15-action policy. Both tests use 1 environment, 8 steps, 32 physics substeps,
zero actions, seed 20261108, one repeat and no scheduled resets.

1. The legacy torso-mounted course camera must be rejected explicitly.
2. The head-mounted development camera must construct and execute accepted
   Metal transitions, with 18 actions and distinct fingerprints. Record any
   rejection, sensor issue, or nonzero failed-step count rather than relaxing
   physics checks.

These tests establish bounded integration only, not learned navigation,
camera biological validity, collision avoidance, or held-out course completion.
They run sequentially after the ongoing grasp controls to avoid GPU contention.

Also attempt the existing feeding policy with the new contract and require
rejection. Save the small 8-frame native replay, including actor observations,
if the zero-action run succeeds; this is sensor plumbing evidence, not a
perception or navigation qualification. Do not overwrite earlier attempts.

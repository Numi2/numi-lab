# Active-head navigation training smoke

The corrected v15 visual runtime completed two fixed-rate PPO updates:
4 environments, 32 steps per update, 256 total samples. No simulation step
failed. This was an integration smoke, not an optimization or promotion trial.

Paired short held-out screens used 8 environments, 256 steps and four physics
substeps on each of two course layouts, comparing the fresh zero-output actor
with the updated actor. Neither actor reached even the first waypoint in any
lane. All four runs had zero failed simulation steps, but episodes did have
physical-failure terminations. This is not successful learned navigation.
The 256-step screen is shorter and smaller than the fixed promotion gate and
must not be interpreted as a qualification result.

The previous v10 navigation actor has 15 actions and a different observation,
physical, and visual contract. It cannot safely be loaded as an 18-action v15
policy without a verified semantic transfer. Retained raw rollout, model and
learner artifacts are archived in Git evidence, with no policy promoted.

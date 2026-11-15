# V3 optimizer contract amendment

The first v3 launch failed before collecting transitions: the native learner correctly rejects restoring optimizer state across a changed GAE lambda (`ppo=False[gae_lambda]`). Preserve this failed launch. Restart from the exact retained v1 policy weights with a newly initialized optimizer for update one, then carry the new optimizer state across subsequent updates. This changes the optimizer history as well as lambda, so the comparison cannot isolate the effect of longer credit alone. All other predeclared v3 settings remain unchanged.

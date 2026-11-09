# Goal-memory imitation

Warm-start from the explicitly rebound prior policy, train 120 epochs at fixed Adam learning rate 0.0001 on successful new-contract demonstrations with every-fifth-step subsampling and stage balancing. Preserve the existing head and beak behavior, train on lanes 0-5 per development layout and use lanes 6-7 for development validation. The new task contract changes observation meaning; old-contract demonstrations cannot be used as if their observations had the new semantics. Teacher-off physical execution, not imitation error, decides performance.

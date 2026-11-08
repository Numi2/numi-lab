# Longer free-rod trajectory result

Two runs of the preregistered 100-step extension produced identical records. Seed 92101 passed 10 steps and then exceeded the 1e-6 m position gate at step 11, with 1.00789794788773e-6 m error. Seed 92105 passed 32 and exceeded the same gate at step 33, with 1.0129633390058043e-6 m error. Seed 92109 passed all 100 steps. Thus 142 of 300 requested steps passed before the two early terminations; these are consecutive steps in three trajectories, not independent trials.

At the two failures, both Metal and the FP64 reference reported successful local solves, and local and accumulated momentum checks remained within their gates. The largest position differences were transverse and varied among nodes. This evidence rules out using short-run success as long-run qualification; it does not by itself assign the error to the Metal solver, initial-state conversion, or reference tolerance. In particular, the reference was allowed a 1e-4 local velocity-unit residual, so its own accumulated error needs a separate refinement check.

All exact inputs, outputs, logs, return codes and source/binary hashes are retained. No production MetalWorld, contact, material-calibration or performance claim follows.

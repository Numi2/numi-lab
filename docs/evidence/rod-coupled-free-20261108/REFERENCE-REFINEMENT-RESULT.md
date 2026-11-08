# Refined FP64 trajectory reference

The reference with 1e-6 local residual tolerance completed every state it was asked to solve in two byte-identical trials. Against that tighter reference, seed 92101 passed nine steps then exceeded the unchanged 1e-6 m position gate at step 10 (1.0040417312196144e-6 m); seed 92105 passed 37 then exceeded it at step 38 (1.0114144606519296e-6 m); seed 92109 passed all 100. These are 146 accepted-and-passing steps before two early terminations, of 300 requested.

The weaker and tighter independently advanced references differed in position by up to 9.6911e-8 m, 8.2511e-8 m and 2.9299e-8 m respectively over their shared reported durations. Thus reference tolerance affects the exact crossing time, but tightening it does not remove the two failures. The Metal solver and all position, local-residual and momentum gates were unchanged. No production or physical-calibration claim follows.

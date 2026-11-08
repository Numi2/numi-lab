# Valid, missing, reacquired sequence

Extend the GPU-only reducer fixture with three submitted frames using one persistent track buffer and no reset after the first frame: valid depth, complete depth dropout, then valid depth at the same point. For each environment and actor/history buffer, require confidence slots [1,1], then [1,0], then [0,1]. In the final frame require zero measured velocity because the immediately preceding observation was invalid. Preserve the original independent per-frame controls. This is a submitted sensor sequence, not a physics-step acceptance/rollback test.

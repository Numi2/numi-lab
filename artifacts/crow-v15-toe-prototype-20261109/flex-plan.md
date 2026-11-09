# Grounded toe flexion and release probe

On the isolated armature prototype, compare a one-second fixed-pose control with a matched arm that commands all eight toe hinges toward 0.35 radians during control steps 10 through 29 and returns to zero afterward. Both use the native contact solver, the nonpenetrating default ground start, 20 ms periods and four physics substeps. Prediction: no rollback, meaningful positive toe joint movement during command and recovery toward the fixed control after release, without loss of finite body state. Preserve full q/v trace and final solver-reported toe-ground impulses. Calibrate trajectory alignment against finalQ. This is ground flexion, not a claim of perch grasp, landing, or biologically validated actuation.

The first probe stopped at its trajectory-length assertion after an accepted fixed arm; no flex arm was run. Inspect the actual native stream length and calibrate its frame offset before interpreting it.

The second readout interpreted the q/v stream with a 61-value stride, but the native contact trajectory appends 13 scene-body values per frame, so its 74-value stride was required; the endpoint calibration rejected the interpretation. That derived trace is invalid. Repeat the same exploratory arms using the native dispatch stride, requiring the last q to agree with finalQ before interpreting any frame.

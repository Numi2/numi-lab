    // Refinement has its own small, fixed work bound. It never changes the
    // Newton/PCG budgets or acceptance tolerances, and old dispatches with a
    // one-float PCG Hessian arena safely skip it.
    if (failureCode == MR_UNIFIED_QUALITY_DID_NOT_CONVERGE &&
        nv <= MR_UNIFIED_QUALITY_MAX_ADJACENT_VELOCITIES &&
        dispatch.hessianStride >= nv * nv &&
        isfinite(scalars[0]) && isfinite(scalars[1]) &&
        scalars[1] <= 16.0f * dispatch.tolerances.x &&
        scalars[2] <= dispatch.tolerances.y) {
        for (uint refinement = 0u;
             refinement < MR_UNIFIED_QUALITY_MAX_ADJACENT_REFINEMENTS;
             ++refinement) {
            if (!chooseAdjacentVelocity(
                    dispatch, problemBlocks, dynamics, jacobian, derivatives,
                    hessian, velocity, gradient, direction,
                    candidateVelocity, lane)) {
                break;
            }
            const float previousObjective = scalars[0];
            const float previousOptimality = scalars[1];
            const bool candidateValid = evaluateState(
                dispatch, problemBlocks, dynamics, jacobian, bias,
                freeVelocity, candidateVelocity, candidateRowVelocity,
                candidateImpulses, candidateGradient, candidateDynamicsAction,
                candidateConstraintAction, derivatives, candidateScalars,
                control, false, lane
            );
            const float objectiveRoundoff = 32.0f * kFloatEpsilon *
                max(abs(previousObjective), 1.0e-12f);
            const bool accepted = candidateValid &&
                candidateScalars[1] < previousOptimality &&
                candidateScalars[2] <= dispatch.tolerances.y &&
                candidateScalars[0] <= previousObjective + objectiveRoundoff;
            if (accepted) {
                objectiveChange = candidateScalars[0] - previousObjective;
                for (uint dof = lane; dof < nv; dof += kWidth) {
                    velocity[dof] = candidateVelocity[dof];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Candidate evaluation aliases current action scratch. Restore
            // the chosen state and derivatives even when the move is rejected.
            if (!evaluateState(
                    dispatch, problemBlocks, dynamics, jacobian, bias,
                    freeVelocity, velocity, rowVelocity, impulses, gradient,
                    dynamicsAction, constraintAction, derivatives, scalars,
                    control, true, lane)) {
                failureCode = control[0] == MR_UNIFIED_QUALITY_INVALID_INDEX
                    ? MR_UNIFIED_QUALITY_NONFINITE_RESULT
                    : MR_UNIFIED_QUALITY_INVALID_BLOCK;
                break;
            }
            if (scalars[1] <= dispatch.tolerances.x &&
                scalars[2] <= dispatch.tolerances.y) {
                converged = true;
                failureCode = MR_UNIFIED_QUALITY_SUCCESS;
                break;
            }
            if (!accepted) break;
        }
    }

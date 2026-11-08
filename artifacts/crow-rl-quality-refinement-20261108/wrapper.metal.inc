
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    MRUnifiedQualityStatusGPU next = statuses[problem];
    const uint nv = dispatch.generalizedVelocityCount, rows = dispatch.rowCount, nb = dispatch.blockCount;
    const uint required = mrUnifiedQualityRefinementScratchFloats(nv, rows, nb);
    if (next.code != MR_UNIFIED_QUALITY_DID_NOT_CONVERGE ||
        required == 0u || dispatch.hessianStride < required) return;
    const uint base = problem * dispatch.vectorStride;
    device const MRUnifiedQualityBlockGPU* pb = blocks +
        (dispatch.blockStride == 0u ? 0u : problem * dispatch.blockStride);
    device const float* A = dynamicsMatrices + problem * dispatch.dynamicsStride;
    device const float* J = jacobianMatrices + problem * dispatch.jacobianStride;
    device const float* bias = biasVectors + base;
    device const float* freev = freeVelocities + base;
    device mr_quality_expansion::DS* x = reinterpret_cast<device mr_quality_expansion::DS*>(
        hessianScratch + problem * dispatch.hessianStride);
    const uint count = 4u * nv + 2u * rows + 36u * nb + nv * nv + 1u;
    device uint* es = reinterpret_cast<device uint*>(x + count);
    mr_quality_expansion::Params c = {0u, 1u, nv, rows, nb,
        MR_UNIFIED_QUALITY_MAX_EXPANSION_REFINEMENTS, 0u, 0u};
    if (!mr_quality_expansion::solveCandidate(c, pb, A, J, bias, freev,
            workspace.velocity, x, es, lane)) return;
    // An expanded iterate alone never authorizes publication.
    for (uint k = lane; k < nv; k += kWidth) {
        const float v = x[k].h + x[k].l;
        workspace.candidateVelocity[k] = v; x[k] = mr_quality_expansion::ds(v);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    device mr_quality_expansion::DS* ew = x + nv;
    device mr_quality_expansion::DS* ep = ew + rows;
    device mr_quality_expansion::DS* eg = ep + rows;
    device mr_quality_expansion::DS* eW = eg + nv;
    device mr_quality_expansion::DS* eo = eW + 36u * nb + nv * nv + 2u * nv;
    const bool rounded = mr_quality_expansion::evaluateDS(c, pb, A, J, bias, freev,
        x, ew, ep, eg, eW, eo, es, lane);
    device float* em = reinterpret_cast<device float*>(es + 16u);
    if (!rounded || em[0] > dispatch.tolerances.x || em[4] > dispatch.tolerances.y) return;
    const bool valid = evaluateState(dispatch, pb, A, J, bias, freev,
        workspace.candidateVelocity, workspace.rowWork, workspace.rowAction,
        workspace.cgDirection, workspace.cgAction, workspace.dynamicsAction,
        derivativeScratch + problem * dispatch.derivativeStride,
        workspace.candidateScalars, workspace.control, false, lane);
    if (!valid || workspace.candidateScalars[1] > dispatch.tolerances.x ||
        workspace.candidateScalars[2] > dispatch.tolerances.y) return;
    float publishedResidual = 0.0f, publishedFeasibility = 0.0f;
    if (!mr_quality_expansion::auditPublished(c, pb, A, J, freev, x,
            workspace.rowAction, publishedResidual, publishedFeasibility, lane) ||
        publishedResidual > dispatch.tolerances.x ||
        publishedFeasibility > dispatch.tolerances.y) return;
    const float optimality = max(workspace.candidateScalars[1], max(em[0], publishedResidual));
    const float feasibility = max(workspace.candidateScalars[2], max(em[4], publishedFeasibility));
    for (uint k = lane; k < nv; k += kWidth) {
        workspace.velocity[k] = workspace.candidateVelocity[k];
        workspace.gradient[k] = workspace.cgDirection[k];
        outputVelocities[base + k] = workspace.candidateVelocity[k];
    }
    for (uint k = lane; k < rows; k += kWidth) {
        workspace.impulses[k] = workspace.rowAction[k];
        workspace.rowVelocity[k] = workspace.rowWork[k];
        outputImpulses[base + k] = workspace.rowAction[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (lane == 0u) {
        const float previous = next.certificates1.w;
        next.code = MR_UNIFIED_QUALITY_SUCCESS;
        next.solvePath = MR_UNIFIED_QUALITY_PATH_REFINED_DIRECT;
        next.failingBlock = MR_UNIFIED_QUALITY_INVALID_INDEX;
        next.firstFailingStableKey = uint4(MR_UNIFIED_QUALITY_INVALID_INDEX);
        next.newtonIterations += es[1];
        next.lineSearchBacktracks += es[10];
        next.certificates0 = float4(optimality, feasibility,
            workspace.candidateScalars[3], workspace.candidateScalars[4]);
        next.certificates1 = float4(max(workspace.candidateScalars[5], publishedResidual),
            0.0f, workspace.candidateScalars[0] - previous, workspace.candidateScalars[0]);
        for (uint k = 0u; k < 16u; ++k) workspace.scalars[k] = workspace.candidateScalars[k];
        workspace.scalars[1] = optimality; workspace.scalars[2] = feasibility;
        statuses[problem] = next;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

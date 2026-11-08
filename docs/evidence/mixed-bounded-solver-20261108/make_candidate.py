from pathlib import Path
whole=Path('/Users/numi/frictionless-mixed-rod-20261108/split.metal').read_text()
lo=whole.index('inline void mrWorldWave32SolvePacket(')
hi=whole.index('// One SIMD32 group owns',lo)
s=whole[lo:hi]
a='''    const uint solverIterations =
        dispatch.velocityIterations +
        (pass.reserved0 != 0u
             ? dispatch.finalVelocityIterations
             : 0u);
    for (uint iteration = 0u;
         iteration < solverIterations;
         ++iteration) {'''
b='''    const uint requestedIterations =
        dispatch.velocityIterations +
        (pass.reserved0 != 0u ? dispatch.finalVelocityIterations : 0u);
    const bool rodWork = (work.flags & MR_ISLAND_WORK_HAS_ROD) != 0u;
    const bool packetHasRod = simd_any(rodWork);
    const uint solverIterations = packetHasRod ? CAP_SWEEPS : requestedIterations;
    uint completedIterations = 0u;
    float finalNormal = 0.0f;
    float finalTangent = 0.0f;
    for (uint iteration = 0u;
         iteration < solverIterations;
         ++iteration) {'''
assert s.count(a)==1,s.count(a);s=s.replace(a,b)
a='''            if (localLane < tile.constraintCount) {
            const uint localConstraint ='''
b='''            if (localLane < tile.constraintCount) {
            const uint localConstraint ='''
# Restrict changes to the selected loop, not earlier code.
start=s.index('    const uint requestedIterations =');end=s.index('    float laneNormalResidual = 0.0f;',start)
block=s[start:end]
a='''            device MRContactConstraintGPU& contact =
                contacts[constraintBase + localConstraint];'''
b='''            if (!rodWork && iteration >= requestedIterations) {
                impulseDeltas[constraintBase + localConstraint] = float4(0.0f);
                continue;
            }
            device MRContactConstraintGPU& contact =
                contacts[constraintBase + localConstraint];'''
assert block.count(a)==1,block.count(a);block=block.replace(a,b)
a='''        threadgroup_barrier(mem_flags::mem_device);
    }

'''
b='''        threadgroup_barrier(mem_flags::mem_device);
        float laneNormal = 0.0f;
        float laneTangent = 0.0f;
        if (rodWork) {
            for (uint localTile = 0u; localTile < work.tileCount; ++localTile) {
                const MRContactTileGPU tile = tiles[tileBase + work.firstTile + localTile];
                if (localLane >= tile.constraintCount) continue;
                const uint localConstraint = tileConstraintIndices[constraintBase + tile.partialOffset + localLane];
                device const MRContactConstraintGPU& contact = contacts[constraintBase + localConstraint];
                device const MREvaluatedConstraintIRRowGPU* rows = evaluatedRows + rowBase + 3u * localConstraint;
                const MRWave32PreconditionerGPU pre = preconditioners[constraintBase + localConstraint];
                const MREvaluatedConstraintIRConeGPU cone = evaluatedCones[constraintBase + localConstraint];
                const float3 rel = typedRelativePointVelocity(localConstraint, contact, bodies, dispatch.articulationIndex,
                    pointJacobians, pointJacobianBase, dispatch.nv, articulationVelocity, rodNodes, rodEdges,
                    rodColliders, rodWitnesses, constraintWitnessIndices, constraintBase);
                const float3 imp = contact.impulses.xyz;
                const float3 rhs = float3(rows[0].targetVelocity - dot(rows[0].direction.xyz, rel) - rows[0].regularization * imp.x,
                    rows[1].targetVelocity - dot(rows[1].direction.xyz, rel) - rows[1].regularization * imp.y,
                    rows[2].targetVelocity - dot(rows[2].direction.xyz, rel) - rows[2].regularization * imp.z);
                if (!finite3(rhs) || !finite3(imp) || !(pre.row1.w > 0.0f) || !isfinite(pre.row1.w) ||
                    cone.effectiveFrictionU != cone.effectiveFrictionV || cone.effectiveFrictionU < 0.0f || cone.maximumNormalImpulse > 0.0f) {
                    laneNormal = INFINITY; laneTangent = INFINITY;
                } else {
                    const float normal = imp.x > kConeEpsilon ? abs(rhs.x) : max(rhs.x, 0.0f);
                    const float3 projected = projectFrictionCone(float3(imp.x, imp.y + pre.row1.w * rhs.y, imp.z + pre.row1.w * rhs.z), cone);
                    const float tangent = length(projected.yz - imp.yz) / pre.row1.w;
                    laneNormal = max(laneNormal, normal);
                    laneTangent = max(laneTangent, tangent);
                }
            }
        }
        finalNormal = waveCohortMaximum(laneNormal, cohortWidth);
        finalTangent = waveCohortMaximum(laneTangent, cohortWidth);
        completedIterations = iteration + 1u;
        const float groupError = simd_max(max(finalNormal, finalTangent));
        if (completedIterations >= requestedIterations && groupError <= 0.002f) break;
    }

'''
assert block.count(a)==1,block.count(a);block=block.replace(a,b);s=s[:start]+block+s[end:]
a='''        result.code =
            maximumFailureCode;
        result.environment = environment;
        result.islandIndex = islandIndex;
        result.iterations = solverIterations;
        if (maximumFailureCode != MR_STEP_SUCCESS) {'''
b='''        result.code = maximumFailureCode;
        if (result.code == MR_STEP_SUCCESS && rodWork &&
            (!isfinite(finalNormal) || !isfinite(finalTangent) || finalNormal > 0.002f || finalTangent > 0.002f))
            result.code = MR_STEP_DID_NOT_CONVERGE;
        result.environment = environment;
        result.islandIndex = islandIndex;
        result.iterations = completedIterations;
        if (maximumFailureCode != MR_STEP_SUCCESS) {'''
assert s.count(a)==1,s.count(a);s=s.replace(a,b)
# On nonconvergence retain normal and tangent diagnostics; w identifies an island-level failure.
a='''        } else {
            result.residuals = float4(
                maximumImpulseDelta,
                maximumNormalResidual,
                maximumConeViolation,
                0.0f
            );
        }
        waveStatuses[islandBase + islandIndex] = result;'''
b='''        } else if (result.code == MR_STEP_DID_NOT_CONVERGE) {
            result.residuals = float4(maximumImpulseDelta, finalNormal, finalTangent, 0.0f);
        } else {
            result.residuals = float4(maximumImpulseDelta, maximumNormalResidual, maximumConeViolation, 0.0f);
        }
        waveStatuses[islandBase + islandIndex] = result;'''
assert s.count(a)==1,s.count(a);s=s.replace(a,b)
for cap in (2,64):Path(f'candidate-{cap}.metal').write_text(whole[:lo]+s.replace('CAP_SWEEPS',str(cap)+'u')+whole[hi:])

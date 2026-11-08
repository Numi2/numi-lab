import pathlib
root = pathlib.Path(__file__).resolve().parent
source = pathlib.Path("/Users/numi/Work/mixed-full-operator-artifacts-20261108/baseline.metal").read_text()
anchor = """        threadgroup_barrier(mem_flags::mem_device);

        if ((work.flags &
             MR_ISLAND_WORK_HAS_ARTICULATION) != 0u) {"""
assert source.count(anchor) == 1
block = r"""
        // Isolated research candidate: exact normal pair on the retained
        // three-node rod only. Unsupported or uncertified blocks fall back.
        if (localLane == 0u && localFailure == MR_STEP_SUCCESS &&
            work.tileCount == 1u && dispatch.rodCount == 1u &&
            dispatch.rodNodeCount == 3u && dispatch.rodEdgeCount == 2u) {
            const MRContactTileGPU bt = tiles[tileBase + work.firstTile];
            uint ids[2] = {0u, 0u};
            uint count = 0u;
            bool valid = bt.constraintCount == 3u;
            for (uint s = 0u; s < bt.constraintCount && valid; ++s) {
                const uint id = tileConstraintIndices[constraintBase + bt.partialOffset + s];
                if (typedRodConstraint(contacts[constraintBase + id])) {
                    if (count >= 2u) { valid = false; break; }
                    ids[count++] = id;
                }
            }
            valid = valid && count == 2u;
            float h[2][2];
            float rhs[2];
            float previous[2];
            for (uint i = 0u; i < 2u && valid; ++i) {
                const uint wi = constraintWitnessIndices[constraintBase + ids[i]];
                if (wi == MR_INVALID_INDEX) { valid = false; break; }
                const MRRodToolWitnessGPU witnessI = rodWitnesses[wi];
                const MRRodColliderGPU colliderI = rodColliders[witnessI.identity.z];
                if (colliderI.rodIndex != 0u ||
                    evaluatedCones[constraintBase + ids[i]].maximumNormalImpulse > 0.0f) {
                    valid = false; break;
                }
                device MRContactConstraintGPU& ci = contacts[constraintBase + ids[i]];
                device const MREvaluatedConstraintIRRowGPU* ri = evaluatedRows + rowBase + 3u * ids[i];
                previous[i] = ci.impulses.x - impulseDeltas[constraintBase + ids[i]].x;
                const float3 velocity = typedRelativePointVelocity(
                    ids[i], ci, bodies, dispatch.articulationIndex, pointJacobians,
                    pointJacobianBase, dispatch.nv, articulationVelocity,
                    rodNodes, rodEdges, rodColliders, rodWitnesses,
                    constraintWitnessIndices, constraintBase);
                rhs[i] = ri[0].targetVelocity - dot(ri[0].direction.xyz, velocity) -
                         ri[0].regularization * previous[i];
                for (uint j = 0u; j < 2u; ++j) {
                    const uint wj = constraintWitnessIndices[constraintBase + ids[j]];
                    if (wj == MR_INVALID_INDEX) { valid = false; break; }
                    const MRRodToolWitnessGPU witnessJ = rodWitnesses[wj];
                    const MRRodColliderGPU colliderJ = rodColliders[witnessJ.identity.z];
                    if (colliderJ.rodIndex != 0u) { valid = false; break; }
                    device MRContactConstraintGPU& cj = contacts[constraintBase + ids[j]];
                    device const MREvaluatedConstraintIRRowGPU* rj = evaluatedRows + rowBase + 3u * ids[j];
                    float3 exact;
                    if (!factorizedRodCrossResponse(dispatch, pass, environment, colliderI, witnessI,
                        colliderJ, witnessJ, rj[0].direction.xyz, rodNodes, rodFactorCaches,
                        rodOperatorArena, exact)) { valid = false; break; }
                    h[i][j] = typedNormalCrossContactResponse(ids[i], ids[j], 0u,
                        ci, cj, ri[0].direction.xyz, rj[0].direction.xyz,
                        bodies, pointJacobians, pointJacobianBase, responseColumns,
                        responseBase, dispatch.nv, dispatch.articulationIndex, rodNodes,
                        inverseRodMasses, inverseRodTwistInertias, rodColliders,
                        rodWitnesses, constraintWitnessIndices, constraintBase) +
                        dot(ri[0].direction.xyz, exact) -
                        rodCrossContactResponse(colliderI, witnessI, colliderJ, witnessJ,
                            rodNodes, inverseRodMasses, inverseRodTwistInertias,
                            ri[0].direction.xyz, rj[0].direction.xyz);
                    if (i == j) h[i][j] += ri[0].regularization;
                    valid = valid && isfinite(h[i][j]);
                }
                valid = valid && isfinite(rhs[i]) && isfinite(previous[i]) && previous[i] >= 0.0f;
            }
            if (valid) {
                const float det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
                valid = h[0][0] > 0.0f && h[1][1] > 0.0f &&
                        isfinite(det) && det > 1e-7f * h[0][0] * h[1][1] &&
                        abs(h[0][1] - h[1][0]) <= 1e-4f * max(max(abs(h[0][1]), abs(h[1][0])), 1.0f);
                if (valid) {
                    const float2 b = float2(rhs[0] + h[0][0]*previous[0] + h[0][1]*previous[1],
                                            rhs[1] + h[1][0]*previous[0] + h[1][1]*previous[1]);
                    float2 p = float2((b.x*h[1][1] - h[0][1]*b.y)/det,
                                      (h[0][0]*b.y - h[1][0]*b.x)/det);
                    bool solved = all(isfinite(p)) && all(p >= 0.0f);
                    if (!solved) {
                        const float one = b.x / h[0][0];
                        if (one >= 0.0f && h[1][0]*one - b.y >= 0.0f) {
                            p = float2(one, 0.0f); solved = true;
                        }
                    }
                    if (!solved) {
                        const float one = b.y / h[1][1];
                        if (one >= 0.0f && h[0][1]*one - b.x >= 0.0f) {
                            p = float2(0.0f, one); solved = true;
                        }
                    }
                    if (!solved && all(b <= 0.0f)) { p = float2(0.0f); solved = true; }
                    if (solved && all(isfinite(p))) {
                        for (uint i = 0u; i < 2u; ++i) {
                            device MRContactConstraintGPU& ci = contacts[constraintBase + ids[i]];
                            const float3 old = ci.impulses.xyz - impulseDeltas[constraintBase + ids[i]].xyz;
                            float3 trial = ci.impulses.xyz;
                            const float relaxation = preconditioners[constraintBase + ids[i]].row2.w;
                            trial.x = old.x + relaxation * (p[i] - old.x);
                            trial = projectFrictionCone(trial, evaluatedCones[constraintBase + ids[i]]);
                            if (finite3(trial)) {
                                ci.impulses.xyz = trial;
                                impulseDeltas[constraintBase + ids[i]] = float4(trial - old, 0.0f);
                            }
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        if ((work.flags &
             MR_ISLAND_WORK_HAS_ARTICULATION) != 0u) {"""
(root / "candidate.metal").write_text(source.replace(anchor, block))

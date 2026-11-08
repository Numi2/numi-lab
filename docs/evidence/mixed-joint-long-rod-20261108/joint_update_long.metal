        // Research-only joint block. The unchanged split deltas survive any
        // failed precondition, numerical solve, or final certificate.
        if (localLane == 0u && localFailure == MR_STEP_SUCCESS &&
            work.tileCount == 1u && dispatch.rodCount == 1u &&
            dispatch.rodNodeCount >= 3u && dispatch.rodEdgeCount + 1u == dispatch.rodNodeCount) {
            const MRContactTileGPU jt=tiles[tileBase+work.firstTile];
            uint ids[3]; bool valid=jt.constraintCount==3u; uint rodCount=0u;
            uint witnessNodeA[2],witnessNodeB[2],witnessEdge[2];
            float j0[9],r0[9],W[81],sn[3],st[3],mu[3],solved[9];
            for (uint i=0u;i<3u && valid;++i) {
                ids[i]=tileConstraintIndices[constraintBase+jt.partialOffset+i];
                device MRContactConstraintGPU& c=contacts[constraintBase+ids[i]];
                device const MREvaluatedConstraintIRRowGPU* rows=evaluatedRows+rowBase+3u*ids[i];
                const MRWave32PreconditionerGPU pre=preconditioners[constraintBase+ids[i]];
                const MREvaluatedConstraintIRConeGPU cone=evaluatedCones[constraintBase+ids[i]];
                const float3 old=c.impulses.xyz-impulseDeltas[constraintBase+ids[i]].xyz;
                const float3 velocity=typedRelativePointVelocity(ids[i],c,bodies,dispatch.articulationIndex,
                    pointJacobians,pointJacobianBase,dispatch.nv,articulationVelocity,rodNodes,rodEdges,
                    rodColliders,rodWitnesses,constraintWitnessIndices,constraintBase);
                if (typedRodConstraint(c)) {
                    if (rodCount>=2u) { valid=false; break; }
                    const uint wi=constraintWitnessIndices[constraintBase+ids[i]];
                    if (wi==MR_INVALID_INDEX) { valid=false; break; }
                    const MRRodToolWitnessGPU w=rodWitnesses[wi];
                    const MRRodColliderGPU collider=rodColliders[w.identity.z];
                    if (collider.rodIndex!=0u || collider.nodeA>=dispatch.rodNodeCount ||
                        collider.nodeB>=dispatch.rodNodeCount || collider.edgeIndex>=dispatch.rodEdgeCount) {
                        valid=false; break;
                    }
                    witnessNodeA[rodCount]=collider.nodeA;
                    witnessNodeB[rodCount]=collider.nodeB;
                    witnessEdge[rodCount]=collider.edgeIndex;
                    ++rodCount;
                }
                sn[i]=pre.row0.w; st[i]=pre.row1.w; mu[i]=cone.effectiveFrictionU;
                valid=finite3(old) && finite3(velocity) && old.x>=0.0f &&
                    isfinite(sn[i]) && sn[i]>0.0f && isfinite(st[i]) && st[i]>0.0f &&
                    isfinite(mu[i]) && mu[i]>=0.0f && isfinite(cone.effectiveFrictionV) &&
                    mu[i]==cone.effectiveFrictionV && isfinite(cone.maximumNormalImpulse) &&
                    cone.maximumNormalImpulse<=0.0f;
                for (uint a=0u;a<3u && valid;++a) {
                    j0[3u*i+a]=old[a];
                    r0[3u*i+a]=rows[a].targetVelocity-dot(rows[a].direction.xyz,velocity)-rows[a].regularization*old[a];
                    valid=valid && isfinite(r0[3u*i+a]);
                }
            }
            valid=valid && rodCount==2u;
            if (valid) valid=mrResearchRodWitnessesFitSelectedBand(
                witnessNodeA[0],witnessNodeB[0],witnessEdge[0],
                witnessNodeA[1],witnessNodeB[1],witnessEdge[1]);
            for (uint i=0u;i<3u && valid;++i) {
                device MRContactConstraintGPU& ci=contacts[constraintBase+ids[i]];
                device const MREvaluatedConstraintIRRowGPU* ri=evaluatedRows+rowBase+3u*ids[i];
                for (uint a=0u;a<3u && valid;++a) for (uint j=0u;j<3u && valid;++j) {
                    device MRContactConstraintGPU& cj=contacts[constraintBase+ids[j]];
                    device const MREvaluatedConstraintIRRowGPU* rj=evaluatedRows+rowBase+3u*ids[j];
                    for (uint b=0u;b<3u && valid;++b) {
                        const float3 di=ri[a].direction.xyz, dj=rj[b].direction.xyz;
                        float value=typedNormalCrossContactResponse(ids[i],ids[j],b,ci,cj,di,dj,
                            bodies,pointJacobians,pointJacobianBase,responseColumns,responseBase,
                            dispatch.nv,dispatch.articulationIndex,rodNodes,inverseRodMasses,
                            inverseRodTwistInertias,rodColliders,rodWitnesses,constraintWitnessIndices,constraintBase);
                        if (typedRodConstraint(ci) && typedRodConstraint(cj)) {
                            const MRRodToolWitnessGPU wi=rodWitnesses[constraintWitnessIndices[constraintBase+ids[i]]];
                            const MRRodToolWitnessGPU wj=rodWitnesses[constraintWitnessIndices[constraintBase+ids[j]]];
                            const MRRodColliderGPU ti=rodColliders[wi.identity.z], tj=rodColliders[wj.identity.z];
                            float3 exact;
                            if (!factorizedRodCrossResponse(dispatch,pass,environment,ti,wi,tj,wj,dj,
                                rodNodes,rodFactorCaches,rodOperatorArena,exact)) { valid=false; break; }
                            value+=dot(di,exact)-rodCrossContactResponse(ti,wi,tj,wj,rodNodes,
                                inverseRodMasses,inverseRodTwistInertias,di,dj);
                        }
                        if (i==j && a==b) value+=ri[a].regularization;
                        W[9u*(3u*i+a)+3u*j+b]=value;
                        valid=valid && isfinite(value);
                    }
                }
            }
            if (valid) {
                float largest=0.0f, asymmetry=0.0f;
                for (uint i=0u;i<9u;++i) for (uint j=0u;j<9u;++j) {
                    largest=max(largest,abs(W[9u*i+j]));
                    asymmetry=max(asymmetry,abs(W[9u*i+j]-W[9u*j+i]));
                    if (i==j && !(W[9u*i+i]>0.0f)) valid=false;
                }
                valid=valid && isfinite(largest) && largest>0.0f && asymmetry<=1e-4f*largest;
            }
            if (valid && mrResearchJointSolve(W,j0,r0,sn,st,mu,solved)) {
                for (uint i=0u;i<3u;++i) {
                    const float3 old=float3(j0[3u*i],j0[3u*i+1u],j0[3u*i+2u]);
                    const float3 next=float3(solved[3u*i],solved[3u*i+1u],solved[3u*i+2u]);
                    contacts[constraintBase+ids[i]].impulses.xyz=next;
                    impulseDeltas[constraintBase+ids[i]]=float4(next-old,0.0f);
                }
            }
        }

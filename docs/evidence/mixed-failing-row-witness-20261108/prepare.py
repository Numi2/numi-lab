from pathlib import Path
s=Path('baseline.metal').read_text()
start=s.index('    float laneNormalResidual = 0.0f;',s.index('const float groupError = simd_max(max(finalNormal, finalTangent));'))
needle='''        const float limitU =
            cone.effectiveFrictionU * contact.impulses.x;'''
pos=s.index(needle,start)
common='''        if (typedRodConstraint(contact)) {
            const uint wi = constraintWitnessIndices[constraintBase + localConstraint];
            if (wi != MR_INVALID_INDEX) {
                const MRRodToolWitnessGPU w = rodWitnesses[wi];
                const MRRodColliderGPU c = rodColliders[w.identity.z];
'''
a='''                contact.friction = float4(float(c.nodeA), float(c.nodeB), w.rodPointAndWeight.w, c.radiusAndOffsets.x);
                contact.response = float4(rodNodes[c.nodeB].position.xyz - rodNodes[c.nodeA].position.xyz, rodEdges[c.edgeIndex].twistAndRate.y);
                contact.targetVelocityAndPreSolveNormal = float4(w.radialAndTwistJacobianV.xyz, localRows[0].targetVelocity);
'''
b='''                const device MRBodyStateGPU& tool = bodies[w.featuresAndFlags.z];
                const float3 tv = tool.flagsAndIndices[1] != MR_INVALID_INDEX
                    ? articulatedPointVelocity(pointJacobians, pointJacobianBase, 2u * localConstraint + 1u, dispatch.nv, articulationVelocity)
                    : pointVelocity(tool, w.toolPointAndSeparation.xyz);
                contact.friction = float4(rodNodes[c.nodeA].velocity.xyz, normalEquation);
                contact.response = float4(rodNodes[c.nodeB].velocity.xyz, localRows[0].regularization);
                contact.targetVelocityAndPreSolveNormal = float4(tv, dot(localRows[0].direction.xyz, relative));
'''
for name,body in [('witness',a),('velocity',b)]:
 Path(name+'.metal').write_text(s[:pos]+common+body+'            }\n        }\n'+s[pos:])
p=Path('probe-original.mm').read_text()
needle='std::cout<<"{\\\"row_trace\\\":["<<c.friction.x<<","<<c.friction.y<<","<<c.friction.z<<"],\\\"flags\\\":"<<c.flags'
replacement='std::cout<<"{\\\"row_trace\\\":["<<c.friction.x<<","<<c.friction.y<<","<<c.friction.z<<"],\\\"scratch\\\":["<<c.friction.x<<","<<c.friction.y<<","<<c.friction.z<<","<<c.friction.w<<","<<c.response.x<<","<<c.response.y<<","<<c.response.z<<","<<c.response.w<<","<<c.targetVelocityAndPreSolveNormal.x<<","<<c.targetVelocityAndPreSolveNormal.y<<","<<c.targetVelocityAndPreSolveNormal.z<<","<<c.targetVelocityAndPreSolveNormal.w<<"],\\\"flags\\\":"<<c.flags'
assert p.count(needle)==1
Path('probe.mm').write_text(p.replace(needle,replacement))

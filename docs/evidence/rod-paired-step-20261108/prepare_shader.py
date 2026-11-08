from pathlib import Path
p=Path('/Users/numi/rod-paired-step-20261108')
s=(p/'base-guarded.metal').read_text()
def once(a,b):
 global s
 assert s.count(a)==1,(a[:80],s.count(a))
 s=s.replace(a,b,1)
once('#include "metalrobo/rod_gpu_shared.h"','#include "metalrobo/rod_gpu_shared.h"\n#include "metalrobo/compensated_geometry_gpu.h"')
once('    const uint environment [[threadgroup_position_in_grid]],\n    const uint lane [[thread_index_in_threadgroup]],\n    const uint laneCount [[threads_per_threadgroup]]\n) {\n    MRRodGPUDispatch dispatch = sourceDispatch;', '''    device const float4* inputPositionLow [[buffer(24)]],
    device const float4* attachmentTargetLow [[buffer(25)]],
    device const float* restLengthLow [[buffer(26)]],
    device const float* stiffnessLow [[buffer(27)]],
    device const float* inverseMassLow [[buffer(28)]],
    device float4* outputPositionLow [[buffer(29)]],
    constant float& timestepLow [[buffer(30)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_threadgroup]],
    const uint laneCount [[threads_per_threadgroup]]
) {
    MRRodGPUDispatch dispatch = sourceDispatch;''')
once('                outputVelocities[nodeBase + node] =\n                    inputVelocities[nodeBase + node];','                outputVelocities[nodeBase + node] =\n                    inputVelocities[nodeBase + node];\n                outputPositionLow[nodeBase + node] = inputPositionLow[nodeBase + node];')
once('    threadgroup float axialCoordinate[MR_ROD_GPU_MAX_NODES];','''    threadgroup float axialCoordinate[MR_ROD_GPU_MAX_NODES];
    threadgroup MRCompensatedScalar pairedX[MR_ROD_GPU_MAX_NODES];
    threadgroup MRCompensatedScalar pairedC[MR_ROD_GPU_MAX_NODES];
    threadgroup MRCompensatedScalar pairedD[MR_ROD_GPU_MAX_NODES];''')
needle='    // Research-only exact axial chain solve. Unsupported geometry, coupling,'
branch='''    // Research-only paired x-axis solve, with a local momentum certificate.
    if (lane == 0u && !eventSegment && dispatch.nodeCount >= 2u &&
        dispatch.attachmentCount >= 2u && dispatch.rigidBodyCount == 0u &&
        dispatch.flags == 0u && dispatch.dampingDerivativeTolerance.x == 0.0f &&
        dispatch.dampingDerivativeTolerance.y == 0.0f &&
        all(dispatch.gravityAndTimestep.xyz == float3(0.0f)) &&
        atomic_load_explicit(&failure, memory_order_relaxed) == MR_ROD_GPU_SUCCESS) {
        const uint base = environment * dispatch.attachmentCount;
        const uint end = dispatch.nodeCount - 1u;
        const auto add = [](MRCompensatedScalar a, MRCompensatedScalar b) { return mrCompensatedAdd(a,b); };
        const auto neg = [](MRCompensatedScalar a) { return MRCompensatedScalar{-a.high,-a.low}; };
        const auto mul = [](MRCompensatedScalar a, MRCompensatedScalar b) { return mrCompensatedMultiply(a,b); };
        const auto div = [](MRCompensatedScalar a, MRCompensatedScalar b) { return mrCompensatedDivide(a,b); };
        const MRCompensatedScalar zero = {0.0f,0.0f};
        const MRCompensatedScalar one = {1.0f,0.0f};
        const MRCompensatedScalar h = {dispatch.gravityAndTimestep.w,timestepLow};
        bool eligible = isfinite(timestepLow);
        for (uint i=0u; i<dispatch.nodeCount; ++i) axialPinIndex[i]=0xffffffffu;
        for (uint i=0u; i<dispatch.nodeCount && eligible; ++i) {
            const float4 lo=inputPositionLow[nodeBase+i];
            eligible = all(isfinite(lo)) && originalPositions[i].y == originalPositions[0].y &&
                originalPositions[i].z == originalPositions[0].z && lo.y == inputPositionLow[nodeBase].y &&
                lo.z == inputPositionLow[nodeBase].z && velocities[i].y == 0.0f && velocities[i].z == 0.0f &&
                inverseMasses[i] > 0.0f && isfinite(inverseMasses[i]) && isfinite(inverseMassLow[i]);
            if (i>0u) eligible = eligible && originalPositions[i].x > originalPositions[i-1u].x;
        }
        for (uint j=0u; j<dispatch.attachmentCount && eligible; ++j) {
            const auto a=attachments[base+j]; const auto lo=attachmentTargetLow[base+j];
            eligible=a.nodeIndex<dispatch.nodeCount && a.targetAndCompliance.w==0.0f &&
                all(isfinite(a.targetAndCompliance)) && all(isfinite(a.velocity)) && all(isfinite(lo)) &&
                a.targetAndCompliance.y==originalPositions[0].y && a.targetAndCompliance.z==originalPositions[0].z &&
                lo.y==inputPositionLow[nodeBase].y && lo.z==inputPositionLow[nodeBase].z &&
                a.velocity.y==0.0f && a.velocity.z==0.0f;
            if (eligible) { eligible=axialPinIndex[a.nodeIndex]==0xffffffffu; if (eligible) axialPinIndex[a.nodeIndex]=j; }
        }
        eligible = eligible && axialPinIndex[0]!=0xffffffffu && axialPinIndex[end]!=0xffffffffu;
        for (uint e=0u;e<dispatch.edgeCount && eligible;++e) {
            eligible=originalTwists[e]==0.0f && twistRates[e]==0.0f && restTwists[e]==0.0f &&
                restLengths[e]>0.0f && stretchStiffness[e]>0.0f && isfinite(restLengths[e]) &&
                isfinite(stretchStiffness[e]) && isfinite(restLengthLow[e]) && isfinite(stiffnessLow[e]);
            if (e+1u<dispatch.edgeCount) eligible=eligible && all(restCurvature[e].xy==float2(0.0f));
        }
        if (eligible) {
            exactAxial=2u;
            bool valid=true;
            for (uint i=0u;i<dispatch.nodeCount;++i) {
                const MRCompensatedScalar initial={inputPositions[nodeBase+i].x,inputPositionLow[nodeBase+i].x};
                pairedX[i]=add(initial,mul(h,{inputVelocities[nodeBase+i].x,0.0f}));
                if (axialPinIndex[i]!=0xffffffffu) {
                    const uint j=base+axialPinIndex[i]; pairedX[i]={attachments[j].targetAndCompliance.x,attachmentTargetLow[j].x};
                }
            }
            for (uint i=1u;i<end && valid;++i) {
                if (axialPinIndex[i]!=0xffffffffu) continue;
                const MRCompensatedScalar l={restLengths[i-1u],restLengthLow[i-1u]}, r={restLengths[i],restLengthLow[i]};
                const auto kl=div({stretchStiffness[i-1u],stiffnessLow[i-1u]},l);
                const auto kr=div({stretchStiffness[i],stiffnessLow[i]},r);
                const bool prev=i>1u && axialPinIndex[i-1u]==0xffffffffu;
                const bool next=i+1u<end && axialPinIndex[i+1u]==0xffffffffu;
                const auto lower=prev?neg(kl):zero;
                auto diagonal=add(div(one,mul({inverseMasses[i],inverseMassLow[i]},mul(h,h))),add(kl,kr));
                auto rhs=add(mul(kl,add(add(pairedX[i-1u],l),neg(pairedX[i]))),mul(kr,add(add(pairedX[i+1u],neg(r)),neg(pairedX[i]))));
                if (prev) { diagonal=add(diagonal,neg(mul(lower,pairedC[i-1u]))); rhs=add(rhs,neg(mul(lower,pairedD[i-1u]))); }
                valid=isfinite(diagonal.high) && isfinite(diagonal.low) && diagonal.high>0.0f && isfinite(rhs.high) && isfinite(rhs.low);
                if (valid) { pairedD[i]=div(rhs,diagonal); pairedC[i]=next?div(neg(kr),diagonal):zero;
                    valid=isfinite(pairedD[i].high)&&isfinite(pairedD[i].low)&&isfinite(pairedC[i].high)&&isfinite(pairedC[i].low); }
            }
            if (valid && end>1u) {
                for (uint i=end-1u;i>0u;--i) { if (axialPinIndex[i]!=0xffffffffu) continue;
                    const bool next=i+1u<end && axialPinIndex[i+1u]==0xffffffffu;
                    pairedD[i]=add(pairedD[i],neg(mul(pairedC[i],next?pairedD[i+1u]:zero))); }
                for (uint i=1u;i<end;++i) if (axialPinIndex[i]==0xffffffffu) pairedX[i]=add(pairedX[i],pairedD[i]);
            }
            for (uint e=0u;e<dispatch.edgeCount && valid;++e) {
                const auto d=add(pairedX[e+1u],neg(pairedX[e])); valid=isfinite(d.high)&&isfinite(d.low)&&d.high>kGeometryFloor;
            }
            for (uint i=1u;i<end && valid;++i) if (axialPinIndex[i]==0xffffffffu) {
                const MRCompensatedScalar l={restLengths[i-1u],restLengthLow[i-1u]},r={restLengths[i],restLengthLow[i]};
                const auto kl=div({stretchStiffness[i-1u],stiffnessLow[i-1u]},l),kr=div({stretchStiffness[i],stiffnessLow[i]},r);
                const auto fl=mul(kl,add(add(pairedX[i],neg(pairedX[i-1u])),neg(l)));
                const auto fr=mul(kr,add(add(pairedX[i+1u],neg(pairedX[i])),neg(r)));
                const auto v=div(add(pairedX[i],neg({inputPositions[nodeBase+i].x,inputPositionLow[nodeBase+i].x})),h);
                const auto residual=add(add(v,neg({inputVelocities[nodeBase+i].x,0.0f})),mul(mul(h,{inverseMasses[i],inverseMassLow[i]}),add(fl,neg(fr))));
                valid=isfinite(residual.high)&&isfinite(residual.low)&&abs(residual.high+residual.low)<=dispatch.dampingDerivativeTolerance.w/dispatch.gravityAndTimestep.w;
            }
            for (uint j=0u;j<dispatch.attachmentCount && valid;++j) {
                const uint node=attachments[base+j].nodeIndex; auto f=zero;
                if (node>0u) { uint e=node-1u; MRCompensatedScalar l={restLengths[e],restLengthLow[e]};
                    f=add(f,neg(mul(div({stretchStiffness[e],stiffnessLow[e]},l),add(add(pairedX[node],neg(pairedX[node-1u])),neg(l))))); }
                if (node<end) { uint e=node; MRCompensatedScalar l={restLengths[e],restLengthLow[e]};
                    f=add(f,mul(div({stretchStiffness[e],stiffnessLow[e]},l),add(add(pairedX[node+1u],neg(pairedX[node])),neg(l)))); }
                const auto v=div(add(pairedX[node],neg({inputPositions[nodeBase+node].x,inputPositionLow[nodeBase+node].x})),h);
                const auto impulse=add(mul(f,h),div(add({inputVelocities[nodeBase+node].x,0.0f},neg(v)),{inverseMasses[node],inverseMassLow[node]}));
                targetImpulses[j]=float3(impulse.high+impulse.low,0.0f,0.0f);
                valid=all(isfinite(targetImpulses[j]));
            }
            if (valid) { for (uint i=0u;i<dispatch.nodeCount;++i) positions[i].x=pairedX[i].high; completedIterations=1u; converged=1u; }
            else recordFailure(failure,MR_ROD_GPU_DID_NOT_CONVERGE);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

'''
once(needle,branch+needle)
once('if (lane == 0u && !eventSegment && dispatch.nodeCount >= 2u &&\n        dispatch.attachmentCount >= 2u && dispatch.rigidBodyCount == 0u &&\n        dispatch.flags == 0u && dispatch.dampingDerivativeTolerance.x == 0.0f &&\n        dispatch.dampingDerivativeTolerance.y == 0.0f &&\n        atomic_load_explicit(&failure, memory_order_relaxed) == MR_ROD_GPU_SUCCESS) {','if (lane == 0u && exactAxial == 0u && !eventSegment && dispatch.nodeCount >= 2u &&\n        dispatch.attachmentCount >= 2u && dispatch.rigidBodyCount == 0u &&\n        dispatch.flags == 0u && dispatch.dampingDerivativeTolerance.x == 0.0f &&\n        dispatch.dampingDerivativeTolerance.y == 0.0f &&\n        atomic_load_explicit(&failure, memory_order_relaxed) == MR_ROD_GPU_SUCCESS) {')
once('        velocities[node] =\n            projectionFailureCode == MR_ROD_GPU_SUCCESS','''        if (exactAxial == 2u && projectionFailureCode == MR_ROD_GPU_SUCCESS) {
            const auto v = mrCompensatedDivide(mrCompensatedAdd(pairedX[node],
                {-inputPositions[nodeBase+node].x,-inputPositionLow[nodeBase+node].x}),
                {dispatch.gravityAndTimestep.w,timestepLow});
            iterationPositions[node] = float3(v.high+v.low,0.0f,0.0f);
        }
        velocities[node] =
            projectionFailureCode == MR_ROD_GPU_SUCCESS''')
once('        outputPositions[nodeBase + node] =\n            float4(position, 1.0f);','''        outputPositions[nodeBase + node] =
            float4(position, 1.0f);
        outputPositionLow[nodeBase + node] = failureCode != MR_ROD_GPU_SUCCESS
            ? inputPositionLow[nodeBase + node]
            : (exactAxial == 2u ? float4(pairedX[node].low,inputPositionLow[nodeBase+node].yz,1.0f) : float4(0.0f));''')
(p/'candidate.metal').write_text(s)

#include <metal_stdlib>
#include "metalrobo/engine_types.h"
#include "metalrobo/ccd_segment_time.metalh"
using namespace metal;

// Resolve immutable actuator templates to ordinary scalar ConstraintIR rows.
// For C = strength*(D+h*K), B = strength*K*(target-q0), low-speed dry friction
// is exactly represented by two interval rows:
//   p+ = clamp(h*(B-C*v-dry), 0, h*cap)
//   p- = clamp(h*(B-C*v+dry), -h*cap, 0).
// Their sum/h is clip(soft(B-C*v,dry),+-cap). At moving source velocity the
// existing signed-friction convention uses one box row and one fixed-zero row.
// The generic quality solver jointly solves these rows and physical contacts.
kernel void mr_world_prepare_coupled_drive_ir(
    device const MRMetalWorldDispatchGPU& world [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(1)]],
    constant MRMetalWorldPassGPU& pass [[buffer(2)]],
    device const MRDofPropertiesGPU* dofs [[buffer(3)]],
    device const MRActuatorProfileGPU* profiles [[buffer(4)]],
    device const float* sourceQ [[buffer(5)]],
    device const float* sourceV [[buffer(6)]],
    device const float* targets [[buffer(7)]],
    device const float4* controllers [[buffer(8)]],
    device const MRConstraintIRBlockGPU* blocks [[buffer(9)]],
    device const MRConstraintIREndpointGPU* endpoints [[buffer(10)]],
    device MRConstraintIRRowGPU* rows [[buffer(11)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(12)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount || statuses[environment].code != MR_STEP_SUCCESS) return;
    const uint cb=environment*dispatch.constraintStride, rb=environment*dispatch.rowStride;
    const uint eb=2u*environment*dispatch.constraintStride;
    const float h=dispatch.timestepAndBias.x;
    const float4 controller=(world.flags & MR_METAL_WORLD_NATIVE_TASK)!=0u ? controllers[environment] : float4(1.0f);
    bool valid=isfinite(h) && h>0.0f && all(isfinite(controller)) && all(controller>=0.0f);
    for (uint b=0u;b<dispatch.authoredConstraintCount && valid;++b) {
        const MRConstraintIRBlockGPU block=blocks[cb+b];
        MRConstraintIRRowGPU row=rows[rb+block.rowOffset];
        if ((row.flags & MR_CONSTRAINT_IR_ROW_POSITION_DRIVE)==0u) continue;
        const bool negative=(row.flags & MR_CONSTRAINT_IR_ROW_DRIVE_NEGATIVE)!=0u;
        const MRConstraintIREndpointGPU endpoint=endpoints[eb+block.endpointOffset];
        const uint i=endpoint.objectIndex;
        if (block.dimension!=1u || i>=dispatch.nv || endpoint.linkIndex>=dispatch.nq) {valid=false;break;}
        const MRDofPropertiesGPU dof=dofs[i];
        const MRActuatorProfileGPU profile=profiles[i];
        const float q=sourceQ[environment*dispatch.qStride+endpoint.linkIndex];
        const float v=sourceV[environment*dispatch.vStride+i];
        float target=targets[pass.controlStep*world.effortStepStride+environment*world.effortEnvironmentStride+i];
        if ((dof.flags & MR_DOF_FLAG_POSITION_LIMIT)!=0u) target=clamp(target,dof.limits.x,dof.limits.y);
        const float K=controller.x*dof.drive.x, D=controller.y*dof.drive.y;
        const float B=controller.w*K*(target-q), C=controller.w*(D+h*K);
        const float dry=controller.w*dof.drive.w;
        const float speed=clamp(abs(v)/max(profile.motorAndSpeed.z,1.175494351e-38f),0.0f,1.0f);
        float cap=min(dof.limits.w,profile.transmissionAndEnvelope.z*profile.motorAndSpeed.w*(1.0f-speed));
        if ((dof.flags & MR_DOF_FLAG_EFFORT_LIMIT)!=0u && dof.limits.w>0.0f) cap=min(cap,controller.w*dof.limits.w);
        valid=isfinite(q) && isfinite(v) && isfinite(target) && isfinite(B) && isfinite(C) && C>=0.0f &&
            isfinite(dry) && dry>=0.0f && isfinite(cap) && cap>=0.0f;
        if (!valid) break;
        row.flags=0u;
        row.targetVelocity=0.0f;row.compliance=h*h;row.impulseLower=row.impulseUpper=0.0f;
        if (C>0.0f && cap>0.0f) {
            if (abs(v)>1.0e-4f) {
                if (!negative) {
                    row.targetVelocity=(B-copysign(dry,v))/C;row.compliance=h/C;
                    row.impulseLower=-h*cap;row.impulseUpper=h*cap;
                }
            } else {
                row.targetVelocity=(B+(negative ? dry : -dry))/C;row.compliance=h/C;
                row.impulseLower=negative ? -h*cap : 0.0f;row.impulseUpper=negative ? 0.0f : h*cap;
            }
        }
        valid=isfinite(row.targetVelocity) && isfinite(row.compliance) && row.compliance>0.0f &&
            isfinite(row.impulseLower) && isfinite(row.impulseUpper);
        rows[rb+block.rowOffset]=row;
    }
    if (!valid) {MRMetalWorldContactStatusGPU status=statuses[environment];status.code=MR_STEP_NONFINITE_RESULT;statuses[environment]=status;}
}

// The accepted physical motor effort is the sum of solved interval impulses/h,
// so native power/reward consumers observe the applied effort after contact.
kernel void mr_world_resolve_coupled_drive_effort(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRConstraintIRBlockGPU* sourceBlocks [[buffer(1)]],
    device const MRConstraintIREndpointGPU* sourceEndpoints [[buffer(2)]],
    device const MRConstraintIRRowGPU* sourceRows [[buffer(3)]],
    device const MRContactConstraintGPU* contacts [[buffer(4)]],
    device float* effort [[buffer(5)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(6)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment>=dispatch.environmentCount || statuses[environment].code!=MR_STEP_SUCCESS) return;
    const uint vb=environment*dispatch.vStride, cb=environment*dispatch.constraintStride;
    const float h=dispatch.timestepAndBias.x;
    for (uint i=0u;i<dispatch.nv;++i) effort[vb+i]=0.0f;
    bool valid=h>0.0f && isfinite(h);
    for (uint b=0u;b<dispatch.authoredConstraintCount && valid;++b) {
        const MRConstraintIRBlockGPU block=sourceBlocks[b];
        if ((sourceRows[block.rowOffset].flags & MR_CONSTRAINT_IR_ROW_POSITION_DRIVE)==0u) continue;
        const uint i=sourceEndpoints[block.endpointOffset].objectIndex;
        valid=i<dispatch.nv && isfinite(contacts[cb+b].impulses.x);
        if (valid) {effort[vb+i]+=contacts[cb+b].impulses.x/h;valid=isfinite(effort[vb+i]);}
    }
    if (!valid) {MRMetalWorldContactStatusGPU status=statuses[environment];status.code=MR_STEP_NONFINITE_RESULT;statuses[environment]=status;}
}

// This translation unit requires -fno-fast-math: two-sum/product residuals
// are part of the convergence certificate, not algebraic redundancies.
// Twofold arithmetic is local to the stiff bounded motor solve. The native
// publication remains FP32; no threshold or physical coefficient is changed.
inline float2 boundedTwoSum(const float a,const float b) {
    const float s=a+b, recovered=s-a;
    return float2(s,(a-(s-recovered))+(b-recovered));
}
inline float2 boundedAdd(const float2 a,const float2 b) {
    const float2 s=boundedTwoSum(a.x,b.x);
    return boundedTwoSum(s.x,(s.y+a.y)+b.y);
}
inline float2 boundedProduct(const float a,const float b) {
    const float p=a*b;
    return float2(p,fma(a,b,-p));
}
inline float2 boundedMultiply(const float2 a,const float b) {
    const float2 p=boundedProduct(a.x,b);
    return boundedTwoSum(p.x,p.y+a.y*b);
}
inline int boundedSign(const float2 a) {
    return a.x>0.0f || (a.x==0.0f && a.y>0.0f) ? 1 :
           a.x<0.0f || (a.x==0.0f && a.y<0.0f) ? -1 : 0;
}
inline float2 boundedSoft(const float2 a,const float friction) {
    const int sign=boundedSign(a);
    const float2 v=boundedAdd(a,float2(-float(sign)*friction,0.0f));
    return boundedSign(v)==sign ? v : float2(0.0f);
}
inline float2 boundedBox(const float2 a,const float cap) {
    if (boundedSign(boundedAdd(a,float2(-cap,0.0f)))>0) return float2(cap,0.0f);
    if (boundedSign(boundedAdd(a,float2(cap,0.0f)))<0) return float2(-cap,0.0f);
    return a;
}
inline float2 boundedQuotient(const float2 numerator,const float2 denominator) {
    const float q=numerator.x/denominator.x;
    const float2 remainder=boundedAdd(numerator,-boundedMultiply(denominator,q));
    return boundedTwoSum(q,(remainder.x+remainder.y)/denominator.x);
}

// Smooth torque-limited motor step. Contact impulses are solved later with the
// physical mass factor; they are not folded into the motor's damping inertia.
// The physical ABA acceleration and source-state factor remain immutable across
// remaining-time prediction and selected-time rematerialization.
kernel void mr_world_bounded_drive_prediction(
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(1)]],
    constant MRMetalWorldPassGPU& pass [[buffer(2)]],
    device const MRDofPropertiesGPU* dofs [[buffer(3)]],
    device const MRActuatorProfileGPU* profiles [[buffer(4)]],
    device const float* sourceQ [[buffer(5)]],
    device const float* sourceV [[buffer(6)]],
    device const float* freeAcceleration [[buffer(7)]],
    device const float* factor [[buffer(8)]],
    device const float* targets [[buffer(9)]],
    device const float4* controllerParameters [[buffer(10)]],
    device const MRArticulatedOperatorStatusGPU* operatorStatuses [[buffer(11)]],
    device MRMetalWorldContactStatusGPU* contactStatuses [[buffer(12)]],
    device MRMetalWorldStatusGPU* worldStatuses [[buffer(13)]],
    device const float* baseEffort [[buffer(14)]],
    device float* effectiveEffort [[buffer(15)]],
    device float* acceleration [[buffer(16)]],
    device const MRCCDEventStateGPU* eventStates [[buffer(17)]],
    constant uint& mode [[buffer(18)]],
    device MRBoundedDriveFailureGPU* failures [[buffer(19)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_threadgroup]]
) {
    constexpr uint capacity = MR_ARTICULATED_ABA_MAX_DOFS;
    threadgroup float inverse[capacity * capacity];
    if (environment >= dispatch.environmentCount ||
        contactStatuses[environment].code != MR_STEP_SUCCESS ||
        worldStatuses[environment].code != MR_STEP_SUCCESS) {
        return;
    }
    const float h = mode == MR_CCD_SEGMENT_FULL_MICROSTEP
        ? dispatch.timestepAndBias.x
        : mrCCDSegmentDuration(eventStates[environment], mode);
    if (!(h > 0.0f)) {
        return; // A finished CCD environment must retain its last motor sample.
    }
    const uint n = dispatch.nv;
    const uint vBase = environment * dispatch.vStride;
    const uint qBase = environment * dispatch.qStride;
    const uint fBase = environment * dispatch.factorStride;
    // Every lane makes the same validity decision before collective work.
    uint invalid = n == 0u || n > capacity || !isfinite(h);
    for (uint owner = 0u; owner < dispatch.articulationCount; ++owner) {
        invalid |= operatorStatuses[owner * dispatch.environmentCount + environment].code != MR_ARTICULATED_OPERATOR_SUCCESS;
    }
    for (uint i = 0u; i < n && i < capacity; ++i) {
        invalid |= !(factor[fBase + i*n + i] > 0.0f) ||
            !isfinite(factor[fBase + i*n + i]) ||
            !isfinite(freeAcceleration[vBase+i]);
    }
    if (invalid != 0u) {
        if (lane == 0u) {
            MRMetalWorldContactStatusGPU status = contactStatuses[environment];
            status.code = MR_STEP_FACTORIZATION_FAILED;
            contactStatuses[environment] = status;
        }
        return;
    }
    // Independent triangular solves compute columns of physical M^-1. Each
    // lane owns one column, including all articulated cross-joint responses.
    for (uint column=lane; column<n; column+=MR_SIMD_WIDTH) {
        float y[capacity], x[capacity];
        for (uint i=0u; i<n; ++i) {
            float value = i == column ? 1.0f : 0.0f;
            for (uint j=0u; j<i; ++j) value -= factor[fBase+i*n+j]*y[j];
            y[i] = value/factor[fBase+i*n+i];
        }
        for (uint back=0u; back<n; ++back) {
            const uint i=n-1u-back;
            float value=y[i];
            for (uint j=i+1u; j<n; ++j) value -= factor[fBase+j*n+i]*x[j];
            x[i]=value/factor[fBase+i*n+i];
            inverse[i*capacity+column]=x[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float4 controller = (worldDispatch.flags & MR_METAL_WORLD_NATIVE_TASK) != 0u
        ? controllerParameters[environment] : float4(1.0f);
    threadgroup float b[capacity],c[capacity],cap[capacity],friction[capacity],a[capacity],aLow[capacity];
    // Replicated coordinate iterates avoid a threadgroup barrier on every visit.
    // All SIMD lanes contribute to each compensated inverse-response dot product.
    float2 torque[capacity];
    for (uint i=0u;i<n;++i) torque[i]=float2(0.0f);
    bool valid=all(isfinite(controller)) && all(controller>=0.0f);
    for (uint i=lane;i<n;i+=MR_SIMD_WIDTH) {
        const MRDofPropertiesGPU dof=dofs[i];
        const MRActuatorProfileGPU profile=profiles[i];
        const float v=sourceV[vBase+i];
        b[i]=0.0f;c[i]=0.0f;cap[i]=0.0f;friction[i]=0.0f;a[i]=freeAcceleration[vBase+i];aLow[i]=0.0f;
        if ((dof.flags & MR_DOF_FLAG_DRIVE)==0u || dof.qIndex>=dispatch.nq) continue;
        float target=targets[pass.controlStep*worldDispatch.effortStepStride+
            environment*worldDispatch.effortEnvironmentStride+i];
        if ((dof.flags & MR_DOF_FLAG_POSITION_LIMIT)!=0u)
            target=clamp(target,dof.limits.x,dof.limits.y);
        const float k=controller.x*dof.drive.x,d=controller.y*dof.drive.y;
        b[i]=controller.w*(k*(target-sourceQ[qBase+dof.qIndex]-h*v)-d*v);
        c[i]=controller.w*h*fma(h,k,d);
        const float dry=controller.w*dof.drive.w;
        if (abs(v)>1.0e-4f) b[i]-=copysign(dry,v);else friction[i]=dry;
        const float speed=clamp(abs(v)/max(profile.motorAndSpeed.z,1.175494351e-38f),0.0f,1.0f);
        cap[i]=min(dof.limits.w,profile.transmissionAndEnvelope.z*profile.motorAndSpeed.w*(1.0f-speed));
        if ((dof.flags & MR_DOF_FLAG_EFFORT_LIMIT)!=0u && dof.limits.w>0.0f)
            cap[i]=min(cap[i],controller.w*dof.limits.w);
        valid &= isfinite(b[i]) && isfinite(c[i]) && c[i]>=0.0f &&
            isfinite(cap[i]) && cap[i]>=0.0f && isfinite(friction[i]) && friction[i]>=0.0f;
        for (uint j=0u;j<n;++j) valid &= isfinite(inverse[i*capacity+j]);
    }
    valid=simd_all(valid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximumScaledResidual=INFINITY;
    uint completed=0u;
    for (uint sweep=0u;valid && sweep<256u;++sweep) {
        for (uint visit=0u;visit<2u*n;++visit) {
            const uint i=visit<n ? visit : 2u*n-1u-visit;
            float2 excluding=lane==0u ? float2(freeAcceleration[vBase+i],0.0f) : float2(0.0f);
            for (uint j=lane;j<n;j+=MR_SIMD_WIDTH) if (j!=i)
                excluding=boundedAdd(excluding,boundedMultiply(torque[j],inverse[i*capacity+j]));
            for (uint offset=MR_SIMD_WIDTH/2u;offset>0u;offset/=2u)
                excluding=boundedAdd(excluding,simd_shuffle_down(excluding,offset));
            float2 next=float2(0.0f);
            if (lane==0u) {
                const float2 rhs=boundedAdd(float2(b[i],0.0f),-boundedMultiply(excluding,c[i]));
                const float2 denominator=boundedAdd(float2(1.0f,0.0f),boundedProduct(c[i],inverse[i*capacity+i]));
                next=boundedBox(boundedQuotient(boundedSoft(rhs,friction[i]),denominator),cap[i]);
            }
            torque[i]=simd_broadcast_first(next);
        }
        float localResidual=0.0f;
        for (uint i=lane;i<n;i+=MR_SIMD_WIDTH) {
            float2 value=float2(freeAcceleration[vBase+i],0.0f);
            for (uint j=0u;j<n;++j)
                value=boundedAdd(value,boundedMultiply(torque[j],inverse[i*capacity+j]));
            a[i]=value.x;aLow[i]=value.y;
            const float2 rhs=boundedAdd(float2(b[i],0.0f),-boundedMultiply(value,c[i]));
            const float2 expected=boundedBox(boundedSoft(rhs,friction[i]),cap[i]);
            const float2 difference=boundedAdd(expected,-torque[i]);
            const float scaled=abs(difference.x+difference.y)/max(cap[i],1.0e-6f);
            valid &= isfinite(a[i]) && isfinite(aLow[i]) && isfinite(scaled);
            localResidual=max(localResidual,scaled);
        }
        valid=simd_all(valid);maximumScaledResidual=simd_max(localResidual);
        completed=sweep+1u;
        if (valid && maximumScaledResidual<=4.0f*1.1920928955078125e-7f) break;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!valid || maximumScaledResidual>4.0f*1.1920928955078125e-7f) {
        if (lane!=0u) return;
        MRMetalWorldContactStatusGPU contact=contactStatuses[environment];
        contact.code=valid ? MR_STEP_DID_NOT_CONVERGE : MR_STEP_NONFINITE_RESULT;
        contactStatuses[environment]=contact;
        MRMetalWorldStatusGPU status=worldStatuses[environment];
        status.code=contact.code;status.failingSubstep=pass.physicsSubstep;
        status.diagnostics=float4(maximumScaledResidual,float(completed),h,0.0f);
        worldStatuses[environment]=status;
        if ((worldDispatch.flags & MR_METAL_WORLD_CAPTURE_BOUNDED_DRIVE_FAILURE)!=0u) {
            device MRBoundedDriveFailureGPU& saved=failures[environment];saved.valid=0u;
            saved.environment=environment;saved.controlStep=pass.controlStep;saved.physicsSubstep=pass.physicsSubstep;
            saved.nv=n;saved.mode=mode;saved.sweeps=completed;saved.code=contact.code;
            saved.diagnostics=float4(h,maximumScaledResidual,0.0f,0.0f);
            for (uint i=0u;i<capacity;++i) {
                for (uint j=0u;j<capacity;++j) {
                    saved.factor[i*capacity+j]=i<n && j<n ? factor[fBase+i*n+j] : 0.0f;
                    saved.inverse[i*capacity+j]=i<n && j<n ? inverse[i*capacity+j] : 0.0f;
                }
                saved.freeAcceleration[i]=i<n ? freeAcceleration[vBase+i] : 0.0f;
                saved.b[i]=i<n ? b[i] : 0.0f;saved.c[i]=i<n ? c[i] : 0.0f;
                saved.cap[i]=i<n ? cap[i] : 0.0f;saved.friction[i]=i<n ? friction[i] : 0.0f;
                saved.torque[i]=i<n ? torque[i].x : 0.0f;saved.acceleration[i]=i<n ? a[i] : 0.0f;
                saved.torqueLow[i]=i<n ? torque[i].y : 0.0f;
                saved.accelerationLow[i]=i<n ? aLow[i] : 0.0f;
            }
            saved.valid=0x42444632u;
        }
        return;
    }
    for (uint i=lane;i<n;i+=MR_SIMD_WIDTH) {
        acceleration[vBase+i]=a[i]+aLow[i];effectiveEffort[vBase+i]=baseEffort[vBase+i]+(torque[i].x+torque[i].y);
    }
}

// Numerical inspection only; never called by the world graph.
kernel void mr_bounded_drive_certificate_probe(
    device const MRBoundedDriveFailureGPU& snapshot [[buffer(0)]],
    device float4* output [[buffer(1)]],
    const uint i [[thread_position_in_grid]]
) {
    if (i>=snapshot.nv) return;
    float2 value=float2(snapshot.freeAcceleration[i],0.0f);
    for (uint j=0u;j<snapshot.nv;++j)
        value=boundedAdd(value,boundedMultiply(float2(snapshot.torque[j],snapshot.torqueLow[j]),snapshot.inverse[i*MR_ARTICULATED_ABA_MAX_DOFS+j]));
    const float2 rhs=boundedAdd(float2(snapshot.b[i],0.0f),-boundedMultiply(value,snapshot.c[i]));
    const float2 expected=boundedBox(boundedSoft(rhs,snapshot.friction[i]),snapshot.cap[i]);
    const float2 difference=boundedAdd(expected,-float2(snapshot.torque[i],snapshot.torqueLow[i]));
    output[i]=float4(value.x,value.y,abs(difference.x+difference.y)/max(snapshot.cap[i],1.0e-6f),expected.x+expected.y);
}

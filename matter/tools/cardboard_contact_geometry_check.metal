#include "../src/matter_kernels.metal"
namespace numi_matter_metal {
kernel void nm_contact_scaled_geometry_replay(device float4* output [[buffer(0)]], uint global [[thread_position_in_grid]]) {
    if (global>=130u) return;
    if (global>=125u && global<130u) {
        const uint test=global-125u;
        const uint base=global*18u;
        for(uint j=0u;j<18u;++j) output[base+j]=float4(0.0f);
        const float3 origin=float3(8192.0f,-4096.0f,2048.0f);
        const float3 localRest=float3(0.0048828125f,0.0f,0.0f);
        const float thickness=1.0e-4f;
        const float stiffness=1.8e13f;
        const float dt=0.0009765625f;
        const float3 correction=float3(1.490116119384765625e-8f,0.0f,0.0f);
        const float scaleValues[4]={0.0008f,0.0016f,0.0032f,0.008f};
        const float3 shift=float3(1024.0f,512.0f,-256.0f);
        const float scale=test<4u?scaleValues[test]:scaleValues[3];
        const float3 localDisplacement[4]={
            float3(0.00022f,0.25f*scale,0.25f*scale),
            float3(0.0002f,0.0f,0.0f),
            float3(0.0002f,scale,0.0f),
            float3(0.0002f,0.0f,scale)};
        NMFEMNodeStateGPU originalStates[4]={};
        NMFEMNodeStateGPU translatedStates[4]={};
        MRCompensatedPositionGPU originalPosition[4];
        MRCompensatedPositionGPU translatedPosition[4];
        for(uint i=0u;i<4u;++i) {
            const float3 rest=origin+localRest;
            const float3 u=localDisplacement[i];
            originalStates[i].restAndFixed=float4(rest,0.0f);
            originalStates[i].positionAndMass=float4(rest+u,1.0f);
            originalStates[i].velocityAndInverseMass=float4(0.0f,0.0f,0.0f,1.0f);
            originalStates[i].referenceDisplacementAndMode=float4(u,1.0f);
            translatedStates[i]=originalStates[i];
            translatedStates[i].restAndFixed.xyz+=shift;
            translatedStates[i].positionAndMass.xyz+=shift;
            originalPosition[i]=femContactPositionRelativeExpansion(
                originalStates[i],origin);
            translatedPosition[i]=femContactPositionRelativeExpansion(
                translatedStates[i],origin+shift);
        }
        NMFEMNodeStateGPU trialPoint=originalStates[0];
        trialPoint=femTrialNodeAtAlpha(originalStates[0],trialPoint,
            correction,1.0f,dt,true);
        NMFEMNodeStateGPU translatedTrialPoint=translatedStates[0];
        translatedTrialPoint=femTrialNodeAtAlpha(translatedStates[0],
            translatedTrialPoint,correction,1.0f,dt,true);
        MRCompensatedPositionGPU trialPosition[4];
        MRCompensatedPositionGPU translatedTrialPosition[4];
        trialPosition[0]=femContactPositionRelativeExpansion(trialPoint,origin);
        translatedTrialPosition[0]=femContactPositionRelativeExpansion(
            translatedTrialPoint,origin+shift);
        for(uint i=1u;i<4u;++i) {
            trialPosition[i]=originalPosition[i];
            translatedTrialPosition[i]=translatedPosition[i];
        }
        NMDeformableContactGPU contact{};
        contact.identity.z=NM_DEFORMABLE_CONTACT_VERTEX_TRIANGLE;
        contact.barrier.z=thickness;
        contact.mollifier.w=stiffness;
        float3 baseBary,baseClosest,trialBary,trialClosest;
        MRCompensatedPositionGPU baseRelative,trialRelative;
        float baseWitnessGap,trialWitnessGap;
        const bool baseWitness=contactCompensatedVertexTriangleWitness(
            originalPosition[0],originalPosition[1],originalPosition[2],
            originalPosition[3],baseBary,baseClosest,baseRelative,baseWitnessGap);
        const bool trialWitness=contactCompensatedVertexTriangleWitness(
            trialPosition[0],trialPosition[1],trialPosition[2],
            trialPosition[3],trialBary,trialClosest,trialRelative,trialWitnessGap);
        float baseEnergy,trialEnergy,baseEnergyGap,trialEnergyGap;
        const bool baseEnergyValid=deformableBarrierEnergyAt(
            contact,originalPosition,baseEnergy,baseEnergyGap);
        const bool trialEnergyValid=deformableBarrierEnergyAt(
            contact,trialPosition,trialEnergy,trialEnergyGap);
        float3 baseProjection[4],trialProjection[4];
        for(uint i=0u;i<4u;++i) {
            baseProjection[i]=femContactPositionRelativeProjection(originalPosition[i]);
            trialProjection[i]=femContactPositionRelativeProjection(trialPosition[i]);
        }
        float legacyBaseEnergy,legacyTrialEnergy,legacyBaseGap,legacyTrialGap;
        const bool legacyBaseValid=deformableBarrierEnergyAt(
            contact,baseProjection,legacyBaseEnergy,legacyBaseGap);
        const bool legacyTrialValid=deformableBarrierEnergyAt(
            contact,trialProjection,legacyTrialEnergy,legacyTrialGap);
        output[base]=float4(float(test),float(baseWitness&&trialWitness),
            float(baseEnergyValid&&legacyBaseValid),
            float(trialEnergyValid&&legacyTrialValid));
        if(test<4u) {
            output[base+1u]=originalPosition[0].high;
            output[base+2u]=originalPosition[0].low;
            output[base+3u]=trialPosition[0].high;
            output[base+4u]=trialPosition[0].low;
            output[base+5u]=originalPosition[1].high;
            output[base+6u]=originalPosition[1].low;
            output[base+7u]=originalPosition[2].high;
            output[base+8u]=originalPosition[2].low;
            output[base+9u]=originalPosition[3].high;
            output[base+10u]=originalPosition[3].low;
            output[base+11u]=float4(baseBary,baseWitnessGap);
            output[base+12u]=float4(trialBary,trialWitnessGap);
            output[base+13u]=float4(baseEnergy,trialEnergy,
                baseEnergyGap,trialEnergyGap);
            output[base+14u]=float4(legacyBaseEnergy,legacyTrialEnergy,
                legacyBaseGap,legacyTrialGap);
            output[base+15u]=float4(baseProjection[0],0.0f);
            output[base+16u]=float4(trialProjection[0],0.0f);
            output[base+17u]=float4(trialPoint.referenceDisplacementAndMode.xyz,
                trialPoint.referenceDisplacementAndMode.w);
        } else {
            for(uint i=0u;i<5u;++i) {
                const uint source=i<2u?0u:i-1u;
                const MRCompensatedPositionGPU left=i==1u
                    ? translatedTrialPosition[0] : translatedPosition[source];
                const MRCompensatedPositionGPU right=i==1u
                    ? trialPosition[0] : originalPosition[source];
                const MRCompensatedPositionGPU delta=mrCompensatedVectorAdd(
                    left,mrCompensatedVectorNegate(right));
                const uint slot=1u+2u*i;
                output[base+slot]=delta.high;
                output[base+slot+1u]=delta.low;
            }
            float3 translatedBaseBary,translatedBaseClosest;
            float3 translatedTrialBary,translatedTrialClosest;
            MRCompensatedPositionGPU translatedBaseRelative,translatedTrialRelative;
            float translatedBaseWitnessGap,translatedTrialWitnessGap;
            const bool translatedBaseWitness=contactCompensatedVertexTriangleWitness(
                translatedPosition[0],translatedPosition[1],translatedPosition[2],
                translatedPosition[3],translatedBaseBary,translatedBaseClosest,
                translatedBaseRelative,translatedBaseWitnessGap);
            const bool translatedTrialWitness=contactCompensatedVertexTriangleWitness(
                translatedTrialPosition[0],translatedTrialPosition[1],
                translatedTrialPosition[2],translatedTrialPosition[3],
                translatedTrialBary,translatedTrialClosest,translatedTrialRelative,
                translatedTrialWitnessGap);
            float translatedBaseEnergy,translatedTrialEnergy;
            float translatedBaseEnergyGap,translatedTrialEnergyGap;
            const bool translatedBaseEnergyValid=deformableBarrierEnergyAt(
                contact,translatedPosition,translatedBaseEnergy,
                translatedBaseEnergyGap);
            const bool translatedTrialEnergyValid=deformableBarrierEnergyAt(
                contact,translatedTrialPosition,translatedTrialEnergy,
                translatedTrialEnergyGap);
            output[base+11u]=float4(baseEnergyGap,trialEnergyGap,0.0f,0.0f);
            output[base+12u]=float4(translatedBaseEnergyGap,
                translatedTrialEnergyGap,0.0f,0.0f);
            output[base+13u]=float4(baseBary,baseWitnessGap);
            output[base+14u]=float4(translatedBaseBary,translatedBaseWitnessGap);
            output[base+15u]=float4(trialBary,trialWitnessGap);
            output[base+16u]=float4(translatedTrialBary,translatedTrialWitnessGap);
            output[base+17u]=float4(baseEnergy,trialEnergy,
                translatedBaseEnergy,translatedTrialEnergy);
            const float3 baseProjection=femContactPositionRelativeProjection(
                originalPosition[0]);
            const float3 trialProjection=femContactPositionRelativeProjection(
                trialPosition[0]);
            const float3 translatedBaseProjection=
                femContactPositionRelativeProjection(translatedPosition[0]);
            const float3 translatedTrialProjection=
                femContactPositionRelativeProjection(translatedTrialPosition[0]);
            output[base+0u].y=float(baseWitness&&trialWitness&&
                translatedBaseWitness&&translatedTrialWitness);
            output[base+0u].z=float(baseEnergyValid&&trialEnergyValid&&
                translatedBaseEnergyValid&&translatedTrialEnergyValid);
            output[base+0u].w=float(
                trialPoint.referenceDisplacementAndMode.w==1.0f &&
                all(baseProjection==trialProjection) &&
                all(translatedBaseProjection==translatedTrialProjection));
        }
        return;
    }
    if (global>=120u && global<125u) {
        const uint test=global-120u;
        const uint base=global*18u;
        for(uint j=0u;j<18u;++j)output[base+j]=float4(0.0f);
        const float3 origin=float3(8192.0f,-4096.0f,2048.0f);
        const float3 correction=float3(0.0625f,-0.03125f,0.125f);
        const float alpha=0.5f;
        const float dt=0.0009765625f;
        if(test==0u) {
            const float3 rest=origin+float3(0.03125f,0.046875f,0.0078125f);
            const float3 u=float3(0.00001f,-0.00002f,0.00003f);
            NMFEMNodeStateGPU state{};
            state.restAndFixed=float4(rest,0.0f);
            state.positionAndMass=float4(rest+u,1.0f);
            state.referenceDisplacementAndMode=float4(u,1.0f);
            const float3 relative=femContactPositionRelativeTo(state,origin);
            const float3 shift=float3(16384.0f,8192.0f,-4096.0f);
            NMFEMNodeStateGPU shifted=state;
            shifted.restAndFixed.xyz+=shift;
            shifted.positionAndMass.xyz+=shift;
            const float3 shiftedOrigin=origin+shift;
            const float3 translatedRelative=femContactPositionRelativeTo(
                shifted,shiftedOrigin);
            output[base]=float4(float(test),1.0f,0.0f,0.0f);
            output[base+1u]=float4(origin,0.0f);
            output[base+2u]=float4(rest,0.0f);
            output[base+3u]=float4(u,0.0f);
            output[base+4u]=float4(state.positionAndMass.xyz,0.0f);
            output[base+5u]=float4(relative,0.0f);
            output[base+6u]=float4(shift,0.0f);
            output[base+7u]=float4(shiftedOrigin,0.0f);
            output[base+8u]=float4(shifted.restAndFixed.xyz,0.0f);
            output[base+9u]=float4(shifted.referenceDisplacementAndMode.xyz,0.0f);
            output[base+10u]=float4(shifted.positionAndMass.xyz,0.0f);
            output[base+11u]=float4(translatedRelative,0.0f);
            output[base+12u]=float4(state.positionAndMass.xyz-origin,0.0f);
            output[base+13u]=float4(shifted.positionAndMass.xyz-shiftedOrigin,0.0f);
            return;
        }
        if(test==1u || test==3u) {
            const float3 rest=origin+float3(0.00390625f,0.005859375f,0.015625f);
            NMFEMNodeStateGPU accepted{};
            accepted.positionAndMass=float4(rest,1.0f);
            accepted.velocityAndInverseMass=float4(0.0f,0.0f,0.0f,1.0f);
            accepted.restAndFixed=float4(rest,0.0f);
            accepted.deltaVelocity=float4(0.0f);
            accepted.referenceDisplacementAndMode=float4(0.0f,0.0f,0.0f,
                test==1u?1.0f:0.0f);
            NMFEMNodeStateGPU next=accepted;
            const NMFEMNodeStateGPU trial=femTrialNodeAtAlpha(
                accepted,next,correction,alpha,dt,true);
            const float3 relative=test==1u
                ? femContactPositionRelativeTo(trial,origin) : float3(0.0f);
            output[base]=float4(float(test),alpha,dt,
                test==1u?1.0f:0.0f);
            output[base+1u]=accepted.restAndFixed;
            output[base+2u]=float4(origin,0.0f);
            output[base+3u]=accepted.positionAndMass;
            output[base+4u]=accepted.velocityAndInverseMass;
            output[base+5u]=float4(accepted.referenceDisplacementAndMode.xyz,
                accepted.referenceDisplacementAndMode.w);
            output[base+6u]=float4(correction,0.0f);
            output[base+7u]=trial.positionAndMass;
            output[base+8u]=trial.velocityAndInverseMass;
            output[base+9u]=float4(trial.referenceDisplacementAndMode.xyz,
                trial.referenceDisplacementAndMode.w);
            output[base+10u]=float4(relative,0.0f);
            output[base+11u]=accepted.positionAndMass;
            output[base+12u]=accepted.velocityAndInverseMass;
            output[base+13u]=float4(accepted.referenceDisplacementAndMode.xyz,
                accepted.referenceDisplacementAndMode.w);
            output[base+14u]=trial.deltaVelocity;
            output[base+15u]=float4(trial.restAndFixed.xyz,
                trial.restAndFixed.w);
            output[base+16u]=accepted.deltaVelocity;
            output[base+17u]=float4(dt,alpha,trial.referenceDisplacementAndMode.w,
                1.0f);
            return;
        }
        if(test==2u) {
            const float3 rest=origin+float3(0.00390625f,0.005859375f,0.015625f);
            NMFEMNodeStateGPU state{};
            state.positionAndMass=float4(rest+float3(0.00001f,0.00002f,0.00003f),1.0f);
            state.velocityAndInverseMass=float4(0.01f,-0.02f,0.03f,1.0f);
            state.restAndFixed=float4(rest,0.0f);
            state.deltaVelocity=float4(0.001f,-0.002f,0.003f,0.0f);
            state.referenceDisplacementAndMode=float4(0.00001f,0.00002f,0.00003f,1.0f);
            NMFEMNodeStateGPU fixed=state;
            fixed.restAndFixed.w=1.0f;
            const NMFEMNodeStateGPU fixedTrial=femTrialNodeAtAlpha(
                fixed,fixed,correction,1.0f,dt,true);
            const NMFEMNodeStateGPU zeroTrial=femTrialNodeAtAlpha(
                state,state,correction,0.0f,dt,true);
            output[base]=float4(float(test),1.0f,0.0f,0.0f);
            output[base+1u]=fixed.positionAndMass;
            output[base+2u]=fixedTrial.positionAndMass;
            output[base+3u]=fixed.velocityAndInverseMass;
            output[base+4u]=fixedTrial.velocityAndInverseMass;
            output[base+5u]=fixed.deltaVelocity;
            output[base+6u]=fixedTrial.deltaVelocity;
            output[base+7u]=fixed.referenceDisplacementAndMode;
            output[base+8u]=fixedTrial.referenceDisplacementAndMode;
            output[base+9u]=state.positionAndMass;
            output[base+10u]=zeroTrial.positionAndMass;
            output[base+11u]=state.velocityAndInverseMass;
            output[base+12u]=zeroTrial.velocityAndInverseMass;
            output[base+13u]=state.deltaVelocity;
            output[base+14u]=zeroTrial.deltaVelocity;
            output[base+15u]=state.referenceDisplacementAndMode;
            output[base+16u]=zeroTrial.referenceDisplacementAndMode;
            const float fixedRestUnchanged=all(
                fixedTrial.restAndFixed==fixed.restAndFixed)?1.0f:0.0f;
            const float zeroRestUnchanged=all(
                zeroTrial.restAndFixed==state.restAndFixed)?1.0f:0.0f;
            output[base+17u]=float4(fixedRestUnchanged,zeroRestUnchanged,
                1.0f,0.0f);
            return;
        }
        const float3 rests[4]={
            origin+float3(0.00390625f,0.005859375f,0.015625f),
            origin,
            origin+float3(0.01953125f,0.0f,0.0f),
            origin+float3(0.0f,0.01953125f,0.0f)};
        NMFEMNodeStateGPU states[4]={};
        float3 baseRelative[4],trialRelative[4],roundedWorldRelative[4];
        for(uint i=0u;i<4u;++i) {
            states[i].positionAndMass=float4(rests[i],1.0f);
            states[i].velocityAndInverseMass=float4(0.0f,0.0f,0.0f,1.0f);
            states[i].restAndFixed=float4(rests[i],0.0f);
            states[i].referenceDisplacementAndMode=float4(0.0f,0.0f,0.0f,1.0f);
            baseRelative[i]=femContactPositionRelativeTo(states[i],origin);
            trialRelative[i]=baseRelative[i];
            roundedWorldRelative[i]=states[i].positionAndMass.xyz-origin;
        }
        const float slopeDt=0.0009765625f;
        const NMFEMNodeStateGPU trialPoint=femTrialNodeAtAlpha(
            states[0],states[0],float3(0.0f,0.0f,0.0625f),1.0f,slopeDt,true);
        trialRelative[0]=femContactPositionRelativeTo(trialPoint,origin);
        roundedWorldRelative[0]=trialPoint.positionAndMass.xyz-origin;
        NMDeformableContactGPU contact{};
        contact.identity.z=NM_DEFORMABLE_CONTACT_VERTEX_TRIANGLE;
        contact.barrier.z=0.02f;
        float3 barycentric,closest;
        MRCompensatedPositionGPU relative;
        float baseSeparation;
        const bool baseWitness=contactCompensatedVertexTriangleWitness(
            baseRelative[0],baseRelative[1],baseRelative[2],baseRelative[3],
            barycentric,closest,relative,baseSeparation);
        const float4 response=ipcBarrierResponse(baseSeparation,contact.barrier.z,
            1.0f,slopeDt,0.0f,false,0.0f,0.0f);
        contact.mollifier.w=response.w;
        float baseEnergy,baseEnergyGap,trialEnergy,trialGap,worldEnergy,worldGap;
        const bool baseEnergyValid=deformableBarrierEnergyAt(
            contact,baseRelative,baseEnergy,baseEnergyGap);
        const bool trialEnergyValid=deformableBarrierEnergyAt(
            contact,trialRelative,trialEnergy,trialGap);
        const bool worldEnergyValid=deformableBarrierEnergyAt(
            contact,roundedWorldRelative,worldEnergy,worldGap);
        output[base]=float4(float(test),baseWitness?1.0f:0.0f,
            baseEnergyValid?1.0f:0.0f,trialEnergyValid&&worldEnergyValid?1.0f:0.0f);
        output[base+1u]=float4(origin,0.0f);
        for(uint i=0u;i<4u;++i)output[base+2u+i]=float4(baseRelative[i],0.0f);
        for(uint i=0u;i<4u;++i)output[base+6u+i]=float4(trialRelative[i],0.0f);
        for(uint i=0u;i<4u;++i)output[base+10u+i]=float4(roundedWorldRelative[i],0.0f);
        output[base+14u]=float4(baseEnergyGap,trialGap,worldGap,baseEnergy);
        output[base+15u]=float4(trialEnergy,worldEnergy,contact.barrier.z,
            response.w);
        output[base+16u]=float4(slopeDt,1.0f,0.0625f,
            trialPoint.referenceDisplacementAndMode.z);
        output[base+17u]=float4(trialPoint.positionAndMass.z,
            states[0].positionAndMass.z,trialPoint.referenceDisplacementAndMode.w,
            float(baseWitness&&baseEnergyValid&&trialEnergyValid&&worldEnergyValid));
        return;
    }
    if (global>=112u) {
        const uint test=global-112u;
        float3 origin=float3(0.0f);
        float3 point=float3(0.004f,0.006f,0.008f);
        float3 a=float3(0.0f), b=float3(0.02f,0.0f,0.0f),
               c=float3(0.0f,0.02f,0.0f);
        float thickness=0.02f;
        const float dt=0.01f;
        if (test==3u) {
            b=float3(0.02f,0.0f,0.0f);
            c=float3(0.01f,0.0001f,0.0f);
            point=float3(0.012f,0.00004f,0.008f);
        }
        if (test>=6u) {
            origin=float3(8192.0f,-4096.0f,2048.0f);
            point=origin+float3(0.00390625f,0.005859375f,0.0078125f);
            a=origin;
            b=origin+float3(0.01953125f,0.0f,0.0f);
            c=origin+float3(0.0f,0.01953125f,0.0f);
            thickness=0.02f;
        }
        const float3 positions[4]={point,a,b,c};
        float3 barycentric,closest;
        MRCompensatedPositionGPU relative;
        float separation;
        const bool witnessValid=contactCompensatedVertexTriangleWitness(
            point,a,b,c,barycentric,closest,relative,separation);
        float3 firstGradient,secondGradient;
        const float mollifier=ipcFeatureMollifierDifferential(
            b-a,c-a,firstGradient,secondGradient);
        const float inverseResponse=1.0f;
        const float closingSpeed=0.0f;
        const float4 baseResponse=ipcBarrierResponse(
            separation,thickness,inverseResponse,dt,closingSpeed,false,0.0f,0.0f);
        const float baseEnergy=ipcBarrierEnergy(
            separation,thickness,baseResponse.w,false,0.0f);
        NMDeformableContactGPU contact{};
        contact.nodes=uint4(0u,1u,2u,3u);
        contact.extraNodes=uint4(NM_INVALID_INDEX);
        contact.identity.z=NM_DEFORMABLE_CONTACT_VERTEX_TRIANGLE;
        contact.weights=float4(1.0f,-barycentric.x,-barycentric.y,-barycentric.z);
        contact.normalAndTOI=float4(separation>0.0f
            ? (relative.high.xyz+relative.low.xyz)/separation : float3(0.0f),0.0f);
        contact.barrier=float4(baseResponse.x*mollifier,
            baseResponse.y*mollifier,thickness,baseResponse.w*mollifier);
        contact.mollifier=float4(mollifier,dt*baseEnergy,
            dt*dt*baseResponse.y,baseResponse.w);
        contact.mollifierGradientRow0=float4(0.0f);
        contact.mollifierGradientRow1=float4(-firstGradient-secondGradient,0.0f);
        contact.mollifierGradientRow2=float4(firstGradient,0.0f);
        contact.mollifierGradientRow3=float4(secondGradient,0.0f);
        float3 analytic[6]={float3(0.0f),float3(0.0f),float3(0.0f),
                            float3(0.0f),float3(0.0f),float3(0.0f)};
        if (test==0u) analytic[0]=float3(0.0f,0.0f,0.0002f);
        if (test==1u) analytic[0]=float3(0.0f,0.0f,-0.0002f);
        if (test==2u) analytic[0]=float3(0.0001f,-0.00015f,0.00008f);
        if (test==3u) analytic[3]=float3(0.0f,0.00001f,0.0f);
        if (test==4u) {
            const float3 translation=float3(0.0002f,-0.0001f,0.0003f);
            for(uint i=0u;i<4u;++i) analytic[i]=translation;
        }
        if (test==6u || test==7u) analytic[0]=float3(0.0f,0.0f,0.00001f);
        float3 rounded[6]={float3(0.0f),float3(0.0f),float3(0.0f),
                           float3(0.0f),float3(0.0f),float3(0.0f)};
        for(uint i=0u;i<6u;++i) rounded[i]=analytic[i];
        if (test>=6u) rounded[0]=(point+analytic[0])-point;
        const float analyticDerivative=contactBarrierDirectionalDerivative(
            contact,analytic,dt);
        const float roundedDerivative=contactBarrierDirectionalDerivative(
            contact,rounded,dt);
        const float usedDerivative=test==7u?roundedDerivative:analyticDerivative;
        const uint base=global*18u;
        for(uint j=0u;j<18u;++j)output[base+j]=float4(0.0f);
        output[base]=float4(float(test),usedDerivative,dt,0.01f);
        for(uint i=0u;i<4u;++i)output[base+1u+i]=float4(positions[i],0.0f);
        for(uint i=0u;i<6u;++i)output[base+5u+i]=float4(analytic[i],0.0f);
        output[base+11u]=float4(thickness,baseResponse.w,mollifier,baseEnergy);
        output[base+12u]=float4(analyticDerivative,roundedDerivative,
            length(rounded[0]),witnessValid?1.0f:0.0f);
        output[base+13u]=float4(barycentric,baseResponse.x);
        return;
    }
    if (global>=109u) {
        const uint test=global-109u;
        const float3 a=float3(0.0f), b=float3(0.02f,0.0f,0.0f), c=float3(0.0f,0.02f,0.0f);
        const float3 point=test==2u?float3(0.004f,0.006f,1.0e-13f):float3(0.004f,0.006f,0.001f);
        const float3 featurePosition[4]={point,a,b,c};
        NMMortarContactData mapping{};
        mapping.found=1u; mapping.kind=NM_DEFORMABLE_CONTACT_VERTEX_TRIANGLE;
        mapping.positiveSide=test==0u?0u:1u;
        mapping.normal=test==2u?float3(0.6f,0.0f,0.8f):float3(0.0f,0.0f,1.0f);
        mapping.featureBarycentric[0]=float3(1.0f,0.0f,0.0f);
        mapping.featureBarycentric[1]=float3(1.0f,0.0f,0.0f);
        mapping.featureBarycentric[2]=float3(0.0f,1.0f,0.0f);
        mapping.featureBarycentric[3]=float3(0.0f,0.0f,1.0f);
        const float3 pointParent[3]={point,point+float3(0.02f,0.0f,0.0f),point+float3(0.0f,0.02f,0.0f)};
        const float3 triangleParent[3]={a,b,c};
        float3 parent0[3],parent1[3];
        for(uint i=0u;i<3u;++i) {
            parent0[i]=mapping.positiveSide==0u?pointParent[i]:triangleParent[i];
            parent1[i]=mapping.positiveSide==0u?triangleParent[i]:pointParent[i];
        }
        float3 normal,closest; float separation;
        float weights[6]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        const bool valid=contactMortarVertexTriangleWitness(mapping,featurePosition,normal,closest,separation,weights);
        float3 weightedPosition=float3(0.0f);
        for(uint i=0u;i<3u;++i) weightedPosition+=weights[i]*parent0[i]+weights[3u+i]*parent1[i];
        const float3 torque=cross(weightedPosition,normal);
        const uint base=global*18u;
        for(uint j=0u;j<18u;++j)output[base+j]=float4(0.0f);
        output[base]=float4(float(test),separation,valid?1.0f:0.0f,float(mapping.positiveSide));
        for(uint i=0u;i<4u;++i)output[base+1u+i]=float4(featurePosition[i],0.0f);
        for(uint i=0u;i<3u;++i) { output[base+5u+i]=float4(parent0[i],0.0f); output[base+8u+i]=float4(parent1[i],0.0f); }
        output[base+11u]=float4(weights[0],weights[1],weights[2],weights[3]);
        output[base+12u]=float4(weights[4],weights[5],0.0f,0.0f);
        output[base+13u]=float4(weightedPosition,0.0f);
        output[base+14u]=float4(normal,length(normal)); output[base+15u]=float4(closest,0.0f);
        output[base+16u]=float4(mapping.normal,separation); output[base+17u]=float4(torque,0.0f);
        return;
    }
    if (global>=102u) {
        const uint test=global-102u;
        float3 point,a,b,c; float thickness=0.005f;
        if(test==0u) {
            point=float3(0.0491591915f,-0.585491121f,0.114537001f);
            a=float3(0.0487377346f,-0.585262775f,0.114244871f);
            b=float3(0.0444787107f,-0.582939923f,0.111115232f);
            c=float3(0.0516800359f,-0.586848319f,0.116216816f); thickness=1.0e-5f;
        } else if(test==1u) {
            point=float3(0.0491592772f,-0.58549118f,0.114537053f);
            a=float3(0.0487376861f,-0.585262775f,0.114244834f);
            b=float3(0.0444787107f,-0.582939923f,0.111115232f);
            c=float3(0.0516800322f,-0.586848319f,0.116216816f); thickness=1.0e-5f;
        } else if(test==2u) {
            point=float3(0.0f,0.0f,0.0002f); a=float3(0.0f); b=float3(0.02f,0.0f,0.0f); c=float3(0.0f,0.02f,0.0f);
        } else if(test==3u) {
            point=float3(0.01f,-0.0002f,0.0001f); a=float3(0.0f); b=float3(0.02f,0.0f,0.0f); c=float3(0.0f,0.02f,0.0f);
        } else if(test==4u) {
            point=float3(0.004f,0.006f,0.0001f); a=float3(0.0f); b=float3(0.02f,0.0f,0.0f); c=float3(0.0f,0.02f,0.0f);
        } else if(test==5u) {
            point=float3(-0.0001f,0.0f,0.0003f); a=float3(0.0f); b=float3(0.02f,0.0f,0.0f); c=float3(0.04f,0.0f,0.0f);
        } else {
            const float3 o=float3(8192.0f,-4096.0f,2048.0f);
            point=o+float3(0.0703125f,0.0595703125f,0.001953125f);
            a=o; b=o+float3(0.3f,0.04f,0.05f); c=o+float3(0.04f,0.3f,-0.03f); thickness=0.01f;
        }
        const float3 positions[4]={point,a,b,c};
        float3 bary,closest; MRCompensatedPositionGPU relative; float separation;
        const bool valid=contactCompensatedVertexTriangleWitness(point,a,b,c,bary,closest,relative,separation);
        float3 legacyBary; const float3 legacyClosest=closestPointOnTriangle(point,a,b,c,legacyBary);
        const float3 legacyRelative=point-legacyClosest; const float legacySeparation=length(legacyRelative);
        float3 firstGradient,secondGradient;
        const float mollifier=ipcFeatureMollifierDifferential(b-a,c-a,firstGradient,secondGradient);
        NMDeformableContactGPU contact{}; contact.identity.z=NM_DEFORMABLE_CONTACT_VERTEX_TRIANGLE;
        contact.barrier.z=thickness; contact.mollifier.w=1.81084e13f;
        float energy,energySeparation;
        const bool energyValid=deformableBarrierEnergyAt(contact,positions,energy,energySeparation);
        const float3 hi=relative.high.xyz,lo=relative.low.xyz,delta=hi+lo;
        const float3 normal=separation>0.0f?delta/separation:float3(0.0f);
        const uint base=global*18u;
        for(uint j=0u;j<18u;++j)output[base+j]=float4(0.0f);
        output[base]=float4(float(test),separation,valid?1.0f:0.0f,thickness);
        output[base+1u]=float4(point,0.0f); output[base+2u]=float4(a,0.0f);
        output[base+3u]=float4(b,0.0f); output[base+4u]=float4(c,0.0f);
        output[base+5u]=float4(bary,0.0f); output[base+6u]=float4(closest,0.0f);
        output[base+7u]=float4(hi,separation); output[base+8u]=float4(lo,0.0f);
        output[base+9u]=float4(normal,length(normal));
        output[base+10u]=float4(thickness,contact.mollifier.w,mollifier,energyValid?1.0f:0.0f);
        output[base+11u]=float4(legacyBary,legacySeparation); output[base+12u]=float4(legacyClosest,0.0f);
        output[base+13u]=float4(legacySeparation>0.0f?legacyRelative/legacySeparation:float3(0.0f),legacySeparation);
        output[base+14u]=float4(energy,energySeparation,energyValid?1.0f:0.0f,mollifier);
        return;
    }
    if (global>=96u) {
        const uint row=global;
        float3 p0,p1,q0,q1;
        if (row==96u) {
            p0=float3(-0.0164212119,-0.519634128,0.0346223861);
            p1=float3(-0.00803090353,-0.526707113,0.0277656447);
            q0=float3(-0.00755289057,-0.54869628,0.0357304141);
            q1=float3(-0.0307032298,-0.461499035,0.0284029972);
        } else if (row==97u) {
            p0=float3(-0.0164212119,-0.519634128,0.0346223861);
            p1=float3(-0.00803074706,-0.526707053,0.0277655888);
            q0=float3(-0.00755289756,-0.54869628,0.0357304364);
            q1=float3(-0.0307032261,-0.461499035,0.0284030009);
        } else if (row==98u) {
            p0=float3(-0.0164212119,-0.519634128,0.0346223861);
            p1=float3(-0.00803090259,-0.526707113,0.0277656447);
            q0=float3(-0.00755289104,-0.54869628,0.0357304141);
            q1=float3(-0.0307032298,-0.461499035,0.0284029972);
        } else if (row==99u) {
            p0=float3(0.001,0.002,0.003); p1=p0;
            q0=float3(0.001,0.002,0.004);
            q1=float3(0.001,0.002,0.005);
        } else if (row==100u) {
            p0=float3(0.0,0.0,0.0); p1=float3(1.0,0.0,0.0);
            q0=float3(1.0,0.2,0.0001); q1=float3(1.0,1.0,0.0001);
        } else {
            p0=float3(-0.5,0.0,0.0); p1=float3(0.5,0.0,0.0);
            q0=float3(-0.5,0.0005,0.0001);
            q1=float3(0.5,0.000501,0.0001);
        }
        float s,t; float3 oldP,oldQ;
        MRCompensatedPositionGPU relative; float separation;
        const bool witnessValid=contactCompensatedSegmentWitness(
            p0,p1,q0,q1,s,t,oldP,oldQ,relative,separation);
        const float3 positions[4]={p0,p1,q0,q1};
        NMDeformableContactGPU contact{};
        contact.identity.z=NM_DEFORMABLE_CONTACT_EDGE_EDGE;
        contact.barrier.z=1.0e-5f;
        contact.mollifier.w=1.0f;
        float energy,energySeparation;
        const bool energyValid=deformableBarrierEnergyAt(
            contact,positions,energy,energySeparation);
        float3 mollifierGradient0,mollifierGradient1;
        const float mollifier=ipcFeatureMollifierDifferential(
            p1-p0,q1-q0,mollifierGradient0,mollifierGradient1);
        const float3 relativeHigh=relative.high.xyz;
        const float3 relativeLow=relative.low.xyz;
        const float3 roundedRelative=relativeHigh+relativeLow;
        const float3 normal=separation>0.0f
            ? roundedRelative/separation : float3(0.0f);
        const uint base=global*18u;
        for(uint j=0u;j<18u;++j)output[base+j]=float4(0.0f);
        output[base]=float4(float(row-96u),separation,float(s),float(t));
        output[base+1u]=float4(p0,0.0f); output[base+2u]=float4(p1,0.0f);
        output[base+3u]=float4(q0,0.0f); output[base+4u]=float4(q1,0.0f);
        output[base+7u]=float4(s,t,witnessValid?1.0f:0.0f,0.0f);
        output[base+9u]=float4(oldP,0.0f); output[base+11u]=float4(oldQ,0.0f);
        output[base+13u]=float4(relativeHigh,separation);
        output[base+14u]=float4(relativeLow,0.0f);
        output[base+15u]=float4(length(oldP-oldQ),0.0f,0.0f,0.0f);
        output[base+16u]=float4(energy,energySeparation,
            energyValid?1.0f:0.0f,mollifier);
        output[base+17u]=float4(normal, length(normal));
        return;
    }
    if (global>=48u) {
        const uint row=global-48u;
        const uint angleIndex=row%6u;
        const float scales[4]={0.0001f,0.0005f,0.002f,0.02f};
        const float angles[6]={0.0f,0.003f,0.01f,0.02f,0.04f,1.5707963267948966f};
        const float scale=scales[(row/6u)%4u];
        const float lengthRatio=row<24u?1.0f:7.0f;
        const float theta=angles[angleIndex];
        const float3 first(scale,0.0f,0.0f);
        const float3 second=angleIndex==5u?float3(0.0f,scale*lengthRatio,0.0f):
            scale*lengthRatio*float3(cos(theta),sin(theta),0.0f);
        float3 g1,g2;
        const float value=ipcFeatureMollifierDifferential(first,second,g1,g2);
        const uint base=global*18u;
        for(uint j=0u;j<18u;++j)output[base+j]=float4(0.0f);
        output[base]=float4(float(angleIndex),scale,lengthRatio,float(global));
        output[base+1u]=float4(first,0.0f);output[base+2u]=float4(second,0.0f);
        output[base+3u]=float4(g1,0.0f);output[base+4u]=float4(g2,0.0f);
        output[base+5u]=float4(value,theta,0.0f,0.0f);
        return;
    }
    const uint kind=global%6u;
    const uint scaleIndex=(global/6u)%4u;
    const float scales[4]={0.0001f,0.0005f,0.002f,0.02f};
    const float s=scales[scaleIndex];
    const float shift=global<24u?0.0f:0.03f;
    const float3 origin(shift,-0.5f*shift,0.25f*shift);
    float3 point=origin,a=origin,b=origin,c=origin,q1=origin;
    float3 newBary(0.0f),newP(0.0f),newQ(0.0f);
    if (kind<3u) {
        a=origin;b=origin+float3(s,0,0);c=origin+float3(0,s,0);
        if(kind==0u)point=origin+float3(0.25f*s,0.25f*s,0.1f*s);
        if(kind==1u)point=origin+float3(0.5f*s,-0.2f*s,0.1f*s);
        if(kind==2u)point=origin+float3(-0.2f*s,1.2f*s,0.1f*s);

        newP=closestPointOnTriangle(point,a,b,c,newBary);
    } else {
        if(kind==3u){a=origin+float3(-0.5f*s,0,0);b=origin+float3(0.5f*s,0,0);c=origin+float3(0,-0.5f*s,0.1f*s);q1=origin+float3(0,0.5f*s,0.1f*s);}
        if(kind==4u){a=origin;b=origin+float3(s,0,0);c=origin+float3(0.25f*s,0.1f*s,0);q1=origin+float3(0.75f*s,0.1f*s,0);}
        if(kind==5u){a=b=origin;c=origin+float3(-0.5f*s,0.2f*s,0);q1=origin+float3(0.5f*s,0.2f*s,0);}
        float newS,newT;

        closestPointsOnSegments(a,b,c,q1,newS,newT,newP,newQ);
        newBary=float3(newS,newT,0);
    }
    const uint base=global*18u;
    for(uint j=0;j<18u;++j)output[base+j]=float4(0.0f);
    output[base]=float4(float(kind),s,shift,float(global));
    output[base+1u]=float4(point,1);output[base+2u]=float4(a,1);
    output[base+3u]=float4(b,1);output[base+4u]=float4(c,1);output[base+5u]=float4(q1,1);
    output[base+7u]=float4(newBary,1);
    output[base+9u]=float4(newP,1);
    output[base+11u]=float4(newQ,1);
}

kernel void nm_contact_vf_smooth_guard(
    constant NMMatterDispatchGPU& dispatch [[buffer(0)]],
    device const NMSchedulerStateGPU* schedulers [[buffer(1)]],
    device const NMFEMNodeStateGPU* candidate [[buffer(2)]],
    device const float4* direction [[buffer(3)]],
    constant uint& traceSlotsPerEnvironment [[buffer(4)]],
    device const NMDeformableContactLimitTraceGPU* traceRecords [[buffer(5)]],
    device uint4* smoothSupport [[buffer(6)]],
    uint global [[thread_position_in_grid]]
) {
    if (traceSlotsPerEnvironment == 0u) return;
    const uint recordCount = dispatch.environmentCount * traceSlotsPerEnvironment;
    if (global >= recordCount) return;
    const uint environment = global / traceSlotsPerEnvironment;
    const uint activeSlot = global - environment * traceSlotsPerEnvironment;
    const NMDeformableContactLimitTraceGPU record = traceRecords[global];
    uint supported[4] = {0u,0u,0u,0u};
    if (record.lineSearch.w == 1.0f &&
        (record.contact.identity.w & NM_CONTACT_VALID) != 0u &&
        record.result.x == activeSlot &&
        record.contact.identity.x < dispatch.objectCount) {
        const float timestep = objectTimestep(
            dispatch, schedulers[environment * dispatch.objectCount +
                                 record.contact.identity.x]);
        for (uint local=0u; local<4u; ++local) {
            float3 action=float3(0.0f);
            const uint node=deformableContactNode(record.contact,local);
            supported[local]=uint(deformableExactVertexTriangleDirectionalForce(
                dispatch,record.contact,candidate,direction,environment,node,
                timestep,action,true));
        }
    }
    smoothSupport[global]=uint4(
        supported[0],supported[1],supported[2],supported[3]);
}
}

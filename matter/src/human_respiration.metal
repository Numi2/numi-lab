// The respiratory muscles use the existing MyoSim activation and compliant
// fibre/tendon implementation. No independent muscle constitutive law.
#include "MujocoMuscleReference.metal"
#include "numi/matter/human_respiration.h"
#include "RespiratoryChemoreflexV1.metal"

kernel void nm_human_respiration_brain_observe(
    constant NMHumanRespirationBrainDispatch& d [[buffer(0)]],
    device const NMHumanRespirationState* accepted [[buffer(1)]],
    device NBNumiRespiratoryChemoreflexInputV1* input [[buffer(2)]],
    uint env [[thread_position_in_grid]]) {
    if(env>=d.environmentCount)return;
    const auto s=accepted[env];
    NBNumiRespiratoryChemoreflexInputV1 sample{};
    sample.sourceTimestampMicroseconds=d.sourceTimeMicroseconds;
    sample.sourceAcceptedRootFingerprint=d.sourceRootIdentity;
    sample.targetTimestampMicroseconds=d.targetTimeMicroseconds;
    sample.targetTransactionFingerprint=d.targetRootIdentity;
    sample.paO2MillimetersMercury=s.observation.x;
    sample.paCO2MillimetersMercury=s.observation.y;
    sample.validityMask=NB_NUMI_RESPIRATORY_VALID_PA_O2|NB_NUMI_RESPIRATORY_VALID_PA_CO2;
    sample.flags=s.status.x==d.sourceStep&&!s.status.w?NB_NUMI_RESPIRATORY_INPUT_ACCEPTED:0;
    input[env]=sample;
}

kernel void nm_human_respiration_brain_deliver(
    constant NMHumanRespirationBrainDispatch& d [[buffer(0)]],
    device const NBNumiRespiratoryChemoreflexOutputV1* output [[buffer(1)]],
    device float4* excitation [[buffer(2)]],
    uint env [[thread_position_in_grid]]) {
    if(env>=d.environmentCount)return;
    auto o=output[env];
    if(!(o.flags&NB_NUMI_RESPIRATORY_RECORD_VALID)||o.targetTransactionFingerprint!=d.targetRootIdentity)
        excitation[env]=float4(NAN);
    else excitation[env]=float4(clamp(d.driveScale*float2(o.diaphragmExcitation,o.intercostalExcitation),0.0f,1.0f),
        o.targetFrequencyBreathsPerMinute,o.targetMinuteVentilationLitresPerMinute);
}

kernel void nm_human_respiration_brain_resolve(
    constant NMHumanRespirationBrainDispatch& d [[buffer(0)]],
    device const NBNumiRespiratoryChemoreflexStateV1* candidate [[buffer(1)]],
    device NBNumiRespiratoryChemoreflexStateV1* accepted [[buffer(2)]],
    device const NMMatterStatusGPU* statuses [[buffer(3)]],
    device const NMHumanRespirationState* respiration [[buffer(4)]],
    uint env [[thread_position_in_grid]]) {
    if(env<d.environmentCount&&statuses[env].code==NM_STATUS_SUCCESS&&!respiration[env].status.w&&
       (candidate[env].flags&NB_NUMI_RESPIRATORY_RECORD_VALID)&&
       candidate[env].acceptedRootFingerprint==d.targetRootIdentity)
        accepted[env]=candidate[env];
}

namespace human_respiration {
inline void addGas(thread float4& amount, float2 delta) {
    float2 y = delta - amount.zw;
    float2 next = amount.xy + y;
    amount.zw = (next - amount.xy) - y;
    amount.xy = next;
}
inline float saturation(float pressure, constant NMHumanRespirationParameters& p) {
    float ratio = pow(max(pressure, 0.0f) / p.oxygen.z, p.oxygen.w);
    return ratio / (1.0f + ratio);
}
inline float oxygenContent(float pressure, constant NMHumanRespirationParameters& p) {
    return p.oxygen.x * saturation(pressure, p) + p.oxygen.y * pressure;
}
inline float oxygenPressure(float content, constant NMHumanRespirationParameters& p) {
    float low = 0.0f, high = 760.0f;
    for (uint i=0; i<24; ++i) {
        float mid = (low + high) * 0.5f;
        if (oxygenContent(mid,p) < content) low=mid; else high=mid;
    }
    return (low+high)*0.5f;
}
inline float physical(device const float4* values,
                      device const NMVascularUnknownGPU* unknowns, uint base, uint row) {
    return values[base+row].x * unknowns[row].initialAndScaling.y;
}
}

kernel void nm_human_respiration_predict(
    constant NMHumanRespirationParameters& p [[buffer(0)]],
    constant NMHumanRespirationDispatch& d [[buffer(1)]],
    device const NMHumanRespirationState* accepted [[buffer(2)]],
    device NMHumanRespirationState* candidate [[buffer(3)]],
    device const float4* excitations [[buffer(4)]],
    uint env [[thread_position_in_grid]]) {
    if (env>=d.environmentCount) return;
    NMHumanRespirationState n=accepted[env];
    n.control=excitations[env];
    n.status.w=0;
    const float dt=p.environment.w;
    float2 pressure=0;
    for(uint m=0; m<2; ++m) {
        auto muscle=p.muscles[m];
        auto state=n.muscles[m];
        float excitation=excitations[env][m];
        if (!isfinite(excitation) || excitation<0 || excitation>1) {n.status.w=1;break;}
        float a=state.excitationAndActivation.y;
        float derivative=activationDerivative(muscle,excitation,a);
        float tau=derivative!=0 ? (excitation-a)/derivative : 1;
        float nextA=excitation-(excitation-a)*exp(-dt/tau);
        const float area=p.geometry[m+2];
        const float path=muscle.compliantArchitecture0.x+muscle.compliantArchitecture0.y-n.motion[m]/area;
        float fibre,velocity,tension,residual;
        if (!solveCompliantFiber(path,-n.motion[m+2]/area,dt,nextA,state,muscle,
                                 fibre,velocity,tension,residual) || abs(residual)>1.e-3f) {
            n.status.w=2;break;
        }
        n.muscles[m].excitationAndActivation=float4(excitation,nextA,fibre,velocity);
        pressure[m]=tension*forceScale(muscle.gainParameters,muscle.lengthRangeAndAcceleration.z)/area;
    }
    // Backward Euler for two muscle-driven thoracic volume coordinates.
    // Lung recoil and mouth resistance act on their sum, giving a symmetric
    // coupled mechanical solve. No phase/clock function writes lung volume.
    const float lungE=1.0f/p.lung.z;
    const float common=dt*(p.lung.w+dt*lungE);
    const float a=p.geometry.x+dt*p.chest.z+dt*dt/p.chest.x+common;
    const float b=common;
    const float c=p.geometry.y+dt*p.chest.w+dt*dt/p.chest.y+common;
    const float recoil=lungE*(n.motion.x+n.motion.y);
    const float2 rhs=p.geometry.xy*n.motion.zw+dt*(pressure-float2(n.motion.x/p.chest.x,n.motion.y/p.chest.y)-recoil);
    const float determinant=a*c-b*b;
    const float2 speed=float2(c*rhs.x-b*rhs.y,a*rhs.y-b*rhs.x)/determinant;
    n.motion.xy+=dt*speed;
    n.motion.zw=speed;
    const float flow=speed.x+speed.y;
    const float volume=p.lung.x+n.motion.x+n.motion.y;
    const float alveolarPressure=-p.lung.w*flow;
    n.mechanics=float4(volume,alveolarPressure,
        p.environment.x+alveolarPressure-lungE*(volume-p.lung.x),flow);
    if (!all(isfinite(n.motion)) || !all(isfinite(n.mechanics)) ||
        !(volume>p.lung.y) || volume>0.008f || determinant<=0 || d.reject) n.status.w=3;
    candidate[env]=n;
}

kernel void nm_human_respiration_exchange(
    constant NMHumanRespirationParameters& p [[buffer(0)]],
    constant NMHumanRespirationDispatch& d [[buffer(1)]],
    device const NMHumanRespirationState* accepted [[buffer(2)]],
    device NMHumanRespirationState* candidate [[buffer(3)]],
    device const float4* vascularBefore [[buffer(4)]],
    device const float4* vascularAfter [[buffer(5)]],
    device const NMVascularUnknownGPU* unknowns [[buffer(6)]],
    device const NMVascularConnectionGPU* connections [[buffer(7)]],
    device NMMatterStatusGPU* statuses [[buffer(8)]],
    device const NMVascularCompartmentGPU* compartments [[buffer(9)]],
    device const float* elastance [[buffer(10)]],
    uint env [[thread_position_in_grid]]) {
    if(env>=d.environmentCount) return;
    NMHumanRespirationState n=candidate[env];
    if(statuses[env].code!=NM_STATUS_SUCCESS) return;
    const NMHumanRespirationState old=accepted[env];
    const float dt=p.environment.w;
    const uint base=env*d.vascularStride;
    float2 delta[21];
    for(uint row=0;row<21;++row) delta[row]=0;
    float2 alveolarDelta=0;
    const float alveolarVolume=old.mechanics.x-p.lung.y;
    const float2 alveolarFraction=old.alveolarGas.xy/(alveolarVolume*p.environment.z);
    const float2 deadFraction=old.deadSpaceGas.xy/(p.lung.y*p.environment.z);
    const float2 alveolarPressure=alveolarFraction*(p.environment.y/133.322387415f);
    const float2 capillaryContent=float2(human_respiration::oxygenContent(alveolarPressure.x,p),
        p.carbonDioxide.x+p.carbonDioxide.y*(alveolarPressure.y-40.0f));
    float pulmonaryO2=0;
    for(uint edge=0;edge<p.topology.w;++edge) {
        const auto con=connections[edge];
        const float flow=human_respiration::physical(vascularAfter,unknowns,base,21+edge);
        const uint from=flow>=0?con.identity.y:con.identity.z;
        const uint to=flow>=0?con.identity.z:con.identity.y;
        const float fromVolume=human_respiration::physical(vascularBefore,unknowns,base,from);
        const float2 content=old.bloodGas[from].xy/fromVolume;
        const float2 transported=dt*abs(flow)*content;
        delta[from]-=transported;
        delta[to]+=transported;
        if(edge==p.topology.x && flow>0) {
            // Perfusion-limited capillary equilibration; an explicit finite
            // effectiveness allows a matched diffusion sensitivity check.
            const float2 exchange=dt*flow*p.carbonDioxide.z*(capillaryContent-content);
            delta[to]+=exchange;
            alveolarDelta-=exchange;
            pulmonaryO2=exchange.x/dt;
        }
        if(dt*abs(flow)>0.1f*fromVolume || !all(isfinite(transported))) n.status.w=4;
    }
    for(uint bed=0;bed<4;++bed) {
        const float2 demand=dt*p.metabolism.zw*p.tissueFractions[bed];
        delta[p.tissueRows[bed]]+=float2(-demand.x,demand.y);
    }
    human_respiration::addGas(n.metabolicGas,dt*p.metabolism.zw);
    // Explicit upwind exchange with one anatomical dead-space reservoir.
    // The same flux leaves one reservoir and enters the next.
    const float transport=dt*n.mechanics.w*p.environment.z;
    const float2 outsideFlux=transport*(transport>=0?p.metabolism.xy:deadFraction);
    const float2 airwayFlux=transport*(transport>=0?deadFraction:alveolarFraction);
    human_respiration::addGas(n.environmentGas,outsideFlux);
    human_respiration::addGas(n.deadSpaceGas,outsideFlux-airwayFlux);
    human_respiration::addGas(n.alveolarGas,airwayFlux+alveolarDelta);
    for(uint row=0;row<21;++row) {
        human_respiration::addGas(n.bloodGas[row],delta[row]);
        if(!all(isfinite(n.bloodGas[row]))||any(n.bloodGas[row].xy<0)) n.status.w=5;
    }
    const uint sensed=p.topology.y;
    const float volume=human_respiration::physical(vascularAfter,unknowns,base,sensed);
    const float2 content=n.bloodGas[sensed].xy/volume;
    const float po2=human_respiration::oxygenPressure(content.x,p);
    const float pco2=40+(content.y-p.carbonDioxide.x)/p.carbonDioxide.y;
    n.observation=float4(po2,pco2,human_respiration::saturation(po2,p),pulmonaryO2);
    // These four source compartments use CVSim's linear/time-varying
    // elastance pressure law (atan venous reservoirs are not sampled here).
    const uint4 pressureRows=uint4(20,16,0,17);
    for(uint i=0;i<4;++i) {
        uint row=pressureRows[i];auto c=compartments[row];
        const float v=human_respiration::physical(vascularAfter,unknowns,base,row);
        n.cardiacPressure[i]=c.compliance.w+c.compliance.y+
            (v-c.compliance.x)*elastance[env*21+row];
    }
    float bloodVolume=0;
    for(uint row=0;row<21;++row) bloodVolume+=human_respiration::physical(vascularAfter,unknowns,base,row);
    n.circulation=float4(bloodVolume,
        human_respiration::physical(vascularAfter,unknowns,base,20),
        human_respiration::physical(vascularAfter,unknowns,base,16),
        max(n.circulation.w,abs(bloodVolume-0.00515f)));
    const float aortic=max(0.0f,human_respiration::physical(vascularAfter,unknowns,base,21));
    const float pulmonary=max(0.0f,human_respiration::physical(vascularAfter,unknowns,base,41));
    const float mitral=max(0.0f,human_respiration::physical(vascularAfter,unknowns,base,44));
    n.cardiacFlow.xy+=dt*float2(aortic,pulmonary);
    n.cardiacFlow.z+=dt*aortic;
    if(mitral>1.e-9f)n.cardiacStatus.y=1;
    if(aortic<=1.e-9f && n.cardiacStatus.z && n.cardiacStatus.y) {
        ++n.cardiacStatus.x;n.cardiacStatus.y=0;
        n.cardiacFlow.w=n.cardiacFlow.z;n.cardiacFlow.z=0;
    }
    n.cardiacStatus.z=aortic>1.e-9f;
    float4 total=0;
    human_respiration::addGas(total,n.alveolarGas.xy);
    human_respiration::addGas(total,n.deadSpaceGas.xy);
    for(uint row=0;row<21;++row) human_respiration::addGas(total,n.bloodGas[row].xy);
    const float2 balance=total.xy-n.environmentGas.xy+
        float2(n.metabolicGas.x,-n.metabolicGas.y)-n.gasBudget.xy;
    n.gasBudget.zw=max(n.gasBudget.zw,abs(balance));
    if(any(abs(balance)>5.e-5f*n.gasBudget.xy)) n.status.w=7;
    n.breath.x=max(n.breath.x,n.mechanics.x);
    n.breath.y=min(n.breath.y,n.mechanics.x);
    const uint inspiration=n.mechanics.w>0;
    if(inspiration&&!n.status.z && n.status.x>0) {
        n.breath.z=n.breath.x-n.breath.y;
        n.breath.xy=float2(n.mechanics.x);
        ++n.status.y;
    }
    n.breath.w+=dt*max(0.0f,n.mechanics.w);
    n.status.z=inspiration;
    ++n.status.x;
    if(!all(isfinite(n.observation))||!all(isfinite(n.alveolarGas))||
       any(n.alveolarGas.xy<0)||any(n.deadSpaceGas.xy<0)||
       abs(transport)>0.1f*p.lung.y*p.environment.z) n.status.w=6;
    if(n.status.w) {statuses[env].code=NM_STATUS_MULTIPHYSICS_FAILURE;}
    candidate[env]=n;
}

kernel void nm_human_respiration_resolve(
    constant NMHumanRespirationDispatch& d [[buffer(0)]],
    device const NMHumanRespirationState* candidate [[buffer(1)]],
    device NMHumanRespirationState* accepted [[buffer(2)]],
    device const NMMatterStatusGPU* statuses [[buffer(3)]],
    uint env [[thread_position_in_grid]]) {
    if(env<d.environmentCount && statuses[env].code==NM_STATUS_SUCCESS && !candidate[env].status.w)
        accepted[env]=candidate[env];
}

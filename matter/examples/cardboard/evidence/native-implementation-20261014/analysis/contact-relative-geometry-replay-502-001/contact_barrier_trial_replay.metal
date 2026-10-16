#include "matter_kernels.metal"

namespace numi_matter_metal {

// Diagnostic only: evaluate the same contact energy after translating the
// nearest-point problem to the query origin. No production source is changed.
inline bool localBarrierEnergyAt(const NMDeformableContactGPU contact,
    thread const float3* position, thread float& energy, thread float& separation) {
    float m=1.0f;
    if (contact.identity.z==NM_DEFORMABLE_CONTACT_POINT_POINT) {
        separation=length(position[0]-position[1]);
    } else if (contact.identity.z==NM_DEFORMABLE_CONTACT_VERTEX_TRIANGLE) {
        float3 bary;
        const float3 a=position[1]-position[0];
        const float3 b=position[2]-position[0];
        const float3 c=position[3]-position[0];
        const float3 q=closestPointOnTriangle(float3(0.0f),a,b,c,bary);
        separation=length(q);
        float3 g1,g2;
        m=ipcFeatureMollifierDifferential(position[2]-position[1],position[3]-position[1],g1,g2);
    } else if(contact.identity.z==NM_DEFORMABLE_CONTACT_EDGE_EDGE) {
        float s,t;float3 q1,q2;
        closestPointsOnSegments(float3(0.0f),position[1]-position[0],
            position[2]-position[0],position[3]-position[0],s,t,q1,q2);
        separation=length(q1-q2);
        float3 g1,g2;
        m=ipcFeatureMollifierDifferential(position[1]-position[0],position[3]-position[2],g1,g2);
    } else {energy=INFINITY;separation=-INFINITY;return false;}
    energy=m*ipcBarrierEnergy(separation,contact.barrier.z,contact.mollifier.w);
    return isfinite(separation)&&isfinite(energy);
}


kernel void nm_contact_barrier_trial_replay(
    device const NMDeformableContactGPU* contacts [[buffer(0)]],
    device const NMFEMNodeStateGPU* accepted [[buffer(1)]],
    device const NMFEMNodeStateGPU* candidate [[buffer(2)]],
    device const float4* correction [[buffer(3)]],
    device const float* alphas [[buffer(4)]],
    device float4* output [[buffer(5)]],
    constant uint& contactCount [[buffer(6)]],
    constant uint& trialsPerContact [[buffer(7)]],
    constant uint& nodeCount [[buffer(8)]],
    device const float* timestepWords [[buffer(9)]],
    uint global [[thread_position_in_grid]])
{
    const uint total = contactCount * trialsPerContact;
    if (global >= total || trialsPerContact == 0u) return;
    const uint contactIndex = global / trialsPerContact;
    const NMDeformableContactGPU contact = contacts[contactIndex];
    const float alpha = alphas[global];
    const float timestep = timestepWords[0];
    float3 current[4];
    float3 contactTrial[4];
    float3 appliedTrial[4];
    bool validNodes = isfinite(alpha) && isfinite(timestep) && timestep > 0.0f;
    for (uint local = 0u; local < 4u; ++local) {
        const uint node = contact.nodes[local];
        if (node >= nodeCount) {
            validNodes = false;
            current[local] = float3(0.0f);
            contactTrial[local] = float3(0.0f);
            appliedTrial[local] = float3(0.0f);
            continue;
        }
        const NMFEMNodeStateGPU old = accepted[node];
        const NMFEMNodeStateGPU next = candidate[node];
        const float3 deltaV = correction[node].xyz;
        current[local] = next.positionAndMass.xyz;

        // Production contact limiter: featurePosition + alpha * (dt * correction).
        const float3 featureDirection = timestep * deltaV;
        contactTrial[local] = current[local] + alpha * featureDirection;

        // Production FEM Armijo update: candidateVelocity += alpha * correction,
        // then accepted position + dt * updated velocity.
        const float3 nextVelocity =
            next.velocityAndInverseMass.xyz + alpha * deltaV;
        if (next.referenceDisplacementAndMode.w == 1.0f) {
            const float3 nextReferenceDisplacement =
                old.referenceDisplacementAndMode.xyz + timestep * nextVelocity;
            appliedTrial[local] =
                next.restAndFixed.xyz + nextReferenceDisplacement;
        } else {
            appliedTrial[local] = old.positionAndMass.xyz + timestep * nextVelocity;
        }
        validNodes = validNodes && all(isfinite(current[local])) &&
            all(isfinite(contactTrial[local])) && all(isfinite(appliedTrial[local]));
    }
    float currentEnergy = INFINITY, currentSeparation = -INFINITY;
    float contactEnergy = INFINITY, contactSeparation = -INFINITY;
    float appliedEnergy = INFINITY, appliedSeparation = -INFINITY;
    const bool currentValid = validNodes && localBarrierEnergyAt(
        contact, current, currentEnergy, currentSeparation);
    const bool contactValid = validNodes && localBarrierEnergyAt(
        contact, contactTrial, contactEnergy, contactSeparation);
    const bool appliedValid = validNodes && localBarrierEnergyAt(
        contact, appliedTrial, appliedEnergy, appliedSeparation);
    const uint out = global * 14u;
    for (uint local = 0u; local < 4u; ++local) {
        output[out + local] = float4(current[local], 1.0f);
        output[out + 4u + local] = float4(contactTrial[local], 1.0f);
        output[out + 8u + local] = float4(appliedTrial[local], 1.0f);
    }
    output[out + 12u] = float4(
        currentEnergy, currentSeparation, contactEnergy, contactSeparation);
    output[out + 13u] = float4(
        appliedEnergy, appliedSeparation,
        currentValid ? 1.0f : 0.0f,
        (contactValid ? 1.0f : 0.0f) + (appliedValid ? 2.0f : 0.0f));
}

} // namespace numi_matter_metal

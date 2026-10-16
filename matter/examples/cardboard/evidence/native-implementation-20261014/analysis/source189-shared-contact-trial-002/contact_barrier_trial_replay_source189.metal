#include "matter_kernels.metal"

namespace numi_matter_metal {

// Diagnostic-only replay of the production barrier energy and the production
// certified CCD helper for both candidate endpoints. No solver/transaction
// state is written. The two sweeps start at the exact captured candidate node
// words and end at the exact rounded endpoint words produced by each path.
kernel void nm_contact_barrier_trial_replay_ccd(
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
    device const float* collisionThickness [[buffer(10)]],
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
    float3 contactDirection[4];
    float3 appliedDirection[4];
    bool validNodes = isfinite(alpha) && isfinite(timestep) && timestep > 0.0f;
    for (uint local = 0u; local < 4u; ++local) {
        const uint node = contact.nodes[local];
        if (node >= nodeCount) {
            validNodes = false;
            current[local] = float3(0.0f);
            contactTrial[local] = float3(0.0f);
            appliedTrial[local] = float3(0.0f);
            contactDirection[local] = float3(0.0f);
            appliedDirection[local] = float3(0.0f);
            continue;
        }
        const NMFEMNodeStateGPU old = accepted[node];
        const NMFEMNodeStateGPU next = candidate[node];
        const float3 deltaV = correction[node].xyz;
        current[local] = next.positionAndMass.xyz;

        // Contact limiter endpoint: candidate position + alpha * (dt * dv).
        const float3 featureDirection = timestep * deltaV;
        contactTrial[local] = current[local] + alpha * featureDirection;
        contactDirection[local] = contactTrial[local] - current[local];

        // Actual FEM Armijo endpoint: candidate velocity += alpha * dv, then
        // rebuild position from the accepted position/displacement semantics.
        const NMFEMNodeStateGPU applied = femTrialNodeAtAlpha(
            old, next, deltaV, alpha, timestep, true);
        appliedTrial[local] = applied.positionAndMass.xyz;
        appliedDirection[local] = appliedTrial[local] - current[local];
        validNodes = validNodes && all(isfinite(current[local])) &&
            all(isfinite(contactTrial[local])) && all(isfinite(appliedTrial[local])) &&
            all(isfinite(contactDirection[local])) && all(isfinite(appliedDirection[local]));
    }
    const float thickness = collisionThickness[contactIndex];
    validNodes = validNodes && isfinite(thickness) && thickness >= 0.0f;
    float currentEnergy = INFINITY, currentSeparation = -INFINITY;
    float contactEnergy = INFINITY, contactSeparation = -INFINITY;
    float appliedEnergy = INFINITY, appliedSeparation = -INFINITY;
    const bool currentValid = validNodes && deformableBarrierEnergyAt(
        contact, current, currentEnergy, currentSeparation);
    const bool contactValid = validNodes && deformableBarrierEnergyAt(
        contact, contactTrial, contactEnergy, contactSeparation);
    const bool appliedValid = validNodes && deformableBarrierEnergyAt(
        contact, appliedTrial, appliedEnergy, appliedSeparation);

    float contactTOI = 0.0f, contactSeparationCCD = INFINITY;
    float3 contactNormal = float3(0.0f), contactPoint = float3(0.0f);
    bool contactCertified = false;
    float contactFraction = 0.0f;
    uint contactRetries = 0u;
    const bool contactHit = validNodes &&
        certifyDeformableNewtonSweepToEndpoint(
            contact.identity.z, current, contactTrial, thickness,
            contactTOI, contactNormal, contactPoint, contactSeparationCCD,
            contactCertified, contactFraction, contactRetries);

    float appliedTOI = 0.0f, appliedSeparationCCD = INFINITY;
    float3 appliedNormal = float3(0.0f), appliedPoint = float3(0.0f);
    bool appliedCertified = false;
    float appliedFraction = 0.0f;
    uint appliedRetries = 0u;
    const bool appliedHit = validNodes &&
        certifyDeformableNewtonSweepToEndpoint(
            contact.identity.z, current, appliedTrial, thickness,
            appliedTOI, appliedNormal, appliedPoint, appliedSeparationCCD,
            appliedCertified, appliedFraction, appliedRetries);

    float3 analyticDisplacement[6] = {
        float3(0.0f), float3(0.0f), float3(0.0f),
        float3(0.0f), float3(0.0f), float3(0.0f)
    };
    bool derivativeInputsValid = validNodes;
    for (uint local = 0u; local < 6u; ++local) {
        const uint node = deformableContactNode(contact, local);
        if (node == NM_INVALID_INDEX) continue;
        if (node >= nodeCount) {
            derivativeInputsValid = false;
            continue;
        }
        const NMFEMNodeStateGPU next = candidate[node];
        if (next.restAndFixed.w <= 0.5f)
            analyticDisplacement[local] = timestep * correction[node].xyz;
    }
    float impulseDotDisplacement = 0.0f;
    for (uint local = 0u; local < 6u; ++local) {
        if (deformableContactNode(contact, local) == NM_INVALID_INDEX)
            continue;
        const float3 barrierImpulse =
            deformableContactWeight(contact, local) *
                contact.barrier.x * contact.normalAndTOI.xyz -
            contact.mollifier.y *
                deformableMollifierGradient(contact, local);
        impulseDotDisplacement += dot(
            barrierImpulse, analyticDisplacement[local]);
    }
    const float directionalDerivative = derivativeInputsValid
        ? -impulseDotDisplacement / max(timestep, 1.0e-20f)
        : NAN;
    const float armijoRight = currentEnergy +
        1.0e-4f * alpha * directionalDerivative;

    const uint out = global * 21u;
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
    output[out + 14u] = float4(
        contactHit ? 1.0f : 0.0f, contactCertified ? 1.0f : 0.0f,
        contactTOI, contactSeparationCCD);
    output[out + 15u] = float4(
        contactFraction, float(contactRetries), contactNormal.x, contactNormal.y);
    output[out + 16u] = float4(
        contactNormal.z, contactPoint.x, contactPoint.y, contactPoint.z);
    output[out + 17u] = float4(
        appliedHit ? 1.0f : 0.0f, appliedCertified ? 1.0f : 0.0f,
        appliedTOI, appliedSeparationCCD);
    output[out + 18u] = float4(
        appliedFraction, float(appliedRetries), appliedNormal.x, appliedNormal.y);
    output[out + 19u] = float4(
        appliedNormal.z, appliedPoint.x, appliedPoint.y, appliedPoint.z);
    output[out + 20u] = float4(
        directionalDerivative, armijoRight, alpha,
        derivativeInputsValid ? 1.0f : 0.0f);
}

} // namespace numi_matter_metal

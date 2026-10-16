inline NMFEMNodeStateGPU femTrialNodeAtAlpha(
    const NMFEMNodeStateGPU acceptedState,
    NMFEMNodeStateGPU next,
    const float3 correction,
    const float alpha,
    const float dt,
    const bool updateEligible
) {
    if (!updateEligible || next.restAndFixed.w > 0.5f || !(alpha > 0.0f))
        return next;
    next.velocityAndInverseMass.xyz += alpha * correction;
    next.deltaVelocity.xyz = next.velocityAndInverseMass.xyz -
        acceptedState.velocityAndInverseMass.xyz;
    if (next.referenceDisplacementAndMode.w == 1.0f) {
        next.referenceDisplacementAndMode.xyz =
            acceptedState.referenceDisplacementAndMode.xyz +
            dt * next.velocityAndInverseMass.xyz;
        next.positionAndMass.xyz = next.restAndFixed.xyz +
            next.referenceDisplacementAndMode.xyz;
    } else {
        next.positionAndMass.xyz = acceptedState.positionAndMass.xyz +
            dt * next.velocityAndInverseMass.xyz;
    }
    return next;
}

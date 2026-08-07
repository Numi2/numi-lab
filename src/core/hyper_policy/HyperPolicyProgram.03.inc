    staged->fingerprint = fingerprintPack(pack, task);
    const std::uint32_t flags =
        MR_HYPER_POLICY_REFERENCE_RESIDUAL |
        (stochastic ? MR_HYPER_POLICY_STOCHASTIC : 0u);
    staged->header = {
        .counts0 = {
            staged->layout.layerCount,
            staged->layout.actorObservationCount,
            staged->layout.actionCount,
            staged->layout.coefficientCount,
        },
        .counts1 = {
            staged->layout.knotCount,
            staged->layout.referenceCount,
            staged->layout.signatureCount,
            staged->layout.maximumRank,
        },
        .counts2 = {
            staged->layout.eventCount,
            staged->layout.contactTrackCount,
            flags,
            0u,
        },
        .offsets0 = {
            layerTableOffset,
            observationMeanOffset,
            observationInverseStdOffset,
            coefficientLimitOffset,
        },
        .offsets1 = {
            actionBiasOffset,
            actionScaleOffset,
            knotPhaseOffset,
            coefficientKnotOffset,
        },
        .offsets2 = {
            coefficientTangentOffset,
            authorityKnotOffset,
            authorityTangentOffset,
            phaseRateKnotOffset,
        },
        .offsets3 = {
            phaseRateTangentOffset,
            referencePhaseOffset,
            referenceActionOffset,
            referenceSignatureOffset,
        },
        .offsets4 = {
            signatureWeightOffset,
            contactMaskOffset,
            eventOffset,
            logStdOffset,
        },
        .offsets5 = {
            contactGroupIndexOffset,
            0u,
            0u,
            0u,
        },
        .limits0 = {
            pack.observationClip,
            pack.actionClip,
            pack.maximumPhaseAdvancePerStep,
            pack.phaseAlignmentBlend,
        },
        .limits1 = {
            pack.phaseAlignmentHuberDelta,
            pack.controlTimeStep,
            1.0e-8f,
            0.0f,
        },
        .policyFingerprint = staged->fingerprint,
        .taskFingerprint = staged->taskFingerprint,
        .sourceMotionFingerprint = pack.sourceMotionFingerprint,
        .revision = staged->revision,
        .abi = {MR_HYPER_POLICY_PROGRAM_ABI_VERSION, 0u, 0u, 0u},
    };

    output.storage_ = std::move(staged);
    return {
        .status = HyperPolicyCompileStatus::success,
        .fingerprint = output.fingerprint(),
        .element = {},
        .message = {},
    };
}

const char* hyperPolicyCompileStatusName(
    const HyperPolicyCompileStatus status
) noexcept {
    switch (status) {
    case HyperPolicyCompileStatus::success:
        return "success";
    case HyperPolicyCompileStatus::invalidTask:
        return "invalid-task";
    case HyperPolicyCompileStatus::invalidPack:
        return "invalid-pack";
    case HyperPolicyCompileStatus::incompatibleContract:
        return "incompatible-contract";
    case HyperPolicyCompileStatus::arithmeticOverflow:
        return "arithmetic-overflow";
    case HyperPolicyCompileStatus::internalFailure:
        return "internal-failure";
    }
    return "unknown";
}

} // namespace metalrobo

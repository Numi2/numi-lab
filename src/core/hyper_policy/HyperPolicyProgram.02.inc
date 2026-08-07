    if (!strictlyNormalizedPhases(pack.knotPhases) ||
        !strictlyNormalizedPhases(pack.referencePhases) ||
        pack.knotPhases.size() > std::numeric_limits<std::uint32_t>::max() ||
        pack.referencePhases.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        pack.signatureCount == 0u ||
        pack.contactTrackCount == 0u || pack.contactTrackCount > 32u) {
        return reject(
            HyperPolicyCompileStatus::invalidPack,
            "phaseProgram",
            "knot/reference phases, signature width, or contact-track count is invalid"
        );
    }
    if (std::any_of(
            pack.contactGroupIndices.begin(),
            pack.contactGroupIndices.end(),
            [&](const std::uint32_t index) {
                return index >= task.contactGroups().size();
            }
        )) {
        return reject(
            HyperPolicyCompileStatus::incompatibleContract,
            "contactGroupIndices",
            "generated contact tracks must resolve to compiled TaskPack contact groups"
        );
    }
    const std::uint64_t knotCount = pack.knotPhases.size();
    const std::uint64_t referenceCount = pack.referencePhases.size();
    const std::uint64_t knotCoefficientCount = knotCount * coefficientCount;
    const std::uint64_t knotActionCount = knotCount * taskLayout.actionCount;
    const std::uint64_t referenceActionCount =
        referenceCount * taskLayout.actionCount;
    const std::uint64_t referenceSignatureCount =
        referenceCount * pack.signatureCount;
    if (pack.coefficientKnots.size() != knotCoefficientCount ||
        pack.coefficientTangents.size() != knotCoefficientCount ||
        pack.authorityKnots.size() != knotActionCount ||
        pack.authorityTangents.size() != knotActionCount ||
        pack.phaseRateKnots.size() != knotCount ||
        pack.phaseRateTangents.size() != knotCount ||
        pack.referenceActions.size() != referenceActionCount ||
        pack.referenceSignatures.size() != referenceSignatureCount ||
        pack.signatureWeights.size() != pack.signatureCount ||
        pack.referenceContactMasks.size() != referenceCount ||
        pack.contactGroupIndices.size() != pack.contactTrackCount ||
        !finite(pack.coefficientKnots) ||
        !finite(pack.coefficientTangents) ||
        !finite(pack.authorityKnots) ||
        !finite(pack.authorityTangents) ||
        !finite(pack.phaseRateKnots) ||
        !finite(pack.phaseRateTangents) ||
        !finite(pack.referenceActions) ||
        !finite(pack.referenceSignatures) ||
        !finite(pack.signatureWeights)) {
        return reject(
            HyperPolicyCompileStatus::invalidPack,
            "phaseTables",
            "generated knots, references, or phase-alignment tables have invalid dimensions"
        );
    }
    for (std::size_t index = 0u;
         index < pack.coefficientKnots.size();
         ++index) {
        const float limit = pack.coefficientLimits[index % coefficientCount];
        if (std::abs(pack.coefficientKnots[index]) > limit + 1.0e-5f) {
            return reject(
                HyperPolicyCompileStatus::invalidPack,
                "coefficientKnots",
                "a generated coefficient exceeds its immutable base limit"
            );
        }
    }
    if (std::any_of(
            pack.authorityKnots.begin(),
            pack.authorityKnots.end(),
            [](const float value) { return value < 0.0f || value > 1.0f; }
        ) ||
        std::any_of(
            pack.phaseRateKnots.begin(),
            pack.phaseRateKnots.end(),
            [](const float value) { return value < 0.0f; }
        ) ||
        std::any_of(
            pack.signatureWeights.begin(),
            pack.signatureWeights.end(),
            [](const float value) { return !(value > 0.0f); }
        )) {
        return reject(
            HyperPolicyCompileStatus::invalidPack,
            "phaseTables",
            "authority, phase rate, and signature weights are outside their ranges"
        );
    }
    const std::uint32_t contactMask = pack.contactTrackCount == 32u
        ? std::numeric_limits<std::uint32_t>::max()
        : (1u << pack.contactTrackCount) - 1u;
    if (std::any_of(
            pack.referenceContactMasks.begin(),
            pack.referenceContactMasks.end(),
            [contactMask](const std::uint32_t value) {
                return (value & ~contactMask) != 0u;
            }
        )) {
        return reject(
            HyperPolicyCompileStatus::invalidPack,
            "referenceContactMasks",
            "reference contact intent exceeds the authored track count"
        );
    }
    float previousEventPhase = -1.0f;
    for (std::size_t index = 0u; index < pack.events.size(); ++index) {
        const HyperPolicyEventGuard& event = pack.events[index];
        if (!std::isfinite(event.phase) || event.phase < 0.0f ||
            event.phase > 1.0f || event.phase < previousEventPhase ||
            !std::isfinite(event.confidence) || event.confidence < 0.0f ||
            event.confidence > 1.0f || event.minimumDwellSteps == 0u ||
            (event.requiredContactOnMask & ~contactMask) != 0u ||
            (event.requiredContactOffMask & ~contactMask) != 0u ||
            (event.requiredContactOnMask & event.requiredContactOffMask) != 0u) {
            return reject(
                HyperPolicyCompileStatus::invalidPack,
                "events[" + std::to_string(index) + "]",
                "event ordering, confidence, dwell, or contact guard is invalid"
            );
        }
        previousEventPhase = event.phase;
    }
    if (pack.events.size() > std::numeric_limits<std::uint32_t>::max() ||
        pack.layers.size() > std::numeric_limits<std::uint32_t>::max()) {
        return reject(
            HyperPolicyCompileStatus::arithmeticOverflow,
            "program",
            "layer or event count exceeds the native ABI"
        );
    }

    auto staged = std::make_shared<CompiledHyperPolicyProgram::Storage>();
    staged->taskFingerprint = task.fingerprint();
    staged->revision = pack.revision;
    staged->layout = {
        .layerCount = static_cast<std::uint32_t>(pack.layers.size()),
        .actorObservationCount = taskLayout.actorObservationSize,
        .actionCount = taskLayout.actionCount,
        .coefficientCount = coefficientCount,
        .knotCount = static_cast<std::uint32_t>(pack.knotPhases.size()),
        .referenceCount = static_cast<std::uint32_t>(pack.referencePhases.size()),
        .signatureCount = pack.signatureCount,
        .eventCount = static_cast<std::uint32_t>(pack.events.size()),
        .contactTrackCount = pack.contactTrackCount,
        .maximumRank = maximumRank,
        .stochastic = stochastic,
    };
    staged->actionLower = pack.actionLower;
    staged->actionUpper = pack.actionUpper;
    staged->maximumActionRate = pack.maximumActionRate;

    std::uint32_t coefficientOffset = 0u;
    staged->layers.reserve(pack.layers.size());
    for (const HyperPolicyLayer& source : pack.layers) {
        MRHyperPolicyLayerGPU layer{};
        layer.counts = {
            source.inputCount,
            source.outputCount,
            source.rank,
            static_cast<std::uint32_t>(source.activation),
        };
        layer.offsets0 = {
            appendArena<float>(staged->arena, source.weights),
            appendArena<float>(staged->arena, source.bias),
            appendArena<float>(staged->arena, source.adapterDown),
            appendArena<float>(staged->arena, source.adapterUp),
        };
        layer.offsets1 = {
            appendArena<float>(staged->arena, source.adapterBiasBasis),
            coefficientOffset,
            0u,
            0u,
        };
        coefficientOffset += source.rank;
        staged->layers.push_back(layer);
    }

    std::vector<MRHyperPolicyEventGuardGPU> events;
    events.reserve(pack.events.size());
    for (const HyperPolicyEventGuard& source : pack.events) {
        events.push_back({
            .phase = {source.phase, source.confidence, 0.0f, 0.0f},
            .contract = {
                source.requiredContactOnMask,
                source.requiredContactOffMask,
                source.minimumDwellSteps,
                source.kind,
            },
        });
    }

    const std::uint32_t layerTableOffset = appendArena<MRHyperPolicyLayerGPU>(
        staged->arena, staged->layers
    );
    const std::uint32_t observationMeanOffset = appendArena<float>(
        staged->arena, pack.observationMean
    );
    const std::uint32_t observationInverseStdOffset = appendArena<float>(
        staged->arena, pack.observationInverseStandardDeviation
    );
    const std::uint32_t coefficientLimitOffset = appendArena<float>(
        staged->arena, pack.coefficientLimits
    );
    const std::uint32_t actionBiasOffset = appendArena<float>(
        staged->arena, pack.actionBias
    );
    const std::uint32_t actionScaleOffset = appendArena<float>(
        staged->arena, pack.actionScale
    );
    const std::uint32_t knotPhaseOffset = appendArena<float>(
        staged->arena, pack.knotPhases
    );
    const std::uint32_t coefficientKnotOffset = appendArena<float>(
        staged->arena, pack.coefficientKnots
    );
    const std::uint32_t coefficientTangentOffset = appendArena<float>(
        staged->arena, pack.coefficientTangents
    );
    const std::uint32_t authorityKnotOffset = appendArena<float>(
        staged->arena, pack.authorityKnots
    );
    const std::uint32_t authorityTangentOffset = appendArena<float>(
        staged->arena, pack.authorityTangents
    );
    const std::uint32_t phaseRateKnotOffset = appendArena<float>(
        staged->arena, pack.phaseRateKnots
    );
    const std::uint32_t phaseRateTangentOffset = appendArena<float>(
        staged->arena, pack.phaseRateTangents
    );
    const std::uint32_t referencePhaseOffset = appendArena<float>(
        staged->arena, pack.referencePhases
    );
    const std::uint32_t referenceActionOffset = appendArena<float>(
        staged->arena, pack.referenceActions
    );
    const std::uint32_t referenceSignatureOffset = appendArena<float>(
        staged->arena, pack.referenceSignatures
    );
    const std::uint32_t signatureWeightOffset = appendArena<float>(
        staged->arena, pack.signatureWeights
    );
    const std::uint32_t contactMaskOffset = appendArena<std::uint32_t>(
        staged->arena, pack.referenceContactMasks
    );
    const std::uint32_t contactGroupIndexOffset = appendArena<std::uint32_t>(
        staged->arena, pack.contactGroupIndices
    );
    const std::uint32_t eventOffset = appendArena<MRHyperPolicyEventGuardGPU>(
        staged->arena, events
    );
    const std::uint32_t logStdOffset = appendArena<float>(
        staged->arena, pack.actionLogStandardDeviation
    );
    const std::uint32_t offsets[] = {
        layerTableOffset,
        observationMeanOffset,
        observationInverseStdOffset,
        coefficientLimitOffset,
        actionBiasOffset,
        actionScaleOffset,
        knotPhaseOffset,
        coefficientKnotOffset,
        coefficientTangentOffset,
        authorityKnotOffset,
        authorityTangentOffset,
        phaseRateKnotOffset,
        phaseRateTangentOffset,
        referencePhaseOffset,
        referenceActionOffset,
        referenceSignatureOffset,
        signatureWeightOffset,
        contactMaskOffset,
        contactGroupIndexOffset,
        eventOffset,
        logStdOffset,
    };
    if (std::any_of(
            std::begin(offsets),
            std::end(offsets),
            [](const std::uint32_t value) {
                return value == MR_HYPER_POLICY_INVALID_INDEX;
            }
        )) {
        return reject(
            HyperPolicyCompileStatus::arithmeticOverflow,
            "arena",
            "immutable hyper-policy arena exceeds 32-bit Metal offsets"
        );
    }


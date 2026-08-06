std::span<const float>
CompiledHyperPolicyProgram::actionLower() const noexcept {
    return valid()
        ? std::span<const float>{storage_->actionLower}
        : std::span<const float>{};
}

std::span<const float>
CompiledHyperPolicyProgram::actionUpper() const noexcept {
    return valid()
        ? std::span<const float>{storage_->actionUpper}
        : std::span<const float>{};
}

std::span<const float>
CompiledHyperPolicyProgram::maximumActionRate() const noexcept {
    return valid()
        ? std::span<const float>{storage_->maximumActionRate}
        : std::span<const float>{};
}

HyperPolicyCompileDiagnostics compileHyperPolicyProgram(
    const HyperPolicyPack& pack,
    const CompiledTaskProgram& task,
    CompiledHyperPolicyProgram& output
) {
    if (!task.valid()) {
        return reject(
            HyperPolicyCompileStatus::invalidTask,
            "task",
            "hyper-policy compilation requires a valid compiled task"
        );
    }
    const TaskProgramLayout& taskLayout = task.layout();
    const bool stochastic = !pack.actionLogStandardDeviation.empty();
    if (pack.id.empty() || pack.revision == 0u ||
        pack.sourceMotionFingerprint == 0u || pack.layers.empty() ||
        !std::isfinite(pack.observationClip) ||
        !(pack.observationClip > 0.0f) ||
        !std::isfinite(pack.actionClip) || !(pack.actionClip > 0.0f) ||
        !std::isfinite(pack.maximumPhaseAdvancePerStep) ||
        pack.maximumPhaseAdvancePerStep < 0.0f ||
        !std::isfinite(pack.phaseAlignmentBlend) ||
        pack.phaseAlignmentBlend < 0.0f ||
        pack.phaseAlignmentBlend > 1.0f ||
        !std::isfinite(pack.phaseAlignmentHuberDelta) ||
        !(pack.phaseAlignmentHuberDelta > 0.0f) ||
        !std::isfinite(pack.controlTimeStep) ||
        !(pack.controlTimeStep > 0.0f)) {
        return reject(
            HyperPolicyCompileStatus::invalidPack,
            "policy",
            "hyper-policy identity, revision, source, clipping, or phase control is invalid"
        );
    }
    if (!pack.contract.exact()) {
        return reject(
            HyperPolicyCompileStatus::invalidPack,
            "contract",
            "hyper-policy contract must be a complete version-1 binding"
        );
    }
    if (pack.contract.worldFingerprint != task.worldFingerprint() ||
        pack.contract.taskFingerprint != task.fingerprint() ||
        pack.contract.observationFingerprint != task.observationFingerprint() ||
        pack.contract.actionFingerprint != task.actionFingerprint()) {
        std::ostringstream details;
        details << "policy=(world=" << pack.contract.worldFingerprint
                << ",task=" << pack.contract.taskFingerprint
                << ",observation=" << pack.contract.observationFingerprint
                << ",action=" << pack.contract.actionFingerprint
                << ") task=(world=" << task.worldFingerprint()
                << ",task=" << task.fingerprint()
                << ",observation=" << task.observationFingerprint()
                << ",action=" << task.actionFingerprint() << ")";
        return reject(
            HyperPolicyCompileStatus::incompatibleContract,
            "contract",
            details.str()
        );
    }
    if (pack.observationMean.size() != taskLayout.actorObservationSize ||
        pack.observationInverseStandardDeviation.size() !=
            taskLayout.actorObservationSize ||
        pack.actionBias.size() != taskLayout.actionCount ||
        pack.actionScale.size() != taskLayout.actionCount ||
        (stochastic && pack.actionLogStandardDeviation.size() !=
            taskLayout.actionCount) ||
        pack.actionLower.size() != taskLayout.actionCount ||
        pack.actionUpper.size() != taskLayout.actionCount ||
        pack.maximumActionRate.size() != taskLayout.actionCount ||
        !finite(pack.observationMean) ||
        !finite(pack.observationInverseStandardDeviation) ||
        !finite(pack.actionBias) || !finite(pack.actionScale) ||
        !finite(pack.actionLogStandardDeviation) ||
        !finite(pack.actionLower) || !finite(pack.actionUpper) ||
        !finite(pack.maximumActionRate)) {
        return reject(
            HyperPolicyCompileStatus::incompatibleContract,
            "normalization",
            "observation, distribution, action transform, or actuator envelope does not match the task"
        );
    }
    if (std::any_of(
            pack.observationInverseStandardDeviation.begin(),
            pack.observationInverseStandardDeviation.end(),
            [](const float value) { return !(value > 0.0f); }
        ) ||
        std::any_of(
            pack.actionScale.begin(),
            pack.actionScale.end(),
            [](const float value) { return std::abs(value) <= 1.0e-12f; }
        ) ||
        std::any_of(
            pack.maximumActionRate.begin(),
            pack.maximumActionRate.end(),
            [](const float value) { return !(value > 0.0f); }
        )) {
        return reject(
            HyperPolicyCompileStatus::invalidPack,
            "normalization",
            "inverse standard deviations, action scales, and action rates must be nonzero"
        );
    }
    for (std::uint32_t action = 0u;
         action < taskLayout.actionCount;
         ++action) {
        if (!(pack.actionLower[action] < pack.actionUpper[action])) {
            return reject(
                HyperPolicyCompileStatus::invalidPack,
                "actionEnvelope",
                "every lower action bound must be strictly below its upper bound"
            );
        }
    }
    if (stochastic && std::any_of(
            pack.actionLogStandardDeviation.begin(),
            pack.actionLogStandardDeviation.end(),
            [](const float value) { return value < -5.0f || value > 2.0f; }
        )) {
        return reject(
            HyperPolicyCompileStatus::invalidPack,
            "actionLogStandardDeviation",
            "Gaussian log standard deviations must remain within [-5, 2]"
        );
    }

    std::uint64_t coefficientCount64 = 0u;
    std::uint32_t expectedInput = taskLayout.actorObservationSize;
    std::uint32_t maximumRank = 0u;
    for (std::size_t index = 0u; index < pack.layers.size(); ++index) {
        const HyperPolicyLayer& layer = pack.layers[index];
        const std::uint64_t weightCount =
            static_cast<std::uint64_t>(layer.inputCount) * layer.outputCount;
        const std::uint64_t downCount =
            static_cast<std::uint64_t>(layer.rank) * layer.inputCount;
        const std::uint64_t upCount =
            static_cast<std::uint64_t>(layer.outputCount) * layer.rank;
        const bool final = index + 1u == pack.layers.size();
        if (layer.inputCount != expectedInput || layer.outputCount == 0u ||
            layer.rank == 0u ||
            layer.rank > std::min(layer.inputCount, layer.outputCount) ||
            weightCount != layer.weights.size() ||
            layer.bias.size() != layer.outputCount ||
            downCount != layer.adapterDown.size() ||
            upCount != layer.adapterUp.size() ||
            upCount != layer.adapterBiasBasis.size() ||
            !supportedActivation(layer.activation) ||
            !finite(layer.weights) || !finite(layer.bias) ||
            !finite(layer.adapterDown) || !finite(layer.adapterUp) ||
            !finite(layer.adapterBiasBasis) ||
            (final && (layer.outputCount != taskLayout.actionCount ||
                layer.activation != PolicyActivation::identity)) ||
            (!final && layer.activation == PolicyActivation::identity)) {
            return reject(
                HyperPolicyCompileStatus::incompatibleContract,
                "layers[" + std::to_string(index) + "]",
                "base layer or low-rank adapter topology is invalid"
            );
        }
        coefficientCount64 += layer.rank;
        maximumRank = std::max(maximumRank, layer.rank);
        expectedInput = layer.outputCount;
    }
    if (coefficientCount64 == 0u ||
        coefficientCount64 > std::numeric_limits<std::uint32_t>::max()) {
        return reject(
            HyperPolicyCompileStatus::arithmeticOverflow,
            "coefficientCount",
            "generated coefficient count exceeds the native ABI"
        );
    }
    const auto coefficientCount =
        static_cast<std::uint32_t>(coefficientCount64);
    if (pack.coefficientLimits.size() != coefficientCount ||
        !finite(pack.coefficientLimits) ||
        std::any_of(
            pack.coefficientLimits.begin(),
            pack.coefficientLimits.end(),
            [](const float value) { return !(value > 0.0f); }
        )) {
        return reject(
            HyperPolicyCompileStatus::invalidPack,
            "coefficientLimits",
            "every generated coefficient requires a finite positive bound"
        );
    }


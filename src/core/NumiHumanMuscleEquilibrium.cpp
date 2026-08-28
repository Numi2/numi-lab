#include "metalrobo/NumiHumanMuscleEquilibrium.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace metalrobo {
namespace {

constexpr double kMinimum = 1.0e-12;

struct PoseState {
    std::vector<double> q;
    std::vector<double> activation;
    std::vector<double> fiberLength;
    std::vector<double> muscleForce;
    std::vector<double> target;
    std::vector<double> residual;
    std::vector<double> limitForce;
    std::vector<double> equalityForce;
    std::vector<double> weights;
    double residualRms = 0.0;
    double maximumResidual = 0.0;
    double objective = 0.0;
    std::uint32_t activationSweeps = 0u;
};

struct ResolvedMuscle {
    double pathLength = 0.0;
    std::vector<double> jacobian;
};

NumiHumanMuscleEquilibriumDiagnostics failure(
    const NumiHumanMuscleEquilibriumStatus status,
    const std::uint32_t failingIndex = MR_INVALID_INDEX
) {
    NumiHumanMuscleEquilibriumDiagnostics diagnostics;
    diagnostics.status = status;
    diagnostics.failingIndex = failingIndex;
    return diagnostics;
}

bool finiteSpan(const std::span<const double> values) {
    return std::all_of(values.begin(), values.end(), [](const double value) {
        return std::isfinite(value);
    });
}

double square(const double value) { return value * value; }

double activationValue(
    const std::vector<double>& samples,
    const std::size_t muscle,
    const std::uint32_t sampleCount,
    const double activation,
    const double activationLimit
) {
    if (activation <= 0.0) return samples[muscle * sampleCount];
    if (activation >= activationLimit) {
        return samples[muscle * sampleCount + sampleCount - 1u];
    }
    const double coordinate = activation / activationLimit *
        static_cast<double>(sampleCount - 1u);
    const std::uint32_t lower = static_cast<std::uint32_t>(coordinate);
    const double fraction = coordinate - static_cast<double>(lower);
    const std::size_t base = muscle * sampleCount + lower;
    return samples[base] + fraction * (samples[base + 1u] - samples[base]);
}

NumiHumanMuscleEquilibriumDiagnostics resolveMuscles(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const MujocoMuscleSite> sites,
    const std::span<const MujocoWrapGeometry> wraps,
    const std::span<const MujocoMuscleDefinition> muscles,
    std::vector<ResolvedMuscle>& resolved,
    const ArticulatedDynamicsConfig& dynamicsConfig
) {
    const std::size_t nv = model.articulations[articulationIndex].nv;
    const std::vector<double> zeroVelocity(nv, 0.0);
    std::vector<ResolvedMuscle> candidate(muscles.size());
    for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
        MujocoMuscleResult path;
        const auto diagnostics = evaluateMujocoMuscle(
            model, articulationIndex, q, zeroVelocity, sites, wraps,
            muscles[muscle], {}, path, dynamicsConfig
        );
        if (!diagnostics.succeeded()) {
            auto failed = failure(
                NumiHumanMuscleEquilibriumStatus::kinematicsFailure,
                static_cast<std::uint32_t>(muscle)
            );
            failed.muscleStatus = diagnostics.status;
            return failed;
        }
        if (!(path.path.length > kMinimum) ||
            path.path.lengthJacobian.size() != nv ||
            !finiteSpan(path.path.lengthJacobian)) {
            return failure(
                NumiHumanMuscleEquilibriumStatus::kinematicsFailure,
                static_cast<std::uint32_t>(muscle)
            );
        }
        candidate[muscle].pathLength = path.path.length;
        candidate[muscle].jacobian = std::move(path.path.lengthJacobian);
    }
    resolved = std::move(candidate);
    return {};
}

NumiHumanMuscleEquilibriumDiagnostics evaluateStaticForce(
    const double pathLength,
    const double activation,
    const double timestep,
    const MujocoMuscleDefinition& definition,
    const MujocoCompliantMuscleArchitecture& architecture,
    double& force,
    double& fiberLength,
    const std::uint32_t muscleIndex
) {
    if (!(architecture.optimalFiberLength > 0.0) ||
        !(architecture.tendonSlackLength > 0.0)) {
        const auto sourceDiagnostics = evaluateMujocoMuscleForceLaw(
            pathLength, 0.0, definition,
            {.excitation = activation, .activation = activation}, force
        );
        if (!sourceDiagnostics.succeeded()) {
            auto failed = failure(
                NumiHumanMuscleEquilibriumStatus::muscleFailure, muscleIndex
            );
            failed.muscleStatus = sourceDiagnostics.status;
            return failed;
        }
        fiberLength = pathLength;
        return {};
    }
    MujocoCompliantMuscleResult compliant;
    const auto diagnostics = evaluateMujocoCompliantMuscle(
        pathLength, 0.0, timestep, definition, architecture,
        {.excitation = activation, .activation = activation}, compliant
    );
    if (!diagnostics.succeeded()) {
        auto failed = failure(
            NumiHumanMuscleEquilibriumStatus::muscleFailure, muscleIndex
        );
        failed.muscleStatus = diagnostics.status;
        return failed;
    }
    force = compliant.actuatorForce;
    fiberLength = compliant.candidateFiberLength;
    if (!std::isfinite(force) || !(fiberLength > 0.0) ||
        !std::isfinite(fiberLength)) {
        return failure(
            NumiHumanMuscleEquilibriumStatus::nonfiniteResult, muscleIndex
        );
    }
    return {};
}

NumiHumanMuscleEquilibriumDiagnostics gravityTarget(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    std::vector<double>& target,
    const ArticulatedDynamicsConfig& dynamicsConfig
) {
    const std::size_t nv = model.articulations[articulationIndex].nv;
    const std::vector<double> zero(nv, 0.0);
    std::vector<double> candidate(nv, 0.0);
    const auto diagnostics = computeArticulatedInverseDynamics(
        model, articulationIndex, q, zero, zero, {}, candidate,
        dynamicsConfig
    );
    if (!diagnostics.succeeded()) {
        auto failed = failure(
            NumiHumanMuscleEquilibriumStatus::dynamicsFailure
        );
        failed.dynamicsStatus = diagnostics.status;
        return failed;
    }
    if (!finiteSpan(candidate)) {
        return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
    }
    target = std::move(candidate);
    return {};
}

double posePenalty(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> initialQ,
    const std::span<const double> q
) {
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    double penalty = 0.0;
    std::size_t count = 0u;
    for (std::uint32_t localV = 0u; localV < articulation.nv; ++localV) {
        const MRDofPropertiesGPU& dof =
            model.dofs[articulation.vOffset + localV];
        if ((dof.flags & MR_DOF_FLAG_ROOT) != 0u ||
            (dof.flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u ||
            dof.qIndex == MR_INVALID_INDEX ||
            dof.qIndex < articulation.qOffset ||
            dof.qIndex >= articulation.qOffset + articulation.nq) {
            continue;
        }
        const double range = static_cast<double>(dof.limits.y) - dof.limits.x;
        if (!(range > kMinimum) || !std::isfinite(range)) continue;
        const std::size_t localQ = dof.qIndex - articulation.qOffset;
        penalty += square((q[localQ] - initialQ[localQ]) / range);
        ++count;
    }
    return count == 0u ? 0.0 : penalty / static_cast<double>(count);
}

void finishResidual(
    const MRArticulationGPU& articulation,
    const NumiHumanMuscleEquilibriumConfig& config,
    const std::span<const double> initialQ,
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const MRNumiHumanJointEqualityGPU> equalities,
    PoseState& state
) {
    const std::size_t firstInternal =
        articulation.rootType == MR_ROOT_FLOATING ? 6u : 0u;
    double sum = 0.0;
    state.maximumResidual = 0.0;
    state.limitForce.assign(articulation.nv, 0.0);
    state.equalityForce.assign(articulation.nv, 0.0);
    for (std::size_t dof = firstInternal; dof < articulation.nv; ++dof) {
        state.residual[dof] = state.muscleForce[dof] - state.target[dof];
    }
    std::vector<std::uint8_t> equalityDependent(articulation.nv, 0u);
    for (const auto& equality : equalities) {
        NumiHumanJointEqualityEvaluation evaluation;
        const auto diagnostics = evaluateNumiHumanJointEquality(
            equality, state.q, evaluation
        );
        if (!diagnostics.succeeded()) {
            state.residualRms = std::numeric_limits<double>::infinity();
            state.objective = std::numeric_limits<double>::infinity();
            return;
        }
        const std::size_t dependent = equality.indices.y;
        const double dependentResidual = state.residual[dependent];
        state.equalityForce[dependent] -= dependentResidual;
        state.residual[dependent] = 0.0;
        state.weights[dependent] = 0.0;
        equalityDependent[dependent] = 1u;
        if (equality.indices.w != MR_INVALID_INDEX) {
            state.equalityForce[equality.indices.w] +=
                evaluation.derivative * dependentResidual;
            state.residual[equality.indices.w] +=
                evaluation.derivative * dependentResidual;
        }
    }
    std::size_t rowCount = 0u;
    for (std::size_t dof = firstInternal; dof < articulation.nv; ++dof) {
        const double rawResidual = state.residual[dof];
        const MRDofPropertiesGPU& properties =
            model.dofs[articulation.vOffset + dof];
        if (equalityDependent[dof] == 0u &&
            (properties.flags & MR_DOF_FLAG_POSITION_LIMIT) != 0u &&
            properties.qIndex != MR_INVALID_INDEX &&
            properties.qIndex >= articulation.qOffset &&
            properties.qIndex < articulation.qOffset + articulation.nq) {
            const double position = state.q[
                properties.qIndex - articulation.qOffset
            ];
            const double lowerGap = position - properties.limits.x;
            const double upperGap = properties.limits.y - position;
            if (rawResidual < 0.0 &&
                lowerGap <= config.positionLimitTolerance) {
                state.limitForce[dof] = -rawResidual;
            } else if (rawResidual > 0.0 &&
                       upperGap <= config.positionLimitTolerance) {
                state.limitForce[dof] = -rawResidual;
            }
        }
        state.residual[dof] = rawResidual + state.limitForce[dof];
        const double normalized = state.weights[dof] * state.residual[dof];
        sum += normalized * normalized;
        if (state.weights[dof] > 0.0) ++rowCount;
        state.maximumResidual = std::max(
            state.maximumResidual, std::abs(state.residual[dof])
        );
    }
    state.residualRms = rowCount == 0u
        ? 0.0
        : std::sqrt(sum / static_cast<double>(rowCount));
    double activationPenalty = 0.0;
    for (const double value : state.activation) {
        activationPenalty += value * value;
    }
    if (!state.activation.empty()) {
        activationPenalty /= static_cast<double>(state.activation.size());
    }
    state.objective = state.residualRms * state.residualRms +
        config.activationRegularization * activationPenalty +
        config.poseRegularization * posePenalty(
            model, articulationIndex, initialQ, state.q
        );
}

NumiHumanMuscleEquilibriumDiagnostics solveActivation(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> initialQ,
    const std::span<const MujocoMuscleSite> sites,
    const std::span<const MujocoWrapGeometry> wraps,
    const std::span<const MujocoMuscleDefinition> muscles,
    const std::span<const MujocoCompliantMuscleArchitecture> architectures,
    const std::span<const MRNumiHumanJointEqualityGPU> equalities,
    const std::span<const std::uint8_t> recruited,
    const NumiHumanMuscleEquilibriumConfig& config,
    const ArticulatedDynamicsConfig& dynamicsConfig,
    PoseState& state
) {
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    const std::size_t nv = articulation.nv;
    const std::size_t firstInternal =
        articulation.rootType == MR_ROOT_FLOATING ? 6u : 0u;
    std::vector<ResolvedMuscle> resolved;
    auto diagnostics = resolveMuscles(
        model, articulationIndex, state.q, sites, wraps, muscles, resolved,
        dynamicsConfig
    );
    if (!diagnostics.succeeded()) return diagnostics;
    diagnostics = gravityTarget(
        model, articulationIndex, state.q, state.target, dynamicsConfig
    );
    if (!diagnostics.succeeded()) return diagnostics;

    std::vector<std::vector<double>> objectiveJacobians;
    objectiveJacobians.reserve(resolved.size());
    for (const auto& muscle : resolved) {
        objectiveJacobians.push_back(muscle.jacobian);
    }
    std::vector<double> objectiveTarget = state.target;
    std::vector<std::uint8_t> equalityDependent(nv, 0u);
    for (std::size_t index = 0u; index < equalities.size(); ++index) {
        NumiHumanJointEqualityEvaluation evaluation;
        const auto equalityDiagnostics = evaluateNumiHumanJointEquality(
            equalities[index], state.q, evaluation
        );
        if (!equalityDiagnostics.succeeded()) {
            auto failed = failure(
                NumiHumanMuscleEquilibriumStatus::equalityFailure,
                static_cast<std::uint32_t>(index)
            );
            failed.equalityStatus = equalityDiagnostics.status;
            return failed;
        }
        const std::size_t dependent = equalities[index].indices.y;
        equalityDependent[dependent] = 1u;
        if (equalities[index].indices.w != MR_INVALID_INDEX) {
            const std::size_t master = equalities[index].indices.w;
            objectiveTarget[master] +=
                evaluation.derivative * objectiveTarget[dependent];
            for (auto& jacobian : objectiveJacobians) {
                jacobian[master] +=
                    evaluation.derivative * jacobian[dependent];
            }
        }
        objectiveTarget[dependent] = 0.0;
        for (auto& jacobian : objectiveJacobians) {
            jacobian[dependent] = 0.0;
        }
    }

    state.weights.assign(nv, 0.0);
    for (std::size_t dof = firstInternal; dof < nv; ++dof) {
        if (equalityDependent[dof] != 0u) continue;
        state.weights[dof] = 1.0 / std::max(
            config.minimumGeneralizedForceScale,
            std::abs(objectiveTarget[dof])
        );
    }
    const std::uint32_t sampleCount = config.activationSamples;
    std::vector<double> forceSamples(muscles.size() * sampleCount, 0.0);
    state.activation.assign(muscles.size(), 0.0);
    state.fiberLength.assign(muscles.size(), 0.0);
    state.muscleForce.assign(nv, 0.0);
    for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
        const std::uint32_t lastSample = recruited[muscle] != 0u
            ? sampleCount : 1u;
        for (std::uint32_t sample = 0u; sample < lastSample; ++sample) {
            const double activation = config.activationLimit *
                static_cast<double>(sample) /
                static_cast<double>(sampleCount - 1u);
            double fiber = 0.0;
            diagnostics = evaluateStaticForce(
                resolved[muscle].pathLength, activation, config.timestep,
                muscles[muscle], architectures[muscle],
                forceSamples[muscle * sampleCount + sample], fiber,
                static_cast<std::uint32_t>(muscle)
            );
            if (!diagnostics.succeeded()) return diagnostics;
            if (sample == 0u) state.fiberLength[muscle] = fiber;
        }
        if (recruited[muscle] == 0u) {
            std::fill_n(
                forceSamples.begin() + muscle * sampleCount,
                sampleCount,
                forceSamples[muscle * sampleCount]
            );
        }
        const double passive = forceSamples[muscle * sampleCount];
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            state.muscleForce[dof] +=
                passive * objectiveJacobians[muscle][dof];
        }
    }
    state.residual.assign(nv, 0.0);
    for (std::size_t dof = firstInternal; dof < nv; ++dof) {
        state.residual[dof] = state.weights[dof] *
            (state.muscleForce[dof] - objectiveTarget[dof]);
    }

    std::uint32_t completedSweeps = 0u;
    for (std::uint32_t sweep = 0u; sweep < config.activationSweeps; ++sweep) {
        double maximumChange = 0.0;
        for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
            if (recruited[muscle] == 0u) continue;
            const double currentActivation = state.activation[muscle];
            const double currentForce = activationValue(
                forceSamples, muscle, sampleCount, currentActivation,
                config.activationLimit
            );
            double bestActivation = currentActivation;
            double bestObjective = std::numeric_limits<double>::infinity();
            for (std::uint32_t sample = 0u; sample + 1u < sampleCount;
                 ++sample) {
                const double lowerActivation = config.activationLimit *
                    static_cast<double>(sample) /
                    static_cast<double>(sampleCount - 1u);
                const double upperActivation = config.activationLimit *
                    static_cast<double>(sample + 1u) /
                    static_cast<double>(sampleCount - 1u);
                const double interval = upperActivation - lowerActivation;
                const std::size_t sampleBase = muscle * sampleCount + sample;
                const double lowerForce = forceSamples[sampleBase];
                const double forceSlope =
                    (forceSamples[sampleBase + 1u] - lowerForce) / interval;
                double gradient =
                    config.activationRegularization * lowerActivation;
                double curvature = config.activationRegularization;
                for (std::size_t dof = firstInternal; dof < nv; ++dof) {
                    const double direction = state.weights[dof] *
                        objectiveJacobians[muscle][dof];
                    const double base = state.residual[dof] + direction *
                        (lowerForce - currentForce);
                    const double column = direction * forceSlope;
                    gradient += column * base;
                    curvature += column * column;
                }
                const double delta = std::clamp(
                    -gradient / std::max(kMinimum, curvature), 0.0, interval
                );
                const double candidateActivation = lowerActivation + delta;
                double objective = 0.5 * config.activationRegularization *
                    candidateActivation * candidateActivation;
                for (std::size_t dof = firstInternal; dof < nv; ++dof) {
                    const double direction = state.weights[dof] *
                        objectiveJacobians[muscle][dof];
                    const double base = state.residual[dof] + direction *
                        (lowerForce - currentForce);
                    const double candidate = base +
                        direction * forceSlope * delta;
                    objective += 0.5 * candidate * candidate;
                }
                if (objective < bestObjective) {
                    bestObjective = objective;
                    bestActivation = candidateActivation;
                }
            }
            const double nextForce = activationValue(
                forceSamples, muscle, sampleCount, bestActivation,
                config.activationLimit
            );
            for (std::size_t dof = firstInternal; dof < nv; ++dof) {
                state.residual[dof] += state.weights[dof] *
                    objectiveJacobians[muscle][dof] *
                    (nextForce - currentForce);
            }
            maximumChange = std::max(
                maximumChange,
                std::abs(bestActivation - currentActivation)
            );
            state.activation[muscle] = bestActivation;
        }
        completedSweeps = sweep + 1u;
        if (maximumChange < config.activationConvergence) break;
    }
    state.activationSweeps = completedSweeps;

    // Publish exact force-law values, not their piecewise-linear optimizer
    // samples. This also supplies the accepted FP64 fibre state.
    std::fill(state.muscleForce.begin(), state.muscleForce.end(), 0.0);
    for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
        double force = 0.0;
        diagnostics = evaluateStaticForce(
            resolved[muscle].pathLength, state.activation[muscle],
            config.timestep, muscles[muscle], architectures[muscle], force,
            state.fiberLength[muscle], static_cast<std::uint32_t>(muscle)
        );
        if (!diagnostics.succeeded()) return diagnostics;
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            state.muscleForce[dof] +=
                force * resolved[muscle].jacobian[dof];
        }
    }
    state.residual.assign(nv, 0.0);
    finishResidual(
        articulation, config, initialQ, model, articulationIndex, equalities,
        state
    );
    if (!std::isfinite(state.residualRms) ||
        !std::isfinite(state.maximumResidual) ||
        !std::isfinite(state.objective)) {
        return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
    }
    return {};
}

NumiHumanMuscleEquilibriumDiagnostics evaluatePoseWithActivation(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> initialQ,
    const std::span<const MujocoMuscleSite> sites,
    const std::span<const MujocoWrapGeometry> wraps,
    const std::span<const MujocoMuscleDefinition> muscles,
    const std::span<const MujocoCompliantMuscleArchitecture> architectures,
    const std::span<const MRNumiHumanJointEqualityGPU> equalities,
    const NumiHumanMuscleEquilibriumConfig& config,
    const ArticulatedDynamicsConfig& dynamicsConfig,
    PoseState& state
) {
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    std::vector<ResolvedMuscle> resolved;
    auto diagnostics = resolveMuscles(
        model, articulationIndex, state.q, sites, wraps, muscles, resolved,
        dynamicsConfig
    );
    if (!diagnostics.succeeded()) return diagnostics;
    diagnostics = gravityTarget(
        model, articulationIndex, state.q, state.target, dynamicsConfig
    );
    if (!diagnostics.succeeded()) return diagnostics;
    state.weights.assign(articulation.nv, 0.0);
    const std::size_t firstInternal =
        articulation.rootType == MR_ROOT_FLOATING ? 6u : 0u;
    for (std::size_t dof = firstInternal; dof < articulation.nv; ++dof) {
        state.weights[dof] = 1.0 / std::max(
            config.minimumGeneralizedForceScale,
            std::abs(state.target[dof])
        );
    }
    state.muscleForce.assign(articulation.nv, 0.0);
    state.fiberLength.assign(muscles.size(), 0.0);
    for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
        double force = 0.0;
        diagnostics = evaluateStaticForce(
            resolved[muscle].pathLength, state.activation[muscle],
            config.timestep, muscles[muscle], architectures[muscle], force,
            state.fiberLength[muscle], static_cast<std::uint32_t>(muscle)
        );
        if (!diagnostics.succeeded()) return diagnostics;
        for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
            state.muscleForce[dof] +=
                force * resolved[muscle].jacobian[dof];
        }
    }
    state.residual.assign(articulation.nv, 0.0);
    finishResidual(
        articulation, config, initialQ, model, articulationIndex, equalities,
        state
    );
    return {};
}

std::vector<std::uint32_t> poseCandidates(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const PoseState& state,
    const std::uint32_t maximumCount
) {
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    std::vector<std::pair<double, std::uint32_t>> ranked;
    for (std::uint32_t localV = 0u; localV < articulation.nv; ++localV) {
        const MRDofPropertiesGPU& dof =
            model.dofs[articulation.vOffset + localV];
        if ((dof.flags & MR_DOF_FLAG_ROOT) != 0u ||
            state.weights[localV] == 0.0 ||
            (dof.flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u ||
            dof.qIndex == MR_INVALID_INDEX ||
            dof.qIndex < articulation.qOffset ||
            dof.qIndex >= articulation.qOffset + articulation.nq ||
            !std::isfinite(dof.limits.x) || !std::isfinite(dof.limits.y) ||
            !(dof.limits.y > dof.limits.x)) {
            continue;
        }
        ranked.emplace_back(
            std::abs(state.weights[localV] * state.residual[localV]), localV
        );
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
        if (left.first != right.first) return left.first > right.first;
        return left.second < right.second;
    });
    if (ranked.size() > maximumCount) ranked.resize(maximumCount);
    std::vector<std::uint32_t> result;
    result.reserve(ranked.size());
    for (const auto& entry : ranked) result.push_back(entry.second);
    return result;
}

double minimumNormalizedLimitMargin(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q
) {
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    double minimum = 1.0;
    bool observed = false;
    for (std::uint32_t localV = 0u; localV < articulation.nv; ++localV) {
        const MRDofPropertiesGPU& dof =
            model.dofs[articulation.vOffset + localV];
        if ((dof.flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u ||
            dof.qIndex == MR_INVALID_INDEX ||
            dof.qIndex < articulation.qOffset ||
            dof.qIndex >= articulation.qOffset + articulation.nq) continue;
        const double range = static_cast<double>(dof.limits.y) - dof.limits.x;
        if (!(range > kMinimum)) continue;
        const double value = q[dof.qIndex - articulation.qOffset];
        minimum = std::min(minimum, std::min(
            (value - dof.limits.x) / range,
            (dof.limits.y - value) / range
        ));
        observed = true;
    }
    return observed ? minimum : 1.0;
}

} // namespace

NumiHumanMuscleEquilibriumDiagnostics compileNumiHumanMuscleEquilibrium(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> initialQ,
    const std::span<const MujocoMuscleSite> sites,
    const std::span<const MujocoWrapGeometry> wraps,
    const std::span<const MujocoMuscleDefinition> muscles,
    const std::span<const MujocoCompliantMuscleArchitecture> architectures,
    const std::span<const MRNumiHumanJointEqualityGPU> jointEqualities,
    const std::span<const std::uint32_t> selectedMuscleIndices,
    NumiHumanMuscleEquilibriumResult& result,
    const NumiHumanMuscleEquilibriumConfig& config
) {
    const bool validConfig = std::isfinite(config.timestep) &&
        config.timestep > 0.0 && std::isfinite(config.activationLimit) &&
        config.activationLimit > 0.0 && config.activationLimit <= 1.0 &&
        config.activationSamples >= 2u && config.activationSamples <= 65u &&
        config.activationSweeps > 0u &&
        std::isfinite(config.activationRegularization) &&
        config.activationRegularization >= 0.0 &&
        std::isfinite(config.activationConvergence) &&
        config.activationConvergence > 0.0 &&
        std::isfinite(config.minimumGeneralizedForceScale) &&
        config.minimumGeneralizedForceScale > 0.0 &&
        std::isfinite(config.balanceTolerance) &&
        config.balanceTolerance > 0.0 && config.poseCandidateCount <= 128u &&
        std::isfinite(config.poseStepFraction) &&
        config.poseStepFraction > 0.0 &&
        std::isfinite(config.maximumPoseStep) &&
        config.maximumPoseStep > 0.0 &&
        std::isfinite(config.positionLimitMarginFraction) &&
        config.positionLimitMarginFraction >= 0.0 &&
        config.positionLimitMarginFraction < 0.5 &&
        std::isfinite(config.poseRegularization) &&
        config.poseRegularization >= 0.0 &&
        std::isfinite(config.poseImprovementTolerance) &&
        config.poseImprovementTolerance >= 0.0 &&
        std::isfinite(config.positionLimitTolerance) &&
        config.positionLimitTolerance >= 0.0;
    if (!validConfig) {
        return failure(
            NumiHumanMuscleEquilibriumStatus::invalidConfiguration
        );
    }
    if (articulationIndex >= model.articulations.size()) {
        return failure(NumiHumanMuscleEquilibriumStatus::invalidArticulation);
    }
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    if (articulation.nv == 0u || articulation.nq == 0u ||
        articulation.vOffset > model.dofs.size() ||
        articulation.nv > model.dofs.size() - articulation.vOffset ||
        initialQ.size() != articulation.nq || muscles.empty() ||
        architectures.size() != muscles.size()) {
        return failure(NumiHumanMuscleEquilibriumStatus::invalidDimensions);
    }
    if (!finiteSpan(initialQ)) {
        return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteInput);
    }
    const auto finite4 = [](const mr_float4 value) {
        return std::isfinite(value.x) && std::isfinite(value.y) &&
            std::isfinite(value.z) && std::isfinite(value.w);
    };
    std::vector<std::uint8_t> dependentDofs(articulation.nv, 0u);
    for (std::size_t index = 0u; index < jointEqualities.size(); ++index) {
        const auto& equality = jointEqualities[index];
        const bool fixed = equality.indices.z == MR_INVALID_INDEX &&
            equality.indices.w == MR_INVALID_INDEX;
        const bool coupled = equality.indices.z < articulation.nq &&
            equality.indices.w < articulation.nv;
        const bool dependentMapping =
            equality.indices.x < articulation.nq &&
            equality.indices.y < articulation.nv &&
            model.dofs[articulation.vOffset + equality.indices.y].qIndex ==
                articulation.qOffset + equality.indices.x;
        const bool masterMapping = fixed ||
            (coupled && model.dofs[
                articulation.vOffset + equality.indices.w
            ].qIndex == articulation.qOffset + equality.indices.z);
        if (!dependentMapping || (!fixed && !coupled) || !masterMapping ||
            (coupled && (equality.indices.x == equality.indices.z ||
                         equality.indices.y == equality.indices.w)) ||
            !finite4(equality.referencesAndCoefficients0) ||
            !finite4(equality.coefficients1) || !finite4(equality.solref) ||
            !finite4(equality.solimp0) || !finite4(equality.solimp1) ||
            equality.coefficients1.w != 0.0f || equality.solref.z != 0.0f ||
            equality.solref.w != 0.0f || equality.solimp1.y != 0.0f ||
            equality.solimp1.z != 0.0f || equality.solimp1.w != 0.0f ||
            dependentDofs[equality.indices.y] != 0u) {
            return failure(
                NumiHumanMuscleEquilibriumStatus::equalityFailure,
                static_cast<std::uint32_t>(index)
            );
        }
        dependentDofs[equality.indices.y] = 1u;
    }
    for (std::size_t index = 0u; index < jointEqualities.size(); ++index) {
        if (jointEqualities[index].indices.w != MR_INVALID_INDEX &&
            dependentDofs[jointEqualities[index].indices.w] != 0u) {
            return failure(
                NumiHumanMuscleEquilibriumStatus::equalityFailure,
                static_cast<std::uint32_t>(index)
            );
        }
    }
    for (std::size_t muscle = 0u; muscle < architectures.size(); ++muscle) {
        const bool legacy = architectures[muscle].optimalFiberLength == 0.0 &&
            architectures[muscle].tendonSlackLength == 0.0;
        const bool compliant = architectures[muscle].optimalFiberLength > 0.0 &&
            architectures[muscle].tendonSlackLength > 0.0;
        if (!legacy && !compliant) {
            return failure(
                NumiHumanMuscleEquilibriumStatus::unsupportedMuscleArchitecture,
                static_cast<std::uint32_t>(muscle)
            );
        }
    }
    std::vector<std::uint8_t> recruited(muscles.size(),
        selectedMuscleIndices.empty() ? 1u : 0u);
    std::uint32_t recruitedCount = selectedMuscleIndices.empty()
        ? static_cast<std::uint32_t>(muscles.size()) : 0u;
    for (const std::uint32_t index : selectedMuscleIndices) {
        if (index >= muscles.size() || recruited[index] != 0u) {
            return failure(
                NumiHumanMuscleEquilibriumStatus::invalidSelection, index
            );
        }
        recruited[index] = 1u;
        ++recruitedCount;
    }

    ArticulatedDynamicsConfig dynamicsConfig;
    dynamicsConfig.gravity = {
        model.world.gravityAndTimestep.x,
        model.world.gravityAndTimestep.y,
        model.world.gravityAndTimestep.z,
    };
    dynamicsConfig.timestep = config.timestep;
    std::vector<double> projectedInitialQ(initialQ.begin(), initialQ.end());
    double maximumInitialEqualityProjection = 0.0;
    const auto projectionDiagnostics = projectNumiHumanJointEqualities(
        jointEqualities, projectedInitialQ, &maximumInitialEqualityProjection
    );
    if (!projectionDiagnostics.succeeded()) {
        auto failed = failure(
            NumiHumanMuscleEquilibriumStatus::equalityFailure,
            projectionDiagnostics.failingIndex
        );
        failed.equalityStatus = projectionDiagnostics.status;
        return failed;
    }
    PoseState current;
    current.q = projectedInitialQ;
    auto diagnostics = solveActivation(
        model, articulationIndex, projectedInitialQ, sites, wraps, muscles,
        architectures, jointEqualities, recruited, config, dynamicsConfig,
        current
    );
    if (!diagnostics.succeeded()) return diagnostics;
    const double initialResidual = current.residualRms;
    std::uint32_t acceptedPoseSteps = 0u;
    for (std::uint32_t sweep = 0u; sweep < config.poseSweeps; ++sweep) {
        const auto candidates = poseCandidates(
            model, articulationIndex, current, config.poseCandidateCount
        );
        PoseState best = current;
        bool found = false;
        for (const std::uint32_t localV : candidates) {
            const MRDofPropertiesGPU& dof =
                model.dofs[articulation.vOffset + localV];
            const std::size_t localQ = dof.qIndex - articulation.qOffset;
            const double range =
                static_cast<double>(dof.limits.y) - dof.limits.x;
            const double margin = config.positionLimitMarginFraction * range;
            const double lower = static_cast<double>(dof.limits.x) + margin;
            const double upper = static_cast<double>(dof.limits.y) - margin;
            if (!(lower < upper)) continue;
            const double step = std::min(
                config.maximumPoseStep, config.poseStepFraction * range
            );
            for (const double direction : {-1.0, 1.0}) {
                PoseState candidate = current;
                candidate.q[localQ] = std::clamp(
                    current.q[localQ] + direction * step, lower, upper
                );
                if (std::abs(candidate.q[localQ] - current.q[localQ]) <
                    kMinimum) continue;
                const auto candidateProjection =
                    projectNumiHumanJointEqualities(
                        jointEqualities, candidate.q
                    );
                if (!candidateProjection.succeeded()) {
                    auto failed = failure(
                        NumiHumanMuscleEquilibriumStatus::equalityFailure,
                        candidateProjection.failingIndex
                    );
                    failed.equalityStatus = candidateProjection.status;
                    return failed;
                }
                diagnostics = evaluatePoseWithActivation(
                    model, articulationIndex, projectedInitialQ, sites, wraps,
                    muscles, architectures, jointEqualities, config,
                    dynamicsConfig, candidate
                );
                if (!diagnostics.succeeded()) return diagnostics;
                if (candidate.objective + config.poseImprovementTolerance <
                    best.objective) {
                    best = std::move(candidate);
                    found = true;
                }
            }
        }
        if (!found) break;
        PoseState recruitedPose;
        recruitedPose.q = best.q;
        diagnostics = solveActivation(
            model, articulationIndex, projectedInitialQ, sites, wraps, muscles,
            architectures, jointEqualities, recruited, config,
            dynamicsConfig, recruitedPose
        );
        if (!diagnostics.succeeded()) return diagnostics;
        if (!(recruitedPose.objective + config.poseImprovementTolerance <
              current.objective)) {
            break;
        }
        current = std::move(recruitedPose);
        ++acceptedPoseSteps;
    }

    NumiHumanMuscleEquilibriumResult candidate;
    candidate.q = current.q;
    candidate.activation = current.activation;
    candidate.fiberLength = current.fiberLength;
    candidate.generalizedMuscleForce = current.muscleForce;
    candidate.generalizedPositionLimitForce = current.limitForce;
    candidate.generalizedJointEqualityForce = current.equalityForce;
    candidate.gravityTarget = current.target;
    candidate.generalizedForceResidual = current.residual;
    candidate.diagnostics.muscleCount =
        static_cast<std::uint32_t>(muscles.size());
    candidate.diagnostics.recruitedMuscleCount = recruitedCount;
    candidate.diagnostics.activationSweeps = current.activationSweeps;
    candidate.diagnostics.acceptedPoseSteps = acceptedPoseSteps;
    candidate.diagnostics.jointEqualityCount =
        static_cast<std::uint32_t>(jointEqualities.size());
    candidate.diagnostics.maximumInitialEqualityProjection =
        maximumInitialEqualityProjection;
    candidate.diagnostics.initialNormalizedResidualRms = initialResidual;
    candidate.diagnostics.normalizedResidualRms = current.residualRms;
    candidate.diagnostics.maximumGeneralizedForceResidual =
        current.maximumResidual;
    for (std::size_t dof = 0u; dof < current.residual.size(); ++dof) {
        const double normalized = std::abs(
            current.weights[dof] * current.residual[dof]
        );
        if (normalized >
            candidate.diagnostics.maximumNormalizedGeneralizedForceResidual) {
            candidate.diagnostics.maximumNormalizedGeneralizedForceResidual =
                normalized;
            candidate.diagnostics.maximumNormalizedResidualDof =
                static_cast<std::uint32_t>(dof);
        }
    }
    candidate.diagnostics.minimumNormalizedPositionLimitMargin =
        minimumNormalizedLimitMargin(
            model, articulationIndex, current.q
        );
    for (const double reaction : current.limitForce) {
        if (reaction != 0.0) {
            ++candidate.diagnostics.activePositionLimitCount;
        }
        candidate.diagnostics.maximumPositionLimitReaction = std::max(
            candidate.diagnostics.maximumPositionLimitReaction,
            std::abs(reaction)
        );
    }
    for (const double reaction : current.equalityForce) {
        candidate.diagnostics.maximumJointEqualityReaction = std::max(
            candidate.diagnostics.maximumJointEqualityReaction,
            std::abs(reaction)
        );
    }
    for (std::size_t index = 0u; index < jointEqualities.size(); ++index) {
        NumiHumanJointEqualityEvaluation evaluation;
        const auto equalityDiagnostics = evaluateNumiHumanJointEquality(
            jointEqualities[index], current.q, evaluation
        );
        if (!equalityDiagnostics.succeeded()) {
            auto failed = failure(
                NumiHumanMuscleEquilibriumStatus::equalityFailure,
                static_cast<std::uint32_t>(index)
            );
            failed.equalityStatus = equalityDiagnostics.status;
            return failed;
        }
        candidate.diagnostics.maximumJointEqualityError = std::max(
            candidate.diagnostics.maximumJointEqualityError,
            std::abs(evaluation.positionError)
        );
    }
    for (const double activation : current.activation) {
        if (activation > 1.0e-5) {
            ++candidate.diagnostics.activeMuscleCount;
        }
        candidate.diagnostics.maximumActivation = std::max(
            candidate.diagnostics.maximumActivation, activation
        );
    }
    candidate.diagnostics.balanced =
        current.residualRms <= config.balanceTolerance;
    result = std::move(candidate);
    return result.diagnostics;
}

const char* numiHumanMuscleEquilibriumStatusName(
    const NumiHumanMuscleEquilibriumStatus status
) noexcept {
    switch (status) {
    case NumiHumanMuscleEquilibriumStatus::success: return "success";
    case NumiHumanMuscleEquilibriumStatus::invalidConfiguration:
        return "invalidConfiguration";
    case NumiHumanMuscleEquilibriumStatus::invalidArticulation:
        return "invalidArticulation";
    case NumiHumanMuscleEquilibriumStatus::invalidDimensions:
        return "invalidDimensions";
    case NumiHumanMuscleEquilibriumStatus::nonfiniteInput:
        return "nonfiniteInput";
    case NumiHumanMuscleEquilibriumStatus::invalidSelection:
        return "invalidSelection";
    case NumiHumanMuscleEquilibriumStatus::unsupportedMuscleArchitecture:
        return "unsupportedMuscleArchitecture";
    case NumiHumanMuscleEquilibriumStatus::equalityFailure:
        return "equalityFailure";
    case NumiHumanMuscleEquilibriumStatus::kinematicsFailure:
        return "kinematicsFailure";
    case NumiHumanMuscleEquilibriumStatus::muscleFailure:
        return "muscleFailure";
    case NumiHumanMuscleEquilibriumStatus::dynamicsFailure:
        return "dynamicsFailure";
    case NumiHumanMuscleEquilibriumStatus::nonfiniteResult:
        return "nonfiniteResult";
    }
    return "unknown";
}

} // namespace metalrobo

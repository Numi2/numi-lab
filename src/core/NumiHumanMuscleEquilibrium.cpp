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
    std::vector<double> muscleTendonForce;
    std::vector<double> passiveMuscleTendonForce;
    std::vector<double> muscleForce;
    std::vector<double> supportNormalForce;
    std::vector<double> supportForce;
    std::vector<double> passiveCoordinateForce;
    std::vector<double> target;
    std::vector<double> residual;
    std::vector<double> limitForce;
    std::vector<double> equalityForce;
    std::vector<double> accelerationResidual;
    std::vector<double> weights;
    double residualRms = 0.0;
    double maximumResidual = 0.0;
    double maximumAccelerationResidual = 0.0;
    double objective = 0.0;
    std::uint32_t activationSweeps = 0u;
    std::uint32_t globalActivationPolishIterations = 0u;
    std::uint32_t acceptedGlobalActivationPolishSteps = 0u;
};

struct ResolvedMuscle {
    double pathLength = 0.0;
    std::vector<double> jacobian;
};

struct AccelerationProjection {
    std::vector<std::uint32_t> independentDofs;
    std::vector<std::vector<std::pair<std::uint32_t, double>>> columns;
    // Lower-triangular Cholesky factor, row-major compact independent space.
    std::vector<double> factor;
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

NumiHumanMuscleEquilibriumDiagnostics resolvePassiveCoordinateForce(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const NumiHumanPassiveCoordinateCoupling> couplings,
    std::vector<double>& force
) {
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    force.assign(articulation.nv, 0.0);
    for (std::size_t index = 0u; index < couplings.size(); ++index) {
        const auto& coupling = couplings[index];
        if (coupling.targetDofIndex >= articulation.nv ||
            coupling.sourceDofIndex >= articulation.nv ||
            !std::isfinite(coupling.sourceRestPosition) ||
            !std::isfinite(coupling.stiffness)) {
            return failure(
                NumiHumanMuscleEquilibriumStatus::invalidDimensions,
                static_cast<std::uint32_t>(index));
        }
        const MRDofPropertiesGPU& source = model.dofs[
            articulation.vOffset + coupling.sourceDofIndex];
        if (source.qIndex == MR_INVALID_INDEX ||
            source.qIndex < articulation.qOffset ||
            source.qIndex >= articulation.qOffset + articulation.nq) {
            return failure(
                NumiHumanMuscleEquilibriumStatus::invalidDimensions,
                static_cast<std::uint32_t>(index));
        }
        const double displacement =
            q[source.qIndex - articulation.qOffset] -
            coupling.sourceRestPosition;
        force[coupling.targetDofIndex] -= coupling.stiffness * displacement;
        if (!std::isfinite(force[coupling.targetDofIndex])) {
            return failure(
                NumiHumanMuscleEquilibriumStatus::nonfiniteResult,
                static_cast<std::uint32_t>(index));
        }
    }
    return {};
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

NumiHumanMuscleEquilibriumDiagnostics resolveStaticSupports(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const NumiHumanStaticSupportContact> supports,
    std::vector<std::vector<double>>& generalizedColumns,
    const ArticulatedDynamicsConfig& dynamicsConfig
) {
    const std::size_t nv = model.articulations[articulationIndex].nv;
    const std::vector<double> zeroVelocity(nv, 0.0);
    std::vector<ArticulatedPointQuery> queries;
    queries.reserve(supports.size());
    for (std::size_t index = 0u; index < supports.size(); ++index) {
        const auto& support = supports[index];
        const double normalLength = std::sqrt(
            square(support.normal[0]) + square(support.normal[1]) +
            square(support.normal[2]));
        if (support.bodyIndex >= model.bodies.size() ||
            model.bodies[support.bodyIndex].articulationIndex !=
                articulationIndex ||
            !finiteSpan(support.localPoint) ||
            !finiteSpan(support.normal) ||
            std::abs(normalLength - 1.0) > 1.0e-6) {
            return failure(
                NumiHumanMuscleEquilibriumStatus::invalidDimensions,
                static_cast<std::uint32_t>(index));
        }
        queries.push_back({support.bodyIndex, support.localPoint});
    }
    if (queries.empty()) {
        generalizedColumns.clear();
        return {};
    }
    std::vector<ArticulatedPointKinematics> points(queries.size());
    std::vector<double> jacobians(queries.size() * 3u * nv, 0.0);
    const auto diagnostics = computeArticulatedPointJacobians(
        model, articulationIndex, q, zeroVelocity, queries, points,
        jacobians, dynamicsConfig);
    if (!diagnostics.succeeded()) {
        auto failed = failure(
            NumiHumanMuscleEquilibriumStatus::kinematicsFailure);
        failed.dynamicsStatus = diagnostics.status;
        return failed;
    }
    generalizedColumns.assign(
        supports.size(), std::vector<double>(nv, 0.0));
    for (std::size_t support = 0u; support < supports.size(); ++support) {
        const std::size_t base = support * 3u * nv;
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            generalizedColumns[support][dof] =
                supports[support].normal[0] * jacobians[base + dof] +
                supports[support].normal[1] * jacobians[base + nv + dof] +
                supports[support].normal[2] *
                    jacobians[base + 2u * nv + dof];
        }
    }
    return {};
}

void solveFloatingRootSupportForces(
    const MRArticulationGPU& articulation,
    const std::span<const double> target,
    const std::span<const std::vector<double>> generalizedColumns,
    const NumiHumanMuscleEquilibriumConfig& config,
    std::vector<double>& normalForce,
    std::vector<double>& generalizedForce
) {
    const std::size_t rootDofCount =
        articulation.rootType == MR_ROOT_FLOATING
        ? std::min<std::size_t>(6u, articulation.nv) : 0u;
    generalizedForce.assign(articulation.nv, 0.0);
    if (normalForce.size() != generalizedColumns.size()) {
        normalForce.assign(generalizedColumns.size(), 0.0);
    }
    if (rootDofCount == 0u || generalizedColumns.empty()) return;

    // Static support is a hard mechanical layer above internal recruitment:
    // solve the floating-base wrench first. Muscle paths are internal force
    // pairs and therefore cannot supply a net world wrench. Mixing these six
    // equations into the soft internal acceleration objective can otherwise
    // trade body weight away to reduce many smaller joint residuals.
    std::array<double, 6u> residual{};
    std::array<double, 6u> weight{};
    for (std::size_t dof = 0u; dof < rootDofCount; ++dof) {
        residual[dof] = -target[dof];
        weight[dof] = 1.0 / std::max(1.0, std::abs(target[dof]));
    }
    for (std::size_t support = 0u;
         support < generalizedColumns.size(); ++support) {
        normalForce[support] = std::clamp(
            normalForce[support], 0.0, config.maximumSupportForceNewtons);
        for (std::size_t dof = 0u; dof < rootDofCount; ++dof) {
            residual[dof] += normalForce[support] *
                generalizedColumns[support][dof];
        }
    }
    for (std::uint32_t sweep = 0u;
         sweep < config.supportForceSweeps; ++sweep) {
        double maximumNormalizedChange = 0.0;
        for (std::size_t support = 0u;
             support < generalizedColumns.size(); ++support) {
            const double currentForce = normalForce[support];
            double gradient =
                config.supportForceRegularization * currentForce;
            double curvature = config.supportForceRegularization;
            for (std::size_t dof = 0u; dof < rootDofCount; ++dof) {
                const double direction = weight[dof] *
                    generalizedColumns[support][dof];
                gradient += direction * weight[dof] * residual[dof];
                curvature += direction * direction;
            }
            const double nextForce = std::clamp(
                currentForce - gradient / std::max(kMinimum, curvature),
                0.0, config.maximumSupportForceNewtons);
            const double delta = nextForce - currentForce;
            for (std::size_t dof = 0u; dof < rootDofCount; ++dof) {
                residual[dof] +=
                    generalizedColumns[support][dof] * delta;
            }
            maximumNormalizedChange = std::max(
                maximumNormalizedChange,
                std::abs(delta) / config.maximumSupportForceNewtons);
            normalForce[support] = nextForce;
        }
        if (maximumNormalizedChange < config.supportForceConvergence) break;
    }
    for (std::size_t support = 0u;
         support < generalizedColumns.size(); ++support) {
        for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
            generalizedForce[dof] += normalForce[support] *
                generalizedColumns[support][dof];
        }
    }
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

NumiHumanMuscleEquilibriumDiagnostics buildAccelerationProjection(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const MRNumiHumanJointEqualityGPU> equalities,
    const ArticulatedDynamicsConfig& dynamicsConfig,
    const bool includeFloatingRoot,
    AccelerationProjection& projection
) {
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    const std::size_t nv = articulation.nv;
    const std::size_t firstInternal =
        articulation.rootType == MR_ROOT_FLOATING && !includeFloatingRoot
        ? 6u : 0u;
    std::vector<std::uint8_t> dependent(nv, 0u);
    std::vector<double> derivative(equalities.size(), 0.0);
    for (std::size_t index = 0u; index < equalities.size(); ++index) {
        NumiHumanJointEqualityEvaluation evaluation;
        const auto diagnostics = evaluateNumiHumanJointEquality(
            equalities[index], q, evaluation
        );
        if (!diagnostics.succeeded() || equalities[index].indices.y >= nv ||
            (equalities[index].indices.w != MR_INVALID_INDEX &&
             equalities[index].indices.w >= nv)) {
            auto failed = failure(
                NumiHumanMuscleEquilibriumStatus::equalityFailure,
                static_cast<std::uint32_t>(index)
            );
            failed.equalityStatus = diagnostics.status;
            return failed;
        }
        dependent[equalities[index].indices.y] = 1u;
        derivative[index] = evaluation.derivative;
    }

    AccelerationProjection candidate;
    std::vector<std::uint32_t> compactIndex(nv, MR_INVALID_INDEX);
    for (std::size_t dof = firstInternal; dof < nv; ++dof) {
        if (dependent[dof] != 0u) continue;
        compactIndex[dof] = static_cast<std::uint32_t>(
            candidate.independentDofs.size()
        );
        candidate.independentDofs.push_back(
            static_cast<std::uint32_t>(dof)
        );
        candidate.columns.push_back({{
            static_cast<std::uint32_t>(dof), 1.0,
        }});
    }
    if (candidate.independentDofs.empty()) {
        return failure(NumiHumanMuscleEquilibriumStatus::invalidDimensions);
    }
    for (std::size_t index = 0u; index < equalities.size(); ++index) {
        const auto& equality = equalities[index];
        if (equality.indices.w == MR_INVALID_INDEX ||
            compactIndex[equality.indices.w] == MR_INVALID_INDEX) {
            continue;
        }
        candidate.columns[compactIndex[equality.indices.w]].push_back({
            equality.indices.y, derivative[index],
        });
    }

    std::vector<double> mass(nv * nv, 0.0);
    const auto massDiagnostics = computeArticulatedMassMatrix(
        model, articulationIndex, q, mass, dynamicsConfig
    );
    if (!massDiagnostics.succeeded()) {
        auto failed = failure(
            NumiHumanMuscleEquilibriumStatus::dynamicsFailure
        );
        failed.dynamicsStatus = massDiagnostics.status;
        return failed;
    }
    const std::size_t count = candidate.independentDofs.size();
    candidate.factor.assign(count * count, 0.0);
    for (std::size_t row = 0u; row < count; ++row) {
        for (std::size_t column = 0u; column < count; ++column) {
            double value = 0.0;
            for (const auto [fullRow, rowScale] : candidate.columns[row]) {
                for (const auto [fullColumn, columnScale] :
                     candidate.columns[column]) {
                    value += rowScale * mass[fullRow * nv + fullColumn] *
                        columnScale;
                }
            }
            candidate.factor[row * count + column] = value;
        }
    }
    for (std::size_t row = 0u; row < count; ++row) {
        double scale = 0.0;
        for (std::size_t column = 0u; column < count; ++column) {
            scale = std::max(
                scale, std::abs(candidate.factor[row * count + column])
            );
        }
        for (std::size_t column = 0u; column <= row; ++column) {
            double value = candidate.factor[row * count + column];
            for (std::size_t inner = 0u; inner < column; ++inner) {
                value -= candidate.factor[row * count + inner] *
                    candidate.factor[column * count + inner];
            }
            if (row == column) {
                const double floor = std::max(
                    kMinimum, scale * 32.0 *
                        std::numeric_limits<double>::epsilon()
                );
                if (!(value > floor) || !std::isfinite(value)) {
                    auto failed = failure(
                        NumiHumanMuscleEquilibriumStatus::dynamicsFailure,
                        static_cast<std::uint32_t>(row)
                    );
                    failed.dynamicsStatus =
                        ArticulatedDynamicsStatus::massMatrixNotPositiveDefinite;
                    return failed;
                }
                candidate.factor[row * count + row] = std::sqrt(value);
            } else {
                candidate.factor[row * count + column] =
                    value / candidate.factor[column * count + column];
            }
        }
    }
    projection = std::move(candidate);
    return {};
}

bool projectForceToAcceleration(
    const AccelerationProjection& projection,
    const std::span<const double> force,
    std::vector<double>& acceleration
) {
    const std::size_t count = projection.independentDofs.size();
    if (projection.factor.size() != count * count) return false;
    std::vector<double> compact(count, 0.0);
    std::vector<double> workspace(count, 0.0);
    for (std::size_t row = 0u; row < count; ++row) {
        const std::uint32_t dof = projection.independentDofs[row];
        if (dof >= force.size() || !std::isfinite(force[dof])) return false;
        double value = force[dof];
        for (std::size_t column = 0u; column < row; ++column) {
            value -= projection.factor[row * count + column] *
                workspace[column];
        }
        workspace[row] = value / projection.factor[row * count + row];
    }
    for (std::size_t reverse = 0u; reverse < count; ++reverse) {
        const std::size_t row = count - 1u - reverse;
        double value = workspace[row];
        for (std::size_t column = row + 1u; column < count; ++column) {
            value -= projection.factor[column * count + row] * compact[column];
        }
        compact[row] = value / projection.factor[row * count + row];
        if (!std::isfinite(compact[row])) return false;
    }
    acceleration.assign(force.size(), 0.0);
    for (std::size_t row = 0u; row < count; ++row) {
        acceleration[projection.independentDofs[row]] = compact[row];
    }
    return true;
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
    const AccelerationProjection& projection,
    const bool includeFloatingRoot,
    PoseState& state
) {
    const std::size_t firstInternal =
        articulation.rootType == MR_ROOT_FLOATING && !includeFloatingRoot
        ? 6u : 0u;
    double sum = 0.0;
    state.maximumResidual = 0.0;
    state.maximumAccelerationResidual = 0.0;
    state.limitForce.assign(articulation.nv, 0.0);
    state.equalityForce.assign(articulation.nv, 0.0);
    for (std::size_t dof = firstInternal; dof < articulation.nv; ++dof) {
        state.residual[dof] = state.muscleForce[dof] +
            state.supportForce[dof] + state.passiveCoordinateForce[dof] -
            state.target[dof];
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
        state.maximumResidual = std::max(
            state.maximumResidual, std::abs(state.residual[dof])
        );
    }
    if (!projectForceToAcceleration(
            projection, state.residual, state.accelerationResidual
        )) {
        state.residualRms = std::numeric_limits<double>::infinity();
        state.objective = std::numeric_limits<double>::infinity();
        return;
    }
    std::size_t rowCount = 0u;
    for (const std::uint32_t dof : projection.independentDofs) {
        const double normalized =
            state.weights[dof] * state.accelerationResidual[dof];
        sum += normalized * normalized;
        ++rowCount;
        state.maximumAccelerationResidual = std::max(
            state.maximumAccelerationResidual,
            std::abs(state.accelerationResidual[dof])
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
        config.supportForceRegularization * std::inner_product(
            state.supportNormalForce.begin(),
            state.supportNormalForce.end(),
            state.supportNormalForce.begin(), 0.0) +
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
    const std::span<const NumiHumanStaticSupportContact> supports,
    const std::span<const NumiHumanPassiveCoordinateCoupling> passiveCouplings,
    const std::span<const std::uint8_t> recruited,
    const bool enableGlobalPolish,
    const bool initializeFromAcceptedState,
    const NumiHumanMuscleEquilibriumConfig& config,
    const ArticulatedDynamicsConfig& dynamicsConfig,
    PoseState& state
) {
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    const std::size_t nv = articulation.nv;
    const std::uint32_t acceptedActivationSweeps = state.activationSweeps;
    if (initializeFromAcceptedState &&
        state.activation.size() != muscles.size()) {
        return failure(NumiHumanMuscleEquilibriumStatus::invalidDimensions);
    }
    std::vector<ResolvedMuscle> resolved;
    auto diagnostics = resolveMuscles(
        model, articulationIndex, state.q, sites, wraps, muscles, resolved,
        dynamicsConfig
    );
    if (!diagnostics.succeeded()) return diagnostics;
    std::vector<std::vector<double>> supportJacobians;
    diagnostics = resolveStaticSupports(
        model, articulationIndex, state.q, supports, supportJacobians,
        dynamicsConfig);
    if (!diagnostics.succeeded()) return diagnostics;
    diagnostics = gravityTarget(
        model, articulationIndex, state.q, state.target, dynamicsConfig
    );
    if (!diagnostics.succeeded()) return diagnostics;
    diagnostics = resolvePassiveCoordinateForce(
        model, articulationIndex, state.q, passiveCouplings,
        state.passiveCoordinateForce);
    if (!diagnostics.succeeded()) return diagnostics;

    std::vector<std::vector<double>> objectiveJacobians;
    objectiveJacobians.reserve(resolved.size() + supportJacobians.size());
    for (const auto& muscle : resolved) {
        objectiveJacobians.push_back(muscle.jacobian);
    }
    for (const auto& support : supportJacobians) {
        objectiveJacobians.push_back(support);
    }
    std::vector<double> objectiveTarget = state.target;
    for (std::size_t dof = 0u; dof < nv; ++dof) {
        objectiveTarget[dof] -= state.passiveCoordinateForce[dof];
    }
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

    AccelerationProjection projection;
    diagnostics = buildAccelerationProjection(
        model, articulationIndex, state.q, equalities, dynamicsConfig,
        !supports.empty(),
        projection
    );
    if (!diagnostics.succeeded()) return diagnostics;
    std::vector<double> objectiveTargetAcceleration;
    if (!projectForceToAcceleration(
            projection, objectiveTarget, objectiveTargetAcceleration
        )) {
        return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
    }
    for (auto& jacobian : objectiveJacobians) {
        std::vector<double> acceleration;
        if (!projectForceToAcceleration(
                projection, jacobian, acceleration
            )) {
            return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
        }
        jacobian = std::move(acceleration);
    }

    state.weights.assign(nv, 0.0);
    for (const std::uint32_t dof : projection.independentDofs) {
        state.weights[dof] = 1.0 / std::max(
            config.minimumGeneralizedAccelerationScale,
            std::abs(objectiveTargetAcceleration[dof])
        );
    }
    const std::uint32_t sampleCount = config.activationSamples;
    std::vector<double> forceSamples(muscles.size() * sampleCount, 0.0);
    if (!initializeFromAcceptedState) {
        state.activation.assign(muscles.size(), 0.0);
    } else {
        for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
            state.activation[muscle] = recruited[muscle] == 0u
                ? 0.0
                : std::clamp(
                    state.activation[muscle], 0.0, config.activationLimit
                );
        }
    }
    state.fiberLength.assign(muscles.size(), 0.0);
    state.muscleTendonForce.assign(muscles.size(), 0.0);
    state.passiveMuscleTendonForce.assign(muscles.size(), 0.0);
    state.muscleForce.assign(nv, 0.0);
    solveFloatingRootSupportForces(
        articulation, objectiveTarget, supportJacobians, config,
        state.supportNormalForce, state.supportForce);
    std::vector<double> objectiveSupportAcceleration;
    if (!projectForceToAcceleration(
            projection, state.supportForce, objectiveSupportAcceleration)) {
        return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
    }
    std::vector<double> objectiveMuscleAcceleration(nv, 0.0);
    std::vector<double> optimizerForce(muscles.size(), 0.0);
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
        state.passiveMuscleTendonForce[muscle] = passive;
        double initialForce = passive;
        if (initializeFromAcceptedState) {
            diagnostics = evaluateStaticForce(
                resolved[muscle].pathLength, state.activation[muscle],
                config.timestep, muscles[muscle], architectures[muscle],
                initialForce, state.fiberLength[muscle],
                static_cast<std::uint32_t>(muscle)
            );
            if (!diagnostics.succeeded()) return diagnostics;
        }
        optimizerForce[muscle] = initialForce;
        state.muscleTendonForce[muscle] = initialForce;
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            state.muscleForce[dof] +=
                initialForce * resolved[muscle].jacobian[dof];
            objectiveMuscleAcceleration[dof] +=
                initialForce * objectiveJacobians[muscle][dof];
        }
    }
    state.residual.assign(nv, 0.0);
    for (const std::uint32_t dof : projection.independentDofs) {
        state.residual[dof] = state.weights[dof] *
            (objectiveMuscleAcceleration[dof] +
             objectiveSupportAcceleration[dof] -
             objectiveTargetAcceleration[dof]);
    }

    std::uint32_t completedSweeps = 0u;
    constexpr double kRootConstraintWeight = 1000.0;
    std::array<double, 6u> rootForceResidual{};
    const std::size_t rootDofCount = !supports.empty() &&
        articulation.rootType == MR_ROOT_FLOATING
        ? std::min<std::size_t>(6u, nv) : 0u;
    for (std::size_t dof = 0u; dof < rootDofCount; ++dof) {
        rootForceResidual[dof] = state.muscleForce[dof] +
            state.supportForce[dof] + state.passiveCoordinateForce[dof] -
            state.target[dof];
    }
    double bestExactObjective = std::numeric_limits<double>::infinity();
    std::vector<double> bestExactActivation;
    std::vector<double> bestExactSupportNormalForce;
    const auto checkpointExactState = [&]()
        -> NumiHumanMuscleEquilibriumDiagnostics {
        PoseState candidate;
        candidate.q = state.q;
        candidate.activation = state.activation;
        candidate.target = state.target;
        candidate.passiveCoordinateForce = state.passiveCoordinateForce;
        candidate.weights = state.weights;
        candidate.supportNormalForce = state.supportNormalForce;
        solveFloatingRootSupportForces(
            articulation, objectiveTarget, supportJacobians, config,
            candidate.supportNormalForce, candidate.supportForce);
        candidate.fiberLength.assign(muscles.size(), 0.0);
        candidate.muscleForce.assign(nv, 0.0);
        std::vector<double> exactForce(muscles.size(), 0.0);
        for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
            auto exactDiagnostics = evaluateStaticForce(
                resolved[muscle].pathLength, candidate.activation[muscle],
                config.timestep, muscles[muscle], architectures[muscle],
                exactForce[muscle], candidate.fiberLength[muscle],
                static_cast<std::uint32_t>(muscle));
            if (!exactDiagnostics.succeeded()) return exactDiagnostics;
            for (std::size_t dof = 0u; dof < nv; ++dof) {
                candidate.muscleForce[dof] +=
                    exactForce[muscle] * resolved[muscle].jacobian[dof];
            }
        }
        candidate.residual.assign(nv, 0.0);
        finishResidual(
            articulation, config, initialQ, model, articulationIndex,
            equalities, projection, !supports.empty(), candidate);
        if (!std::isfinite(candidate.objective)) {
            return failure(
                NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
        }
        if (candidate.objective < bestExactObjective) {
            bestExactObjective = candidate.objective;
            bestExactActivation = candidate.activation;
            bestExactSupportNormalForce = candidate.supportNormalForce;
        }
        // Re-anchor the coordinate optimizer to the exact nonlinear force
        // law. Otherwise piecewise interpolation error accumulates across
        // sweeps and a nominal descent direction can leave the exact state.
        optimizerForce = std::move(exactForce);
        state.muscleForce = candidate.muscleForce;
        state.supportNormalForce = candidate.supportNormalForce;
        state.supportForce = candidate.supportForce;
        std::fill(state.residual.begin(), state.residual.end(), 0.0);
        for (const std::uint32_t dof : projection.independentDofs) {
            double acceleration = -objectiveTargetAcceleration[dof];
            for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
                acceleration += optimizerForce[muscle] *
                    objectiveJacobians[muscle][dof];
            }
            for (std::size_t support = 0u; support < supports.size(); ++support) {
                acceleration += state.supportNormalForce[support] *
                    objectiveJacobians[muscles.size() + support][dof];
            }
            state.residual[dof] = state.weights[dof] * acceleration;
        }
        for (std::size_t dof = 0u; dof < rootDofCount; ++dof) {
            rootForceResidual[dof] = state.muscleForce[dof] +
                state.supportForce[dof] + state.passiveCoordinateForce[dof] -
                state.target[dof];
        }
        return {};
    };
    diagnostics = checkpointExactState();
    if (!diagnostics.succeeded()) return diagnostics;
    for (std::uint32_t sweep = 0u;
         !initializeFromAcceptedState && sweep < config.activationSweeps;
         ++sweep) {
        double maximumChange = 0.0;
        for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
            if (recruited[muscle] == 0u) continue;
            const double currentActivation = state.activation[muscle];
            const double currentForce = optimizerForce[muscle];
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
                for (const std::uint32_t dof : projection.independentDofs) {
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
                for (const std::uint32_t dof : projection.independentDofs) {
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
            for (const std::uint32_t dof : projection.independentDofs) {
                state.residual[dof] += state.weights[dof] *
                    objectiveJacobians[muscle][dof] *
                    (nextForce - currentForce);
            }
            maximumChange = std::max(
                maximumChange,
                std::abs(bestActivation - currentActivation)
            );
            state.activation[muscle] = bestActivation;
            optimizerForce[muscle] = nextForce;
        }
        for (std::size_t support = 0u;
             support < supports.size(); ++support) {
            const std::size_t columnIndex = muscles.size() + support;
            const double currentForce = state.supportNormalForce[support];
            double gradient =
                config.supportForceRegularization * currentForce;
            double curvature = config.supportForceRegularization;
            for (const std::uint32_t dof : projection.independentDofs) {
                const double direction = state.weights[dof] *
                    objectiveJacobians[columnIndex][dof];
                gradient += direction * state.residual[dof];
                curvature += direction * direction;
            }
            for (std::size_t dof = 0u; dof < rootDofCount; ++dof) {
                const double scale = std::max(
                    1.0, std::abs(state.target[dof]));
                const double direction = kRootConstraintWeight *
                    supportJacobians[support][dof] / scale;
                const double residual = kRootConstraintWeight *
                    rootForceResidual[dof] / scale;
                gradient += direction * residual;
                curvature += direction * direction;
            }
            const double nextForce = std::clamp(
                currentForce - gradient / std::max(kMinimum, curvature),
                0.0, config.maximumSupportForceNewtons);
            const double delta = nextForce - currentForce;
            for (const std::uint32_t dof : projection.independentDofs) {
                state.residual[dof] += state.weights[dof] *
                    objectiveJacobians[columnIndex][dof] * delta;
            }
            for (std::size_t dof = 0u; dof < rootDofCount; ++dof) {
                rootForceResidual[dof] +=
                    supportJacobians[support][dof] * delta;
            }
            maximumChange = std::max(
                maximumChange,
                std::abs(delta) / config.maximumSupportForceNewtons);
            state.supportNormalForce[support] = nextForce;
        }
        completedSweeps = sweep + 1u;
        if (completedSweeps % config.activationExactCheckpointInterval == 0u ||
            completedSweeps == config.activationSweeps ||
            maximumChange < config.activationConvergence) {
            diagnostics = checkpointExactState();
            if (!diagnostics.succeeded()) return diagnostics;
        }
        if (maximumChange < config.activationConvergence) break;
    }
    state.activationSweeps = initializeFromAcceptedState
        ? acceptedActivationSweeps : completedSweeps;
    if (bestExactActivation.empty()) {
        return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
    }
    state.activation = std::move(bestExactActivation);
    state.supportNormalForce = std::move(bestExactSupportNormalForce);

    // Remove the tiny finite-penalty drift while preserving the optimized
    // contact-force nullspace as much as coordinate projection permits.
    solveFloatingRootSupportForces(
        articulation, objectiveTarget, supportJacobians, config,
        state.supportNormalForce, state.supportForce);

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
        state.muscleTendonForce[muscle] = force;
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            state.muscleForce[dof] +=
                force * resolved[muscle].jacobian[dof];
        }
    }
    std::fill(state.supportForce.begin(), state.supportForce.end(), 0.0);
    for (std::size_t support = 0u;
         support < supportJacobians.size(); ++support) {
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            state.supportForce[dof] += state.supportNormalForce[support] *
                supportJacobians[support][dof];
        }
    }
    state.residual.assign(nv, 0.0);
    finishResidual(
        articulation, config, initialQ, model, articulationIndex, equalities,
        projection, !supports.empty(), state
    );
    if (!std::isfinite(state.residualRms) ||
        !std::isfinite(state.maximumResidual) ||
        !std::isfinite(state.objective)) {
        return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
    }

    // The source-ordered coordinate pass is a robust initializer, but a
    // finite sweep budget can leave order-dependent force sharing. Polish all
    // recruited activations simultaneously in the exact reported objective.
    // The diagonal Gauss-Newton proposal uses the fixed-pose acceleration
    // columns; exact compliant-force evaluation and backtracking decide every
    // accepted step, so this cannot promote an interpolated-force regression.
    const std::size_t objectiveRowCount = projection.independentDofs.size();
    const double activationPenaltyScale = config.activationRegularization /
        static_cast<double>(muscles.size());
    const double activationInterval = config.activationLimit /
        static_cast<double>(sampleCount - 1u);
    const std::uint32_t globalPolishIterations = enableGlobalPolish
        ? config.globalActivationPolishIterations : 0u;
    for (std::uint32_t iteration = 0u;
         iteration < globalPolishIterations; ++iteration) {
        state.globalActivationPolishIterations = iteration + 1u;
        std::vector<double> proposal = state.activation;
        double maximumProposalChange = 0.0;
        for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
            if (recruited[muscle] == 0u) continue;
            const double activation = std::clamp(
                state.activation[muscle], 0.0, config.activationLimit);
            const double sampleCoordinate = activation /
                config.activationLimit * static_cast<double>(sampleCount - 1u);
            const std::uint32_t lowerSample = std::min<std::uint32_t>(
                static_cast<std::uint32_t>(sampleCoordinate),
                sampleCount - 2u);
            const std::size_t sampleBase =
                muscle * sampleCount + lowerSample;
            const double forceSlope =
                (forceSamples[sampleBase + 1u] -
                 forceSamples[sampleBase]) / activationInterval;
            double gradient = activationPenaltyScale * activation;
            double curvature = activationPenaltyScale;
            if (objectiveRowCount != 0u) {
                for (const std::uint32_t dof : projection.independentDofs) {
                    const double direction = state.weights[dof] *
                        objectiveJacobians[muscle][dof] * forceSlope;
                    const double normalizedResidual = state.weights[dof] *
                        state.accelerationResidual[dof];
                    gradient += direction * normalizedResidual /
                        static_cast<double>(objectiveRowCount);
                    curvature += direction * direction /
                        static_cast<double>(objectiveRowCount);
                }
            }
            const double next = std::clamp(
                activation - gradient / std::max(kMinimum, curvature),
                0.0, config.activationLimit);
            proposal[muscle] = next;
            maximumProposalChange = std::max(
                maximumProposalChange, std::abs(next - activation));
        }
        if (maximumProposalChange < config.globalActivationConvergence) break;

        bool accepted = false;
        double lineScale = 1.0;
        for (std::uint32_t line = 0u;
             line < config.globalActivationLineSearchSteps; ++line) {
            PoseState candidate;
            candidate.q = state.q;
            candidate.activation.resize(muscles.size(), 0.0);
            for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
                candidate.activation[muscle] = std::clamp(
                    state.activation[muscle] + lineScale *
                        (proposal[muscle] - state.activation[muscle]),
                    0.0, config.activationLimit);
            }
            candidate.target = state.target;
            candidate.passiveCoordinateForce = state.passiveCoordinateForce;
            candidate.passiveMuscleTendonForce =
                state.passiveMuscleTendonForce;
            candidate.weights = state.weights;
            candidate.supportNormalForce = state.supportNormalForce;
            solveFloatingRootSupportForces(
                articulation, objectiveTarget, supportJacobians, config,
                candidate.supportNormalForce, candidate.supportForce);
            candidate.fiberLength.assign(muscles.size(), 0.0);
            candidate.muscleTendonForce.assign(muscles.size(), 0.0);
            candidate.muscleForce.assign(nv, 0.0);
            for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
                double force = 0.0;
                diagnostics = evaluateStaticForce(
                    resolved[muscle].pathLength,
                    candidate.activation[muscle], config.timestep,
                    muscles[muscle], architectures[muscle], force,
                    candidate.fiberLength[muscle],
                    static_cast<std::uint32_t>(muscle));
                if (!diagnostics.succeeded()) return diagnostics;
                candidate.muscleTendonForce[muscle] = force;
                for (std::size_t dof = 0u; dof < nv; ++dof) {
                    candidate.muscleForce[dof] +=
                        force * resolved[muscle].jacobian[dof];
                }
            }
            candidate.residual.assign(nv, 0.0);
            finishResidual(
                articulation, config, initialQ, model, articulationIndex,
                equalities, projection, !supports.empty(), candidate);
            if (!std::isfinite(candidate.objective)) {
                return failure(
                    NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
            }
            const double improvementFloor = 1.0e-14 *
                std::max(1.0, std::abs(state.objective));
            if (candidate.objective + improvementFloor < state.objective) {
                candidate.activationSweeps = state.activationSweeps;
                candidate.globalActivationPolishIterations = iteration + 1u;
                candidate.acceptedGlobalActivationPolishSteps =
                    state.acceptedGlobalActivationPolishSteps + 1u;
                state = std::move(candidate);
                accepted = true;
                break;
            }
            lineScale *= 0.5;
        }
        if (!accepted) break;
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
    const std::span<const NumiHumanStaticSupportContact> supports,
    const std::span<const NumiHumanPassiveCoordinateCoupling> passiveCouplings,
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
    std::vector<std::vector<double>> supportJacobians;
    diagnostics = resolveStaticSupports(
        model, articulationIndex, state.q, supports, supportJacobians,
        dynamicsConfig);
    if (!diagnostics.succeeded()) return diagnostics;
    diagnostics = gravityTarget(
        model, articulationIndex, state.q, state.target, dynamicsConfig
    );
    if (!diagnostics.succeeded()) return diagnostics;
    diagnostics = resolvePassiveCoordinateForce(
        model, articulationIndex, state.q, passiveCouplings,
        state.passiveCoordinateForce);
    if (!diagnostics.succeeded()) return diagnostics;
    std::vector<double> reducedTarget = state.target;
    for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
        reducedTarget[dof] -= state.passiveCoordinateForce[dof];
    }
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
        if (equalities[index].indices.w != MR_INVALID_INDEX) {
            reducedTarget[equalities[index].indices.w] +=
                evaluation.derivative * reducedTarget[dependent];
        }
        reducedTarget[dependent] = 0.0;
    }
    AccelerationProjection projection;
    diagnostics = buildAccelerationProjection(
        model, articulationIndex, state.q, equalities, dynamicsConfig,
        !supports.empty(),
        projection
    );
    if (!diagnostics.succeeded()) return diagnostics;
    std::vector<double> targetAcceleration;
    if (!projectForceToAcceleration(
            projection, reducedTarget, targetAcceleration
        )) {
        return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
    }
    state.weights.assign(articulation.nv, 0.0);
    for (const std::uint32_t dof : projection.independentDofs) {
        state.weights[dof] = 1.0 / std::max(
            config.minimumGeneralizedAccelerationScale,
            std::abs(targetAcceleration[dof])
        );
    }
    state.muscleForce.assign(articulation.nv, 0.0);
    solveFloatingRootSupportForces(
        articulation, reducedTarget, supportJacobians, config,
        state.supportNormalForce, state.supportForce);
    state.fiberLength.assign(muscles.size(), 0.0);
    state.muscleTendonForce.assign(muscles.size(), 0.0);
    state.passiveMuscleTendonForce.assign(muscles.size(), 0.0);
    for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
        double force = 0.0;
        diagnostics = evaluateStaticForce(
            resolved[muscle].pathLength, state.activation[muscle],
            config.timestep, muscles[muscle], architectures[muscle], force,
            state.fiberLength[muscle], static_cast<std::uint32_t>(muscle)
        );
        if (!diagnostics.succeeded()) return diagnostics;
        state.muscleTendonForce[muscle] = force;
        double passiveForce = 0.0;
        double passiveFiberLength = 0.0;
        diagnostics = evaluateStaticForce(
            resolved[muscle].pathLength, 0.0, config.timestep,
            muscles[muscle], architectures[muscle], passiveForce,
            passiveFiberLength, static_cast<std::uint32_t>(muscle));
        if (!diagnostics.succeeded()) return diagnostics;
        state.passiveMuscleTendonForce[muscle] = passiveForce;
        for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
            state.muscleForce[dof] +=
                force * resolved[muscle].jacobian[dof];
        }
    }
    state.residual.assign(articulation.nv, 0.0);
    finishResidual(
        articulation, config, initialQ, model, articulationIndex, equalities,
        projection, !supports.empty(), state
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
            std::abs(
                state.weights[localV] * state.accelerationResidual[localV]
            ),
            localV
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
    const std::span<const NumiHumanStaticSupportContact> supportContacts,
    const std::span<const NumiHumanPassiveCoordinateCoupling> passiveCouplings,
    NumiHumanMuscleEquilibriumResult& result,
    const NumiHumanMuscleEquilibriumConfig& config
) {
    const bool validConfig = std::isfinite(config.timestep) &&
        config.timestep > 0.0 && std::isfinite(config.activationLimit) &&
        config.activationLimit > 0.0 && config.activationLimit <= 1.0 &&
        config.activationSamples >= 2u && config.activationSamples <= 65u &&
        config.activationSweeps > 0u &&
        config.activationExactCheckpointInterval > 0u &&
        std::isfinite(config.activationRegularization) &&
        config.activationRegularization >= 0.0 &&
        std::isfinite(config.activationConvergence) &&
        config.activationConvergence > 0.0 &&
        config.globalActivationLineSearchSteps > 0u &&
        config.globalActivationLineSearchSteps <= 64u &&
        std::isfinite(config.globalActivationConvergence) &&
        config.globalActivationConvergence > 0.0 &&
        std::isfinite(config.minimumGeneralizedAccelerationScale) &&
        config.minimumGeneralizedAccelerationScale > 0.0 &&
        std::isfinite(config.balanceTolerance) &&
        config.balanceTolerance > 0.0 && config.poseCandidateCount <= 128u &&
        config.poseRecruitmentCandidateCount > 0u &&
        config.poseRecruitmentCandidateCount <= 256u &&
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
        config.positionLimitTolerance >= 0.0 &&
        std::isfinite(config.maximumSupportForceNewtons) &&
        config.maximumSupportForceNewtons > 0.0 &&
        std::isfinite(config.supportForceRegularization) &&
        config.supportForceRegularization >= 0.0 &&
        config.supportForceSweeps > 0u &&
        std::isfinite(config.supportForceConvergence) &&
        config.supportForceConvergence > 0.0;
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
    if (!supportContacts.empty() &&
        articulation.rootType != MR_ROOT_FLOATING) {
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
        architectures, jointEqualities, supportContacts, passiveCouplings,
        recruited, config.poseSweeps == 0u, false, config, dynamicsConfig,
        current
    );
    if (!diagnostics.succeeded()) return diagnostics;
    const double initialResidual = current.residualRms;
    std::uint32_t acceptedPoseSteps = 0u;
    for (std::uint32_t sweep = 0u; sweep < config.poseSweeps; ++sweep) {
        const auto candidates = poseCandidates(
            model, articulationIndex, current, config.poseCandidateCount
        );
        std::vector<PoseState> poseTrials;
        constexpr std::array<double, 4u> kPoseStepScales{
            1.0, 0.5, 0.25, 0.125,
        };
        poseTrials.reserve(candidates.size() * 2u * kPoseStepScales.size());
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
                for (const double scale : kPoseStepScales) {
                    PoseState candidate = current;
                    candidate.q[localQ] = std::clamp(
                        current.q[localQ] + direction * scale * step,
                        lower, upper);
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
                        model, articulationIndex, projectedInitialQ, sites,
                        wraps, muscles, architectures, jointEqualities,
                        supportContacts, passiveCouplings, config,
                        dynamicsConfig, candidate
                    );
                    if (!diagnostics.succeeded()) return diagnostics;
                    poseTrials.push_back(std::move(candidate));
                }
            }
        }
        std::stable_sort(
            poseTrials.begin(), poseTrials.end(),
            [](const PoseState& left, const PoseState& right) {
                return left.objective < right.objective;
            });
        PoseState best = current;
        bool found = false;
        const std::size_t recruitmentCandidateCount = std::min<std::size_t>(
            config.poseRecruitmentCandidateCount, poseTrials.size());
        for (std::size_t index = 0u;
             index < recruitmentCandidateCount; ++index) {
            PoseState recruitedPose;
            recruitedPose.q = poseTrials[index].q;
            diagnostics = solveActivation(
                model, articulationIndex, projectedInitialQ, sites, wraps,
                muscles, architectures, jointEqualities, supportContacts,
                passiveCouplings, recruited, false, false, config,
                dynamicsConfig, recruitedPose
            );
            if (!diagnostics.succeeded()) return diagnostics;
            if (recruitedPose.objective + config.poseImprovementTolerance <
                best.objective) {
                best = std::move(recruitedPose);
                found = true;
            }
        }
        if (!found) break;
        current = std::move(best);
        ++acceptedPoseSteps;
    }

    // Pose trials use the robust coordinate initializer only. Apply the
    // simultaneous exact-objective polish once to the accepted posture so its
    // cost does not multiply across discarded candidates.
    if (config.poseSweeps > 0u &&
        config.globalActivationPolishIterations > 0u) {
        PoseState polished = current;
        diagnostics = solveActivation(
            model, articulationIndex, projectedInitialQ, sites, wraps,
            muscles, architectures, jointEqualities, supportContacts,
            passiveCouplings, recruited, true, true, config, dynamicsConfig,
            polished);
        if (!diagnostics.succeeded()) return diagnostics;
        const double comparisonTolerance = 1.0e-12 *
            std::max(1.0, std::abs(current.objective));
        if (polished.objective <= current.objective + comparisonTolerance) {
            current = std::move(polished);
        }
    }

    NumiHumanMuscleEquilibriumResult candidate;
    candidate.q = current.q;
    candidate.activation = current.activation;
    candidate.fiberLength = current.fiberLength;
    candidate.muscleTendonForce = current.muscleTendonForce;
    candidate.passiveMuscleTendonForce = current.passiveMuscleTendonForce;
    candidate.generalizedMuscleForce = current.muscleForce;
    candidate.generalizedPositionLimitForce = current.limitForce;
    candidate.generalizedJointEqualityForce = current.equalityForce;
    candidate.supportNormalForce = current.supportNormalForce;
    candidate.generalizedSupportForce = current.supportForce;
    candidate.generalizedPassiveCoordinateForce =
        current.passiveCoordinateForce;
    candidate.gravityTarget = current.target;
    candidate.generalizedForceResidual = current.residual;
    candidate.generalizedAccelerationResidual = current.accelerationResidual;
    candidate.diagnostics.muscleCount =
        static_cast<std::uint32_t>(muscles.size());
    candidate.diagnostics.recruitedMuscleCount = recruitedCount;
    candidate.diagnostics.activationSweeps = current.activationSweeps;
    candidate.diagnostics.globalActivationPolishIterations =
        current.globalActivationPolishIterations;
    candidate.diagnostics.acceptedGlobalActivationPolishSteps =
        current.acceptedGlobalActivationPolishSteps;
    candidate.diagnostics.acceptedPoseSteps = acceptedPoseSteps;
    candidate.diagnostics.jointEqualityCount =
        static_cast<std::uint32_t>(jointEqualities.size());
    candidate.diagnostics.supportContactCount =
        static_cast<std::uint32_t>(supportContacts.size());
    candidate.diagnostics.floatingRootIncluded = !supportContacts.empty();
    candidate.diagnostics.maximumInitialEqualityProjection =
        maximumInitialEqualityProjection;
    candidate.diagnostics.initialNormalizedResidualRms = initialResidual;
    candidate.diagnostics.normalizedResidualRms = current.residualRms;
    candidate.diagnostics.maximumGeneralizedForceResidual =
        current.maximumResidual;
    candidate.diagnostics.maximumGeneralizedAccelerationResidual =
        current.maximumAccelerationResidual;
    for (std::size_t dof = 0u;
         dof < current.accelerationResidual.size(); ++dof) {
        const double acceleration = std::abs(
            current.accelerationResidual[dof]
        );
        if (acceleration == current.maximumAccelerationResidual) {
            candidate.diagnostics.maximumAccelerationResidualDof =
                static_cast<std::uint32_t>(dof);
        }
        const double normalized = std::abs(
            current.weights[dof] * current.accelerationResidual[dof]
        );
        if (normalized >
            candidate.diagnostics.maximumNormalizedAccelerationResidual) {
            candidate.diagnostics.maximumNormalizedAccelerationResidual =
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
    for (const double force : current.supportNormalForce) {
        if (force > 1.0e-6) {
            ++candidate.diagnostics.activeSupportContactCount;
        }
        candidate.diagnostics.totalSupportForceNewtons += force;
        candidate.diagnostics.maximumSupportForceNewtons = std::max(
            candidate.diagnostics.maximumSupportForceNewtons, force);
    }
    if (!supportContacts.empty() && articulation.rootType == MR_ROOT_FLOATING) {
        for (std::size_t dof = 0u;
             dof < std::min<std::size_t>(6u, articulation.nv); ++dof) {
            candidate.diagnostics.maximumFloatingRootForceResidual = std::max(
                candidate.diagnostics.maximumFloatingRootForceResidual,
                std::abs(current.residual[dof]));
            candidate.diagnostics.maximumFloatingRootAccelerationResidual =
                std::max(
                    candidate.diagnostics.maximumFloatingRootAccelerationResidual,
                    std::abs(current.accelerationResidual[dof]));
        }
    }
    candidate.diagnostics.balanced =
        current.residualRms <= config.balanceTolerance;
    result = std::move(candidate);
    return result.diagnostics;
}

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
    const std::span<const NumiHumanStaticSupportContact> supportContacts,
    NumiHumanMuscleEquilibriumResult& result,
    const NumiHumanMuscleEquilibriumConfig& config
) {
    return compileNumiHumanMuscleEquilibrium(
        model, articulationIndex, initialQ, sites, wraps, muscles,
        architectures, jointEqualities, selectedMuscleIndices, supportContacts,
        std::span<const NumiHumanPassiveCoordinateCoupling>{}, result, config);
}

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
    return compileNumiHumanMuscleEquilibrium(
        model, articulationIndex, initialQ, sites, wraps, muscles,
        architectures, jointEqualities, selectedMuscleIndices,
        std::span<const NumiHumanStaticSupportContact>{}, result, config);
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

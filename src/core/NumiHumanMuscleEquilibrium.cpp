#include "metalrobo/NumiHumanMuscleEquilibrium.hpp"
#include "metalrobo/QualityContactSolver.hpp"

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

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
    std::vector<double> supportPlaneGapMeters;
    std::vector<double> supportForce;
    std::vector<double> passiveCoordinateForce;
    std::vector<double> target;
    std::vector<double> residual;
    std::vector<double> limitForce;
    std::vector<double> equalityForce;
    std::vector<double> accelerationResidual;
    std::vector<double> weights;
    std::vector<double> limitMultipliers;
    double limitKktResidual = 0.0;
    double residualRms = 0.0;
    double maximumResidual = 0.0;
    double maximumAccelerationResidual = 0.0;
    double objective = 0.0;
    bool coupledPoseTrial = false;
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
    std::vector<double> fullMass;
    struct LimitRow {
        std::uint32_t sourceDof;
        std::uint32_t independentDof;
        double direction;
        double tangent;
        double scale;
        std::vector<double> acceleration;
    };
    std::vector<LimitRow> limits;
    // Unit-diagonal, equality-reduced J M^-1 J'. The owning cone solver's
    // numerical regularizer is audited against physical complementarity;
    // it is not source tissue compliance.
    ContactSpaceConicProblem limitProblem;
};

constexpr double kLimitRegularization = 1.0e-12;
constexpr double kLimitCertificateTolerance = 1.0e-8;

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

NumiHumanMuscleEquilibriumDiagnostics validatePositionLimits(
    const EngineModel& model,
    const MRArticulationGPU& articulation,
    const std::span<const double> q,
    const double tolerance
) {
    for (std::uint32_t localV = 0u; localV < articulation.nv; ++localV) {
        const auto& dof = model.dofs[articulation.vOffset + localV];
        if ((dof.flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u) continue;
        if (dof.qIndex == MR_INVALID_INDEX || dof.qIndex < articulation.qOffset ||
            dof.qIndex >= articulation.qOffset + articulation.nq ||
            !std::isfinite(dof.limits.x) || !std::isfinite(dof.limits.y) ||
            dof.limits.x > dof.limits.y) {
            return failure(NumiHumanMuscleEquilibriumStatus::invalidDimensions, localV);
        }
        const double position = q[dof.qIndex - articulation.qOffset];
        if (position < dof.limits.x - tolerance || position > dof.limits.y + tolerance) {
            return failure(NumiHumanMuscleEquilibriumStatus::positionLimitViolation, localV);
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
    std::vector<double>& planeGaps,
    const double gapTolerance,
    const ArticulatedDynamicsConfig& dynamicsConfig,
    const bool requireAdmissiblePose = true
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
            !std::isfinite(support.supportRadius) || support.supportRadius < 0.0 ||
            !finiteSpan(support.localPoint) ||
            !finiteSpan(support.normal) ||
            !finiteSpan(support.planePoint) ||
            std::abs(normalLength - 1.0) > 1.0e-6) {
            return failure(
                NumiHumanMuscleEquilibriumStatus::invalidDimensions,
                static_cast<std::uint32_t>(index));
        }
        queries.push_back({support.bodyIndex, support.localPoint, support.supportRadius, support.normal, support.supportRadii, support.supportOrientation});
    }
    if (queries.empty()) {
        generalizedColumns.clear();
        planeGaps.clear();
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
    planeGaps.assign(supports.size(), 0.0);
    for (std::size_t support = 0u; support < supports.size(); ++support) {
        double gap = 0.0;
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            gap += (points[support].position[axis] -
                    supports[support].planePoint[axis]) *
                supports[support].normal[axis];
        }
        if (!std::isfinite(gap)) {
            return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult,
                           static_cast<std::uint32_t>(support));
        }
        if (requireAdmissiblePose && gap < -gapTolerance) {
            return failure(NumiHumanMuscleEquilibriumStatus::supportPenetration,
                           static_cast<std::uint32_t>(support));
        }
        planeGaps[support] = gap;
        // Zero columns keep source indexing stable while removing the force
        // variable for a separated witness from every recruitment objective.
        if (requireAdmissiblePose && gap > gapTolerance) continue;
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
        const bool separated = std::all_of(
            generalizedColumns[support].begin(),
            generalizedColumns[support].end(),
            [](const double value) { return value == 0.0; });
        normalForce[support] = separated ? 0.0 : std::clamp(
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
    candidate.fullMass = mass;
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

bool buildLimitReactionProjection(
    const EngineModel& model,
    const MRArticulationGPU& articulation,
    const std::span<const double> q,
    const double positionTolerance,
    AccelerationProjection& projection
) {
    for (std::size_t column = 0u; column < projection.columns.size(); ++column) {
        for (const auto [sourceDof, tangent] : projection.columns[column]) {
            const auto& dof = model.dofs[articulation.vOffset + sourceDof];
            if ((dof.flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u ||
                dof.qIndex == MR_INVALID_INDEX ||
                dof.qIndex < articulation.qOffset ||
                dof.qIndex >= articulation.qOffset + articulation.nq ||
                tangent == 0.0) continue;
            const double position = q[dof.qIndex - articulation.qOffset];
            for (const double direction : {1.0, -1.0}) {
                const double gap = direction > 0.0
                    ? position - dof.limits.x : dof.limits.y - position;
                if (gap > positionTolerance) continue;
                AccelerationProjection::LimitRow row{
                    sourceDof, projection.independentDofs[column], direction,
                    tangent * direction, 0.0, {}};
                // Scalar source equalities can place several stops on the
                // same signed independent acceleration. They are identical
                // inequalities here. Retain the first source row in stable
                // tangent-column order instead of regularizing a duplicate
                // dual nullspace. Unselected duplicate reactions remain zero.
                if (std::any_of(projection.limits.begin(), projection.limits.end(),
                        [&](const auto& existing) {
                            return existing.independentDof == row.independentDof &&
                                std::signbit(existing.tangent) == std::signbit(row.tangent);
                        })) continue;
                std::vector<double> force(articulation.nv, 0.0);
                force[row.independentDof] = row.tangent;
                if (!projectForceToAcceleration(projection, force, row.acceleration)) {
                    return false;
                }
                const double diagonal = row.tangent * row.acceleration[row.independentDof];
                if (!(diagonal > 0.0) || !std::isfinite(diagonal)) return false;
                row.scale = 1.0 / std::sqrt(diagonal);
                projection.limits.push_back(std::move(row));
            }
        }
    }
    const std::size_t count = projection.limits.size();
    const std::size_t dimension = 3u * count;
    auto& problem = projection.limitProblem;
    problem.contacts.resize(count);
    problem.freeContactVelocity.assign(dimension, 0.0);
    problem.delassus.assign(dimension * dimension, 0.0);
    for (std::size_t row = 0u; row < count; ++row) {
        problem.contacts[row].friction = 0.0;
        problem.contacts[row].regularization.fill(kLimitRegularization);
        for (std::size_t column = 0u; column <= row; ++column) {
            const auto& a = projection.limits[row];
            const auto& b = projection.limits[column];
            const double value = a.scale * a.tangent *
                b.acceleration[a.independentDof] * b.scale;
            problem.delassus[(3u * row) * dimension + 3u * column] = value;
            problem.delassus[(3u * column) * dimension + 3u * row] = value;
        }
    }
    return true;
}

// The same converged native cone solver used by articulated joint limits,
// evaluated at zero velocity in acceleration units. q never changes here.
bool solveLimitReactions(
    const AccelerationProjection& projection,
    const std::span<const double> reducedForce,
    const std::span<const double> warmLimitForce,
    PoseState& state
) {
    if (!projectForceToAcceleration(projection, reducedForce, state.accelerationResidual)) {
        return false;
    }
    state.limitMultipliers.assign(projection.limits.size(), 0.0);
    state.limitKktResidual = 0.0;
    if (projection.limits.empty()) return true;
    auto problem = projection.limitProblem;
    double forceScale = 1.0;
    for (std::size_t index = 0u; index < projection.limits.size(); ++index) {
        const auto& row = projection.limits[index];
        problem.freeContactVelocity[3u * index] = row.scale * row.tangent *
            state.accelerationResidual[row.independentDof];
        forceScale = std::max(forceScale, std::abs(problem.freeContactVelocity[3u * index]));
    }
    // The cone is homogeneous. Scale its linear term and unknown together so
    // a large passive preload cannot consume the solver's absolute precision.
    for (double& value : problem.freeContactVelocity) value /= forceScale;
    // Seed by source DoF, sign and physical force, since the admitted stop
    // rows and their mass scaling can change between posture candidates.
    // The owning solver still enforces its unchanged KKT tolerance and the
    // independent unregularized certificate below.
    for (std::size_t index = 0u; index < projection.limits.size(); ++index) {
        const auto& row = projection.limits[index];
        if (row.sourceDof < warmLimitForce.size() && std::isfinite(warmLimitForce[row.sourceDof])) {
            problem.contacts[index].warmImpulse[0] =
                std::max(0.0, row.direction * warmLimitForce[row.sourceDof]) / (row.scale * forceScale);
        }
    }
    QualityContactSolverConfig config;
    const auto solution = solveQualityContactSpaceProblem(problem, config);
    if (!solution.converged()) return false;
    for (std::size_t index = 0u; index < projection.limits.size(); ++index) {
        const auto& row = projection.limits[index];
        const double multiplier = solution.impulses[3u * index] * forceScale;
        state.limitMultipliers[index] = multiplier;
        const double force = row.scale * multiplier;
        state.limitForce[row.sourceDof] += row.direction * force;
        for (const auto dof : projection.independentDofs) {
            state.accelerationResidual[dof] += row.acceleration[dof] * force;
        }
    }
    // Independently check physical (unregularized) complementarity. A solver
    // success flag alone must not turn a numerical regularizer into support.
    for (std::size_t index = 0u; index < projection.limits.size(); ++index) {
        const auto& row = projection.limits[index];
        const double lambda = state.limitMultipliers[index];
        const double normalAcceleration = row.scale * row.tangent *
            state.accelerationResidual[row.independentDof];
        const double scale = 1.0 + std::abs(problem.freeContactVelocity[3u * index] * forceScale) + lambda;
        state.limitKktResidual = std::max(state.limitKktResidual,
            std::max({0.0, -normalAcceleration, -lambda,
                std::abs(std::min(lambda, normalAcceleration))}) / scale);
    }
    return finiteSpan(state.accelerationResidual) &&
        std::isfinite(state.limitKktResidual) &&
        state.limitKktResidual <= kLimitCertificateTolerance;
}

// Derivative of the converged reaction solve at its current active set.
// Used for recruitment directions only; every accepted state is evaluated
// through the exact nonlinear muscle law and a fresh complementarity solve.
bool projectLimitTangent(
    const AccelerationProjection& projection,
    const PoseState& state,
    std::vector<std::vector<double>>& accelerationColumns
) {
    std::vector<std::size_t> active;
    for (std::size_t index = 0u; index < projection.limits.size(); ++index) {
        if (state.limitMultipliers[index] > 1.0e-10) active.push_back(index);
    }
    const std::size_t count = active.size();
    if (count == 0u) return true;
    const std::size_t dimension = projection.limitProblem.freeContactVelocity.size();
    std::vector<double> lower(count * count, 0.0);
    for (std::size_t row = 0u; row < count; ++row) {
        for (std::size_t column = 0u; column <= row; ++column) {
            double value = projection.limitProblem.delassus[
                3u * active[row] * dimension + 3u * active[column]];
            if (row == column) value += kLimitRegularization;
            for (std::size_t inner = 0u; inner < column; ++inner) {
                value -= lower[row * count + inner] * lower[column * count + inner];
            }
            if (row == column) {
                if (!(value > 0.0) || !std::isfinite(value)) return false;
                lower[row * count + row] = std::sqrt(value);
            } else {
                lower[row * count + column] = value / lower[column * count + column];
            }
        }
    }
    for (auto& acceleration : accelerationColumns) {
        std::vector<double> reaction(count, 0.0);
        for (std::size_t index = 0u; index < count; ++index) {
            const auto& row = projection.limits[active[index]];
            double value = -row.scale * row.tangent * acceleration[row.independentDof];
            for (std::size_t inner = 0u; inner < index; ++inner) {
                value -= lower[index * count + inner] * reaction[inner];
            }
            reaction[index] = value / lower[index * count + index];
        }
        for (std::size_t reverse = 0u; reverse < count; ++reverse) {
            const std::size_t index = count - 1u - reverse;
            for (std::size_t inner = index + 1u; inner < count; ++inner) {
                reaction[index] -= lower[inner * count + index] * reaction[inner];
            }
            reaction[index] /= lower[index * count + index];
        }
        for (std::size_t index = 0u; index < count; ++index) {
            const auto& row = projection.limits[active[index]];
            for (const auto dof : projection.independentDofs) {
                acceleration[dof] += row.acceleration[dof] * row.scale * reaction[index];
            }
        }
        if (!finiteSpan(acceleration)) return false;
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

NumiHumanMuscleEquilibriumDiagnostics finishResidual(
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
    const auto warmLimitForce = state.limitForce;
    state.limitForce.assign(articulation.nv, 0.0);
    state.equalityForce.assign(articulation.nv, 0.0);
    for (std::size_t dof = firstInternal; dof < articulation.nv; ++dof) {
        state.residual[dof] = state.muscleForce[dof] +
            state.supportForce[dof] + state.passiveCoordinateForce[dof] -
            state.target[dof];
    }
    const auto externalForce = state.residual;
    std::vector<double> reducedForce(articulation.nv, 0.0);
    for (std::size_t column = 0u; column < projection.columns.size(); ++column) {
        for (const auto [dof, scale] : projection.columns[column]) {
            reducedForce[projection.independentDofs[column]] += scale * externalForce[dof];
        }
    }
    if (!solveLimitReactions(projection, reducedForce, warmLimitForce, state)) {
        state.residualRms = std::numeric_limits<double>::infinity();
        state.objective = std::numeric_limits<double>::infinity();
        return failure(NumiHumanMuscleEquilibriumStatus::constraintSolveFailure);
    }
    // Lift accelerations through the exact zero-velocity equality tangent.
    // Equality reactions must balance M*a - f, not simply cancel f on a
    // dependent coordinate while the rest of the articulation accelerates.
    for (std::size_t column = 0u; column < projection.columns.size(); ++column) {
        const double acceleration = state.accelerationResidual[projection.independentDofs[column]];
        for (const auto [dof, scale] : projection.columns[column]) {
            state.accelerationResidual[dof] = scale * acceleration;
        }
    }
    for (const auto& equality : equalities) {
        NumiHumanJointEqualityEvaluation evaluation;
        if (!evaluateNumiHumanJointEquality(equality, state.q, evaluation).succeeded()) {
            state.residualRms = std::numeric_limits<double>::infinity();
            state.objective = std::numeric_limits<double>::infinity();
            return failure(NumiHumanMuscleEquilibriumStatus::equalityFailure);
        }
        const auto dependent = equality.indices.y;
        double inertialForce = 0.0;
        for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
            inertialForce += projection.fullMass[dependent * articulation.nv + dof] *
                state.accelerationResidual[dof];
        }
        const double reaction = inertialForce - externalForce[dependent] - state.limitForce[dependent];
        state.equalityForce[dependent] += reaction;
        if (equality.indices.w != MR_INVALID_INDEX) {
            state.equalityForce[equality.indices.w] -= evaluation.derivative * reaction;
        }
    }
    for (std::size_t dof = firstInternal; dof < articulation.nv; ++dof) {
        state.residual[dof] = externalForce[dof] + state.limitForce[dof] + state.equalityForce[dof];
        state.maximumResidual = std::max(state.maximumResidual, std::abs(state.residual[dof]));
        state.maximumAccelerationResidual = std::max(
            state.maximumAccelerationResidual, std::abs(state.accelerationResidual[dof]));
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
    if (!std::isfinite(state.objective) || !std::isfinite(state.maximumResidual) ||
        !std::isfinite(state.maximumAccelerationResidual)) {
        return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
    }
    return {};
}

// Feasible active-set solve of a coupled quadratic recruitment proposal.
// This is an offline search direction, never an additional force authority.
// Diagonal scaling and a roundoff-size LM term condition the normal system;
// only a fresh exact nonlinear objective can accept the resulting direction.
bool boundedRecruitmentDirection(
    const std::vector<std::vector<double>>& columns,
    const std::span<const double> residual,
    const std::span<const double> values,
    const std::span<const double> caps,
    const std::span<const double> penaltyGradient,
    const std::span<const double> penaltyCurvature,
    std::vector<double>& direction
) {
    const std::size_t n = columns.size();
    if (n > static_cast<std::size_t>(std::numeric_limits<__LAPACK_int>::max())) return false;
    if (n == 0u) { direction.clear(); return true; }
    std::vector<double> hessian(n * n), gradient(n), scale(n), lower(n), upper(n);
    for (std::size_t i = 0u; i < n; ++i) {
        gradient[i] = penaltyGradient[i] + std::inner_product(
            columns[i].begin(), columns[i].end(), residual.begin(), 0.0);
        for (std::size_t j = 0u; j <= i; ++j) {
            const double value = std::inner_product(columns[i].begin(),
                columns[i].end(), columns[j].begin(), 0.0) + (i == j ? penaltyCurvature[i] : 0.0);
            hessian[i * n + j] = hessian[j * n + i] = value;
        }
        scale[i] = 1.0 / std::sqrt(std::max(1.0e-20, hessian[i * n + i]));
        lower[i] = -values[i] / scale[i];
        upper[i] = (caps[i] - values[i]) / scale[i];
    }
    for (std::size_t i = 0u; i < n; ++i) {
        gradient[i] *= scale[i];
        for (std::size_t j = 0u; j < n; ++j) hessian[i * n + j] *= scale[i] * scale[j];
        hessian[i * n + i] += 1.0e-12;
    }
    std::vector<double> x(n, 0.0), currentGradient(n), step(n);
    std::vector<int> bound(n, 0);
    for (std::size_t i = 0u; i < n; ++i) {
        if (values[i] == 0.0) bound[i] = -1;
        else if (values[i] == caps[i]) bound[i] = 1;
    }
    for (std::size_t iteration = 0u; iteration < 4u * n + 32u; ++iteration) {
        std::vector<std::size_t> free;
        for (std::size_t i = 0u; i < n; ++i) {
            currentGradient[i] = gradient[i] + std::inner_product(
                hessian.begin() + i * n, hessian.begin() + (i + 1u) * n, x.begin(), 0.0);
            if (bound[i] == 0) free.push_back(i);
        }
        std::fill(step.begin(), step.end(), 0.0);
        if (!free.empty()) {
            const __LAPACK_int order = static_cast<__LAPACK_int>(free.size());
            const __LAPACK_int one = 1;
            const char triangle = 'L';
            __LAPACK_int info = 0;
            std::vector<double> factor(free.size() * free.size()), rhs(free.size());
            for (std::size_t i = 0u; i < free.size(); ++i) {
                rhs[i] = -currentGradient[free[i]];
                for (std::size_t j = 0u; j < free.size(); ++j) {
                    factor[i * free.size() + j] = hessian[free[i] * n + free[j]];
                }
            }
            dposv_(&triangle, &order, &one, factor.data(), &order, rhs.data(), &order, &info);
            if (info != 0 || !finiteSpan(rhs)) return false;
            for (std::size_t i = 0u; i < free.size(); ++i) step[free[i]] = rhs[i];
        }
        double alpha = 1.0;
        std::size_t blocker = n;
        int blockerSide = 0;
        for (const auto i : free) {
            const double distance = step[i] > 0.0 ? upper[i] - x[i] : lower[i] - x[i];
            if (step[i] != 0.0 && distance / step[i] < alpha) {
                alpha = std::max(0.0, distance / step[i]);
                blocker = i;
                blockerSide = step[i] > 0.0 ? 1 : -1;
            }
        }
        for (const auto i : free) x[i] = std::clamp(x[i] + alpha * step[i], lower[i], upper[i]);
        if (blocker != n) {
            bound[blocker] = blockerSide;
            x[blocker] = blockerSide > 0 ? upper[blocker] : lower[blocker];
            continue;
        }
        double worst = 1.0e-10;
        std::size_t release = n;
        for (std::size_t i = 0u; i < n; ++i) {
            const double g = gradient[i] + std::inner_product(
                hessian.begin() + i * n, hessian.begin() + (i + 1u) * n, x.begin(), 0.0);
            const double violation = static_cast<double>(bound[i]) * g;
            if (violation > worst) { worst = violation; release = i; }
        }
        if (release == n) break;
        bound[release] = 0;
    }
    direction.resize(n);
    for (std::size_t i = 0u; i < n; ++i) direction[i] = scale[i] * x[i];
    return finiteSpan(direction);
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
    std::uint32_t& rejectedConstraintCandidates,
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
    const auto limitAdmission = validatePositionLimits(
        model, articulation, state.q, config.positionLimitTolerance);
    if (!limitAdmission.succeeded()) return limitAdmission;
    std::vector<std::vector<double>> supportJacobians;
    auto diagnostics = resolveStaticSupports(
        model, articulationIndex, state.q, supports, supportJacobians,
        state.supportPlaneGapMeters, config.supportGapToleranceMeters,
        dynamicsConfig);
    if (!diagnostics.succeeded()) return diagnostics;
    std::vector<ResolvedMuscle> resolved;
    diagnostics = resolveMuscles(
        model, articulationIndex, state.q, sites, wraps, muscles, resolved,
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
    if (!buildLimitReactionProjection(model, articulation, state.q,
            config.positionLimitTolerance, projection)) {
        return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
    }
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
    auto optimizerJacobians = objectiveJacobians;
    double bestExactObjective = std::numeric_limits<double>::infinity();
    std::vector<double> bestExactActivation;
    std::vector<double> bestExactSupportNormalForce;
    const auto checkpointExactState = [&]()
        -> NumiHumanMuscleEquilibriumDiagnostics {
        PoseState candidate;
        candidate.q = state.q;
        candidate.limitForce = state.limitForce;
        candidate.activation = state.activation;
        candidate.target = state.target;
        candidate.passiveCoordinateForce = state.passiveCoordinateForce;
        candidate.weights = state.weights;
        candidate.supportNormalForce = state.supportNormalForce;
        candidate.supportPlaneGapMeters = state.supportPlaneGapMeters;
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
        const auto residualDiagnostics = finishResidual(
            articulation, config, initialQ, model, articulationIndex,
            equalities, projection, !supports.empty(), candidate);
        if (!residualDiagnostics.succeeded()) return residualDiagnostics;
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
        optimizerJacobians = objectiveJacobians;
        if (!projectLimitTangent(projection, candidate, optimizerJacobians)) {
            return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
        }
        std::fill(state.residual.begin(), state.residual.end(), 0.0);
        for (const auto dof : projection.independentDofs) {
            state.residual[dof] = state.weights[dof] * candidate.accelerationResidual[dof];
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
                        optimizerJacobians[muscle][dof];
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
                        optimizerJacobians[muscle][dof];
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
                    optimizerJacobians[muscle][dof] *
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
                    optimizerJacobians[columnIndex][dof];
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
                    optimizerJacobians[columnIndex][dof] * delta;
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
    const auto residualDiagnostics = finishResidual(
        articulation, config, initialQ, model, articulationIndex, equalities,
        projection, !supports.empty(), state
    );
    if (!residualDiagnostics.succeeded()) return residualDiagnostics;
    if (!std::isfinite(state.residualRms) ||
        !std::isfinite(state.maximumResidual) ||
        !std::isfinite(state.objective)) {
        return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
    }

    // The source-ordered coordinate pass is a robust initializer, but a
    // finite sweep budget can leave order-dependent force sharing. Polish all
    // recruited activations simultaneously in the exact reported objective.
    // The coupled Gauss-Newton proposal uses the fixed-pose acceleration
    // columns; exact compliant-force evaluation and backtracking decide every
    // accepted step, so this cannot promote an interpolated-force regression.
    const std::size_t objectiveRowCount = projection.independentDofs.size();
    const double activationPenaltyScale = config.activationRegularization /
        static_cast<double>(muscles.size());
    const std::uint32_t globalPolishIterations = enableGlobalPolish
        ? config.globalActivationPolishIterations : 0u;
    for (std::uint32_t iteration = 0u;
         iteration < globalPolishIterations; ++iteration) {
        auto constrainedJacobians = objectiveJacobians;
        if (!projectLimitTangent(projection, state, constrainedJacobians)) {
            return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
        }
        state.globalActivationPolishIterations = iteration + 1u;
        std::vector<double> proposal = state.activation;
        std::vector<std::size_t> selected;
        std::vector<std::vector<double>> coupledColumns;
        std::vector<double> selectedActivation, normalizedResidual;
        const double rowScale = 1.0 / std::sqrt(static_cast<double>(std::max<std::size_t>(1u, objectiveRowCount)));
        for (const auto dof : projection.independentDofs) {
            normalizedResidual.push_back(rowScale * state.weights[dof] * state.accelerationResidual[dof]);
        }
        for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
            if (recruited[muscle] == 0u) continue;
            const double activation = std::clamp(
                state.activation[muscle], 0.0, config.activationLimit);
            // Differentiate the exact static force law locally. A secant
            // across the coarse initializer table can cease to be a descent
            // direction near slack/tendon transitions even after convergence
            // of the coupled quadratic model.
            const double lowerActivation = std::max(0.0, activation - 1.0e-5);
            const double upperActivation = std::min(config.activationLimit, activation + 1.0e-5);
            double lowerForce = 0.0, upperForce = 0.0, fiber = 0.0;
            diagnostics = evaluateStaticForce(resolved[muscle].pathLength, lowerActivation,
                config.timestep, muscles[muscle], architectures[muscle], lowerForce, fiber,
                static_cast<std::uint32_t>(muscle));
            if (!diagnostics.succeeded()) return diagnostics;
            diagnostics = evaluateStaticForce(resolved[muscle].pathLength, upperActivation,
                config.timestep, muscles[muscle], architectures[muscle], upperForce, fiber,
                static_cast<std::uint32_t>(muscle));
            if (!diagnostics.succeeded()) return diagnostics;
            const double forceSlope = (upperForce - lowerForce) / (upperActivation - lowerActivation);
            std::vector<double> column;
            for (const auto dof : projection.independentDofs) {
                column.push_back(rowScale * state.weights[dof] * constrainedJacobians[muscle][dof] * forceSlope);
            }
            coupledColumns.push_back(std::move(column));
            selected.push_back(muscle);
            selectedActivation.push_back(activation);
        }
        std::vector<double> direction;
        const std::vector<double> caps(selected.size(), config.activationLimit);
        const std::vector<double> penaltyCurvature(selected.size(), activationPenaltyScale);
        std::vector<double> penaltyGradient;
        for (const double value : selectedActivation) penaltyGradient.push_back(activationPenaltyScale * value);
        if (!boundedRecruitmentDirection(coupledColumns, normalizedResidual,
                selectedActivation, caps, penaltyGradient, penaltyCurvature, direction)) {
            return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
        }
        double maximumProposalChange = 0.0;
        for (std::size_t i = 0u; i < selected.size(); ++i) {
            const auto muscle = selected[i];
            proposal[muscle] = std::clamp(state.activation[muscle] + direction[i], 0.0, config.activationLimit);
            maximumProposalChange = std::max(
                maximumProposalChange, std::abs(proposal[muscle] - state.activation[muscle]));
        }
        if (maximumProposalChange < config.globalActivationConvergence) break;

        bool accepted = false;
        double lineScale = 1.0;
        for (std::uint32_t line = 0u;
             line < config.globalActivationLineSearchSteps; ++line) {
            PoseState candidate;
            candidate.q = state.q;
            candidate.limitForce = state.limitForce;
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
            candidate.supportPlaneGapMeters = state.supportPlaneGapMeters;
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
            const auto residualDiagnostics = finishResidual(
                articulation, config, initialQ, model, articulationIndex,
                equalities, projection, !supports.empty(), candidate);
            if (!residualDiagnostics.succeeded()) {
                if (residualDiagnostics.status == NumiHumanMuscleEquilibriumStatus::constraintSolveFailure) {
                    ++rejectedConstraintCandidates;
                    lineScale *= 0.5;
                    continue;
                }
                return residualDiagnostics;
            }
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
    const auto limitAdmission = validatePositionLimits(
        model, articulation, state.q, config.positionLimitTolerance);
    if (!limitAdmission.succeeded()) return limitAdmission;
    std::vector<std::vector<double>> supportJacobians;
    auto diagnostics = resolveStaticSupports(
        model, articulationIndex, state.q, supports, supportJacobians,
        state.supportPlaneGapMeters, config.supportGapToleranceMeters,
        dynamicsConfig);
    if (!diagnostics.succeeded()) return diagnostics;
    std::vector<ResolvedMuscle> resolved;
    diagnostics = resolveMuscles(
        model, articulationIndex, state.q, sites, wraps, muscles, resolved,
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
    if (!buildLimitReactionProjection(model, articulation, state.q,
            config.positionLimitTolerance, projection)) {
        return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
    }
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
    const auto residualDiagnostics = finishResidual(
        articulation, config, initialQ, model, articulationIndex, equalities,
        projection, !supports.empty(), state
    );
    if (!residualDiagnostics.succeeded()) return residualDiagnostics;
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

NumiHumanMuscleEquilibriumDiagnostics compileNumiHumanSupportPose(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> initialQ,
    const std::span<const MRNumiHumanJointEqualityGPU> equalities,
    const std::span<const NumiHumanStaticSupportContact> supports,
    const std::span<const std::uint32_t> activeSupportIndices,
    const std::span<const NumiHumanSupportPoseCoordinate> coordinates,
    NumiHumanSupportPoseResult& result,
    const NumiHumanSupportPoseConfig& config
) {
    if (config.maximumIterations == 0u || config.maximumIterations > 1024u ||
        config.lineSearchSteps == 0u || config.lineSearchSteps > 64u ||
        !std::isfinite(config.gapToleranceMeters) ||
        config.gapToleranceMeters <= 0.0 ||
        !std::isfinite(config.normalizedStepLimit) ||
        config.normalizedStepLimit <= 0.0 || config.normalizedStepLimit > 1.0) {
        return failure(NumiHumanMuscleEquilibriumStatus::invalidConfiguration);
    }
    if (articulationIndex >= model.articulations.size()) {
        return failure(NumiHumanMuscleEquilibriumStatus::invalidArticulation);
    }
    const auto& art = model.articulations[articulationIndex];
    if (art.nv == 0u || art.nq == 0u || initialQ.size() != art.nq ||
        art.vOffset > model.dofs.size() ||
        art.nv > model.dofs.size() - art.vOffset || supports.empty() ||
        activeSupportIndices.empty() || coordinates.empty()) {
        return failure(NumiHumanMuscleEquilibriumStatus::invalidDimensions);
    }
    if (!finiteSpan(initialQ)) {
        return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteInput);
    }
    std::vector<bool> active(supports.size(), false), dependent(art.nv, false);
    for (const auto index : activeSupportIndices) {
        if (index >= supports.size() || active[index]) {
            return failure(NumiHumanMuscleEquilibriumStatus::invalidSelection, index);
        }
        active[index] = true;
    }
    for (std::size_t index = 0; index < equalities.size(); ++index) {
        const auto& e = equalities[index];
        const bool fixed = e.indices.z == MR_INVALID_INDEX &&
            e.indices.w == MR_INVALID_INDEX;
        if (e.indices.x >= art.nq || e.indices.y >= art.nv ||
            dependent[e.indices.y] ||
            model.dofs[art.vOffset + e.indices.y].qIndex != art.qOffset + e.indices.x ||
            (!fixed && (e.indices.z >= art.nq || e.indices.w >= art.nv ||
              model.dofs[art.vOffset + e.indices.w].qIndex != art.qOffset + e.indices.z))) {
            return failure(NumiHumanMuscleEquilibriumStatus::equalityFailure,
                           static_cast<std::uint32_t>(index));
        }
        dependent[e.indices.y] = true;
    }
    for (const auto& e : equalities) {
        if (e.indices.w != MR_INVALID_INDEX && dependent[e.indices.w]) {
            return failure(NumiHumanMuscleEquilibriumStatus::equalityFailure);
        }
    }
    std::vector<double> seed(initialQ.begin(), initialQ.end());
    const auto projection = projectNumiHumanJointEqualities(equalities, seed);
    if (!projection.succeeded() || !finiteSpan(seed)) {
        return failure(NumiHumanMuscleEquilibriumStatus::equalityFailure,
                       projection.failingIndex);
    }
    const auto withinJointLimits = [&](const std::vector<double>& q) {
        for (std::size_t i = 0; i < art.nv; ++i) {
            const auto& d = model.dofs[art.vOffset + i];
            if ((d.flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u) continue;
            if (d.qIndex == MR_INVALID_INDEX || d.qIndex < art.qOffset ||
                d.qIndex >= art.qOffset + art.nq || !std::isfinite(d.limits.x) ||
                !std::isfinite(d.limits.y) || d.limits.x > d.limits.y) return false;
            const double value = q[d.qIndex - art.qOffset];
            if (value < d.limits.x || value > d.limits.y) return false;
        }
        return true;
    };
    if (!withinJointLimits(seed)) {
        return failure(NumiHumanMuscleEquilibriumStatus::supportPoseInfeasible);
    }
    const std::size_t n = coordinates.size();
    std::vector<bool> selected(art.nv, false);
    std::vector<std::size_t> qIndices;
    std::vector<double> lower, upper;
    for (std::size_t index = 0; index < n; ++index) {
        const auto& c = coordinates[index];
        if (c.dofIndex >= art.nv || selected[c.dofIndex] || dependent[c.dofIndex] ||
            !std::isfinite(c.maximumDisplacement) || c.maximumDisplacement <= 0.0) {
            return failure(NumiHumanMuscleEquilibriumStatus::invalidSelection,
                           static_cast<std::uint32_t>(index));
        }
        const auto& d = model.dofs[art.vOffset + c.dofIndex];
        const bool root = (d.flags & MR_DOF_FLAG_ROOT) != 0u;
        if (d.qIndex == MR_INVALID_INDEX || d.qIndex < art.qOffset ||
            d.qIndex >= art.qOffset + art.nq ||
            (root && (art.rootType != MR_ROOT_FLOATING || c.dofIndex >= 3u ||
                      d.qIndex != art.qOffset + c.dofIndex)) ||
            (!root && ((d.flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u ||
                       !std::isfinite(d.limits.x) || !std::isfinite(d.limits.y) ||
                       d.limits.x >= d.limits.y))) {
            return failure(NumiHumanMuscleEquilibriumStatus::invalidSelection,
                           static_cast<std::uint32_t>(index));
        }
        selected[c.dofIndex] = true;
        const auto qi = d.qIndex - art.qOffset;
        qIndices.push_back(qi);
        lower.push_back(root ? seed[qi] - c.maximumDisplacement :
            std::max(seed[qi] - c.maximumDisplacement, double(d.limits.x)));
        upper.push_back(root ? seed[qi] + c.maximumDisplacement :
            std::min(seed[qi] + c.maximumDisplacement, double(d.limits.y)));
        if (seed[qi] < lower.back() || seed[qi] > upper.back()) {
            return failure(NumiHumanMuscleEquilibriumStatus::invalidSelection,
                           static_cast<std::uint32_t>(index));
        }
    }
    ArticulatedDynamicsConfig dynamics;
    NumiHumanSupportPoseResult candidate;
    candidate.q = seed;
    std::vector<std::vector<double>> columns;
    auto evaluate = [&](const std::vector<double>& q,
                        std::vector<std::vector<double>>& jac,
                        std::vector<double>& gaps) {
        return resolveStaticSupports(model, articulationIndex, q, supports,
            jac, gaps, config.gapToleranceMeters, dynamics, false);
    };
    auto diagnostics = evaluate(candidate.q, columns, candidate.supportPlaneGapMeters);
    if (!diagnostics.succeeded()) return diagnostics;
    const auto objective = [&](const std::vector<double>& gaps) {
        double value = 0.0;
        for (std::size_t c = 0; c < gaps.size(); ++c) {
            // Active contacts are equalities; unselected contacts are exact
            // unilateral inequalities. No force is introduced by this fit.
            const double residual = active[c] ? gaps[c] : std::min(0.0, gaps[c]);
            value += residual * residual;
        }
        return value;
    };
    for (std::uint32_t iteration = 0; iteration <= config.maximumIterations; ++iteration) {
        candidate.iterations = iteration;
        candidate.minimumGapMeters = *std::min_element(
            candidate.supportPlaneGapMeters.begin(), candidate.supportPlaneGapMeters.end());
        candidate.maximumActiveGapMeters = 0.0;
        for (const auto index : activeSupportIndices) {
            candidate.maximumActiveGapMeters = std::max(candidate.maximumActiveGapMeters,
                std::abs(candidate.supportPlaneGapMeters[index]));
        }
        if (candidate.minimumGapMeters >= -config.gapToleranceMeters &&
            candidate.maximumActiveGapMeters <= config.gapToleranceMeters) {
            result = std::move(candidate);
            return {};
        }
        if (iteration == config.maximumIterations) break;
        std::vector<double> derivatives(equalities.size());
        for (std::size_t e = 0; e < equalities.size(); ++e) {
            NumiHumanJointEqualityEvaluation value;
            if (!evaluateNumiHumanJointEquality(equalities[e], candidate.q, value).succeeded()) {
                return failure(NumiHumanMuscleEquilibriumStatus::equalityFailure);
            }
            derivatives[e] = value.derivative;
        }
        std::vector<double> matrix(n * n, 0.0), step(n, 0.0), row(n);
        for (std::size_t c = 0; c < supports.size(); ++c) {
            const double gap = candidate.supportPlaneGapMeters[c];
            if (!active[c] && gap >= 0.0) continue;
            for (std::size_t j = 0; j < n; ++j) {
                const auto dof = coordinates[j].dofIndex;
                double derivative = columns[c][dof];
                for (std::size_t e = 0; e < equalities.size(); ++e) {
                    if (equalities[e].indices.w == dof) {
                        derivative += columns[c][equalities[e].indices.y] * derivatives[e];
                    }
                }
                row[j] = derivative * coordinates[j].maximumDisplacement;
            }
            for (std::size_t j = 0; j < n; ++j) {
                step[j] -= row[j] * gap;
                for (std::size_t k = 0; k < n; ++k) matrix[j*n+k] += row[j] * row[k];
            }
        }
        // Damped minimum-displacement Gauss-Newton in dimensionless bounded
        // coordinates. Damping regularizes redundant contact rows; exact
        // nonlinear geometry and backtracking decide whether a step is used.
        double scale = 0.0;
        for (std::size_t j = 0; j < n; ++j) scale = std::max(scale, matrix[j*n+j]);
        const double damping = std::max(1.0e-16, scale * 1.0e-10);
        for (std::size_t j = 0; j < n; ++j) matrix[j*n+j] += damping;
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k <= j; ++k) {
                double value = matrix[j*n+k];
                for (std::size_t l = 0; l < k; ++l) value -= matrix[j*n+l] * matrix[k*n+l];
                if (j == k) {
                    if (!(value > 0.0) || !std::isfinite(value)) {
                        return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
                    }
                    matrix[j*n+j] = std::sqrt(value);
                } else matrix[j*n+k] = value / matrix[k*n+k];
            }
            for (std::size_t k = 0; k < j; ++k) step[j] -= matrix[j*n+k] * step[k];
            step[j] /= matrix[j*n+j];
        }
        for (std::size_t reverse = 0; reverse < n; ++reverse) {
            const auto j = n - reverse - 1;
            for (std::size_t k = j + 1; k < n; ++k) step[j] -= matrix[k*n+j] * step[k];
            step[j] /= matrix[j*n+j];
        }
        if (!finiteSpan(step)) return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
        double maximumStep = 0.0;
        for (const double value : step) maximumStep = std::max(maximumStep, std::abs(value));
        double fraction = maximumStep > config.normalizedStepLimit
            ? config.normalizedStepLimit / maximumStep : 1.0;
        bool admitted = false;
        const double currentObjective = objective(candidate.supportPlaneGapMeters);
        for (std::uint32_t search = 0; search < config.lineSearchSteps; ++search, fraction *= 0.5) {
            auto q = candidate.q;
            for (std::size_t j = 0; j < n; ++j) {
                q[qIndices[j]] = std::clamp(q[qIndices[j]] +
                    fraction * step[j] * coordinates[j].maximumDisplacement, lower[j], upper[j]);
            }
            if (!projectNumiHumanJointEqualities(equalities, q).succeeded()) {
                return failure(NumiHumanMuscleEquilibriumStatus::equalityFailure);
            }
            if (!withinJointLimits(q)) continue;
            std::vector<std::vector<double>> trialColumns;
            std::vector<double> gaps;
            diagnostics = evaluate(q, trialColumns, gaps);
            if (!diagnostics.succeeded()) return diagnostics;
            if (objective(gaps) < currentObjective) {
                candidate.q = std::move(q);
                candidate.supportPlaneGapMeters = std::move(gaps);
                columns = std::move(trialColumns);
                admitted = true;
                break;
            }
        }
        if (!admitted) break;
    }
    return failure(NumiHumanMuscleEquilibriumStatus::supportPoseInfeasible);
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
        std::isfinite(config.supportGapToleranceMeters) &&
        config.supportGapToleranceMeters >= 0.0 &&
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
    std::uint32_t rejectedConstraintCandidates = 0u;
    PoseState current;
    current.q = projectedInitialQ;
    auto diagnostics = solveActivation(
        model, articulationIndex, projectedInitialQ, sites, wraps, muscles,
        architectures, jointEqualities, supportContacts, passiveCouplings,
        recruited, true, false, config, dynamicsConfig,
        rejectedConstraintCandidates, current
    );
    if (!diagnostics.succeeded()) return diagnostics;
    const double initialResidual = current.residualRms;
    std::vector<NumiHumanEquilibriumSearchRecord> searchTrace{
        {0u, 0u, current.residualRms, current.objective, false, rejectedConstraintCandidates}};
    std::uint32_t acceptedPoseSteps = 0u;
    std::uint32_t acceptedCoupledPoseSteps = 0u;
    std::uint32_t rejectedSupportManifoldPoseCandidates = 0u;
    std::uint32_t rejectedPenetratingPoseCandidates = 0u;
    std::uint32_t rejectedPositionLimitPoseCandidates = 0u;
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
                    candidate.coupledPoseTrial = false;
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
                    if (diagnostics.status ==
                        NumiHumanMuscleEquilibriumStatus::supportPenetration) {
                        ++rejectedPenetratingPoseCandidates;
                        continue;
                    }
                    if (diagnostics.status == NumiHumanMuscleEquilibriumStatus::positionLimitViolation) {
                        ++rejectedPositionLimitPoseCandidates;
                        continue;
                    }
                    if (diagnostics.status == NumiHumanMuscleEquilibriumStatus::constraintSolveFailure) {
                        ++rejectedConstraintCandidates;
                        continue;
                    }
                    if (!diagnostics.succeeded()) return diagnostics;
                    poseTrials.push_back(std::move(candidate));
                }
            }
        }
        if (!candidates.empty()) {
            // A connected limb can require several joints to relax together.
            // Form a bounded joint posture/activation proposal from exact,
            // equality-projected accelerations at frozen current load scales.
            // Renormalizing every perturbation can make a changing spring load
            // falsely constant (a/abs(a)) and erase its search direction. Root coordinates remain fixed and any trial
            // violating the authored ground or a source range is discarded.
            const auto poseResidual = [&](const PoseState& pose) {
                std::vector<double> values;
                const auto count = std::count_if(current.weights.begin(), current.weights.end(),
                    [](double value) { return value != 0.0; });
                const double accelerationScale = 1.0 / std::sqrt(static_cast<double>(std::max<std::ptrdiff_t>(1, count)));
                for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
                    if (current.weights[dof] != 0.0) values.push_back(accelerationScale * current.weights[dof] * pose.accelerationResidual[dof]);
                }
                std::vector<double> posture;
                for (const auto& dof : std::span(model.dofs).subspan(articulation.vOffset, articulation.nv)) {
                    if ((dof.flags & MR_DOF_FLAG_ROOT) != 0u ||
                        (dof.flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u ||
                        dof.qIndex == MR_INVALID_INDEX || dof.qIndex < articulation.qOffset ||
                        dof.qIndex >= articulation.qOffset + articulation.nq) continue;
                    const double range = static_cast<double>(dof.limits.y) - dof.limits.x;
                    if (!(range > kMinimum)) continue;
                    const auto qi = dof.qIndex - articulation.qOffset;
                    posture.push_back((pose.q[qi] - projectedInitialQ[qi]) / range);
                }
                const double postureScale = std::sqrt(config.poseRegularization /
                    static_cast<double>(std::max<std::size_t>(1u, posture.size())));
                for (const double value : posture) values.push_back(postureScale * value);
                const double activationScale = std::sqrt(config.activationRegularization /
                    static_cast<double>(std::max<std::size_t>(1u, pose.activation.size())));
                for (const double value : pose.activation) values.push_back(activationScale * value);
                return values;
            };
            // Fixed-pose activation derivatives share native geometry and inertia.
            // The accepted reaction active-set tangent differentiates their force
            // columns; exact trial admission independently re-solves reactions.
            std::vector<ResolvedMuscle> postureMuscles;
            diagnostics = resolveMuscles(model, articulationIndex, current.q, sites, wraps,
                muscles, postureMuscles, dynamicsConfig);
            if (!diagnostics.succeeded()) return diagnostics;
            AccelerationProjection postureProjection;
            diagnostics = buildAccelerationProjection(model, articulationIndex, current.q,
                jointEqualities, dynamicsConfig, !supportContacts.empty(), postureProjection);
            if (!diagnostics.succeeded()) return diagnostics;
            if (!buildLimitReactionProjection(model, articulation, current.q,
                    config.positionLimitTolerance, postureProjection)) return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
            const auto baseResidual = poseResidual(current);
            std::vector<std::vector<double>> poseColumns;
            std::vector<double> poseValues, poseCaps, poseLower;
            std::vector<std::size_t> poseQ;
            std::vector<bool> activationVariable;
            std::vector<bool> loadedStop(articulation.nv, false);
            for (std::size_t i = 0u; i < postureProjection.limits.size(); ++i) {
                if (current.limitMultipliers[i] > 1.0e-10) loadedStop[postureProjection.limits[i].independentDof] = true;
            }
            std::vector<std::uint32_t> activeContacts;
            for (std::size_t i = 0u; i < current.supportNormalForce.size(); ++i) {
                if (current.supportNormalForce[i] > 0.0) activeContacts.push_back(static_cast<std::uint32_t>(i));
            }
            std::vector<NumiHumanSupportPoseCoordinate> contactCoordinates;
            for (const auto dof : poseCandidates(model, articulationIndex, current, articulation.nv)) {
                if (loadedStop[dof]) continue;
                const auto& property = model.dofs[articulation.vOffset + dof];
                contactCoordinates.push_back({dof, std::min(config.maximumPoseStep,
                    config.poseStepFraction * (static_cast<double>(property.limits.y) - property.limits.x))});
            }
            const auto projectContactManifold = [&](PoseState& pose)
                -> NumiHumanMuscleEquilibriumDiagnostics {
                const auto equalityStatus = projectNumiHumanJointEqualities(jointEqualities, pose.q);
                if (!equalityStatus.succeeded()) {
                    auto failed = failure(NumiHumanMuscleEquilibriumStatus::equalityFailure, equalityStatus.failingIndex);
                    failed.equalityStatus = equalityStatus.status;
                    return failed;
                }
                if (!activeContacts.empty() && !contactCoordinates.empty()) {
                    NumiHumanSupportPoseResult placed;
                    const auto placedStatus = compileNumiHumanSupportPose(model, articulationIndex,
                        pose.q, jointEqualities, supportContacts, activeContacts, contactCoordinates, placed);
                    if (!placedStatus.succeeded()) return placedStatus;
                    pose.q = std::move(placed.q);
                }
                for (const auto& coordinate : contactCoordinates) {
                    const auto& property = model.dofs[articulation.vOffset + coordinate.dofIndex];
                    const auto qi = property.qIndex - articulation.qOffset;
                    if (std::abs(pose.q[qi] - current.q[qi]) > coordinate.maximumDisplacement + 1.0e-10) {
                        return failure(NumiHumanMuscleEquilibriumStatus::supportPoseInfeasible);
                    }
                }
                return {};
            };
            const auto geometricRejection = [](const NumiHumanMuscleEquilibriumStatus status) {
                return status == NumiHumanMuscleEquilibriumStatus::supportPoseInfeasible ||
                    status == NumiHumanMuscleEquilibriumStatus::supportPenetration ||
                    status == NumiHumanMuscleEquilibriumStatus::positionLimitViolation;
            };
            for (const auto localV : poseCandidates(model, articulationIndex, current, articulation.nv)) {
                if (loadedStop[localV]) continue;
                const auto& dof = model.dofs[articulation.vOffset + localV];
                const auto qi = dof.qIndex - articulation.qOffset;
                const double range = static_cast<double>(dof.limits.y) - dof.limits.x;
                const double stepLimit = std::min(config.maximumPoseStep, config.poseStepFraction * range);
                const double lower = std::max(static_cast<double>(dof.limits.x), current.q[qi] - stepLimit);
                const double upper = std::min(static_cast<double>(dof.limits.y), current.q[qi] + stepLimit);
                if (!(upper > lower) || current.q[qi] < lower || current.q[qi] > upper) continue;
                std::vector<double> column;
                for (const double sign : {1.0, -1.0}) {
                    PoseState probe = current;
                    probe.q[qi] = std::clamp(current.q[qi] + sign * std::min(1.0e-5, 1.0e-4 * range), lower, upper);
                    const double delta = probe.q[qi] - current.q[qi];
                    if (std::abs(delta) < 1.0e-12) continue;
                    const auto placement = projectContactManifold(probe);
                    if (!placement.succeeded()) {
                        if (geometricRejection(placement.status)) continue;
                        return placement;
                    }
                    const auto probeStatus = evaluatePoseWithActivation(model, articulationIndex,
                        projectedInitialQ, sites, wraps, muscles, architectures, jointEqualities,
                        supportContacts, passiveCouplings, config, dynamicsConfig, probe);
                    if (!probeStatus.succeeded()) {
                        if (probeStatus.status == NumiHumanMuscleEquilibriumStatus::constraintSolveFailure) {
                            ++rejectedConstraintCandidates;
                            continue;
                        }
                        if (probeStatus.status == NumiHumanMuscleEquilibriumStatus::supportPenetration ||
                            probeStatus.status == NumiHumanMuscleEquilibriumStatus::positionLimitViolation) continue;
                        return probeStatus;
                    }
                    const auto perturbedResidual = poseResidual(probe);
                    if (perturbedResidual.size() != baseResidual.size()) continue;
                    column.resize(baseResidual.size());
                    for (std::size_t row = 0u; row < column.size(); ++row) {
                        column[row] = (perturbedResidual[row] - baseResidual[row]) / delta;
                    }
                    break;
                }
                if (column.empty() || !finiteSpan(column)) continue;
                poseColumns.push_back(std::move(column));
                poseQ.push_back(qi);
                activationVariable.push_back(false);
                poseValues.push_back(current.q[qi] - lower);
                poseCaps.push_back(upper - lower);
                poseLower.push_back(lower);
            }
            std::vector<std::vector<double>> muscleAccelerationColumns;
            for (const auto& muscle : postureMuscles) {
                std::vector<double> reduced(articulation.nv, 0.0), acceleration;
                for (std::size_t index = 0u; index < postureProjection.columns.size(); ++index) {
                    for (const auto [dof, scale] : postureProjection.columns[index]) {
                        reduced[postureProjection.independentDofs[index]] += scale * muscle.jacobian[dof];
                    }
                }
                if (!projectForceToAcceleration(postureProjection, reduced, acceleration)) return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
                muscleAccelerationColumns.push_back(std::move(acceleration));
            }
            if (!projectLimitTangent(postureProjection, current, muscleAccelerationColumns)) return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
            for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
                if (recruited[muscle] == 0u) continue;
                const double before = current.activation[muscle];
                const double lower = std::max(0.0, before - 1.0e-5);
                const double upper = std::min(config.activationLimit, before + 1.0e-5);
                double lowerForce = 0.0, upperForce = 0.0, fiber = 0.0;
                diagnostics = evaluateStaticForce(postureMuscles[muscle].pathLength, lower,
                    config.timestep, muscles[muscle], architectures[muscle], lowerForce, fiber,
                    static_cast<std::uint32_t>(muscle));
                if (!diagnostics.succeeded()) return diagnostics;
                diagnostics = evaluateStaticForce(postureMuscles[muscle].pathLength, upper,
                    config.timestep, muscles[muscle], architectures[muscle], upperForce, fiber,
                    static_cast<std::uint32_t>(muscle));
                if (!diagnostics.succeeded()) return diagnostics;
                const double slope = (upperForce - lowerForce) / (upper - lower);
                std::vector<double> column(baseResidual.size(), 0.0);
                std::size_t row = 0u;
                const double rowScale = 1.0 / std::sqrt(static_cast<double>(postureProjection.independentDofs.size()));
                for (const auto dof : postureProjection.independentDofs) {
                    column[row++] = rowScale * current.weights[dof] * muscleAccelerationColumns[muscle][dof] * slope;
                }
                column[column.size() - muscles.size() + muscle] = std::sqrt(config.activationRegularization /
                    static_cast<double>(muscles.size()));
                if (!finiteSpan(column)) return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
                poseColumns.push_back(std::move(column));
                poseQ.push_back(muscle);
                activationVariable.push_back(true);
                poseValues.push_back(before);
                poseCaps.push_back(config.activationLimit);
                poseLower.push_back(0.0);
            }
            if (!poseQ.empty()) {
                const std::vector<double> zeroPenalty(poseQ.size(), 0.0);
                std::vector<double> direction;
                if (!boundedRecruitmentDirection(poseColumns, baseResidual, poseValues,
                        poseCaps, zeroPenalty, zeroPenalty, direction)) {
                    return failure(NumiHumanMuscleEquilibriumStatus::nonfiniteResult);
                }
                double scale = 1.0;
                for (std::uint32_t line = 0u; line < config.globalActivationLineSearchSteps; ++line, scale *= 0.5) {
                    PoseState trial = current;
                    trial.coupledPoseTrial = true;
                    for (std::size_t i = 0u; i < poseQ.size(); ++i) {
                        const double value = poseLower[i] + std::clamp(poseValues[i] + scale * direction[i], 0.0, poseCaps[i]);
                        if (activationVariable[i]) trial.activation[poseQ[i]] = value;
                        else trial.q[poseQ[i]] = value;
                    }
                    const auto placement = projectContactManifold(trial);
                    if (!placement.succeeded()) {
                        if (geometricRejection(placement.status)) {
                            ++rejectedSupportManifoldPoseCandidates;
                            continue;
                        }
                        return placement;
                    }
                    const auto trialStatus = evaluatePoseWithActivation(model, articulationIndex,
                        projectedInitialQ, sites, wraps, muscles, architectures, jointEqualities,
                        supportContacts, passiveCouplings, config, dynamicsConfig, trial);
                    if (trialStatus.succeeded()) {
                        if (trial.objective + config.poseImprovementTolerance < current.objective) {
                            poseTrials.push_back(std::move(trial));
                            break;
                        }
                    }
                    else if (trialStatus.status == NumiHumanMuscleEquilibriumStatus::supportPenetration) ++rejectedPenetratingPoseCandidates;
                    else if (trialStatus.status == NumiHumanMuscleEquilibriumStatus::positionLimitViolation) ++rejectedPositionLimitPoseCandidates;
                    else if (trialStatus.status == NumiHumanMuscleEquilibriumStatus::constraintSolveFailure) ++rejectedConstraintCandidates;
                    else return trialStatus;
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
            PoseState recruitedPose = poseTrials[index];
            diagnostics = solveActivation(
                model, articulationIndex, projectedInitialQ, sites, wraps,
                muscles, architectures, jointEqualities, supportContacts,
                passiveCouplings, recruited, true, true, config,
                dynamicsConfig, rejectedConstraintCandidates, recruitedPose
            );
            if (diagnostics.status == NumiHumanMuscleEquilibriumStatus::constraintSolveFailure) {
                ++rejectedConstraintCandidates;
                continue;
            }
            if (!diagnostics.succeeded()) return diagnostics;
            recruitedPose.coupledPoseTrial = poseTrials[index].coupledPoseTrial;
            if (recruitedPose.objective + config.poseImprovementTolerance <
                best.objective) {
                best = std::move(recruitedPose);
                found = true;
            }
        }
        if (!found) break;
        current = std::move(best);
        if (current.coupledPoseTrial) ++acceptedCoupledPoseSteps;
        ++acceptedPoseSteps;
        searchTrace.push_back({1u, acceptedPoseSteps, current.residualRms, current.objective,
            current.coupledPoseTrial, rejectedConstraintCandidates});
    }

    // Refine the accepted posture once more without resetting its recruited
    // state. Every pose candidate was compared after the same coupled polish.
    if (config.poseSweeps > 0u &&
        config.globalActivationPolishIterations > 0u) {
        PoseState polished = current;
        diagnostics = solveActivation(
            model, articulationIndex, projectedInitialQ, sites, wraps,
            muscles, architectures, jointEqualities, supportContacts,
            passiveCouplings, recruited, true, true, config, dynamicsConfig,
            rejectedConstraintCandidates, polished);
        if (diagnostics.status == NumiHumanMuscleEquilibriumStatus::constraintSolveFailure) {
            ++rejectedConstraintCandidates;
            polished = current;
        } else if (!diagnostics.succeeded()) return diagnostics;
        const double comparisonTolerance = 1.0e-12 *
            std::max(1.0, std::abs(current.objective));
        if (polished.objective <= current.objective + comparisonTolerance) {
            current = std::move(polished);
        }
    }

    searchTrace.push_back({2u, acceptedPoseSteps, current.residualRms, current.objective,
        false, rejectedConstraintCandidates});
    NumiHumanMuscleEquilibriumResult candidate;
    candidate.searchTrace = std::move(searchTrace);
    candidate.q = current.q;
    candidate.activation = current.activation;
    candidate.fiberLength = current.fiberLength;
    candidate.muscleTendonForce = current.muscleTendonForce;
    candidate.passiveMuscleTendonForce = current.passiveMuscleTendonForce;
    candidate.generalizedMuscleForce = current.muscleForce;
    candidate.generalizedPositionLimitForce = current.limitForce;
    candidate.generalizedJointEqualityForce = current.equalityForce;
    candidate.supportNormalForce = current.supportNormalForce;
    candidate.supportPlaneGapMeters = current.supportPlaneGapMeters;
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
    candidate.diagnostics.acceptedCoupledPoseSteps = acceptedCoupledPoseSteps;
    candidate.diagnostics.rejectedConstraintCandidates = rejectedConstraintCandidates;
    candidate.diagnostics.rejectedSupportManifoldPoseCandidates = rejectedSupportManifoldPoseCandidates;
    candidate.diagnostics.rejectedPenetratingPoseCandidates =
        rejectedPenetratingPoseCandidates;
    candidate.diagnostics.rejectedPositionLimitPoseCandidates = rejectedPositionLimitPoseCandidates;
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
    candidate.diagnostics.positionLimitKktResidual = current.limitKktResidual;
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
    case NumiHumanMuscleEquilibriumStatus::supportPoseInfeasible:
        return "supportPoseInfeasible";
    case NumiHumanMuscleEquilibriumStatus::supportPenetration:
        return "supportPenetration";
    case NumiHumanMuscleEquilibriumStatus::positionLimitViolation:
        return "positionLimitViolation";
    case NumiHumanMuscleEquilibriumStatus::constraintSolveFailure:
        return "constraintSolveFailure";
    }
    return "unknown";
}

} // namespace metalrobo

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalArticulatedInverseMass.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/SurgicalPSM.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kEnvironmentCount = 5u;
constexpr std::size_t kPointCount = 3u;
constexpr std::size_t kRightHandSideCount = 3u;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
bool byteEqual(
    const std::vector<T>& left,
    const std::vector<T>& right
) {
    return
        left.size() == right.size() &&
        (
            left.empty() ||
            std::memcmp(
                left.data(),
                right.data(),
                left.size() * sizeof(T)
            ) == 0
        );
}

bool samePayload(
    const metalrobo::MetalArticulatedInverseMassResult& left,
    const metalrobo::MetalArticulatedInverseMassResult& right
) {
    return
        byteEqual(left.output, right.output) &&
        byteEqual(left.statuses, right.statuses);
}

bool samePayload(
    const metalrobo::MetalArticulatedOperatorResult& left,
    const metalrobo::MetalArticulatedOperatorResult& right
) {
    return
        byteEqual(left.bodyPoses, right.bodyPoses) &&
        byteEqual(left.pointWorld, right.pointWorld) &&
        byteEqual(
            left.diagnosticMassMatrix,
            right.diagnosticMassMatrix
        ) &&
        byteEqual(left.pointJacobians, right.pointJacobians) &&
        byteEqual(
            left.generalizedImpulse,
            right.generalizedImpulse
        ) &&
        byteEqual(left.deltaVelocity, right.deltaVelocity) &&
        byteEqual(left.statuses, right.statuses);
}

std::array<double, 3> xyz(const mr_float4 value) {
    return {
        static_cast<double>(value.x),
        static_cast<double>(value.y),
        static_cast<double>(value.z),
    };
}

MRArticulatedPointImpulseGPU makePoint(
    const std::uint32_t bodyIndex,
    const std::array<double, 3>& localPoint,
    const std::array<float, 3>& worldImpulse
) {
    MRArticulatedPointImpulseGPU point{};
    point.bodyIndex = bodyIndex;
    point.localPoint = {
        static_cast<float>(localPoint[0]),
        static_cast<float>(localPoint[1]),
        static_cast<float>(localPoint[2]),
        0.0f,
    };
    point.worldImpulse = {
        worldImpulse[0],
        worldImpulse[1],
        worldImpulse[2],
        0.0f,
    };
    return point;
}

struct CholeskyFactor {
    std::vector<double> lower;
    double minimumPivot = std::numeric_limits<double>::infinity();
    double maximumPivot = 0.0;
};

CholeskyFactor factorPositiveDefinite(
    const std::span<const double> matrix,
    const std::size_t dimension
) {
    require(
        matrix.size() == dimension * dimension,
        "FP64 mass matrix has invalid dimensions"
    );
    CholeskyFactor factor;
    factor.lower.assign(dimension * dimension, 0.0);
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = 0u; column <= row; ++column) {
            double value = matrix[row * dimension + column];
            for (std::size_t inner = 0u; inner < column; ++inner) {
                value -=
                    factor.lower[row * dimension + inner] *
                    factor.lower[column * dimension + inner];
            }
            if (row == column) {
                require(
                    value > 1.0e-15 && std::isfinite(value),
                    "FP64 CRBA mass matrix is not positive definite"
                );
                const double pivot = std::sqrt(value);
                factor.lower[row * dimension + row] = pivot;
                factor.minimumPivot =
                    std::min(factor.minimumPivot, pivot);
                factor.maximumPivot =
                    std::max(factor.maximumPivot, pivot);
            } else {
                factor.lower[row * dimension + column] =
                    value /
                    factor.lower[column * dimension + column];
            }
        }
    }
    return factor;
}

std::vector<double> solve(
    const CholeskyFactor& factor,
    const std::span<const double> rightHandSide
) {
    const std::size_t dimension = rightHandSide.size();
    require(
        factor.lower.size() == dimension * dimension,
        "FP64 Cholesky solve has invalid dimensions"
    );
    std::vector<double> intermediate(dimension, 0.0);
    std::vector<double> result(dimension, 0.0);
    for (std::size_t row = 0u; row < dimension; ++row) {
        double value = rightHandSide[row];
        for (std::size_t column = 0u; column < row; ++column) {
            value -=
                factor.lower[row * dimension + column] *
                intermediate[column];
        }
        intermediate[row] =
            value / factor.lower[row * dimension + row];
    }
    for (std::size_t reverse = 0u;
         reverse < dimension;
         ++reverse) {
        const std::size_t row = dimension - 1u - reverse;
        double value = intermediate[row];
        for (std::size_t column = row + 1u;
             column < dimension;
             ++column) {
            value -=
                factor.lower[column * dimension + row] *
                result[column];
        }
        result[row] =
            value / factor.lower[row * dimension + row];
    }
    return result;
}

double scaledVectorError(
    const double actual,
    const double expected
) {
    return
        std::abs(actual - expected) /
        (1.0 + std::abs(expected));
}

double scaledMassResidual(
    const std::span<const double> mass,
    const std::span<const double> solution,
    const std::span<const double> rightHandSide
) {
    const std::size_t dimension = solution.size();
    require(
        mass.size() == dimension * dimension &&
            rightHandSide.size() == dimension,
        "mass residual has invalid dimensions"
    );
    double residualInfinityNorm = 0.0;
    double massInfinityNorm = 0.0;
    double solutionInfinityNorm = 0.0;
    double rhsInfinityNorm = 0.0;
    for (std::size_t row = 0u; row < dimension; ++row) {
        double action = 0.0;
        double rowNorm = 0.0;
        for (std::size_t column = 0u;
             column < dimension;
             ++column) {
            const double value = mass[row * dimension + column];
            action += value * solution[column];
            rowNorm += std::abs(value);
        }
        residualInfinityNorm = std::max(
            residualInfinityNorm,
            std::abs(action - rightHandSide[row])
        );
        massInfinityNorm = std::max(massInfinityNorm, rowNorm);
        rhsInfinityNorm = std::max(
            rhsInfinityNorm,
            std::abs(rightHandSide[row])
        );
    }
    for (const double value : solution) {
        solutionInfinityNorm = std::max(
            solutionInfinityNorm,
            std::abs(value)
        );
    }
    return residualInfinityNorm /
        (
            1.0 +
            rhsInfinityNorm +
            massInfinityNorm * solutionInfinityNorm
        );
}

struct Workload {
    std::vector<float> q;
    std::vector<MRArticulatedPointImpulseGPU> points;
    std::vector<float> rightHandSides;

    [[nodiscard]]
    metalrobo::MetalArticulatedOperatorInput operatorInput() const {
        return {
            .articulationIndex = 0u,
            .environmentCount = kEnvironmentCount,
            .pointCount = kPointCount,
            .q = q,
            .points = points,
        };
    }

    [[nodiscard]]
    metalrobo::MetalArticulatedInverseMassInput
    inverseMassInput() const {
        return {
            .articulationIndex = 0u,
            .environmentCount = kEnvironmentCount,
            .rhsCount = kRightHandSideCount,
            .q = q,
            .rightHandSides = rightHandSides,
        };
    }
};

std::array<double, 3> distalPointOnBody(
    const metalrobo::EngineModel& model,
    const std::uint32_t bodyIndex
) {
    const MRShapeGPU* distalShape = nullptr;
    for (const MRShapeGPU& shape : model.shapes) {
        if (shape.bodyIndex != bodyIndex) {
            continue;
        }
        if (distalShape == nullptr ||
            shape.localPosition.z >
                distalShape->localPosition.z) {
            distalShape = &shape;
        }
    }
    require(
        distalShape != nullptr,
        "PSM jaw has no executable distal collision shape"
    );
    return xyz(distalShape->localPosition);
}

Workload makeWorkload(const metalrobo::EngineModel& model) {
    require(
        model.articulations.size() == 1u,
        "surgical probe expects exactly one articulation"
    );
    const MRArticulationGPU& articulation =
        model.articulations.front();
    require(
        articulation.rootType == MR_ROOT_FIXED &&
            articulation.nq == metalrobo::kSurgicalPSMJointCount &&
            articulation.nv == metalrobo::kSurgicalPSMJointCount &&
            model.joints.size() ==
                metalrobo::kSurgicalPSMJointCount &&
            model.joints[2].jointType == MR_JOINT_PRISMATIC,
        "canonical surgical PSM topology is not executable"
    );

    constexpr std::array<
        std::array<float, metalrobo::kSurgicalPSMJointCount>,
        kEnvironmentCount
    > configurations{{
        {{0.01f, 0.01f, 0.060f, 0.01f, 0.01f, 0.01f,
          -0.09f, 0.09f}},
        {{-0.55f, 0.35f, 0.090f, -0.80f, 0.45f, -0.25f,
          -0.20f, 0.20f}},
        {{0.72f, -0.43f, 0.135f, 1.20f, -0.58f, 0.31f,
          -0.32f, 0.32f}},
        {{-1.05f, -0.61f, 0.185f, -2.00f, 0.82f, -0.44f,
          -0.45f, 0.45f}},
        {{1.31f, 0.77f, 0.225f, 3.10f, -1.05f, 0.49f,
          -0.50f, 0.50f}},
    }};

    Workload workload;
    workload.q.reserve(kEnvironmentCount * articulation.nq);
    for (std::size_t environment = 0u;
         environment < kEnvironmentCount;
         ++environment) {
        for (std::size_t dof = 0u; dof < articulation.nq; ++dof) {
            const float value = configurations[environment][dof];
            require(
                value >= model.dofs[dof].limits.x &&
                    value <= model.dofs[dof].limits.y,
                "surgical probe configuration violates a joint limit"
            );
            workload.q.push_back(value);
        }
    }
    require(
        configurations.back()[2] - configurations.front()[2] >
            0.16f,
        "surgical probe does not meaningfully exercise insertion"
    );

    const metalrobo::SurgicalPSMModelMetadata& metadata =
        metalrobo::surgicalPSMMetadata();
    const std::array<double, 3> toolControlPoint =
        xyz(metadata.researchToolControlPointLocalPosition);
    const std::array<double, 3> jawOneDistal =
        distalPointOnBody(model, 7u);
    const std::array<double, 3> jawTwoDistal =
        distalPointOnBody(model, 8u);
    workload.points.reserve(kEnvironmentCount * kPointCount);
    for (std::size_t environment = 0u;
         environment < kEnvironmentCount;
         ++environment) {
        const float e = static_cast<float>(environment + 1u);
        workload.points.push_back(makePoint(
            metadata.researchToolControlPointBodyIndex,
            toolControlPoint,
            {
                0.18f + 0.03f * e,
                -0.11f + 0.02f * e,
                0.09f - 0.01f * e,
            }
        ));
        workload.points.push_back(makePoint(
            7u,
            jawOneDistal,
            {
                -0.06f + 0.015f * e,
                0.13f - 0.009f * e,
                0.05f + 0.011f * e,
            }
        ));
        workload.points.push_back(makePoint(
            8u,
            jawTwoDistal,
            {
                0.04f - 0.007f * e,
                -0.08f + 0.013f * e,
                0.12f - 0.006f * e,
            }
        ));
    }
    return workload;
}

struct CpuReference {
    std::vector<metalrobo::ArticulatedBodyKinematics> bodies;
    std::vector<metalrobo::ArticulatedPointKinematics> points;
    std::vector<double> pointJacobian;
    std::vector<double> mass;
    std::vector<double> generalizedImpulse;
    std::vector<double> deltaVelocity;
    CholeskyFactor massFactor;
};

CpuReference makeCpuReference(
    const metalrobo::EngineModel& model,
    const std::span<const float> qFloat,
    const std::span<const MRArticulatedPointImpulseGPU> points
) {
    const MRArticulationGPU& articulation =
        model.articulations.front();
    std::vector<double> q(qFloat.begin(), qFloat.end());
    std::vector<double> zeroVelocity(articulation.nv, 0.0);

    CpuReference reference;
    reference.bodies.resize(articulation.bodyCount);
    auto diagnostics =
        metalrobo::computeArticulatedBodyKinematics(
            model,
            0u,
            q,
            zeroVelocity,
            reference.bodies
        );
    require(
        diagnostics.succeeded(),
        "FP64 PSM body kinematics failed"
    );

    std::vector<metalrobo::ArticulatedPointQuery> queries(
        points.size()
    );
    for (std::size_t point = 0u; point < points.size(); ++point) {
        queries[point].bodyIndex = points[point].bodyIndex;
        queries[point].localPoint = xyz(points[point].localPoint);
    }
    reference.points.resize(points.size());
    reference.pointJacobian.resize(
        points.size() * 3u * articulation.nv
    );
    diagnostics =
        metalrobo::computeArticulatedPointJacobians(
            model,
            0u,
            q,
            zeroVelocity,
            queries,
            reference.points,
            reference.pointJacobian
        );
    require(
        diagnostics.succeeded(),
        "FP64 PSM point Jacobian failed"
    );

    reference.mass.resize(
        articulation.nv * articulation.nv
    );
    diagnostics = metalrobo::computeArticulatedMassMatrix(
        model,
        0u,
        q,
        reference.mass
    );
    require(
        diagnostics.succeeded(),
        "FP64 PSM CRBA mass matrix failed"
    );
    reference.massFactor = factorPositiveDefinite(
        reference.mass,
        articulation.nv
    );

    reference.generalizedImpulse.assign(
        articulation.nv,
        0.0
    );
    for (std::size_t point = 0u; point < points.size(); ++point) {
        const std::array<double, 3> impulse =
            xyz(points[point].worldImpulse);
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            for (std::size_t dof = 0u;
                 dof < articulation.nv;
                 ++dof) {
                reference.generalizedImpulse[dof] +=
                    reference.pointJacobian[
                        (point * 3u + axis) *
                            articulation.nv +
                        dof
                    ] *
                    impulse[axis];
            }
        }
    }
    reference.deltaVelocity = solve(
        reference.massFactor,
        reference.generalizedImpulse
    );
    require(
        scaledMassResidual(
            reference.mass,
            reference.deltaVelocity,
            reference.generalizedImpulse
        ) < 1.0e-13,
        "FP64 PSM reference solve residual is too large"
    );
    return reference;
}

std::vector<CpuReference> buildReferences(
    const metalrobo::EngineModel& model,
    const Workload& workload
) {
    const MRArticulationGPU& articulation =
        model.articulations.front();
    std::vector<CpuReference> references;
    references.reserve(kEnvironmentCount);
    for (std::size_t environment = 0u;
         environment < kEnvironmentCount;
         ++environment) {
        references.push_back(makeCpuReference(
            model,
            std::span<const float>{
                workload.q.data() +
                    environment * articulation.nq,
                articulation.nq,
            },
            std::span<const MRArticulatedPointImpulseGPU>{
                workload.points.data() +
                    environment * kPointCount,
                kPointCount,
            }
        ));
    }
    return references;
}

void fillRightHandSides(
    const metalrobo::EngineModel& model,
    const std::span<const CpuReference> references,
    Workload& workload
) {
    const std::size_t nv = model.articulations.front().nv;
    workload.rightHandSides.assign(
        kEnvironmentCount * kRightHandSideCount * nv,
        0.0f
    );
    for (std::size_t environment = 0u;
         environment < kEnvironmentCount;
         ++environment) {
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            const std::size_t base =
                environment * kRightHandSideCount * nv;
            workload.rightHandSides[base + dof] =
                static_cast<float>(
                    references[environment]
                        .generalizedImpulse[dof]
                );
            workload.rightHandSides[base + nv + dof] =
                0.35f *
                    std::sin(
                        0.29f *
                        static_cast<float>(
                            (environment + 1u) * (dof + 1u)
                        )
                    ) +
                0.12f *
                    std::cos(
                        0.17f *
                        static_cast<float>(dof + 2u)
                    );
            workload.rightHandSides[base + 2u * nv + dof] =
                (
                    dof == 2u
                        ? 0.85f +
                            0.04f *
                            static_cast<float>(environment)
                        : -0.16f *
                            std::cos(
                                0.23f *
                                static_cast<float>(
                                    (environment + 2u) *
                                    (dof + 1u)
                                )
                            )
                );
        }
    }
}

void requireSuccess(
    const metalrobo::MetalArticulatedInverseMassDiagnostics&
        diagnostics,
    const std::string& operation
) {
    require(
        diagnostics.succeeded() &&
            diagnostics.dispatched &&
            diagnostics.published,
        operation + " failed: " +
            metalrobo::metalArticulatedInverseMassHostStatusName(
                diagnostics.status
            ) +
            " " + diagnostics.message
    );
}

void requireSuccess(
    const metalrobo::MetalArticulatedOperatorDiagnostics&
        diagnostics,
    const std::string& operation
) {
    require(
        diagnostics.succeeded() &&
            diagnostics.dispatched &&
            diagnostics.published,
        operation + " failed: " +
            metalrobo::metalArticulatedOperatorHostStatusName(
                diagnostics.status
            ) +
            " " + diagnostics.message
    );
}

struct Metrics {
    double bodyPosition = 0.0;
    double bodyOrientation = 0.0;
    double pointPosition = 0.0;
    double diagnosticMass = 0.0;
    double diagnosticMassScaled = 0.0;
    double pointJacobian = 0.0;
    double pointJacobianScaled = 0.0;
    double generalizedImpulse = 0.0;
    double generalizedImpulseScaled = 0.0;
    double operatorDeltaVelocity = 0.0;
    double operatorDeltaVelocityScaled = 0.0;
    double operatorEquationResidualScaled = 0.0;
    double inverseMassDeltaVelocity = 0.0;
    double inverseMassDeltaVelocityScaled = 0.0;
    double inverseMassEquationResidualScaled = 0.0;
    double crossKernelDeltaVelocityScaled = 0.0;
    double statusBackwardError = 0.0;
    double minimumCpuPivot =
        std::numeric_limits<double>::infinity();
    double maximumCpuPivot = 0.0;
};

void compareInverseMass(
    const metalrobo::EngineModel& model,
    const Workload& workload,
    const std::span<const CpuReference> references,
    const metalrobo::MetalArticulatedInverseMassResult& gpu,
    Metrics& metrics
) {
    const MRArticulationGPU& articulation =
        model.articulations.front();
    require(
        gpu.output.size() ==
                kEnvironmentCount * kRightHandSideCount *
                    articulation.nv &&
            gpu.statuses.size() == kEnvironmentCount,
        "Metal inverse-mass output has invalid dimensions"
    );
    for (std::size_t environment = 0u;
         environment < kEnvironmentCount;
         ++environment) {
        const MRInverseMassStatusGPU& status =
            gpu.statuses[environment];
        require(
            status.code == MR_INVERSE_MASS_SUCCESS &&
                status.environment == environment &&
                status.articulationIndex == 0u &&
                status.bodyCount == articulation.bodyCount &&
                status.nq == articulation.nq &&
                status.nv == articulation.nv &&
                status.rhsCount == kRightHandSideCount &&
                std::isfinite(status.diagnostics.x) &&
                status.diagnostics.x > 0.0f &&
                std::isfinite(status.diagnostics.y) &&
                status.diagnostics.y >= status.diagnostics.x &&
                std::isfinite(status.diagnostics.z) &&
                std::isfinite(status.diagnostics.w),
            "Metal inverse-mass environment failed"
        );

        const CpuReference& cpu = references[environment];
        metrics.minimumCpuPivot = std::min(
            metrics.minimumCpuPivot,
            cpu.massFactor.minimumPivot
        );
        metrics.maximumCpuPivot = std::max(
            metrics.maximumCpuPivot,
            cpu.massFactor.maximumPivot
        );
        for (std::size_t rhsIndex = 0u;
             rhsIndex < kRightHandSideCount;
             ++rhsIndex) {
            const std::size_t base =
                (
                    environment * kRightHandSideCount +
                    rhsIndex
                ) *
                articulation.nv;
            std::vector<double> rhs(articulation.nv, 0.0);
            std::vector<double> actual(articulation.nv, 0.0);
            for (std::size_t dof = 0u;
                 dof < articulation.nv;
                 ++dof) {
                rhs[dof] = workload.rightHandSides[base + dof];
                actual[dof] = gpu.output[base + dof];
            }
            const std::vector<double> expected =
                solve(cpu.massFactor, rhs);
            for (std::size_t dof = 0u;
                 dof < articulation.nv;
                 ++dof) {
                const double error =
                    std::abs(actual[dof] - expected[dof]);
                metrics.inverseMassDeltaVelocity = std::max(
                    metrics.inverseMassDeltaVelocity,
                    error
                );
                metrics.inverseMassDeltaVelocityScaled = std::max(
                    metrics.inverseMassDeltaVelocityScaled,
                    scaledVectorError(
                        actual[dof],
                        expected[dof]
                    )
                );
            }
            metrics.inverseMassEquationResidualScaled = std::max(
                metrics.inverseMassEquationResidualScaled,
                scaledMassResidual(cpu.mass, actual, rhs)
            );
        }
    }
}

void compareOperator(
    const metalrobo::EngineModel& model,
    const Workload& workload,
    const std::span<const CpuReference> references,
    const metalrobo::MetalArticulatedOperatorResult& gpu,
    const metalrobo::MetalArticulatedInverseMassResult& inverseMass,
    Metrics& metrics
) {
    const MRArticulationGPU& articulation =
        model.articulations.front();
    require(
        gpu.bodyPoses.size() ==
                kEnvironmentCount * articulation.bodyCount &&
            gpu.pointWorld.size() ==
                kEnvironmentCount * kPointCount &&
            gpu.diagnosticMassMatrix.size() ==
                kEnvironmentCount *
                    articulation.nv * articulation.nv &&
            gpu.pointJacobians.size() ==
                kEnvironmentCount * kPointCount * 3u *
                    articulation.nv &&
            gpu.generalizedImpulse.size() ==
                kEnvironmentCount * articulation.nv &&
            gpu.deltaVelocity.size() ==
                kEnvironmentCount * articulation.nv &&
            gpu.statuses.size() == kEnvironmentCount,
        "Metal articulated-operator output has invalid dimensions"
    );

    for (std::size_t environment = 0u;
         environment < kEnvironmentCount;
         ++environment) {
        const CpuReference& cpu = references[environment];
        const MRArticulatedOperatorStatusGPU& status =
            gpu.statuses[environment];
        require(
            status.code == MR_ARTICULATED_OPERATOR_SUCCESS &&
                status.environment == environment &&
                status.articulationIndex == 0u &&
                status.bodyCount == articulation.bodyCount &&
                status.nq == articulation.nq &&
                status.nv == articulation.nv &&
                status.pointCount == kPointCount &&
                std::isfinite(status.diagnostics.x) &&
                status.diagnostics.x > 0.0f &&
                std::isfinite(status.diagnostics.y) &&
                status.diagnostics.y >= status.diagnostics.x &&
                std::isfinite(status.diagnostics.z) &&
                std::isfinite(status.diagnostics.w),
            "Metal articulated-operator environment failed"
        );
        metrics.statusBackwardError = std::max(
            metrics.statusBackwardError,
            static_cast<double>(status.diagnostics.z)
        );

        const std::size_t bodyBase =
            environment * gpu.layout.dispatch.bodyPoseStride;
        for (std::size_t body = 0u;
             body < articulation.bodyCount;
             ++body) {
            const MRArticulatedBodyPoseGPU& actual =
                gpu.bodyPoses[bodyBase + body];
            const metalrobo::ArticulatedBodyKinematics& expected =
                cpu.bodies[body];
            const std::array<double, 3> actualPosition =
                xyz(actual.position);
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                metrics.bodyPosition = std::max(
                    metrics.bodyPosition,
                    std::abs(
                        actualPosition[axis] -
                        expected.centerOfMassPosition[axis]
                    )
                );
            }
            const std::array<double, 4> actualOrientation{
                actual.orientation.x,
                actual.orientation.y,
                actual.orientation.z,
                actual.orientation.w,
            };
            double quaternionDot = 0.0;
            for (std::size_t component = 0u;
                 component < 4u;
                 ++component) {
                quaternionDot +=
                    actualOrientation[component] *
                    expected.orientation[component];
            }
            const double sign =
                quaternionDot < 0.0 ? -1.0 : 1.0;
            for (std::size_t component = 0u;
                 component < 4u;
                 ++component) {
                metrics.bodyOrientation = std::max(
                    metrics.bodyOrientation,
                    std::abs(
                        sign * actualOrientation[component] -
                        expected.orientation[component]
                    )
                );
            }
        }

        const std::size_t pointBase =
            environment * gpu.layout.dispatch.pointWorldStride;
        for (std::size_t point = 0u;
             point < kPointCount;
             ++point) {
            const std::array<double, 3> actual =
                xyz(gpu.pointWorld[pointBase + point].position);
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                metrics.pointPosition = std::max(
                    metrics.pointPosition,
                    std::abs(
                        actual[axis] -
                        cpu.points[point].position[axis]
                    )
                );
            }
        }

        const std::size_t massBase =
            environment * gpu.layout.dispatch.massMatrixStride;
        for (std::size_t index = 0u;
             index < cpu.mass.size();
             ++index) {
            const double error = std::abs(
                gpu.diagnosticMassMatrix[massBase + index] -
                cpu.mass[index]
            );
            metrics.diagnosticMass = std::max(
                metrics.diagnosticMass,
                error
            );
            metrics.diagnosticMassScaled = std::max(
                metrics.diagnosticMassScaled,
                error / (1.0 + std::abs(cpu.mass[index]))
            );
        }

        const std::size_t jacobianBase =
            environment *
            gpu.layout.dispatch.pointJacobianStride;
        for (std::size_t index = 0u;
             index < cpu.pointJacobian.size();
             ++index) {
            const double actual =
                gpu.pointJacobians[jacobianBase + index];
            const double expected = cpu.pointJacobian[index];
            metrics.pointJacobian = std::max(
                metrics.pointJacobian,
                std::abs(actual - expected)
            );
            metrics.pointJacobianScaled = std::max(
                metrics.pointJacobianScaled,
                scaledVectorError(actual, expected)
            );
        }

        const std::size_t generalizedBase =
            environment * gpu.layout.dispatch.generalizedStride;
        std::vector<double> actualDelta(
            articulation.nv,
            0.0
        );
        for (std::size_t dof = 0u;
             dof < articulation.nv;
             ++dof) {
            const double actualImpulse =
                gpu.generalizedImpulse[generalizedBase + dof];
            const double expectedImpulse =
                cpu.generalizedImpulse[dof];
            metrics.generalizedImpulse = std::max(
                metrics.generalizedImpulse,
                std::abs(actualImpulse - expectedImpulse)
            );
            metrics.generalizedImpulseScaled = std::max(
                metrics.generalizedImpulseScaled,
                scaledVectorError(
                    actualImpulse,
                    expectedImpulse
                )
            );

            actualDelta[dof] =
                gpu.deltaVelocity[generalizedBase + dof];
            metrics.operatorDeltaVelocity = std::max(
                metrics.operatorDeltaVelocity,
                std::abs(
                    actualDelta[dof] -
                    cpu.deltaVelocity[dof]
                )
            );
            metrics.operatorDeltaVelocityScaled = std::max(
                metrics.operatorDeltaVelocityScaled,
                scaledVectorError(
                    actualDelta[dof],
                    cpu.deltaVelocity[dof]
                )
            );

            const std::size_t inverseBase =
                environment * kRightHandSideCount *
                articulation.nv;
            metrics.crossKernelDeltaVelocityScaled = std::max(
                metrics.crossKernelDeltaVelocityScaled,
                scaledVectorError(
                    actualDelta[dof],
                    inverseMass.output[inverseBase + dof]
                )
            );
        }
        metrics.operatorEquationResidualScaled = std::max(
            metrics.operatorEquationResidualScaled,
            scaledMassResidual(
                cpu.mass,
                actualDelta,
                cpu.generalizedImpulse
            )
        );
    }
    require(
        workload.points.size() ==
            kEnvironmentCount * kPointCount,
        "operator workload changed during comparison"
    );
}

void validateMetrics(const Metrics& metrics) {
    require(
        metrics.bodyPosition < 1.0e-6 &&
            metrics.bodyOrientation < 1.0e-6 &&
            metrics.pointPosition < 1.0e-6 &&
            metrics.diagnosticMassScaled < 2.0e-6 &&
            metrics.pointJacobianScaled < 2.0e-6 &&
            metrics.generalizedImpulseScaled < 2.0e-6 &&
            metrics.operatorDeltaVelocityScaled < 2.0e-5 &&
            metrics.operatorEquationResidualScaled < 1.0e-6 &&
            metrics.inverseMassDeltaVelocityScaled < 2.0e-5 &&
            metrics.inverseMassEquationResidualScaled < 1.0e-6 &&
            metrics.crossKernelDeltaVelocityScaled < 2.0e-5 &&
            metrics.statusBackwardError <
                MR_ARTICULATED_OPERATOR_MAX_RELATIVE_RESIDUAL,
        "surgical PSM FP64/Metal FP32 parity exceeded tolerance"
    );
}

} // namespace

int main() {
    try {
        const metalrobo::EngineModel model =
            metalrobo::makeDvrkPsmLargeNeedleDriverEngineModel();
        Workload workload = makeWorkload(model);
        const std::vector<CpuReference> references =
            buildReferences(model, workload);
        fillRightHandSides(model, references, workload);

        metalrobo::MetalArticulatedInverseMassContext
            inverseMassContext;
        metalrobo::MetalArticulatedInverseMassResult inverseMass;
        const auto inverseMassDiagnostics = inverseMassContext.run(
            model,
            workload.inverseMassInput(),
            inverseMass
        );
        requireSuccess(
            inverseMassDiagnostics,
            "surgical inverse-mass batch"
        );
        metalrobo::MetalArticulatedInverseMassResult
            inverseMassReplay;
        requireSuccess(
            inverseMassContext.run(
                model,
                workload.inverseMassInput(),
                inverseMassReplay
            ),
            "surgical inverse-mass replay"
        );
        require(
            samePayload(inverseMass, inverseMassReplay),
            "surgical inverse-mass replay was not bit deterministic"
        );

        metalrobo::MetalArticulatedOperatorContext operatorContext{
            {
                .writeDiagnosticMassMatrix = true,
            }
        };
        metalrobo::MetalArticulatedOperatorResult articulated;
        const auto operatorDiagnostics = operatorContext.run(
            model,
            workload.operatorInput(),
            articulated
        );
        requireSuccess(
            operatorDiagnostics,
            "surgical articulated-operator batch"
        );
        metalrobo::MetalArticulatedOperatorResult
            articulatedReplay;
        requireSuccess(
            operatorContext.run(
                model,
                workload.operatorInput(),
                articulatedReplay
            ),
            "surgical articulated-operator replay"
        );
        require(
            samePayload(articulated, articulatedReplay),
            "surgical articulated-operator replay was not "
            "bit deterministic"
        );

        Metrics metrics;
        compareInverseMass(
            model,
            workload,
            references,
            inverseMass,
            metrics
        );
        compareOperator(
            model,
            workload,
            references,
            articulated,
            inverseMass,
            metrics
        );
        validateMetrics(metrics);

        std::cout
            << std::scientific
            << std::setprecision(6)
            << "surgical_metal_operator=Metal"
            << " device=\"" << operatorDiagnostics.deviceName << '"'
            << " environments=" << kEnvironmentCount
            << " rhs=" << kRightHandSideCount
            << " points=" << kPointCount
            << " insertion_span="
            << (
                workload.q[
                    (kEnvironmentCount - 1u) *
                        metalrobo::kSurgicalPSMJointCount +
                    2u
                ] -
                workload.q[2u]
            )
            << " body_pose=" << metrics.bodyPosition
            << " body_orientation=" << metrics.bodyOrientation
            << " point=" << metrics.pointPosition
            << " mass=" << metrics.diagnosticMass
            << " mass_scaled="
            << metrics.diagnosticMassScaled
            << " jacobian=" << metrics.pointJacobian
            << " jacobian_scaled="
            << metrics.pointJacobianScaled
            << " JTp=" << metrics.generalizedImpulse
            << " JTp_scaled="
            << metrics.generalizedImpulseScaled
            << " operator_dv="
            << metrics.operatorDeltaVelocity
            << " operator_dv_scaled="
            << metrics.operatorDeltaVelocityScaled
            << " operator_residual="
            << metrics.operatorEquationResidualScaled
            << " inverse_dv="
            << metrics.inverseMassDeltaVelocity
            << " inverse_dv_scaled="
            << metrics.inverseMassDeltaVelocityScaled
            << " inverse_residual="
            << metrics.inverseMassEquationResidualScaled
            << " cross_kernel_dv_scaled="
            << metrics.crossKernelDeltaVelocityScaled
            << " status_backward_error="
            << metrics.statusBackwardError
            << " cpu_pivot_min=" << metrics.minimumCpuPivot
            << " cpu_pivot_max=" << metrics.maximumCpuPivot
            << " inverse_ms="
            << inverseMassDiagnostics.elapsedMilliseconds
            << " operator_ms="
            << operatorDiagnostics.elapsedMilliseconds
            << " deterministic_replay=exact"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "surgical_metal_operator=failed error=\""
            << error.what() << "\"\n";
        return 1;
    }
}

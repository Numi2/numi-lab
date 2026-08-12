#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/Franka.hpp"
#include "metalrobo/G1.hpp"
#include "metalrobo/MetalArticulatedABA.hpp"
#include "metalrobo/MetalWorld.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

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
    return left.size() == right.size() &&
        (left.empty() ||
         std::memcmp(
             left.data(),
             right.data(),
             left.size() * sizeof(T)
         ) == 0);
}

bool samePayload(
    const metalrobo::MetalWorldResult& left,
    const metalrobo::MetalWorldResult& right
) {
    return byteEqual(left.finalQ, right.finalQ) &&
        byteEqual(left.finalV, right.finalV) &&
        byteEqual(left.observations, right.observations) &&
        byteEqual(left.accelerations, right.accelerations) &&
        byteEqual(left.statuses, right.statuses);
}

double percentile(
    std::vector<double> values,
    const double probability
) {
    require(
        !values.empty() &&
            probability >= 0.0 &&
            probability <= 1.0,
        "invalid percentile request"
    );
    std::sort(values.begin(), values.end());
    const double position =
        probability * static_cast<double>(values.size() - 1u);
    const auto lower =
        static_cast<std::size_t>(std::floor(position));
    const auto upper =
        static_cast<std::size_t>(std::ceil(position));
    const double fraction =
        position - static_cast<double>(lower);
    return values[lower] +
        fraction * (values[upper] - values[lower]);
}

struct OwnedBatch {
    std::size_t environmentCount = 0u;
    std::size_t controlStepCount = 0u;
    std::vector<float> initialQ;
    std::vector<float> initialV;
    std::vector<float> efforts;
    std::vector<std::uint32_t> resetMasks;
    std::vector<float> resetQ;
    std::vector<float> resetV;

    [[nodiscard]] metalrobo::MetalWorldBatch view() const {
        return {
            .environmentCount = environmentCount,
            .controlStepCount = controlStepCount,
            .initialQ = initialQ,
            .initialV = initialV,
            .efforts = efforts,
            .resetMasks = resetMasks,
            .resetQ = resetQ,
            .resetV = resetV,
        };
    }
};

OwnedBatch makeBatch(
    const metalrobo::EngineModel& model,
    const std::size_t environmentCount,
    const std::size_t controlStepCount,
    const bool withResets
) {
    const MRArticulationGPU& articulation =
        model.articulations[0];
    OwnedBatch batch{
        .environmentCount = environmentCount,
        .controlStepCount = controlStepCount,
    };
    batch.initialQ.resize(environmentCount * articulation.nq);
    batch.initialV.resize(environmentCount * articulation.nv);
    batch.efforts.resize(
        controlStepCount * environmentCount * articulation.nv
    );
    if (withResets) {
        batch.resetMasks.assign(
            controlStepCount * environmentCount,
            0u
        );
        batch.resetQ.resize(environmentCount * articulation.nq);
        batch.resetV.resize(environmentCount * articulation.nv);
    }

    for (std::size_t environment = 0u;
         environment < environmentCount;
         ++environment) {
        const std::size_t qBase =
            environment * articulation.nq;
        const std::size_t vBase =
            environment * articulation.nv;
        std::copy_n(
            model.defaultQ.begin() + articulation.qOffset,
            articulation.nq,
            batch.initialQ.begin() +
                static_cast<std::ptrdiff_t>(qBase)
        );
        std::copy_n(
            model.defaultV.begin() + articulation.vOffset,
            articulation.nv,
            batch.initialV.begin() +
                static_cast<std::ptrdiff_t>(vBase)
        );
        for (std::size_t coordinate = 0u;
             coordinate < articulation.nq;
             ++coordinate) {
            const bool floatingRootCoordinate =
                articulation.rootType == MR_ROOT_FLOATING &&
                coordinate < 7u;
            if (!floatingRootCoordinate) {
                batch.initialQ[qBase + coordinate] +=
                    0.003f *
                    std::sin(
                        static_cast<float>(
                            1u + coordinate +
                            3u * environment
                        )
                    );
            }
        }
        if (articulation.rootType == MR_ROOT_FLOATING) {
            batch.initialQ[qBase + 0u] +=
                0.01f * static_cast<float>(environment % 3u);
            const float x =
                0.002f * static_cast<float>(environment % 5u);
            const float y =
                -0.0015f * static_cast<float>(environment % 3u);
            const float z =
                0.001f * static_cast<float>(environment % 7u);
            batch.initialQ[qBase + 3u] = x;
            batch.initialQ[qBase + 4u] = y;
            batch.initialQ[qBase + 5u] = z;
            batch.initialQ[qBase + 6u] =
                std::sqrt(1.0f - x * x - y * y - z * z);
        }
        for (std::size_t dof = 0u;
             dof < articulation.nv;
             ++dof) {
            batch.initialV[vBase + dof] =
                0.01f *
                std::sin(
                    0.17f *
                    static_cast<float>(
                        1u + dof + 2u * environment
                    )
                );
        }

        if (withResets) {
            std::copy_n(
                batch.initialQ.begin() +
                    static_cast<std::ptrdiff_t>(qBase),
                articulation.nq,
                batch.resetQ.begin() +
                    static_cast<std::ptrdiff_t>(qBase)
            );
            std::copy_n(
                batch.initialV.begin() +
                    static_cast<std::ptrdiff_t>(vBase),
                articulation.nv,
                batch.resetV.begin() +
                    static_cast<std::ptrdiff_t>(vBase)
            );
            if (articulation.rootType == MR_ROOT_FLOATING) {
                batch.resetQ[qBase + 1u] += 0.02f;
            } else if (articulation.nq != 0u) {
                batch.resetQ[qBase] +=
                    0.004f *
                    static_cast<float>(1u + environment % 3u);
            }
            for (std::size_t dof = 0u;
                 dof < articulation.nv;
                 ++dof) {
                batch.resetV[vBase + dof] *= -0.5f;
            }
        }
    }

    const std::size_t effortStepStride =
        environmentCount * articulation.nv;
    for (std::size_t step = 0u;
         step < controlStepCount;
         ++step) {
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            for (std::size_t dof = 0u;
                 dof < articulation.nv;
                 ++dof) {
                const MRDofPropertiesGPU& properties =
                    model.dofs[articulation.vOffset + dof];
                const bool actuatorAcceptsEffort =
                    (properties.flags & MR_DOF_FLAG_ACTUATED) != 0u &&
                    (properties.flags & MR_DOF_FLAG_EFFORT_LIMIT) != 0u &&
                    properties.limits.w > 0.0f;
                // Root and unactuated generalized coordinates are not
                // actuators. External loads are covered by the ABA body-wrench
                // path, so the parity fixture must not inject forbidden root
                // effort that MetalWorld correctly resolves to zero.
                batch.efforts[
                    step * effortStepStride +
                    environment * articulation.nv +
                    dof
                ] = actuatorAcceptsEffort
                    ? 0.25f *
                    std::cos(
                        0.11f *
                        static_cast<float>(
                            1u + 3u * step + 5u * environment +
                            dof
                        )
                    )
                    : 0.0f;
            }
        }
    }
    if (withResets && controlStepCount > 2u) {
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            if ((environment % 2u) != 0u) {
                batch.resetMasks[
                    2u * environmentCount + environment
                ] = 1u;
            }
        }
    }
    return batch;
}

struct CPUOracle {
    std::vector<double> finalQ;
    std::vector<double> finalV;
    std::vector<double> observations;
    std::vector<double> accelerations;
};

CPUOracle runCPUOracle(
    const metalrobo::EngineModel& model,
    const OwnedBatch& batch,
    const metalrobo::MetalWorldStepConfig& stepConfig
) {
    const MRArticulationGPU& articulation =
        model.articulations[0];
    const std::size_t observationWidth =
        articulation.nq + articulation.nv;
    CPUOracle oracle;
    oracle.finalQ.assign(
        batch.initialQ.begin(),
        batch.initialQ.end()
    );
    oracle.finalV.assign(
        batch.initialV.begin(),
        batch.initialV.end()
    );
    oracle.observations.resize(
        batch.controlStepCount *
        batch.environmentCount *
        observationWidth
    );
    oracle.accelerations.resize(
        batch.controlStepCount *
        batch.environmentCount *
        articulation.nv
    );

    metalrobo::ArticulatedDynamicsConfig dynamicsConfig;
    dynamicsConfig.gravity = {
        model.world.gravityAndTimestep.x,
        model.world.gravityAndTimestep.y,
        model.world.gravityAndTimestep.z,
    };
    dynamicsConfig.timestep =
        static_cast<double>(stepConfig.timestepSeconds) /
        stepConfig.physicsSubsteps;
    dynamicsConfig.applyBodyDamping =
        stepConfig.applyBodyDamping;

    const std::size_t effortStepStride =
        batch.environmentCount * articulation.nv;
    for (std::size_t step = 0u;
         step < batch.controlStepCount;
         ++step) {
        for (std::size_t environment = 0u;
             environment < batch.environmentCount;
             ++environment) {
            const std::size_t qBase =
                environment * articulation.nq;
            const std::size_t vBase =
                environment * articulation.nv;
            if (!batch.resetMasks.empty() &&
                batch.resetMasks[
                    step * batch.environmentCount + environment
                ] != 0u) {
                std::copy_n(
                    batch.resetQ.begin() +
                        static_cast<std::ptrdiff_t>(qBase),
                    articulation.nq,
                    oracle.finalQ.begin() +
                        static_cast<std::ptrdiff_t>(qBase)
                );
                std::copy_n(
                    batch.resetV.begin() +
                        static_cast<std::ptrdiff_t>(vBase),
                    articulation.nv,
                    oracle.finalV.begin() +
                        static_cast<std::ptrdiff_t>(vBase)
                );
            }

            std::span<double> q{
                oracle.finalQ.data() + qBase,
                articulation.nq,
            };
            std::span<double> v{
                oracle.finalV.data() + vBase,
                articulation.nv,
            };
            std::vector<double> effort(articulation.nv);
            for (std::size_t dof = 0u;
                 dof < articulation.nv;
                 ++dof) {
                effort[dof] = batch.efforts[
                    step * effortStepStride +
                    environment * articulation.nv +
                    dof
                ];
            }
            std::vector<double> acceleration(
                articulation.nv
            );
            for (std::uint32_t substep = 0u;
                 substep < stepConfig.physicsSubsteps;
                 ++substep) {
                const auto dynamics =
                    metalrobo::computeArticulatedForwardDynamics(
                        model,
                        0u,
                        q,
                        v,
                        effort,
                        {},
                        acceleration,
                        dynamicsConfig
                    );
                require(
                    dynamics.succeeded(),
                    "CPU forward-dynamics oracle failed"
                );
                for (std::size_t dof = 0u;
                     dof < articulation.nv;
                     ++dof) {
                    v[dof] +=
                        dynamicsConfig.timestep *
                        acceleration[dof];
                }
                const auto integration =
                    metalrobo::integrateArticulatedConfiguration(
                        model,
                        0u,
                        q,
                        v,
                        dynamicsConfig
                    );
                require(
                    integration.succeeded(),
                    "CPU configuration-integration oracle failed"
                );
            }

            const std::size_t observationBase =
                (step * batch.environmentCount + environment) *
                observationWidth;
            std::copy(
                q.begin(),
                q.end(),
                oracle.observations.begin() +
                    static_cast<std::ptrdiff_t>(
                        observationBase
                    )
            );
            std::copy(
                v.begin(),
                v.end(),
                oracle.observations.begin() +
                    static_cast<std::ptrdiff_t>(
                        observationBase + articulation.nq
                    )
            );
            std::copy(
                acceleration.begin(),
                acceleration.end(),
                oracle.accelerations.begin() +
                    static_cast<std::ptrdiff_t>(
                        (step * batch.environmentCount +
                         environment) *
                        articulation.nv
                    )
            );
        }
    }
    return oracle;
}

struct Parity {
    double qMaximum = 0.0;
    double qGPU = 0.0;
    double qCPU = 0.0;
    std::size_t qStep = 0u;
    std::size_t qEnvironment = 0u;
    std::size_t qCoordinate = 0u;
    double vMaximum = 0.0;
    double vGPU = 0.0;
    double vCPU = 0.0;
    std::size_t vStep = 0u;
    std::size_t vEnvironment = 0u;
    std::size_t vCoordinate = 0u;
    double accelerationScaledMaximum = 0.0;
    double accelerationGPU = 0.0;
    double accelerationCPU = 0.0;
    std::size_t accelerationStep = 0u;
    std::size_t accelerationEnvironment = 0u;
    std::size_t accelerationCoordinate = 0u;
};

Parity compareCPU(
    const metalrobo::MetalWorldResult& gpu,
    const CPUOracle& cpu
) {
    const MRMetalWorldDispatchGPU& dispatch =
        gpu.layout.dispatch;
    Parity parity;
    for (std::size_t step = 0u;
         step < dispatch.controlStepCount;
         ++step) {
        for (std::size_t environment = 0u;
             environment < dispatch.environmentCount;
             ++environment) {
            const std::size_t gpuObservationBase =
                step * dispatch.observationStepStride +
                environment *
                    dispatch.observationEnvironmentStride;
            const std::size_t cpuObservationBase =
                (step * dispatch.environmentCount + environment) *
                (dispatch.nq + dispatch.nv);
            for (std::size_t coordinate = 0u;
                 coordinate < dispatch.nq;
                 ++coordinate) {
                const double gpuValue = gpu.observations[
                    gpuObservationBase + coordinate
                ];
                const double cpuValue = cpu.observations[
                    cpuObservationBase + coordinate
                ];
                const double error = std::abs(gpuValue - cpuValue);
                if (error > parity.qMaximum) {
                    parity.qMaximum = error;
                    parity.qGPU = gpuValue;
                    parity.qCPU = cpuValue;
                    parity.qStep = step;
                    parity.qEnvironment = environment;
                    parity.qCoordinate = coordinate;
                }
            }
            for (std::size_t dof = 0u;
                 dof < dispatch.nv;
                 ++dof) {
                const double gpuVelocity = gpu.observations[
                    gpuObservationBase + dispatch.nq + dof
                ];
                const double cpuVelocity = cpu.observations[
                    cpuObservationBase + dispatch.nq + dof
                ];
                const double velocityError =
                    std::abs(gpuVelocity - cpuVelocity);
                if (velocityError > parity.vMaximum) {
                    parity.vMaximum = velocityError;
                    parity.vGPU = gpuVelocity;
                    parity.vCPU = cpuVelocity;
                    parity.vStep = step;
                    parity.vEnvironment = environment;
                    parity.vCoordinate = dof;
                }
                const std::size_t accelerationIndex =
                    step * dispatch.accelerationStepStride +
                    environment * dispatch.nv + dof;
                const double gpuAcceleration =
                    gpu.accelerations[accelerationIndex];
                const double cpuAcceleration =
                    cpu.accelerations[accelerationIndex];
                const double accelerationError =
                    std::abs(gpuAcceleration - cpuAcceleration) /
                    (1.0 + std::abs(cpuAcceleration));
                if (accelerationError >
                    parity.accelerationScaledMaximum) {
                    parity.accelerationScaledMaximum =
                        accelerationError;
                    parity.accelerationGPU = gpuAcceleration;
                    parity.accelerationCPU = cpuAcceleration;
                    parity.accelerationStep = step;
                    parity.accelerationEnvironment = environment;
                    parity.accelerationCoordinate = dof;
                }
            }
        }
    }
    return parity;
}

void requireSuccess(
    const metalrobo::MetalWorldDiagnostics& diagnostics,
    const std::string& operation
) {
    require(
        diagnostics.succeeded(),
        operation + " failed: " +
            metalrobo::metalWorldHostStatusName(
                diagnostics.status
            ) +
            " (" + diagnostics.message + ")"
    );
}

void requireParity(
    const Parity& parity,
    const std::string& modelName
) {
    if (parity.qMaximum < 7.5e-5 &&
        parity.vMaximum < 1.5e-4 &&
        parity.accelerationScaledMaximum < 7.5e-5) {
        return;
    }
    std::ostringstream message;
    message << std::setprecision(10)
            << modelName
            << " multi-step CPU/Metal parity gate failed: q="
            << parity.qMaximum
            << " (gpu=" << parity.qGPU
            << ", cpu=" << parity.qCPU
            << ", step=" << parity.qStep
            << ", environment=" << parity.qEnvironment
            << ", coordinate=" << parity.qCoordinate
            << "), v=" << parity.vMaximum
            << " (gpu=" << parity.vGPU
            << ", cpu=" << parity.vCPU
            << ", step=" << parity.vStep
            << ", environment=" << parity.vEnvironment
            << ", coordinate=" << parity.vCoordinate
            << "), acceleration_scaled="
            << parity.accelerationScaledMaximum
            << " (gpu=" << parity.accelerationGPU
            << ", cpu=" << parity.accelerationCPU
            << ", step=" << parity.accelerationStep
            << ", environment="
            << parity.accelerationEnvironment
            << ", coordinate="
            << parity.accelerationCoordinate << ')';
    throw std::runtime_error(message.str());
}

void requireFailureRollback(
    const metalrobo::EngineModel& model,
    const OwnedBatch& batch,
    const metalrobo::MetalWorldResult& result
) {
    const MRArticulationGPU& articulation =
        model.articulations[0];
    std::vector<float> acceptedQ = batch.initialQ;
    std::vector<float> acceptedV = batch.initialV;
    const std::size_t observationWidth =
        articulation.nq + articulation.nv;
    for (std::size_t step = 0u;
         step < batch.controlStepCount;
         ++step) {
        for (std::size_t environment = 0u;
             environment < batch.environmentCount;
             ++environment) {
            const std::size_t qBase =
                environment * articulation.nq;
            const std::size_t vBase =
                environment * articulation.nv;
            if (!batch.resetMasks.empty() &&
                batch.resetMasks[
                    step * batch.environmentCount + environment
                ] != 0u) {
                std::copy_n(
                    batch.resetQ.begin() +
                        static_cast<std::ptrdiff_t>(qBase),
                    articulation.nq,
                    acceptedQ.begin() +
                        static_cast<std::ptrdiff_t>(qBase)
                );
                std::copy_n(
                    batch.resetV.begin() +
                        static_cast<std::ptrdiff_t>(vBase),
                    articulation.nv,
                    acceptedV.begin() +
                        static_cast<std::ptrdiff_t>(vBase)
                );
            }
            const std::size_t observationBase =
                (step * batch.environmentCount + environment) *
                observationWidth;
            require(
                std::memcmp(
                    result.observations.data() + observationBase,
                    acceptedQ.data() + qBase,
                    articulation.nq * sizeof(float)
                ) == 0 &&
                    std::memcmp(
                        result.observations.data() +
                            observationBase + articulation.nq,
                        acceptedV.data() + vBase,
                        articulation.nv * sizeof(float)
                    ) == 0,
                "failed control step did not restore its checkpoint"
            );
        }
    }
    require(
        byteEqual(result.finalQ, acceptedQ) &&
            byteEqual(result.finalV, acceptedV) &&
            std::all_of(
                result.accelerations.begin(),
                result.accelerations.end(),
                [](const float value) {
                    return value == 0.0f;
                }
            ),
        "failed rollout published candidate state or acceleration"
    );
}

} // namespace

int main() {
    try {
        const metalrobo::EngineModel franka =
            metalrobo::makeFrankaPandaEngineModel();
        metalrobo::CompiledWorld compiled;
        const auto compile =
            metalrobo::compileMetalWorld(franka, 0u, compiled);
        require(
            compile.succeeded() &&
                compiled.valid() &&
                compiled.fingerprint() == compile.fingerprint &&
                compiled.nq() == franka.articulations[0].nq &&
                compiled.nv() == franka.articulations[0].nv &&
                compiled.capacityClass() ==
                    metalrobo::MetalWorldCapacityClass::
                        compactABA12,
            "Franka CompiledWorld admission failed"
        );

        auto invalidModel = franka;
        invalidModel.world.abiVersion = 0u;
        const std::uint64_t retainedFingerprint =
            compiled.fingerprint();
        const auto invalidCompile =
            metalrobo::compileMetalWorld(
                invalidModel,
                0u,
                compiled
            );
        require(
            !invalidCompile.succeeded() &&
                compiled.fingerprint() == retainedFingerprint,
            "failed CompiledWorld admission mutated the last good snapshot"
        );

        constexpr std::size_t smallEnvironmentCount = 4u;
        constexpr std::size_t smallControlStepCount = 7u;
        const OwnedBatch small = makeBatch(
            franka,
            smallEnvironmentCount,
            smallControlStepCount,
            true
        );
        const metalrobo::MetalWorldStepConfig stepConfig{
            .timestepSeconds = 1.0f / 240.0f,
            .physicsSubsteps = 3u,
            .solverMode =
                metalrobo::MetalWorldSolverMode::freeMotionABA,
            .applyBodyDamping = true,
            .deterministic = true,
        };

        metalrobo::MetalWorldContext context;
        const auto coldStats = context.stats();
        require(
            coldStats.pipelineCreationCount == 0u &&
                coldStats.bufferAllocationCount == 0u &&
                coldStats.modelUploadCount == 0u,
            "MetalWorld context performed eager Metal work"
        );

        metalrobo::MetalWorldResult first;
        const auto firstDiagnostics = context.run(
            compiled,
            small.view(),
            stepConfig,
            first
        );
        requireSuccess(
            firstDiagnostics,
            "first persistent MetalWorld rollout"
        );
        require(
            firstDiagnostics.dispatched &&
                firstDiagnostics.published &&
                firstDiagnostics.successfulStepCount ==
                    smallEnvironmentCount *
                    smallControlStepCount &&
                firstDiagnostics.failedStepCount == 0u,
            "MetalWorld did not publish complete step accounting"
        );

        // High-gain model drives are integrated through M+hD+h^2K in ABA,
        // not converted into an unstable explicit torque.
        OwnedBatch driveBatch{
            .environmentCount = 1u,
            .controlStepCount = 120u,
        };
        driveBatch.initialQ = franka.defaultQ;
        driveBatch.initialV.assign(compiled.nv(), 0.0f);
        driveBatch.initialQ[0] += 0.2f;
        driveBatch.efforts.resize(
            driveBatch.controlStepCount * compiled.nv()
        );
        for (std::size_t step = 0u;
             step < driveBatch.controlStepCount;
             ++step) {
            for (std::size_t dof = 0u;
                 dof < compiled.nv();
                 ++dof) {
                const MRDofPropertiesGPU& properties =
                    franka.dofs[dof];
                driveBatch.efforts[
                    step * compiled.nv() + dof
                ] = properties.qIndex != MR_INVALID_INDEX
                    ? franka.defaultQ[properties.qIndex]
                    : 0.0f;
            }
        }
        metalrobo::MetalWorldStepConfig driveConfig = stepConfig;
        driveConfig.physicsSubsteps = 1u;
        driveConfig.actuationMode =
            metalrobo::MetalWorldActuationMode::
                implicitPositionDrive;
        metalrobo::MetalWorldContext driveContext;
        metalrobo::MetalWorldResult driven;
        const auto driveDiagnostics = driveContext.run(
            compiled,
            driveBatch.view(),
            driveConfig,
            driven
        );
        requireSuccess(
            driveDiagnostics,
            "implicit position-drive rollout"
        );
        require(
            std::abs(
                driven.finalQ[0] - franka.defaultQ[0]
            ) < 0.12f,
            "implicit position drive did not reduce Franka error"
        );

        const CPUOracle cpu =
            runCPUOracle(franka, small, stepConfig);
        const Parity parity = compareCPU(first, cpu);
        requireParity(parity, "Franka");

        auto genericKernelModel = franka;
        genericKernelModel.world.gravityAndTimestep.w =
            stepConfig.timestepSeconds /
            static_cast<float>(stepConfig.physicsSubsteps);
        std::vector<float> genericQ = small.initialQ;
        std::vector<float> genericV = small.initialV;
        const std::size_t smallEffortStepStride =
            smallEnvironmentCount * compiled.nv();
        std::vector<float> firstEffort(
            small.efforts.begin(),
            small.efforts.begin() +
                static_cast<std::ptrdiff_t>(
                    smallEffortStepStride
                )
        );
        metalrobo::MetalArticulatedABAContext genericABA;
        metalrobo::MetalArticulatedABAResult genericStep;
        for (std::uint32_t substep = 0u;
             substep < stepConfig.physicsSubsteps;
             ++substep) {
            const metalrobo::MetalArticulatedABAInput input{
                .articulationIndex = 0u,
                .environmentCount = smallEnvironmentCount,
                .q = genericQ,
                .v = genericV,
                .effort = firstEffort,
                .bodyWrenches = {},
                .applyBodyDamping =
                    stepConfig.applyBodyDamping,
            };
            const auto diagnostics =
                genericABA.run(
                    genericKernelModel,
                    input,
                    genericStep
                );
            require(
                diagnostics.succeeded(),
                "generic-capacity ABA equivalence run failed"
            );
            genericQ = genericStep.nextQ;
            genericV = genericStep.nextV;
        }
        std::vector<float> smallBucketQ(
            smallEnvironmentCount * compiled.nq()
        );
        std::vector<float> smallBucketV(
            smallEnvironmentCount * compiled.nv()
        );
        std::vector<float> smallBucketAcceleration(
            smallEnvironmentCount * compiled.nv()
        );
        for (std::size_t environment = 0u;
             environment < smallEnvironmentCount;
             ++environment) {
            const std::size_t observationBase =
                environment *
                (compiled.nq() + compiled.nv());
            std::copy_n(
                first.observations.begin() +
                    static_cast<std::ptrdiff_t>(
                        observationBase
                    ),
                compiled.nq(),
                smallBucketQ.begin() +
                    static_cast<std::ptrdiff_t>(
                        environment * compiled.nq()
                    )
            );
            std::copy_n(
                first.observations.begin() +
                    static_cast<std::ptrdiff_t>(
                        observationBase + compiled.nq()
                    ),
                compiled.nv(),
                smallBucketV.begin() +
                    static_cast<std::ptrdiff_t>(
                        environment * compiled.nv()
                    )
            );
            std::copy_n(
                first.accelerations.begin() +
                    static_cast<std::ptrdiff_t>(
                        environment * compiled.nv()
                    ),
                compiled.nv(),
                smallBucketAcceleration.begin() +
                    static_cast<std::ptrdiff_t>(
                        environment * compiled.nv()
                    )
            );
        }
        require(
            byteEqual(genericQ, smallBucketQ) &&
                byteEqual(genericV, smallBucketV) &&
                byteEqual(
                    genericStep.acceleration,
                    smallBucketAcceleration
                ),
            "small and generic ABA capacity buckets diverged"
        );

        const auto warmStats = context.stats();
        require(
            warmStats.pipelineCreationCount != 0u &&
                warmStats.modelUploadCount == 1u &&
                warmStats.bufferAllocationCount != 0u &&
                warmStats.bufferGrowthCount == 0u &&
                warmStats.submissionCount == 1u &&
                warmStats.completedSubmissionCount == 1u &&
                !warmStats.hasInFlightSubmission,
            "cold MetalWorld resource counters are inconsistent: pipelines=" +
                std::to_string(warmStats.pipelineCreationCount) +
                " uploads=" +
                std::to_string(warmStats.modelUploadCount) +
                " allocations=" +
                std::to_string(warmStats.bufferAllocationCount) +
                " growths=" +
                std::to_string(warmStats.bufferGrowthCount) +
                " submissions=" +
                std::to_string(warmStats.submissionCount) +
                " completed=" +
                std::to_string(
                    warmStats.completedSubmissionCount
                )
        );

        metalrobo::MetalWorldResult replay;
        requireSuccess(
            context.run(
                compiled,
                small.view(),
                stepConfig,
                replay
            ),
            "bitwise MetalWorld replay"
        );
        require(
            samePayload(first, replay) &&
                context.stats().pipelineCreationCount ==
                    warmStats.pipelineCreationCount &&
                context.stats().modelUploadCount == 1u &&
                context.stats().bufferAllocationCount ==
                    warmStats.bufferAllocationCount,
            "same-device replay changed payload or rebuilt resources"
        );

        OwnedBatch asynchronousBatch = small;
        metalrobo::MetalWorldSubmission pending;
        const auto submitted = context.submit(
            compiled,
            asynchronousBatch.view(),
            stepConfig,
            pending
        );
        require(
            submitted.succeeded() &&
                submitted.dispatched &&
                !submitted.published &&
                pending.valid() &&
                context.stats().hasInFlightSubmission,
            "asynchronous MetalWorld submit did not return a live ticket"
        );
        metalrobo::MetalWorldSubmission rejected;
        const auto busy = context.submit(
            compiled,
            small.view(),
            stepConfig,
            rejected
        );
        const bool rejectedWhileBusy =
            busy.status ==
                metalrobo::MetalWorldHostStatus::contextBusy &&
            !busy.dispatched &&
            !rejected.valid();
        const bool firstSubmissionAlreadyCompleted =
            busy.succeeded() &&
            busy.dispatched &&
            rejected.valid();
        require(
            rejectedWhileBusy ||
                firstSubmissionAlreadyCompleted,
            "shared MetalWorld arena admitted an overlapping submission"
        );
        if (firstSubmissionAlreadyCompleted) {
            metalrobo::MetalWorldResult redundant;
            requireSuccess(
                rejected.wait(redundant),
                "completed-before-overlap replay"
            );
        }
        std::fill(
            asynchronousBatch.initialQ.begin(),
            asynchronousBatch.initialQ.end(),
            std::numeric_limits<float>::quiet_NaN()
        );
        asynchronousBatch.initialV.clear();
        asynchronousBatch.efforts.clear();
        asynchronousBatch.resetMasks.clear();
        asynchronousBatch.resetQ.clear();
        asynchronousBatch.resetV.clear();
        metalrobo::MetalWorldSubmission moved =
            std::move(pending);
        require(
            !pending.valid() && moved.valid(),
            "MetalWorld ticket move did not transfer ownership"
        );
        metalrobo::MetalWorldResult asynchronous;
        requireSuccess(
            moved.wait(asynchronous),
            "asynchronous MetalWorld wait"
        );
        require(
            !moved.valid() &&
                samePayload(first, asynchronous) &&
                !context.stats().hasInFlightSubmission,
            "MetalWorld submit did not snapshot caller-owned spans"
        );

        metalrobo::MetalWorldResult sentinel = first;
        OwnedBatch wrongDimensions = small;
        wrongDimensions.efforts.pop_back();
        const auto badDimensions = context.run(
            compiled,
            wrongDimensions.view(),
            stepConfig,
            sentinel
        );
        require(
            badDimensions.status ==
                    metalrobo::MetalWorldHostStatus::
                        invalidDimensions &&
                samePayload(first, sentinel),
            "dimension rejection was not transactional"
        );
        OwnedBatch nonfinite = small;
        nonfinite.efforts[0] =
            std::numeric_limits<float>::infinity();
        const auto badFinite = context.run(
            compiled,
            nonfinite.view(),
            stepConfig,
            sentinel
        );
        require(
            badFinite.status ==
                    metalrobo::MetalWorldHostStatus::
                        nonfiniteInput &&
                samePayload(first, sentinel),
            "non-finite rejection was not transactional"
        );
        OwnedBatch badReset = small;
        badReset.resetMasks[0] = 2u;
        const auto badResetDiagnostics = context.run(
            compiled,
            badReset.view(),
            stepConfig,
            sentinel
        );
        require(
            badResetDiagnostics.status ==
                    metalrobo::MetalWorldHostStatus::invalidReset &&
                samePayload(first, sentinel),
            "reset rejection was not transactional"
        );
        auto unsupportedConfig = stepConfig;
        unsupportedConfig.solverMode =
            static_cast<metalrobo::MetalWorldSolverMode>(99u);
        const auto unsupported = context.run(
            compiled,
            small.view(),
            unsupportedConfig,
            sentinel
        );
        require(
            unsupported.status ==
                    metalrobo::MetalWorldHostStatus::
                        unsupportedSolverMode &&
                samePayload(first, sentinel),
            "unknown solver mode did not fail closed"
        );
        auto subnormalTimestep = stepConfig;
        subnormalTimestep.timestepSeconds =
            std::numeric_limits<float>::denorm_min();
        const auto badTimestep = context.run(
            compiled,
            small.view(),
            subnormalTimestep,
            sentinel
        );
        require(
            badTimestep.status ==
                    metalrobo::MetalWorldHostStatus::
                        invalidDimensions &&
                samePayload(first, sentinel),
            "subnormal timestep admission was not transactional"
        );

        const OwnedBatch grownBatch =
            makeBatch(franka, 19u, 9u, true);
        metalrobo::MetalWorldResult grown;
        requireSuccess(
            context.run(
                compiled,
                grownBatch.view(),
                stepConfig,
                grown
            ),
            "grown MetalWorld rollout"
        );
        const auto grownStats = context.stats();
        require(
            grownStats.bufferGrowthCount >=
                    warmStats.bufferGrowthCount &&
                grownStats.bufferAllocationCount >=
                    warmStats.bufferAllocationCount &&
                grownStats.retainedBufferBytes >=
                    warmStats.retainedBufferBytes &&
                grownStats.modelUploadCount >=
                    warmStats.modelUploadCount,
            "larger rollout regressed reusable-arena accounting: "
            "warm_growths=" +
                std::to_string(warmStats.bufferGrowthCount) +
                " grown_growths=" +
                std::to_string(grownStats.bufferGrowthCount) +
                " warm_allocations=" +
                std::to_string(
                    warmStats.bufferAllocationCount
                ) +
                " grown_allocations=" +
                std::to_string(
                    grownStats.bufferAllocationCount
                ) +
                " warm_bytes=" +
                std::to_string(warmStats.retainedBufferBytes) +
                " grown_bytes=" +
                std::to_string(grownStats.retainedBufferBytes) +
                " uploads=" +
                std::to_string(grownStats.modelUploadCount)
        );
        const auto allocationsAfterGrowth =
            grownStats.bufferAllocationCount;
        metalrobo::MetalWorldResult shrunk;
        requireSuccess(
            context.run(
                compiled,
                small.view(),
                stepConfig,
                shrunk
            ),
            "post-growth small MetalWorld rollout"
        );
        require(
            context.stats().bufferAllocationCount ==
                    allocationsAfterGrowth &&
                samePayload(first, shrunk),
            "persistent MetalWorld arena shrank or changed replay"
        );

        const metalrobo::EngineModel g1 =
            metalrobo::makeUnitreeG1EngineModel();
        metalrobo::CompiledWorld compiledG1;
        const auto g1Compile =
            metalrobo::compileMetalWorld(g1, 0u, compiledG1);
        require(
            g1Compile.succeeded() &&
                compiledG1.nq() == 36u &&
                compiledG1.nv() == 35u &&
                compiledG1.capacityClass() ==
                    metalrobo::MetalWorldCapacityClass::
                        fullABA32,
            "floating-base G1 did not compile into MetalWorld"
        );
        const OwnedBatch g1Batch = makeBatch(g1, 2u, 4u, true);
        const metalrobo::MetalWorldStepConfig g1Config{
            .timestepSeconds = 1.0f / 480.0f,
            .physicsSubsteps = 2u,
            .solverMode =
                metalrobo::MetalWorldSolverMode::freeMotionABA,
            .applyBodyDamping = true,
            .deterministic = true,
        };
        metalrobo::MetalWorldResult g1Result;
        requireSuccess(
            context.run(
                compiledG1,
                g1Batch.view(),
                g1Config,
                g1Result
            ),
            "floating-base G1 MetalWorld rollout"
        );
        const Parity g1Parity = compareCPU(
            g1Result,
            runCPUOracle(g1, g1Batch, g1Config)
        );
        requireParity(g1Parity, "G1");

        auto tinyPivot =
            metalrobo::makeFreeSphereEngineModel();
        MRBodyPropertiesGPU& tinyBody =
            tinyPivot.bodies[tinyPivot.articulations[0].rootBody];
        tinyBody.massAndInverseMass = {
            1.0e-20f,
            1.0e20f,
            0.0f,
            0.0f,
        };
        tinyBody.inertiaRow0 =
            {1.0e-20f, 0.0f, 0.0f, 0.0f};
        tinyBody.inertiaRow1 =
            {0.0f, 1.0e-20f, 0.0f, 0.0f};
        tinyBody.inertiaRow2 =
            {0.0f, 0.0f, 1.0e-20f, 0.0f};
        tinyBody.inverseInertiaRow0 =
            {1.0e20f, 0.0f, 0.0f, 0.0f};
        tinyBody.inverseInertiaRow1 =
            {0.0f, 1.0e20f, 0.0f, 0.0f};
        tinyBody.inverseInertiaRow2 =
            {0.0f, 0.0f, 1.0e20f, 0.0f};
        std::string reason;
        require(
            tinyPivot.valid(&reason),
            "tiny-pivot rollback canary is invalid: " + reason
        );
        metalrobo::CompiledWorld failingWorld;
        const auto failingCompile =
            metalrobo::compileMetalWorld(
                tinyPivot,
                0u,
                failingWorld
            );
        require(
            failingCompile.succeeded(),
            "tiny-pivot rollback world did not compile"
        );
        OwnedBatch failingBatch =
            makeBatch(tinyPivot, 2u, 4u, true);
        failingBatch.resetMasks[2u] = 1u;
        metalrobo::MetalWorldResult failed;
        const auto gpuFailure = context.run(
            failingWorld,
            failingBatch.view(),
            stepConfig,
            failed
        );
        require(
            gpuFailure.status ==
                    metalrobo::MetalWorldHostStatus::
                        gpuEnvironmentFailure &&
                gpuFailure.published &&
                gpuFailure.failedStepCount ==
                    failingBatch.environmentCount *
                    failingBatch.controlStepCount &&
                std::all_of(
                    failed.statuses.begin(),
                    failed.statuses.end(),
                    [](const MRMetalWorldStatusGPU& status) {
                        return status.code ==
                                   MR_STEP_FACTORIZATION_FAILED &&
                            status.abaCode ==
                                MR_ABA_FACTORIZATION_FAILED &&
                            status.successfulSubsteps == 0u &&
                            status.failingSubstep == 0u;
                    }
                ),
            "GPU failure did not latch typed per-step status"
        );
        requireFailureRollback(
            tinyPivot,
            failingBatch,
            failed
        );

        constexpr std::size_t throughputEnvironmentCount =
            4096u;
        constexpr std::size_t throughputControlStepCount = 16u;
        const OwnedBatch throughputBatch = makeBatch(
            franka,
            throughputEnvironmentCount,
            throughputControlStepCount,
            false
        );
        const metalrobo::MetalWorldStepConfig throughputConfig{
            .timestepSeconds = 1.0f / 240.0f,
            .physicsSubsteps = 4u,
            .solverMode =
                metalrobo::MetalWorldSolverMode::freeMotionABA,
            .applyBodyDamping = true,
            .deterministic = true,
        };
        metalrobo::MetalWorldResult throughputWarmup;
        requireSuccess(
            context.run(
                compiled,
                throughputBatch.view(),
                throughputConfig,
                throughputWarmup
            ),
            "4,096-environment MetalWorld warmup"
        );
        constexpr std::size_t throughputSampleCount = 5u;
        std::vector<double> gpuMilliseconds;
        std::vector<double> wallMillisecondsSamples;
        gpuMilliseconds.reserve(throughputSampleCount);
        wallMillisecondsSamples.reserve(throughputSampleCount);
        metalrobo::MetalWorldResult throughput;
        metalrobo::MetalWorldDiagnostics throughputDiagnostics;
        for (std::size_t sample = 0u;
             sample < throughputSampleCount;
             ++sample) {
            const auto wallStart =
                std::chrono::steady_clock::now();
            auto sampleDiagnostics = context.run(
                compiled,
                throughputBatch.view(),
                throughputConfig,
                throughput
            );
            const auto wallEnd =
                std::chrono::steady_clock::now();
            requireSuccess(
                sampleDiagnostics,
                "4,096-environment MetalWorld throughput rollout"
            );
            require(
                std::isfinite(
                    sampleDiagnostics.gpuElapsedMilliseconds
                ) &&
                    sampleDiagnostics.gpuElapsedMilliseconds > 0.0,
                "Metal did not publish a positive GPU execution interval"
            );
            gpuMilliseconds.push_back(
                sampleDiagnostics.gpuElapsedMilliseconds
            );
            wallMillisecondsSamples.push_back(
                std::chrono::duration<double, std::milli>(
                    wallEnd - wallStart
                ).count()
            );
            throughputDiagnostics =
                std::move(sampleDiagnostics);
        }
        const double controlSteps =
            static_cast<double>(throughputEnvironmentCount) *
            throughputControlStepCount;
        double totalGpuMilliseconds = 0.0;
        double totalWallMilliseconds = 0.0;
        for (const double milliseconds : gpuMilliseconds) {
            totalGpuMilliseconds += milliseconds;
        }
        for (const double milliseconds :
             wallMillisecondsSamples) {
            totalWallMilliseconds += milliseconds;
        }
        const double gpuControlStepsPerSecond =
            1000.0 * controlSteps * throughputSampleCount /
            totalGpuMilliseconds;
        const double wallControlStepsPerSecond =
            1000.0 * controlSteps * throughputSampleCount /
            totalWallMilliseconds;
        const double gpuP50Milliseconds =
            percentile(gpuMilliseconds, 0.50);
        const double gpuP95Milliseconds =
            percentile(gpuMilliseconds, 0.95);
        const double wallP50Milliseconds =
            percentile(wallMillisecondsSamples, 0.50);
        const double wallP95Milliseconds =
            percentile(wallMillisecondsSamples, 0.95);
        require(
            gpuControlStepsPerSecond >= 150000.0 &&
                wallControlStepsPerSecond >= 150000.0,
            "4,096-environment Franka free-space throughput "
            "fell below 150,000 control-steps/s on the GPU or "
            "end-to-end wall clock: " +
                std::to_string(gpuControlStepsPerSecond) +
                " GPU, " +
                std::to_string(wallControlStepsPerSecond) +
                " wall"
        );

        std::cout
            << "metal_world=metal"
            << " device=\"" << firstDiagnostics.deviceName << "\""
            << " abi=" << MR_METAL_WORLD_ABI_VERSION
            << " graph=free_motion_aba"
            << " environments=" << smallEnvironmentCount
            << " control_steps=" << smallControlStepCount
            << " physics_substeps=" << stepConfig.physicsSubsteps
            << " q_error=" << parity.qMaximum
            << " v_error=" << parity.vMaximum
            << " acceleration_scaled_error="
            << parity.accelerationScaledMaximum
            << " g1_q_error=" << g1Parity.qMaximum
            << " g1_v_error=" << g1Parity.vMaximum
            << " g1_acceleration_scaled_error="
            << g1Parity.accelerationScaledMaximum
            << " pipeline_creations="
            << context.stats().pipelineCreationCount
            << " model_uploads="
            << context.stats().modelUploadCount
            << " buffer_growths="
            << context.stats().bufferGrowthCount
            << " retained_bytes="
            << context.stats().retainedBufferBytes
            << " throughput_batch="
            << throughputEnvironmentCount
            << " throughput_horizon="
            << throughputControlStepCount
            << " gpu_control_steps_per_s="
            << gpuControlStepsPerSecond
            << " wall_control_steps_per_s="
            << wallControlStepsPerSecond
            << " gpu_p50_ms=" << gpuP50Milliseconds
            << " gpu_p95_ms=" << gpuP95Milliseconds
            << " wall_p50_ms=" << wallP50Milliseconds
            << " wall_p95_ms=" << wallP95Milliseconds
            << " thermal=" << throughputDiagnostics.thermalState
            << " replay=bitwise"
            << " async=pass"
            << " input_snapshot=pass"
            << " busy_gate=pass"
            << " reset=pass"
            << " rollback=pass"
            << " g1_free_motion=pass"
            << " capacity_bucket_equivalence=bitwise"
            << " grow_only=pass"
            << " host_transaction=pass"
            << " no_host_sync_between_control_steps=yes"
                  << " contact_graph=device_resident"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "metal_world=failed reason=\""
            << exception.what() << "\"\n";
        return 1;
    }
}

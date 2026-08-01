#include "metalrobo/MetalArticulatedInverseMass.hpp"
#include "metalrobo/MetalMultiArticulatedInverseMass.hpp"
#include "metalrobo/SurgicalWorld.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

float maximumDifference(
    const std::span<const float> left,
    const std::span<const float> right
) {
    require(left.size() == right.size(), "comparison size mismatch");
    float maximum = 0.0f;
    for (std::size_t index = 0u; index < left.size(); ++index) {
        maximum = std::max(
            maximum,
            std::abs(left[index] - right[index])
        );
    }
    return maximum;
}

} // namespace

int main() {
    try {
        constexpr std::size_t environmentCount = 4u;
        constexpr std::size_t rhsCount = 3u;
        const metalrobo::DualPsmWorld dual =
            metalrobo::makeDualDvrkPsmWorld();
        const metalrobo::EngineModel& model = dual.model;
        require(
            model.articulations.size() == 2u,
            "dual PSM probe does not contain two articulations"
        );

        std::vector<float> q(
            environmentCount * model.world.nq
        );
        std::vector<float> rhs(
            environmentCount * rhsCount * model.world.nv
        );
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            std::copy(
                model.defaultQ.begin(),
                model.defaultQ.end(),
                q.begin() + environment * model.world.nq
            );
            for (std::size_t rhsIndex = 0u;
                 rhsIndex < rhsCount;
                 ++rhsIndex) {
                for (std::size_t dof = 0u;
                     dof < model.world.nv;
                     ++dof) {
                    rhs[
                        (environment * rhsCount + rhsIndex) *
                            model.world.nv +
                        dof
                    ] =
                        0.125f *
                        std::sin(float(
                            1u +
                            5u * environment +
                            3u * rhsIndex +
                            dof
                        ));
                }
            }
        }

        const metalrobo::MetalMultiArticulatedInverseMassInput
            input{
                .environmentCount = environmentCount,
                .rhsCount = rhsCount,
                .q = q,
                .rightHandSides = rhs,
            };
        metalrobo::MetalMultiArticulatedInverseMassResult result;
        const auto diagnostics =
            metalrobo::runMetalMultiArticulatedInverseMass(
                model,
                input,
                result
            );
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{
                    "multi-articulation inverse mass failed: "
                } + diagnostics.message
            );
        }
        require(
            diagnostics.dispatched &&
                diagnostics.published &&
                result.output.size() == rhs.size() &&
                result.statuses.size() ==
                    environmentCount *
                    model.articulations.size(),
            "multi-articulation response was not published"
        );

        float maximumError = 0.0f;
        for (std::size_t articulationIndex = 0u;
             articulationIndex < model.articulations.size();
             ++articulationIndex) {
            const MRArticulationGPU& articulation =
                model.articulations[articulationIndex];
            std::vector<float> localQ(
                environmentCount * articulation.nq
            );
            std::vector<float> localRhs(
                environmentCount * rhsCount * articulation.nv
            );
            for (std::size_t environment = 0u;
                 environment < environmentCount;
                 ++environment) {
                std::copy_n(
                    q.begin() +
                        environment * model.world.nq +
                        articulation.qOffset,
                    articulation.nq,
                    localQ.begin() +
                        environment * articulation.nq
                );
                for (std::size_t rhsIndex = 0u;
                     rhsIndex < rhsCount;
                     ++rhsIndex) {
                    std::copy_n(
                        rhs.begin() +
                            (environment * rhsCount + rhsIndex) *
                                model.world.nv +
                            articulation.vOffset,
                        articulation.nv,
                        localRhs.begin() +
                            (environment * rhsCount + rhsIndex) *
                                articulation.nv
                    );
                }
            }

            const metalrobo::MetalArticulatedInverseMassInput
                referenceInput{
                    .articulationIndex =
                        static_cast<std::uint32_t>(
                            articulationIndex
                        ),
                    .environmentCount = environmentCount,
                    .rhsCount = rhsCount,
                    .q = localQ,
                    .rightHandSides = localRhs,
                };
            metalrobo::MetalArticulatedInverseMassResult reference;
            const auto referenceDiagnostics =
                metalrobo::runMetalArticulatedInverseMass(
                    model,
                    referenceInput,
                    reference
                );
            require(
                referenceDiagnostics.succeeded(),
                "single-articulation inverse-mass reference failed"
            );
            for (std::size_t environment = 0u;
                 environment < environmentCount;
                 ++environment) {
                for (std::size_t rhsIndex = 0u;
                     rhsIndex < rhsCount;
                     ++rhsIndex) {
                    maximumError = std::max(
                        maximumError,
                        maximumDifference(
                            std::span{
                                result.output.data() +
                                    (environment * rhsCount +
                                     rhsIndex) *
                                        model.world.nv +
                                    articulation.vOffset,
                                articulation.nv,
                            },
                            std::span{
                                reference.output.data() +
                                    (environment * rhsCount +
                                     rhsIndex) *
                                        articulation.nv,
                                articulation.nv,
                            }
                        )
                    );
                }
            }
        }
        require(
            maximumError == 0.0f,
            "2-D inverse-mass grid diverged from reference kernels"
        );

        std::vector<mr_float4> bodyParameters(
            environmentCount * model.world.bodyCount,
            mr_float4{1.0f, 1.0f, 1.0f, 1.0f}
        );
        std::vector<mr_float4> controllerParameters(
            environmentCount,
            mr_float4{1.0f, 1.0f, 0.0f, 0.0f}
        );
        auto parameterizedInput = input;
        parameterizedInput.bodyParameters = bodyParameters;
        parameterizedInput.controllerParameters =
            controllerParameters;
        metalrobo::MetalMultiArticulatedInverseMassResult
            parameterizedIdentity;
        const auto parameterizedIdentityDiagnostics =
            metalrobo::runMetalMultiArticulatedInverseMass(
                model,
                parameterizedInput,
                parameterizedIdentity
            );
        require(
            parameterizedIdentityDiagnostics.succeeded() &&
                parameterizedIdentity.output == result.output,
            "identity physical parameters changed inverse mass"
        );

        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            const float massScale =
                0.75f + 0.25f * float(environment);
            for (std::size_t body = 0u;
                 body < model.world.bodyCount;
                 ++body) {
                bodyParameters[
                    environment * model.world.bodyCount + body
                ].x = massScale;
            }
            controllerParameters[environment].x =
                0.8f + 0.1f * float(environment);
            controllerParameters[environment].y =
                0.9f + 0.05f * float(environment);
        }
        parameterizedInput.bodyParameters = bodyParameters;
        parameterizedInput.controllerParameters =
            controllerParameters;
        parameterizedInput.implicitDrives = true;
        metalrobo::MetalMultiArticulatedInverseMassResult
            effectiveResponse;
        const auto effectiveDiagnostics =
            metalrobo::runMetalMultiArticulatedInverseMass(
                model,
                parameterizedInput,
                effectiveResponse
            );
        require(
            effectiveDiagnostics.succeeded() &&
                effectiveResponse.output != result.output,
            "effective randomized operator was not applied"
        );
        float effectiveReferenceError = 0.0f;
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            metalrobo::EngineModel effectiveModel = model;
            const float massScale =
                bodyParameters[
                    environment * model.world.bodyCount
                ].x;
            for (MRBodyPropertiesGPU& body : effectiveModel.bodies) {
                body.massAndInverseMass.x *= massScale;
                body.massAndInverseMass.y /= massScale;
                body.inertiaRow0.x *= massScale;
                body.inertiaRow0.y *= massScale;
                body.inertiaRow0.z *= massScale;
                body.inertiaRow1.x *= massScale;
                body.inertiaRow1.y *= massScale;
                body.inertiaRow1.z *= massScale;
                body.inertiaRow2.x *= massScale;
                body.inertiaRow2.y *= massScale;
                body.inertiaRow2.z *= massScale;
                body.inverseInertiaRow0.x /= massScale;
                body.inverseInertiaRow0.y /= massScale;
                body.inverseInertiaRow0.z /= massScale;
                body.inverseInertiaRow1.x /= massScale;
                body.inverseInertiaRow1.y /= massScale;
                body.inverseInertiaRow1.z /= massScale;
                body.inverseInertiaRow2.x /= massScale;
                body.inverseInertiaRow2.y /= massScale;
                body.inverseInertiaRow2.z /= massScale;
            }
            const float timestep =
                effectiveModel.world.gravityAndTimestep.w;
            for (MRDofPropertiesGPU& dof : effectiveModel.dofs) {
                if ((dof.flags & MR_DOF_FLAG_DRIVE) != 0u) {
                    dof.drive.z +=
                        timestep *
                            controllerParameters[environment].y *
                            dof.drive.y +
                        timestep * timestep *
                            controllerParameters[environment].x *
                            dof.drive.x;
                }
            }
            const metalrobo::MetalMultiArticulatedInverseMassInput
                effectiveReferenceInput{
                    .environmentCount = 1u,
                    .rhsCount = rhsCount,
                    .q = std::span{
                        q.data() + environment * model.world.nq,
                        model.world.nq,
                    },
                    .rightHandSides = std::span{
                        rhs.data() + environment * rhsCount *
                            model.world.nv,
                        rhsCount * model.world.nv,
                    },
                };
            metalrobo::MetalMultiArticulatedInverseMassResult
                effectiveReference;
            const auto effectiveReferenceDiagnostics =
                metalrobo::runMetalMultiArticulatedInverseMass(
                    effectiveModel,
                    effectiveReferenceInput,
                    effectiveReference
                );
            require(
                effectiveReferenceDiagnostics.succeeded(),
                "materialized effective-operator reference failed"
            );
            effectiveReferenceError = std::max(
                effectiveReferenceError,
                maximumDifference(
                    std::span{
                        effectiveResponse.output.data() +
                            environment * rhsCount * model.world.nv,
                        rhsCount * model.world.nv,
                    },
                    effectiveReference.output
                )
            );
        }
        require(
            effectiveReferenceError <= 2.0e-5f,
            "parameterized inverse mass diverged from materialized "
            "effective operator"
        );
        metalrobo::MetalMultiArticulatedInverseMassResult
            effectiveReplay;
        const auto effectiveReplayDiagnostics =
            metalrobo::runMetalMultiArticulatedInverseMass(
                model,
                parameterizedInput,
                effectiveReplay
            );
        require(
            effectiveReplayDiagnostics.succeeded() &&
                effectiveReplay.output == effectiveResponse.output,
            "effective randomized operator is not deterministic"
        );

        metalrobo::MetalMultiArticulatedInverseMassResult replay;
        const auto replayDiagnostics =
            metalrobo::runMetalMultiArticulatedInverseMass(
                model,
                input,
                replay
            );
        require(
            replayDiagnostics.succeeded() &&
                replay.output == result.output,
            "multi-articulation inverse mass is not deterministic"
        );

        metalrobo::MetalMultiArticulatedInverseMassResult sentinel =
            result;
        const std::size_t sentinelSize = sentinel.output.size();
        const float sentinelValue = sentinel.output.front();
        std::vector<float> invalidRhs = rhs;
        invalidRhs.back() =
            std::numeric_limits<float>::quiet_NaN();
        auto invalidInput = input;
        invalidInput.rightHandSides = invalidRhs;
        const auto rejected =
            metalrobo::runMetalMultiArticulatedInverseMass(
                model,
                invalidInput,
                sentinel
            );
        require(
            !rejected.succeeded() &&
                !rejected.dispatched &&
                sentinel.output.size() == sentinelSize &&
                sentinel.output.front() == sentinelValue,
            "inverse-mass rejection was not transactional"
        );

        std::cout
            << "multi_articulated_inverse_mass=ok"
            << " device=\"" << diagnostics.deviceName << "\""
            << " articulations=" << model.articulations.size()
            << " environments=" << environmentCount
            << " rhs=" << rhsCount
            << " grid_packets="
            << model.articulations.size() * environmentCount
            << " maximum_error=" << maximumError
            << " parameterized_identity=yes"
            << " implicit_drives=yes"
            << " effective_reference_error="
            << effectiveReferenceError
            << " elapsed_ms=" << diagnostics.elapsedMilliseconds
            << " deterministic=yes transactional=yes\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "multi_articulated_inverse_mass=failed reason="
            << exception.what() << '\n';
        return 1;
    }
}

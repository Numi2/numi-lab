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

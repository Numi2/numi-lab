#include "metalrobo/MetalArticulatedABA.hpp"
#include "metalrobo/MetalMultiArticulatedABA.hpp"
#include "metalrobo/SurgicalWorld.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
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
        constexpr std::size_t environmentCount = 3u;
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
        std::vector<float> v(
            environmentCount * model.world.nv,
            0.0f
        );
        std::vector<float> effort(
            environmentCount * model.world.nv,
            0.0f
        );
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            std::copy(
                model.defaultQ.begin(),
                model.defaultQ.end(),
                q.begin() + environment * model.world.nq
            );
            for (std::size_t localV = 0u;
                 localV < model.world.nv;
                 ++localV) {
                v[environment * model.world.nv + localV] =
                    0.0025f *
                    std::sin(
                        float(1u + environment + localV)
                    );
                effort[environment * model.world.nv + localV] =
                    0.01f *
                    std::cos(
                        float(3u + 2u * environment + localV)
                    );
            }
        }

        metalrobo::MetalMultiArticulatedABAInput input{
            .environmentCount = environmentCount,
            .q = q,
            .v = v,
            .effort = effort,
        };
        metalrobo::MetalMultiArticulatedABAResult result;
        const auto diagnostics =
            metalrobo::runMetalMultiArticulatedABA(
                model,
                input,
                result
            );
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                std::string{"multi-articulated Metal ABA failed: "} +
                diagnostics.message
            );
        }
        require(
            diagnostics.dispatched &&
                diagnostics.published &&
                result.statuses.size() ==
                    environmentCount *
                    model.articulations.size(),
            "multi-articulated result was not published"
        );

        float accelerationError = 0.0f;
        float velocityError = 0.0f;
        float configurationError = 0.0f;
        for (std::size_t articulationIndex = 0u;
             articulationIndex < model.articulations.size();
             ++articulationIndex) {
            const MRArticulationGPU& articulation =
                model.articulations[articulationIndex];
            std::vector<float> localQ(
                environmentCount * articulation.nq
            );
            std::vector<float> localV(
                environmentCount * articulation.nv
            );
            std::vector<float> localEffort(
                environmentCount * articulation.nv
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
                std::copy_n(
                    v.begin() +
                        environment * model.world.nv +
                        articulation.vOffset,
                    articulation.nv,
                    localV.begin() +
                        environment * articulation.nv
                );
                std::copy_n(
                    effort.begin() +
                        environment * model.world.nv +
                        articulation.vOffset,
                    articulation.nv,
                    localEffort.begin() +
                        environment * articulation.nv
                );
            }
            metalrobo::MetalArticulatedABAResult reference;
            const metalrobo::MetalArticulatedABAInput referenceInput{
                .articulationIndex =
                    static_cast<std::uint32_t>(articulationIndex),
                .environmentCount = environmentCount,
                .q = localQ,
                .v = localV,
                .effort = localEffort,
            };
            const auto referenceDiagnostics =
                metalrobo::runMetalArticulatedABA(
                    model,
                    referenceInput,
                    reference
                );
            require(
                referenceDiagnostics.succeeded(),
                "single-articulation reference failed"
            );
            for (std::size_t environment = 0u;
                 environment < environmentCount;
                 ++environment) {
                accelerationError = std::max(
                    accelerationError,
                    maximumDifference(
                        std::span{
                            result.acceleration.data() +
                                environment * model.world.nv +
                                articulation.vOffset,
                            articulation.nv,
                        },
                        std::span{
                            reference.acceleration.data() +
                                environment * articulation.nv,
                            articulation.nv,
                        }
                    )
                );
                velocityError = std::max(
                    velocityError,
                    maximumDifference(
                        std::span{
                            result.nextV.data() +
                                environment * model.world.nv +
                                articulation.vOffset,
                            articulation.nv,
                        },
                        std::span{
                            reference.nextV.data() +
                                environment * articulation.nv,
                            articulation.nv,
                        }
                    )
                );
                configurationError = std::max(
                    configurationError,
                    maximumDifference(
                        std::span{
                            result.nextQ.data() +
                                environment * model.world.nq +
                                articulation.qOffset,
                            articulation.nq,
                        },
                        std::span{
                            reference.nextQ.data() +
                                environment * articulation.nq,
                            articulation.nq,
                        }
                    )
                );
            }
        }
        if (accelerationError != 0.0f ||
            velocityError != 0.0f ||
            configurationError != 0.0f) {
            std::cerr
                << "multi_articulated_aba_mismatch"
                << " acceleration=" << accelerationError
                << " velocity=" << velocityError
                << " configuration=" << configurationError
                << '\n';
        }
        require(
            accelerationError == 0.0f &&
                velocityError == 0.0f &&
                configurationError == 0.0f,
            "2-D multi-articulation grid diverged from ABA reference"
        );

        metalrobo::MetalMultiArticulatedABAResult sentinel = result;
        const std::size_t sentinelSize = sentinel.nextQ.size();
        const float sentinelValue = sentinel.nextQ.front();
        std::vector<float> invalidQ = q;
        invalidQ.front() =
            std::numeric_limits<float>::quiet_NaN();
        input.q = invalidQ;
        const auto rejected =
            metalrobo::runMetalMultiArticulatedABA(
                model,
                input,
                sentinel
            );
        require(
            !rejected.succeeded() &&
                !rejected.dispatched &&
                sentinel.nextQ.size() == sentinelSize &&
                sentinel.nextQ.front() == sentinelValue,
            "multi-articulated rejection was not transactional"
        );

        std::cout
            << "multi_articulated_aba=ok"
            << " device=\"" << diagnostics.deviceName << "\""
            << " articulations=" << model.articulations.size()
            << " environments=" << environmentCount
            << " grid_packets="
            << model.articulations.size() * environmentCount
            << " acceleration_error=" << accelerationError
            << " velocity_error=" << velocityError
            << " configuration_error=" << configurationError
            << " elapsed_ms=" << diagnostics.elapsedMilliseconds
            << " transactional=yes\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "multi_articulated_aba=failed reason="
            << exception.what() << '\n';
        return 1;
    }
}

#include "metalrobo/Franka.hpp"
#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/MetalWorldFamily.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    try {
        metalrobo::WorldTemplate worldTemplate;
        const metalrobo::WorldCompileResult twinResult =
            metalrobo::compileEpisodeTwin(
                metalrobo::makeFrankaPickPlaceEpisodeTwin(),
                metalrobo::makeFrankaPickPlaceEngineModel(),
                worldTemplate
            );
        if (!twinResult.succeeded()) {
            throw std::runtime_error(twinResult.message);
        }

        metalrobo::WorldFamily family;
        const metalrobo::WorldCompileResult familyResult =
            metalrobo::compileWorldFamily(
                worldTemplate,
                metalrobo::makeFrankaPickPlaceWorldProgram(),
                family
            );
        if (!familyResult.succeeded()) {
            throw std::runtime_error(familyResult.message);
        }

        constexpr std::uint32_t kWorldCount = 4096u;
        metalrobo::MetalWorldFamilyContext context;
        const auto compile = context.compile(family, kWorldCount);
        if (!compile.succeeded()) {
            throw std::runtime_error(
                std::string{
                    metalrobo::metalWorldFamilyStatusName(
                        compile.status
                    )
                } + ": " + compile.message
            );
        }
        const auto sample = context.sample(
            kWorldCount,
            0x123456789abcdef0ull
        );
        if (!sample.succeeded()) {
            throw std::runtime_error(
                std::string{
                    metalrobo::metalWorldFamilyStatusName(
                        sample.status
                    )
                } + ": " + sample.message
            );
        }
        for (const auto buffer : {
                 metalrobo::MetalWorldFamilyBuffer::instanceHeaders,
                 metalrobo::MetalWorldFamilyBuffer::assetInstances,
                 metalrobo::MetalWorldFamilyBuffer::sensorInstances,
                 metalrobo::MetalWorldFamilyBuffer::appearanceInstances,
                 metalrobo::MetalWorldFamilyBuffer::assetBindings,
                 metalrobo::MetalWorldFamilyBuffer::bindingIndices,
                 metalrobo::MetalWorldFamilyBuffer::resetQ,
                 metalrobo::MetalWorldFamilyBuffer::resetV,
                 metalrobo::MetalWorldFamilyBuffer::resetSceneBodies,
                 metalrobo::MetalWorldFamilyBuffer::bodyParameters,
                 metalrobo::MetalWorldFamilyBuffer::controllerParameters,
                 metalrobo::MetalWorldFamilyBuffer::scenarioHeaders,
                 metalrobo::MetalWorldFamilyBuffer::scenarioValues,
             }) {
            if (context.nativeBuffer(buffer) == nullptr) {
                throw std::runtime_error(
                    "persistent private output buffer is unavailable"
                );
            }
        }

        metalrobo::WorldInstanceBatch batch;
        const auto readback = context.readback(batch);
        if (!readback.succeeded()) {
            throw std::runtime_error(
                std::string{
                    metalrobo::metalWorldFamilyStatusName(
                        readback.status
                    )
                } + ": " + readback.message
            );
        }
        const std::uint32_t objectIndex =
            worldTemplate.assetIndex("pick_object");
        const float firstObjectX =
            batch.assets[objectIndex].positionAndScale.x;
        const float lastObjectX =
            batch.assets[
                (kWorldCount - 1u) * worldTemplate.assets.size() +
                objectIndex
            ].positionAndScale.x;
        if (std::abs(firstObjectX - lastObjectX) < 1.0e-6f) {
            throw std::runtime_error(
                "GPU object configurations collapsed"
            );
        }
        if (batch.scenarioHeaders.size() != kWorldCount ||
            batch.scenarioValues.size() !=
                kWorldCount * family.program.variations.size() ||
            batch.scenarioHeaders.front().identity.x !=
                batch.instances.front().identity.x ||
            batch.scenarioHeaders.front().identity.y !=
                batch.instances.front().identity.y) {
            throw std::runtime_error(
                "GPU scenario provenance is incomplete"
            );
        }
        metalrobo::MetalWorldFamilyPhysicsBatch physics;
        const auto physicsReadback =
            context.readbackPhysics(physics);
        if (!physicsReadback.succeeded()) {
            throw std::runtime_error(
                std::string{
                    metalrobo::metalWorldFamilyStatusName(
                        physicsReadback.status
                    )
                } + ": " + physicsReadback.message
            );
        }
        if (physics.primaryArticulationIndex != 0u ||
            physics.nq != 9u || physics.nv != 9u ||
            physics.bodyCount != 15u ||
            physics.sceneBodyCount != 4u ||
            physics.articulationCount != 1u ||
            physics.resetSceneBodies.size() !=
                kWorldCount * physics.sceneBodyCount ||
            std::abs(
                physics.resetSceneBodies.front().position.x -
                firstObjectX
            ) > 1.0e-6f ||
            std::abs(
                physics.resetSceneBodies[
                    (kWorldCount - 1u) *
                    physics.sceneBodyCount
                ].position.x -
                lastObjectX
            ) > 1.0e-6f) {
            throw std::runtime_error(
                "GPU physics resets do not match sampled assets"
            );
        }
        const auto& firstObjectParameters =
            physics.bodyParameters[11u];
        const auto& lastObjectParameters =
            physics.bodyParameters[
                (kWorldCount - 1u) * physics.bodyCount + 11u
            ];
        if (firstObjectParameters.identity.x != objectIndex ||
            std::abs(
                firstObjectParameters.physical.y -
                batch.assets[objectIndex].physical.y
            ) > 1.0e-6f ||
            std::abs(
                firstObjectParameters.physical.x -
                lastObjectParameters.physical.x
            ) < 1.0e-6f ||
            !(physics.resetSceneBodies.front()
                  .linearVelocityAndInverseMass.w > 0.0f) ||
            std::abs(
                physics.resetSceneBodies.front()
                    .linearVelocityAndInverseMass.w -
                physics.resetSceneBodies[
                    (kWorldCount - 1u) *
                    physics.sceneBodyCount
                ].linearVelocityAndInverseMass.w
            ) < 1.0e-6f) {
            throw std::runtime_error(
                "GPU mass/inertia parameters are not causal in resets"
            );
        }
        const std::uint32_t robotIndex =
            worldTemplate.assetIndex("franka");
        if (physics.controllerParameters.front().identity.x !=
                robotIndex ||
            std::abs(
                physics.controllerParameters.front().controller.z -
                batch.assets[robotIndex].controller.z
            ) > 1.0e-6f) {
            throw std::runtime_error(
                "GPU controller parameters do not match robot state"
            );
        }

        const metalrobo::MetalWorldFamilyStats stats =
            context.stats();
        std::cout
            << "device=\"" << sample.deviceName << "\""
            << " worlds=" << batch.instances.size()
            << " variations=" << family.program.variations.size()
            << " resident_private_bytes="
            << stats.retainedPrivateBytes
            << " sample_ms=" << sample.elapsedMilliseconds
            << " explicit_readback_ms="
            << readback.elapsedMilliseconds
            << " physics_resets=yes"
            << " gpu_resident=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_metal_world_family_probe: "
                  << error.what() << '\n';
        return 1;
    }
}

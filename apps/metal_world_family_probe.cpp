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
                metalrobo::makeFrankaPandaEngineModel(),
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
            << " gpu_resident=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_metal_world_family_probe: "
                  << error.what() << '\n';
        return 1;
    }
}

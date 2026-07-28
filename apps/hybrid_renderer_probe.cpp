#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/MetalHybridRenderer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

template <typename Result>
void require(const Result& result, const char* operation) {
    if (!result.succeeded()) {
        throw std::runtime_error(std::string{operation} + ": " +
                                 result.message);
    }
}

} // namespace

int main() {
    try {
        metalrobo::WorldTemplate worldTemplate;
        require(metalrobo::compileEpisodeTwin(
                    metalrobo::makeFrankaPickPlaceEpisodeTwin(),
                    metalrobo::makeFrankaPickPlaceEngineModel(), worldTemplate),
                "episode compile");
        metalrobo::WorldFamily family;
        require(metalrobo::compileWorldFamily(
                    worldTemplate, metalrobo::makeFrankaPickPlaceWorldProgram(),
                    family),
                "family compile");
        metalrobo::MetalWorldFamilyContext worlds;
        require(worlds.compile(family, 4u), "world compile");
        require(worlds.sample(4u, 29u), "world sample");

        metalrobo::HybridGaussianScene scene;
        scene.id = "camera_aligned_probe";
        scene.assetCount =
            static_cast<std::uint32_t>(worldTemplate.assets.size());
        scene.gaussians.push_back({
            {0.8f, -0.6f, 1.3f, 0.98f},
            {0.06f, 0.02f, 0.04f, 1.0f},
            {0.0f, 0.0f, 0.0f, 1.0f},
            {1.0f, 0.1f, 0.05f, 0.0f},
            {
                0u,
                MR_INVALID_INDEX,
                42u,
                MR_HYBRID_GAUSSIAN_ASSET_LOCAL,
            },
        });
        metalrobo::MetalHybridRenderer renderer;
        require(renderer.compile(scene, 4u), "renderer compile");
        const auto rendered = renderer.render(worlds, 4u, 0u);
        require(rendered, "render");
        metalrobo::HybridObservationBatch observations;
        require(renderer.readback(observations), "readback");

        const std::size_t center = 60u * observations.width + 80u;
        const float depth = observations.depth[center];
        const std::uint32_t semantic = observations.segmentation[center];
        if (!std::isfinite(depth) || std::abs(depth - 0.5f) > 0.05f ||
            semantic != 42u ||
            observations.rgb[center].x <= observations.rgb[center].y) {
            throw std::runtime_error(
                "center pixel did not preserve Gaussian depth, "
                "semantic identity, and color");
        }

        std::cout << "device=\"" << rendered.deviceName << "\""
                  << " environments=" << observations.environmentCount
                  << " resolution=" << observations.width << 'x'
                  << observations.height << " center_depth=" << depth
                  << " center_semantic=" << semantic
                  << " render_ms=" << rendered.elapsedMilliseconds
                  << " private_bytes=" << renderer.layout().retainedPrivateBytes
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_hybrid_renderer_probe: " << error.what()
                  << '\n';
        return 1;
    }
}

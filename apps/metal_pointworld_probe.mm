#include "metalrobo/MetalPointWorld.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::uint64_t fingerprint(const void* data, const std::size_t bytes, std::uint64_t hash) {
    const auto* values = static_cast<const std::uint8_t*>(data);
    for (std::size_t index = 0u; index < bytes; ++index) {
        hash ^= values[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

} // namespace

int main() {
    constexpr std::uint32_t width = 320u;
    constexpr std::uint32_t height = 180u;
    constexpr std::size_t pixels = static_cast<std::size_t>(width) * height;
    std::vector<std::uint8_t> rgba(pixels * 4u, 255u);
    std::vector<float> depth(pixels, 1.0f);
    std::vector<std::uint8_t> validity(pixels, 1u);
    validity[0] = 0u;
    metalrobo::MetalPointWorldPreprocessor context;
    metalrobo::PointWorldPreprocessResult output;
    const auto diagnostics = context.run({
        .rgba = rgba, .metricDepth = depth, .depthValidity = validity,
        .intrinsics = {200.0f, 200.0f, 160.0f, 90.0f},
        .width = width, .height = height,
    }, output);
    if (!diagnostics.succeeded()) {
        throw std::runtime_error(diagnostics.message);
    }
    const auto& center = output.cameraPoints[90u * width + 160u];
    const auto& invalid = output.cameraPoints.front();
    const auto& rgb = output.normalizedRGBA.front();
    if (std::abs(center[0]) > 1.0e-6f || std::abs(center[1]) > 1.0e-6f ||
        std::abs(center[2] - 1.0f) > 1.0e-6f || center[3] != 1.0f ||
        std::abs(rgb[0] - ((1.0f - 0.485f) / 0.229f)) > 1.0e-5f ||
        invalid[0] != 0.0f || invalid[1] != 0.0f || invalid[2] != 0.0f || invalid[3] != 0.0f) {
        throw std::runtime_error("PointWorld Metal preprocessing result differs from the release contract");
    }
    std::uint64_t outputFingerprint = 1469598103934665603ull;
    outputFingerprint = fingerprint(
        output.normalizedRGBA.data(), output.normalizedRGBA.size() * sizeof(output.normalizedRGBA.front()), outputFingerprint
    );
    outputFingerprint = fingerprint(
        output.cameraPoints.data(), output.cameraPoints.size() * sizeof(output.cameraPoints.front()), outputFingerprint
    );
    std::cout << "device=" << diagnostics.deviceName
              << " elapsed_ms=" << diagnostics.elapsedMilliseconds
              << " retained_bytes=" << diagnostics.retainedBytes
              << " center_z=" << center[2]
              << " fingerprint=" << outputFingerprint << '\n';
    return 0;
}

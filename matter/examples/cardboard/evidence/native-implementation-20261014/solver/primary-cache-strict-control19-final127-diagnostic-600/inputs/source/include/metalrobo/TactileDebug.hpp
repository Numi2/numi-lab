#pragma once

#include "metalrobo/Tactile.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace metalrobo {

struct TactileDebugExportConfig {
    std::filesystem::path directory;
    std::string prefix = "tactile";
    std::uint32_t environment = 0u;
    std::uint32_t sensor = 0u;
    // Visual-only scale for the net-force line in the OBJ output.
    float forceVectorMetersPerNewton = 0.01f;
};

struct TactileDebugExportResult {
    std::filesystem::path metricDepthCSV;
    std::filesystem::path depthPreviewPGM;
    std::filesystem::path validityPreviewPGM;
    std::filesystem::path geometryOBJ;
    std::filesystem::path summaryJSON;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return message == "ok";
    }
};

// Explicit inspection path. It writes metric values, atlas previews, surface
// samples/normals/sensing segments/hits, centroid, CoP, and net force. The
// normal headless observation path never calls this function.
[[nodiscard]] TactileDebugExportResult exportTactileDebugFrame(
    const CookedTactileSystem& tactile,
    const TactileObservationBatch& observation,
    const TactileDebugExportConfig& config
);

} // namespace metalrobo

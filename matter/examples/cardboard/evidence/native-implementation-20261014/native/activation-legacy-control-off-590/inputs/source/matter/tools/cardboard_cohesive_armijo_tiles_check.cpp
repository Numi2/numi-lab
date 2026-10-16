#include "cardboard_cohesive_armijo_tiles.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
unsigned checks = 0u;

void require(const bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

using namespace numi::matter::cardboard::cohesive_tiles;

void checkLayout(const std::uint32_t environments,
                 const std::uint32_t faces,
                 const std::uint32_t requestedPage,
                 const std::uint64_t deviceLimit,
                 const bool expected) {
    Layout layout;
    const bool admitted = makeLayout(
        environments, faces, requestedPage, deviceLimit, layout);
    require(admitted == expected, "cohesive tile layout admission mismatch");
    if (!admitted) return;
    require(layout.environmentCount == environments &&
            layout.totalFaceCount == faces &&
            !layout.pages.empty(),
        "cohesive tile layout metadata mismatch");
    std::uint64_t cursor = 0u;
    for (const Page page : layout.pages) {
        require(page.firstFace == cursor && page.faceCount != 0u &&
                page.dispatchThreads == environments * page.faceCount,
            "cohesive tile pages are not contiguous or dispatch-bounded");
        cursor += page.faceCount;
    }
    require(cursor == faces && layout.pageBytes <= kPrivatePageByteCap &&
            layout.pageBytes <= deviceLimit,
        "cohesive tile plan truncated faces or exceeded a byte bound");
    require(layout.legacyPrivateBytes ==
                std::uint64_t(environments) * faces *
                    (sizeof(NMCohesiveFaceGPU) +
                     sizeof(NMCohesiveFaceVectorGPU)),
        "legacy cohesive face+force scratch accounting mismatch");
}

void checkOrderedGather() {
    std::vector<FaceForces> faces(3u);
    // Deliberate FP32 cancellation makes face-order changes observable.
    faces[0][0] = {1.0e20f, 0.25f, -2.0f};
    faces[1][1] = {-1.0e20f, 1.0e20f, 2.0f};
    faces[2][3] = {3.25f, -1.0e20f, -7.0f};
    faces[2][4] = {0.5f, 0.75f, 1.25f};
    // The duplicate face/slot entries model repeated incidence contributions;
    // neither path may canonicalize or discard them.
    const std::vector<std::vector<IncidenceEntry>> incidence{
        {{0u, 0u}, {1u, 1u}, {2u, 3u}, {2u, 4u}, {2u, 4u}},
        {{0u, 0u}, {2u, 3u}, {2u, 3u}},
    };
    const auto expected = legacyGather(faces, incidence);
    require(expected[0][0] == 4.25f,
        "legacy cancellation witness did not exercise ordered FP32 arithmetic");
    for (const std::uint32_t pageSize : {1u, 2u, 3u}) {
        Layout layout;
        require(makeLayout(1u, 3u, pageSize, kPrivatePageByteCap, layout),
            "small multipage layout was not admitted");
        const auto actual = tiledGather(faces, incidence, layout);
        require(bitIdentical(expected, actual),
            "ordered tiled incidence gather changed FP32 accumulation bytes");
    }
    const std::vector<std::vector<IncidenceEntry>> unsorted{
        {{2u, 3u}, {0u, 0u}},
    };
    Layout unsortedLayout;
    require(makeLayout(1u, 3u, 1u, kPrivatePageByteCap, unsortedLayout),
        "unsorted-incidence negative control layout failed");
    bool rejectedUnsorted = false;
    try {
        (void)tiledGather(faces, unsorted, unsortedLayout);
    } catch (const std::runtime_error&) {
        rejectedUnsorted = true;
    }
    require(rejectedUnsorted,
        "tiled page gather accepted noncanonical face/slot order");
}

void run() {
    static_assert(sizeof(NMCohesiveFaceGPU) == 384u);
    static_assert(sizeof(NMCohesiveFaceVectorGPU) == 96u);
    checkLayout(1u, 1u, 0u, kPrivatePageByteCap, true);
    checkLayout(2u, 3u, 1u, kPrivatePageByteCap, true);
    checkLayout(4u, 7u, 1u, kPrivatePageByteCap, true);
    checkLayout(1u, 260835u, 0u, kPrivatePageByteCap, true);
    checkLayout(4u, 260835u, 0u, kPrivatePageByteCap, true);
    checkLayout(1u, 260835u, 1u,
                sizeof(NMCohesiveFaceVectorGPU) - 1u, false);
    checkLayout(1u, 0u, 0u, kPrivatePageByteCap, false);
    checkLayout(5u, 1u, 0u, kPrivatePageByteCap, false);
    checkLayout(1u, 260835u, 1u,
                std::uint64_t(sizeof(NMCohesiveFaceVectorGPU)) - 1u, false);
    checkLayout(1u, std::numeric_limits<std::uint32_t>::max() / 6u + 1u,
                1u, kPrivatePageByteCap, false);
    checkLayout(4u, 260835u,
                static_cast<std::uint32_t>(
                    kPrivatePageByteCap /
                    (4u * sizeof(NMCohesiveFaceVectorGPU)) + 1u),
                kPrivatePageByteCap, false);
    checkOrderedGather();
}

} // namespace

int main() {
    try {
        run();
        std::cout << "PASS cohesive private page layout and ordered gather checks="
                  << checks
                  << " face_bytes=" << sizeof(NMCohesiveFaceGPU)
                  << " face_force_bytes=" << sizeof(NMCohesiveFaceVectorGPU)
                  << " legacy_private_bytes_per_face="
                  << sizeof(NMCohesiveFaceGPU) + sizeof(NMCohesiveFaceVectorGPU)
                  << " physical_qualification=false\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cohesive private tile check failed: " << error.what() << '\n';
        return 1;
    }
}

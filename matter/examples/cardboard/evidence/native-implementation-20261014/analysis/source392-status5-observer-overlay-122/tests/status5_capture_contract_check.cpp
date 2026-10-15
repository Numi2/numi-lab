#include "numi/matter/shared.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

static_assert(sizeof(NMMatterStatusGPU) == 48);
static_assert(sizeof(NMFEMNodeStateGPU) == 80);
static_assert(sizeof(NMFEMStatus5CaptureGPU) % 16 == 0);
static_assert(sizeof(((NMFEMStatus5CaptureGPU*)nullptr)->acceptedNodes) == 4u * 80u);
static_assert(sizeof(((NMFEMStatus5CaptureGPU*)nullptr)->candidateNodes) == 4u * 80u);
static_assert(sizeof(((NMFEMStatus5CaptureGPU*)nullptr)->acceptedInternalState) == 32u * sizeof(float));
static_assert(sizeof(((NMFEMStatus5CaptureGPU*)nullptr)->parameterWords) == 512u * sizeof(float));
static_assert(offsetof(NMFEMStatus5CaptureGPU, status) % 16u == 0u);
static_assert(offsetof(NMFEMStatus5CaptureGPU, deformationHigh) % 16u == 0u);

constexpr std::uint32_t captureSlot(const std::uint32_t environment,
    const std::uint32_t domain) noexcept {
    return environment * 2u + domain;
}

int main() {
    bool seen[16]{};
    for (std::uint32_t environment = 0; environment < 8; ++environment) {
        for (std::uint32_t domain = 0; domain < 2; ++domain) {
            const auto slot = captureSlot(environment, domain);
            assert(slot < 16u && !seen[slot]);
            seen[slot] = true;
        }
    }
    for (const bool value : seen) assert(value);
    assert(NM_STATUS5_INDEX_NONE == 0u);
    assert(NM_STATUS5_INDEX_NODE == 1u);
    assert(NM_STATUS5_INDEX_TETRAHEDRON == 2u);
    assert(NM_STATUS5_INDEX_SOLVER_ITERATION == 3u);
    assert(NM_STATUS5_CAPTURE_DOMAIN_GLOBAL != NM_STATUS5_CAPTURE_DOMAIN_PRIVATE);
    assert(NM_STATUS5_CAPTURE_MAX_PARAMETER_WORDS == 512u);
    return 0;
}

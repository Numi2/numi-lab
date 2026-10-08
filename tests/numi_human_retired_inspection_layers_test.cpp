#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include "NumiHumanRestingInspectionLayers.hpp"

struct TestVisualInstance {
    std::array<std::uint32_t, 4> identity;
    std::uint32_t inspectionMask;
};

int main() {
    constexpr std::uint32_t boneSemantic = 51004u;
    constexpr std::uint32_t muscleSemantic = 51005u;
    constexpr std::uint32_t organSemantic = 51010u;
    constexpr std::uint32_t organInspectionBit = 1u << 3u;
    constexpr std::uint32_t boneInspectionMask = 6u;
    constexpr std::uint32_t muscleInspectionMask = 2u;
    const std::set<unsigned> receiptValidatedRetiredIDs{22u};
    const auto layerSelected = [](std::uint32_t mask, unsigned layer) {
        return layer < 7u && (mask & (1u << layer)) != 0u;
    };

    for (unsigned activePatch = 14u; activePatch <= 21u; ++activePatch) {
        const auto mask = numiHumanRestingInspectionLayerMask(
            organSemantic, activePatch, organInspectionBit, organSemantic,
            receiptValidatedRetiredIDs);
        if (mask != organInspectionBit) {
            std::cerr << "active liver patch lost its organ inspection mask: " << activePatch << "\n";
            return 1;
        }
        for (unsigned layer = 0; layer < 7u; ++layer) {
            if (layerSelected(mask, layer) != (layer == 3u)) {
                std::cerr << "active patch visibility differs at layer " << layer << "\n";
                return 1;
            }
        }
    }

    const TestVisualInstance retiredOrgan{{organSemantic, 22u, 20u, 22u}, organInspectionBit};
    const auto retiredMask = numiHumanRestingInspectionLayerMask(
        retiredOrgan.identity[0], retiredOrgan.identity[3], retiredOrgan.inspectionMask,
        organSemantic, receiptValidatedRetiredIDs);
    if (retiredMask != 0u) {
        std::cerr << "receipt-retired organ alias retained a selectable inspection mask\n";
        return 1;
    }
    for (unsigned layer = 0; layer < 7u; ++layer) {
        if (layerSelected(retiredMask, layer)) {
            std::cerr << "retired organ alias is visible in inspection layer " << layer << "\n";
            return 1;
        }
    }

    const TestVisualInstance boneCollision{{boneSemantic, 22u, 137u, 22u}, boneInspectionMask};
    const TestVisualInstance muscleCollision{{muscleSemantic, 22u,
        std::numeric_limits<std::uint32_t>::max(), 22u}, muscleInspectionMask};
    if (boneCollision.identity != std::array<std::uint32_t, 4>{51004u, 22u, 137u, 22u} ||
        muscleCollision.identity != std::array<std::uint32_t, 4>{
            51005u, 22u, std::numeric_limits<std::uint32_t>::max(), 22u}) {
        std::cerr << "semantic-collision regression fixtures changed identity tuples\n";
        return 1;
    }
    for (const TestVisualInstance* collision : {&boneCollision, &muscleCollision}) {
        const auto mask = numiHumanRestingInspectionLayerMask(
            collision->identity[0], collision->identity[3], collision->inspectionMask,
            organSemantic, receiptValidatedRetiredIDs);
        if (mask != collision->inspectionMask) {
            std::cerr << "non-organ semantic collision was hidden for semantic "
                      << collision->identity[0] << "\n";
            return 1;
        }
        for (unsigned layer = 0; layer < 7u; ++layer) {
            const bool expected = (collision->inspectionMask & (1u << layer)) != 0u;
            if (layerSelected(mask, layer) != expected) {
                std::cerr << "non-organ semantic collision changed selection at layer " << layer << "\n";
                return 1;
            }
        }
    }

    const std::set<unsigned> noRetirementReceipt;
    if (numiHumanRestingInspectionLayerMask(
            organSemantic, 22u, organInspectionBit, organSemantic, noRetirementReceipt) != organInspectionBit) {
        std::cerr << "unregistered organ identity was hidden without receipt retirement\n";
        return 1;
    }
    return 0;
}

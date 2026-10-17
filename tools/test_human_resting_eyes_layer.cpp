#include "NumiHumanRestingInspectionLayers.hpp"
#include "NumiHumanRestingOcularWindow.hpp"

#include <cstdlib>
#include <iostream>
#include <set>

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << "\n";
        std::exit(1);
}
}

}

int main() {
    constexpr std::uint32_t organSemantic = 100u;
    metalrobo::VisualAssetPackV2 emptyPack;
    const auto absent = numi_human_resting_ocular_window::inspectCanonicalExteriorSet(
        emptyPack, organSemantic);
    require(absent.state == numi_human_resting_ocular_window::CanonicalExteriorSetState::absent &&
            absent.presentSurfaceCount == 0u && absent.expectedSurfaceCount == 6u,
        "older scenes with no canonical eye exterior must leave the feature inactive");

    metalrobo::VisualAssetPackV2 completePack;
    completePack.materials.push_back({});
    completePack.indices = {0u, 1u, 2u};
    completePack.vertices.resize(3u);
    constexpr std::array<std::uint32_t, 6> exteriorIDs{{386u, 383u, 384u, 395u, 392u, 393u}};
    for (std::uint32_t index = 0u; index < exteriorIDs.size(); ++index) {
        MRVisualPrimitiveGPUV2 primitive{};
        primitive.geometry = {0u, 3u, 0u, index};
        primitive.identity = {organSemantic, index, 0u, exteriorIDs[index]};
        completePack.primitives.push_back(primitive);
        MRVisualInstanceGPUV2 instance{};
        instance.geometry = {index, 1u, 0u, 0u};
        instance.identity = {organSemantic, index, 0u, exteriorIDs[index]};
        completePack.instances.push_back(instance);
    }
    const auto complete = numi_human_resting_ocular_window::inspectCanonicalExteriorSet(
        completePack, organSemantic);
    require(complete.state == numi_human_resting_ocular_window::CanonicalExteriorSetState::complete &&
            complete.presentSurfaceCount == 6u,
        "the six exact sclera/cornea/iris surfaces must activate both Skin windows");

    auto partialPack = completePack;
    partialPack.primitives.pop_back();
    partialPack.instances.pop_back();
    const auto partial = numi_human_resting_ocular_window::inspectCanonicalExteriorSet(
        partialPack, organSemantic);
    require(partial.state == numi_human_resting_ocular_window::CanonicalExteriorSetState::partial &&
            partial.presentSurfaceCount == 5u,
        "a partially present canonical eye exterior must fail closed");

    auto ambiguousPack = completePack;
    ambiguousPack.primitives.push_back(ambiguousPack.primitives.front());
    const auto ambiguous = numi_human_resting_ocular_window::inspectCanonicalExteriorSet(
        ambiguousPack, organSemantic);
    require(ambiguous.state == numi_human_resting_ocular_window::CanonicalExteriorSetState::ambiguous,
        "duplicate typed eye surfaces must fail closed");

    auto brokenLinkPack = completePack;
    brokenLinkPack.primitives.front().geometry.w = 5u;
    const auto brokenLink = numi_human_resting_ocular_window::inspectCanonicalExteriorSet(
        brokenLinkPack, organSemantic);
    require(brokenLink.state == numi_human_resting_ocular_window::CanonicalExteriorSetState::ambiguous,
        "typed eye surfaces must preserve exact primitive-instance links");
    constexpr std::uint32_t organLayerBit = 1u << 3u;
    const std::set<unsigned> none;

    for (std::uint32_t stableId = kNumiHumanFirstOcularSurfaceStableID;
         stableId <= kNumiHumanLastOcularSurfaceStableID; ++stableId) {
        const auto role = numiHumanRestingOcularAppearance(organSemantic, stableId, organSemantic);
        require(role != NumiHumanRestingOcularAppearance::none,
            "every accepted ocular source ID must have an authenticated presentation role");
        const auto mask = numiHumanRestingInspectionLayerMask(
            organSemantic, stableId, organLayerBit, organSemantic, none);
        const bool exposedInSkin = role == NumiHumanRestingOcularAppearance::iris;
        const auto expected = organLayerBit |
            (exposedInSkin ? kNumiHumanSkinInspectionLayerBit : 0u);
        require(mask == expected,
            "only the source iris joins Skin; lens and other internal source anatomy stay Organ-only");
    }
    require(numiHumanRestingOcularAppearance(organSemantic, 384u, organSemantic) ==
                NumiHumanRestingOcularAppearance::iris &&
            numiHumanRestingOcularAppearance(organSemantic, 393u, organSemantic) ==
                NumiHumanRestingOcularAppearance::iris,
        "right and left iris identities must map to their source role");
    require(numiHumanRestingOcularAppearance(organSemantic, 386u, organSemantic) ==
                NumiHumanRestingOcularAppearance::sclera &&
            numiHumanRestingOcularAppearance(organSemantic, 395u, organSemantic) ==
                NumiHumanRestingOcularAppearance::sclera,
        "right and left sclera identities must map to their source role");
    require(numiHumanRestingOcularAppearance(organSemantic, 398u, organSemantic) ==
                NumiHumanRestingOcularAppearance::none &&
            numiHumanRestingOcularAppearance(organSemantic + 1u, 384u, organSemantic) ==
                NumiHumanRestingOcularAppearance::none,
        "unmapped or non-organ identities must not acquire an ocular appearance role");
    require(numiHumanRestingInspectionLayerMask(organSemantic, 380u, organLayerBit,
            organSemantic, none) == organLayerBit,
        "non-ocular organ immediately below the source range changed layer");
    require(numiHumanRestingInspectionLayerMask(organSemantic, 398u, organLayerBit,
            organSemantic, none) == organLayerBit &&
            numiHumanRestingInspectionLayerMask(organSemantic, 399u, organLayerBit,
                organSemantic, none) == organLayerBit,
        "derived window IDs must not collide with source ocular-role identities");
    require(numiHumanRestingInspectionLayerMask(organSemantic, 384u, organLayerBit,
            organSemantic, none, false) == organLayerBit,
        "older scenes with no eye exterior must not activate ocular Skin presentation");
    require(numiHumanRestingScleraWindowLayerMask() == kNumiHumanSkinInspectionLayerBit &&
            kNumiHumanRightScleraWindowStableID != kNumiHumanLeftScleraWindowStableID,
        "derived sclera windows must use unique Skin-only presentation identities");
    require(numiHumanRestingPosteriorLensBackingLayerMask() == kNumiHumanSkinInspectionLayerBit &&
            kNumiHumanRightPosteriorLensBackingStableID != kNumiHumanLeftPosteriorLensBackingStableID &&
            kNumiHumanRightPosteriorLensBackingStableID != 385u &&
            kNumiHumanLeftPosteriorLensBackingStableID != 394u,
        "derived posterior lens backings must not collide with source identities");
    require(numiHumanRestingInspectionLayerMask(organSemantic + 1u, 381u, organLayerBit,
            organSemantic, none) == organLayerBit,
        "non-organ identity must not be classified as an ocular surface");
    const std::set<unsigned> retiredEye{381u};
    require(numiHumanRestingInspectionLayerMask(organSemantic, 381u, organLayerBit,
            organSemantic, retiredEye) == 0u,
        "retired source identity must remain hidden in every inspection layer");
    std::cout << "6-surface eye readiness is all-or-none; source irises and derived sclera/backing faces are Skin-visible\n";
}

#pragma once
#include <cstdint>
#include <set>

// The accepted visual source assigns stable IDs 381..397 to its 17 ocular
// surfaces. Keep all canonical surfaces in Organ. Only the source iris joins
// Skin directly; source-face sclera windows and a small posterior lens backing
// are presentation-only clones and leave the anatomical source untouched.
inline constexpr std::uint32_t kNumiHumanFirstOcularSurfaceStableID = 381u;
inline constexpr std::uint32_t kNumiHumanLastOcularSurfaceStableID = 397u;
inline constexpr std::uint32_t kNumiHumanSkinInspectionLayerBit = 1u << 0u;

enum class NumiHumanRestingOcularAppearance : std::uint8_t {
    none,
    choroid,
    cornea,
    iris,
    lens,
    sclera,
    suspensoryLigament,
    vitreous,
    anteriorChamber,
};

// Authenticated BodyParts3D source-member crosswalk for the accepted eye surfaces.
// These are presentation roles; they do not create missing structures such as a
// separate pupil mesh.
inline constexpr NumiHumanRestingOcularAppearance numiHumanRestingOcularAppearance(
    std::uint32_t semantic, std::uint32_t stableId, std::uint32_t organSemantic) noexcept {
    if (semantic != organSemantic) return NumiHumanRestingOcularAppearance::none;
    switch (stableId) {
        case 381u: case 382u: case 390u: case 391u:
            return NumiHumanRestingOcularAppearance::choroid;
        case 383u: case 392u:
            return NumiHumanRestingOcularAppearance::cornea;
        case 384u: case 393u:
            return NumiHumanRestingOcularAppearance::iris;
        case 385u: case 394u:
            return NumiHumanRestingOcularAppearance::lens;
        case 386u: case 395u:
            return NumiHumanRestingOcularAppearance::sclera;
        case 387u: case 396u:
            return NumiHumanRestingOcularAppearance::suspensoryLigament;
        case 388u: case 397u:
            return NumiHumanRestingOcularAppearance::vitreous;
        case 389u:
            return NumiHumanRestingOcularAppearance::anteriorChamber;
        default:
            return NumiHumanRestingOcularAppearance::none;
    }
}

// Receipt-validated retired source identities remain in the visual pack for
// provenance, but must never acquire a selectable inspection-layer bit.
inline std::uint32_t numiHumanRestingInspectionLayerMask(
    std::uint32_t semantic,
    std::uint32_t stableId,
    std::uint32_t sourceLayerMask,
    std::uint32_t organSemantic,
    const std::set<unsigned>& retiredInspectionStableIDs,
    bool ocularPresentationEnabled = true) noexcept {
    const bool retiredIdentity = semantic == organSemantic &&
        retiredInspectionStableIDs.count(stableId) != 0u;
    if (retiredIdentity) return 0u;
    if (!ocularPresentationEnabled) return sourceLayerMask;

    const auto ocularAppearance =
        numiHumanRestingOcularAppearance(semantic, stableId, organSemantic);
    // The source cornea remains selectable in the organ layer. It is excluded
    // from Skin inspection because this renderer has no order-independent
    // transparency; omitting this one optical shell lets the authentic iris
    // remain visible without changing geometry or source identity.
    const bool skinVisibleOcular = ocularAppearance == NumiHumanRestingOcularAppearance::iris;
    return sourceLayerMask | (skinVisibleOcular ? kNumiHumanSkinInspectionLayerBit : 0u);
}

// Derived opaque Skin-only sclera windows use reserved presentation identities
// and always retain their canonical source ID in a sidecar face map.
inline constexpr std::uint32_t kNumiHumanRightScleraWindowStableID = 0xFFFFFE01u;
inline constexpr std::uint32_t kNumiHumanLeftScleraWindowStableID = 0xFFFFFE02u;
inline constexpr std::uint32_t kNumiHumanRightPosteriorLensBackingStableID = 0xFFFFFE03u;
inline constexpr std::uint32_t kNumiHumanLeftPosteriorLensBackingStableID = 0xFFFFFE04u;
inline constexpr std::uint32_t numiHumanRestingScleraWindowLayerMask() noexcept {
    return kNumiHumanSkinInspectionLayerBit;
}
inline constexpr std::uint32_t numiHumanRestingPosteriorLensBackingLayerMask() noexcept {
    return kNumiHumanSkinInspectionLayerBit;
}

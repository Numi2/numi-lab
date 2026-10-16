#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace metalrobo {

inline constexpr std::uint32_t kInteractionRootTargetCount = 7u;
inline constexpr std::uint32_t kInteractionContactFeatureCount = 13u;
inline constexpr std::uint32_t kInteractionContactFeatureMask =
    (1u << kInteractionContactFeatureCount) - 1u;
inline constexpr const char* kInteractionCoordinateFrame =
    "metalrobo_z_up_x_forward_xyzw";

// Contact intent remains distinct from measured solver contact. Approach and
// release are non-load-bearing transition modes; stick, roll, and slide expect
// physical contact.
enum class InteractionContactMode : std::uint32_t {
    free = 0u,
    approach = 1u,
    stick = 2u,
    roll = 3u,
    slide = 4u,
    release = 5u,
};

enum InteractionSampleFlags : std::uint32_t {
    interactionSamplePredicted = 1u << 0u,
    interactionSamplePhysicsCertified = 1u << 1u,
};

// Feature order deliberately matches SensorPack support-patch observations:
// local force xyz (N), local torque xyz (N*m), local CoP xy (m), occupied
// area (m^2), then a 2x2 row-major pressure field (Pa). A validity bit gates
// every feature independently, so a motion generator's binary foot contact
// cannot impersonate force truth.
struct InteractionContactTrack {
    std::string id;
    std::string taskContactGroup;
    std::string counterpart;
};

struct InteractionClip {
    std::string id;
    std::string desiredOutcome;
    float framesPerSecond = 0.0f;
    std::uint32_t frameCount = 0u;
    bool loop = false;

    // Frame-major MetalRobo root-link pose: link-origin position xyz followed
    // by quaternion xyzw. The compiler/runtime converts this authored link
    // frame to the solver's floating-root COM coordinates; artifact producers
    // must never perform that mechanism-specific conversion themselves.
    // Joint targets follow InteractionPack::jointNames.
    std::vector<float> rootTargets;
    std::vector<float> jointTargets;

    // Frame-major contact records follow InteractionPack::contactTracks.
    std::vector<std::uint32_t> contactModes;
    std::vector<std::uint32_t> contactFeatureMasks;
    std::vector<std::uint32_t> contactSampleFlags;
    std::vector<float> contactConfidence;
    std::vector<float> contactTargets;
    std::vector<float> contactTolerances;
};

// Canonical contact-first reference artifact. Generated intent is a target;
// only physics-certified samples may claim solved wrench/CoP/pressure values.
struct InteractionPack {
    std::string id;
    std::string sourceRepository;
    std::string sourceRevision;
    std::string license;
    std::string coordinateFrame = kInteractionCoordinateFrame;
    std::vector<std::string> jointNames;
    std::vector<InteractionContactTrack> contactTracks;
    std::vector<InteractionClip> clips;
};

// Shared structural validator used by artifact I/O and task compilation.
// It does not resolve robot or TaskPack semantics.
[[nodiscard]] bool validInteractionPack(
    const InteractionPack& pack
) noexcept;

} // namespace metalrobo

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

#include "metalrobo/EngineModelComposer.hpp"
#include "metalrobo/MatterSnapshotArchive.hpp"
#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/MetalWorldFamily.hpp"
#include "metalrobo/SurgicalAssets.hpp"
#include "metalrobo/SurgicalPSM.hpp"
#include "metalrobo/SurgicalVisual.hpp"
#include "metalrobo/SurgicalWorld.hpp"
#include "metalrobo/VisualPlatform.hpp"
#include "metalrobo/WorldCompiler.hpp"
#include "numi/matter/surgical_tissue.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kWidth = 1280u;
constexpr std::uint32_t kHeight = 960u;
constexpr std::uint32_t kInstrumentSemantic = 101u;
constexpr std::uint32_t kReceiverInstrumentSemantic = 102u;
constexpr std::uint32_t kNeedleSemantic = 201u;
constexpr std::uint32_t kThreadSemantic = 202u;
constexpr std::uint32_t kFieldSemantic = 301u;
constexpr std::uint32_t kTissueSemantic = 302u;
constexpr std::uint32_t kQualifiedThreadNodeCount = 25u;
constexpr double kQualifiedThreadRadius = 1.2e-4;
constexpr double kQualifiedThreadLength = 0.18;
constexpr std::uint32_t kHandoffThreadNodeCount = 128u;
constexpr double kHandoffThreadRadius = 1.0e-4;
constexpr double kHandoffThreadLength = 0.25;
constexpr double kHandoffTissueCenterX = 0.042;

struct Arguments {
    std::filesystem::path statePath;
    std::filesystem::path outputDirectory;
    bool geometryOnly = false;
};

struct PickupState {
    bool dualHandoff = false;
    bool medicallyMatchedSinglePickup = false;
    std::string phase;
    std::string model;
    std::uint64_t step = 0u;
    double controlTimestepSeconds = 0.0005;
    std::vector<float> q;
    std::vector<float> v;
    MRBodyStateGPU needle{};
    double threadRadius = 0.0;
    std::uint32_t threadNodeCount = 0u;
    double threadLength = 0.0;
    std::vector<std::array<double, 3>> threadPositions;
    std::vector<std::array<double, 3>> threadVelocities;
    std::vector<std::array<double, 2>> threadTwists;
    double tissueLength = 0.0;
    double tissueWidth = 0.0;
    double tissueThickness = 0.0;
    double tissueIncisionGap = 0.0;
    std::string matterSnapshotFilename;
    std::uint64_t matterSnapshotContentHash = 0u;
    std::uint64_t matterSnapshotPayloadBytes = 0u;
    std::uint64_t matterSourcePhysicsFingerprint = 0u;
    std::uint64_t matterDeviceProgramFingerprint = 0u;
    std::shared_ptr<const numi::matter::RuntimeStateSnapshot> matterSnapshot;
    double needleLift = 0.0;
    double jawTravel = 0.0;
    double followRatio = 0.0;
    double graspSeatDrift = 0.0;
    double orientationDrift = 0.0;
    bool grasped = false;
    bool modelSeen = false;
    bool phaseSeen = false;
    bool stepSeen = false;
    bool controlTimestepSeen = false;
    bool qSeen = false;
    bool vSeen = false;
    bool needlePositionSeen = false;
    bool needleOrientationSeen = false;
    bool needleLinearVelocitySeen = false;
    bool needleAngularVelocitySeen = false;
    bool threadModelSeen = false;
    bool tissueModelSeen = false;
    bool matterSnapshotRequired = false;
    bool matterSnapshotSeen = false;
    std::uint64_t knotProtocolFingerprint = 0u;
    std::uint32_t knotThrowIndex = 0u;
    std::uint32_t knotCompletedSample = 0u;
    bool knotProtocolSeen = false;
    bool knotTargetSeen = false;
    bool knotFrameOriginSeen = false;
    bool knotFrameXSeen = false;
    bool knotFrameYSeen = false;
    bool knotFrameZSeen = false;
    bool needleLiftSeen = false;
    bool jawTravelSeen = false;
    bool followRatioSeen = false;
    bool graspSeatDriftSeen = false;
    bool orientationDriftSeen = false;
    bool graspedSeen = false;
    std::vector<std::uint8_t> threadPositionsSeen;
    std::vector<std::uint8_t> threadVelocitiesSeen;
    std::vector<std::uint8_t> threadTwistsSeen;
};

struct VisibilityMetrics {
    std::size_t instrumentPixels = 0u;
    std::size_t receiverInstrumentPixels = 0u;
    std::size_t needlePixels = 0u;
    std::size_t threadPixels = 0u;
    std::size_t fieldPixels = 0u;
    std::size_t tissuePixels = 0u;
    std::size_t validPixels = 0u;
    float minimumDepth = std::numeric_limits<float>::infinity();
    float maximumDepth = 0.0f;
};

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Diagnostics>
void requireSucceeded(
    const Diagnostics& diagnostics,
    const char* operation
) {
    if (!diagnostics.succeeded()) {
        throw std::runtime_error(
            std::string{operation} + ": " + diagnostics.message
        );
    }
}

Arguments parseArguments(
    const int argumentCount,
    char* const arguments[]
) {
    Arguments result;
    for (int index = 1; index < argumentCount; ++index) {
        const std::string_view argument{arguments[index]};
        if (argument == "--state" && result.statePath.empty() &&
            index + 1 < argumentCount) {
            result.statePath = arguments[++index];
        } else if (argument == "--output-dir" &&
                   result.outputDirectory.empty() &&
                   index + 1 < argumentCount) {
            result.outputDirectory = arguments[++index];
        } else if (argument == "--geometry-only" &&
                   !result.geometryOnly) {
            result.geometryOnly = true;
        } else {
            throw std::invalid_argument(
                "usage: metalrobo_suture_visual_probe --state PATH "
                "--output-dir DIRECTORY [--geometry-only]"
            );
        }
    }
    if (result.statePath.empty() || result.outputDirectory.empty()) {
        throw std::invalid_argument(
            "usage: metalrobo_suture_visual_probe --state PATH "
            "--output-dir DIRECTORY [--geometry-only]"
        );
    }
    return result;
}

std::vector<std::string> splitTabs(const std::string& line) {
    std::vector<std::string> result;
    std::size_t begin = 0u;
    while (begin <= line.size()) {
        const std::size_t end = line.find('\t', begin);
        result.push_back(line.substr(
            begin,
            end == std::string::npos ? end : end - begin
        ));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1u;
    }
    return result;
}

double number(const std::string& text, const char* field) {
    std::size_t consumed = 0u;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument(
            std::string{"invalid pickup state "} + field
        );
    }
    return value;
}

std::uint64_t integer(const std::string& text, const char* field) {
    std::size_t consumed = 0u;
    const std::uint64_t value = std::stoull(text, &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument(
            std::string{"invalid pickup state "} + field
        );
    }
    return value;
}

bool isKnotVisualPhase(const std::string_view phase) {
    return phase == "tissue-knot-first-throw-staged" ||
        phase == "tissue-knot-first-double-throw";
}

bool isAcceptedDualHandoffVisualPhase(const std::string_view phase) {
    // Every entry is a transactionally published q/v + rigid + DER snapshot.
    // Keep this finite so an arbitrary or partially written phase cannot be
    // presented as operative evidence merely because its array widths match.
    static constexpr std::array<std::string_view, 38u> phases{
        "giver-closed",
        "giver-lift",
        "giver-handoff-stage",
        "receiver-approach",
        "receiver-aligned",
        "positive-control-motion",
        "positive-control-overlap",
        "load-exchange",
        "giver-release",
        "receiver-transfer-motion",
        "receiver-transfer",
        "receiver-extraction-giver-hold",
        "receiver-extraction-approach",
        "receiver-extraction-positive-control",
        "receiver-extraction-load-exchange",
        "receiver-extraction-giver-release",
        "receiver-extraction-giver-retreated",
        "receiver-extraction-preload-released",
        "receiver-extraction-retraction-settled",
        "receiver-extraction-retracted",
        "tissue-rest",
        "tissue-receiver-bridge-start",
        "tissue-receiver-dynamic-bridge",
        "tissue-receiver-alignment-motion",
        "tissue-receiver-alignment-settled",
        "tissue-receiver-acquisition",
        "tissue-receiver-extraction",
        "tissue-opposing-bite-distal-clearance",
        "tissue-opposing-bite-reoriented",
        "tissue-opposing-bite-ready",
        "tissue-opposing-bite-passage",
        "tissue-opposing-bite-thread-root",
        "tissue-suture-pull-stroke",
        "tissue-suture-pull-complete",
        "tissue-thread-approached",
        "tissue-thread-acquired",
        "tissue-knot-first-throw-staged",
        "tissue-knot-first-double-throw",
    };
    return std::ranges::find(phases, phase) != phases.end();
}

bool isPostHandoffOperativeVisualPhase(const std::string_view phase) {
    return phase.starts_with("receiver-extraction-") ||
        phase.starts_with("tissue-");
}

PickupState readPickupState(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("could not open pickup state artifact");
    }
    PickupState result;
    result.needle.orientation.w = 1.0f;
    bool schema = false;
    std::string line;
    while (std::getline(input, line)) {
        const std::vector<std::string> fields = splitTabs(line);
        if (fields.empty()) {
            continue;
        }
        if (fields[0] == "schema") {
            require(
                fields.size() == 2u && !schema &&
                    (fields[1] == "numi.surgical-pickup-state.v1" ||
                     fields[1] == "numi.surgical-pickup-state.v2" ||
                     fields[1] ==
                         "numi.dual-psm-suture-handoff-state.v2" ||
                     fields[1] ==
                         "numi.dual-psm-suture-handoff-state.v3"),
                "pickup state schema is unsupported"
            );
            result.dualHandoff = fields[1] !=
                    "numi.surgical-pickup-state.v1" &&
                fields[1] != "numi.surgical-pickup-state.v2";
            result.medicallyMatchedSinglePickup =
                fields[1] == "numi.surgical-pickup-state.v2";
            if (result.dualHandoff) {
                result.controlTimestepSeconds = 0.002;
            }
            result.matterSnapshotRequired = fields[1] ==
                "numi.dual-psm-suture-handoff-state.v3";
            schema = true;
        } else if (fields[0] == "phase") {
            require(
                fields.size() == 2u && !result.phaseSeen,
                "handoff phase row is invalid"
            );
            result.phase = fields[1];
            result.phaseSeen = true;
        } else if (fields[0] == "model") {
            require(
                fields.size() == 2u && !result.modelSeen,
                "pickup model row is invalid"
            );
            result.model = fields[1];
            result.modelSeen = true;
        } else if (fields[0] == "step") {
            require(
                fields.size() == 2u && !result.stepSeen,
                "pickup step row is invalid"
            );
            result.step = integer(fields[1], "step");
            result.stepSeen = true;
        } else if (fields[0] == "control_timestep_s") {
            require(
                fields.size() == 2u && !result.controlTimestepSeen &&
                    number(fields[1], "control timestep") > 0.0,
                "pickup control timestep row is invalid"
            );
            result.controlTimestepSeconds =
                number(fields[1], "control timestep");
            result.controlTimestepSeen = true;
        } else if (fields[0] == "q" || fields[0] == "v") {
            bool& seen = fields[0] == "q"
                ? result.qSeen
                : result.vSeen;
            require(
                !seen && fields.size() > 1u,
                "pickup generalized-state row is invalid"
            );
            seen = true;
            std::vector<float>& values =
                fields[0] == "q" ? result.q : result.v;
            for (std::size_t index = 1u; index < fields.size(); ++index) {
                values.push_back(static_cast<float>(number(
                    fields[index],
                    fields[0].c_str()
                )));
            }
        } else if (fields[0] == "needle_position") {
            require(
                fields.size() == 4u && !result.needlePositionSeen,
                "needle position row is invalid"
            );
            result.needle.position = {
                static_cast<float>(number(fields[1], "needle position")),
                static_cast<float>(number(fields[2], "needle position")),
                static_cast<float>(number(fields[3], "needle position")),
                1.0f,
            };
            result.needlePositionSeen = true;
        } else if (fields[0] == "needle_orientation_xyzw") {
            require(
                fields.size() == 5u && !result.needleOrientationSeen,
                "needle orientation row is invalid"
            );
            result.needle.orientation = {
                static_cast<float>(number(fields[1], "needle orientation")),
                static_cast<float>(number(fields[2], "needle orientation")),
                static_cast<float>(number(fields[3], "needle orientation")),
                static_cast<float>(number(fields[4], "needle orientation")),
            };
            result.needleOrientationSeen = true;
        } else if (fields[0] == "needle_linear_velocity") {
            require(
                fields.size() == 4u && !result.needleLinearVelocitySeen,
                "needle velocity row is invalid"
            );
            result.needle.linearVelocityAndInverseMass = {
                static_cast<float>(number(fields[1], "needle velocity")),
                static_cast<float>(number(fields[2], "needle velocity")),
                static_cast<float>(number(fields[3], "needle velocity")),
                0.0f,
            };
            result.needleLinearVelocitySeen = true;
        } else if (fields[0] == "needle_angular_velocity") {
            require(
                fields.size() == 4u && !result.needleAngularVelocitySeen,
                "needle angular row is invalid"
            );
            result.needle.angularVelocity = {
                static_cast<float>(number(fields[1], "needle angular")),
                static_cast<float>(number(fields[2], "needle angular")),
                static_cast<float>(number(fields[3], "needle angular")),
                0.0f,
            };
            result.needleAngularVelocitySeen = true;
        } else if (fields[0] == "thread_model") {
            require(
                fields.size() == 4u && !result.threadModelSeen,
                "thread model row is invalid"
            );
            result.threadRadius = number(fields[1], "thread radius");
            const std::uint64_t threadNodeCount = integer(
                fields[2],
                "thread node count"
            );
            require(threadNodeCount >= 2u, "thread node count is invalid");
            result.threadNodeCount = static_cast<std::uint32_t>(
                threadNodeCount
            );
            result.threadLength = number(fields[3], "thread length");
            result.threadPositions.resize(result.threadNodeCount);
            result.threadVelocities.resize(result.threadNodeCount);
            result.threadTwists.resize(result.threadNodeCount - 1u);
            result.threadPositionsSeen.resize(result.threadNodeCount, 0u);
            result.threadVelocitiesSeen.resize(result.threadNodeCount, 0u);
            result.threadTwistsSeen.resize(result.threadNodeCount - 1u, 0u);
            result.threadModelSeen = true;
        } else if (fields[0] == "thread_position" ||
                   fields[0] == "thread_velocity") {
            require(fields.size() == 5u, "thread state row is invalid");
            const std::size_t index = static_cast<std::size_t>(
                integer(fields[1], "thread node index")
            );
            std::vector<std::array<double, 3>>& values =
                fields[0] == "thread_position"
                ? result.threadPositions
                : result.threadVelocities;
            std::vector<std::uint8_t>& seen =
                fields[0] == "thread_position"
                ? result.threadPositionsSeen
                : result.threadVelocitiesSeen;
            require(index < values.size(), "thread node index is outside model");
            require(seen[index] == 0u, "thread node row is duplicated");
            values[index] = {
                number(fields[2], fields[0].c_str()),
                number(fields[3], fields[0].c_str()),
                number(fields[4], fields[0].c_str()),
            };
            seen[index] = 1u;
        } else if (fields[0] == "thread_twist") {
            require(
                fields.size() == 4u && result.threadModelSeen,
                "thread twist row is invalid"
            );
            const std::size_t index = static_cast<std::size_t>(
                integer(fields[1], "thread edge index")
            );
            require(
                index < result.threadTwists.size(),
                "thread edge index is outside model"
            );
            require(
                result.threadTwistsSeen[index] == 0u,
                "thread twist row is duplicated"
            );
            result.threadTwists[index] = {
                number(fields[2], "thread twist"),
                number(fields[3], "thread twist rate"),
            };
            result.threadTwistsSeen[index] = 1u;
        } else if (fields[0] == "needle_lift_m") {
            require(
                fields.size() == 2u && !result.needleLiftSeen,
                "needle lift row is invalid"
            );
            result.needleLift = number(fields[1u], "needle lift");
            result.needleLiftSeen = true;
        } else if (fields[0] == "jaw_travel_m") {
            require(
                fields.size() == 2u && !result.jawTravelSeen,
                "jaw travel row is invalid"
            );
            result.jawTravel = number(fields[1u], "jaw travel");
            result.jawTravelSeen = true;
        } else if (fields[0] == "follow_ratio") {
            require(
                fields.size() == 2u && !result.followRatioSeen,
                "follow ratio row is invalid"
            );
            result.followRatio = number(fields[1u], "follow ratio");
            result.followRatioSeen = true;
        } else if (fields[0] == "orientation_drift_rad") {
            require(
                fields.size() == 2u && !result.orientationDriftSeen,
                "orientation drift row is invalid"
            );
            result.orientationDrift = number(fields[1u], "orientation drift");
            result.orientationDriftSeen = true;
        } else if (fields[0] == "jaw_needle_seat_drift_m") {
            require(
                fields.size() == 2u && !result.graspSeatDriftSeen,
                "pickup grasp-seat row is invalid"
            );
            result.graspSeatDrift = number(
                fields[1u],
                "pickup grasp-seat drift"
            );
            result.graspSeatDriftSeen = true;
        } else if (fields[0] == "grasped") {
            require(
                fields.size() == 2u && !result.graspedSeen,
                "grasped row is invalid"
            );
            const std::uint64_t grasped = integer(fields[1u], "grasped");
            require(grasped <= 1u, "grasped state is not boolean");
            result.grasped = grasped == 1u;
            result.graspedSeen = true;
        } else if (fields[0] == "tissue_model") {
            require(
                fields.size() == 6u && !result.tissueModelSeen,
                "handoff tissue row is invalid"
            );
            const double length = number(fields[2], "tissue length");
            const double width = number(fields[3], "tissue width");
            const double thickness = number(fields[4], "tissue thickness");
            const double incisionGap = number(
                fields[5],
                "tissue incision gap"
            );
            require(
                fields[1] == "porcine_jejunum_fung" &&
                    length > 0.0 && width > 0.0 && thickness > 0.0 &&
                    incisionGap > 0.0,
                "handoff tissue row is invalid"
            );
            result.tissueLength = length;
            result.tissueWidth = width;
            result.tissueThickness = thickness;
            result.tissueIncisionGap = incisionGap;
            result.tissueModelSeen = true;
        } else if (fields[0] == "matter_snapshot") {
            require(
                fields.size() == 6u && result.dualHandoff &&
                    !result.matterSnapshotSeen,
                "handoff Matter snapshot row is invalid"
            );
            const std::filesystem::path filename = fields[1];
            result.matterSnapshotFilename = fields[1];
            result.matterSnapshotContentHash = integer(
                fields[2],
                "Matter snapshot content hash"
            );
            result.matterSnapshotPayloadBytes = integer(
                fields[3],
                "Matter snapshot payload bytes"
            );
            result.matterSourcePhysicsFingerprint = integer(
                fields[4],
                "Matter source physics fingerprint"
            );
            result.matterDeviceProgramFingerprint = integer(
                fields[5],
                "Matter device program fingerprint"
            );
            require(
                !filename.empty() && filename == filename.filename() &&
                    result.matterSnapshotFilename.ends_with(".matter.bin") &&
                    result.matterSnapshotContentHash != 0u &&
                    result.matterSnapshotPayloadBytes != 0u &&
                    result.matterSourcePhysicsFingerprint != 0u &&
                    result.matterDeviceProgramFingerprint != 0u,
                "handoff Matter snapshot identity is invalid"
            );
            result.matterSnapshotSeen = true;
        } else if (fields[0] == "knot_protocol") {
            require(
                fields.size() == 8u && !result.knotProtocolSeen,
                "handoff knot protocol row is invalid"
            );
            const std::uint64_t throwIndex = integer(
                fields[3],
                "knot throw index"
            );
            const std::uint64_t completedSample = integer(
                fields[4],
                "knot completed sample"
            );
            const std::uint64_t protocolFingerprint = integer(
                fields[2],
                "knot protocol fingerprint"
            );
            require(
                integer(fields[1], "knot protocol revision") != 0u &&
                    protocolFingerprint != 0u &&
                    integer(fields[5], "knot whole turns") != 0u &&
                    throwIndex <=
                        std::numeric_limits<std::uint32_t>::max() &&
                    completedSample <=
                        std::numeric_limits<std::uint32_t>::max() &&
                    std::abs(number(fields[6], "knot winding sign")) ==
                        1.0 &&
                    std::abs(number(fields[7], "knot transfer sign")) ==
                        1.0,
                "handoff knot protocol row is invalid"
            );
            result.knotThrowIndex =
                static_cast<std::uint32_t>(throwIndex);
            result.knotProtocolFingerprint = protocolFingerprint;
            result.knotCompletedSample =
                static_cast<std::uint32_t>(completedSample);
            result.knotProtocolSeen = true;
        } else if (fields[0] == "knot_standing_target") {
            require(
                fields.size() == 4u && !result.knotTargetSeen &&
                    integer(fields[2], "knot window first") <=
                        integer(fields[1], "knot center edge") &&
                    integer(fields[1], "knot center edge") <
                        integer(fields[3], "knot window last"),
                "handoff knot target row is invalid"
            );
            result.knotTargetSeen = true;
        } else if (fields[0] == "knot_frame_origin" ||
                   fields[0] == "knot_frame_x" ||
                   fields[0] == "knot_frame_y" ||
                   fields[0] == "knot_frame_z") {
            require(
                fields.size() == 4u,
                "handoff knot frame row is invalid"
            );
            (void)number(fields[1], "knot frame x");
            (void)number(fields[2], "knot frame y");
            (void)number(fields[3], "knot frame z");
            bool* seen = fields[0] == "knot_frame_origin"
                ? &result.knotFrameOriginSeen
                : (fields[0] == "knot_frame_x"
                    ? &result.knotFrameXSeen
                    : (fields[0] == "knot_frame_y"
                        ? &result.knotFrameYSeen
                        : &result.knotFrameZSeen));
            require(!*seen, "duplicate handoff knot frame row");
            *seen = true;
        } else {
            throw std::invalid_argument(
                "pickup state contains an unknown row: " + fields[0]
            );
        }
    }
    require(schema, "pickup state has no schema");
    const bool commonValid =
        result.modelSeen && result.stepSeen && result.qSeen && result.vSeen &&
            result.needlePositionSeen && result.needleOrientationSeen &&
            result.needleLinearVelocitySeen &&
            result.needleAngularVelocitySeen && result.threadModelSeen &&
            std::ranges::all_of(
                result.threadPositionsSeen,
                [](const std::uint8_t value) { return value == 1u; }
            ) &&
            std::ranges::all_of(
                result.threadVelocitiesSeen,
                [](const std::uint8_t value) { return value == 1u; }
            ) &&
            result.threadPositions.size() == result.threadNodeCount &&
            result.threadVelocities.size() == result.threadNodeCount;
    const bool singlePickupValid = !result.dualHandoff &&
            result.needleLiftSeen && result.jawTravelSeen &&
            result.followRatioSeen && result.orientationDriftSeen &&
            result.graspedSeen &&
            (result.model ==
                 "dvrk_psm_classic_lnd_source_coupled_abi_v3" ||
             result.model ==
                 "dvrk_psm_classic_lnd_source_coupled_abi_v4") &&
            result.step == 3030u &&
            result.q.size() == metalrobo::kSurgicalPSMJointCount &&
            result.v.size() == metalrobo::kSurgicalPSMJointCount &&
            result.threadNodeCount == kQualifiedThreadNodeCount &&
            std::abs(result.threadRadius - kQualifiedThreadRadius) <=
                1.0e-12 &&
            std::abs(result.threadLength - kQualifiedThreadLength) <=
                1.0e-12 &&
            result.grasped &&
            result.needleLift > 0.007 &&
            (
                result.medicallyMatchedSinglePickup
                ? result.graspSeatDriftSeen &&
                    result.graspSeatDrift < 1.0e-4
                : result.followRatio > 0.9
            );
    const bool dualHandoffValid = result.dualHandoff &&
        result.phaseSeen && result.controlTimestepSeen &&
        result.tissueModelSeen &&
        (!result.matterSnapshotRequired || result.matterSnapshotSeen) &&
        result.model == "dual_psm_bowel_suture_neutral_zone_world" &&
        result.q.size() == metalrobo::kDualPsmQCount &&
        result.v.size() == metalrobo::kDualPsmVCount &&
        isAcceptedDualHandoffVisualPhase(result.phase) &&
        result.threadNodeCount == kHandoffThreadNodeCount &&
        result.threadTwists.size() + 1u == result.threadNodeCount &&
        std::ranges::all_of(
            result.threadTwistsSeen,
            [](const std::uint8_t value) { return value == 1u; }
        ) &&
        std::abs(result.threadRadius - kHandoffThreadRadius) <= 1.0e-12 &&
        std::abs(result.threadLength - kHandoffThreadLength) <= 1.0e-12;
    const std::uint32_t knotMetadataRows =
        static_cast<std::uint32_t>(result.knotProtocolSeen) +
        static_cast<std::uint32_t>(result.knotTargetSeen) +
        static_cast<std::uint32_t>(result.knotFrameOriginSeen) +
        static_cast<std::uint32_t>(result.knotFrameXSeen) +
        static_cast<std::uint32_t>(result.knotFrameYSeen) +
        static_cast<std::uint32_t>(result.knotFrameZSeen);
    const bool knotMetadataValid = isKnotVisualPhase(result.phase)
        ? knotMetadataRows == 6u && result.knotThrowIndex == 0u &&
            (result.phase == "tissue-knot-first-throw-staged"
                ? result.knotCompletedSample == 0u
                : result.knotCompletedSample > 0u)
        : knotMetadataRows == 0u;
    require(
        commonValid && (singlePickupValid || dualHandoffValid) &&
            (!result.dualHandoff || knotMetadataValid),
        "pickup state did not pass the physical acquisition contract"
    );
    if (result.matterSnapshotRequired) {
        auto snapshot =
            std::make_shared<numi::matter::RuntimeStateSnapshot>();
        const metalrobo::MatterSnapshotArchiveResult archive =
            metalrobo::readMatterSnapshotArchive(
                path.parent_path() / result.matterSnapshotFilename,
                *snapshot
            );
        require(
            archive.succeeded() && snapshot->available &&
                archive.contentHash == result.matterSnapshotContentHash &&
                archive.payloadBytes == result.matterSnapshotPayloadBytes &&
                snapshot->sourcePhysicsFingerprint ==
                    result.matterSourcePhysicsFingerprint &&
                snapshot->deviceProgramFingerprint ==
                    result.matterDeviceProgramFingerprint,
            "Matter visual snapshot failed archive identity validation: " +
                archive.message
        );
        result.matterSnapshot = std::move(snapshot);
    }
    return result;
}

metalrobo::EngineModel makeNeedleModel(
    const metalrobo::CurvedSutureNeedleAsset& needle
) {
    metalrobo::EngineModel model;
    model.name = needle.spec.arcLengthM.basis ==
            metalrobo::SurgicalValueBasis::pdsII3_0ProductGeometry
        ? "pdsii_3_0_bowel_needle_scene"
        : "gs21_curved_needle_scene";
    model.bodies = {needle.rigid.body};
    model.bodies[0u].articulationIndex = MR_INVALID_INDEX;
    model.bodies[0u].parentBody = MR_INVALID_INDEX;
    model.bodies[0u].inboundJoint = MR_INVALID_INDEX;
    model.materials = {needle.rigid.material};
    model.shapes = needle.rigid.shapes;
    for (MRShapeGPU& shape : model.shapes) {
        shape.bodyIndex = 0u;
        shape.materialIndex = 0u;
    }
    model.world.abiVersion = MR_ENGINE_ABI_VERSION;
    model.world.bodyCount = 1u;
    model.world.shapeCount = static_cast<std::uint32_t>(model.shapes.size());
    model.world.materialCount = 1u;
    const std::uint64_t pairCount =
        static_cast<std::uint64_t>(model.shapes.size()) *
        (model.shapes.size() - 1u) / 2u;
    model.world.pairCapacity = static_cast<std::uint32_t>(pairCount);
    model.world.contactCapacity = static_cast<std::uint32_t>(4u * pairCount);
    model.world.constraintCapacity = model.world.contactCapacity;
    model.world.islandCapacity = 1u;
    model.world.solverType = MR_SOLVER_TEMPORAL_CONE;
    model.world.frictionConeType = MR_FRICTION_CONE_ELLIPTIC;
    model.world.gravityAndTimestep = {0.0f, 0.0f, -9.81f, 0.0005f};
    model.world.solverScales = {1.0e-7f, 1.0e-9f, 2.0f, 1.0e-5f};
    return model;
}

metalrobo::EngineModel makeVisualEngineModel(
    const metalrobo::CurvedSutureNeedleAsset& needle,
    const PickupState& state
) {
    const metalrobo::EngineModel needleModel = makeNeedleModel(needle);
    metalrobo::EngineModel result;
    metalrobo::EngineModelComposeDiagnostics diagnostics;
    if (state.dualHandoff) {
        const metalrobo::DualPsmWorld dual =
            metalrobo::makeDualDvrkPsmWorld();
        const std::array components{
            metalrobo::EngineModelComponent{
                .model = &dual.model,
                .instanceId = "dual_dvrk_psm",
                .preserveSemanticNames = true,
            },
            metalrobo::EngineModelComponent{
                .model = &needleModel,
                .instanceId = "bowel_suture_needle",
            },
        };
        diagnostics = metalrobo::composeEngineModels(
            components,
            result,
            {
                .name = "dual_dvrk_psm_suture_handoff_visual_world",
                .gravityAndTimestep = {
                    0.0f,
                    0.0f,
                    -9.81f,
                    static_cast<float>(state.controlTimestepSeconds),
                },
            }
        );
    } else {
        const metalrobo::EngineModel psm =
            metalrobo::makeDvrkPsmLargeNeedleDriverEngineModel();
        const std::string_view needleInstance =
            state.medicallyMatchedSinglePickup
            ? "bowel_suture_needle"
            : "gs21_needle";
        const std::array components{
            metalrobo::EngineModelComponent{
                .model = &psm,
                .instanceId = "dvrk_psm",
                .preserveSemanticNames = true,
            },
            metalrobo::EngineModelComponent{
                .model = &needleModel,
                .instanceId = needleInstance,
            },
        };
        diagnostics = metalrobo::composeEngineModels(
            components,
            result,
            {
                .name = state.medicallyMatchedSinglePickup
                    ? "dvrk_psm_bowel_suture_pickup_visual_world"
                    : "dvrk_psm_gs21_suture_visual_world",
                .gravityAndTimestep = {
                    0.0f,
                    0.0f,
                    state.medicallyMatchedSinglePickup ? 9.81f : -9.81f,
                    0.0005f,
                },
            }
        );
    }
    requireSucceeded(diagnostics, "visual engine composition");
    require(result.world.nq == state.q.size(), "visual model q width changed");
    std::copy(state.q.begin(), state.q.end(), result.defaultQ.begin());
    std::copy(state.v.begin(), state.v.end(), result.defaultV.begin());
    std::string reason;
    require(result.valid(&reason), "visual engine model is invalid: " + reason);
    return result;
}

metalrobo::WorldPose cameraToward(
    const mr_float4 position,
    const mr_float4 target
) {
    const float x = target.x - position.x;
    const float y = target.y - position.y;
    const float z = target.z - position.z;
    const float inverseLength = 1.0f / std::sqrt(x * x + y * y + z * z);
    const mr_float4 forward{
        x * inverseLength,
        y * inverseLength,
        z * inverseLength,
        0.0f,
    };
    mr_float4 orientation{
        -forward.y,
        forward.x,
        0.0f,
        1.0f + forward.z,
    };
    if (orientation.w < 1.0e-5f) {
        orientation = {1.0f, 0.0f, 0.0f, 0.0f};
    }
    const float inverseNorm = 1.0f / std::sqrt(
        orientation.x * orientation.x +
        orientation.y * orientation.y +
        orientation.z * orientation.z +
        orientation.w * orientation.w
    );
    orientation.x *= inverseNorm;
    orientation.y *= inverseNorm;
    orientation.z *= inverseNorm;
    orientation.w *= inverseNorm;
    return {position, orientation};
}

metalrobo::VisualLightRigV1 makeSurgicalThreePointLightRig() {
    metalrobo::VisualLightRigV1 result;
    result.id = "surgical_three_point";
    result.contentHash = "procedural:surgical-three-point-v1";
    MRVisualLightGPUV1 key{};
    key.positionAndRange = {0.16f, -0.22f, 0.60f, 1.2f};
    key.directionAndSpot = {-0.33f, 0.45f, -0.83f, -1.0f};
    key.colorAndIntensity = {1.0f, 0.95f, 0.90f, 720.0f};
    key.shape = {0.18f, 0.14f, -1.0f, 0.025f};
    key.identity = {
        MR_VISUAL_LIGHT_RECTANGLE,
        MR_VISUAL_LIGHT_UNIT_NIT,
        120u,
        1001u,
    };
    key.shadow = {1u, 0u, 12u, 0u};

    MRVisualLightGPUV1 fill{};
    fill.positionAndRange = {-0.18f, -0.02f, 0.38f, 1.0f};
    fill.directionAndSpot = {0.705f, 0.078f, -0.705f, -1.0f};
    fill.colorAndIntensity = {0.72f, 0.84f, 1.0f, 230.0f};
    fill.shape = {0.16f, 0.12f, -1.0f, 0.02f};
    fill.identity = {
        MR_VISUAL_LIGHT_RECTANGLE,
        MR_VISUAL_LIGHT_UNIT_NIT,
        90u,
        1002u,
    };
    fill.shadow = {0u, 1u, 8u, 0u};

    MRVisualLightGPUV1 rim{};
    rim.positionAndRange = {0.04f, 0.20f, 0.48f, 1.0f};
    rim.directionAndSpot = {-0.127f, -0.635f, -0.762f, -1.0f};
    rim.colorAndIntensity = {0.76f, 0.88f, 1.0f, 390.0f};
    rim.shape = {0.12f, 0.10f, -1.0f, 0.02f};
    rim.identity = {
        MR_VISUAL_LIGHT_RECTANGLE,
        MR_VISUAL_LIGHT_UNIT_NIT,
        100u,
        1003u,
    };
    rim.shadow = {0u, 2u, 8u, 0u};

    result.lights = {key, fill, rim};
    result.fingerprint = 0x5355545552454c31ull;
    return result;
}

metalrobo::WorldAsset asset(
    std::string id,
    std::string semanticClass,
    const MRWorldAssetRole role,
    const MRWorldDynamicsRepresentation dynamics
) {
    metalrobo::WorldAsset result;
    result.id = std::move(id);
    result.semanticClass = std::move(semanticClass);
    result.role = role;
    result.render = MR_WORLD_RENDER_MESH_PBR;
    result.collision = MR_WORLD_COLLISION_PRIMITIVES;
    result.dynamics = dynamics;
    result.topologyCohort = "dvrk_suture_pickup";
    return result;
}

mr_float4 rotateDirection(
    const mr_float4 quaternion,
    const mr_float4 value
) {
    const mr_float4 doubled{
        2.0f * (
            quaternion.y * value.z - quaternion.z * value.y
        ),
        2.0f * (
            quaternion.z * value.x - quaternion.x * value.z
        ),
        2.0f * (
            quaternion.x * value.y - quaternion.y * value.x
        ),
        0.0f,
    };
    return {
        value.x + quaternion.w * doubled.x +
            quaternion.y * doubled.z - quaternion.z * doubled.y,
        value.y + quaternion.w * doubled.y +
            quaternion.z * doubled.x - quaternion.x * doubled.z,
        value.z + quaternion.w * doubled.z +
            quaternion.x * doubled.y - quaternion.y * doubled.x,
        0.0f,
    };
}

void assignRotatedInverseInertia(
    MRBodyStateGPU& state,
    const MRBodyPropertiesGPU& properties
) {
    const std::array<mr_float4, 3u> axes{{
        rotateDirection(state.orientation, {1.0f, 0.0f, 0.0f, 0.0f}),
        rotateDirection(state.orientation, {0.0f, 1.0f, 0.0f, 0.0f}),
        rotateDirection(state.orientation, {0.0f, 0.0f, 1.0f, 0.0f}),
    }};
    const auto component = [](const mr_float4 value, const std::size_t axis) {
        return axis == 0u ? value.x : axis == 1u ? value.y : value.z;
    };
    const float body[3][3] = {
        {properties.inverseInertiaRow0.x,
         properties.inverseInertiaRow0.y,
         properties.inverseInertiaRow0.z},
        {properties.inverseInertiaRow1.x,
         properties.inverseInertiaRow1.y,
         properties.inverseInertiaRow1.z},
        {properties.inverseInertiaRow2.x,
         properties.inverseInertiaRow2.y,
         properties.inverseInertiaRow2.z},
    };
    float world[3][3]{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            for (std::size_t bodyRow = 0u; bodyRow < 3u; ++bodyRow) {
                for (std::size_t bodyColumn = 0u;
                     bodyColumn < 3u;
                     ++bodyColumn) {
                    world[row][column] +=
                        component(axes[bodyRow], row) *
                        body[bodyRow][bodyColumn] *
                        component(axes[bodyColumn], column);
                }
            }
        }
    }
    state.inverseInertiaWorldRow0 = {
        world[0][0], world[0][1], world[0][2], 0.0f,
    };
    state.inverseInertiaWorldRow1 = {
        world[1][0], world[1][1], world[1][2], 0.0f,
    };
    state.inverseInertiaWorldRow2 = {
        world[2][0], world[2][1], world[2][2], 0.0f,
    };
}

std::array<double, 3> visualTissueCenter(
    const PickupState& state,
    const metalrobo::DvrkSutureVisualScene& scene
) {
    const metalrobo::SurgicalNeutralZonePadSpec pad;
    if (scene.tissuePositions.empty()) {
        return {
            kHandoffTissueCenterX,
            0.0,
            0.5 * pad.thicknessM.value + 0.5 * state.tissueThickness,
        };
    }
    std::array<double, 3> lower{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    std::array<double, 3> upper{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    for (const auto& position : scene.tissuePositions) {
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            const double value =
                position[axis] + scene.tissueTranslationM[axis];
            require(
                std::isfinite(value),
                "visual tissue bounds contain a non-finite node"
            );
            lower[axis] = std::min(lower[axis], value);
            upper[axis] = std::max(upper[axis], value);
        }
    }
    return {
        0.5 * (lower[0] + upper[0]),
        0.5 * (lower[1] + upper[1]),
        0.5 * (lower[2] + upper[2]),
    };
}

metalrobo::WorldTemplate makeWorldTemplate(
    const metalrobo::EngineModel& model,
    const PickupState& state,
    const metalrobo::DvrkSutureVisualScene& visualScene
) {
    metalrobo::EpisodeTwin episode;
    episode.id = state.dualHandoff
        ? "dual_dvrk_suture_handoff_visual_v1"
        : state.medicallyMatchedSinglePickup
            ? "dvrk_bowel_suture_pickup_visual_v2"
            : "dvrk_gs21_suture_pickup_visual_v1";
    metalrobo::WorldAsset robot = asset(
        state.dualHandoff ? "giver_dvrk_psm" : "dvrk_psm",
        "surgical_instrument",
        MR_WORLD_ASSET_ROBOT,
        MR_WORLD_DYNAMICS_ARTICULATED
    );
    robot.articulationIndex = 0u;
    for (std::uint32_t body = 0u; body < 9u; ++body) {
        robot.bodyIndices.push_back(body);
    }
    for (std::uint32_t shape = 0u;
         shape < metalrobo::kSurgicalPSMShapeCount;
         ++shape) {
        robot.shapeIndices.push_back(shape);
    }
    robot.materialIndices = {0u, 1u};
    std::vector<metalrobo::WorldAsset> assets;
    assets.push_back(std::move(robot));
    if (state.dualHandoff) {
        metalrobo::WorldAsset receiver = asset(
            "receiver_dvrk_psm",
            "surgical_instrument",
            MR_WORLD_ASSET_ROBOT,
            MR_WORLD_DYNAMICS_ARTICULATED
        );
        receiver.articulationIndex = 1u;
        for (std::uint32_t body = 9u; body < 18u; ++body) {
            receiver.bodyIndices.push_back(body);
        }
        for (std::uint32_t shape =
                 metalrobo::kSurgicalPSMShapeCount;
             shape < 2u * metalrobo::kSurgicalPSMShapeCount;
             ++shape) {
            receiver.shapeIndices.push_back(shape);
        }
        receiver.materialIndices = {2u, 3u};
        assets.push_back(std::move(receiver));
    }
    metalrobo::WorldAsset needle = asset(
        state.dualHandoff || state.medicallyMatchedSinglePickup
            ? "bowel_suture_needle"
            : "gs21_needle",
        "suture_needle",
        MR_WORLD_ASSET_MANIPULATED,
        MR_WORLD_DYNAMICS_RIGID
    );
    const std::uint32_t needleBodyIndex = state.dualHandoff ? 18u : 9u;
    const std::uint32_t needleShapeBegin = state.dualHandoff
        ? 2u * metalrobo::kSurgicalPSMShapeCount
        : metalrobo::kSurgicalPSMShapeCount;
    needle.bodyIndices = {needleBodyIndex};
    for (std::uint32_t shape = needleShapeBegin;
         shape < model.shapes.size();
         ++shape) {
        needle.shapeIndices.push_back(shape);
    }
    needle.materialIndices = {state.dualHandoff ? 4u : 2u};
    needle.initialPose.position = state.needle.position;
    needle.initialPose.orientation = state.needle.orientation;
    assets.push_back(std::move(needle));
    episode.assets = std::move(assets);

    const mr_float4 needlePosition = state.needle.position;
    const mr_float4 needlePlaneNormal = rotateDirection(
        state.needle.orientation,
        {0.0f, 0.0f, 1.0f, 0.0f}
    );
    mr_float4 threadCenter{};
    for (const auto& point : state.threadPositions) {
        threadCenter.x += static_cast<float>(point[0]);
        threadCenter.y += static_cast<float>(point[1]);
        threadCenter.z += static_cast<float>(point[2]);
    }
    const float inverseThreadCount =
        1.0f / static_cast<float>(state.threadPositions.size());
    threadCenter.x *= inverseThreadCount;
    threadCenter.y *= inverseThreadCount;
    threadCenter.z *= inverseThreadCount;
    threadCenter.w = 1.0f;

    metalrobo::SensorSpec close;
    close.id = "pickup_close_rgbd";
    close.parentAssetId = state.dualHandoff
        ? "giver_dvrk_psm"
        : "dvrk_psm";
    close.parentKind = MR_WORLD_SENSOR_PARENT_WORLD;
    close.kind = MR_WORLD_SENSOR_RGBD;
    close.width = kWidth;
    close.height = kHeight;
    close.intrinsics = state.dualHandoff
        ? isPostHandoffOperativeVisualPhase(state.phase)
            ? mr_float4{1750.0f, 1750.0f, 640.0f, 480.0f}
            : mr_float4{1500.0f, 1500.0f, 640.0f, 480.0f}
        : state.medicallyMatchedSinglePickup
            ? mr_float4{1600.0f, 1600.0f, 640.0f, 480.0f}
            : mr_float4{1250.0f, 1250.0f, 640.0f, 480.0f};
    const mr_float4 closePosition = state.dualHandoff
        ? mr_float4{
              needlePosition.x - 0.050f,
              needlePosition.y - 0.070f,
              needlePosition.z + 0.045f,
              1.0f,
          }
        : mr_float4{
              needlePosition.x +
                  (state.medicallyMatchedSinglePickup
                      ? 0.060f * needlePlaneNormal.x
                      : 0.045f),
              needlePosition.y -
                  (state.medicallyMatchedSinglePickup
                      ? 0.020f - 0.060f * needlePlaneNormal.y
                      : 0.050f),
              needlePosition.z +
                  (state.medicallyMatchedSinglePickup
                      ? 0.060f * needlePlaneNormal.z
                      : 0.032f),
              1.0f,
          };
    const mr_float4 closeTarget = state.dualHandoff
        ? mr_float4{
              needlePosition.x,
              needlePosition.y,
              needlePosition.z,
              1.0f,
          }
        : mr_float4{
              needlePosition.x,
              needlePosition.y,
              needlePosition.z +
                  (state.medicallyMatchedSinglePickup ? 0.0f : -0.006f),
              1.0f,
          };
    close.localPose = cameraToward(
        closePosition,
        closeTarget
    );
    close.nominalRateHz = 60.0f;
    close.exposureSeconds = 1.0f / 240.0f;
    close.minimumDepthMeters = 0.015f;
    close.maximumDepthMeters = 2.0f;
    close.depthQuantumMeters = 1.0e-5f;

    metalrobo::SensorSpec overview = close;
    overview.id = "pickup_overview_rgbd";
    overview.intrinsics = state.dualHandoff
        ? mr_float4{2000.0f, 2000.0f, 640.0f, 480.0f}
        : state.medicallyMatchedSinglePickup
            ? mr_float4{1700.0f, 1700.0f, 640.0f, 480.0f}
            : mr_float4{1030.0f, 1030.0f, 640.0f, 480.0f};
    const std::array<double, 3> tissueCenter = visualTissueCenter(
        state,
        visualScene
    );
    const mr_float4 overviewTarget = state.dualHandoff
        ? mr_float4{
              0.35f * needlePosition.x + 0.40f * threadCenter.x +
                  0.25f * static_cast<float>(tissueCenter[0]),
              0.35f * needlePosition.y + 0.40f * threadCenter.y +
                  0.25f * static_cast<float>(tissueCenter[1]),
              0.35f * needlePosition.z + 0.40f * threadCenter.z +
                  0.25f * static_cast<float>(tissueCenter[2]),
              1.0f,
          }
        : mr_float4{
              (state.medicallyMatchedSinglePickup ? 0.80f : 0.55f) *
                      needlePosition.x +
                  (state.medicallyMatchedSinglePickup ? 0.20f : 0.45f) *
                      threadCenter.x,
              (state.medicallyMatchedSinglePickup ? 0.80f : 0.55f) *
                      needlePosition.y +
                  (state.medicallyMatchedSinglePickup ? 0.20f : 0.45f) *
                      threadCenter.y,
              (state.medicallyMatchedSinglePickup ? 0.80f : 0.55f) *
                      needlePosition.z +
                  (state.medicallyMatchedSinglePickup ? 0.20f : 0.45f) *
                      threadCenter.z,
              1.0f,
          };
    const mr_float4 overviewPosition = state.dualHandoff
        ? mr_float4{
              overviewTarget.x - 0.140f,
              overviewTarget.y - 0.160f,
              overviewTarget.z + 0.140f,
              1.0f,
          }
        : mr_float4{
              overviewTarget.x +
                  (state.medicallyMatchedSinglePickup
                      ? 0.100f * needlePlaneNormal.x
                      : 0.155f),
              overviewTarget.y -
                  (state.medicallyMatchedSinglePickup
                      ? 0.060f - 0.100f * needlePlaneNormal.y
                      : 0.175f),
              overviewTarget.z +
                  (state.medicallyMatchedSinglePickup
                      ? 0.100f * needlePlaneNormal.z
                      : 0.075f),
              1.0f,
          };
    overview.localPose = cameraToward(
        overviewPosition,
        overviewTarget
    );
    episode.sensors = {std::move(close), std::move(overview)};
    episode.task = {
        .id = state.dualHandoff
            ? "dual_dvrk_suture_handoff_visual_qualification"
            : state.medicallyMatchedSinglePickup
                ? "bowel_suture_pickup_visual_qualification"
                : "gs21_suture_pickup_visual_qualification",
        .robotAssetId = state.dualHandoff
            ? "giver_dvrk_psm"
            : "dvrk_psm",
        .controlPeriodSeconds = state.controlTimestepSeconds,
        .horizonSeconds =
            state.controlTimestepSeconds * static_cast<double>(state.step),
    };
    metalrobo::WorldTemplate result;
    const auto diagnostics = metalrobo::compileEpisodeTwin(
        episode,
        model,
        result
    );
    requireSucceeded(diagnostics, "visual episode compile");
    return result;
}

metalrobo::WorldFamily makeWorldFamily(
    const metalrobo::WorldTemplate& world
) {
    metalrobo::WorldProgram program;
    program.id = "dvrk_suture_pickup_visual_program_v2";
    metalrobo::WorldFamily result;
    const auto diagnostics = metalrobo::compileWorldFamily(
        world,
        program,
        result
    );
    requireSucceeded(diagnostics, "visual world family compile");
    return result;
}

metalrobo::DvrkSutureVisualScene makeHandoffVisualScene(
    const PickupState& state
) {
    metalrobo::DvrkSutureVisualScene result;
    if (!state.dualHandoff) {
        if (state.medicallyMatchedSinglePickup) {
            std::array<double, 3> center{};
            double maximumThreadZ =
                -std::numeric_limits<double>::infinity();
            for (const auto& point : state.threadPositions) {
                center[0u] += point[0u];
                center[1u] += point[1u];
                maximumThreadZ = std::max(maximumThreadZ, point[2u]);
            }
            const double inverseCount =
                1.0 / static_cast<double>(state.threadPositions.size());
            center[0u] *= inverseCount;
            center[1u] *= inverseCount;
            center[2u] = maximumThreadZ + 0.006;
            result.hasSurgicalFieldGeometry = true;
            result.surgicalFieldCenterM = center;
            result.surgicalFieldHalfExtentM = {0.115, 0.085, 0.0025};
        }
        return result;
    }
    result.hasSecondaryInstrument = true;
    result.secondaryInstrument = {
        .shaftBodyIndex = 12u,
        .wristPitchBodyIndex = 13u,
        .wristYawBodyIndex = 14u,
        .toolBodyIndex = 15u,
        .jawABodyIndex = 16u,
        .jawBBodyIndex = 17u,
        .needleBodyIndex = 18u,
    };
    const numi::matter::PorcineJejunumFungSpec tissueSpec;
    require(
        std::abs(state.tissueLength - tissueSpec.lengthM.value) <= 1.0e-12 &&
            std::abs(state.tissueWidth - tissueSpec.widthM.value) <=
                1.0e-12 &&
            std::abs(state.tissueThickness - tissueSpec.thicknessM.value) <=
                1.0e-12 &&
            std::abs(state.tissueIncisionGap -
                     tissueSpec.incisionGapM.value) <= 1.0e-12,
        "handoff replay tissue does not match the authored coupon"
    );
    const metalrobo::SurgicalNeutralZonePadSpec pad;
    result.hasSurgicalFieldGeometry = true;
    result.surgicalFieldCenterM = {0.0, 0.0, 0.0};
    result.surgicalFieldHalfExtentM = {
        0.5 * pad.sizeXM.value,
        0.5 * pad.sizeYM.value,
        0.5 * pad.thicknessM.value,
    };
    std::vector<std::array<std::uint32_t, 4>> activeTetrahedra;
    if (state.matterSnapshot != nullptr) {
        const numi::matter::RuntimeStateSnapshot& snapshot =
            *state.matterSnapshot;
        require(
            snapshot.available && !snapshot.femNodes.empty() &&
                !snapshot.femTopologyTetrahedra.empty(),
            "live Matter visual snapshot has no FEM state"
        );
        std::vector<std::array<std::uint32_t, 4>> sourceTetrahedra;
        std::vector<std::uint8_t> referenced(snapshot.femNodes.size(), 0u);
        for (const NMTetrahedronGPU& tetrahedron :
             snapshot.femTopologyTetrahedra) {
            if ((tetrahedron.identity.w & NM_OBJECT_ACTIVE) == 0u) {
                continue;
            }
            const std::array<std::uint32_t, 4> nodes{
                tetrahedron.nodes.x,
                tetrahedron.nodes.y,
                tetrahedron.nodes.z,
                tetrahedron.nodes.w,
            };
            for (const std::uint32_t node : nodes) {
                require(
                    node < snapshot.femNodes.size() &&
                        (snapshot.femTopologyNodes.empty() ||
                         (node < snapshot.femTopologyNodes.size() &&
                          (snapshot.femTopologyNodes[node].identity.w &
                           NM_TOPOLOGY_ACTIVE) != 0u)),
                    "live Matter visual topology references an inactive node"
                );
                referenced[node] = 1u;
            }
            sourceTetrahedra.push_back(nodes);
        }
        require(
            !sourceTetrahedra.empty(),
            "live Matter visual snapshot has no active tetrahedron"
        );
        std::vector<std::uint32_t> remap(
            snapshot.femNodes.size(),
            NM_INVALID_INDEX
        );
        result.tissuePositions.reserve(snapshot.femNodes.size());
        for (std::size_t node = 0u; node < snapshot.femNodes.size(); ++node) {
            if (referenced[node] == 0u) {
                continue;
            }
            const nm_float4 position =
                snapshot.femNodes[node].positionAndMass;
            require(
                std::isfinite(position.x) && std::isfinite(position.y) &&
                    std::isfinite(position.z),
                "live Matter visual FEM node is non-finite"
            );
            remap[node] = static_cast<std::uint32_t>(
                result.tissuePositions.size()
            );
            result.tissuePositions.push_back({
                static_cast<double>(position.x),
                static_cast<double>(position.y),
                static_cast<double>(position.z),
            });
        }
        activeTetrahedra.reserve(sourceTetrahedra.size());
        for (const auto& source : sourceTetrahedra) {
            activeTetrahedra.push_back({
                remap[source[0]],
                remap[source[1]],
                remap[source[2]],
                remap[source[3]],
            });
        }
        // Snapshot FEM positions are already in the live world frame; adding
        // the legacy neutral-zone offset would render a second, false coupon.
        result.tissueTranslationM = {};
    } else {
        const numi::matter::PorcineJejunumClosureCoupon coupon =
            numi::matter::makePorcineJejunumClosureCoupon(0u);
        result.tissuePositions = coupon.object.femNodes;
        activeTetrahedra.reserve(coupon.object.tetrahedra.size());
        for (const numi::matter::TetrahedronSource& tetrahedron :
             coupon.object.tetrahedra) {
            activeTetrahedra.push_back(tetrahedron.nodes);
        }
        // The pre-tissue neutral-zone fixture keeps its authored coupon on the
        // pad beside the pickup. Live v3 tissue checkpoints take the branch
        // above and never receive this presentation-only translation.
        result.tissueTranslationM = {
            kHandoffTissueCenterX,
            0.0,
            0.5 * pad.thicknessM.value +
                0.5 * tissueSpec.thicknessM.value,
        };
    }

    struct BoundaryFace {
        std::array<std::uint32_t, 3> oriented{};
        std::uint32_t count = 0u;
    };
    std::map<std::array<std::uint32_t, 3>, BoundaryFace> faces;
    constexpr std::array<std::array<std::uint32_t, 4>, 4> localFaces{{
        {{1u, 2u, 3u, 0u}},
        {{0u, 3u, 2u, 1u}},
        {{0u, 1u, 3u, 2u}},
        {{0u, 2u, 1u, 3u}},
    }};
    const auto subtract = [](const std::array<double, 3>& left,
                             const std::array<double, 3>& right) {
        return std::array<double, 3>{
            left[0] - right[0],
            left[1] - right[1],
            left[2] - right[2],
        };
    };
    const auto cross3 = [](const std::array<double, 3>& left,
                           const std::array<double, 3>& right) {
        return std::array<double, 3>{
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        };
    };
    const auto dot3 = [](const std::array<double, 3>& left,
                         const std::array<double, 3>& right) {
        return left[0] * right[0] + left[1] * right[1] +
            left[2] * right[2];
    };
    for (const std::array<std::uint32_t, 4>& tetrahedron :
         activeTetrahedra) {
        for (const auto& local : localFaces) {
            std::array<std::uint32_t, 3> oriented{
                tetrahedron[local[0]],
                tetrahedron[local[1]],
                tetrahedron[local[2]],
            };
            const std::uint32_t opposite = tetrahedron[local[3]];
            const auto& a = result.tissuePositions[oriented[0]];
            const auto& b = result.tissuePositions[oriented[1]];
            const auto& c = result.tissuePositions[oriented[2]];
            const auto& d = result.tissuePositions[opposite];
            const double signedSixVolume = dot3(
                cross3(subtract(b, a), subtract(c, a)),
                subtract(d, a)
            );
            require(
                std::isfinite(signedSixVolume) &&
                    std::abs(signedSixVolume) > 1.0e-24,
                "live visual tissue contains a degenerate tetrahedron"
            );
            if (signedSixVolume > 0.0) {
                std::swap(oriented[1], oriented[2]);
            }
            std::array<std::uint32_t, 3> key = oriented;
            std::ranges::sort(key);
            BoundaryFace& face = faces[key];
            if (face.count == 0u) {
                face.oriented = oriented;
            }
            ++face.count;
            require(
                face.count <= 2u,
                "jejunal coupon has a non-manifold tetrahedral face"
            );
        }
    }
    for (const auto& [key, face] : faces) {
        static_cast<void>(key);
        if (face.count == 1u) {
            result.tissueTriangles.push_back(face.oriented);
        }
    }
    require(
        !result.tissueTriangles.empty(),
        "jejunal coupon produced no visual boundary"
    );
    return result;
}

float linearToSrgb(const float value) {
    const float mapped = std::max(value, 0.0f) /
        (1.0f + std::max(value, 0.0f));
    return mapped <= 0.0031308f
        ? 12.92f * mapped
        : 1.055f * std::pow(mapped, 1.0f / 2.4f) - 0.055f;
}

bool writePng(
    const std::filesystem::path& path,
    const std::span<const std::uint8_t> rgba,
    const std::uint32_t width,
    const std::uint32_t height
) {
    if (rgba.size() != static_cast<std::size_t>(width) * height * 4u) {
        return false;
    }
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        kCFAllocatorDefault,
        reinterpret_cast<const UInt8*>(path.c_str()),
        static_cast<CFIndex>(path.string().size()),
        false
    );
    if (url == nullptr) {
        return false;
    }
    CGDataProviderRef provider = CGDataProviderCreateWithData(
        nullptr,
        rgba.data(),
        rgba.size(),
        nullptr
    );
    CGColorSpaceRef colorSpace = CGColorSpaceCreateWithName(
        kCGColorSpaceSRGB
    );
    CGImageRef image = provider != nullptr && colorSpace != nullptr
        ? CGImageCreate(
              width,
              height,
              8u,
              32u,
              static_cast<std::size_t>(width) * 4u,
              colorSpace,
              static_cast<CGBitmapInfo>(
                  static_cast<std::uint32_t>(kCGBitmapByteOrderDefault) |
                  static_cast<std::uint32_t>(kCGImageAlphaLast)
              ),
              provider,
              nullptr,
              false,
              kCGRenderingIntentDefault
          )
        : nullptr;
    CGImageDestinationRef destination = image != nullptr
        ? CGImageDestinationCreateWithURL(
              url,
              CFSTR("public.png"),
              1u,
              nullptr
          )
        : nullptr;
    bool succeeded = false;
    if (destination != nullptr) {
        CGImageDestinationAddImage(destination, image, nullptr);
        succeeded = CGImageDestinationFinalize(destination);
    }
    if (destination != nullptr) {
        CFRelease(destination);
    }
    if (image != nullptr) {
        CGImageRelease(image);
    }
    if (colorSpace != nullptr) {
        CGColorSpaceRelease(colorSpace);
    }
    if (provider != nullptr) {
        CGDataProviderRelease(provider);
    }
    CFRelease(url);
    return succeeded;
}

std::vector<std::uint8_t> colorImage(
    const metalrobo::HybridObservationBatch& observations
) {
    std::vector<std::uint8_t> result(observations.rgb.size() * 4u);
    for (std::size_t pixel = 0u; pixel < observations.rgb.size(); ++pixel) {
        const mr_float4 value = observations.rgb[pixel];
        result[4u * pixel + 0u] = static_cast<std::uint8_t>(
            std::lround(255.0f * std::clamp(linearToSrgb(value.x), 0.0f, 1.0f))
        );
        result[4u * pixel + 1u] = static_cast<std::uint8_t>(
            std::lround(255.0f * std::clamp(linearToSrgb(value.y), 0.0f, 1.0f))
        );
        result[4u * pixel + 2u] = static_cast<std::uint8_t>(
            std::lround(255.0f * std::clamp(linearToSrgb(value.z), 0.0f, 1.0f))
        );
        result[4u * pixel + 3u] = 255u;
    }
    return result;
}

std::vector<std::uint8_t> identityImage(
    const metalrobo::HybridObservationBatch& observations
) {
    std::vector<std::uint8_t> result(observations.segmentation.size() * 4u);
    for (std::size_t pixel = 0u;
         pixel < observations.segmentation.size();
         ++pixel) {
        std::array<std::uint8_t, 3u> color{8u, 10u, 14u};
        switch (observations.segmentation[pixel]) {
        case kInstrumentSemantic:
            color = {210u, 218u, 226u};
            break;
        case kReceiverInstrumentSemantic:
            color = {130u, 178u, 224u};
            break;
        case kNeedleSemantic:
            color = {250u, 190u, 40u};
            break;
        case kThreadSemantic:
            color = {164u, 72u, 216u};
            break;
        case kFieldSemantic:
            color = {20u, 120u, 100u};
            break;
        case kTissueSemantic:
            color = {190u, 74u, 68u};
            break;
        default:
            break;
        }
        result[4u * pixel + 0u] = color[0u];
        result[4u * pixel + 1u] = color[1u];
        result[4u * pixel + 2u] = color[2u];
        result[4u * pixel + 3u] = 255u;
    }
    return result;
}

VisibilityMetrics visibility(
    const metalrobo::HybridObservationBatch& observations
) {
    VisibilityMetrics result;
    for (std::size_t pixel = 0u;
         pixel < observations.segmentation.size();
         ++pixel) {
        switch (observations.segmentation[pixel]) {
        case kInstrumentSemantic:
            ++result.instrumentPixels;
            break;
        case kReceiverInstrumentSemantic:
            ++result.receiverInstrumentPixels;
            break;
        case kNeedleSemantic:
            ++result.needlePixels;
            break;
        case kThreadSemantic:
            ++result.threadPixels;
            break;
        case kFieldSemantic:
            ++result.fieldPixels;
            break;
        case kTissueSemantic:
            ++result.tissuePixels;
            break;
        default:
            break;
        }
        if ((observations.validity[pixel] &
             MR_VISUAL_VALIDITY_GEOMETRY) != 0u &&
            std::isfinite(observations.depth[pixel])) {
            ++result.validPixels;
            result.minimumDepth = std::min(
                result.minimumDepth,
                observations.depth[pixel]
            );
            result.maximumDepth = std::max(
                result.maximumDepth,
                observations.depth[pixel]
            );
        }
    }
    return result;
}

void writeEvidence(
    const std::filesystem::path& path,
    const PickupState& pickup,
    const metalrobo::DvrkSutureVisualAsset& asset,
    const metalrobo::VisualSceneManifestV3& manifest,
    const metalrobo::MetalHybridRendererDiagnostics& render,
    const std::array<VisibilityMetrics, 2u>& views
) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("could not open visual evidence output");
    }
    const std::string_view schema = pickup.dualHandoff
        ? pickup.matterSnapshot != nullptr
            ? "numi.dual-psm-suture-handoff-visual-evidence.v2"
            : "numi.dual-psm-suture-handoff-visual-evidence.v1"
        : pickup.medicallyMatchedSinglePickup
            ? "numi.surgical-suture-visual-evidence.v2"
            : "numi.surgical-suture-visual-evidence.v1";
    const std::string_view physicsBoundary = pickup.dualHandoff
        ? pickup.matterSnapshot != nullptr
            ? "accepted Apple Metal articulated/contact state plus DER thread "
              "and the same checkpoint's verified live FEM nodes and active "
              "topology; the reconstructed surface remains presentation-only"
            : "accepted Apple Metal articulated/contact handoff plus DER "
              "thread; the jejunal surface is generated from the calibrated "
              "FEM rest mesh and remains presentation-only"
        : pickup.medicallyMatchedSinglePickup
            ? "accepted deterministic CPU articulated/contact pickup plus "
              "CPU DER thread; Apple Metal owns this render only; the support "
              "field is presentation geometry derived from the accepted task "
              "frame"
            : "accepted deterministic CPU articulated/contact pickup plus "
              "CPU DER thread; Apple Metal owns this render only";
    output << std::setprecision(12)
        << "{\n"
        << "  \"schema\": \"" << schema << "\",\n"
        << "  \"source\": \"simulation\",\n"
        << "  \"device\": \"" << render.deviceName << "\",\n"
        << "  \"pickup_step\": " << pickup.step << ",\n"
        << "  \"phase\": \""
        << (pickup.dualHandoff ? pickup.phase : "single-pickup")
        << "\",\n"
        << "  \"dual_instrument\": "
        << (pickup.dualHandoff ? "true" : "false") << ",\n"
        << "  \"matter_checkpoint\": "
        << (pickup.matterSnapshotSeen ? "true" : "false") << ",\n";
    if (pickup.matterSnapshotSeen) {
        output
            << "  \"matter_snapshot_file\": \""
            << pickup.matterSnapshotFilename << "\",\n"
            << "  \"matter_snapshot_content_hash\": "
            << pickup.matterSnapshotContentHash << ",\n"
            << "  \"matter_snapshot_payload_bytes\": "
            << pickup.matterSnapshotPayloadBytes << ",\n"
            << "  \"matter_source_physics_fingerprint\": "
            << pickup.matterSourcePhysicsFingerprint << ",\n"
            << "  \"matter_device_program_fingerprint\": "
            << pickup.matterDeviceProgramFingerprint << ",\n";
    }
    if (pickup.knotProtocolSeen) {
        output
            << "  \"knot_protocol_fingerprint\": "
            << pickup.knotProtocolFingerprint << ",\n"
            << "  \"knot_throw_index\": "
            << pickup.knotThrowIndex << ",\n"
            << "  \"knot_completed_sample\": "
            << pickup.knotCompletedSample << ",\n";
    }
    output
        << "  \"tissue_geometry_source\": \""
        << (pickup.matterSnapshot != nullptr
                ? "verified_live_matter_fem_snapshot"
                : pickup.dualHandoff
                    ? "authored_fem_rest_mesh"
                    : "none")
        << "\",\n"
        << "  \"visual_pack_hash\": \"" << asset.pack.contentHash << "\",\n"
        << "  \"visual_scene_fingerprint\": " << manifest.fingerprint << ",\n"
        << "  \"vertices\": " << asset.metrics.vertexCount << ",\n"
        << "  \"triangles\": " << asset.metrics.triangleCount << ",\n"
        << "  \"needle_triangles\": "
        << asset.metrics.needleTriangleCount << ",\n"
        << "  \"thread_triangles\": "
        << asset.metrics.threadTriangleCount << ",\n"
        << "  \"tissue_triangles\": "
        << asset.metrics.tissueTriangleCount << ",\n"
        << "  \"thread_centerline_length_m\": "
        << asset.metrics.threadCenterlineLengthM << ",\n"
        << "  \"render_ms\": " << render.elapsedMilliseconds << ",\n"
        << "  \"views\": [\n";
    for (std::size_t index = 0u; index < views.size(); ++index) {
        const VisibilityMetrics& view = views[index];
        output << "    {\"name\": \""
            << (index == 0u ? "close" : "overview") << "\", "
            << "\"instrument_pixels\": "
            << view.instrumentPixels + view.receiverInstrumentPixels << ", "
            << "\"giver_instrument_pixels\": "
            << view.instrumentPixels << ", "
            << "\"receiver_instrument_pixels\": "
            << view.receiverInstrumentPixels << ", "
            << "\"needle_pixels\": " << view.needlePixels << ", "
            << "\"thread_pixels\": " << view.threadPixels << ", "
            << "\"field_pixels\": " << view.fieldPixels << ", "
            << "\"tissue_pixels\": " << view.tissuePixels << ", "
            << "\"valid_pixels\": " << view.validPixels << ", "
            << "\"minimum_depth_m\": " << view.minimumDepth << ", "
            << "\"maximum_depth_m\": " << view.maximumDepth << "}"
            << (index + 1u == views.size() ? "\n" : ",\n");
    }
    output << "  ],\n"
        << "  \"physics_boundary\": \"" << physicsBoundary << "\",\n"
        << "  \"clinical_validation\": false\n"
        << "}\n";
    output.close();
    require(static_cast<bool>(output), "could not publish visual evidence");
}

} // namespace

int main(const int argumentCount, char* argv[]) {
    @autoreleasepool {
        try {
            const Arguments options = parseArguments(
                argumentCount,
                argv
            );
            std::error_code error;
            std::filesystem::create_directories(
                options.outputDirectory,
                error
            );
            require(!error, "could not create visual output directory");

            const PickupState pickup = readPickupState(options.statePath);
            const metalrobo::CurvedSutureNeedleAsset needle =
                metalrobo::makeCurvedSutureNeedleAsset({
                    .bodyIndex = 0u,
                    .materialIndex = 0u,
                    .slotGenerationBase = 410210u,
                    .collisionGroup = 1u,
                    .collisionMask = ~1u,
                    .motionType = MR_MOTION_DYNAMIC,
                }, pickup.dualHandoff ||
                    pickup.medicallyMatchedSinglePickup
                    ? metalrobo::makeBowelAnastomosisNeedleSpec()
                    : metalrobo::CurvedSutureNeedleSpec{});
            metalrobo::DiscreteRodMaterial visualRodMaterial;
            visualRodMaterial.radius = pickup.threadRadius;
            metalrobo::DiscreteElasticRodModel threadModel =
                metalrobo::makeStraightSutureRod(
                    pickup.threadNodeCount,
                    pickup.threadLength,
                    visualRodMaterial
                );
            metalrobo::DiscreteElasticRodState threadState;
            threadState.positions = pickup.threadPositions;
            threadState.velocities = pickup.threadVelocities;
            threadState.twists.assign(
                pickup.threadNodeCount - 1u,
                0.0
            );
            threadState.twistRates.assign(
                pickup.threadNodeCount - 1u,
                0.0
            );
            metalrobo::DvrkSutureVisualBindings primaryBindings;
            if (pickup.dualHandoff) {
                primaryBindings.needleBodyIndex = 18u;
            }
            const metalrobo::DvrkSutureVisualScene visualScene =
                makeHandoffVisualScene(pickup);
            const metalrobo::DvrkSutureVisualAsset visualAsset =
                metalrobo::makeDvrkSutureVisualAsset(
                    needle,
                    threadModel,
                    threadState,
                    primaryBindings,
                    {},
                    visualScene
                );
            if (options.geometryOnly) {
                std::cout << std::setprecision(12)
                    << "suture_visual_geometry status=ok"
                    << " phase=\""
                    << (pickup.dualHandoff
                            ? pickup.phase
                            : "single-pickup")
                    << "\" vertices=" << visualAsset.metrics.vertexCount
                    << " triangles=" << visualAsset.metrics.triangleCount
                    << " needle_triangles="
                    << visualAsset.metrics.needleTriangleCount
                    << " thread_triangles="
                    << visualAsset.metrics.threadTriangleCount
                    << " tissue_triangles="
                    << visualAsset.metrics.tissueTriangleCount
                    << " tissue_geometry_source="
                    << (pickup.matterSnapshot != nullptr
                            ? "verified_live_matter_fem_snapshot"
                            : pickup.dualHandoff
                                ? "authored_fem_rest_mesh"
                                : "none")
                    << " matter_content_hash="
                    << pickup.matterSnapshotContentHash
                    << " pack_hash=" << visualAsset.pack.contentHash
                    << " gpu_dispatched=no\n";
                return 0;
            }
            const std::filesystem::path packPath =
                options.outputDirectory /
                (pickup.dualHandoff
                    ? "dual-dvrk-suture-handoff.mrvpack"
                    : pickup.medicallyMatchedSinglePickup
                        ? "dvrk-pdsii-3-0-bowel-suture-pickup.mrvpack"
                        : "dvrk-gs21-suture-pickup.mrvpack");
            std::string reason;
            require(
                metalrobo::writeVisualAssetPack(
                    visualAsset.pack,
                    packPath,
                    &reason
                ),
                "could not write surgical visual pack: " + reason
            );

            metalrobo::EngineModel model = makeVisualEngineModel(
                needle,
                pickup
            );
            const float inverseMass = needle.rigid.body.massAndInverseMass.y;
            PickupState livePickup = pickup;
            livePickup.needle.linearVelocityAndInverseMass.w = inverseMass;
            assignRotatedInverseInertia(
                livePickup.needle,
                needle.rigid.body
            );
            livePickup.needle.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
            livePickup.needle.flagsAndIndices[1] = MR_INVALID_INDEX;
            livePickup.needle.flagsAndIndices[2] = 41021u;

            const metalrobo::WorldTemplate world = makeWorldTemplate(
                model,
                livePickup,
                visualScene
            );
            const metalrobo::WorldFamily family = makeWorldFamily(world);
            metalrobo::MetalWorldFamilyContext worlds;
            requireSucceeded(
                worlds.compile(family, 1u),
                "Metal world family compile"
            );
            requireSucceeded(
                worlds.sample(1u, 0x535554555245ull),
                "Metal world sample"
            );

            const std::array references{
                metalrobo::VisualAssetReferenceV3{
                    packPath,
                    visualAsset.pack.contentHash,
                    0u,
                    kInstrumentSemantic,
                    1000u,
                },
            };
            metalrobo::VisualSceneManifestV3 manifest;
            require(
                metalrobo::compileVisualSceneManifestV3(
                    world,
                    references,
                    metalrobo::makeNeutralStudioEnvironmentV2(),
                    makeSurgicalThreePointLightRig(),
                    manifest,
                    &reason
                ),
                "surgical visual scene compile: " + reason
            );
            const std::filesystem::path manifestPath =
                options.outputDirectory /
                (pickup.dualHandoff
                    ? "dual-dvrk-suture-handoff.visual.v3.json"
                    : pickup.medicallyMatchedSinglePickup
                        ? "dvrk-pdsii-3-0-bowel-suture-pickup.visual.v3.json"
                        : "dvrk-gs21-suture-pickup.visual.v3.json");
            require(
                metalrobo::writeVisualSceneManifestV3(
                    manifest,
                    manifestPath,
                    &reason
                ),
                "surgical visual manifest write: " + reason
            );

            metalrobo::MetalHybridRendererConfig rendererConfig;
            rendererConfig.width = kWidth;
            rendererConfig.height = kHeight;
            rendererConfig.maximumReferenceFramesInFlight = 2u;
            rendererConfig.clearColorAndDepth = {
                0.003f,
                0.008f,
                0.012f,
                1.0e30f,
            };
            metalrobo::MetalHybridRenderer renderer(rendererConfig);
            const metalrobo::VisualSceneManifestV3 evidenceManifest = manifest;
            requireSucceeded(
                renderer.compile(
                    std::move(manifest.renderScene),
                    metalrobo::VisualRendererProfileV1::sensorReference(),
                    1u
                ),
                "surgical reference renderer compile"
            );
            std::vector<MRBodyStateGPU> bodyStates;
            require(
                metalrobo::composeVisualBodyStates(
                    model,
                    1u,
                    livePickup.q,
                    livePickup.v,
                    std::span<const MRBodyStateGPU>{&livePickup.needle, 1u},
                    bodyStates,
                    &reason
                ),
                "surgical visual body composition: " + reason
            );
            const double captureTimestamp =
                livePickup.step * livePickup.controlTimestepSeconds;
            constexpr double exposureSeconds = 1.0 / 240.0;
            metalrobo::VisualMotionSampleBatchV1 motion;
            motion.environmentCount = 1u;
            motion.bodyCount = static_cast<std::uint32_t>(
                model.bodies.size()
            );
            motion.sampleCount = 2u;
            motion.exposureOpenSeconds =
                captureTimestamp - 0.5 * exposureSeconds;
            motion.exposureCloseSeconds =
                captureTimestamp + 0.5 * exposureSeconds;
            motion.timestampsSeconds = {
                motion.exposureOpenSeconds,
                motion.exposureCloseSeconds,
            };
            motion.bodyStates.reserve(2u * bodyStates.size());
            motion.bodyStates.insert(
                motion.bodyStates.end(),
                bodyStates.begin(),
                bodyStates.end()
            );
            motion.bodyStates.insert(
                motion.bodyStates.end(),
                bodyStates.begin(),
                bodyStates.end()
            );
            motion.scenarioIdentity = 0x535554555245ull;
            motion.frameIndex = livePickup.step;
            motion.source = MR_VISUAL_SOURCE_SIMULATION;

            const std::array<std::string, 2u> names{
                pickup.dualHandoff ? "handoff-close" : "pickup-close",
                pickup.dualHandoff ? "handoff-overview" : "pickup-overview",
            };
            const std::array<std::array<std::size_t, 7u>, 2u>
                minimumCoverage = pickup.dualHandoff
                    ? std::array<std::array<std::size_t, 7u>, 2u>{{
                          {{15000u, 15000u, 1200u, 1000u, 5000u, 500u,
                            100000u}},
                          {{2500u, 2500u, 250u, 400u, 1000u, 200u,
                            100000u}},
                      }}
                    : std::array<std::array<std::size_t, 7u>, 2u>{{
                          {{30000u, 0u, 1800u, 1000u,
                            pickup.medicallyMatchedSinglePickup ? 5000u : 0u,
                            pickup.medicallyMatchedSinglePickup ? 500u : 0u,
                            100000u}},
                          {{5000u, 0u, 250u, 400u,
                            pickup.medicallyMatchedSinglePickup ? 1000u : 0u,
                            pickup.medicallyMatchedSinglePickup ? 200u : 0u,
                            100000u}},
                      }};
            std::array<VisibilityMetrics, 2u> viewMetrics{};
            metalrobo::MetalHybridRendererDiagnostics lastRender;
            for (std::uint32_t camera = 0u; camera < names.size(); ++camera) {
                motion.sensorIdentity = camera + 1u;
                motion.sensorSequence = camera + 1u;
                lastRender = renderer.renderFrame(worlds, motion, camera);
                requireSucceeded(lastRender, "surgical reference render");
                metalrobo::HybridObservationBatch observations;
                requireSucceeded(
                    renderer.readback(observations),
                    "surgical render readback"
                );
                viewMetrics[camera] = visibility(observations);
                require(
                    writePng(
                        options.outputDirectory /
                            (names[camera] + ".png"),
                        colorImage(observations),
                        kWidth,
                        kHeight
                    ),
                    "could not write surgical color PNG"
                );
                require(
                    writePng(
                        options.outputDirectory /
                            (names[camera] + "-identity.png"),
                        identityImage(observations),
                        kWidth,
                        kHeight
                    ),
                    "could not write surgical identity PNG"
                );
            }
            for (std::uint32_t camera = 0u;
                 camera < names.size();
                 ++camera) {
                require(
                    viewMetrics[camera].instrumentPixels >=
                            minimumCoverage[camera][0u] &&
                        (
                            !pickup.dualHandoff ||
                            viewMetrics[camera]
                                    .receiverInstrumentPixels >=
                                minimumCoverage[camera][1u]
                        ) &&
                        viewMetrics[camera].needlePixels >=
                            minimumCoverage[camera][2u] &&
                        viewMetrics[camera].threadPixels >=
                            minimumCoverage[camera][3u] &&
                        viewMetrics[camera].fieldPixels >=
                            minimumCoverage[camera][4u] &&
                        viewMetrics[camera].tissuePixels >=
                            minimumCoverage[camera][5u] &&
                        viewMetrics[camera].validPixels >=
                            minimumCoverage[camera][6u],
                    std::string{"visual binding coverage failed for "} +
                        names[camera] +
                        " instrument=" +
                        std::to_string(viewMetrics[camera].instrumentPixels) +
                        " receiver_instrument=" +
                        std::to_string(
                            viewMetrics[camera].receiverInstrumentPixels
                        ) +
                        " needle=" +
                        std::to_string(viewMetrics[camera].needlePixels) +
                        " thread=" +
                        std::to_string(viewMetrics[camera].threadPixels) +
                        " field=" +
                        std::to_string(viewMetrics[camera].fieldPixels) +
                        " tissue=" +
                        std::to_string(viewMetrics[camera].tissuePixels) +
                        " valid=" +
                        std::to_string(viewMetrics[camera].validPixels)
                );
            }
            const std::filesystem::path evidencePath =
                options.outputDirectory / "visual-evidence.json";
            writeEvidence(
                evidencePath,
                livePickup,
                visualAsset,
                evidenceManifest,
                lastRender,
                viewMetrics
            );
            std::cout << std::setprecision(12)
                << "suture_visual status=ok"
                << " device=\"" << lastRender.deviceName << "\""
                << " resolution=" << kWidth << "x" << kHeight
                << " vertices=" << visualAsset.metrics.vertexCount
                << " triangles=" << visualAsset.metrics.triangleCount
                << " needle_triangles="
                << visualAsset.metrics.needleTriangleCount
                << " thread_triangles="
                << visualAsset.metrics.threadTriangleCount
                << " tissue_triangles="
                << visualAsset.metrics.tissueTriangleCount
                << " tissue_geometry_source="
                << (livePickup.matterSnapshot != nullptr
                        ? "verified_live_matter_fem_snapshot"
                        : livePickup.dualHandoff
                            ? "authored_fem_rest_mesh"
                            : "none")
                << " close_pixels="
                << viewMetrics[0u].instrumentPixels << "/"
                << viewMetrics[0u].receiverInstrumentPixels << "/"
                << viewMetrics[0u].needlePixels << "/"
                << viewMetrics[0u].threadPixels << "/"
                << viewMetrics[0u].fieldPixels << "/"
                << viewMetrics[0u].tissuePixels
                << " overview_pixels="
                << viewMetrics[1u].instrumentPixels << "/"
                << viewMetrics[1u].receiverInstrumentPixels << "/"
                << viewMetrics[1u].needlePixels << "/"
                << viewMetrics[1u].threadPixels << "/"
                << viewMetrics[1u].fieldPixels << "/"
                << viewMetrics[1u].tissuePixels
                << " pack_hash=" << visualAsset.pack.contentHash
                << " scene_fingerprint=" << evidenceManifest.fingerprint
                << " source=simulation clinical_validation=no\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "suture_visual status=failed reason=\""
                << exception.what() << "\"\n";
            return 1;
        }
    }
}

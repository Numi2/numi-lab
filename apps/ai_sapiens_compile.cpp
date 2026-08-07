#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/RobotDescriptionCooker.hpp"

#include <libxml/parser.h>
#include <libxml/tree.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kSourceRepository =
    "https://github.com/ROBOTIS-GIT/ai_sapiens";
constexpr std::string_view kSourceRevision =
    "c2880e89fb3451a07b6d2600e274224ffcf912e4";

constexpr std::array<std::string_view, 23u> kPolicyJoints{{
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_elbow_joint", "right_elbow_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
}};

constexpr std::array<float, 23u> kVelocityOffsets{{
    -0.205F, -0.205F, 0.0F, -0.0106F, 0.0106F, 0.218F, 0.218F,
    0.00225F, -0.00225F, 0.315F, -0.315F, 0.517F, 0.517F,
    -0.0695F, 0.0695F, -0.307F, -0.307F, 1.08F, 1.08F,
    0.0108F, -0.0108F, 0.00186F, -0.00186F,
}};

constexpr std::array<std::string_view, 23u> kMotionJoints{{
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint",
}};

constexpr std::array<float, 23u> kMimicOffsets{{
    -0.29F, -0.307F, 0.00124F, 0.000442F, 0.00489F, 0.202F, 0.209F,
    0.00796F, 0.00546F, 0.203F, -0.199F, 0.63F, 0.632F, -0.00489F,
    0.00793F, -0.336F, -0.326F, 0.594F, 0.61F, 0.00258F, -0.00029F,
    0.00475F, -0.00448F,
}};

struct Arguments {
    std::filesystem::path source;
    std::filesystem::path output;
};

[[nodiscard]] Arguments arguments(const int argc, const char* const* argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option{argv[index]};
        if (option == "--source" || option == "--output") {
            if (++index == argc) {
                throw std::invalid_argument("missing value for " +
                    std::string{option});
            }
            if (option == "--source") {
                result.source = argv[index];
            } else {
                result.output = argv[index];
            }
        } else if (option == "--help" || option == "-h") {
            std::cout << "usage: metalrobo_ai_sapiens_compile "
                         "--source AI_SAPIENS --output ARTIFACT_DIRECTORY\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option " +
                std::string{option});
        }
    }
    if (result.source.empty() || result.output.empty()) {
        throw std::invalid_argument("--source and --output are required");
    }
    return result;
}

[[nodiscard]] std::filesystem::path urdfPath(
    const std::filesystem::path& source
) {
    const auto result = source / "ai_sapiens_description" /
        "urdf" / "k1_rev1" / "k1.urdf";
    if (!std::filesystem::is_regular_file(result)) {
        throw std::invalid_argument("official K1 URDF is missing: " +
            result.string());
    }
    return result;
}

[[nodiscard]] std::filesystem::path writeCollisionURDF(
    const std::filesystem::path& source,
    const std::filesystem::path& output
) {
    const std::filesystem::path target = output / "k1-collision.urdf";
    xmlDocPtr document = xmlReadFile(
        urdfPath(source).c_str(), nullptr, XML_PARSE_NONET);
    if (document == nullptr) {
        throw std::runtime_error("could not parse official K1 URDF");
    }
    std::size_t removedMeshes = 0u;
    for (xmlNodePtr link = xmlDocGetRootElement(document)->children;
         link != nullptr;
         link = link->next) {
        if (link->type != XML_ELEMENT_NODE ||
            xmlStrcmp(link->name, BAD_CAST "link") != 0) {
            continue;
        }
        for (xmlNodePtr child = link->children, next = nullptr;
             child != nullptr;
             child = next) {
            next = child->next;
            if (child->type != XML_ELEMENT_NODE ||
                xmlStrcmp(child->name, BAD_CAST "collision") != 0) {
                continue;
            }
            bool mesh = false;
            for (xmlNodePtr geometry = child->children;
                 geometry != nullptr && !mesh;
                 geometry = geometry->next) {
                if (geometry->type != XML_ELEMENT_NODE ||
                    xmlStrcmp(geometry->name, BAD_CAST "geometry") != 0) {
                    continue;
                }
                for (xmlNodePtr shape = geometry->children;
                     shape != nullptr;
                     shape = shape->next) {
                    if (shape->type == XML_ELEMENT_NODE &&
                        xmlStrcmp(shape->name, BAD_CAST "mesh") == 0) {
                        mesh = true;
                        break;
                    }
                }
            }
            if (mesh) {
                xmlUnlinkNode(child);
                xmlFreeNode(child);
                ++removedMeshes;
            }
        }
    }
    const int saved = xmlSaveFormatFileEnc(
        target.c_str(), document, "UTF-8", 1);
    xmlFreeDoc(document);
    if (removedMeshes != 3u || saved < 0) {
        throw std::runtime_error(
            "official K1 collision normalization did not remove the three "
            "high-detail mesh colliders");
    }
    return target;
}

void requireOfficialAssets(const std::filesystem::path& source) {
    const std::array<std::filesystem::path, 9u> required{{
        urdfPath(source),
        source / "ai_sapiens_sim2real" / "config" / "k1_config.yaml",
        source / "ai_sapiens_sim2real" / "assets" / "k1" /
            "locomotion" / "velocity" / "walk_default" /
            "exported" / "policy.onnx",
        source / "ai_sapiens_sim2real" / "assets" / "k1" /
            "mimic" / "squat" / "exported" / "policy.onnx",
        source / "ai_sapiens_sim2real" / "assets" / "k1" /
            "mimic" / "dance1" / "exported" / "policy.onnx",
        source / "ai_sapiens_sim2real" / "assets" / "k1" /
            "mimic" / "dance2" / "exported" / "policy.onnx",
        source / "ai_sapiens_sim2real" / "assets" / "k1" /
            "mimic" / "squat" / "params" / "squat.csv",
        source / "ai_sapiens_sim2real" / "assets" / "k1" /
            "mimic" / "dance1" / "params" / "dance1.csv",
        source / "ai_sapiens_sim2real" / "assets" / "k1" /
            "mimic" / "dance2" / "params" / "dance2.csv",
    }};
    for (const auto& path : required) {
        if (!std::filesystem::is_regular_file(path)) {
            throw std::invalid_argument("official AI Sapiens asset is missing: " +
                path.string());
        }
    }
}

[[nodiscard]] metalrobo::TaskObservationOperatorSpec observation(
    const metalrobo::TaskObservationSource source,
    const std::string_view target,
    const std::uint32_t component,
    const float scale = 1.0F,
    const float offset = 0.0F
) {
    return {
        .source = source,
        .target = std::string{target},
        .component = component,
        .scale = scale,
        .offset = offset,
    };
}

[[nodiscard]] metalrobo::SensorProgramPack velocitySensors() {
    metalrobo::SensorProgramPack result;
    result.id = "robotis_ai_sapiens_k1_velocity_sensor_v1";
    auto& program = result.observation;
    program.actorHistoryLength = 5u;
    program.criticHistoryLength = 5u;
    program.criticIncludesCleanHistory = true;
    for (std::uint32_t component = 0u; component < 3u; ++component) {
        program.actorFrame.push_back(observation(
            metalrobo::TaskObservationSource::rootAngularVelocityLocal,
            {}, component, 0.2F));
    }
    for (std::uint32_t component = 0u; component < 3u; ++component) {
        program.actorFrame.push_back(observation(
            metalrobo::TaskObservationSource::projectedGravity,
            {}, component));
    }
    for (std::uint32_t component = 0u; component < 3u; ++component) {
        program.actorFrame.push_back(observation(
            metalrobo::TaskObservationSource::command,
            {}, component));
    }
    for (std::size_t index = 0u; index < kPolicyJoints.size(); ++index) {
        program.actorFrame.push_back(observation(
            metalrobo::TaskObservationSource::jointPositionError,
            kPolicyJoints[index], 0u, 1.0F, -kVelocityOffsets[index]));
    }
    for (const std::string_view joint : kPolicyJoints) {
        program.actorFrame.push_back(observation(
            metalrobo::TaskObservationSource::jointVelocity,
            joint, 0u, 0.05F));
    }
    for (const std::string_view joint : kPolicyJoints) {
        program.actorFrame.push_back(observation(
            metalrobo::TaskObservationSource::previousAction,
            joint, 0u));
    }
    program.critic = program.actorFrame;
    return result;
}

[[nodiscard]] metalrobo::SensorProgramPack mimicSensors() {
    metalrobo::SensorProgramPack result;
    result.id = "robotis_ai_sapiens_k1_mimic_sensor_v1";
    auto& program = result.observation;
    program.actorHistoryLength = 1u;
    program.criticHistoryLength = 1u;
    program.criticIncludesCleanHistory = true;
    for (const std::string_view joint : kPolicyJoints) {
        program.actorFrame.push_back(observation(
            metalrobo::TaskObservationSource::interactionJointTarget,
            joint, 0u));
    }
    for (const std::string_view joint : kPolicyJoints) {
        program.actorFrame.push_back(observation(
            metalrobo::TaskObservationSource::interactionJointTargetVelocity,
            joint, 0u));
    }
    for (std::uint32_t component = 0u; component < 6u; ++component) {
        program.actorFrame.push_back(observation(
            metalrobo::TaskObservationSource::interactionAnchorOrientation,
            "waist_yaw_joint", component));
    }
    for (std::uint32_t component = 0u; component < 3u; ++component) {
        program.actorFrame.push_back(observation(
            metalrobo::TaskObservationSource::rootAngularVelocityLocal,
            {}, component));
    }
    for (std::size_t index = 0u; index < kPolicyJoints.size(); ++index) {
        program.actorFrame.push_back(observation(
            metalrobo::TaskObservationSource::jointPositionError,
            kPolicyJoints[index], 0u, 1.0F, -kMimicOffsets[index]));
    }
    for (const std::string_view joint : kPolicyJoints) {
        program.actorFrame.push_back(observation(
            metalrobo::TaskObservationSource::jointVelocity, joint, 0u));
    }
    for (const std::string_view joint : kPolicyJoints) {
        program.actorFrame.push_back(observation(
            metalrobo::TaskObservationSource::previousAction, joint, 0u));
    }
    program.critic = program.actorFrame;
    if (program.actorFrame.size() != 124u) {
        throw std::logic_error("K1 mimic observation ABI is not 124");
    }
    return result;
}

[[nodiscard]] metalrobo::RobotActuatorPack velocityActuators() {
    metalrobo::RobotActuatorPack result;
    result.id = "robotis_ai_sapiens_k1_velocity_actuators_v1";
    for (const std::string_view joint : kPolicyJoints) {
        result.actuators.push_back({
            .id = std::string{joint},
            .kind = metalrobo::RobotActuatorKind::jointPosition,
            .target = std::string{joint},
            .scale = 1.0F,
        });
    }
    return result;
}

[[nodiscard]] metalrobo::RobotActuatorPack mimicActuators() {
    metalrobo::RobotActuatorPack result;
    result.id = "robotis_ai_sapiens_k1_mimic_actuators_v1";
    for (std::size_t index = 0u; index < kPolicyJoints.size(); ++index) {
        const bool leg = index == 0u || index == 1u || index == 2u ||
            index == 3u || index == 4u || index == 7u || index == 8u ||
            index == 11u || index == 12u;
        const bool ankle = index == 15u || index == 16u ||
            index == 19u || index == 20u;
        const float stiffness = leg ? 76.5F : ankle ? 44.6F : 22.3F;
        const float damping = leg ? 4.87F : ankle ? 2.84F : 1.42F;
        result.actuators.push_back({
            .id = std::string{kPolicyJoints[index]},
            .kind = metalrobo::RobotActuatorKind::jointPosition,
            .target = std::string{kPolicyJoints[index]},
            .scale = 1.0F,
            .parameters = {stiffness, damping, 0.0F, 0.0F},
        });
    }
    return result;
}

[[nodiscard]] metalrobo::TaskPack velocityTask() {
    metalrobo::TaskPack result;
    result.id = "robotis_ai_sapiens_k1_velocity_v1";
    result.maximumEpisodeSteps = 1'000u;
    result.difficultyBandCount = 1u;
    result.baseHeightTarget = 0.78F;
    result.successTrackingThreshold = 0.8F;
    result.commands.lower = {0.0F, 0.0F, 0.0F, 0.0F};
    result.commands.upper = {0.0F, 0.0F, 0.0F, 0.0F};
    result.commands.limitLower = {-0.5F, -0.3F, -1.0F, 0.0F};
    result.commands.limitUpper = {1.0F, 0.3F, 1.0F, 0.0F};
    result.commands.minimumDurationSeconds = 10.0F;
    result.commands.maximumDurationSeconds = 10.0F;
    result.outcomes = {
        {"tracking", "ratio", metalrobo::TaskOutcomeSource::trackingScore,
            metalrobo::TaskOutcomeDirection::higherIsBetter},
        {"root_height", "m", metalrobo::TaskOutcomeSource::rootHeight,
            metalrobo::TaskOutcomeDirection::neutral},
        {"tilt", "rad", metalrobo::TaskOutcomeSource::tilt,
            metalrobo::TaskOutcomeDirection::lowerIsBetter},
    };
    for (const std::string_view joint : kPolicyJoints) {
        result.actions.push_back({.actuator = std::string{joint}});
    }
    result.contactGroups = {
        {.id = "left_foot", .bodies = {"left_ankle_roll_link"},
            .support = true, .referenceBody = "left_ankle_roll_link"},
        {.id = "right_foot", .bodies = {"right_ankle_roll_link"},
            .support = true, .referenceBody = "right_ankle_roll_link"},
    };
    result.rewards = {
        {.operation = metalrobo::TaskRewardOperator::uprightness,
            .weight = 0.5F},
        {.operation = metalrobo::TaskRewardOperator::rootHeightErrorSquared,
            .weight = -1.0F, .parameters = {0.78F, 0.0F, 0.0F, 0.0F}},
    };
    result.terminations = {
        {.operation = metalrobo::TaskTerminationOperator::minimumRootHeight,
            .reason = MR_TASK_TERMINATION_HEIGHT, .priority = 0u,
            .threshold = 0.38F, .failurePenalty = -2.0F},
        {.operation = metalrobo::TaskTerminationOperator::maximumTilt,
            .reason = MR_TASK_TERMINATION_TILT, .priority = 1u,
            .threshold = 1.2F, .failurePenalty = -2.0F},
    };
    return result;
}

[[nodiscard]] metalrobo::RealityProgramPack velocityReality() {
    metalrobo::RealityProgramPack result;
    result.id = "robotis_ai_sapiens_k1_velocity_reality_v1";
    result.program.id = "robotis_ai_sapiens_k1_nominal_world_v1";
    result.reset.operators.push_back({
        .operation = metalrobo::TaskRandomizationOperator::rootHeight,
        .parameters = {0.78F, 0.78F, 0.0F, 0.0F},
    });
    for (std::size_t index = 0u; index < kPolicyJoints.size(); ++index) {
        result.reset.operators.push_back({
            .operation = metalrobo::TaskRandomizationOperator::jointPosition,
            .target = std::string{kPolicyJoints[index]},
            .parameters = {
                kVelocityOffsets[index], kVelocityOffsets[index], 0.0F, 0.0F,
            },
        });
    }
    return result;
}

[[nodiscard]] metalrobo::TaskPack mimicTask() {
    metalrobo::TaskPack result;
    result.id = "robotis_ai_sapiens_k1_mimic_v1";
    result.maximumEpisodeSteps = 300u;
    result.difficultyBandCount = 1u;
    result.interactionControlReference = false;
    result.outcomes = {
        {"interaction_tracking", "ratio",
            metalrobo::TaskOutcomeSource::trackingScore,
            metalrobo::TaskOutcomeDirection::higherIsBetter},
        {"root_height", "m", metalrobo::TaskOutcomeSource::rootHeight,
            metalrobo::TaskOutcomeDirection::neutral},
        {"tilt", "rad", metalrobo::TaskOutcomeSource::tilt,
            metalrobo::TaskOutcomeDirection::lowerIsBetter},
    };
    for (const std::string_view joint : kPolicyJoints) {
        result.actions.push_back({.actuator = std::string{joint}});
    }
    result.contactGroups = {
        {.id = "left_foot", .bodies = {"left_ankle_roll_link"},
            .support = true, .referenceBody = "left_ankle_roll_link"},
        {.id = "right_foot", .bodies = {"right_ankle_roll_link"},
            .support = true, .referenceBody = "right_ankle_roll_link"},
    };
    result.rewards = {
        {.operation = metalrobo::TaskRewardOperator::interactionJointTracking,
            .weight = 1.0F, .parameters = {0.5F, 0.25F, 0.0F, 0.0F}},
        {.operation = metalrobo::TaskRewardOperator::interactionRootTracking,
            .weight = 1.0F, .parameters = {1.0F, 0.25F, 1.0F, 1.0F}},
        {.operation = metalrobo::TaskRewardOperator::jointVelocitySquared,
            .weight = -0.001F},
        {.operation = metalrobo::TaskRewardOperator::actionRateSquared,
            .weight = -0.01F},
    };
    result.terminations = {
        {.operation = metalrobo::TaskTerminationOperator::minimumRootHeight,
            .reason = MR_TASK_TERMINATION_HEIGHT, .priority = 0u,
            .threshold = 0.25F, .failurePenalty = -2.0F},
        {.operation = metalrobo::TaskTerminationOperator::maximumTilt,
            .reason = MR_TASK_TERMINATION_TILT, .priority = 1u,
            .threshold = 1.5F, .failurePenalty = -2.0F},
    };
    return result;
}

[[nodiscard]] metalrobo::RealityProgramPack mimicReality() {
    metalrobo::RealityProgramPack result;
    result.id = "robotis_ai_sapiens_k1_mimic_reality_v1";
    result.program.id = "robotis_ai_sapiens_k1_mimic_world_v1";
    return result;
}

[[nodiscard]] std::vector<float> csvRow(
    const std::string& line,
    const std::filesystem::path& path
) {
    std::vector<float> values;
    std::stringstream stream{line};
    std::string value;
    while (std::getline(stream, value, ',')) {
        std::size_t consumed = 0u;
        const float parsed = std::stof(value, &consumed);
        if (consumed != value.size() || !std::isfinite(parsed)) {
            throw std::invalid_argument("invalid K1 motion CSV value: " +
                path.string());
        }
        values.push_back(parsed);
    }
    if (values.size() != 30u) {
        throw std::invalid_argument("K1 motion CSV row must contain 30 fields: " +
            path.string());
    }
    return values;
}

[[nodiscard]] metalrobo::InteractionPack mimicInteraction(
    const std::filesystem::path& source,
    const std::string_view id
) {
    const std::filesystem::path csv = source / "ai_sapiens_sim2real" /
        "assets" / "k1" / "mimic" / id / "params" /
        (std::string{id} + ".csv");
    std::ifstream input{csv};
    if (!input) {
        throw std::invalid_argument("official K1 mimic CSV is missing: " +
            csv.string());
    }
    metalrobo::InteractionPack result{
        .id = "robotis_ai_sapiens_k1_" + std::string{id},
        .sourceRepository = std::string{kSourceRepository},
        .sourceRevision = std::string{kSourceRevision},
        .license = "Apache-2.0",
        .coordinateFrame = metalrobo::kInteractionCoordinateFrame,
    };
    for (const std::string_view joint : kMotionJoints) {
        result.jointNames.push_back(std::string{joint});
    }
    metalrobo::InteractionClip clip{
        .id = std::string{id},
        .desiredOutcome = "ROBOTIS AI Sapiens K1 source mimic " +
            std::string{id},
        .framesPerSecond = 50.0F,
        .loop = false,
    };
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<float> row = csvRow(line, csv);
        clip.rootTargets.insert(clip.rootTargets.end(), row.begin(), row.begin() + 7);
        clip.jointTargets.insert(clip.jointTargets.end(), row.begin() + 7, row.end());
        ++clip.frameCount;
    }
    if (clip.frameCount < 2u) {
        throw std::invalid_argument("K1 mimic CSV has fewer than two frames: " +
            csv.string());
    }
    result.clips.push_back(std::move(clip));
    return result;
}

void writeOrThrow(
    const metalrobo::LearningPackResult result,
    const std::filesystem::path& target
) {
    if (!result.succeeded()) {
        throw std::runtime_error("could not write " + target.string() + ": " +
            result.message);
    }
}

} // namespace

int main(const int argc, const char* const* argv) {
    try {
        const Arguments options = arguments(argc, argv);
        requireOfficialAssets(options.source);

        std::filesystem::create_directories(options.output);
        const std::filesystem::path collisionURDF = writeCollisionURDF(
            options.source, options.output);
        metalrobo::EngineModel cooked;
        metalrobo::RobotDescriptionCookOptions cookOptions;
        cookOptions.rootMode = metalrobo::RobotDescriptionRootMode::floating;
        cookOptions.meshMode =
            metalrobo::RobotDescriptionMeshMode::requireConvexSurface;
        cookOptions.packageSearchRoots = {options.source};
        const auto cookedResult = metalrobo::cookRobotDescriptionFiles(
            collisionURDF, {}, cooked, cookOptions);
        if (!cookedResult.succeeded() || cookedResult.dofCount != 29u) {
            throw std::runtime_error("official K1 URDF cook contract failed [" +
                std::string{metalrobo::robotDescriptionStatusName(
                    cookedResult.status)} + "] " + cookedResult.element +
                ": " + cookedResult.message + " joints=" +
                std::to_string(cookedResult.jointCount) + " dofs=" +
                std::to_string(cookedResult.dofCount));
        }
        for (const std::string_view joint : kPolicyJoints) {
            if (std::ranges::find(cookedResult.jointNames, joint) ==
                cookedResult.jointNames.end()) {
                throw std::runtime_error("official K1 joint contract changed: " +
                    std::string{joint});
            }
        }

        const auto task = options.output / "k1-velocity.taskpack";
        const auto actuators = options.output / "k1-velocity.actuatorpack";
        const auto sensors = options.output / "k1-velocity.sensorpack";
        const auto reality = options.output / "k1-velocity.realitypack";
        const auto mimicTaskPath = options.output / "k1-mimic.taskpack";
        const auto mimicActuatorsPath = options.output / "k1-mimic.actuatorpack";
        const auto mimicSensorsPath = options.output / "k1-mimic.sensorpack";
        const auto mimicRealityPath = options.output / "k1-mimic.realitypack";
        writeOrThrow(metalrobo::writeTaskPack(velocityTask(), task), task);
        writeOrThrow(metalrobo::writeRobotActuatorPack(
            velocityActuators(), actuators), actuators);
        writeOrThrow(metalrobo::writeSensorProgramPack(
            velocitySensors(), sensors), sensors);
        writeOrThrow(metalrobo::writeRealityProgramPack(
            velocityReality(), reality), reality);
        writeOrThrow(metalrobo::writeTaskPack(mimicTask(), mimicTaskPath),
            mimicTaskPath);
        writeOrThrow(metalrobo::writeRobotActuatorPack(mimicActuators(),
            mimicActuatorsPath), mimicActuatorsPath);
        writeOrThrow(metalrobo::writeSensorProgramPack(mimicSensors(),
            mimicSensorsPath), mimicSensorsPath);
        writeOrThrow(metalrobo::writeRealityProgramPack(mimicReality(),
            mimicRealityPath), mimicRealityPath);
        for (const std::string_view id : {"squat", "dance1", "dance2"}) {
            const auto interaction = options.output /
                ("k1-" + std::string{id} + ".interactionpack");
            writeOrThrow(metalrobo::writeInteractionPack(
                mimicInteraction(options.source, id), interaction), interaction);
        }

        std::ofstream manifest{options.output / "source.json"};
        manifest << "{\n"
                 << "  \"schema\": \"numi.ai-sapiens-k1.v1\",\n"
                 << "  \"source_repository\": \"" << kSourceRepository << "\",\n"
                 << "  \"expected_source_revision\": \"" << kSourceRevision << "\",\n"
                 << "  \"source_urdf\": \""
                 << urdfPath(options.source).string() << "\",\n"
                 << "  \"collision_urdf\": \""
                 << collisionURDF.string() << "\",\n"
                 << "  \"urdf_fingerprint\": " << cookedResult.sourceFingerprint << ",\n"
                 << "  \"joint_count\": " << cookedResult.jointCount << ",\n"
                 << "  \"dof_count\": " << cookedResult.dofCount << ",\n"
                 << "  \"velocity_actor_observations\": 390,\n"
                 << "  \"velocity_actions\": 23,\n"
                 << "  \"mimic_actor_observations\": 124,\n"
                 << "  \"mimic_actions\": 23,\n"
                 << "  \"mimic_clips\": [\"squat\", \"dance1\", \"dance2\"]\n"
                 << "}\n";
        if (!manifest) {
            throw std::runtime_error("could not write source manifest");
        }
        std::cout << "wrote " << options.output << "\n";
    } catch (const std::exception& error) {
        std::cerr << "ai_sapiens compile failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

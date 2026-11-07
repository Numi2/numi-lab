#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/RobotDescriptionCooker.hpp"
#include "metalrobo/VisualPresentation.hpp"

#include <libxml/parser.h>
#include <libxml/tree.h>

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
    "https://github.com/BoosterRobotics/booster_assets";
constexpr std::string_view kSourceRevision =
    "508cbee6ca9ae6fbc8c0b38dd58785a6f3fc61a2";

constexpr std::array<std::string_view, 22u> kJoints{{
    "AAHead_yaw", "Head_pitch", "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll",
    "Right_Elbow_Pitch", "Right_Elbow_Yaw", "Left_Hip_Pitch",
    "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll", "Right_Hip_Pitch",
    "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
    "Right_Ankle_Pitch", "Right_Ankle_Roll",
}};

constexpr std::array<float, 22u> kStiffness{{
    10.0F, 10.0F, 4.0F, 4.0F, 4.0F, 4.0F, 4.0F, 4.0F, 4.0F,
    4.0F, 80.0F, 80.0F, 80.0F, 80.0F, 30.0F, 30.0F, 80.0F,
    80.0F, 80.0F, 80.0F, 30.0F, 30.0F,
}};

constexpr std::array<float, 22u> kDamping{{
    2.0F, 2.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F,
    1.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F,
    2.0F, 2.0F, 2.0F, 2.0F,
}};

struct Arguments {
    std::filesystem::path source;
    std::filesystem::path output;
    bool reuseVisual = false;
};

Arguments arguments(const int argc, const char* const* argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option{argv[index]};
        if (option == "--reuse-visual") {
            result.reuseVisual = true;
        } else if (option == "--source" || option == "--output") {
            if (++index == argc) {
                throw std::invalid_argument("missing value for " +
                    std::string{option});
            }
            (option == "--source" ? result.source : result.output) =
                argv[index];
        } else if (option == "--help" || option == "-h") {
            std::cout << "usage: metalrobo_booster_k1_compile --source "
                         "BOOSTER_ASSETS --output ARTIFACT_DIRECTORY\n";
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

std::filesystem::path sourceURDF(const std::filesystem::path& source) {
    const auto path = source / "robots" / "K1" / "K1_22dof.urdf";
    if (!std::filesystem::is_regular_file(path)) {
        throw std::invalid_argument("official Booster K1 URDF is missing: " +
            path.string());
    }
    return path;
}

std::filesystem::path writeResolvedURDF(
    const std::filesystem::path& source,
    const std::filesystem::path& output
) {
    const auto input = sourceURDF(source);
    const auto target = output / "booster-k1.urdf";
    xmlDocPtr document = xmlReadFile(input.c_str(), nullptr, XML_PARSE_NONET);
    if (document == nullptr) {
        throw std::runtime_error("could not parse official Booster K1 URDF");
    }
    std::size_t resolved = 0u;
    const auto resolveNode = [&](this const auto& self, xmlNodePtr node) -> void {
        for (xmlNodePtr child = node; child != nullptr; child = child->next) {
            if (child->type == XML_ELEMENT_NODE &&
                xmlStrcmp(child->name, BAD_CAST "mesh") == 0) {
                xmlChar* raw = xmlGetProp(child, BAD_CAST "filename");
                const std::string filename = raw == nullptr
                    ? std::string{}
                    : reinterpret_cast<const char*>(raw);
                xmlFree(raw);
                std::filesystem::path mesh{filename};
                if (mesh.is_relative()) {
                    mesh = input.parent_path() / mesh;
                }
                mesh = std::filesystem::weakly_canonical(mesh);
                if (!std::filesystem::is_regular_file(mesh)) {
                    xmlFreeDoc(document);
                    throw std::runtime_error("Booster K1 mesh is missing: " +
                        mesh.string());
                }
                const std::string absolute = mesh.string();
                xmlSetProp(child, BAD_CAST "filename",
                    BAD_CAST absolute.c_str());
                ++resolved;
            }
            self(child->children);
        }
    };
    resolveNode(xmlDocGetRootElement(document));
    const int saved = xmlSaveFormatFileEnc(
        target.c_str(), document, "UTF-8", 1);
    xmlFreeDoc(document);
    if (resolved == 0u || saved < 0) {
        throw std::runtime_error("Booster K1 URDF mesh resolution failed");
    }
    return target;
}

metalrobo::TaskObservationOperatorSpec observation(
    const metalrobo::TaskObservationSource source,
    const std::string_view target
) {
    return {.source = source, .target = std::string{target}};
}

metalrobo::SensorProgramPack sensors() {
    metalrobo::SensorProgramPack result;
    result.id = "booster_k1_motion_sensor_v1";
    result.observation.actorHistoryLength = 1u;
    result.observation.criticHistoryLength = 1u;
    result.observation.criticIncludesCleanHistory = true;
    for (const std::string_view joint : kJoints) {
        result.observation.actorFrame.push_back(observation(
            metalrobo::TaskObservationSource::jointPositionError, joint));
    }
    for (const std::string_view joint : kJoints) {
        result.observation.actorFrame.push_back(observation(
            metalrobo::TaskObservationSource::jointVelocity, joint));
    }
    for (const std::string_view joint : kJoints) {
        result.observation.actorFrame.push_back(observation(
            metalrobo::TaskObservationSource::previousPolicyAction, joint));
    }
    result.observation.critic = result.observation.actorFrame;
    return result;
}

metalrobo::RobotActuatorPack actuators() {
    metalrobo::RobotActuatorPack result;
    result.id = "booster_k1_motion_actuators_v1";
    for (std::size_t index = 0u; index < kJoints.size(); ++index) {
        result.actuators.push_back({
            .id = std::string{kJoints[index]},
            .kind = metalrobo::RobotActuatorKind::jointPosition,
            .target = std::string{kJoints[index]},
            .scale = 1.0F,
            .parameters = {kStiffness[index], kDamping[index], 0.0F, 0.0F},
        });
    }
    return result;
}

metalrobo::TaskPack task() {
    metalrobo::TaskPack result;
    result.id = "booster_k1_motion_v1";
    result.maximumEpisodeSteps = 2'000u;
    result.difficultyBandCount = 1u;
    result.interactionControlReference = true;
    result.interactionInitializeFromReference = true;
    result.interactionAlignReferenceYaw = true;
    for (const std::string_view joint : kJoints) {
        result.actions.push_back({.actuator = std::string{joint}});
    }
    result.contactGroups = {
        {.id = "left_foot", .bodies = {"left_foot_link"},
            .support = true, .referenceBody = "left_foot_link"},
        {.id = "right_foot", .bodies = {"right_foot_link"},
            .support = true, .referenceBody = "right_foot_link"},
    };
    result.rewards = {
        {.operation = metalrobo::TaskRewardOperator::interactionJointTracking,
            .weight = 1.0F, .parameters = {0.5F, 0.25F, 0.0F, 0.0F}},
        {.operation = metalrobo::TaskRewardOperator::interactionRootTracking,
            .weight = 1.0F, .parameters = {1.0F, 0.25F, 1.0F, 1.0F}},
    };
    result.outcomes = {
        {"interaction_tracking", "ratio",
            metalrobo::TaskOutcomeSource::trackingScore,
            metalrobo::TaskOutcomeDirection::higherIsBetter},
        {"root_height", "m", metalrobo::TaskOutcomeSource::rootHeight,
            metalrobo::TaskOutcomeDirection::neutral},
        {"tilt", "rad", metalrobo::TaskOutcomeSource::tilt,
            metalrobo::TaskOutcomeDirection::lowerIsBetter},
    };
    return result;
}

metalrobo::RealityProgramPack reality() {
    metalrobo::RealityProgramPack result;
    result.id = "booster_k1_motion_reality_v1";
    result.program.id = "booster_k1_nominal_ground_v1";
    return result;
}

std::vector<float> csvRow(
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
            throw std::invalid_argument("invalid Booster motion value: " +
                path.string());
        }
        values.push_back(parsed);
    }
    if (values.size() != 7u + kJoints.size()) {
        throw std::invalid_argument("Booster K1 motion row is not 29 values");
    }
    return values;
}

metalrobo::InteractionPack interaction(
    const std::filesystem::path& source,
    const std::string_view id,
    const std::string_view filename,
    const float fps,
    const metalrobo::EngineModel& model
) {
    const auto path = source / "motions" / "K1" / filename;
    std::ifstream input{path};
    if (!input) {
        throw std::invalid_argument("official Booster motion is missing: " +
            path.string());
    }
    metalrobo::InteractionPack result{
        .id = "booster_k1_" + std::string{id},
        .sourceRepository = std::string{kSourceRepository},
        .sourceRevision = std::string{kSourceRevision},
        .license = "BSD-3-Clause",
        .coordinateFrame = metalrobo::kInteractionCoordinateFrame,
    };
    for (const auto joint : kJoints) {
        result.jointNames.push_back(std::string{joint});
    }
    metalrobo::InteractionClip clip{
        .id = std::string{id},
        .desiredOutcome = "Booster K1 source motion " + std::string{id},
        .framesPerSecond = fps,
        .loop = true,
    };
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        auto row = csvRow(line, path);
        const float quaternionNorm = std::sqrt(
            row[3] * row[3] + row[4] * row[4] +
            row[5] * row[5] + row[6] * row[6]
        );
        if (!std::isfinite(quaternionNorm) || quaternionNorm < 1.0e-6F) {
            throw std::invalid_argument(
                "Booster K1 motion contains an invalid root quaternion"
            );
        }
        for (std::size_t component = 3u; component < 7u; ++component) {
            row[component] /= quaternionNorm;
        }
        for (std::size_t joint = 0u; joint < kJoints.size(); ++joint) {
            const auto found = std::find(
                model.dofNames.begin(), model.dofNames.end(), kJoints[joint]);
            if (found == model.dofNames.end()) {
                throw std::runtime_error(
                    "Booster K1 motion joint is absent from cooked mechanics");
            }
            const std::size_t dof = static_cast<std::size_t>(
                std::distance(model.dofNames.begin(), found));
            row[7u + joint] = std::clamp(
                row[7u + joint],
                model.dofs[dof].limits.x,
                model.dofs[dof].limits.y
            );
        }
        clip.rootTargets.insert(clip.rootTargets.end(), row.begin(), row.begin() + 7);
        clip.jointTargets.insert(clip.jointTargets.end(), row.begin() + 7, row.end());
        ++clip.frameCount;
    }
    result.clips.push_back(std::move(clip));
    return result;
}

void writeOrThrow(
    const metalrobo::LearningPackResult result,
    const std::filesystem::path& path
) {
    if (!result.succeeded()) {
        throw std::runtime_error("could not write " + path.string() + ": " +
            result.message);
    }
}

void writeVisual(
    const std::filesystem::path& urdf,
    const metalrobo::EngineModel& cooked,
    const std::filesystem::path& output
) {
    metalrobo::VisualAssetCookOptions options;
    options.id = "booster_k1";
    options.license = "BSD-3-Clause";
    options.preprocessingProvenance = "Booster K1;source=" +
        std::string{kSourceRepository} + ";revision=" +
        std::string{kSourceRevision};
    for (std::uint32_t body = 0u; body < cooked.bodyNames.size(); ++body) {
        options.linkBodyIndices.emplace(cooked.bodyNames[body], body);
        options.linkCenterOfMassOffsets.emplace(
            cooked.bodyNames[body], cooked.bodies[body].centerOfMass);
    }
    std::vector<metalrobo::VisualAssetPackV2> packs;
    const auto diagnostics = metalrobo::cookUrdfVisualDescription(
        urdf, packs, options);
    if (!diagnostics.succeeded()) {
        throw std::runtime_error("Booster K1 visual cook failed: " +
            diagnostics.message);
    }
    const auto directory = output / "visual";
    std::filesystem::create_directories(directory);
    std::ofstream config{output / "booster-k1-visual-observation.json"};
    config << "{\n  \"format\": \"numi.visual-observation.v1\",\n"
           << "  \"id\": \"booster_k1_presentation_v1\",\n"
           << "  \"packs\": [\n";
    for (std::size_t index = 0u; index < packs.size(); ++index) {
        const auto path = directory /
            (packs[index].id + "_" + std::to_string(index) + ".mrvpack");
        std::string reason;
        if (!metalrobo::writeVisualAssetPack(packs[index], path, &reason)) {
            throw std::runtime_error("could not write Booster visual: " + reason);
        }
        config << "    {\"path\": \"visual/" << path.filename().string()
               << "\", \"asset_id\": \"robot\", \"semantic_id\": 1, "
                  "\"instance_id\": 1}"
               << (index + 1u == packs.size() ? "\n" : ",\n");
    }
    config << "  ],\n  \"camera\": {\n"
           << "    \"parent_body\": \"Trunk\",\n"
           << "    \"position\": [1.15, -1.45, 0.30],\n"
           << "    \"orientation\": [-0.73858354, -0.26102070, "
              "0.20711737, 0.58605883],\n"
           << "    \"width\": 16, \"height\": 9,\n"
           << "    \"minimum_visible_pixels\": 1,\n"
           << "    \"vertical_field_of_view_degrees\": 50.0,\n"
           << "    \"nominal_rate_hz\": 50.0,\n"
           << "    \"maximum_retained_bytes\": 0\n"
           << "  }\n}\n";
}

void writeWindowScene(
    const std::filesystem::path& output,
    const std::string_view id,
    const std::string_view name
) {
    const auto path = output /
        ("booster-k1-" + std::string{id} + ".numi-window.json");
    std::ofstream stream{path};
    stream << "{\n  \"format\": \"numi.window.scene.v1\",\n"
           << "  \"id\": \"booster-k1-" << id << "\",\n"
           << "  \"robot_id\": \"booster-k1\",\n"
           << "  \"robot_name\": \"Booster K1\",\n"
           << "  \"scene_id\": \"" << id << "\",\n"
           << "  \"scene_name\": \"" << name << "\",\n"
           << "  \"available\": false,\n"
           << "  \"visual_observation\": "
              "\"booster-k1-visual-observation.json\",\n"
           << "  \"arguments\": [\n"
           << "    \"--scene\", \"ground\",\n"
           << "    \"--urdf\", \"booster-k1.urdf\",\n"
           << "    \"--task-pack\", \"booster-k1.taskpack\",\n"
           << "    \"--robot-actuator-pack\", "
              "\"booster-k1.actuatorpack\",\n"
           << "    \"--sensor-program-pack\", "
              "\"booster-k1.sensorpack\",\n"
           << "    \"--reality-program-pack\", "
              "\"booster-k1.realitypack\",\n"
           << "    \"--interaction-pack\", \"booster-k1-" << id
           << ".interactionpack\",\n"
           << "    \"--interaction-clip\", \"" << id << "\",\n"
           << "    \"--zero-actions\"\n  ]\n}\n";
}

} // namespace

int main(const int argc, const char* const* argv) {
    try {
        const auto options = arguments(argc, argv);
        std::filesystem::create_directories(options.output);
        const auto urdf = writeResolvedURDF(options.source, options.output);
        metalrobo::EngineModel cooked;
        metalrobo::RobotDescriptionCookOptions cookOptions;
        cookOptions.rootMode = metalrobo::RobotDescriptionRootMode::floating;
        cookOptions.meshMode = metalrobo::RobotDescriptionMeshMode::convexHull;
        cookOptions.friction = {0.4F, 0.4F, 0.0F, 0.0F};
        const auto diagnostics = metalrobo::cookRobotDescriptionFiles(
            urdf, {}, cooked, cookOptions);
        if (!diagnostics.succeeded() || diagnostics.dofCount != 28u) {
            throw std::runtime_error("Booster K1 mechanics cook failed: " +
                diagnostics.message + " dofs=" +
                std::to_string(diagnostics.dofCount));
        }
        if (!options.reuseVisual || !std::filesystem::is_regular_file(
                options.output / "booster-k1-visual-observation.json")) {
            writeVisual(urdf, cooked, options.output);
        }
        writeOrThrow(metalrobo::writeTaskPack(task(),
            options.output / "booster-k1.taskpack"),
            options.output / "booster-k1.taskpack");
        writeOrThrow(metalrobo::writeRobotActuatorPack(actuators(),
            options.output / "booster-k1.actuatorpack"),
            options.output / "booster-k1.actuatorpack");
        writeOrThrow(metalrobo::writeSensorProgramPack(sensors(),
            options.output / "booster-k1.sensorpack"),
            options.output / "booster-k1.sensorpack");
        writeOrThrow(metalrobo::writeRealityProgramPack(reality(),
            options.output / "booster-k1.realitypack"),
            options.output / "booster-k1.realitypack");
        writeOrThrow(metalrobo::writeInteractionPack(interaction(
            options.source, "fight", "k1_fight_001_30fps.csv", 30.0F,
            cooked),
            options.output / "booster-k1-fight.interactionpack"),
            options.output / "booster-k1-fight.interactionpack");
        writeOrThrow(metalrobo::writeInteractionPack(interaction(
            options.source, "mj", "k1_mj2_seg1_50fps.csv", 50.0F,
            cooked),
            options.output / "booster-k1-mj.interactionpack"),
            options.output / "booster-k1-mj.interactionpack");
        writeWindowScene(options.output, "fight", "Fight Motion");
        writeWindowScene(options.output, "mj", "MJ Motion");
        std::ofstream source{options.output / "source.json"};
        source << "{\n  \"schema\": \"numi.booster-k1.v1\",\n"
               << "  \"source_repository\": \"" << kSourceRepository
               << "\",\n  \"source_revision\": \"" << kSourceRevision
               << "\",\n  \"joint_count\": 22,\n"
               << "  \"motion_scenes\": [\"fight\", \"mj\"]\n}\n";
        std::cout << "wrote " << options.output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Booster K1 compile failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

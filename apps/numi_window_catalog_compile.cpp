#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/G1.hpp"
#include "metalrobo/RunProgram.hpp"
#include "metalrobo/VisualPresentation.hpp"

#include <libxml/parser.h>
#include <libxml/tree.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using metalrobo::EngineModel;

struct Arguments {
    std::filesystem::path workspace;
    std::filesystem::path output;
};

Arguments arguments(const int argc, const char* const* argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option{argv[index]};
        if (option == "--workspace" || option == "--output") {
            if (++index == argc) {
                throw std::invalid_argument("missing value for " +
                    std::string{option});
            }
            (option == "--workspace" ? result.workspace : result.output) =
                argv[index];
        } else {
            throw std::invalid_argument("unknown option " +
                std::string{option});
        }
    }
    if (result.workspace.empty() || result.output.empty()) {
        throw std::invalid_argument("--workspace and --output are required");
    }
    return result;
}

void resolveMeshes(
    xmlNodePtr node,
    const std::filesystem::path& sourceDirectory
) {
    for (xmlNodePtr current = node; current != nullptr;
         current = current->next) {
        if (current->type == XML_ELEMENT_NODE &&
            xmlStrcmp(current->name, BAD_CAST "mesh") == 0) {
            xmlChar* raw = xmlGetProp(current, BAD_CAST "filename");
            if (raw != nullptr) {
                std::filesystem::path path{
                    reinterpret_cast<const char*>(raw)};
                xmlFree(raw);
                if (path.is_relative()) {
                    path = sourceDirectory / path;
                }
                path = std::filesystem::weakly_canonical(path);
                const std::string absolute = path.string();
                xmlSetProp(current, BAD_CAST "filename",
                    BAD_CAST absolute.c_str());
            }
        }
        resolveMeshes(current->children, sourceDirectory);
    }
}

std::optional<std::string> xmlProperty(
    xmlNodePtr node,
    const char* name
) {
    xmlChar* raw = xmlGetProp(node, BAD_CAST name);
    if (raw == nullptr) {
        return std::nullopt;
    }
    std::string value{reinterpret_cast<const char*>(raw)};
    xmlFree(raw);
    return value;
}

xmlNodePtr xmlChild(xmlNodePtr node, const char* name) {
    for (xmlNodePtr child = node == nullptr ? nullptr : node->children;
         child != nullptr; child = child->next) {
        if (child->type == XML_ELEMENT_NODE &&
            xmlStrcmp(child->name, BAD_CAST name) == 0) {
            return child;
        }
    }
    return nullptr;
}

using Matrix = std::array<double, 16u>;

Matrix identityMatrix() {
    return {{1.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0}};
}

Matrix multiply(const Matrix& left, const Matrix& right) {
    Matrix result{};
    for (std::size_t row = 0u; row < 4u; ++row) {
        for (std::size_t column = 0u; column < 4u; ++column) {
            for (std::size_t inner = 0u; inner < 4u; ++inner) {
                result[row * 4u + column] +=
                    left[row * 4u + inner] * right[inner * 4u + column];
            }
        }
    }
    return result;
}

std::array<double, 3u> vector3(
    const std::optional<std::string>& text
) {
    std::array<double, 3u> result{};
    if (!text.has_value()) {
        return result;
    }
    std::istringstream stream{*text};
    if (!(stream >> result[0] >> result[1] >> result[2]) ||
        !(stream >> std::ws).eof()) {
        throw std::runtime_error("invalid fixed-joint origin in visual URDF");
    }
    return result;
}

Matrix jointOrigin(xmlNodePtr joint) {
    const xmlNodePtr origin = xmlChild(joint, "origin");
    const auto xyz = vector3(origin == nullptr
        ? std::nullopt : xmlProperty(origin, "xyz"));
    const auto rpy = vector3(origin == nullptr
        ? std::nullopt : xmlProperty(origin, "rpy"));
    const double cr = std::cos(rpy[0]);
    const double sr = std::sin(rpy[0]);
    const double cp = std::cos(rpy[1]);
    const double sp = std::sin(rpy[1]);
    const double cy = std::cos(rpy[2]);
    const double sy = std::sin(rpy[2]);
    Matrix result = identityMatrix();
    result[0] = cy * cp;
    result[1] = cy * sp * sr - sy * cr;
    result[2] = cy * sp * cr + sy * sr;
    result[4] = sy * cp;
    result[5] = sy * sp * sr + cy * cr;
    result[6] = sy * sp * cr - cy * sr;
    result[8] = -sp;
    result[9] = cp * sr;
    result[10] = cp * cr;
    result[3] = xyz[0];
    result[7] = xyz[1];
    result[11] = xyz[2];
    return result;
}

std::filesystem::path resolvedURDF(
    const std::filesystem::path& source,
    const std::filesystem::path& target
) {
    xmlDocPtr document = xmlReadFile(source.c_str(), nullptr, XML_PARSE_NONET);
    if (document == nullptr) {
        throw std::runtime_error("could not read authored robot visual URDF");
    }
    xmlNodePtr root = xmlDocGetRootElement(document);
    resolveMeshes(root, source.parent_path());
    const int saved = xmlSaveFormatFileEnc(
        target.c_str(), document, "UTF-8", 1);
    xmlFreeDoc(document);
    if (saved < 0) {
        throw std::runtime_error("could not write filtered visual URDF");
    }
    return target;
}

void bindFixedVisualLinks(
    const std::filesystem::path& urdf,
    const EngineModel& model,
    metalrobo::VisualAssetCookOptions& options
) {
    xmlDocPtr document = xmlReadFile(urdf.c_str(), nullptr, XML_PARSE_NONET);
    if (document == nullptr) {
        throw std::runtime_error("could not read resolved visual URDF");
    }
    struct FixedParent {
        std::string parent;
        Matrix parentFromChild;
    };
    std::map<std::string, FixedParent> fixedParents;
    xmlNodePtr root = xmlDocGetRootElement(document);
    for (xmlNodePtr node = root->children; node != nullptr;
         node = node->next) {
        if (node->type != XML_ELEMENT_NODE ||
            xmlStrcmp(node->name, BAD_CAST "joint") != 0 ||
            xmlProperty(node, "type").value_or("") != "fixed") {
            continue;
        }
        const xmlNodePtr parent = xmlChild(node, "parent");
        const xmlNodePtr child = xmlChild(node, "child");
        if (parent == nullptr || child == nullptr) {
            continue;
        }
        fixedParents.emplace(
            xmlProperty(child, "link").value_or(""),
            FixedParent{
                xmlProperty(parent, "link").value_or(""),
                jointOrigin(node),
            }
        );
    }
    std::map<std::string, std::uint32_t> bodies;
    for (std::uint32_t body = 0u; body < model.bodyNames.size(); ++body) {
        bodies.emplace(model.bodyNames[body], body);
    }
    for (const auto& [link, ignored] : fixedParents) {
        (void)ignored;
        std::string current = link;
        Matrix bodyOriginFromLinkOrigin = identityMatrix();
        std::set<std::string> visited;
        while (!bodies.contains(current)) {
            if (!visited.insert(current).second) {
                xmlFreeDoc(document);
                throw std::runtime_error("fixed visual-link cycle in URDF");
            }
            const auto parent = fixedParents.find(current);
            if (parent == fixedParents.end()) {
                break;
            }
            bodyOriginFromLinkOrigin = multiply(
                parent->second.parentFromChild,
                bodyOriginFromLinkOrigin
            );
            current = parent->second.parent;
        }
        const auto body = bodies.find(current);
        if (body == bodies.end()) {
            continue;
        }
        options.linkBodyIndices.emplace(link, body->second);
        options.linkCenterOfMassOffsets.emplace(
            link, model.bodies[body->second].centerOfMass);
        std::array<float, 16u> encoded{};
        std::ranges::transform(
            bodyOriginFromLinkOrigin,
            encoded.begin(),
            [](const double value) { return static_cast<float>(value); }
        );
        options.linkOriginTransforms.emplace(link, encoded);
    }
    xmlFreeDoc(document);
}

void syntheticURDF(
    const std::filesystem::path& target,
    const std::vector<std::pair<std::string, std::filesystem::path>>& links
) {
    std::ofstream stream{target};
    stream << "<?xml version=\"1.0\"?><robot name=\"numi_window\">\n";
    for (const auto& [name, mesh] : links) {
        stream << "  <link name=\"" << name << "\"><visual><geometry>"
               << "<mesh filename=\"" << mesh.string()
               << "\"/></geometry><material name=\"studio\"><color "
                  "rgba=\"0.78 0.80 0.86 1\"/></material></visual></link>\n";
    }
    stream << "</robot>\n";
    if (!stream) {
        throw std::runtime_error("could not write visual URDF");
    }
}

void px4VisualURDF(
    const std::filesystem::path& target,
    const std::filesystem::path& clockwise,
    const std::filesystem::path& counterClockwise
) {
    std::ofstream stream{target};
    stream << "<?xml version=\"1.0\"?><robot name=\"px4_x500\">\n"
           << "  <link name=\"x500_base\">\n";
    const std::array<std::array<float, 3u>, 4u> positions{{
        {{0.174f, 0.174f, 0.04f}}, {{-0.174f, 0.174f, 0.04f}},
        {{0.174f, -0.174f, 0.04f}}, {{-0.174f, -0.174f, 0.04f}},
    }};
    for (std::size_t index = 0u; index < positions.size(); ++index) {
        const auto& position = positions[index];
        const auto& mesh = index % 2u == 0u ? clockwise : counterClockwise;
        stream << "    <visual><origin xyz=\"" << position[0] << ' '
               << position[1] << ' ' << position[2]
               << "\" rpy=\"0 0 0\"/><geometry><mesh filename=\""
               << mesh.string() << "\"/></geometry><material name=\"prop\">"
                  "<color rgba=\"0.18 0.20 0.24 1\"/></material></visual>\n";
    }
    stream << "  </link>\n</robot>\n";
}

void cookPresentation(
    const std::filesystem::path& urdf,
    const EngineModel& model,
    const std::string& id,
    const std::string& parentBody,
    const std::filesystem::path& output,
    const std::array<float, 3u>& position,
    const std::array<float, 4u>& orientation
) {
    metalrobo::VisualAssetCookOptions options;
    options.id = id;
    options.license = "NOASSERTION";
    options.preprocessingProvenance =
        "Numi Window source-authored robot catalog";
    for (std::uint32_t body = 0u; body < model.bodyNames.size(); ++body) {
        options.linkBodyIndices.emplace(model.bodyNames[body], body);
        options.linkCenterOfMassOffsets.emplace(
            model.bodyNames[body], model.bodies[body].centerOfMass);
    }
    bindFixedVisualLinks(urdf, model, options);
    std::vector<metalrobo::VisualAssetPackV2> packs;
    const auto diagnostics = metalrobo::cookUrdfVisualDescription(
        urdf, packs, options);
    if (!diagnostics.succeeded() || packs.empty()) {
        throw std::runtime_error("visual cook failed for " + id + ": " +
            diagnostics.message);
    }
    const auto visualDirectory = output / "visual";
    std::filesystem::create_directories(visualDirectory);
    std::ofstream config{output / (id + "-visual-observation.json")};
    config << "{\n  \"format\": \"numi.visual-observation.v1\",\n"
           << "  \"id\": \"" << id << "_presentation_v1\",\n"
           << "  \"packs\": [\n";
    for (std::size_t index = 0u; index < packs.size(); ++index) {
        const auto path = visualDirectory /
            (packs[index].id + "_" + std::to_string(index) + ".mrvpack");
        std::string reason;
        if (!metalrobo::writeVisualAssetPack(packs[index], path, &reason)) {
            throw std::runtime_error("could not write visual pack: " + reason);
        }
        config << "    {\"path\": \"visual/" << path.filename().string()
               << "\", \"asset_id\": \"robot\", \"semantic_id\": 1, "
                  "\"instance_id\": 1}"
               << (index + 1u == packs.size() ? "\n" : ",\n");
    }
    config << "  ],\n  \"camera\": {\n    \"parent_body\": \""
           << parentBody << "\",\n    \"position\": ["
           << position[0] << ", " << position[1] << ", " << position[2]
           << "],\n    \"orientation\": [" << orientation[0] << ", "
           << orientation[1] << ", " << orientation[2] << ", "
           << orientation[3] << "],\n    \"width\": 16, \"height\": 9,\n"
           << "    \"minimum_visible_pixels\": 1,\n"
           << "    \"vertical_field_of_view_degrees\": 52.0,\n"
           << "    \"nominal_rate_hz\": 50.0,\n"
           << "    \"maximum_retained_bytes\": 0\n  }\n}\n";
}

void scene(
    const std::filesystem::path& output,
    const std::string& id,
    const std::string& robotID,
    const std::string& robotName,
    const std::string& sceneName,
    const std::string& visual,
    const std::vector<std::string>& arguments
) {
    std::ofstream stream{output / (id + ".numi-window.json")};
    stream << "{\n  \"format\": \"numi.window.scene.v1\",\n"
           << "  \"id\": \"" << id << "\",\n"
           << "  \"robot_id\": \"" << robotID << "\",\n"
           << "  \"robot_name\": \"" << robotName << "\",\n"
           << "  \"scene_id\": \"" << id << "\",\n"
           << "  \"scene_name\": \"" << sceneName << "\",\n"
           << "  \"visual_observation\": \"" << visual << "\",\n"
           << "  \"arguments\": [";
    for (std::size_t index = 0u; index < arguments.size(); ++index) {
        stream << (index == 0u ? "\n    " : ",\n    ")
               << "\"" << arguments[index] << "\"";
    }
    stream << "\n  ]\n}\n";
}

} // namespace

int main(const int argc, const char* const* argv) {
    try {
        const Arguments options = arguments(argc, argv);
        std::filesystem::create_directories(options.output);
        constexpr std::array<float, 4u> cameraQ{{
            -0.73858354f, -0.26102070f, 0.20711737f, 0.58605883f}};

        const EngineModel g1 = metalrobo::makeUnitreeG1EngineModel();
        const auto g1Source = options.workspace /
            "build/unitree_ros/robots/g1_description/g1_29dof.urdf";
        const auto g1URDF = resolvedURDF(
            g1Source, options.output / "unitree-g1-visual.urdf");
        cookPresentation(g1URDF, g1, "unitree-g1", "pelvis",
            options.output, {1.05f, -1.30f, 0.32f}, cameraQ);
        const std::vector<std::pair<std::string, std::string>> g1Scenes{{
            "velocity", "Velocity"}, {"adult-locomotion", "Adult Locomotion"},
            {"g1-legs-locomotion", "Legs Locomotion"},
            {"disturbance-recovery", "Disturbance Recovery"},
            {"supine-get-up", "Supine Get Up"},
            {"developmental-recovery", "Developmental Recovery"},
            {"ball-recovery", "Ball Recovery"}, {"ball-dodge", "Ball Dodge"},
        };
        for (const auto& [task, name] : g1Scenes) {
            scene(options.output, "unitree-g1-" + task, "unitree-g1",
                "Unitree G1", name, "unitree-g1-visual-observation.json",
                {"--robot-source", "unitree-g1", "--scene", "ground",
                 "--task", task, "--native-policy"});
        }

        const EngineModel franka = metalrobo::makeFrankaPickPlaceEngineModel();
        const auto frankaMeshes = options.workspace /
            "build/franka_description/meshes/robots/fer/collision";
        std::vector<std::pair<std::string, std::filesystem::path>> frankaLinks;
        for (int link = 0; link <= 7; ++link) {
            frankaLinks.emplace_back("panda_link" + std::to_string(link),
                frankaMeshes / ("link" + std::to_string(link) + ".stl"));
        }
        frankaLinks.emplace_back("panda_hand", frankaMeshes / "hand.stl");
        const auto frankaURDF = options.output / "franka-visual.urdf";
        syntheticURDF(frankaURDF, frankaLinks);
        cookPresentation(frankaURDF, franka, "franka-panda", "panda_link0",
            options.output, {1.1f, -1.35f, 0.7f}, cameraQ);
        scene(options.output, "franka-pick-place", "franka-panda",
            "Franka Panda + Hand", "Pick & Place",
            "franka-panda-visual-observation.json",
            {"--robot-source", "franka-panda", "--scene", "ground",
             "--zero-actions"});
        scene(options.output, "franka-hand-motion", "franka-panda",
            "Franka Panda + Hand", "Hand Motion",
            "franka-panda-visual-observation.json",
            {"--robot-source", "franka-panda", "--scene", "ground"});

        const auto px4Pack = metalrobo::builtinRobotPack("px4_x500");
        if (!px4Pack.has_value()) {
            throw std::runtime_error("PX4 X500 built-in robot is unavailable");
        }
        const auto px4Meshes = options.workspace /
            ".numi/runs/px4-x500-source/source/models/x500_base/meshes";
        const auto px4URDF = options.output / "px4-x500-visual.urdf";
        px4VisualURDF(px4URDF, px4Meshes / "1345_prop_cw.stl",
            px4Meshes / "1345_prop_ccw.stl");
        cookPresentation(px4URDF, px4Pack->mechanics, "px4-x500", "x500_base",
            options.output, {0.35f, -0.46f, 0.16f}, cameraQ);
        scene(options.output, "px4-x500-hover", "px4-x500", "PX4 X500 Drone",
            "Hover Task", "px4-x500-visual-observation.json",
            {"--robot-source", "px4-x500", "--scene", "ground",
             "--native-policy"});
        scene(options.output, "px4-x500-motor-sweep", "px4-x500",
            "PX4 X500 Drone", "Motor Sweep",
            "px4-x500-visual-observation.json",
            {"--robot-source", "px4-x500", "--scene", "ground"});

        {
            std::ofstream version{options.output / "catalog.version"};
            version << "2\n";
            if (!version) {
                throw std::runtime_error("could not publish catalog version");
            }
        }
        std::cout << "wrote Numi Window built-in catalog "
                  << options.output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Numi Window catalog compile failed: "
                  << error.what() << '\n';
        return 1;
    }
    return 0;
}

#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/G1.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/WorldPack.hpp"

#include <CommonCrypto/CommonDigest.h>
#include "metalrobo/RunProgram.hpp"
#include "metalrobo/RobotDescriptionCooker.hpp"
#include "metalrobo/VisualPresentation.hpp"
#include "numi_mjcf_import.hpp"

#include <libxml/parser.h>
#include <libxml/tree.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <numeric>
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
    std::filesystem::path pandaModel;
    std::filesystem::path metallib;
    std::filesystem::path urdf;
    std::filesystem::path mjcf;
    std::filesystem::path srdf;
    std::string projectID = "robot-project";
    std::string projectName;
    std::optional<float> gravity;
    float friction = 0.8f;
    float effort = 0.1f;
    std::optional<float> pandaCubeMass;
    std::optional<float> pandaCubeFriction;
    bool starter = false;
};

Arguments arguments(const int argc, const char* const* argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option{argv[index]};
        if (option == "--starter") {
            result.starter = true;
        } else if (option == "--workspace" || option == "--output" ||
                   option == "--panda-model" || option == "--metallib" ||
                   option == "--urdf" || option == "--mjcf" || option == "--srdf" ||
                   option == "--project-id" || option == "--project-name" ||
                   option == "--gravity" || option == "--friction" || option == "--effort" ||
                   option == "--panda-cube-mass" || option == "--panda-cube-friction") {
            if (++index == argc) {
                throw std::invalid_argument("missing value for " +
                    std::string{option});
            }
            if (option == "--workspace") result.workspace = argv[index];
            else if (option == "--output") result.output = argv[index];
            else if (option == "--panda-model") result.pandaModel = argv[index];
            else if (option == "--metallib") result.metallib = argv[index];
            else if (option == "--urdf") result.urdf = argv[index];
            else if (option == "--mjcf") result.mjcf = argv[index];
            else if (option == "--srdf") result.srdf = argv[index];
            else if (option == "--project-id") result.projectID = argv[index];
            else if (option == "--project-name") result.projectName = argv[index];
            else {
                const std::string value = argv[index];
                std::size_t end = 0u;
                const float number = std::stof(value, &end);
                const bool mass = option == "--panda-cube-mass";
                const bool friction = option == "--friction" || option == "--panda-cube-friction";
                const float lower = option == "--effort" ? -1.0f : (mass ? 0.01f : 0.0f);
                const float upper = option == "--gravity" ? 100.0f : (friction ? 4.0f : 1.0f);
                if (end != value.size() || !std::isfinite(number) ||
                    number < lower || number > upper) {
                    throw std::invalid_argument("Invalid value for " + std::string{option});
                }
                if (option == "--gravity") result.gravity = number;
                else if (option == "--friction") result.friction = number;
                else if (option == "--effort") result.effort = number;
                else if (mass) result.pandaCubeMass = number;
                else result.pandaCubeFriction = number;
            }
        } else {
            throw std::invalid_argument("unknown option " +
                std::string{option});
        }
    }
    const auto importCount = !result.urdf.empty() + !result.mjcf.empty() + !result.pandaModel.empty();
    if (result.output.empty() || importCount > 1 ||
        (importCount == 0 && result.workspace.empty()) ||
        (importCount != 0 && result.starter) ||
        (!result.srdf.empty() && result.urdf.empty()) ||
        ((result.pandaCubeMass || result.pandaCubeFriction) && result.pandaModel.empty())) {
        throw std::invalid_argument(
            "Specify --output and one of --workspace [--starter], --panda-model, --mjcf, or --urdf [--srdf]."
        );
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

void scaleURDFOriginsAndMeshes(xmlNodePtr node, const double scale) {
    for (xmlNodePtr current = node; current != nullptr;
         current = current->next) {
        if (current->type == XML_ELEMENT_NODE &&
            xmlStrcmp(current->name, BAD_CAST "origin") == 0) {
            const auto xyz = vector3(xmlProperty(current, "xyz"));
            std::ostringstream encoded;
            encoded << xyz[0] * scale << ' ' << xyz[1] * scale << ' '
                    << xyz[2] * scale;
            const std::string value = encoded.str();
            xmlSetProp(current, BAD_CAST "xyz", BAD_CAST value.c_str());
        } else if (current->type == XML_ELEMENT_NODE &&
                   xmlStrcmp(current->name, BAD_CAST "mesh") == 0) {
            std::ostringstream encoded;
            encoded << scale << ' ' << scale << ' ' << scale;
            const std::string value = encoded.str();
            xmlSetProp(current, BAD_CAST "scale", BAD_CAST value.c_str());
        }
        scaleURDFOriginsAndMeshes(current->children, scale);
    }
}

bool numiflyLegName(const std::string_view name) {
    return name.starts_with("left_hip_") ||
        name.starts_with("left_knee_") ||
        name.starts_with("left_ankle_") ||
        name.starts_with("right_hip_") ||
        name.starts_with("right_knee_") ||
        name.starts_with("right_ankle_");
}

void removeNumiflyLegVisuals(xmlNodePtr root) {
    std::uint32_t removedLinks = 0u;
    std::uint32_t removedJoints = 0u;
    for (xmlNodePtr node = root->children; node != nullptr;) {
        xmlNodePtr next = node->next;
        if (node->type == XML_ELEMENT_NODE) {
            const std::string name = xmlProperty(node, "name").value_or("");
            const bool link = xmlStrcmp(node->name, BAD_CAST "link") == 0;
            const bool joint = xmlStrcmp(node->name, BAD_CAST "joint") == 0;
            if ((link || joint) && numiflyLegName(name)) {
                xmlUnlinkNode(node);
                xmlFreeNode(node);
                removedLinks += link ? 1u : 0u;
                removedJoints += joint ? 1u : 0u;
            }
        }
        node = next;
    }
    if (removedLinks != 12u || removedJoints != 12u) {
        throw std::runtime_error(
            "Numifly no-legs visual filter did not remove exactly 12 leg links and joints"
        );
    }
}

std::filesystem::path numiflyVisualURDF(
    const std::filesystem::path& resolvedG1,
    const std::filesystem::path& wingSurface,
    const std::filesystem::path& target,
    const bool noLegs = false
) {
    xmlDocPtr document = xmlReadFile(
        resolvedG1.c_str(), nullptr, XML_PARSE_NONET);
    if (document == nullptr) {
        throw std::runtime_error("could not read resolved G1 visual URDF");
    }
    xmlNodePtr root = xmlDocGetRootElement(document);
    scaleURDFOriginsAndMeshes(root, metalrobo::kNumiflyLinearScale);
    if (noLegs) {
        removeNumiflyLegVisuals(root);
        xmlSetProp(root, BAD_CAST "name", BAD_CAST "numifly_no_legs");
    }
    xmlNodePtr torso = nullptr;
    for (xmlNodePtr node = root->children; node != nullptr;
         node = node->next) {
        if (node->type == XML_ELEMENT_NODE &&
            xmlStrcmp(node->name, BAD_CAST "link") == 0 &&
            xmlProperty(node, "name").value_or("") == "torso_link") {
            torso = node;
            break;
        }
    }
    if (torso == nullptr) {
        xmlFreeDoc(document);
        throw std::runtime_error("G1 visual URDF has no torso_link");
    }
    xmlNodePtr visual = xmlNewChild(torso, nullptr, BAD_CAST "visual", nullptr);
    xmlSetProp(visual, BAD_CAST "name", BAD_CAST "numifly_maeda_wings");
    xmlNodePtr geometry = xmlNewChild(
        visual, nullptr, BAD_CAST "geometry", nullptr);
    xmlNodePtr mesh = xmlNewChild(
        geometry, nullptr, BAD_CAST "mesh", nullptr);
    const std::string wingPath =
        std::filesystem::weakly_canonical(wingSurface).string();
    xmlSetProp(mesh, BAD_CAST "filename", BAD_CAST wingPath.c_str());
    xmlSetProp(mesh, BAD_CAST "scale", BAD_CAST "1 1 1");
    xmlNodePtr material = xmlNewChild(
        visual, nullptr, BAD_CAST "material", nullptr);
    xmlSetProp(material, BAD_CAST "name", BAD_CAST "numifly_wing");
    xmlNodePtr color = xmlNewChild(
        material, nullptr, BAD_CAST "color", nullptr);
    xmlSetProp(color, BAD_CAST "rgba", BAD_CAST "0.18 0.42 0.85 0.82");
    const int saved = xmlSaveFormatFileEnc(
        target.c_str(), document, "UTF-8", 1);
    xmlFreeDoc(document);
    if (saved < 0) {
        throw std::runtime_error("could not write Numifly visual URDF");
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

metalrobo::VisualAssetPackV2 surfacePresentation(
    const bool terrain
) {
    constexpr std::uint32_t side = 161u;
    constexpr float spacing = 0.05f;
    constexpr float halfWidth = 4.0f;
    metalrobo::VisualAssetPackV2 pack;
    pack.id = terrain ? "numi_uneven_terrain" : "numi_flat_ground";
    pack.sourceUri = terrain
        ? "builtin://locomotion-terrain" : "builtin://locomotion-ground";
    pack.sourceContentHash = terrain
        ? "builtin:locomotion-terrain-v1" : "builtin:locomotion-ground-v1";
    pack.license = "NOASSERTION";
    pack.preprocessingProvenance =
        "Numi Window physical locomotion surface presentation v1";
    float minimumHeight = 0.0f;
    float maximumHeight = 0.0f;
    for (std::uint32_t y = 0u; y < side; ++y) {
        for (std::uint32_t x = 0u; x < side; ++x) {
            const float px = -halfWidth + spacing * static_cast<float>(x);
            const float py = -halfWidth + spacing * static_cast<float>(y);
            const auto height = [&](const float sx, const float sy) {
                return terrain
                    ? metalrobo::locomotionTerrainHeight(sx, sy) : 0.0f;
            };
            const float pz = height(px, py);
            minimumHeight = std::min(minimumHeight, pz);
            maximumHeight = std::max(maximumHeight, pz);
            const float dx =
                (height(px + spacing, py) - height(px - spacing, py)) /
                (2.0f * spacing);
            const float dy =
                (height(px, py + spacing) - height(px, py - spacing)) /
                (2.0f * spacing);
            const float inverseLength = 1.0f /
                std::sqrt(dx * dx + dy * dy + 1.0f);
            const mr_float4 normal{
                -dx * inverseLength, -dy * inverseLength,
                inverseLength, 1.0f};
            const float tangentLength = std::sqrt(
                normal.z * normal.z + normal.x * normal.x);
            pack.vertices.push_back({
                {px, py, pz, 1.0f},
                normal,
                {normal.z / tangentLength, 0.0f,
                 -normal.x / tangentLength, 0.0f},
                {static_cast<float>(x) / static_cast<float>(side - 1u),
                 static_cast<float>(y) / static_cast<float>(side - 1u),
                 0.0f, 0.0f},
                {1.0f, 1.0f, 1.0f, 1.0f},
            });
        }
    }
    for (std::uint32_t y = 0u; y + 1u < side; ++y) {
        for (std::uint32_t x = 0u; x + 1u < side; ++x) {
            const std::uint32_t a = y * side + x;
            const std::uint32_t b = a + 1u;
            const std::uint32_t c = a + side;
            const std::uint32_t d = c + 1u;
            pack.indices.insert(pack.indices.end(), {a, b, c, b, d, c});
        }
    }
    MRVisualMaterialGPUV2 material{};
    material.baseColorAndOpacity = terrain
        ? mr_float4{0.20f, 0.31f, 0.18f, 1.0f}
        : mr_float4{0.32f, 0.35f, 0.40f, 1.0f};
    material.surface = {0.82f, 0.02f, 1.0f, 1.0f};
    material.coatingAndAlphaCutoff = {0.0f, 0.0f, 1.0f, 0.5f};
    material.textureIndices0 = {MR_INVALID_INDEX, MR_INVALID_INDEX,
                                MR_INVALID_INDEX, MR_INVALID_INDEX};
    material.textureIndices1 = material.textureIndices0;
    material.reserved = material.textureIndices0;
    material.flags = {MR_VISUAL_ALPHA_OPAQUE,
        MR_VISUAL_MATERIAL_DOUBLE_SIDED, 0u, 1u};
    pack.materials = {material};
    MRVisualPrimitiveGPUV2 primitive{};
    primitive.geometry = {0u, static_cast<std::uint32_t>(pack.indices.size()),
                          0u, 0u};
    primitive.identity = {2u, 2u, 0u, 1u};
    primitive.boundsMinimum = {-halfWidth, -halfWidth, minimumHeight, 1.0f};
    primitive.boundsMaximum = {halfWidth, halfWidth, maximumHeight, 1.0f};
    pack.primitives = {primitive};
    MRVisualInstanceGPUV2 instance{};
    instance.translationAndScale = {0.0f, 0.0f, 0.0f, 1.0f};
    instance.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    instance.binding = {0u, MR_INVALID_INDEX, MR_VISUAL_BINDING_ASSET,
        MR_VISUAL_INSTANCE_CASTS_SHADOW |
        MR_VISUAL_INSTANCE_RECEIVES_SHADOW |
        MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR};
    instance.identity = {2u, 2u, 0u, 1u};
    instance.geometry = {0u, 1u, 0u, 0u};
    pack.instances = {instance};
    pack.symbolicBindings = {{"surface", "", 0u, MR_INVALID_INDEX,
                              MR_VISUAL_BINDING_ASSET}};
    pack.contentHash = metalrobo::computeVisualAssetPackContentHash(pack);
    return pack;
}

metalrobo::VisualAssetPackV2 projectilePresentation(
    const float radius
) {
    metalrobo::VisualAssetPackV2 pack;
    const auto radiusMillimetres = static_cast<std::uint32_t>(
        std::lround(radius * 1000.0f));
    pack.id = "numi_projectile_ball_" +
        std::to_string(radiusMillimetres) + "mm";
    pack.sourceUri = "builtin://locomotion-projectile-ball";
    pack.sourceContentHash = "builtin:locomotion-projectile-ball-v1-" +
        std::to_string(radiusMillimetres) + "mm";
    pack.license = "NOASSERTION";
    pack.preprocessingProvenance =
        "Numi Window physics-bound projectile presentation v1";
    constexpr std::uint32_t latitudeSegments = 16u;
    constexpr std::uint32_t longitudeSegments = 32u;
    constexpr float pi = 3.14159265358979323846f;
    for (std::uint32_t latitude = 0u;
         latitude <= latitudeSegments; ++latitude) {
        const float v = static_cast<float>(latitude) /
            static_cast<float>(latitudeSegments);
        const float polar = v * pi;
        const float ring = std::sin(polar);
        const float z = std::cos(polar);
        for (std::uint32_t longitude = 0u;
             longitude <= longitudeSegments; ++longitude) {
            const float u = static_cast<float>(longitude) /
                static_cast<float>(longitudeSegments);
            const float azimuth = u * 2.0f * pi;
            const float x = ring * std::cos(azimuth);
            const float y = ring * std::sin(azimuth);
            pack.vertices.push_back({
                {radius * x, radius * y, radius * z, 1.0f},
                {x, y, z, 1.0f},
                {-std::sin(azimuth), std::cos(azimuth), 0.0f, 0.0f},
                {u, v, 0.0f, 0.0f},
                {1.0f, 1.0f, 1.0f, 1.0f}});
        }
    }
    constexpr std::uint32_t ringStride = longitudeSegments + 1u;
    for (std::uint32_t latitude = 0u;
         latitude < latitudeSegments; ++latitude) {
        for (std::uint32_t longitude = 0u;
             longitude < longitudeSegments; ++longitude) {
            const std::uint32_t a = latitude * ringStride + longitude;
            const std::uint32_t b = a + ringStride;
            const std::uint32_t c = b + 1u;
            const std::uint32_t d = a + 1u;
            if (latitude != 0u) {
                pack.indices.insert(pack.indices.end(), {a, b, d});
            }
            if (latitude + 1u != latitudeSegments) {
                pack.indices.insert(pack.indices.end(), {d, b, c});
            }
        }
    }
    MRVisualMaterialGPUV2 material{};
    material.baseColorAndOpacity = {0.92f, 0.20f, 0.06f, 1.0f};
    material.surface = {0.28f, 0.04f, 1.0f, 1.0f};
    material.coatingAndAlphaCutoff = {0.18f, 0.12f, 1.0f, 0.5f};
    material.textureIndices0 = {MR_INVALID_INDEX, MR_INVALID_INDEX,
                                MR_INVALID_INDEX, MR_INVALID_INDEX};
    material.textureIndices1 = material.textureIndices0;
    material.reserved = material.textureIndices0;
    material.flags = {MR_VISUAL_ALPHA_OPAQUE,
        MR_VISUAL_MATERIAL_DOUBLE_SIDED, 0u, 1u};
    pack.materials = {material};
    MRVisualPrimitiveGPUV2 primitive{};
    primitive.geometry = {0u, static_cast<std::uint32_t>(pack.indices.size()),
                          0u, 0u};
    primitive.identity = {3u, 3u, 0u, 1u};
    primitive.boundsMinimum = {-radius, -radius, -radius, 1.0f};
    primitive.boundsMaximum = { radius,  radius,  radius, 1.0f};
    pack.primitives = {primitive};
    MRVisualInstanceGPUV2 instance{};
    instance.translationAndScale = {0.0f, 0.0f, 0.0f, 1.0f};
    instance.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    instance.binding = {0u, MR_INVALID_INDEX, MR_VISUAL_BINDING_ASSET,
        MR_VISUAL_INSTANCE_CASTS_SHADOW |
        MR_VISUAL_INSTANCE_RECEIVES_SHADOW |
        MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR};
    instance.identity = {3u, 3u, 0u, 1u};
    instance.geometry = {0u, 1u, 0u, 0u};
    pack.instances = {instance};
    pack.symbolicBindings = {{"projectile", "", 0u, MR_INVALID_INDEX,
                              MR_VISUAL_BINDING_ASSET}};
    pack.contentHash = metalrobo::computeVisualAssetPackContentHash(pack);
    return pack;
}

void writeSurfacePresentation(
    const std::filesystem::path& output,
    const std::string& robotConfig,
    const std::string& surface,
    const std::string& assetID,
    const metalrobo::VisualAssetPackV2& pack
) {
    const auto visualDirectory = output / "visual";
    const auto packPath = visualDirectory / (pack.id + ".mrvpack");
    std::string reason;
    if (!metalrobo::writeVisualAssetPack(pack, packPath, &reason)) {
        throw std::runtime_error("surface visual pack write failed: " + reason);
    }
    std::ifstream source{output / robotConfig};
    std::string contents{
        std::istreambuf_iterator<char>(source),
        std::istreambuf_iterator<char>()};
    const std::string marker = "  \"packs\": [\n";
    const auto position = contents.find(marker);
    if (position == std::string::npos) {
        throw std::runtime_error("robot visual config has no pack table");
    }
    contents.insert(position + marker.size(),
        "    {\"path\": \"visual/" + packPath.filename().string() +
        "\", \"asset_id\": \"" + assetID +
        "\", \"semantic_id\": 2, \"instance_id\": 2},\n");
    constexpr std::string_view suffix = "-visual-observation.json";
    if (!robotConfig.ends_with(suffix)) {
        throw std::runtime_error(
            "robot visual config does not use the catalog naming contract");
    }
    const std::string robotPrefix = robotConfig.substr(
        0u, robotConfig.size() - suffix.size());
    std::ofstream target{output /
        (robotPrefix + "-" + surface + "-visual-observation.json")};
    target << contents;
    if (!target) {
        throw std::runtime_error("surface visual config write failed");
    }
}

void writeBallPresentation(
    const std::filesystem::path& output,
    const std::string& surface,
    const std::string& task,
    const std::array<float, 6u>& radii
) {
    std::ifstream source{
        output / ("unitree-g1-" + surface + "-visual-observation.json")};
    std::string contents{
        std::istreambuf_iterator<char>(source),
        std::istreambuf_iterator<char>()};
    const std::string marker = "  \"packs\": [\n";
    const auto position = contents.find(marker);
    if (position == std::string::npos) {
        throw std::runtime_error("surface visual config has no pack table");
    }
    std::string entries;
    for (std::uint32_t index = 0u; index < radii.size(); ++index) {
        const auto pack = projectilePresentation(radii[index]);
        const auto packPath = output / "visual" / (pack.id + ".mrvpack");
        std::string reason;
        if (!metalrobo::writeVisualAssetPack(pack, packPath, &reason)) {
            throw std::runtime_error(
                "projectile visual pack write failed: " + reason);
        }
        entries += "    {\"path\": \"visual/" +
            packPath.filename().string() +
            "\", \"asset_id\": \"locomotion_dynamic_sphere_" +
            std::to_string(index) +
            "\", \"semantic_id\": 3, \"instance_id\": " +
            std::to_string(30u + index) + "},\n";
    }
    contents.insert(position + marker.size(), entries);
    std::ofstream target{output /
        ("unitree-g1-" + surface + "-" + task +
         "-visual-observation.json")};
    target << contents;
    if (!target) {
        throw std::runtime_error("ball visual config write failed");
    }
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

std::string jsonString(std::string_view value);

void cookPresentation(
    const std::filesystem::path& urdf,
    const EngineModel& model,
    const std::string& id,
    const std::string& parentBody,
    const std::filesystem::path& output,
    const std::array<float, 3u>& position,
    const std::array<float, 4u>& orientation,
    const std::string_view deformableSourceFilename = {},
    metalrobo::VisualAssetCookOptions options = {},
    const bool projectRelativeSources = false
) {
    options.id = id;
    if (options.license == "NOASSERTION") {
        options.preprocessingProvenance =
            "Numi Window source-authored robot catalog";
    }
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
    std::uint32_t deformablePackCount = 0u;
    for (std::size_t index = 0u; index < packs.size(); ++index) {
        auto& pack = packs[index];
        const auto source = std::filesystem::path{pack.sourceUri}.filename().string();
        if (projectRelativeSources) {
            pack.sourceUri = source.starts_with("primitive-")
                ? "builtin://numi/" + source
                : "source/meshes/" + source;
            pack.contentHash = metalrobo::computeVisualAssetPackContentHash(pack);
        }
        if (id == "px4-x500") {
            for (std::uint32_t rotor = 0u; rotor < 4u; ++rotor) {
                if (source != "px4-x500-rotor-" + std::to_string(rotor) + ".glb") {
                    continue;
                }
                for (auto& instance : pack.instances) {
                    instance.binding.w |= MR_VISUAL_INSTANCE_INSPECTION_X500_ROTOR |
                        (rotor << MR_VISUAL_X500_ROTOR_INDEX_SHIFT);
                }
                pack.contentHash = metalrobo::computeVisualAssetPackContentHash(pack);
            }
        }
        const auto path = visualDirectory /
            (pack.id + "_" + std::to_string(index) + ".mrvpack");
        std::string reason;
        if (!metalrobo::writeVisualAssetPack(pack, path, &reason)) {
            throw std::runtime_error("could not write visual pack: " + reason);
        }
        const bool deformable = !deformableSourceFilename.empty() &&
            source == deformableSourceFilename;
        deformablePackCount += deformable ? 1u : 0u;
        config << "    {\"path\": " << jsonString("visual/" + path.filename().string())
               << ", \"asset_id\": \"robot\", \"semantic_id\": "
               << (deformable ? 2 : 1) << ", \"instance_id\": "
               << (deformable ? 2 : 1)
               << (deformable
                    ? ", \"deformation\": \"measured_surface\"}"
                    : "}")
               << (index + 1u == packs.size() ? "\n" : ",\n");
    }
    if (!deformableSourceFilename.empty() && deformablePackCount != 1u) {
        throw std::runtime_error(
            "visual cook did not resolve exactly one authored deformable surface"
        );
    }
    config << "  ],\n  \"camera\": {\n    \"parent_body\": "
           << jsonString(parentBody) << ",\n    \"position\": ["
           << position[0] << ", " << position[1] << ", " << position[2]
           << "],\n    \"orientation\": [" << orientation[0] << ", "
           << orientation[1] << ", " << orientation[2] << ", "
           << orientation[3] << "],\n    \"width\": 16, \"height\": 9,\n"
           << "    \"minimum_visible_pixels\": 1,\n"
           << "    \"vertical_field_of_view_degrees\": 52.0,\n"
           << "    \"nominal_rate_hz\": 50.0,\n"
           << "    \"maximum_retained_bytes\": 0\n  }\n}\n";
}

std::string jsonString(std::string_view value) {
    std::ostringstream stream;
    stream << '"';
    for (const unsigned char byte : value) {
        if (byte == '"' || byte == '\\') stream << '\\' << static_cast<char>(byte);
        else if (byte < 32u) stream << "\\u00" << std::hex << std::setw(2)
                                   << std::setfill('0') << static_cast<unsigned>(byte);
        else stream << static_cast<char>(byte);
    }
    stream << '"';
    return stream.str();
}

void scene(
    const std::filesystem::path& output,
    const std::string& id,
    const std::string& robotID,
    const std::string& robotName,
    const std::string& sceneID,
    const std::string& sceneName,
    const std::string& taskID,
    const std::string& taskName,
    const std::string& visual,
    const std::vector<std::string>& arguments,
    const std::vector<std::string>& trainingArguments = {}
) {
    std::ofstream stream{output / (id + ".numi-window.json")};
    stream << "{\n  \"format\": \"numi.window.scene.v1\",\n"
           << "  \"id\": " << jsonString(id) << ",\n"
           << "  \"robot_id\": " << jsonString(robotID) << ",\n"
           << "  \"robot_name\": " << jsonString(robotName) << ",\n"
           << "  \"scene_id\": " << jsonString(sceneID) << ",\n"
           << "  \"scene_name\": " << jsonString(sceneName) << ",\n"
           << "  \"task_id\": " << jsonString(taskID) << ",\n"
           << "  \"task_name\": " << jsonString(taskName) << ",\n"
           << "  \"visual_observation\": " << jsonString(visual) << ",\n"
           << "  \"arguments\": [";
    for (std::size_t index = 0u; index < arguments.size(); ++index) {
        stream << (index == 0u ? "\n    " : ",\n    ")
               << jsonString(arguments[index]);
    }
    stream << "\n  ]";
    if (!trainingArguments.empty()) {
        stream << ",\n  \"training_arguments\": [";
        for (std::size_t index = 0u;
             index < trainingArguments.size(); ++index) {
            stream << (index == 0u ? "\n    " : ",\n    ")
                   << jsonString(trainingArguments[index]);
        }
        stream << "\n  ]";
    }
    stream << "\n}\n";
    stream.close();
    if (!stream) {
        throw std::runtime_error("could not publish window scene " + id);
    }
}

void writeCatalogVersion(
    const std::filesystem::path& output,
    const std::string_view version
) {
    std::ofstream stream{output / "catalog.version"};
    stream << version << '\n';
    stream.close();
    if (!stream) {
        throw std::runtime_error("could not publish catalog version");
    }
}

void writeX500HoverScene(const std::filesystem::path& output) {
    scene(output, "px4-x500-hover", "px4-x500", "PX4 X500 Drone",
        "flight-area", "Flight Test Area", "hover", "Hover",
        "px4-x500-visual-observation.json",
        {"--robot-source", "px4-x500", "--scene", "ground",
         "--zero-actions"});
}

void writeX500FlightPreview(const std::filesystem::path& output) {
    constexpr unsigned ticks = 300u;
    constexpr float duration = 6.0f;
    constexpr float frequency = std::numbers::pi_v<float> / duration;
    const auto robot = metalrobo::builtinRobotPack("px4_x500").value();
    const float hover = robot.multicopter->mixer.hoverAndScales.x;
    const float collectiveScale = robot.multicopter->mixer.hoverAndScales.y;
    std::ofstream clip{output / "px4-x500-flight.actions.f32", std::ios::binary};
    for (unsigned tick = 0u; tick < ticks; ++tick) {
        const float progress = std::numbers::pi_v<float> *
            (static_cast<float>(tick) + 0.5f) / static_cast<float>(ticks);
        const float sinSquared = std::pow(std::sin(progress), 2.0f);
        // Acceleration profile of a 0.55 m climb and descent with zero
        // velocity and acceleration at both ends.
        const float shape = frequency * frequency * sinSquared * (12.0f - 16.0f * sinSquared);
        const float speed = hover * std::sqrt(1.0f + (0.55f * shape - 0.003f) / 9.81f);
        const std::array actions{(speed - hover) / collectiveScale, 0.0f, 0.0f, 0.72f * shape};
        clip.write(reinterpret_cast<const char*>(actions.data()), sizeof(actions));
    }
    clip.close();
    if (!clip) throw std::runtime_error("Could not write the X500 flight preview clip.");
    scene(output, "px4-x500-hover", "px4-x500", "PX4 X500 Drone",
        "flight-area", "Flight Test Area", "flight-demo", "Flight Demo",
        "px4-x500-ground-visual-observation.json",
        {"--robot-source", "px4-x500-flight-demo", "--scene", "ground",
         "--no-scheduled-resets", "--action-clip", "px4-x500-flight.actions.f32",
         "--loop-action-clip"});
}

constexpr std::string_view pandaRevision =
    "eea4c7431b63da8278b0ff31638e0b0e941cec0e";

std::string checkedPandaXML(
    const std::filesystem::path& path,
    const std::string_view expectedHash
) {
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::file_size(path) > 65536u) {
        throw std::runtime_error("Choose the stock Menagerie panda.xml or scene.xml.");
    }
    std::ifstream input{path, std::ios::binary};
    std::string contents(65537u, '\0');
    input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
    contents.resize(static_cast<std::size_t>(input.gcount()));
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(contents.data(), static_cast<CC_LONG>(contents.size()), digest.data());
    std::ostringstream hash;
    hash << std::hex << std::setfill('0');
    for (const unsigned char byte : digest) hash << std::setw(2) << unsigned(byte);
    if (hash.str() != expectedHash) {
        throw std::runtime_error(
            "Unsupported Panda MJCF revision. Choose an unmodified model from "
            "MuJoCo Menagerie commit " + std::string{pandaRevision} + ".");
    }
    return contents;
}

void writePandaVisualDescription(
    const std::filesystem::path& selection,
    const std::filesystem::path& output
) {
    const auto folder = std::filesystem::canonical(selection.parent_path());
    const bool sceneSelected = selection.filename() == "scene.xml";
    if (sceneSelected) {
        (void)checkedPandaXML(selection,
            "d3933265c01e4301f192163364b2389013c283a9b1645ab158da8a5d837789a7");
    }
    const std::string xml = checkedPandaXML(sceneSelected ? folder / "panda.xml" : selection,
        "c047a49f3c770ff2393e2a74f7e0fbd9d99080a305407254f8ff53ac053cde3b");
    const std::unique_ptr<xmlDoc, decltype(&xmlFreeDoc)> source{
        xmlReadMemory(xml.data(), static_cast<int>(xml.size()), nullptr,
            nullptr, XML_PARSE_NONET), xmlFreeDoc};
    if (!source) throw std::runtime_error("Could not read Panda MJCF.");
    xmlNodePtr root = xmlDocGetRootElement(source.get());
    const auto assetFolder = folder /
        xmlProperty(xmlChild(root, "compiler"), "meshdir").value_or("assets");
    std::map<std::string, std::filesystem::path> meshes;
    std::map<std::string, std::string> materials;
    for (xmlNodePtr node = xmlChild(root, "asset")->children;
         node != nullptr; node = node->next) {
        if (node->type != XML_ELEMENT_NODE) continue;
        if (xmlStrEqual(node->name, BAD_CAST "mesh")) {
            const std::filesystem::path file = xmlProperty(node, "file").value();
            meshes.emplace(xmlProperty(node, "name").value_or(file.stem().string()), file);
        } else if (xmlStrEqual(node->name, BAD_CAST "material")) {
            materials.emplace(xmlProperty(node, "name").value(),
                xmlProperty(node, "rgba").value());
        }
    }

    const std::unique_ptr<xmlDoc, decltype(&xmlFreeDoc)> visual{
        xmlNewDoc(BAD_CAST "1.0"), xmlFreeDoc};
    xmlNodePtr robot = xmlNewNode(nullptr, BAD_CAST "robot");
    xmlDocSetRootElement(visual.get(), robot);
    xmlNewProp(robot, BAD_CAST "name", BAD_CAST "panda_menagerie");

    const auto addBody = [&](auto&& visit, xmlNodePtr body) -> void {
        const std::string name = xmlProperty(body, "name").value();
        const std::string nativeName = name == "left_finger" ? "panda_leftfinger"
            : name == "right_finger" ? "panda_rightfinger" : "panda_" + name;
        xmlNodePtr link = xmlNewChild(robot, nullptr, BAD_CAST "link", nullptr);
        xmlNewProp(link, BAD_CAST "name", BAD_CAST nativeName.c_str());
        for (xmlNodePtr child = body->children; child != nullptr; child = child->next) {
            if (child->type != XML_ELEMENT_NODE) continue;
            if (xmlStrEqual(child->name, BAD_CAST "body")) {
                visit(visit, child);
            } else if (xmlStrEqual(child->name, BAD_CAST "geom") &&
                       xmlProperty(child, "class") == "visual") {
                const auto relative = meshes.at(xmlProperty(child, "mesh").value());
                const auto path = assetFolder / relative;
                if (!std::filesystem::is_regular_file(path) ||
                    std::filesystem::canonical(path).parent_path() != assetFolder ||
                    std::filesystem::file_size(path) > 8u * 1024u * 1024u) {
                    throw std::runtime_error("Missing or invalid Panda mesh " +
                        relative.string() + ". Keep the stock assets folder beside the XML.");
                }
                xmlNodePtr part = xmlNewChild(link, nullptr, BAD_CAST "visual", nullptr);
                xmlNodePtr geometry = xmlNewChild(part, nullptr, BAD_CAST "geometry", nullptr);
                xmlNodePtr mesh = xmlNewChild(geometry, nullptr, BAD_CAST "mesh", nullptr);
                xmlNewProp(mesh, BAD_CAST "filename", BAD_CAST path.c_str());
                xmlNodePtr material = xmlNewChild(part, nullptr, BAD_CAST "material", nullptr);
                xmlNodePtr color = xmlNewChild(material, nullptr, BAD_CAST "color", nullptr);
                const std::string& rgba = materials.at(xmlProperty(child, "material").value());
                xmlNewProp(color, BAD_CAST "rgba", BAD_CAST rgba.c_str());
            }
        }
    };
    addBody(addBody, xmlChild(xmlChild(root, "worldbody"), "body"));
    if (xmlSaveFormatFileEnc(output.c_str(), visual.get(), "UTF-8", 1) < 0) {
        throw std::runtime_error("Could not write Panda presentation description.");
    }
}

void requirePandaPack(const auto& result) {
    if (!result.succeeded()) {
        throw std::runtime_error("Panda tabletop compilation failed: " + result.message);
    }
}

// Native nominal home and scale are taken from the WorldPack and its actuators.
// This clip uses control ticks at 50 Hz with zero endpoint velocity/acceleration.
void writePandaClip(
    const std::filesystem::path& output,
    const std::vector<float>& home,
    const std::vector<metalrobo::RobotActuatorSpec>& actuators
) {
    struct Keyframe {
        unsigned tick;
        float shoulder, elbow, wrist, finger;
    };
    constexpr std::array<Keyframe, 9u> keys{{
        {35, -0.7853982f, -2.3561945f, 1.5707963f, 0.039f},
        {75, 0.0855445f, -2.3873337f, 2.4728781f, 0.039f},
        {115, 0.3361294f, -2.3918243f, 2.7279537f, 0.039f},
        {150, 0.4587454f, -2.3658219f, 2.8245673f, 0.039f},
        {185, 0.4587454f, -2.3658219f, 2.8245673f, 0.006f},
        {225, 0.4587454f, -2.3658219f, 2.8245673f, 0.006f},
        {275, 0.2712647f, -2.3990754f, 2.6703400f, 0.006f},
        {310, 0.1283840f, -2.3952529f, 2.5236369f, 0.006f},
        {370, 0.1283840f, -2.3952529f, 2.5236369f, 0.006f},
    }};
    if (home.size() != 9u || actuators.size() != home.size()) {
        throw std::runtime_error("Panda clip requires seven arm joints and two fingers.");
    }
    std::ofstream clip{output / "tabletop.actions.f32", std::ios::binary};
    std::ofstream targets{output / "tabletop.targets.csv"};
    targets << std::setprecision(std::numeric_limits<float>::max_digits10);
    targets << "tick,joint1,joint2,joint3,joint4,joint5,joint6,joint7,"
               "finger_joint1,finger_joint2\n";
    std::vector<float> previous = home;
    unsigned previousTick = 0;
    for (const Keyframe& key : keys) {
        const std::array<float, 9u> goal{
            0.0f, key.shoulder, 0.0f, key.elbow, 0.0f, key.wrist,
            0.785398163f, key.finger, key.finger,
        };
        for (unsigned tick = previousTick + 1; tick <= key.tick; ++tick) {
            const float t = static_cast<float>(tick - previousTick) /
                static_cast<float>(key.tick - previousTick);
            const float blend = t * t * t * (10.0f + t * (-15.0f + 6.0f * t));
            targets << tick;
            for (std::size_t joint = 0; joint < goal.size(); ++joint) {
                const float q = std::lerp(previous[joint], goal[joint], blend);
                const float action = (q - home[joint]) / actuators[joint].scale;
                if (std::abs(action) > 1.0f) {
                    throw std::runtime_error("Panda clip target exceeds actuator range.");
                }
                clip.write(reinterpret_cast<const char*>(&action), sizeof(action));
                targets << ',' << q;
            }
            targets << '\n';
        }
        previous.assign(goal.begin(), goal.end());
        previousTick = key.tick;
    }
    clip.close();
    targets.close();
    if (!clip || !targets) throw std::runtime_error("Could not write Panda clip.");
}

void writeBoxOBJ(const std::filesystem::path& path, mr_float4 half) {
    std::ofstream stream{path};
    const std::array<float, 3> extents{half.x, half.y, half.z};
    for (unsigned corner = 0u; corner < 8u; ++corner) {
        stream << "v";
        for (unsigned axis = 0u; axis < extents.size(); ++axis) {
            stream << ' ' << (corner & (1u << axis) ? extents[axis] : -extents[axis]);
        }
        stream << '\n';
    }
    stream << "f 1 3 4 2\nf 5 6 8 7\nf 1 2 6 5\nf 3 7 8 4\n"
              "f 1 5 7 3\nf 2 4 8 6\n";
    stream.close();
    if (!stream) throw std::runtime_error("Could not write tabletop box geometry.");
}

using XmlDocument = std::unique_ptr<xmlDoc, decltype(&xmlFreeDoc)>;

XmlDocument readProjectURDF(const std::filesystem::path& path) {
    XmlDocument document{xmlReadFile(path.c_str(), nullptr, XML_PARSE_NONET |
        XML_PARSE_NOERROR | XML_PARSE_NOWARNING), xmlFreeDoc};
    const auto* root = document ? xmlDocGetRootElement(document.get()) : nullptr;
    if (root == nullptr || document->intSubset || document->extSubset ||
        xmlStrcmp(root->name, BAD_CAST "robot") != 0) {
        throw std::runtime_error("Expected a URDF robot without an XML document type.");
    }
    return document;
}

void saveProjectURDF(const std::filesystem::path& path, const XmlDocument& document) {
    if (xmlSaveFormatFileEnc(path.c_str(), document.get(), "UTF-8", 1) < 0) {
        throw std::runtime_error("Could not save project URDF.");
    }
}

void localizeProjectMeshes(
    xmlNode* node, const std::filesystem::path& source,
    const std::filesystem::path& destination,
    std::map<std::filesystem::path, std::filesystem::path>& copied
) {
    for (; node != nullptr; node = node->next) {
        if (node->type == XML_ELEMENT_NODE &&
            xmlStrcmp(node->name, BAD_CAST "mesh") == 0) {
            const std::string name = xmlProperty(node, "filename").value_or("");
            const std::filesystem::path requested{name};
            if (name.empty() || name.find("://") != std::string::npos ||
                requested.is_absolute()) {
                throw std::runtime_error("Use mesh paths relative to the URDF directory: " + name);
            }
            const auto mesh = std::filesystem::weakly_canonical(source / requested);
            const auto relative = mesh.lexically_relative(source);
            const auto type = mesh.extension().string();
            if (relative.empty() || *relative.begin() == ".." ||
                (type != ".obj" && type != ".stl" && type != ".glb") ||
                !std::filesystem::is_regular_file(mesh)) {
                throw std::runtime_error("Expected a local OBJ, STL, or GLB mesh: " + name);
            }
            if (type == ".obj") {
                std::ifstream input{mesh};
                for (std::string line; std::getline(input, line); ) {
                    std::istringstream tokens{line};
                    std::string first;
                    if ((tokens >> first) && first == "mtllib") {
                        throw std::runtime_error("Use embedded GLB materials or URDF colors: " + name);
                    }
                }
            }
            auto [entry, insert] = copied.try_emplace(mesh,
                std::filesystem::path{"meshes"} / (std::to_string(copied.size()) + type));
            if (insert) std::filesystem::copy_file(mesh, destination / entry->second);
            xmlSetProp(node, BAD_CAST "filename", BAD_CAST entry->second.c_str());
        }
        localizeProjectMeshes(node->children, source, destination, copied);
    }
}

float dimension(xmlNode* node, const char* attribute) {
    const std::string value = xmlProperty(node, attribute).value_or("");
    std::size_t end = 0;
    const float result = std::stof(value, &end);
    if (end != value.size() || !std::isfinite(result) || result <= 0.0f) {
        throw std::runtime_error("Invalid primitive dimension: " + std::string{attribute});
    }
    return result;
}

void writeRevolvedOBJ(const std::filesystem::path& mesh, float radius, float length, bool sphere) {
    constexpr unsigned slices = 32;
    const unsigned rings = sphere ? 15 : 2;
    std::ofstream file{mesh};
    file << "v 0 0 " << (-length / 2) << '\n';
    for (unsigned ring = 0; ring < rings; ++ring) {
        const double latitude = std::numbers::pi * (ring + 1) / (rings + 1);
        const double r = sphere ? radius * std::sin(latitude) : radius;
        const double z = sphere ? -radius * std::cos(latitude) :
            (static_cast<double>(ring) - 0.5) * length;
        for (unsigned slice = 0; slice < slices; ++slice) {
            const double azimuth = 2 * std::numbers::pi * slice / slices;
            file << "v " << r * std::cos(azimuth) << ' ' << r * std::sin(azimuth)
                 << ' ' << z << '\n';
        }
    }
    const unsigned top = 2 + rings * slices;
    file << "v 0 0 " << (length / 2) << '\n';
    for (unsigned slice = 0; slice < slices; ++slice) {
        const unsigned next = (slice + 1) % slices;
        file << "f 1 " << 2 + next << ' ' << 2 + slice << '\n';
        for (unsigned ring = 0; ring + 1 < rings; ++ring) {
            const unsigned first = 2 + ring * slices;
            file << "f " << first + slice << ' ' << first + next << ' '
                 << first + next + slices << ' ' << first + slice + slices << '\n';
        }
        const unsigned last = 2 + (rings - 1) * slices;
        file << "f " << top << ' ' << last + slice << ' ' << last + next << '\n';
    }
    file.close();
    if (!file) throw std::runtime_error("Could not write project presentation geometry.");
}

void projectVisualMeshes(xmlNode* node, const std::filesystem::path& output) {
    unsigned index = 0;
    for (xmlNode* link = node->children; link != nullptr; link = link->next) {
        if (link->type != XML_ELEMENT_NODE ||
            xmlStrcmp(link->name, BAD_CAST "link") != 0) continue;
        const std::string name = xmlProperty(link, "name").value_or("");
        if (name.empty() || name.find_first_of("/\\") != std::string::npos ||
            std::ranges::any_of(name, [](unsigned char c) { return c < 32; })) {
            throw std::runtime_error("Project link names must be printable and cannot contain path separators.");
        }
        for (xmlNode* visual = link->children; visual != nullptr; visual = visual->next) {
            if (visual->type != XML_ELEMENT_NODE ||
                xmlStrcmp(visual->name, BAD_CAST "visual") != 0) continue;
            xmlNode* geometry = xmlChild(visual, "geometry");
            if (geometry == nullptr || xmlChild(geometry, "mesh") != nullptr) continue;
            const auto mesh = output / ("primitive-" + std::to_string(index++) + ".obj");
            xmlNode* primitive = xmlChild(geometry, "box");
            if (primitive != nullptr) {
                const auto size = vector3(xmlProperty(primitive, "size"));
                mr_float4 half{static_cast<float>(size[0] / 2.0),
                               static_cast<float>(size[1] / 2.0),
                               static_cast<float>(size[2] / 2.0), 0.0f};
                if (!(half.x > 0 && half.y > 0 && half.z > 0) ||
                    !std::isfinite(half.x) || !std::isfinite(half.y) || !std::isfinite(half.z)) {
                    throw std::runtime_error("Visual box dimensions must be finite and positive.");
                }
                writeBoxOBJ(mesh, half);
            } else {
                primitive = xmlChild(geometry, "sphere");
                const bool sphere = primitive != nullptr;
                if (!sphere) primitive = xmlChild(geometry, "cylinder");
                if (primitive == nullptr) {
                    throw std::runtime_error("Use a mesh, box, sphere, or cylinder for visual geometry.");
                }
                const float radius = dimension(primitive, "radius");
                writeRevolvedOBJ(mesh, radius, sphere ? 2 * radius : dimension(primitive, "length"), sphere);
            }
            xmlUnlinkNode(primitive);
            xmlFreeNode(primitive);
            auto* replacement = xmlNewChild(geometry, nullptr, BAD_CAST "mesh", nullptr);
            xmlSetProp(replacement, BAD_CAST "filename", BAD_CAST mesh.filename().c_str());
        }
    }
}

void requireProjectPack(const auto& result) {
    if (!result.succeeded()) throw std::runtime_error("Project compilation failed: " + result.message);
}

void writeRobotProject(const Arguments& options) {
    if (options.projectID.empty() || options.projectID.size() > 64 ||
        !std::ranges::all_of(options.projectID, [](unsigned char c) {
            return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-';
        })) {
        throw std::runtime_error("The project ID must use lowercase ASCII letters, digits, and hyphens.");
    }
    if (std::filesystem::exists(options.output) && !std::filesystem::is_empty(options.output)) {
        throw std::runtime_error("Robot compilation requires an empty output directory.");
    }
    const bool isMJCF = !options.mjcf.empty();
    const auto source = std::filesystem::canonical(isMJCF ? options.mjcf : options.urdf);
    auto mjcf = isMJCF ? numi::importMJCF(source) : numi::MJCFImport{};
    auto document = isMJCF ? std::move(mjcf.urdf) : readProjectURDF(source);
    const float gravity = options.gravity.value_or(mjcf.gravity);
    const std::string name = options.projectName.empty()
        ? xmlProperty(xmlDocGetRootElement(document.get()), "name").value_or("Imported Robot")
        : options.projectName;
    if (name.empty() || name.size() > 200) {
        throw std::runtime_error("The project name must have 1 to 200 UTF-8 bytes.");
    }
    std::filesystem::create_directories(options.output / "source/meshes");
    const auto output = std::filesystem::canonical(options.output);
    const auto sourceDirectory = output / "source";
    const auto projectURDF = sourceDirectory / "model.urdf";
    std::map<std::filesystem::path, std::filesystem::path> copied;
    localizeProjectMeshes(xmlDocGetRootElement(document.get()), source.parent_path(),
        sourceDirectory, copied);
    saveProjectURDF(projectURDF, document);
    if (isMJCF) std::filesystem::copy_file(source, sourceDirectory / "model.xml");
    std::filesystem::path srdf;
    if (!options.srdf.empty()) {
        srdf = sourceDirectory / "model.srdf";
        std::filesystem::copy_file(options.srdf, srdf);
    }

    EngineModel mechanics;
    metalrobo::RobotDescriptionCookOptions cook;
    cook.gravityAndTimestep.z = -gravity;
    cook.friction.x = options.friction;
    cook.friction.y = options.friction * 0.75f;
    requireProjectPack(metalrobo::cookRobotDescriptionFiles(projectURDF, srdf, mechanics, cook));

    metalrobo::TaskPack task;
    task.id = "robot_effort_v1";
    metalrobo::RobotActuatorPack actuators;
    actuators.id = "robot_actuators_v1";
    metalrobo::SensorProgramPack sensors;
    sensors.id = "robot_joint_state_v1";
    for (const auto& dof : mechanics.dofs) {
        if ((dof.flags & MR_DOF_FLAG_ACTUATED) == 0u) continue;
        const std::string& joint = mechanics.dofNames.at(dof.vIndex);
        if ((dof.flags & MR_DOF_FLAG_EFFORT_LIMIT) == 0u) {
            throw std::runtime_error("Add a positive URDF effort limit to joint " + joint);
        }
        actuators.actuators.push_back({
            .id = "effort_" + std::to_string(task.actions.size()),
            .kind = metalrobo::RobotActuatorKind::jointEffort,
            .target = joint, .scale = dof.limits.w,
        });
        task.actions.push_back({actuators.actuators.back().id});
        sensors.observation.actorFrame.push_back({
            .source = metalrobo::TaskObservationSource::jointVelocity, .target = joint,
        });
    }
    if (task.actions.empty()) {
        throw std::runtime_error("The workbench requires at least one actuated joint.");
    }
    sensors.observation.critic = sensors.observation.actorFrame;
    const auto root = mechanics.articulations.front().rootBody;
    const std::string rootName = mechanics.bodyNames.at(root);
    // The solver anchors a fixed root at zero. Keep URDF link origins at
    // that anchor, including models whose fixed root has an inertial offset.
    const auto center = mechanics.bodies.at(root).centerOfMass;
    for (auto& joint : mechanics.joints) {
        if (joint.parentBody == root) {
            joint.parentAnchor.x += center.x;
            joint.parentAnchor.y += center.y;
            joint.parentAnchor.z += center.z;
        }
    }
    mechanics.bodies[root].centerOfMass = {};
    // Keep robot self-collision enabled. Floor rows on an immovable base or
    // fixed descendants have zero effective mass and should never be emitted.
    std::vector<bool> fixed(mechanics.bodies.size(), false);
    fixed[root] = true;
    for (const auto& joint : mechanics.joints) {
        fixed[joint.childBody] = fixed[joint.parentBody] && joint.jointType == MR_JOINT_FIXED;
    }
    constexpr auto floorGroup = 2u;
    constexpr auto fixedGroup = 4u;
    for (auto& shape : mechanics.shapes) {
        if (shape.bodyIndex == root) {
            shape.localPosition.x += center.x;
            shape.localPosition.y += center.y;
            shape.localPosition.z += center.z;
        }
        if (fixed.at(shape.bodyIndex)) {
            shape.collisionGroup = fixedGroup;
            shape.collisionMask &= ~(floorGroup | fixedGroup);
        }
    }
    metalrobo::WorldAsset robot;
    robot.id = "robot";
    robot.semanticClass = "robot";
    robot.role = MR_WORLD_ASSET_ROBOT;
    robot.render = MR_WORLD_RENDER_MESH_PBR;
    robot.collision = MR_WORLD_COLLISION_CONVEX;
    robot.dynamics = MR_WORLD_DYNAMICS_ARTICULATED;
    robot.articulationIndex = 0;
    robot.bodyIndices.resize(mechanics.bodies.size());
    robot.shapeIndices.resize(mechanics.shapes.size());
    robot.materialIndices.resize(mechanics.materials.size());
    std::iota(robot.bodyIndices.begin(), robot.bodyIndices.end(), 0u);
    std::iota(robot.shapeIndices.begin(), robot.shapeIndices.end(), 0u);
    std::iota(robot.materialIndices.begin(), robot.materialIndices.end(), 0u);
    metalrobo::WorldAsset floor;
    floor.id = "locomotion_ground";
    floor.semanticClass = "ground";
    floor.role = MR_WORLD_ASSET_BACKGROUND;
    floor.render = MR_WORLD_RENDER_MESH_PBR;
    floor.collision = MR_WORLD_COLLISION_PRIMITIVES;
    floor.bodyIndices = {static_cast<std::uint32_t>(mechanics.bodies.size())};
    floor.shapeIndices = {static_cast<std::uint32_t>(mechanics.shapes.size())};
    floor.materialIndices = {0u};
    std::vector<MRBodyStateGPU> sceneBodies;
    metalrobo::appendLocomotionSurface(mechanics, sceneBodies, metalrobo::LocomotionSurface::ground);
    mechanics.shapes.back().collisionGroup = floorGroup;
    if (isMJCF && !mjcf.ground) {
        for (auto& shape : mechanics.shapes) shape.collisionMask &= ~floorGroup;
    }

    metalrobo::EpisodeTwin episode;
    episode.id = "robot_project_v1";
    episode.assets = {std::move(robot), std::move(floor)};
    episode.task = {.id = task.id, .robotAssetId = "robot", .controlPeriodSeconds = 0.02};
    metalrobo::WorldTemplate world;
    requireProjectPack(metalrobo::compileEpisodeTwin(episode, mechanics, world));
    metalrobo::WorldProgram program;
    program.id = "robot_nominal_v1";
    metalrobo::WorldFamily family;
    requireProjectPack(metalrobo::compileWorldFamily(world, program, family));
    metalrobo::MRWorldPack pack;
    requireProjectPack(metalrobo::compileWorldPack(family, pack));
    requireProjectPack(metalrobo::writeWorldPack(pack, output / "robot.worldpack"));
    requireProjectPack(metalrobo::writeTaskPack(task, output / "robot.taskpack"));
    requireProjectPack(metalrobo::writeRobotActuatorPack(actuators, output / "robot.actuators"));
    requireProjectPack(metalrobo::writeSensorProgramPack(sensors, output / "robot.sensors"));
    requireProjectPack(metalrobo::writeRealityProgramPack(
        {.id = program.id, .program = std::move(program),
         .sourceProgramFingerprint = family.program.fingerprint}, output / "robot.reality"));

    std::ofstream clip{output / "robot.actions.f32", std::ios::binary};
    std::vector<float> action(task.actions.size(), 0.0f);
    for (unsigned tick = 0; tick < 200; ++tick) {
        action.front() = options.effort * std::sin(
            2 * std::numbers::pi_v<float> * static_cast<float>(tick) / 200.0f);
        clip.write(reinterpret_cast<const char*>(action.data()), action.size() * sizeof(float));
    }
    clip.close();
    if (!clip) throw std::runtime_error("Could not write project effort clip.");
    std::ofstream settings{output / "project.json"};
    settings << std::setprecision(std::numeric_limits<float>::max_digits10)
             << "{\n  \"format\": \"numi.robot-project.v1\",\n  \"name\": "
             << jsonString(name) << ",\n  \"gravity\": "
             << gravity << ",\n  \"friction\": " << options.friction
             << ",\n  \"effort\": " << options.effort << ",\n  \"action_count\": " << task.actions.size();
    if (isMJCF) settings << ",\n  \"source_format\": \"mjcf\"";
    settings << "\n}\n";
    settings.close();
    if (!settings) throw std::runtime_error("Could not save project settings.");

    const auto presentationSource = sourceDirectory / "presentation.urdf";
    projectVisualMeshes(xmlDocGetRootElement(document.get()), sourceDirectory);
    saveProjectURDF(presentationSource, document);
    metalrobo::VisualAssetCookOptions visual;
    visual.metallibPath = options.metallib;
    cookPresentation(presentationSource, mechanics, options.projectID, rootName,
        output, {1.35f, -1.4f, 0.8f},
        {-0.73858354f, -0.26102070f, 0.20711737f, 0.58605883f}, {}, std::move(visual), true);
    const std::string visualConfig = options.projectID + "-visual-observation.json";
    writeSurfacePresentation(output, visualConfig, "ground", "locomotion_ground", surfacePresentation(false));
    std::filesystem::remove(output / visualConfig);
    std::filesystem::remove(presentationSource);
    for (const auto& entry : std::filesystem::directory_iterator(sourceDirectory)) {
        if (entry.path().filename().string().starts_with("primitive-")) {
            std::filesystem::remove(entry.path());
        }
    }
    std::vector<std::string> launchArguments{
         "--world-pack", "robot.worldpack", "--task-pack", "robot.taskpack",
         "--robot-actuator-pack", "robot.actuators", "--sensor-program-pack", "robot.sensors",
         "--reality-program-pack", "robot.reality", "--no-scheduled-resets",
         "--action-clip", "robot.actions.f32", "--loop-action-clip"};
    if (isMJCF) {
        launchArguments.insert(launchArguments.end(),
            {"--physics-substeps", std::to_string(mjcf.physicsSubsteps)});
    }
    scene(output, options.projectID, options.projectID, name, "ground", "Ground",
        "effort", "Joint Effort", options.projectID + "-ground-visual-observation.json", launchArguments);
}

void writePandaTabletop(const Arguments& options) {
    if (std::filesystem::exists(options.output) &&
        !std::filesystem::is_empty(options.output)) {
        throw std::runtime_error("Panda tabletop requires an empty output directory.");
    }
    std::filesystem::create_directories(options.output);
    const auto output = std::filesystem::canonical(options.output);
    const auto visualDescription = output / "panda-visual.urdf";
    writePandaVisualDescription(
        std::filesystem::absolute(options.pandaModel).lexically_normal(), visualDescription);

    EngineModel mechanics = metalrobo::makeFrankaPickPlaceEngineModel();
    // Anchor the fixed mount at the stock MJCF link origin. The solver places
    // a fixed root at zero, so express its joint, collision, and visual frames
    // relative to the link origin for this scene.
    const auto root = mechanics.articulations.front().rootBody;
    const auto center = mechanics.bodies.at(root).centerOfMass;
    for (auto& joint : mechanics.joints) {
        if (joint.parentBody == root) {
            joint.parentAnchor.x += center.x;
            joint.parentAnchor.y += center.y;
            joint.parentAnchor.z += center.z;
        }
    }
    mechanics.bodies[root].centerOfMass = {};
    // The authored cube is a fingertip grasp fixture. Broad arm proxies
    // remain available to the rest of the workcell, but cannot lift the cube.
    constexpr std::uint32_t fingertips = 8u;
    for (auto& shape : mechanics.shapes) {
        if (shape.bodyIndex == root) {
            shape.localPosition.x += center.x;
            shape.localPosition.y += center.y;
            shape.localPosition.z += center.z;
        }
        const std::string& body = mechanics.bodyNames.at(shape.bodyIndex);
        if (body == "panda_leftfinger" || body == "panda_rightfinger") {
            shape.collisionGroup |= fingertips;
        } else if (body == "pick_object") {
            shape.collisionMask = 1u | 4u | fingertips;
        }
    }
    const auto cube = std::ranges::find(mechanics.bodyNames, "pick_object");
    if (cube == mechanics.bodyNames.end()) {
        throw std::runtime_error("Missing Panda tabletop cube.");
    }
    const auto cubeIndex = static_cast<std::size_t>(cube - mechanics.bodyNames.begin());
    auto& cubeBody = mechanics.bodies.at(cubeIndex);
    if (options.pandaCubeMass) {
        const float mass = *options.pandaCubeMass;
        const float ratio = mass / cubeBody.massAndInverseMass.x;
        cubeBody.massAndInverseMass = {mass, 1.0f / mass, 0.0f, 0.0f};
        for (auto* row : {&cubeBody.inertiaRow0, &cubeBody.inertiaRow1, &cubeBody.inertiaRow2}) {
            row->x *= ratio; row->y *= ratio; row->z *= ratio;
        }
        for (auto* row : {&cubeBody.inverseInertiaRow0, &cubeBody.inverseInertiaRow1, &cubeBody.inverseInertiaRow2}) {
            row->x /= ratio; row->y /= ratio; row->z /= ratio;
        }
    }
    std::optional<std::uint32_t> cubeMaterial;
    if (options.pandaCubeFriction) {
        const float friction = *options.pandaCubeFriction;
        for (auto& shape : mechanics.shapes) {
            if (shape.bodyIndex != cubeIndex) continue;
            if (!cubeMaterial) {
                auto material = mechanics.materials.at(shape.materialIndex);
                material.friction.x = friction;
                material.friction.y = friction * (6.0f / 7.0f);
                cubeMaterial = static_cast<std::uint32_t>(mechanics.materials.size());
                mechanics.materials.push_back(material);
                mechanics.world.materialCount = *cubeMaterial + 1u;
            }
            shape.materialIndex = *cubeMaterial;
        }
    }
    std::ofstream fixture{output / "tabletop.experiment.json"};
    fixture << std::setprecision(std::numeric_limits<float>::max_digits10)
            << "{\"cube_mass_kg\":" << cubeBody.massAndInverseMass.x
            << ",\"cube_static_friction\":" << options.pandaCubeFriction.value_or(0.7f) << "}\n";
    fixture.close();
    if (!fixture) throw std::runtime_error("Could not write Panda experiment settings.");

    auto episode = metalrobo::makeFrankaPickPlaceEpisodeTwin();
    episode.id = "panda_tabletop_v1";
    if (cubeMaterial) {
        for (auto& asset : episode.assets) {
            if (asset.id == "pick_object") asset.materialIndices = {*cubeMaterial};
        }
    }
    episode.artifacts.clear();
    episode.sensors.clear();
    episode.task.controlPeriodSeconds = 0.02;
    metalrobo::WorldTemplate world;
    requirePandaPack(metalrobo::compileEpisodeTwin(episode, mechanics, world));
    metalrobo::WorldProgram program;
    program.id = "panda_tabletop_nominal";
    metalrobo::WorldFamily family;
    requirePandaPack(metalrobo::compileWorldFamily(world, program, family));
    metalrobo::MRWorldPack worldPack;
    requirePandaPack(metalrobo::compileWorldPack(family, worldPack));
    requirePandaPack(metalrobo::writeWorldPack(worldPack, output / "tabletop.worldpack"));

    metalrobo::TaskObservationProgram observations;
    metalrobo::TaskResetProgram reset;
    auto task = metalrobo::makeFrankaPickPlaceTaskPack(observations, reset);
    task.id = "panda_tabletop_v1";
    auto robot = metalrobo::builtinRobotPack("franka_panda").value();
    constexpr std::array<float, 7> armStiffness{4500, 4500, 3500, 3500, 2000, 2000, 2000};
    for (std::size_t index = 0; index < robot.actuators.size(); ++index) {
        auto& actuator = robot.actuators[index];
        actuator.scale = actuator.kind == metalrobo::RobotActuatorKind::gripperPosition
            ? 0.04f : 2.5f;
        if (index < armStiffness.size()) {
            const float stiffness = armStiffness[index];
            actuator.parameters = {stiffness, stiffness / 10.0f, 0, 0};
        }
    }
    writePandaClip(output, mechanics.defaultQ, robot.actuators);
    requirePandaPack(metalrobo::writeTaskPack(task, output / "tabletop.taskpack"));
    requirePandaPack(metalrobo::writeRobotActuatorPack(
        {.id = "panda_tabletop_actuators", .actuators = std::move(robot.actuators)},
        output / "tabletop.actuators"));
    requirePandaPack(metalrobo::writeSensorProgramPack(
        {.id = "panda_tabletop_sensors", .observation = std::move(observations)},
        output / "tabletop.sensors"));
    requirePandaPack(metalrobo::writeRealityProgramPack(
        {.id = "panda_tabletop_nominal", .program = std::move(program),
         .sourceProgramFingerprint = family.program.fingerprint},
        output / "tabletop.reality"));
    metalrobo::VisualAssetCookOptions visual;
    visual.license = "Apache-2.0";
    visual.preprocessingProvenance = "MuJoCo Menagerie Panda " + std::string{pandaRevision};
    visual.metallibPath = options.metallib;
    cookPresentation(visualDescription, mechanics, "panda-tabletop", "panda_link0",
        output, {0.88f, -0.58f, 0.37f},
        {-0.73858354f, -0.26102070f, 0.20711737f, 0.58605883f},
        {}, std::move(visual));
    const std::string config = "panda-tabletop-visual-observation.json";
    const auto ground = surfacePresentation(false);
    writeSurfacePresentation(output, config, "ground", "workspace", ground);
    const auto tabletopConfig = output / "panda-tabletop-ground-visual-observation.json";
    std::ifstream input{tabletopConfig};
    std::string contents{std::istreambuf_iterator<char>(input), {}};
    input.close();
    const std::string marker = "  \"packs\": [\n";
    const auto offset = contents.find(marker);
    if (offset == std::string::npos) {
        throw std::runtime_error("Panda visual config has no pack table.");
    }
    constexpr std::array boxNames{"pick_object", "target_fixture", "clutter"};
    constexpr std::array<mr_float4, 3u> colors{{
        {0.82f, 0.19f, 0.045f, 1.0f},
        {0.09f, 0.43f, 0.25f, 1.0f},
        {0.09f, 0.28f, 0.62f, 1.0f},
    }};
    for (unsigned index = 0u; index < boxNames.size(); ++index) {
        const std::string box = boxNames[index];
        const auto body = std::ranges::find(mechanics.bodyNames, box);
        const auto bodyIndex = std::distance(mechanics.bodyNames.begin(), body);
        const auto shape = std::ranges::find_if(mechanics.shapes, [&](const auto& candidate) {
            return candidate.bodyIndex == bodyIndex && candidate.shapeType == MR_SHAPE_BOX;
        });
        if (body == mechanics.bodyNames.end() || shape == mechanics.shapes.end()) {
            throw std::runtime_error("Panda tabletop box is unresolved: " + box);
        }
        const auto mesh = output / (box + ".obj");
        writeBoxOBJ(mesh, shape->dimensions);
        metalrobo::VisualAssetCookOptions boxOptions;
        boxOptions.id = box;
        boxOptions.metallibPath = options.metallib;
        metalrobo::VisualAssetPackV2 pack;
        requirePandaPack(metalrobo::cookVisualAsset(mesh, pack, boxOptions));
        const auto visualBody = static_cast<std::uint32_t>(bodyIndex);
        for (auto& instance : pack.instances) {
            instance.binding.y = visualBody;
            instance.binding.z = MR_VISUAL_BINDING_RIGID_BODY;
            instance.identity.z = visualBody;
        }
        for (auto& binding : pack.symbolicBindings) {
            binding.link = box;
            binding.bodyIndex = visualBody;
            binding.binding = MR_VISUAL_BINDING_RIGID_BODY;
        }
        for (auto& material : pack.materials) {
            material.baseColorAndOpacity = colors[index];
        }
        pack.contentHash = metalrobo::computeVisualAssetPackContentHash(pack);
        std::filesystem::remove(mesh);
        std::string error;
        const std::string filename = box + ".mrvpack";
        if (!metalrobo::writeVisualAssetPack(pack, output / "visual" / filename, &error)) {
            throw std::runtime_error("Could not write tabletop " + box + ": " + error);
        }
        const std::string instance = std::to_string(index + 3u);
        contents.insert(offset + marker.size(),
            "    {\"path\": \"visual/" + filename + "\", \"asset_id\": \"" + box +
            "\", \"semantic_id\": " + instance + ", \"instance_id\": " + instance + "},\n");
    }
    std::ofstream tabletopVisual{tabletopConfig};
    tabletopVisual << contents;
    tabletopVisual.close();
    if (!tabletopVisual) throw std::runtime_error("Could not write Panda tabletop presentation.");
    std::filesystem::remove(visualDescription);
    std::filesystem::remove(output / config);
    scene(output, "panda-tabletop", "panda-tabletop", "Franka Panda",
        "tabletop", "Tabletop", "grasp", "Tabletop Grasp",
        "panda-tabletop-ground-visual-observation.json",
        {"--world-pack", "tabletop.worldpack", "--task-pack", "tabletop.taskpack",
         "--robot-actuator-pack", "tabletop.actuators",
         "--sensor-program-pack", "tabletop.sensors",
         "--reality-program-pack", "tabletop.reality", "--no-scheduled-resets",
         "--action-clip", "tabletop.actions.f32"});
}

} // namespace

int main(const int argc, const char* const* argv) {
    try {
        const Arguments options = arguments(argc, argv);
        if (!options.urdf.empty() || !options.mjcf.empty()) {
            writeRobotProject(options);
            return 0;
        }
        if (!options.pandaModel.empty()) {
            writePandaTabletop(options);
            return 0;
        }
        std::filesystem::create_directories(options.output);
        std::filesystem::remove(options.output / "catalog.version");
        for (const auto& entry :
             std::filesystem::directory_iterator(options.output)) {
            if (entry.is_regular_file() &&
                entry.path().filename().string().ends_with(
                    ".numi-window.json")) {
                std::filesystem::remove(entry.path());
            }
        }
        constexpr std::array<float, 4u> cameraQ{{
            -0.73858354f, -0.26102070f, 0.20711737f, 0.58605883f}};

        const auto px4Pack = metalrobo::builtinRobotPack("px4_x500");
        if (!px4Pack.has_value()) {
            throw std::runtime_error("PX4 X500 built-in robot is unavailable");
        }
        if (options.starter) {
            const auto urdf = options.workspace /
                "assets/window/px4-x500-starter.urdf";
            metalrobo::VisualAssetCookOptions visual;
            visual.license = "BSD-3-Clause";
            visual.preprocessingProvenance = "PX4-gazebo-models e00d3b9cde682dbcb3bf6f30a2f2b8ef4325dae8";
            cookPresentation(urdf, px4Pack->mechanics, "px4-x500", "world",
                options.output, {0.78f, -1.03f, 2.59f}, cameraQ, {}, std::move(visual));
            writeSurfacePresentation(options.output, "px4-x500-visual-observation.json",
                "ground", "locomotion_ground", surfacePresentation(false));
            writeX500FlightPreview(options.output);
            writeCatalogVersion(options.output, "starter-2");
            std::cout << "wrote Numi Window starter catalog "
                      << options.output << '\n';
            return 0;
        }

        const EngineModel g1 = metalrobo::makeUnitreeG1EngineModel();
        const auto g1Source = options.workspace /
            "build/unitree_ros/robots/g1_description/g1_29dof.urdf";
        const auto g1URDF = resolvedURDF(
            g1Source, options.output / "unitree-g1-visual.urdf");
        cookPresentation(g1URDF, g1, "unitree-g1", "pelvis",
            options.output, {1.05f, -1.30f, 0.32f}, cameraQ);
        writeSurfacePresentation(
            options.output,
            "unitree-g1-visual-observation.json",
            "ground",
            "locomotion_ground",
            surfacePresentation(false)
        );
        writeSurfacePresentation(
            options.output,
            "unitree-g1-visual-observation.json",
            "terrain",
            "locomotion_terrain",
            surfacePresentation(true)
        );
        constexpr std::array<float, 6u> dodgeRadii{
            0.10f, 0.10f, 0.10f, 0.10f, 0.10f, 0.10f};
        constexpr std::array<float, 6u> recoveryRadii{
            0.10f, 0.12f, 0.14f, 0.16f, 0.18f, 0.20f};
        for (const auto& surface : {"ground", "terrain"}) {
            writeBallPresentation(
                options.output, surface, "ball-dodge", dodgeRadii);
            writeBallPresentation(
                options.output, surface, "ball-recovery", recoveryRadii);
        }
        const std::vector<std::pair<std::string, std::string>> g1Scenes{{
            "velocity", "Stand & Follow Velocity Commands"},
            {"adult-locomotion", "Walk with Disturbances"},
            {"g1-legs-locomotion", "12-Joint Legs Locomotion"},
            {"disturbance-recovery", "Recover from Pushes"},
            {"supine-get-up", "Learn to Get Up from the Floor"},
            {"developmental-recovery", "Tuck, Brace & Recover"},
            {"ball-recovery", "Recover from Ball Impacts"},
            {"ball-dodge", "See and Dodge Balls"},
        };
        for (const auto& [surface, surfaceName] :
             std::vector<std::pair<std::string, std::string>>{{
                 "ground", "Flat Ground"}, {"terrain", "Uneven Terrain"}}) {
            for (const auto& [task, name] : g1Scenes) {
                const bool hasProjectiles =
                    task == "ball-recovery" || task == "ball-dodge";
                scene(options.output,
                    "unitree-g1-" + surface + "-" + task,
                    "unitree-g1", "Unitree G1",
                    surface, surfaceName, task, name,
                    "unitree-g1-" + surface +
                        (hasProjectiles ? "-" + task : "") +
                        "-visual-observation.json",
                    {"--robot-source", "unitree-g1", "--scene", surface,
                     "--task", task, "--zero-actions"},
                    {"--scene", surface, "--task", task});
            }
        }

        const auto numiflyWingDirectory = options.workspace /
            "assets/numifly/maeda-wing-pack-v1";
        const auto wingManifest = std::filesystem::relative(
            numiflyWingDirectory / "manifest.json", options.output).string();
        const auto numiflyPack = metalrobo::makeNumiflyRobotPack(
            metalrobo::loadNumiflyMaedaWingPack(
                numiflyWingDirectory / "manifest.json"));
        const auto numiflyURDF = numiflyVisualURDF(
            g1URDF,
            numiflyWingDirectory / "numifly-maeda-wings-phase0.stl",
            options.output / "numifly-visual.urdf");
        cookPresentation(
            numiflyURDF, numiflyPack.mechanics, "numifly", "pelvis",
            options.output,
            {1.05f * metalrobo::kNumiflyLinearScale,
             -1.30f * metalrobo::kNumiflyLinearScale,
             0.32f * metalrobo::kNumiflyLinearScale},
            cameraQ,
            "numifly-maeda-wings-phase0.stl");
        writeSurfacePresentation(
            options.output,
            "numifly-visual-observation.json",
            "ground",
            "locomotion_ground",
            surfacePresentation(false));
        scene(
            options.output,
            "numifly-ground-flight",
            "numifly", "Numifly",
            "ground", "Flight Test Ground",
            "numifly-flight", "Learn Bilateral Wing Flight",
            "numifly-ground-visual-observation.json",
            {"--robot-source", "numifly", "--scene", "ground",
             "--task", "numifly-flight", "--numifly-wing-manifest",
             wingManifest,
             "--zero-actions"},
            {"--scene", "ground", "--task", "numifly-flight",
             "--numifly-wing-manifest",
             wingManifest});
        scene(
            options.output,
            "numifly-ground-forward-flight",
            "numifly", "Numifly",
            "ground", "Flight Test Ground",
            "numifly-forward-flight", "Learn Forward Flight and Locomotion",
            "numifly-ground-visual-observation.json",
            {"--robot-source", "numifly-forward-flight", "--scene", "ground",
             "--task", "numifly-forward-flight", "--numifly-wing-manifest",
             wingManifest,
             "--zero-actions"},
            {"--scene", "ground", "--task", "numifly-forward-flight",
             "--numifly-wing-manifest",
             wingManifest});

        const auto numiflyNoLegsPack =
            metalrobo::makeNumiflyNoLegsRobotPack(
                metalrobo::loadNumiflyMaedaWingPack(
                    numiflyWingDirectory / "manifest.json"));
        const auto numiflyNoLegsURDF = numiflyVisualURDF(
            g1URDF,
            numiflyWingDirectory / "numifly-maeda-wings-phase0.stl",
            options.output / "numifly-no-legs-visual.urdf",
            true);
        cookPresentation(
            numiflyNoLegsURDF, numiflyNoLegsPack.mechanics,
            "numifly-no-legs", "pelvis", options.output,
            {1.05f * metalrobo::kNumiflyLinearScale,
             -1.30f * metalrobo::kNumiflyLinearScale,
             0.32f * metalrobo::kNumiflyLinearScale},
            cameraQ,
            "numifly-maeda-wings-phase0.stl");
        writeSurfacePresentation(
            options.output,
            "numifly-no-legs-visual-observation.json",
            "ground",
            "locomotion_ground",
            surfacePresentation(false));
        scene(
            options.output,
            "numifly-no-legs-ground-flight",
            "numifly-no-legs", "Numifly No Legs",
            "ground", "Flight Test Ground",
            "numifly-no-legs-flight", "Learn Legless Wing Flight",
            "numifly-no-legs-ground-visual-observation.json",
            {"--robot-source", "numifly-no-legs", "--scene", "ground",
             "--task", "numifly-no-legs-flight",
             "--numifly-wing-manifest",
             wingManifest,
             "--zero-actions"},
            {"--scene", "ground", "--task",
             "numifly-no-legs-flight", "--numifly-wing-manifest",
             wingManifest});

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
            "Franka Panda + Hand", "workcell", "Manipulation Workcell",
            "pick-place", "Pick & Place",
            "franka-panda-visual-observation.json",
            {"--robot-source", "franka-panda", "--scene", "ground",
             "--zero-actions"});
        scene(options.output, "franka-hand-motion", "franka-panda",
            "Franka Panda + Hand", "workcell", "Manipulation Workcell",
            "hand-motion", "Hand Motion",
            "franka-panda-visual-observation.json",
            {"--robot-source", "franka-panda", "--scene", "ground",
             "--zero-actions"});

        const auto px4Meshes = options.workspace /
            ".numi/runs/px4-x500-source/source/models/x500_base/meshes";
        const auto px4URDF = options.output / "px4-x500-visual.urdf";
        px4VisualURDF(px4URDF, px4Meshes / "1345_prop_cw.stl",
            px4Meshes / "1345_prop_ccw.stl");
        cookPresentation(px4URDF, px4Pack->mechanics, "px4-x500", "x500_base",
            options.output, {0.35f, -0.46f, 0.16f}, cameraQ);
        writeX500HoverScene(options.output);
        scene(options.output, "px4-x500-motor-sweep", "px4-x500",
            "PX4 X500 Drone", "flight-area", "Flight Test Area",
            "motor-sweep", "Motor Sweep",
            "px4-x500-visual-observation.json",
            {"--robot-source", "px4-x500", "--scene", "ground",
             "--zero-actions"});

        writeCatalogVersion(options.output, "10");
        std::cout << "wrote Numi Window built-in catalog "
                  << options.output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Numi Window catalog compile failed: "
                  << error.what() << '\n';
        return 1;
    }
    return 0;
}

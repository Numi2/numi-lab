#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/G1.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include "metalrobo/RunProgram.hpp"
#include "metalrobo/VisualPresentation.hpp"

#include <libxml/parser.h>
#include <libxml/tree.h>

#include <array>
#include <cmath>
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
    std::ofstream target{
        output / ("unitree-g1-" + surface + "-visual-observation.json")};
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
           << "  \"id\": \"" << id << "\",\n"
           << "  \"robot_id\": \"" << robotID << "\",\n"
           << "  \"robot_name\": \"" << robotName << "\",\n"
           << "  \"scene_id\": \"" << sceneID << "\",\n"
           << "  \"scene_name\": \"" << sceneName << "\",\n"
           << "  \"task_id\": \"" << taskID << "\",\n"
           << "  \"task_name\": \"" << taskName << "\",\n"
           << "  \"visual_observation\": \"" << visual << "\",\n"
           << "  \"arguments\": [";
    for (std::size_t index = 0u; index < arguments.size(); ++index) {
        stream << (index == 0u ? "\n    " : ",\n    ")
               << "\"" << arguments[index] << "\"";
    }
    stream << "\n  ]";
    if (!trainingArguments.empty()) {
        stream << ",\n  \"training_arguments\": [";
        for (std::size_t index = 0u;
             index < trainingArguments.size(); ++index) {
            stream << (index == 0u ? "\n    " : ",\n    ")
                   << "\"" << trainingArguments[index] << "\"";
        }
        stream << "\n  ]";
    }
    stream << "\n}\n";
}

} // namespace

int main(const int argc, const char* const* argv) {
    try {
        const Arguments options = arguments(argc, argv);
        std::filesystem::create_directories(options.output);
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
            "flight-area", "Flight Test Area", "hover", "Hover Task",
            "px4-x500-visual-observation.json",
            {"--robot-source", "px4-x500", "--scene", "ground",
             "--zero-actions"});
        scene(options.output, "px4-x500-motor-sweep", "px4-x500",
            "PX4 X500 Drone", "flight-area", "Flight Test Area",
            "motor-sweep", "Motor Sweep",
            "px4-x500-visual-observation.json",
            {"--robot-source", "px4-x500", "--scene", "ground",
             "--zero-actions"});

        {
            std::ofstream version{options.output / "catalog.version"};
            version << "4\n";
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

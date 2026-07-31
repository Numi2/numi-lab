#include "metalrobo/RobotDescriptionCooker.hpp"
#include "metalrobo/ConstraintIR.hpp"
#include "metalrobo/GeometryCooker.hpp"

#include <libxml/parser.h>
#include <libxml/tree.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFnvOffset =
    14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Mat3 {
    double m[3][3]{};
};

struct Transform {
    Vec3 translation{};
    Mat3 rotation{{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
};

enum class ParsedShapeKind {
    sphere,
    box,
    cylinder,
    capsule,
    convexMesh,
};

struct ParsedShape {
    ParsedShapeKind kind = ParsedShapeKind::sphere;
    Transform origin{};
    Vec3 dimensions{};
    std::uint32_t meshAsset = MR_INVALID_INDEX;
};

struct ParsedMeshAsset {
    std::filesystem::path resolvedPath;
    std::string sourceBytes;
    std::vector<mr_float4> vertices;
    std::vector<std::uint32_t> indices;
    std::uint32_t geometryIndex = MR_INVALID_INDEX;
};

struct ParsedLink {
    std::string name;
    double mass = 0.0;
    Vec3 centerOfMass{};
    Mat3 inertia{};
    std::vector<ParsedShape> collisions;
};

struct ParsedJoint {
    std::string name;
    std::string parent;
    std::string child;
    mr_u32 type = MR_JOINT_FIXED;
    Transform origin{};
    Vec3 axis{1.0, 0.0, 0.0};
    bool hasPositionLimit = false;
    double lower = 0.0;
    double upper = 0.0;
    double velocity = 0.0;
    double effort = 0.0;
    double damping = 0.0;
    double friction = 0.0;
    std::string mimicJoint;
    double mimicMultiplier = 1.0;
    double mimicOffset = 0.0;
};

using XmlDocument = std::unique_ptr<
    xmlDoc,
    decltype(&xmlFreeDoc)
>;

RobotDescriptionDiagnostics fail(
    RobotDescriptionDiagnostics diagnostics,
    const RobotDescriptionStatus status,
    std::string message,
    std::string element = {}
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    diagnostics.element = std::move(element);
    return diagnostics;
}

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const Vec3 value) {
    return finite(value.x) &&
        finite(value.y) &&
        finite(value.z);
}

bool finite(const Mat3& value) {
    for (const auto& row : value.m) {
        for (const double entry : row) {
            if (!finite(entry)) {
                return false;
            }
        }
    }
    return true;
}

std::string nodeName(const xmlNode* node) {
    if (node == nullptr || node->name == nullptr) {
        return {};
    }
    return reinterpret_cast<const char*>(node->name);
}

std::optional<std::string> property(
    const xmlNode* node,
    const char* name
) {
    xmlChar* value = xmlGetProp(
        const_cast<xmlNode*>(node),
        reinterpret_cast<const xmlChar*>(name)
    );
    if (value == nullptr) {
        return std::nullopt;
    }
    std::string result{
        reinterpret_cast<const char*>(value)
    };
    xmlFree(value);
    return result;
}

xmlNode* firstChild(
    xmlNode* parent,
    const std::string_view name
) {
    if (parent == nullptr) {
        return nullptr;
    }
    for (xmlNode* child = parent->children;
         child != nullptr;
         child = child->next) {
        if (child->type == XML_ELEMENT_NODE &&
            nodeName(child) == name) {
            return child;
        }
    }
    return nullptr;
}

std::vector<xmlNode*> children(
    xmlNode* parent,
    const std::string_view name
) {
    std::vector<xmlNode*> result;
    if (parent == nullptr) {
        return result;
    }
    for (xmlNode* child = parent->children;
         child != nullptr;
         child = child->next) {
        if (child->type == XML_ELEMENT_NODE &&
            nodeName(child) == name) {
            result.push_back(child);
        }
    }
    return result;
}

bool parseDouble(
    const std::string_view text,
    double& value
) {
    if (text.empty()) {
        return false;
    }
    std::string owned{text};
    char* end = nullptr;
    errno = 0;
    const double parsed =
        std::strtod(owned.c_str(), &end);
    if (errno != 0 ||
        end != owned.c_str() + owned.size() ||
        !finite(parsed)) {
        return false;
    }
    value = parsed;
    return true;
}

bool parseVec3(
    const std::string_view text,
    Vec3& value
) {
    std::istringstream stream{std::string{text}};
    Vec3 parsed;
    if (!(stream >> parsed.x >> parsed.y >> parsed.z)) {
        return false;
    }
    stream >> std::ws;
    if (!stream.eof() || !finite(parsed)) {
        return false;
    }
    value = parsed;
    return true;
}

Mat3 transpose(const Mat3& value) {
    Mat3 result{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u;
             column < 3u;
             ++column) {
            result.m[row][column] =
                value.m[column][row];
        }
    }
    return result;
}

Mat3 operator*(const Mat3& left, const Mat3& right) {
    Mat3 result{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u;
             column < 3u;
             ++column) {
            for (std::size_t inner = 0u;
                 inner < 3u;
                 ++inner) {
                result.m[row][column] +=
                    left.m[row][inner] *
                    right.m[inner][column];
            }
        }
    }
    return result;
}

Mat3 rotationFromRpy(const Vec3 rpy) {
    const double cr = std::cos(rpy.x);
    const double sr = std::sin(rpy.x);
    const double cp = std::cos(rpy.y);
    const double sp = std::sin(rpy.y);
    const double cy = std::cos(rpy.z);
    const double sy = std::sin(rpy.z);
    return {{
        {
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
        },
        {
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
        },
        {-sp, cp * sr, cp * cr},
    }};
}

std::array<double, 4> quaternion(const Mat3& matrix) {
    std::array<double, 4> result{};
    const double trace =
        matrix.m[0][0] +
        matrix.m[1][1] +
        matrix.m[2][2];
    if (trace > 0.0) {
        const double scale =
            2.0 * std::sqrt(trace + 1.0);
        result = {
            (matrix.m[2][1] - matrix.m[1][2]) / scale,
            (matrix.m[0][2] - matrix.m[2][0]) / scale,
            (matrix.m[1][0] - matrix.m[0][1]) / scale,
            0.25 * scale,
        };
    } else if (matrix.m[0][0] > matrix.m[1][1] &&
               matrix.m[0][0] > matrix.m[2][2]) {
        const double scale = 2.0 * std::sqrt(
            1.0 + matrix.m[0][0] -
                matrix.m[1][1] - matrix.m[2][2]
        );
        result = {
            0.25 * scale,
            (matrix.m[0][1] + matrix.m[1][0]) / scale,
            (matrix.m[0][2] + matrix.m[2][0]) / scale,
            (matrix.m[2][1] - matrix.m[1][2]) / scale,
        };
    } else if (matrix.m[1][1] > matrix.m[2][2]) {
        const double scale = 2.0 * std::sqrt(
            1.0 + matrix.m[1][1] -
                matrix.m[0][0] - matrix.m[2][2]
        );
        result = {
            (matrix.m[0][1] + matrix.m[1][0]) / scale,
            0.25 * scale,
            (matrix.m[1][2] + matrix.m[2][1]) / scale,
            (matrix.m[0][2] - matrix.m[2][0]) / scale,
        };
    } else {
        const double scale = 2.0 * std::sqrt(
            1.0 + matrix.m[2][2] -
                matrix.m[0][0] - matrix.m[1][1]
        );
        result = {
            (matrix.m[0][2] + matrix.m[2][0]) / scale,
            (matrix.m[1][2] + matrix.m[2][1]) / scale,
            0.25 * scale,
            (matrix.m[1][0] - matrix.m[0][1]) / scale,
        };
    }
    const double norm = std::sqrt(
        result[0] * result[0] +
        result[1] * result[1] +
        result[2] * result[2] +
        result[3] * result[3]
    );
    for (double& component : result) {
        component /= norm;
    }
    if (result[3] < 0.0) {
        for (double& component : result) {
            component = -component;
        }
    }
    return result;
}

double determinant(const Mat3& value) {
    return
        value.m[0][0] *
            (
                value.m[1][1] * value.m[2][2] -
                value.m[1][2] * value.m[2][1]
            ) -
        value.m[0][1] *
            (
                value.m[1][0] * value.m[2][2] -
                value.m[1][2] * value.m[2][0]
            ) +
        value.m[0][2] *
            (
                value.m[1][0] * value.m[2][1] -
                value.m[1][1] * value.m[2][0]
            );
}

bool positiveDefinite(const Mat3& value) {
    return
        value.m[0][0] > 0.0 &&
        value.m[0][0] * value.m[1][1] -
            value.m[0][1] * value.m[1][0] > 0.0 &&
        determinant(value) > 0.0;
}

bool inverse(const Mat3& value, Mat3& result) {
    const double det = determinant(value);
    if (!(det > 0.0) || !finite(det)) {
        return false;
    }
    result.m[0][0] =
        (value.m[1][1] * value.m[2][2] -
         value.m[1][2] * value.m[2][1]) / det;
    result.m[0][1] =
        (value.m[0][2] * value.m[2][1] -
         value.m[0][1] * value.m[2][2]) / det;
    result.m[0][2] =
        (value.m[0][1] * value.m[1][2] -
         value.m[0][2] * value.m[1][1]) / det;
    result.m[1][0] =
        (value.m[1][2] * value.m[2][0] -
         value.m[1][0] * value.m[2][2]) / det;
    result.m[1][1] =
        (value.m[0][0] * value.m[2][2] -
         value.m[0][2] * value.m[2][0]) / det;
    result.m[1][2] =
        (value.m[0][2] * value.m[1][0] -
         value.m[0][0] * value.m[1][2]) / det;
    result.m[2][0] =
        (value.m[1][0] * value.m[2][1] -
         value.m[1][1] * value.m[2][0]) / det;
    result.m[2][1] =
        (value.m[0][1] * value.m[2][0] -
         value.m[0][0] * value.m[2][1]) / det;
    result.m[2][2] =
        (value.m[0][0] * value.m[1][1] -
         value.m[0][1] * value.m[1][0]) / det;
    return finite(result);
}

mr_float4 f4(
    const double x,
    const double y,
    const double z,
    const double w = 0.0
) {
    return {
        static_cast<float>(x),
        static_cast<float>(y),
        static_cast<float>(z),
        static_cast<float>(w),
    };
}

mr_float4 row(const Mat3& matrix, const std::size_t index) {
    return f4(
        matrix.m[index][0],
        matrix.m[index][1],
        matrix.m[index][2]
    );
}

bool parseOrigin(xmlNode* parent, Transform& transform) {
    xmlNode* origin = firstChild(parent, "origin");
    if (origin == nullptr) {
        return true;
    }
    if (const auto xyz = property(origin, "xyz");
        xyz.has_value() &&
        !parseVec3(*xyz, transform.translation)) {
        return false;
    }
    Vec3 rpy{};
    if (const auto value = property(origin, "rpy");
        value.has_value() && !parseVec3(*value, rpy)) {
        return false;
    }
    transform.rotation = rotationFromRpy(rpy);
    return true;
}

std::uint64_t sourceFingerprint(
    const std::string_view urdf,
    const std::string_view srdf,
    const RobotDescriptionCookOptions& options
) {
    std::uint64_t hash = kFnvOffset;
    const auto append = [&hash](const void* data, const std::size_t size) {
        const auto* bytes =
            static_cast<const unsigned char*>(data);
        for (std::size_t index = 0u; index < size; ++index) {
            hash ^= bytes[index];
            hash *= kFnvPrime;
        }
    };
    append(urdf.data(), urdf.size());
    append(srdf.data(), srdf.size());
    const auto rootMode =
        static_cast<std::uint32_t>(options.rootMode);
    const auto meshMode =
        static_cast<std::uint32_t>(options.meshMode);
    const std::uint32_t actuated =
        options.actuateMovableJoints ? 1u : 0u;
    const std::uint32_t respectTransmissions =
        options.respectTransmissions ? 1u : 0u;
    append(&rootMode, sizeof(rootMode));
    append(&meshMode, sizeof(meshMode));
    append(
        &options.gravityAndTimestep,
        sizeof(options.gravityAndTimestep)
    );
    append(&options.friction, sizeof(options.friction));
    append(&options.response, sizeof(options.response));
    append(&options.contactOffset, sizeof(options.contactOffset));
    append(&options.restOffset, sizeof(options.restOffset));
    append(&options.defaultArmature, sizeof(options.defaultArmature));
    append(&actuated, sizeof(actuated));
    append(
        &respectTransmissions,
        sizeof(respectTransmissions)
    );
    append(
        &options.collisionGroup,
        sizeof(options.collisionGroup)
    );
    append(
        &options.collisionMask,
        sizeof(options.collisionMask)
    );
    return hash;
}

std::optional<std::string> readFile(
    const std::filesystem::path& path
) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return std::nullopt;
    }
    std::ostringstream output;
    output << stream.rdbuf();
    if (!stream.good() && !stream.eof()) {
        return std::nullopt;
    }
    return output.str();
}

void appendFingerprint(
    std::uint64_t& hash,
    const std::string_view bytes
) {
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= kFnvPrime;
    }
}

void appendFingerprintSize(
    std::uint64_t& hash,
    const std::uint64_t value
) {
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        hash ^= static_cast<unsigned char>(
            value >> (8u * byte)
        );
        hash *= kFnvPrime;
    }
}

bool hasUriScheme(const std::string_view value) {
    return value.find("://") != std::string_view::npos;
}

std::filesystem::path effectiveMeshRoot(
    const RobotDescriptionCookOptions& options,
    const std::string& sourceName
) {
    if (!options.meshAssetRoot.empty()) {
        return options.meshAssetRoot;
    }
    if (sourceName.empty() || hasUriScheme(sourceName)) {
        return {};
    }
    return std::filesystem::path(sourceName).parent_path();
}

bool regularFile(
    const std::filesystem::path& candidate,
    std::filesystem::path& resolved
) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(candidate, error) ||
        error) {
        return false;
    }
    resolved = std::filesystem::weakly_canonical(
        candidate,
        error
    );
    return !error;
}

bool resolveMeshUri(
    const std::string_view uri,
    const RobotDescriptionCookOptions& options,
    const std::string& sourceName,
    std::filesystem::path& resolved,
    std::string& message
) {
    constexpr std::string_view packagePrefix =
        "package://";
    constexpr std::string_view filePrefix = "file://";
    if (uri.starts_with(packagePrefix)) {
        const std::string_view payload =
            uri.substr(packagePrefix.size());
        const std::size_t separator = payload.find('/');
        if (separator == std::string_view::npos ||
            separator == 0u ||
            separator + 1u >= payload.size()) {
            message = "package mesh URI must contain package and path";
            return false;
        }
        const std::filesystem::path package{
            payload.substr(0u, separator)
        };
        const std::filesystem::path relative{
            payload.substr(separator + 1u)
        };
        if (relative.is_absolute()) {
            message = "package mesh path must be relative";
            return false;
        }
        std::vector<std::filesystem::path> roots =
            options.packageSearchRoots;
        const std::filesystem::path assetRoot =
            effectiveMeshRoot(options, sourceName);
        if (!assetRoot.empty()) {
            roots.push_back(assetRoot);
        }
        for (const std::filesystem::path& root : roots) {
            if (regularFile(
                    root / package / relative,
                    resolved
                )) {
                return true;
            }
            if (root.filename() == package &&
                regularFile(root / relative, resolved)) {
                return true;
            }
        }
        message = "package mesh URI was not found in configured roots";
        return false;
    }
    if (uri.starts_with(filePrefix)) {
        const std::filesystem::path absolute{
            uri.substr(filePrefix.size())
        };
        if (!absolute.is_absolute()) {
            message = "file mesh URI must be absolute";
            return false;
        }
        if (regularFile(absolute, resolved)) {
            return true;
        }
        message = "file mesh URI does not name a regular file";
        return false;
    }
    if (hasUriScheme(uri)) {
        message = "mesh URI scheme is unsupported";
        return false;
    }
    const std::filesystem::path authored{uri};
    if (authored.is_absolute()) {
        if (regularFile(authored, resolved)) {
            return true;
        }
        message = "absolute mesh path does not name a regular file";
        return false;
    }
    const std::filesystem::path root =
        effectiveMeshRoot(options, sourceName);
    if (root.empty()) {
        message =
            "relative mesh URI requires meshAssetRoot or a file source name";
        return false;
    }
    if (regularFile(root / authored, resolved)) {
        return true;
    }
    message = "relative mesh URI was not found under the asset root";
    return false;
}

bool parseObjIndex(
    const std::string_view token,
    const std::size_t vertexCount,
    std::uint32_t& index
) {
    const std::size_t separator = token.find('/');
    const std::string_view vertex =
        token.substr(0u, separator);
    int parsed = 0;
    const auto [end, error] = std::from_chars(
        vertex.data(),
        vertex.data() + vertex.size(),
        parsed
    );
    if (error != std::errc{} ||
        end != vertex.data() + vertex.size() ||
        parsed == 0) {
        return false;
    }
    const std::int64_t resolved = parsed > 0
        ? static_cast<std::int64_t>(parsed - 1)
        : static_cast<std::int64_t>(vertexCount) + parsed;
    if (resolved < 0 ||
        resolved >= static_cast<std::int64_t>(vertexCount)) {
        return false;
    }
    index = static_cast<std::uint32_t>(resolved);
    return true;
}

bool parseObj(
    const std::string& source,
    ParsedMeshAsset& mesh,
    std::string& message
) {
    std::istringstream lines(source);
    std::string line;
    std::size_t lineNumber = 0u;
    while (std::getline(lines, line)) {
        ++lineNumber;
        if (const std::size_t comment = line.find('#');
            comment != std::string::npos) {
            line.resize(comment);
        }
        std::istringstream tokens(line);
        std::string kind;
        if (!(tokens >> kind)) {
            continue;
        }
        if (kind == "v") {
            std::string x;
            std::string y;
            std::string z;
            Vec3 vertex;
            if (!(tokens >> x >> y >> z) ||
                !parseDouble(x, vertex.x) ||
                !parseDouble(y, vertex.y) ||
                !parseDouble(z, vertex.z) ||
                mesh.vertices.size() >=
                    std::numeric_limits<std::uint32_t>::max()) {
                message = "invalid OBJ vertex at line " +
                    std::to_string(lineNumber);
                return false;
            }
            mesh.vertices.push_back(f4(
                vertex.x,
                vertex.y,
                vertex.z,
                1.0
            ));
        } else if (kind == "f") {
            std::vector<std::uint32_t> polygon;
            std::string token;
            while (tokens >> token) {
                std::uint32_t index = 0u;
                if (!parseObjIndex(
                        token,
                        mesh.vertices.size(),
                        index
                    )) {
                    message = "invalid OBJ face at line " +
                        std::to_string(lineNumber);
                    return false;
                }
                polygon.push_back(index);
            }
            if (polygon.size() < 3u) {
                message = "OBJ face has fewer than three vertices at line " +
                    std::to_string(lineNumber);
                return false;
            }
            const std::uint64_t addedIndices =
                3ull * (polygon.size() - 2u);
            if (mesh.indices.size() >
                    std::numeric_limits<std::uint32_t>::max() ||
                addedIndices >
                    std::numeric_limits<std::uint32_t>::max() -
                        mesh.indices.size()) {
                message = "OBJ triangle count exceeds the cooked ABI";
                return false;
            }
            for (std::size_t corner = 1u;
                 corner + 1u < polygon.size();
                 ++corner) {
                mesh.indices.push_back(polygon[0]);
                mesh.indices.push_back(polygon[corner]);
                mesh.indices.push_back(polygon[corner + 1u]);
            }
        }
    }
    if (mesh.vertices.size() < 4u ||
        mesh.indices.size() < 12u) {
        message = "OBJ has no closed convex surface";
        return false;
    }
    return true;
}

std::uint32_t littleU32(
    const unsigned char* bytes
) {
    return static_cast<std::uint32_t>(bytes[0]) |
        (static_cast<std::uint32_t>(bytes[1]) << 8u) |
        (static_cast<std::uint32_t>(bytes[2]) << 16u) |
        (static_cast<std::uint32_t>(bytes[3]) << 24u);
}

float littleFloat(const unsigned char* bytes) {
    return std::bit_cast<float>(littleU32(bytes));
}

bool parseBinaryStl(
    const std::string& source,
    const std::uint32_t triangleCount,
    ParsedMeshAsset& mesh,
    std::string& message
) {
    if (triangleCount >
        std::numeric_limits<std::uint32_t>::max() / 3u) {
        message = "binary STL triangle count exceeds the cooked ABI";
        return false;
    }
    mesh.vertices.reserve(
        static_cast<std::size_t>(triangleCount) * 3u
    );
    mesh.indices.reserve(
        static_cast<std::size_t>(triangleCount) * 3u
    );
    const auto* bytes = reinterpret_cast<
        const unsigned char*
    >(source.data());
    for (std::uint32_t triangle = 0u;
         triangle < triangleCount;
         ++triangle) {
        const std::size_t vertexBase =
            84u + 50u * triangle + 12u;
        for (std::uint32_t corner = 0u;
             corner < 3u;
             ++corner) {
            const std::size_t offset =
                vertexBase + 12u * corner;
            const Vec3 vertex{
                littleFloat(bytes + offset),
                littleFloat(bytes + offset + 4u),
                littleFloat(bytes + offset + 8u),
            };
            if (!finite(vertex)) {
                message = "binary STL contains a non-finite vertex";
                return false;
            }
            mesh.indices.push_back(
                static_cast<std::uint32_t>(
                    mesh.vertices.size()
                )
            );
            mesh.vertices.push_back(f4(
                vertex.x,
                vertex.y,
                vertex.z,
                1.0
            ));
        }
    }
    if (triangleCount < 4u) {
        message = "binary STL has no closed convex surface";
        return false;
    }
    return true;
}

bool parseAsciiStl(
    const std::string& source,
    ParsedMeshAsset& mesh,
    std::string& message
) {
    std::istringstream tokens(source);
    std::string token;
    while (tokens >> token) {
        if (token != "vertex") {
            continue;
        }
        std::string x;
        std::string y;
        std::string z;
        Vec3 vertex;
        if (!(tokens >> x >> y >> z) ||
            !parseDouble(x, vertex.x) ||
            !parseDouble(y, vertex.y) ||
            !parseDouble(z, vertex.z) ||
            mesh.vertices.size() >=
                std::numeric_limits<std::uint32_t>::max()) {
            message = "ASCII STL contains an invalid vertex";
            return false;
        }
        mesh.indices.push_back(
            static_cast<std::uint32_t>(mesh.vertices.size())
        );
        mesh.vertices.push_back(f4(
            vertex.x,
            vertex.y,
            vertex.z,
            1.0
        ));
    }
    if (mesh.vertices.size() < 12u ||
        mesh.vertices.size() % 3u != 0u) {
        message = "ASCII STL has no complete closed surface";
        return false;
    }
    return true;
}

bool parseStl(
    const std::string& source,
    ParsedMeshAsset& mesh,
    std::string& message
) {
    if (source.size() >= 84u) {
        const auto* bytes = reinterpret_cast<
            const unsigned char*
        >(source.data());
        const std::uint32_t count =
            littleU32(bytes + 80u);
        const std::uint64_t expected =
            84ull + 50ull * count;
        if (expected == source.size()) {
            return parseBinaryStl(
                source,
                count,
                mesh,
                message
            );
        }
    }
    return parseAsciiStl(source, mesh, message);
}

bool loadResolvedMeshAsset(
    const std::filesystem::path& resolvedPath,
    ParsedMeshAsset& mesh,
    std::string& message
) {
    mesh.resolvedPath = resolvedPath;
    const std::optional<std::string> loaded =
        readFile(mesh.resolvedPath);
    constexpr std::size_t maximumMeshBytes =
        256u * 1024u * 1024u;
    if (!loaded.has_value()) {
        message = "failed to read resolved mesh";
        return false;
    }
    if (loaded->size() > maximumMeshBytes) {
        message = "mesh exceeds the 256 MiB cooker limit";
        return false;
    }
    mesh.sourceBytes = *loaded;
    std::string extension =
        mesh.resolvedPath.extension().string();
    std::ranges::transform(
        extension,
        extension.begin(),
        [](const unsigned char value) {
            return static_cast<char>(std::tolower(value));
        }
    );
    if (extension == ".obj") {
        return parseObj(mesh.sourceBytes, mesh, message);
    }
    if (extension == ".stl") {
        return parseStl(mesh.sourceBytes, mesh, message);
    }
    message =
        "only OBJ and STL articulated collision meshes are supported";
    return false;
}

} // namespace

RobotDescriptionDiagnostics cookRobotDescription(
    const std::string_view urdf,
    const std::string_view srdf,
    EngineModel& output,
    const RobotDescriptionCookOptions& options,
    std::string sourceName
) {
    RobotDescriptionDiagnostics diagnostics;
    diagnostics.sourceName = std::move(sourceName);
    diagnostics.sourceFingerprint =
        sourceFingerprint(urdf, srdf, options);
    try {
        if (urdf.empty() ||
            urdf.size() >
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max()
                ) ||
            srdf.size() >
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max()
                )) {
            return fail(
                std::move(diagnostics),
                RobotDescriptionStatus::malformedXml,
                "robot description is empty or exceeds XML parser bounds"
            );
        }
        constexpr int parseFlags =
            XML_PARSE_NONET |
            XML_PARSE_NOBLANKS |
            XML_PARSE_NOERROR |
            XML_PARSE_NOWARNING |
            XML_PARSE_COMPACT;
        XmlDocument urdfDocument{
            xmlReadMemory(
                urdf.data(),
                static_cast<int>(urdf.size()),
                diagnostics.sourceName.empty()
                    ? "robot.urdf"
                    : diagnostics.sourceName.c_str(),
                nullptr,
                parseFlags
            ),
            &xmlFreeDoc,
        };
        if (!urdfDocument) {
            return fail(
                std::move(diagnostics),
                RobotDescriptionStatus::malformedXml,
                "URDF XML parsing failed"
            );
        }
        xmlNode* robot = xmlDocGetRootElement(
            urdfDocument.get()
        );
        if (robot == nullptr || nodeName(robot) != "robot") {
            return fail(
                std::move(diagnostics),
                RobotDescriptionStatus::invalidRobot,
                "URDF root must be <robot>"
            );
        }
        const auto robotName = property(robot, "name");
        if (!robotName.has_value() || robotName->empty()) {
            return fail(
                std::move(diagnostics),
                RobotDescriptionStatus::invalidRobot,
                "URDF robot has no name",
                "robot"
            );
        }

        std::vector<ParsedMeshAsset> meshAssets;
        std::map<std::string, std::uint32_t> meshByPath;
        std::map<std::string, ParsedLink> links;
        for (xmlNode* linkNode : children(robot, "link")) {
            const auto name = property(linkNode, "name");
            if (!name.has_value() || name->empty() ||
                links.contains(*name)) {
                return fail(
                    std::move(diagnostics),
                    RobotDescriptionStatus::invalidRobot,
                    "URDF link names must be nonempty and unique",
                    name.value_or("link")
                );
            }
            ParsedLink link;
            link.name = *name;
            xmlNode* inertial =
                firstChild(linkNode, "inertial");
            xmlNode* mass =
                firstChild(inertial, "mass");
            xmlNode* inertia =
                firstChild(inertial, "inertia");
            Transform inertialOrigin;
            const auto massValue = property(mass, "value");
            if (inertial == nullptr ||
                mass == nullptr ||
                inertia == nullptr ||
                !massValue.has_value() ||
                !parseDouble(*massValue, link.mass) ||
                !(link.mass > 0.0) ||
                !parseOrigin(inertial, inertialOrigin)) {
                return fail(
                    std::move(diagnostics),
                    RobotDescriptionStatus::invalidInertial,
                    "every executable link requires a positive inertial",
                    link.name
                );
            }
            link.centerOfMass =
                inertialOrigin.translation;
            Mat3 authored{};
            const std::array<const char*, 6> names{
                "ixx", "ixy", "ixz",
                "iyy", "iyz", "izz",
            };
            std::array<double, 6> values{};
            for (std::size_t index = 0u;
                 index < names.size();
                 ++index) {
                const auto value =
                    property(inertia, names[index]);
                if (!value.has_value() ||
                    !parseDouble(*value, values[index])) {
                    return fail(
                        std::move(diagnostics),
                        RobotDescriptionStatus::invalidInertial,
                        "link inertia tensor is incomplete",
                        link.name
                    );
                }
            }
            authored = {{
                {values[0], values[1], values[2]},
                {values[1], values[3], values[4]},
                {values[2], values[4], values[5]},
            }};
            if (!positiveDefinite(authored)) {
                return fail(
                    std::move(diagnostics),
                    RobotDescriptionStatus::invalidInertial,
                    "link inertia is not positive definite",
                    link.name
                );
            }
            link.inertia =
                inertialOrigin.rotation *
                authored *
                transpose(inertialOrigin.rotation);
            if (!positiveDefinite(link.inertia)) {
                return fail(
                    std::move(diagnostics),
                    RobotDescriptionStatus::invalidInertial,
                    "rotated link inertia is invalid",
                    link.name
                );
            }

            for (xmlNode* collision :
                 children(linkNode, "collision")) {
                ParsedShape shape;
                if (!parseOrigin(collision, shape.origin)) {
                    return fail(
                        std::move(diagnostics),
                        RobotDescriptionStatus::invalidRobot,
                        "collision origin is invalid",
                        link.name
                    );
                }
                xmlNode* geometry =
                    firstChild(collision, "geometry");
                if (geometry == nullptr) {
                    return fail(
                        std::move(diagnostics),
                        RobotDescriptionStatus::invalidRobot,
                        "collision has no geometry",
                        link.name
                    );
                }
                if (xmlNode* sphere =
                        firstChild(geometry, "sphere");
                    sphere != nullptr) {
                    const auto radius =
                        property(sphere, "radius");
                    if (!radius.has_value() ||
                        !parseDouble(
                            *radius,
                            shape.dimensions.x
                        ) ||
                        !(shape.dimensions.x > 0.0)) {
                        return fail(
                            std::move(diagnostics),
                            RobotDescriptionStatus::
                                invalidRobot,
                            "sphere radius is invalid",
                            link.name
                        );
                    }
                    shape.kind = ParsedShapeKind::sphere;
                } else if (xmlNode* box =
                               firstChild(geometry, "box");
                           box != nullptr) {
                    const auto size = property(box, "size");
                    if (!size.has_value() ||
                        !parseVec3(*size, shape.dimensions) ||
                        !(shape.dimensions.x > 0.0) ||
                        !(shape.dimensions.y > 0.0) ||
                        !(shape.dimensions.z > 0.0)) {
                        return fail(
                            std::move(diagnostics),
                            RobotDescriptionStatus::
                                invalidRobot,
                            "box size is invalid",
                            link.name
                        );
                    }
                    shape.dimensions.x *= 0.5;
                    shape.dimensions.y *= 0.5;
                    shape.dimensions.z *= 0.5;
                    shape.kind = ParsedShapeKind::box;
                } else if (xmlNode* cylinder =
                               firstChild(
                                   geometry,
                                   "cylinder"
                               );
                           cylinder != nullptr) {
                    const auto radius =
                        property(cylinder, "radius");
                    const auto length =
                        property(cylinder, "length");
                    if (!radius.has_value() ||
                        !length.has_value() ||
                        !parseDouble(
                            *radius,
                            shape.dimensions.x
                        ) ||
                        !parseDouble(
                            *length,
                            shape.dimensions.y
                        ) ||
                        !(shape.dimensions.x > 0.0) ||
                        !(shape.dimensions.y > 0.0)) {
                        return fail(
                            std::move(diagnostics),
                            RobotDescriptionStatus::
                                invalidRobot,
                            "cylinder dimensions are invalid",
                            link.name
                        );
                    }
                    shape.dimensions.y *= 0.5;
                    shape.kind =
                        ParsedShapeKind::cylinder;
                } else if (xmlNode* capsule =
                               firstChild(
                                   geometry,
                                   "capsule"
                               );
                           capsule != nullptr) {
                    const auto radius =
                        property(capsule, "radius");
                    const auto length =
                        property(capsule, "length");
                    if (!radius.has_value() ||
                        !length.has_value() ||
                        !parseDouble(
                            *radius,
                            shape.dimensions.x
                        ) ||
                        !parseDouble(
                            *length,
                            shape.dimensions.y
                        ) ||
                        !(shape.dimensions.x > 0.0) ||
                        !(shape.dimensions.y >= 0.0)) {
                        return fail(
                            std::move(diagnostics),
                            RobotDescriptionStatus::
                                invalidRobot,
                            "capsule dimensions are invalid",
                            link.name
                        );
                    }
                    shape.dimensions.y *= 0.5;
                    shape.kind =
                        ParsedShapeKind::capsule;
                } else if (xmlNode* mesh =
                               firstChild(geometry, "mesh");
                           mesh != nullptr) {
                    const auto filename =
                        property(mesh, "filename");
                    const auto scale =
                        property(mesh, "scale");
                    shape.dimensions = {1.0, 1.0, 1.0};
                    if (!filename.has_value() ||
                        filename->empty() ||
                        (scale.has_value() &&
                         !parseVec3(
                             *scale,
                             shape.dimensions
                         )) ||
                        !(shape.dimensions.x > 0.0) ||
                        !(shape.dimensions.y > 0.0) ||
                        !(shape.dimensions.z > 0.0)) {
                        return fail(
                            std::move(diagnostics),
                            RobotDescriptionStatus::
                                invalidRobot,
                            "mesh filename or scale is invalid",
                            link.name
                        );
                    }
                    std::filesystem::path resolvedMesh;
                    std::string meshMessage;
                    if (!resolveMeshUri(
                            *filename,
                            options,
                            diagnostics.sourceName,
                            resolvedMesh,
                            meshMessage
                        )) {
                        return fail(
                            std::move(diagnostics),
                            RobotDescriptionStatus::
                                unsupportedGeometry,
                            std::move(meshMessage),
                            link.name
                        );
                    }
                    const std::string key =
                        resolvedMesh.generic_string();
                    const auto existing = meshByPath.find(key);
                    if (existing == meshByPath.end()) {
                        if (meshAssets.size() >=
                            std::numeric_limits<
                                std::uint32_t
                            >::max()) {
                            return fail(
                                std::move(diagnostics),
                                RobotDescriptionStatus::
                                    capacityOverflow,
                                "mesh asset count exceeds the cooked ABI",
                                link.name
                            );
                        }
                        ParsedMeshAsset loaded;
                        if (!loadResolvedMeshAsset(
                                resolvedMesh,
                                loaded,
                                meshMessage
                            )) {
                            return fail(
                                std::move(diagnostics),
                                RobotDescriptionStatus::
                                    unsupportedGeometry,
                                std::move(meshMessage),
                                link.name
                            );
                        }
                        shape.meshAsset =
                            static_cast<std::uint32_t>(
                                meshAssets.size()
                            );
                        meshByPath.emplace(
                            key,
                            shape.meshAsset
                        );
                        meshAssets.push_back(std::move(loaded));
                    } else {
                        shape.meshAsset = existing->second;
                    }
                    appendFingerprintSize(
                        diagnostics.sourceFingerprint,
                        filename->size()
                    );
                    appendFingerprint(
                        diagnostics.sourceFingerprint,
                        *filename
                    );
                    appendFingerprintSize(
                        diagnostics.sourceFingerprint,
                        meshAssets[shape.meshAsset]
                            .sourceBytes.size()
                    );
                    appendFingerprint(
                        diagnostics.sourceFingerprint,
                        meshAssets[shape.meshAsset].sourceBytes
                    );
                    shape.kind =
                        ParsedShapeKind::convexMesh;
                } else {
                    return fail(
                        std::move(diagnostics),
                        RobotDescriptionStatus::
                            unsupportedGeometry,
                        "collision geometry type is unsupported",
                        link.name
                    );
                }
                link.collisions.push_back(shape);
            }
            links.emplace(link.name, std::move(link));
        }
        if (links.empty()) {
            return fail(
                std::move(diagnostics),
                RobotDescriptionStatus::invalidRobot,
                "URDF contains no links"
            );
        }

        std::vector<ParsedJoint> joints;
        std::map<std::string, std::size_t> childJoint;
        std::map<std::string, std::vector<std::size_t>>
            outgoing;
        for (xmlNode* jointNode : children(robot, "joint")) {
            ParsedJoint joint;
            const auto name = property(jointNode, "name");
            const auto type = property(jointNode, "type");
            xmlNode* parentNode =
                firstChild(jointNode, "parent");
            xmlNode* childNode =
                firstChild(jointNode, "child");
            const auto parent =
                property(parentNode, "link");
            const auto child =
                property(childNode, "link");
            if (!name.has_value() || name->empty() ||
                !type.has_value() ||
                !parent.has_value() ||
                !child.has_value() ||
                !links.contains(*parent) ||
                !links.contains(*child) ||
                *parent == *child ||
                childJoint.contains(*child) ||
                std::ranges::any_of(
                    joints,
                    [&name](const ParsedJoint& existing) {
                        return existing.name == *name;
                    }
                )) {
                return fail(
                    std::move(diagnostics),
                    RobotDescriptionStatus::invalidTopology,
                    "joint names/links do not define a tree",
                    name.value_or("joint")
                );
            }
            joint.name = *name;
            joint.parent = *parent;
            joint.child = *child;
            if (*type == "fixed") {
                joint.type = MR_JOINT_FIXED;
            } else if (*type == "revolute") {
                joint.type = MR_JOINT_REVOLUTE;
            } else if (*type == "continuous") {
                joint.type = MR_JOINT_CONTINUOUS;
            } else if (*type == "prismatic") {
                joint.type = MR_JOINT_PRISMATIC;
            } else {
                return fail(
                    std::move(diagnostics),
                    RobotDescriptionStatus::unsupportedJoint,
                    "joint type is not executable",
                    joint.name
                );
            }
            if (!parseOrigin(jointNode, joint.origin)) {
                return fail(
                    std::move(diagnostics),
                    RobotDescriptionStatus::invalidTopology,
                    "joint origin is invalid",
                    joint.name
                );
            }
            if (xmlNode* axis =
                    firstChild(jointNode, "axis");
                axis != nullptr) {
                const auto xyz = property(axis, "xyz");
                if (!xyz.has_value() ||
                    !parseVec3(*xyz, joint.axis)) {
                    return fail(
                        std::move(diagnostics),
                        RobotDescriptionStatus::
                            invalidTopology,
                        "joint axis is invalid",
                        joint.name
                    );
                }
            }
            const double axisNorm = std::sqrt(
                joint.axis.x * joint.axis.x +
                joint.axis.y * joint.axis.y +
                joint.axis.z * joint.axis.z
            );
            if (joint.type != MR_JOINT_FIXED &&
                (!(axisNorm > 0.0) ||
                 !finite(axisNorm))) {
                return fail(
                    std::move(diagnostics),
                    RobotDescriptionStatus::invalidTopology,
                    "joint axis has zero length",
                    joint.name
                );
            }
            if (joint.type != MR_JOINT_FIXED) {
                joint.axis.x /= axisNorm;
                joint.axis.y /= axisNorm;
                joint.axis.z /= axisNorm;
                xmlNode* limit =
                    firstChild(jointNode, "limit");
                if (limit != nullptr) {
                    if (const auto value =
                            property(limit, "velocity");
                        value.has_value() &&
                        !parseDouble(
                            *value,
                            joint.velocity
                        )) {
                        return fail(
                            std::move(diagnostics),
                            RobotDescriptionStatus::
                                invalidTopology,
                            "joint velocity limit is invalid",
                            joint.name
                        );
                    }
                    if (const auto value =
                            property(limit, "effort");
                        value.has_value() &&
                        !parseDouble(
                            *value,
                            joint.effort
                        )) {
                        return fail(
                            std::move(diagnostics),
                            RobotDescriptionStatus::
                                invalidTopology,
                            "joint effort limit is invalid",
                            joint.name
                        );
                    }
                    if (joint.type !=
                        MR_JOINT_CONTINUOUS) {
                        const auto lower =
                            property(limit, "lower");
                        const auto upper =
                            property(limit, "upper");
                        if (!lower.has_value() ||
                            !upper.has_value() ||
                            !parseDouble(
                                *lower,
                                joint.lower
                            ) ||
                            !parseDouble(
                                *upper,
                                joint.upper
                            ) ||
                            joint.lower > joint.upper) {
                            return fail(
                                std::move(diagnostics),
                                RobotDescriptionStatus::
                                    invalidTopology,
                                "joint position limits are invalid",
                                joint.name
                            );
                        }
                        joint.hasPositionLimit = true;
                    }
                } else if (joint.type !=
                           MR_JOINT_CONTINUOUS) {
                    return fail(
                        std::move(diagnostics),
                        RobotDescriptionStatus::
                            invalidTopology,
                        "bounded joint has no limit",
                        joint.name
                    );
                }
                if (xmlNode* dynamics =
                        firstChild(jointNode, "dynamics");
                    dynamics != nullptr) {
                    if (const auto value =
                            property(dynamics, "damping");
                        value.has_value() &&
                        (!parseDouble(
                             *value,
                             joint.damping
                         ) ||
                         joint.damping < 0.0)) {
                        return fail(
                            std::move(diagnostics),
                            RobotDescriptionStatus::
                                invalidTopology,
                            "joint damping is invalid",
                            joint.name
                        );
                    }
                    if (const auto value =
                            property(dynamics, "friction");
                        value.has_value() &&
                        (!parseDouble(
                             *value,
                             joint.friction
                         ) ||
                         joint.friction < 0.0)) {
                        return fail(
                            std::move(diagnostics),
                            RobotDescriptionStatus::
                                invalidTopology,
                            "joint friction is invalid",
                            joint.name
                        );
                    }
                }
            }
            if (xmlNode* mimic =
                    firstChild(jointNode, "mimic");
                mimic != nullptr) {
                const auto source =
                    property(mimic, "joint");
                if (joint.type == MR_JOINT_FIXED ||
                    !source.has_value() ||
                    source->empty()) {
                    return fail(
                        std::move(diagnostics),
                        RobotDescriptionStatus::invalidMimic,
                        "mimic requires a movable joint and source",
                        joint.name
                    );
                }
                joint.mimicJoint = *source;
                if (const auto multiplier =
                        property(mimic, "multiplier");
                    multiplier.has_value() &&
                    !parseDouble(
                        *multiplier,
                        joint.mimicMultiplier
                    )) {
                    return fail(
                        std::move(diagnostics),
                        RobotDescriptionStatus::invalidMimic,
                        "mimic multiplier is invalid",
                        joint.name
                    );
                }
                if (const auto offset =
                        property(mimic, "offset");
                    offset.has_value() &&
                    !parseDouble(
                        *offset,
                        joint.mimicOffset
                    )) {
                    return fail(
                        std::move(diagnostics),
                        RobotDescriptionStatus::invalidMimic,
                        "mimic offset is invalid",
                        joint.name
                    );
                }
            }
            const std::size_t index = joints.size();
            childJoint.emplace(joint.child, index);
            outgoing[joint.parent].push_back(index);
            joints.push_back(std::move(joint));
        }
        std::map<std::string, std::size_t> jointByName;
        for (std::size_t index = 0u;
             index < joints.size();
             ++index) {
            jointByName.emplace(joints[index].name, index);
        }
        for (std::size_t index = 0u;
             index < joints.size();
             ++index) {
            const ParsedJoint& joint = joints[index];
            if (joint.mimicJoint.empty()) {
                continue;
            }
            const auto source =
                jointByName.find(joint.mimicJoint);
            if (source == jointByName.end() ||
                source->second == index ||
                joints[source->second].type == MR_JOINT_FIXED) {
                return fail(
                    std::move(diagnostics),
                    RobotDescriptionStatus::invalidMimic,
                    "mimic source is missing, fixed, or recursive",
                    joint.name
                );
            }
            const bool jointRotational =
                joint.type == MR_JOINT_REVOLUTE ||
                joint.type == MR_JOINT_CONTINUOUS;
            const ParsedJoint& sourceJoint =
                joints[source->second];
            const bool sourceRotational =
                sourceJoint.type == MR_JOINT_REVOLUTE ||
                sourceJoint.type == MR_JOINT_CONTINUOUS;
            if (jointRotational != sourceRotational) {
                return fail(
                    std::move(diagnostics),
                    RobotDescriptionStatus::invalidMimic,
                    "mimic joints use incompatible coordinates",
                    joint.name
                );
            }
            std::set<std::size_t> chain;
            std::size_t cursor = index;
            while (!joints[cursor].mimicJoint.empty()) {
                if (!chain.insert(cursor).second) {
                    return fail(
                        std::move(diagnostics),
                        RobotDescriptionStatus::invalidMimic,
                        "mimic dependency graph contains a cycle",
                        joint.name
                    );
                }
                const auto next = jointByName.find(
                    joints[cursor].mimicJoint
                );
                if (next == jointByName.end()) {
                    return fail(
                        std::move(diagnostics),
                        RobotDescriptionStatus::invalidMimic,
                        "mimic dependency source is missing",
                        joint.name
                    );
                }
                cursor = next->second;
            }
        }

        std::set<std::string> transmissionJoints;
        std::set<std::string> transmissionNames;
        for (xmlNode* transmission :
             children(robot, "transmission")) {
            const auto name = property(transmission, "name");
            if (!name.has_value() || name->empty() ||
                !transmissionNames.insert(*name).second) {
                return fail(
                    std::move(diagnostics),
                    RobotDescriptionStatus::
                        invalidTransmission,
                    "transmission names must be nonempty and unique",
                    name.value_or("transmission")
                );
            }
            const auto transmittedJoints =
                children(transmission, "joint");
            if (transmittedJoints.empty()) {
                return fail(
                    std::move(diagnostics),
                    RobotDescriptionStatus::
                        invalidTransmission,
                    "transmission has no joint",
                    *name
                );
            }
            for (xmlNode* transmitted :
                 transmittedJoints) {
                const auto jointName =
                    property(transmitted, "name");
                const auto found = jointName.has_value()
                    ? jointByName.find(*jointName)
                    : jointByName.end();
                if (!jointName.has_value() ||
                    found == jointByName.end() ||
                    joints[found->second].type ==
                        MR_JOINT_FIXED ||
                    !transmissionJoints.insert(
                        *jointName
                    ).second) {
                    return fail(
                        std::move(diagnostics),
                        RobotDescriptionStatus::
                            invalidTransmission,
                        "transmission joint is missing, fixed, or duplicated",
                        *name
                    );
                }
            }
        }
        if (joints.size() + 1u != links.size()) {
            return fail(
                std::move(diagnostics),
                RobotDescriptionStatus::invalidTopology,
                "URDF must contain one connected tree"
            );
        }
        std::vector<std::string> roots;
        for (const auto& [name, link] : links) {
            static_cast<void>(link);
            if (!childJoint.contains(name)) {
                roots.push_back(name);
            }
        }
        if (roots.size() != 1u) {
            return fail(
                std::move(diagnostics),
                RobotDescriptionStatus::invalidTopology,
                "URDF must have exactly one root link"
            );
        }
        for (auto& [parent, indices] : outgoing) {
            static_cast<void>(parent);
            std::ranges::sort(
                indices,
                [&joints](
                    const std::size_t left,
                    const std::size_t right
                ) {
                    return joints[left].name <
                        joints[right].name;
                }
            );
        }
        std::vector<std::string> linkOrder;
        std::vector<std::size_t> jointOrder;
        const auto visit = [&](
            this auto&& self,
            const std::string& link
        ) -> bool {
            if (std::ranges::find(linkOrder, link) !=
                linkOrder.end()) {
                return false;
            }
            linkOrder.push_back(link);
            const auto found = outgoing.find(link);
            if (found == outgoing.end()) {
                return true;
            }
            for (const std::size_t jointIndex :
                 found->second) {
                jointOrder.push_back(jointIndex);
                if (!self(joints[jointIndex].child)) {
                    return false;
                }
            }
            return true;
        };
        if (!visit(roots.front()) ||
            linkOrder.size() != links.size() ||
            jointOrder.size() != joints.size()) {
            return fail(
                std::move(diagnostics),
                RobotDescriptionStatus::invalidTopology,
                "URDF tree contains a cycle or disconnected link"
            );
        }

        EngineModel staged;
        staged.name = *robotName;
        staged.world.abiVersion = MR_ENGINE_ABI_VERSION;
        staged.world.articulationCount = 1u;
        staged.world.jointCount =
            static_cast<mr_u32>(joints.size());
        staged.world.bodyCount =
            static_cast<mr_u32>(links.size());
        staged.world.materialCount = 1u;
        staged.world.solverType =
            MR_SOLVER_THROUGHPUT_PGS;
        staged.world.frictionConeType =
            MR_FRICTION_CONE_ELLIPTIC;
        staged.world.gravityAndTimestep =
            options.gravityAndTimestep;
        staged.world.solverScales =
            f4(1.0e-6, 1.0e-9, 2.0, 1.0e-4);

        MRMaterialGPU material{};
        material.friction = options.friction;
        material.response = options.response;
        material.geometry = f4(
            options.contactOffset,
            options.restOffset,
            0.0,
            0.0
        );
        staged.materials.push_back(material);

        std::map<std::string, std::uint32_t> bodyIndex;
        for (std::uint32_t index = 0u;
             index < linkOrder.size();
             ++index) {
            bodyIndex.emplace(linkOrder[index], index);
        }

        MRArticulationGPU articulation{};
        articulation.rootBody = 0u;
        articulation.rootType =
            options.rootMode ==
                RobotDescriptionRootMode::floating
            ? MR_ROOT_FLOATING
            : MR_ROOT_FIXED;
        articulation.firstBody = 0u;
        articulation.bodyCount =
            staged.world.bodyCount;
        articulation.firstJoint = 0u;
        articulation.jointCount =
            staged.world.jointCount;
        articulation.qOffset = 0u;
        articulation.vOffset = 0u;
        articulation.nq =
            articulation.rootType ==
                MR_ROOT_FLOATING
            ? 7u
            : 0u;
        articulation.nv =
            articulation.rootType ==
                MR_ROOT_FLOATING
            ? 6u
            : 0u;
        staged.articulations.push_back(articulation);
        if (articulation.rootType == MR_ROOT_FLOATING) {
            staged.defaultQ = {
                0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 1.0F,
            };
            staged.defaultV.assign(6u, 0.0F);
            for (std::uint32_t local = 0u;
                 local < 6u;
                 ++local) {
                MRDofPropertiesGPU dof{};
                dof.articulationIndex = 0u;
                dof.jointIndex = MR_INVALID_INDEX;
                dof.qIndex =
                    local < 3u
                    ? local
                    : MR_INVALID_INDEX;
                dof.vIndex = local;
                dof.localDof = local;
                dof.flags = MR_DOF_FLAG_ROOT;
                staged.dofs.push_back(dof);
            }
        }

        std::map<std::string, std::uint32_t>
            jointIndexByChild;
        std::map<std::string, std::uint32_t>
            jointIndexByName;
        for (std::size_t ordered = 0u;
             ordered < jointOrder.size();
             ++ordered) {
            const ParsedJoint& source =
                joints[jointOrder[ordered]];
            MRJointDescriptorGPU joint{};
            joint.parentBody =
                bodyIndex.at(source.parent);
            joint.childBody =
                bodyIndex.at(source.child);
            joint.jointType = source.type;
            joint.qOffset =
                static_cast<mr_u32>(
                    staged.defaultQ.size()
                );
            joint.vOffset =
                static_cast<mr_u32>(
                    staged.defaultV.size()
                );
            const bool scalar =
                source.type != MR_JOINT_FIXED;
            joint.nq = scalar ? 1u : 0u;
            joint.nv = scalar ? 1u : 0u;
            joint.axis0 = f4(
                source.axis.x,
                source.axis.y,
                source.axis.z
            );
            const ParsedLink& parent =
                links.at(source.parent);
            const ParsedLink& child =
                links.at(source.child);
            joint.parentAnchor = f4(
                source.origin.translation.x -
                    parent.centerOfMass.x,
                source.origin.translation.y -
                    parent.centerOfMass.y,
                source.origin.translation.z -
                    parent.centerOfMass.z
            );
            joint.childAnchor = f4(
                -child.centerOfMass.x,
                -child.centerOfMass.y,
                -child.centerOfMass.z
            );
            const auto parentRotation =
                quaternion(source.origin.rotation);
            joint.parentRotation = f4(
                parentRotation[0],
                parentRotation[1],
                parentRotation[2],
                parentRotation[3]
            );
            joint.childRotation =
                f4(0.0, 0.0, 0.0, 1.0);
            staged.joints.push_back(joint);
            jointIndexByChild.emplace(
                source.child,
                static_cast<std::uint32_t>(ordered)
            );
            jointIndexByName.emplace(
                source.name,
                static_cast<std::uint32_t>(ordered)
            );
            if (scalar) {
                MRDofPropertiesGPU dof{};
                dof.articulationIndex = 0u;
                dof.jointIndex =
                    static_cast<mr_u32>(ordered);
                dof.qIndex = joint.qOffset;
                dof.vIndex = joint.vOffset;
                dof.localDof = 0u;
                const bool transmissionSelected =
                    !options.respectTransmissions ||
                    transmissionJoints.empty() ||
                    transmissionJoints.contains(source.name);
                if (options.actuateMovableJoints &&
                    transmissionSelected &&
                    source.mimicJoint.empty()) {
                    dof.flags |=
                        MR_DOF_FLAG_ACTUATED;
                }
                if (source.hasPositionLimit) {
                    dof.flags |=
                        MR_DOF_FLAG_POSITION_LIMIT;
                    dof.limits.x =
                        static_cast<float>(source.lower);
                    dof.limits.y =
                        static_cast<float>(source.upper);
                }
                if (source.velocity > 0.0) {
                    dof.flags |=
                        MR_DOF_FLAG_VELOCITY_LIMIT;
                    dof.limits.z =
                        static_cast<float>(source.velocity);
                }
                if (source.effort > 0.0 &&
                    (dof.flags & MR_DOF_FLAG_ACTUATED) != 0u) {
                    dof.flags |=
                        MR_DOF_FLAG_EFFORT_LIMIT;
                    dof.limits.w =
                        static_cast<float>(source.effort);
                }
                if (source.damping > 0.0) {
                    dof.flags |= MR_DOF_FLAG_DRIVE;
                    dof.drive.y =
                        static_cast<float>(source.damping);
                }
                dof.drive.z = options.defaultArmature;
                dof.drive.w =
                    static_cast<float>(source.friction);
                staged.dofs.push_back(dof);
                staged.defaultQ.push_back(0.0F);
                staged.defaultV.push_back(0.0F);
            }
        }
        staged.articulations[0].nq =
            static_cast<mr_u32>(
                staged.defaultQ.size()
            );
        staged.articulations[0].nv =
            static_cast<mr_u32>(
                staged.defaultV.size()
            );
        staged.world.nq =
            staged.articulations[0].nq;
        staged.world.nv =
            staged.articulations[0].nv;

        std::map<std::string, std::uint32_t> mimicVisitState;
        const auto resolveMimicDefault = [&](
            this auto&& self,
            const std::string& name
        ) -> bool {
            std::uint32_t& state = mimicVisitState[name];
            if (state == 2u) {
                return true;
            }
            if (state == 1u) {
                return false;
            }
            state = 1u;
            const ParsedJoint& source =
                joints[jointByName.at(name)];
            if (!source.mimicJoint.empty()) {
                if (!self(source.mimicJoint)) {
                    return false;
                }
                const MRJointDescriptorGPU& mimicDescriptor =
                    staged.joints[
                        jointIndexByName.at(name)
                    ];
                const MRJointDescriptorGPU& sourceDescriptor =
                    staged.joints[
                        jointIndexByName.at(
                            source.mimicJoint
                        )
                    ];
                const double defaultValue =
                    source.mimicMultiplier *
                        staged.defaultQ[
                            sourceDescriptor.qOffset
                        ] +
                    source.mimicOffset;
                if (!finite(defaultValue) ||
                    std::abs(defaultValue) >
                        std::numeric_limits<float>::max() ||
                    (source.hasPositionLimit &&
                     (defaultValue < source.lower ||
                      defaultValue > source.upper))) {
                    return false;
                }
                staged.defaultQ[mimicDescriptor.qOffset] =
                    static_cast<float>(defaultValue);
            }
            state = 2u;
            return true;
        };
        for (const ParsedJoint& joint : joints) {
            if (!resolveMimicDefault(joint.name)) {
                return fail(
                    std::move(diagnostics),
                    RobotDescriptionStatus::invalidMimic,
                    "mimic defaults are cyclic, non-finite, or outside limits",
                    joint.name
                );
            }
        }

        struct MimicConstraintStorage {
            ConstraintIRStableKey key{};
            std::vector<ConstraintIREndpoint> endpoints;
            ConstraintIRRow row{};
            float warmImpulse = 0.0f;
        };
        std::vector<MimicConstraintStorage> mimicConstraints;
        mimicConstraints.reserve(joints.size());
        for (const ParsedJoint& source : joints) {
            if (source.mimicJoint.empty()) {
                continue;
            }
            const std::uint32_t mimicIndex =
                jointIndexByName.at(source.name);
            const std::uint32_t sourceIndex =
                jointIndexByName.at(source.mimicJoint);
            const MRJointDescriptorGPU& mimic =
                staged.joints[mimicIndex];
            const MRJointDescriptorGPU& driver =
                staged.joints[sourceIndex];
            MimicConstraintStorage storage;
            storage.key.words[0] = 0x4d494d49u;
            storage.key.words[1] = mimicIndex;
            storage.key.words[2] = sourceIndex;
            storage.endpoints.reserve(2u);
            storage.endpoints.push_back(
                makeConstraintIRGeneralizedEndpoint(
                    0u,
                    mimic.qOffset,
                    mimic.vOffset,
                    0u,
                    1.0f
                )
            );
            if (source.mimicMultiplier != 0.0) {
                storage.endpoints.push_back(
                    makeConstraintIRGeneralizedEndpoint(
                        0u,
                        driver.qOffset,
                        driver.vOffset,
                        0u,
                        static_cast<float>(
                            -source.mimicMultiplier
                        )
                    )
                );
            }
            storage.row.targetVelocity = 0.0f;
            storage.row.compliance = 0.0f;
            storage.row.dissipation = 0.0f;
            storage.row.impulseLower =
                -kConstraintIRUnbounded;
            storage.row.impulseUpper =
                kConstraintIRUnbounded;
            mimicConstraints.push_back(std::move(storage));
        }
        if (!mimicConstraints.empty()) {
            std::vector<ConstraintIRSourceBlock> sources;
            sources.reserve(mimicConstraints.size());
            for (const MimicConstraintStorage& storage :
                 mimicConstraints) {
                sources.push_back({
                    .key = storage.key,
                    .type = MR_CONSTRAINT_GEAR,
                    .flags = 0u,
                    .islandIndex = 0u,
                    .eventSlot = kConstraintIRInvalidIndex,
                    .endpoints = storage.endpoints,
                    .rows = std::span{&storage.row, 1u},
                    .cone = std::nullopt,
                    .warmImpulses = std::span{
                        &storage.warmImpulse,
                        1u,
                    },
                });
            }
            const ConstraintIRCompilationResult compiled =
                compileConstraintIR(sources);
            if (!compiled.succeeded()) {
                return fail(
                    std::move(diagnostics),
                    RobotDescriptionStatus::invalidMimic,
                    "mimic ConstraintIR compilation failed: " +
                        compiled.diagnostics.message
                );
            }
            staged.constraintProgram = compiled.ir;
        }

        for (std::uint32_t ordered = 0u;
             ordered < linkOrder.size();
             ++ordered) {
            const ParsedLink& source =
                links.at(linkOrder[ordered]);
            Mat3 inverseInertia;
            if (!inverse(source.inertia, inverseInertia)) {
                return fail(
                    std::move(diagnostics),
                    RobotDescriptionStatus::invalidInertial,
                    "link inertia inversion failed",
                    source.name
                );
            }
            MRBodyPropertiesGPU body{};
            body.articulationIndex = 0u;
            body.parentBody = MR_INVALID_INDEX;
            body.inboundJoint = MR_INVALID_INDEX;
            if (ordered != 0u) {
                const std::uint32_t jointIndex =
                    jointIndexByChild.at(source.name);
                body.inboundJoint = jointIndex;
                body.parentBody =
                    staged.joints[jointIndex].parentBody;
            }
            body.motionType = MR_MOTION_DYNAMIC;
            body.massAndInverseMass = f4(
                source.mass,
                1.0 / source.mass,
                0.0
            );
            body.centerOfMass = f4(
                source.centerOfMass.x,
                source.centerOfMass.y,
                source.centerOfMass.z
            );
            body.inertiaRow0 = row(source.inertia, 0u);
            body.inertiaRow1 = row(source.inertia, 1u);
            body.inertiaRow2 = row(source.inertia, 2u);
            body.inverseInertiaRow0 =
                row(inverseInertia, 0u);
            body.inverseInertiaRow1 =
                row(inverseInertia, 1u);
            body.inverseInertiaRow2 =
                row(inverseInertia, 2u);
            body.dampingAndSpeedLimits =
                f4(0.0, 0.0, 1.0e6, 1.0e6);
            staged.bodies.push_back(body);

            for (const ParsedShape& sourceShape :
                 source.collisions) {
                MRShapeGPU shape{};
                shape.bodyIndex = ordered;
                shape.materialIndex = 0u;
                shape.collisionGroup =
                    options.collisionGroup;
                shape.collisionMask =
                    options.collisionMask;
                shape.slotGeneration =
                    static_cast<mr_u32>(
                        staged.shapes.size() + 1u
                    );
                shape.localPosition = f4(
                    sourceShape.origin.translation.x -
                        source.centerOfMass.x,
                    sourceShape.origin.translation.y -
                        source.centerOfMass.y,
                    sourceShape.origin.translation.z -
                        source.centerOfMass.z
                );
                const auto rotation =
                    quaternion(sourceShape.origin.rotation);
                shape.localRotation = f4(
                    rotation[0],
                    rotation[1],
                    rotation[2],
                    rotation[3]
                );
                double radius = 0.0;
                switch (sourceShape.kind) {
                case ParsedShapeKind::sphere:
                    shape.shapeType = MR_SHAPE_SPHERE;
                    shape.dimensions = f4(
                        sourceShape.dimensions.x,
                        0.0,
                        0.0
                    );
                    radius = sourceShape.dimensions.x;
                    break;
                case ParsedShapeKind::box:
                    shape.shapeType = MR_SHAPE_BOX;
                    shape.dimensions = f4(
                        sourceShape.dimensions.x,
                        sourceShape.dimensions.y,
                        sourceShape.dimensions.z
                    );
                    radius = std::sqrt(
                        sourceShape.dimensions.x *
                            sourceShape.dimensions.x +
                        sourceShape.dimensions.y *
                            sourceShape.dimensions.y +
                        sourceShape.dimensions.z *
                            sourceShape.dimensions.z
                    );
                    break;
                case ParsedShapeKind::cylinder:
                    shape.shapeType =
                        MR_SHAPE_CYLINDER;
                    shape.dimensions = f4(
                        sourceShape.dimensions.x,
                        sourceShape.dimensions.y,
                        0.0
                    );
                    radius = std::hypot(
                        sourceShape.dimensions.x,
                        sourceShape.dimensions.y
                    );
                    break;
                case ParsedShapeKind::capsule:
                    shape.shapeType = MR_SHAPE_CAPSULE;
                    shape.dimensions = f4(
                        sourceShape.dimensions.x,
                        sourceShape.dimensions.y,
                        0.0
                    );
                    radius =
                        sourceShape.dimensions.x +
                        sourceShape.dimensions.y;
                    break;
                case ParsedShapeKind::convexMesh: {
                    if (sourceShape.meshAsset >=
                        meshAssets.size()) {
                        return fail(
                            std::move(diagnostics),
                            RobotDescriptionStatus::
                                internalFailure,
                            "mesh asset reference is invalid",
                            source.name
                        );
                    }
                    ParsedMeshAsset& mesh =
                        meshAssets[sourceShape.meshAsset];
                    if (mesh.geometryIndex ==
                        MR_INVALID_INDEX) {
                        const GeometryCookResult cooked =
                            options.meshMode ==
                                RobotDescriptionMeshMode::
                                    convexHull
                            ? cookConvexHullGeometry(
                                  staged,
                                  mesh.vertices
                              )
                            : cookConvexGeometry(
                                  staged,
                                  mesh.vertices,
                                  mesh.indices
                              );
                        if (!cooked.succeeded()) {
                            return fail(
                                std::move(diagnostics),
                                cooked.status ==
                                    GeometryCookStatus::
                                        capacityOverflow
                                    ? RobotDescriptionStatus::
                                        capacityOverflow
                                    : RobotDescriptionStatus::
                                        unsupportedGeometry,
                                "articulated mesh is not a valid "
                                "closed convex surface: " +
                                    cooked.message,
                                source.name
                            );
                        }
                        mesh.geometryIndex =
                            cooked.geometryIndex;
                    }
                    shape.shapeType = MR_SHAPE_CONVEX;
                    shape.geometryOffset =
                        mesh.geometryIndex;
                    shape.geometryCount = 1u;
                    shape.dimensions = f4(
                        sourceShape.dimensions.x,
                        sourceShape.dimensions.y,
                        sourceShape.dimensions.z
                    );
                    double radiusSquared = 0.0;
                    for (const mr_float4& vertex :
                         mesh.vertices) {
                        radiusSquared = std::max(
                            radiusSquared,
                            static_cast<double>(vertex.x) *
                                    vertex.x *
                                    sourceShape.dimensions.x *
                                    sourceShape.dimensions.x +
                                static_cast<double>(vertex.y) *
                                    vertex.y *
                                    sourceShape.dimensions.y *
                                    sourceShape.dimensions.y +
                                static_cast<double>(vertex.z) *
                                    vertex.z *
                                    sourceShape.dimensions.z *
                                    sourceShape.dimensions.z
                        );
                    }
                    radius = std::sqrt(radiusSquared);
                    break;
                }
                }
                shape.contactRestAndBoundingRadius = f4(
                    options.contactOffset,
                    options.restOffset,
                    radius,
                    0.0
                );
                staged.shapes.push_back(shape);
            }
        }
        staged.world.shapeCount =
            static_cast<mr_u32>(staged.shapes.size());
        const std::uint64_t logicalPairs =
            static_cast<std::uint64_t>(
                staged.shapes.size()
            ) * (
                staged.shapes.size() > 0u
                ? staged.shapes.size() - 1u
                : 0u
            ) / 2u;
        if (logicalPairs >
            std::numeric_limits<mr_u32>::max()) {
            return fail(
                std::move(diagnostics),
                RobotDescriptionStatus::capacityOverflow,
                "collider pair count exceeds the engine ABI"
            );
        }
        const std::uint64_t pairCapacity =
            std::max<std::uint64_t>(logicalPairs, 1u);
        const std::uint64_t contactCapacity =
            std::max<std::uint64_t>(
                4u * pairCapacity,
                8u
            );
        const std::uint64_t constraintCapacity =
            std::max<std::uint64_t>(
                contactCapacity +
                    2u * staged.world.nv,
                8u
            );
        if (contactCapacity >
                std::numeric_limits<mr_u32>::max() ||
            constraintCapacity >
                std::numeric_limits<mr_u32>::max()) {
            return fail(
                std::move(diagnostics),
                RobotDescriptionStatus::capacityOverflow,
                "derived contact or constraint capacity exceeds the engine ABI"
            );
        }
        staged.world.pairCapacity =
            static_cast<mr_u32>(pairCapacity);
        staged.world.contactCapacity =
            static_cast<mr_u32>(contactCapacity);
        staged.world.constraintCapacity =
            static_cast<mr_u32>(constraintCapacity);
        staged.world.islandCapacity =
            std::max<mr_u32>(
                staged.world.bodyCount,
                1u
            );

        std::set<std::string> passiveJoints;
        if (!srdf.empty()) {
            XmlDocument srdfDocument{
                xmlReadMemory(
                    srdf.data(),
                    static_cast<int>(srdf.size()),
                    "robot.srdf",
                    nullptr,
                    parseFlags
                ),
                &xmlFreeDoc,
            };
            xmlNode* srdfRobot = srdfDocument
                ? xmlDocGetRootElement(srdfDocument.get())
                : nullptr;
            if (srdfRobot == nullptr ||
                nodeName(srdfRobot) != "robot") {
                return fail(
                    std::move(diagnostics),
                    RobotDescriptionStatus::invalidSrdf,
                    "SRDF XML/root is invalid"
                );
            }
            if (const auto srdfName =
                    property(srdfRobot, "name");
                srdfName.has_value() &&
                *srdfName != *robotName) {
                return fail(
                    std::move(diagnostics),
                    RobotDescriptionStatus::invalidSrdf,
                    "SRDF robot name does not match URDF"
                );
            }
            for (xmlNode* passive :
                 children(srdfRobot, "passive_joint")) {
                const auto name = property(passive, "name");
                const auto found = name.has_value()
                    ? jointIndexByName.find(*name)
                    : jointIndexByName.end();
                if (!name.has_value() ||
                    found == jointIndexByName.end() ||
                    staged.joints[found->second].nv != 1u ||
                    !passiveJoints.insert(*name).second) {
                    return fail(
                        std::move(diagnostics),
                        RobotDescriptionStatus::invalidSrdf,
                        "SRDF passive joint is missing, fixed, or duplicated",
                        name.value_or("passive_joint")
                    );
                }
                MRDofPropertiesGPU& dof = staged.dofs[
                    staged.joints[found->second].vOffset
                ];
                dof.flags &= ~(
                    MR_DOF_FLAG_ACTUATED |
                    MR_DOF_FLAG_EFFORT_LIMIT
                );
            }
            std::map<std::string, std::vector<std::uint32_t>>
                linkShapes;
            for (std::uint32_t shapeIndex = 0u;
                 shapeIndex < staged.shapes.size();
                 ++shapeIndex) {
                linkShapes[
                    linkOrder[
                        staged.shapes[shapeIndex].bodyIndex
                    ]
                ].push_back(shapeIndex);
            }
            for (xmlNode* disabled :
                 children(
                     srdfRobot,
                     "disable_collisions"
                 )) {
                const auto link1 =
                    property(disabled, "link1");
                const auto link2 =
                    property(disabled, "link2");
                if (!link1.has_value() ||
                    !link2.has_value() ||
                    !links.contains(*link1) ||
                    !links.contains(*link2) ||
                    *link1 == *link2) {
                    return fail(
                        std::move(diagnostics),
                        RobotDescriptionStatus::invalidSrdf,
                        "SRDF disabled-collision pair is invalid"
                    );
                }
                for (const std::uint32_t first :
                     linkShapes[*link1]) {
                    for (const std::uint32_t second :
                         linkShapes[*link2]) {
                        staged.collisionExclusions.push_back({
                            .colliderA =
                                std::min(first, second),
                            .colliderB =
                                std::max(first, second),
                        });
                    }
                }
            }
            std::ranges::sort(
                staged.collisionExclusions,
                [](const CollisionPairExclusion& left,
                   const CollisionPairExclusion& right) {
                    return
                        std::pair{
                            left.colliderA,
                            left.colliderB,
                        } <
                        std::pair{
                            right.colliderA,
                            right.colliderB,
                        };
                }
            );
            const auto uniqueEnd = std::ranges::unique(
                staged.collisionExclusions,
                {},
                [](const CollisionPairExclusion& value) {
                    return std::pair{
                        value.colliderA,
                        value.colliderB,
                    };
                }
            ).begin();
            staged.collisionExclusions.erase(
                uniqueEnd,
                staged.collisionExclusions.end()
            );
        }

        std::string reason;
        if (!staged.valid(&reason)) {
            return fail(
                std::move(diagnostics),
                RobotDescriptionStatus::invalidEngineModel,
                std::move(reason)
            );
        }
        diagnostics.linkCount =
            staged.world.bodyCount;
        diagnostics.jointCount =
            staged.world.jointCount;
        diagnostics.dofCount = staged.world.nv;
        diagnostics.colliderCount =
            staged.world.shapeCount;
        diagnostics.exclusionCount =
            static_cast<std::uint32_t>(
                staged.collisionExclusions.size()
            );
        diagnostics.mimicConstraintCount =
            static_cast<std::uint32_t>(
                mimicConstraints.size()
            );
        diagnostics.transmissionJointCount =
            static_cast<std::uint32_t>(
                transmissionJoints.size()
            );
        diagnostics.passiveJointCount =
            static_cast<std::uint32_t>(
                passiveJoints.size()
            );
        diagnostics.meshAssetCount =
            static_cast<std::uint32_t>(
                meshAssets.size()
            );
        std::uint64_t meshVertices = 0u;
        std::uint64_t meshTriangles = 0u;
        for (const ParsedMeshAsset& mesh : meshAssets) {
            meshVertices += mesh.vertices.size();
            meshTriangles += mesh.indices.size() / 3u;
        }
        if (meshVertices >
                std::numeric_limits<std::uint32_t>::max() ||
            meshTriangles >
                std::numeric_limits<std::uint32_t>::max()) {
            return fail(
                std::move(diagnostics),
                RobotDescriptionStatus::capacityOverflow,
                "mesh diagnostics exceed the cooked ABI"
            );
        }
        diagnostics.meshVertexCount =
            static_cast<std::uint32_t>(meshVertices);
        diagnostics.meshTriangleCount =
            static_cast<std::uint32_t>(meshTriangles);
        diagnostics.bodyNames = linkOrder;
        diagnostics.jointNames.reserve(jointOrder.size());
        diagnostics.dofNames.reserve(staged.world.nv);
        if (options.rootMode ==
            RobotDescriptionRootMode::floating) {
            diagnostics.dofNames.insert(
                diagnostics.dofNames.end(),
                {
                    "root_linear_x",
                    "root_linear_y",
                    "root_linear_z",
                    "root_angular_x",
                    "root_angular_y",
                    "root_angular_z",
                }
            );
        }
        for (const std::size_t ordered : jointOrder) {
            const ParsedJoint& joint = joints[ordered];
            diagnostics.jointNames.push_back(joint.name);
            if (joint.type != MR_JOINT_FIXED) {
                diagnostics.dofNames.push_back(joint.name);
            }
        }
        for (const std::string& linkName : linkOrder) {
            const ParsedLink& link = links.at(linkName);
            diagnostics.shapeLinkNames.insert(
                diagnostics.shapeLinkNames.end(),
                link.collisions.size(),
                linkName
            );
        }
        if (
            diagnostics.bodyNames.size() != staged.bodies.size() ||
            diagnostics.jointNames.size() != staged.joints.size() ||
            diagnostics.dofNames.size() != staged.dofs.size() ||
            diagnostics.shapeLinkNames.size() !=
                staged.shapes.size()
        ) {
            return fail(
                std::move(diagnostics),
                RobotDescriptionStatus::internalFailure,
                "semantic index maps do not match cooked runtime arrays"
            );
        }
        staged.bodyNames = diagnostics.bodyNames;
        staged.jointNames = diagnostics.jointNames;
        staged.dofNames = diagnostics.dofNames;
        staged.shapeNames.reserve(
            diagnostics.shapeLinkNames.size()
        );
        for (std::size_t shapeIndex = 0u;
             shapeIndex < diagnostics.shapeLinkNames.size();
             ++shapeIndex) {
            staged.shapeNames.push_back(
                diagnostics.shapeLinkNames[shapeIndex] +
                "/collision_" + std::to_string(shapeIndex)
            );
        }
        reason.clear();
        if (!staged.valid(&reason)) {
            return fail(
                std::move(diagnostics),
                RobotDescriptionStatus::invalidEngineModel,
                "cooked semantic maps are invalid: " + reason
            );
        }
        output = std::move(staged);
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return fail(
            std::move(diagnostics),
            RobotDescriptionStatus::capacityOverflow,
            "robot-description allocation failed"
        );
    } catch (const std::exception& exception) {
        return fail(
            std::move(diagnostics),
            RobotDescriptionStatus::internalFailure,
            exception.what()
        );
    }
}

RobotDescriptionDiagnostics cookRobotDescriptionFiles(
    const std::filesystem::path& urdfPath,
    const std::filesystem::path& srdfPath,
    EngineModel& output,
    const RobotDescriptionCookOptions& options
) {
    const std::optional<std::string> urdf =
        readFile(urdfPath);
    if (!urdf.has_value()) {
        RobotDescriptionDiagnostics diagnostics;
        diagnostics.sourceName = urdfPath.string();
        return fail(
            std::move(diagnostics),
            RobotDescriptionStatus::ioFailure,
            "failed to read URDF"
        );
    }
    std::string srdf;
    if (!srdfPath.empty()) {
        const std::optional<std::string> loaded =
            readFile(srdfPath);
        if (!loaded.has_value()) {
            RobotDescriptionDiagnostics diagnostics;
            diagnostics.sourceName = srdfPath.string();
            return fail(
                std::move(diagnostics),
                RobotDescriptionStatus::ioFailure,
                "failed to read SRDF"
            );
        }
        srdf = *loaded;
    }
    RobotDescriptionCookOptions resolvedOptions = options;
    if (resolvedOptions.meshAssetRoot.empty()) {
        resolvedOptions.meshAssetRoot =
            urdfPath.parent_path();
    }
    return cookRobotDescription(
        *urdf,
        srdf,
        output,
        resolvedOptions,
        urdfPath.string()
    );
}

const char* robotDescriptionStatusName(
    const RobotDescriptionStatus status
) noexcept {
    switch (status) {
    case RobotDescriptionStatus::success:
        return "success";
    case RobotDescriptionStatus::ioFailure:
        return "io_failure";
    case RobotDescriptionStatus::malformedXml:
        return "malformed_xml";
    case RobotDescriptionStatus::invalidRobot:
        return "invalid_robot";
    case RobotDescriptionStatus::invalidTopology:
        return "invalid_topology";
    case RobotDescriptionStatus::invalidInertial:
        return "invalid_inertial";
    case RobotDescriptionStatus::unsupportedJoint:
        return "unsupported_joint";
    case RobotDescriptionStatus::unsupportedGeometry:
        return "unsupported_geometry";
    case RobotDescriptionStatus::invalidMimic:
        return "invalid_mimic";
    case RobotDescriptionStatus::invalidTransmission:
        return "invalid_transmission";
    case RobotDescriptionStatus::invalidSrdf:
        return "invalid_srdf";
    case RobotDescriptionStatus::capacityOverflow:
        return "capacity_overflow";
    case RobotDescriptionStatus::invalidEngineModel:
        return "invalid_engine_model";
    case RobotDescriptionStatus::internalFailure:
        return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo

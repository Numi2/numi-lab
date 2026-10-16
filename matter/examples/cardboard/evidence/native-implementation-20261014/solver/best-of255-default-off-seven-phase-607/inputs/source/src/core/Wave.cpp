#include "metalrobo/Wave.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <regex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFnvOffset =
    1469598103934665603ull;
constexpr std::uint64_t kFnvPrime =
    1099511628211ull;

struct NpyFloatArray {
    std::array<std::size_t, 3u> shape{};
    std::vector<float> values;
    std::vector<std::byte> sourceBytes;
};

struct AtlasPoint {
    mr_float4 position{};
    mr_float4 normal{};
    bool valid = false;
};

WaveAssetDiagnostics fail(
    WaveAssetDiagnostics diagnostics,
    const WaveAssetStatus status,
    std::string message,
    std::string element = {}
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    diagnostics.element = std::move(element);
    return diagnostics;
}

void hashBytes(
    std::uint64_t& hash,
    const void* data,
    const std::size_t size
) {
    const auto* bytes =
        static_cast<const unsigned char*>(data);
    for (std::size_t index = 0u; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
}

void hashString(
    std::uint64_t& hash,
    const std::string_view value
) {
    hashBytes(hash, value.data(), value.size());
    constexpr unsigned char separator = 0xffu;
    hashBytes(hash, &separator, sizeof(separator));
}

std::optional<std::string> readText(
    const std::filesystem::path& path
) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return std::nullopt;
    }
    std::ostringstream contents;
    contents << stream.rdbuf();
    if (!stream.good() && !stream.eof()) {
        return std::nullopt;
    }
    return contents.str();
}

std::string trimmed(std::string value) {
    const auto isSpace = [](const unsigned char character) {
        return std::isspace(character) != 0;
    };
    value.erase(
        value.begin(),
        std::find_if_not(value.begin(), value.end(), isSpace)
    );
    value.erase(
        std::find_if_not(
            value.rbegin(),
            value.rend(),
            isSpace
        ).base(),
        value.end()
    );
    return value;
}

bool gitRevision(
    const std::filesystem::path& root,
    std::string& revision
) {
    const std::filesystem::path git = root / ".git";
    const auto headValue = readText(git / "HEAD");
    if (!headValue.has_value()) {
        return false;
    }
    const std::string head = trimmed(*headValue);
    static const std::regex hashPattern{"[0-9a-f]{40}"};
    if (std::regex_match(head, hashPattern)) {
        revision = head;
        return true;
    }
    constexpr std::string_view prefix = "ref: ";
    if (!head.starts_with(prefix)) {
        return false;
    }
    const std::string reference{
        head.substr(prefix.size())
    };
    if (const auto loose = readText(git / reference);
        loose.has_value()) {
        revision = trimmed(*loose);
        return std::regex_match(revision, hashPattern);
    }
    const auto packed = readText(git / "packed-refs");
    if (!packed.has_value()) {
        return false;
    }
    std::istringstream lines{*packed};
    std::string line;
    while (std::getline(lines, line)) {
        if (line.empty() || line.front() == '#' ||
            line.front() == '^') {
            continue;
        }
        const std::size_t separator = line.find(' ');
        if (separator == std::string::npos) {
            continue;
        }
        if (line.substr(separator + 1u) == reference) {
            revision = line.substr(0u, separator);
            return std::regex_match(revision, hashPattern);
        }
    }
    return false;
}

bool readNpy(
    const std::filesystem::path& path,
    NpyFloatArray& output,
    std::string& message
) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        message = "failed to open NumPy tactile array";
        return false;
    }
    const std::streamsize size = stream.tellg();
    if (size < 12) {
        message = "NumPy tactile array is truncated";
        return false;
    }
    stream.seekg(0);
    std::vector<std::byte> bytes(
        static_cast<std::size_t>(size)
    );
    stream.read(
        reinterpret_cast<char*>(bytes.data()),
        size
    );
    if (!stream) {
        message = "failed to read NumPy tactile array";
        return false;
    }
    constexpr std::array<unsigned char, 6u> magic{
        0x93u, 'N', 'U', 'M', 'P', 'Y',
    };
    if (!std::equal(
            magic.begin(),
            magic.end(),
            reinterpret_cast<const unsigned char*>(
                bytes.data()
            )
        )) {
        message = "tactile array has no NumPy magic";
        return false;
    }
    const auto* raw = reinterpret_cast<const unsigned char*>(
        bytes.data()
    );
    const std::uint32_t major = raw[6u];
    std::size_t headerLength = 0u;
    std::size_t headerOffset = 0u;
    if (major == 1u) {
        headerLength =
            static_cast<std::size_t>(raw[8u]) |
            (static_cast<std::size_t>(raw[9u]) << 8u);
        headerOffset = 10u;
    } else if (major == 2u || major == 3u) {
        headerLength =
            static_cast<std::size_t>(raw[8u]) |
            (static_cast<std::size_t>(raw[9u]) << 8u) |
            (static_cast<std::size_t>(raw[10u]) << 16u) |
            (static_cast<std::size_t>(raw[11u]) << 24u);
        headerOffset = 12u;
    } else {
        message = "tactile array uses an unsupported NumPy version";
        return false;
    }
    if (headerOffset + headerLength > bytes.size()) {
        message = "NumPy tactile header exceeds the file";
        return false;
    }
    const std::string header{
        reinterpret_cast<const char*>(
            bytes.data() + headerOffset
        ),
        headerLength,
    };
    const bool float32 =
        header.find("'descr': '<f4'") != std::string::npos ||
        header.find("\"descr\": \"<f4\"") != std::string::npos ||
        header.find("'descr': '=f4'") != std::string::npos;
    if (!float32 ||
        header.find("fortran_order") == std::string::npos ||
        (
            header.find("'fortran_order': False") ==
                std::string::npos &&
            header.find("\"fortran_order\": false") ==
                std::string::npos
        )) {
        message =
            "tactile array must be little-endian C-order float32";
        return false;
    }
    static const std::regex shapePattern{
        R"('shape'\s*:\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,?\s*\))"
    };
    std::smatch match;
    if (!std::regex_search(header, match, shapePattern)) {
        message = "NumPy tactile array shape is missing";
        return false;
    }
    std::array<std::size_t, 3u> shape{};
    try {
        for (std::size_t index = 0u;
             index < shape.size();
             ++index) {
            shape[index] = std::stoull(match[index + 1u].str());
        }
    } catch (const std::exception&) {
        message = "NumPy tactile array shape is invalid";
        return false;
    }
    if (
        shape[0] == 0u ||
        shape[1] == 0u ||
        shape[2] != 3u ||
        shape[0] >
            std::numeric_limits<std::size_t>::max() /
                shape[1] /
                shape[2]
    ) {
        message = "NumPy tactile array dimensions are invalid";
        return false;
    }
    const std::size_t valueCount =
        shape[0] * shape[1] * shape[2];
    const std::size_t dataOffset =
        headerOffset + headerLength;
    if (
        valueCount >
            std::numeric_limits<std::size_t>::max() /
                sizeof(float) ||
        dataOffset + valueCount * sizeof(float) != bytes.size()
    ) {
        message = "NumPy tactile payload size is invalid";
        return false;
    }
    static_assert(std::endian::native == std::endian::little);
    NpyFloatArray candidate;
    candidate.shape = shape;
    candidate.values.resize(valueCount);
    std::memcpy(
        candidate.values.data(),
        bytes.data() + dataOffset,
        valueCount * sizeof(float)
    );
    if (!std::ranges::all_of(
            candidate.values,
            [](const float value) {
                return std::isfinite(value);
            }
        )) {
        message = "NumPy tactile payload is non-finite";
        return false;
    }
    candidate.sourceBytes = std::move(bytes);
    output = std::move(candidate);
    return true;
}

mr_float4 add(
    const mr_float4 left,
    const mr_float4 right
) {
    return {
        left.x + right.x,
        left.y + right.y,
        left.z + right.z,
        0.0f,
    };
}

mr_float4 subtract(
    const mr_float4 left,
    const mr_float4 right
) {
    return {
        left.x - right.x,
        left.y - right.y,
        left.z - right.z,
        0.0f,
    };
}

mr_float4 multiply(
    const mr_float4 value,
    const float scale
) {
    return {
        value.x * scale,
        value.y * scale,
        value.z * scale,
        0.0f,
    };
}

float dot(const mr_float4 left, const mr_float4 right) {
    return
        left.x * right.x +
        left.y * right.y +
        left.z * right.z;
}

mr_float4 cross(
    const mr_float4 left,
    const mr_float4 right
) {
    return {
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
        0.0f,
    };
}

float length(const mr_float4 value) {
    return std::sqrt(dot(value, value));
}

mr_float4 normalized(const mr_float4 value) {
    const float magnitude = length(value);
    if (!(magnitude > 1.0e-12f)) {
        throw std::runtime_error(
            "tactile atlas vector is degenerate"
        );
    }
    return multiply(value, 1.0f / magnitude);
}

std::size_t flatIndex(
    const NpyFloatArray& array,
    const std::size_t v,
    const std::size_t u,
    const std::size_t lane
) {
    return (v * array.shape[1] + u) * 3u + lane;
}

std::vector<AtlasPoint> resampleAtlas(
    const NpyFloatArray& points,
    const NpyFloatArray& normals,
    const std::uint32_t width,
    const std::uint32_t height
) {
    if (points.shape != normals.shape ||
        points.shape[0] < height ||
        points.shape[1] < width) {
        throw std::runtime_error(
            "published tactile point and normal maps disagree"
        );
    }
    std::vector<AtlasPoint> result(
        static_cast<std::size_t>(width) * height
    );
    for (std::uint32_t v = 0u; v < height; ++v) {
        const std::size_t sourceV0 =
            static_cast<std::size_t>(v) *
            points.shape[0] / height;
        const std::size_t sourceV1 =
            static_cast<std::size_t>(v + 1u) *
            points.shape[0] / height;
        for (std::uint32_t u = 0u; u < width; ++u) {
            const std::size_t sourceU0 =
                static_cast<std::size_t>(u) *
                points.shape[1] / width;
            const std::size_t sourceU1 =
                static_cast<std::size_t>(u + 1u) *
                points.shape[1] / width;
            mr_float4 pointSum{};
            mr_float4 normalSum{};
            std::uint32_t count = 0u;
            for (std::size_t sourceV = sourceV0;
                 sourceV < sourceV1;
                 ++sourceV) {
                for (std::size_t sourceU = sourceU0;
                     sourceU < sourceU1;
                     ++sourceU) {
                    const mr_float4 normal{
                        normals.values[
                            flatIndex(
                                normals,
                                sourceV,
                                sourceU,
                                0u
                            )
                        ],
                        normals.values[
                            flatIndex(
                                normals,
                                sourceV,
                                sourceU,
                                1u
                            )
                        ],
                        normals.values[
                            flatIndex(
                                normals,
                                sourceV,
                                sourceU,
                                2u
                            )
                        ],
                        0.0f,
                    };
                    if (length(normal) < 0.5f) {
                        continue;
                    }
                    const mr_float4 point{
                        points.values[
                            flatIndex(
                                points,
                                sourceV,
                                sourceU,
                                0u
                            )
                        ] * 1.0e-3f,
                        points.values[
                            flatIndex(
                                points,
                                sourceV,
                                sourceU,
                                1u
                            )
                        ] * 1.0e-3f,
                        points.values[
                            flatIndex(
                                points,
                                sourceV,
                                sourceU,
                                2u
                            )
                        ] * 1.0e-3f,
                        0.0f,
                    };
                    pointSum = add(pointSum, point);
                    normalSum = add(normalSum, normal);
                    ++count;
                }
            }
            if (count == 0u || length(normalSum) < 1.0e-6f) {
                continue;
            }
            AtlasPoint& point =
                result[static_cast<std::size_t>(v) * width + u];
            point.position =
                multiply(pointSum, 1.0f / count);
            point.normal = normalized(normalSum);
            point.valid = true;
        }
    }
    return result;
}

bool derivative(
    const std::vector<AtlasPoint>& points,
    const std::uint32_t width,
    const std::uint32_t height,
    const std::uint32_t u,
    const std::uint32_t v,
    const bool horizontal,
    mr_float4& value
) {
    const auto sample = [&](const std::uint32_t x,
                            const std::uint32_t y)
        -> const AtlasPoint& {
        return points[static_cast<std::size_t>(y) * width + x];
    };
    std::optional<std::pair<std::uint32_t, mr_float4>> lower;
    std::optional<std::pair<std::uint32_t, mr_float4>> upper;
    const std::uint32_t coordinate = horizontal ? u : v;
    const std::uint32_t limit = horizontal ? width : height;
    for (std::uint32_t distance = 1u;
         distance < limit;
         ++distance) {
        if (!lower.has_value() && coordinate >= distance) {
            const std::uint32_t x =
                horizontal ? u - distance : u;
            const std::uint32_t y =
                horizontal ? v : v - distance;
            if (sample(x, y).valid) {
                lower = {
                    coordinate - distance,
                    sample(x, y).position,
                };
            }
        }
        if (!upper.has_value() &&
            coordinate + distance < limit) {
            const std::uint32_t x =
                horizontal ? u + distance : u;
            const std::uint32_t y =
                horizontal ? v : v + distance;
            if (sample(x, y).valid) {
                upper = {
                    coordinate + distance,
                    sample(x, y).position,
                };
            }
        }
        if (lower.has_value() && upper.has_value()) {
            break;
        }
    }
    const mr_float4 center = sample(u, v).position;
    if (lower.has_value() && upper.has_value()) {
        value = multiply(
            subtract(upper->second, lower->second),
            1.0f / static_cast<float>(
                upper->first - lower->first
            )
        );
        return true;
    }
    if (upper.has_value()) {
        value = multiply(
            subtract(upper->second, center),
            1.0f / static_cast<float>(
                upper->first - coordinate
            )
        );
        return true;
    }
    if (lower.has_value()) {
        value = multiply(
            subtract(center, lower->second),
            1.0f / static_cast<float>(
                coordinate - lower->first
            )
        );
        return true;
    }
    return false;
}

std::vector<TactileSampleSpec> tactileSamples(
    const NpyFloatArray& points,
    const NpyFloatArray& normals,
    const std::uint32_t width,
    const std::uint32_t height,
    const float maximumDepth,
    std::uint32_t& validCount
) {
    const std::vector<AtlasPoint> atlas =
        resampleAtlas(points, normals, width, height);
    std::vector<TactileSampleSpec> result;
    result.reserve(atlas.size());
    for (std::uint32_t v = 0u; v < height; ++v) {
        for (std::uint32_t u = 0u; u < width; ++u) {
            TactileSampleSpec sample;
            sample.atlasU = u;
            sample.atlasV = v;
            const AtlasPoint& point =
                atlas[static_cast<std::size_t>(v) * width + u];
            mr_float4 deltaU{};
            mr_float4 deltaV{};
            if (!point.valid ||
                !derivative(
                    atlas,
                    width,
                    height,
                    u,
                    v,
                    true,
                    deltaU
                ) ||
                !derivative(
                    atlas,
                    width,
                    height,
                    u,
                    v,
                    false,
                    deltaV
                )) {
                sample.valid = false;
                result.push_back(sample);
                continue;
            }
            deltaU = subtract(
                deltaU,
                multiply(
                    point.normal,
                    dot(deltaU, point.normal)
                )
            );
            if (length(deltaU) < 1.0e-8f) {
                sample.valid = false;
                result.push_back(sample);
                continue;
            }
            const mr_float4 tangentU = normalized(deltaU);
            mr_float4 tangentV =
                normalized(cross(point.normal, tangentU));
            if (dot(tangentV, deltaV) < 0.0f) {
                tangentV = multiply(tangentV, -1.0f);
            }
            const float area = length(cross(deltaU, deltaV));
            if (!(area > 1.0e-12f) || !std::isfinite(area)) {
                sample.valid = false;
                result.push_back(sample);
                continue;
            }
            sample.localPosition = point.position;
            sample.localNormal = point.normal;
            sample.localTangentU = tangentU;
            sample.localTangentV = tangentV;
            sample.areaSquareMeters = area;
            sample.maximumDepthMeters = maximumDepth;
            sample.valid = true;
            result.push_back(sample);
            ++validCount;
        }
    }
    return result;
}

std::array<std::string, 22u> controllerJointNames(
    const std::string_view side
) {
    const std::array<std::string_view, 22u> suffixes{
        "thumb_CMC_FE",
        "thumb_CMC_AA",
        "thumb_MCP_FE",
        "thumb_MCP_AA",
        "thumb_IP",
        "index_MCP_FE",
        "index_MCP_AA",
        "index_PIP",
        "index_DIP",
        "middle_MCP_FE",
        "middle_MCP_AA",
        "middle_PIP",
        "middle_DIP",
        "ring_MCP_FE",
        "ring_MCP_AA",
        "ring_PIP",
        "ring_DIP",
        "pinky_CMC",
        "pinky_MCP_FE",
        "pinky_MCP_AA",
        "pinky_PIP",
        "pinky_DIP",
    };
    std::array<std::string, 22u> result;
    for (std::size_t index = 0u;
         index < suffixes.size();
         ++index) {
        result[index] =
            std::string{side} + "_" +
            std::string{suffixes[index]};
    }
    return result;
}

std::uint32_t indexOf(
    const std::vector<std::string>& values,
    const std::string& target
) {
    const auto found = std::ranges::find(values, target);
    return found == values.end()
        ? MR_INVALID_INDEX
        : static_cast<std::uint32_t>(
              std::distance(values.begin(), found)
          );
}

template <std::size_t Count>
std::array<std::uint32_t, Count> sequentialIndices(
    const std::uint32_t first
) {
    std::array<std::uint32_t, Count> result{};
    for (std::size_t index = 0u; index < Count; ++index) {
        result[index] =
            first + static_cast<std::uint32_t>(index);
    }
    return result;
}

} // namespace

WaveAssetDiagnostics cookSharpaWaveHand(
    const WaveAssetConfig& config,
    WaveHandAssets& output
) {
    WaveAssetDiagnostics diagnostics;
    if (
        config.urdfRepositoryRoot.empty() ||
        config.tactileRepositoryRoot.empty() ||
        config.atlasWidth == 0u ||
        config.atlasHeight == 0u ||
        !std::isfinite(config.maximumDepthMeters) ||
        !(config.maximumDepthMeters > 0.0f) ||
        config.maximumDepthMeters > 0.02f
    ) {
        return fail(
            std::move(diagnostics),
            WaveAssetStatus::invalidConfiguration,
            "Wave asset configuration is invalid"
        );
    }
    try {
        std::string urdfRevision;
        std::string tactileRevision;
        const bool hasUrdfRevision = gitRevision(
            config.urdfRepositoryRoot,
            urdfRevision
        );
        const bool hasTactileRevision = gitRevision(
            config.tactileRepositoryRoot,
            tactileRevision
        );
        if (config.requirePinnedGitRevisions &&
            (
                !hasUrdfRevision ||
                !hasTactileRevision ||
                urdfRevision != kSharpaWaveUrdfRevision ||
                tactileRevision != kSharpaWaveTactileRevision
            )) {
            return fail(
                std::move(diagnostics),
                WaveAssetStatus::sourceRevisionMismatch,
                "Wave URDF or tactile checkout is not at the pinned "
                "revision"
            );
        }
        const std::string side =
            config.side == WaveHandSide::left
            ? "left"
            : "right";
        const std::filesystem::path packageRoot =
            config.urdfRepositoryRoot / "wave_01";
        const std::filesystem::path urdf =
            packageRoot /
            (side + "_sharpa_wave") /
            (side + "_sharpa_wave.urdf");
        RobotDescriptionCookOptions robotOptions;
        robotOptions.rootMode =
            RobotDescriptionRootMode::fixed;
        robotOptions.meshAssetRoot = urdf.parent_path();
        robotOptions.packageSearchRoots = {packageRoot};
        robotOptions.meshMode =
            RobotDescriptionMeshMode::convexHull;
        EngineModel model;
        RobotDescriptionDiagnostics robotDiagnostics =
            cookRobotDescriptionFiles(
                urdf,
                {},
                model,
                robotOptions
            );
        if (!robotDiagnostics.succeeded()) {
            return fail(
                std::move(diagnostics),
                WaveAssetStatus::robotCookFailure,
                robotDiagnostics.message,
                robotDiagnostics.element
            );
        }
        const auto controllerNames = controllerJointNames(side);
        std::array<std::uint32_t, 22u> controllerDofIndices{};
        for (std::size_t index = 0u;
             index < controllerNames.size();
             ++index) {
            controllerDofIndices[index] = indexOf(
                robotDiagnostics.dofNames,
                controllerNames[index]
            );
            if (controllerDofIndices[index] == MR_INVALID_INDEX) {
                return fail(
                    std::move(diagnostics),
                    WaveAssetStatus::semanticMismatch,
                    "published controller joint is absent from cooked "
                    "generalized coordinates",
                    controllerNames[index]
                );
            }
        }

        const std::filesystem::path tactileRoot =
            config.tactileRepositoryRoot / "wave_01";
        NpyFloatArray fingerPoints;
        NpyFloatArray fingerNormals;
        NpyFloatArray thumbPoints;
        NpyFloatArray thumbNormals;
        std::string arrayMessage;
        const std::array<
            std::pair<std::filesystem::path, NpyFloatArray*>,
            4u
        > arrays{{
            {
                tactileRoot /
                    "tactileSensor_map_4F_point.npy",
                &fingerPoints,
            },
            {
                tactileRoot /
                    "tactileSensor_map_4F_normal.npy",
                &fingerNormals,
            },
            {
                tactileRoot /
                    "tactileSensor_map_TH_point.npy",
                &thumbPoints,
            },
            {
                tactileRoot /
                    "tactileSensor_map_TH_normal.npy",
                &thumbNormals,
            },
        }};
        for (const auto& [path, destination] : arrays) {
            if (!readNpy(path, *destination, arrayMessage)) {
                return fail(
                    std::move(diagnostics),
                    WaveAssetStatus::invalidTactileArray,
                    arrayMessage,
                    path.string()
                );
            }
        }

        const std::array<std::pair<std::string_view, std::string_view>, 5u>
            fingers{{
                {"thumb", "thumb"},
                {"index", "index"},
                {"middle", "middle"},
                {"ring", "ring"},
                {"pinky", "little"},
            }};
        std::vector<TactileSensorSpec> sensors;
        sensors.reserve(fingers.size());
        std::uint32_t validSampleCount = 0u;
        for (const auto& [urdfFinger, sensorFinger] : fingers) {
            const std::string link =
                side + "_" + std::string{urdfFinger} +
                "_elastomer";
            const std::uint32_t bodyIndex = indexOf(
                robotDiagnostics.bodyNames,
                link
            );
            if (bodyIndex == MR_INVALID_INDEX) {
                return fail(
                    std::move(diagnostics),
                    WaveAssetStatus::semanticMismatch,
                    "Wave elastomer link is absent",
                    link
                );
            }
            std::vector<std::uint32_t> backingShapes;
            for (std::uint32_t shapeIndex = 0u;
                 shapeIndex <
                    robotDiagnostics.shapeLinkNames.size();
                 ++shapeIndex) {
                if (
                    robotDiagnostics.shapeLinkNames[shapeIndex] ==
                    link
                ) {
                    backingShapes.push_back(shapeIndex);
                }
            }
            if (backingShapes.empty()) {
                return fail(
                    std::move(diagnostics),
                    WaveAssetStatus::semanticMismatch,
                    "Wave elastomer has no collision backing",
                    link
                );
            }
            if (config.compliantShell) {
                for (const std::uint32_t shapeIndex :
                     backingShapes) {
                    MRShapeGPU& shape = model.shapes[shapeIndex];
                    shape.contactRestAndBoundingRadius.y =
                        config.maximumDepthMeters;
                    shape.contactRestAndBoundingRadius.x =
                        std::max(
                            shape.contactRestAndBoundingRadius.x,
                            config.maximumDepthMeters
                        );
                }
            }
            TactileSensorSpec sensor;
            sensor.id =
                side + "_" + std::string{sensorFinger};
            sensor.parentBodyIndex = bodyIndex;
            sensor.backingShapeIndices =
                std::move(backingShapes);
            const mr_float4 centerOfMass =
                model.bodies[bodyIndex].centerOfMass;
            sensor.localPose.position = {
                -centerOfMass.x,
                -centerOfMass.y,
                -centerOfMass.z,
                0.0f,
            };
            sensor.width = config.atlasWidth;
            sensor.height = config.atlasHeight;
            sensor.surfaceKind =
                MR_TACTILE_SURFACE_CUSTOM_ATLAS;
            sensor.maximumDepthMeters =
                config.maximumDepthMeters;
            sensor.maximumTangentialDisplacementMeters =
                config.maximumDepthMeters;
            sensor.activeDepthThresholdMeters = 1.0e-6f;
            sensor.queryEpsilonMeters = 2.5e-7f;
            sensor.flags = config.compliantShell
                ? MR_TACTILE_SENSOR_COMPLIANT_SHELL
                : 0u;
            sensor.samples = tactileSamples(
                urdfFinger == "thumb"
                    ? thumbPoints
                    : fingerPoints,
                urdfFinger == "thumb"
                    ? thumbNormals
                    : fingerNormals,
                config.atlasWidth,
                config.atlasHeight,
                config.maximumDepthMeters,
                validSampleCount
            );
            sensors.push_back(std::move(sensor));
        }
        CookedTactileSystem cooked;
        const TactileCookResult tactileCook =
            cookTactileSystem(sensors, model, cooked);
        if (!tactileCook.succeeded()) {
            return fail(
                std::move(diagnostics),
                WaveAssetStatus::tactileCookFailure,
                tactileCook.message
            );
        }
        std::uint64_t fingerprint = kFnvOffset;
        hashBytes(
            fingerprint,
            &robotDiagnostics.sourceFingerprint,
            sizeof(robotDiagnostics.sourceFingerprint)
        );
        for (const auto& [path, array] : arrays) {
            hashString(fingerprint, path.filename().string());
            hashBytes(
                fingerprint,
                array->sourceBytes.data(),
                array->sourceBytes.size()
            );
        }
        hashString(fingerprint, urdfRevision);
        hashString(fingerprint, tactileRevision);
        hashBytes(
            fingerprint,
            &config.atlasWidth,
            sizeof(config.atlasWidth)
        );
        hashBytes(
            fingerprint,
            &config.atlasHeight,
            sizeof(config.atlasHeight)
        );
        hashBytes(
            fingerprint,
            &config.maximumDepthMeters,
            sizeof(config.maximumDepthMeters)
        );
        hashBytes(
            fingerprint,
            &config.compliantShell,
            sizeof(config.compliantShell)
        );
        const std::uint32_t handVectorOffset =
            config.side == WaveHandSide::left ? 7u : 36u;
        const std::uint32_t wrenchOffset =
            config.side == WaveHandSide::left ? 0u : 30u;
        const auto origamiStateIndices =
            sequentialIndices<22u>(handVectorOffset);
        const auto origamiActionIndices =
            sequentialIndices<22u>(handVectorOffset);
        const auto origamiWrenchIndices =
            sequentialIndices<30u>(wrenchOffset);
        hashBytes(
            fingerprint,
            controllerDofIndices.data(),
            sizeof(controllerDofIndices)
        );
        hashBytes(
            fingerprint,
            origamiStateIndices.data(),
            sizeof(origamiStateIndices)
        );
        hashBytes(
            fingerprint,
            origamiActionIndices.data(),
            sizeof(origamiActionIndices)
        );
        hashBytes(
            fingerprint,
            origamiWrenchIndices.data(),
            sizeof(origamiWrenchIndices)
        );

        WaveHandAssets candidate;
        candidate.model = std::move(model);
        candidate.robotDiagnostics =
            std::move(robotDiagnostics);
        candidate.tactileSensors = std::move(sensors);
        candidate.controllerDofIndices =
            controllerDofIndices;
        candidate.origamiStateIndices =
            origamiStateIndices;
        candidate.origamiActionIndices =
            origamiActionIndices;
        candidate.origamiWrenchIndices =
            origamiWrenchIndices;
        candidate.sourceFingerprint = fingerprint;
        candidate.urdfRevision = std::move(urdfRevision);
        candidate.tactileRevision =
            std::move(tactileRevision);
        diagnostics.validSampleCount = validSampleCount;
        diagnostics.sensorCount =
            static_cast<std::uint32_t>(
                candidate.tactileSensors.size()
            );
        diagnostics.sourceFingerprint = fingerprint;
        output = std::move(candidate);
        return diagnostics;
    } catch (const std::exception& exception) {
        return fail(
            std::move(diagnostics),
            WaveAssetStatus::internalFailure,
            exception.what()
        );
    }
}

WaveAssetDiagnostics cookSharpaWavePair(
    const WaveAssetConfig& config,
    WavePairAssets& output
) {
    WaveAssetConfig leftConfig = config;
    leftConfig.side = WaveHandSide::left;
    WaveHandAssets left;
    WaveAssetDiagnostics leftDiagnostics =
        cookSharpaWaveHand(leftConfig, left);
    if (!leftDiagnostics.succeeded()) {
        leftDiagnostics.element =
            leftDiagnostics.element.empty()
            ? "left"
            : "left:" + leftDiagnostics.element;
        return leftDiagnostics;
    }

    WaveAssetConfig rightConfig = config;
    rightConfig.side = WaveHandSide::right;
    WaveHandAssets right;
    WaveAssetDiagnostics rightDiagnostics =
        cookSharpaWaveHand(rightConfig, right);
    if (!rightDiagnostics.succeeded()) {
        rightDiagnostics.element =
            rightDiagnostics.element.empty()
            ? "right"
            : "right:" + rightDiagnostics.element;
        return rightDiagnostics;
    }

    WavePairAssets candidate;
    candidate.left = std::move(left);
    candidate.right = std::move(right);
    for (std::size_t index = 0u; index < 22u; ++index) {
        candidate.origamiStateIndices[index] =
            candidate.left.origamiStateIndices[index];
        candidate.origamiStateIndices[index + 22u] =
            candidate.right.origamiStateIndices[index];
        candidate.origamiActionIndices[index] =
            candidate.left.origamiActionIndices[index];
        candidate.origamiActionIndices[index + 22u] =
            candidate.right.origamiActionIndices[index];
    }
    for (std::size_t index = 0u; index < 30u; ++index) {
        candidate.origamiWrenchIndices[index] =
            candidate.left.origamiWrenchIndices[index];
        candidate.origamiWrenchIndices[index + 30u] =
            candidate.right.origamiWrenchIndices[index];
    }
    std::uint64_t fingerprint = kFnvOffset;
    hashString(fingerprint, "sharpa_wave_pair");
    hashBytes(
        fingerprint,
        &candidate.left.sourceFingerprint,
        sizeof(candidate.left.sourceFingerprint)
    );
    hashBytes(
        fingerprint,
        &candidate.right.sourceFingerprint,
        sizeof(candidate.right.sourceFingerprint)
    );
    hashBytes(
        fingerprint,
        candidate.origamiStateIndices.data(),
        sizeof(candidate.origamiStateIndices)
    );
    hashBytes(
        fingerprint,
        candidate.origamiActionIndices.data(),
        sizeof(candidate.origamiActionIndices)
    );
    hashBytes(
        fingerprint,
        candidate.origamiWrenchIndices.data(),
        sizeof(candidate.origamiWrenchIndices)
    );
    candidate.sourceFingerprint = fingerprint;

    WaveAssetDiagnostics diagnostics;
    diagnostics.validSampleCount =
        leftDiagnostics.validSampleCount +
        rightDiagnostics.validSampleCount;
    diagnostics.sensorCount =
        leftDiagnostics.sensorCount +
        rightDiagnostics.sensorCount;
    diagnostics.sourceFingerprint = fingerprint;
    output = std::move(candidate);
    return diagnostics;
}

std::vector<TactileSensorSpec>
rebaseSharpaWaveTactileSensors(
    const std::span<const TactileSensorSpec> sensors,
    const std::uint32_t bodyOffset,
    const std::uint32_t shapeOffset,
    const std::span<const std::uint32_t> targetShapeIndices
) {
    std::vector<TactileSensorSpec> result(
        sensors.begin(),
        sensors.end()
    );
    for (TactileSensorSpec& sensor : result) {
        if (
            sensor.parentBodyIndex >
            std::numeric_limits<std::uint32_t>::max() -
                bodyOffset
        ) {
            throw std::overflow_error(
                "Wave tactile body offset overflows"
            );
        }
        sensor.parentBodyIndex += bodyOffset;
        for (std::uint32_t& shape :
             sensor.backingShapeIndices) {
            if (
                shape >
                std::numeric_limits<std::uint32_t>::max() -
                    shapeOffset
            ) {
                throw std::overflow_error(
                    "Wave tactile shape offset overflows"
                );
            }
            shape += shapeOffset;
        }
        sensor.targetShapeIndices.assign(
            targetShapeIndices.begin(),
            targetShapeIndices.end()
        );
    }
    return result;
}

const char* waveAssetStatusName(
    const WaveAssetStatus status
) noexcept {
    switch (status) {
    case WaveAssetStatus::success:
        return "success";
    case WaveAssetStatus::invalidConfiguration:
        return "invalid_configuration";
    case WaveAssetStatus::sourceRevisionMismatch:
        return "source_revision_mismatch";
    case WaveAssetStatus::ioFailure:
        return "io_failure";
    case WaveAssetStatus::invalidTactileArray:
        return "invalid_tactile_array";
    case WaveAssetStatus::robotCookFailure:
        return "robot_cook_failure";
    case WaveAssetStatus::semanticMismatch:
        return "semantic_mismatch";
    case WaveAssetStatus::tactileCookFailure:
        return "tactile_cook_failure";
    case WaveAssetStatus::internalFailure:
        return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>

#include "cardboard_glue_mesh.hpp"
#include "metalrobo/engine_types.h"
#include "numi/matter/language.hpp"
#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <ranges>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace numi::matter;

constexpr std::uint32_t kEnvironmentCount = 2u;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct CrossSectionPoint {
    double x = 0.0;
    double z = 0.0;
};

struct Arguments {
    std::filesystem::path output;
    std::filesystem::path linerMaterial;
    std::filesystem::path mediumMaterial;
    std::filesystem::path glueMaterial;
    double glueGapM = 0.0;
    double bondWidthM = 0.0;
    double upperGlueGapM = 0.0;
    double upperBondWidthM = 0.0;
    bool disableSelfContact = false;
    std::string preset = "literature2009";
    bool withoutMedium = false;
    bool compileOnly = false;
    bool help = false;
    std::uint32_t steps = 8u;
    std::uint32_t holdSteps = 0u;
    std::uint32_t unloadSteps = 0u;
    std::uint32_t relaxSteps = 0u;
    std::uint32_t fgmresIterations = NM_MIXED_FGMRES_ITERATIONS;
    std::uint32_t localMaterialIterations = 8u;
    std::uint32_t newtonIterations = NM_MIXED_NEWTON_ITERATIONS;
    std::uint32_t nxPerPitch = 16u;
    std::uint32_t ny = 4u;
    std::uint32_t thicknessSlices = 2u;
    double bendAngleDegrees = 1.0;
    double timestepSeconds = 1.0e-4;
    double lengthM = 0.036;
    double widthM = 0.020;
    double totalHeightM = 0.005;
    double pitchM = 0.009;
    double linerThicknessM = 0.00025;
    double mediumThicknessM = 0.00025;
};

struct TetMetadata {
    std::array<std::uint32_t, 4> nodes{};
    std::uint32_t materialIndex = 0u;
    double frameAngle = 0.0;
};

struct TriangleCell {
    std::array<std::uint32_t, 3> points{};
    std::uint32_t materialIndex = 0u;
    double frameAngle = 0.0;
};

struct MeshSource {
    WorldSource world;
    std::vector<TetMetadata> tetrahedra;
    std::vector<std::uint32_t> leftGripNodes;
    std::vector<std::uint32_t> rightGripNodes;
    std::vector<Vec3> restNodes;
    std::vector<std::array<double, 4>> materialFrames;
    std::vector<std::uint32_t> materialIndices;
    std::string materialMapDigest;
    std::string frameMapDigest;
    std::vector<numi::cardboard::GlueFootprint> glueFootprints;
};

struct Digest {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> bytes{};
};

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string jsonEscape(const std::string_view input) {
    std::ostringstream result;
    for (const unsigned char value : input) {
        switch (value) {
        case '"': result << "\\\""; break;
        case '\\': result << "\\\\"; break;
        case '\b': result << "\\b"; break;
        case '\f': result << "\\f"; break;
        case '\n': result << "\\n"; break;
        case '\r': result << "\\r"; break;
        case '\t': result << "\\t"; break;
        default:
            if (value < 0x20u) {
                result << "\\u" << std::hex << std::setw(4)
                       << std::setfill('0') << static_cast<unsigned>(value)
                       << std::dec;
            } else {
                result << static_cast<char>(value);
            }
        }
    }
    return result.str();
}

Digest sha256(const void* bytes, const std::size_t count) {
    Digest result;
    require(count <= std::numeric_limits<CC_LONG>::max(),
            "SHA-256 input exceeds CommonCrypto's bounded length");
    CC_SHA256(bytes, static_cast<CC_LONG>(count), result.bytes.data());
    return result;
}

Digest sha256(const std::string_view value) {
    return sha256(value.data(), value.size());
}

Digest sha256File(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "cannot read material file: " + path.string());
    std::ostringstream contents;
    contents << input.rdbuf();
    require(input.good() || input.eof(),
            "failed while reading material file: " + path.string());
    const std::string bytes = contents.str();
    return sha256(bytes.data(), bytes.size());
}

std::string hexDigest(const Digest& digest) {
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (const unsigned char byte : digest.bytes)
        result << std::setw(2) << static_cast<unsigned>(byte);
    return result.str();
}

std::array<std::uint64_t, 4> digestIdentity(const Digest& digest) {
    std::array<std::uint64_t, 4> identity{};
    std::memcpy(identity.data(), digest.bytes.data(), digest.bytes.size());
    return identity;
}

std::string diagnosticText(const std::vector<Diagnostic>& diagnostics) {
    std::string result;
    for (const Diagnostic& diagnostic : diagnostics) {
        if (diagnostic.severity != Diagnostic::Severity::error) continue;
        if (!result.empty()) result += "; ";
        result += diagnostic.message;
    }
    return result;
}

double parseDouble(const std::string& value, const std::string& option) {
    std::size_t used = 0u;
    const double parsed = std::stod(value, &used);
    require(used == value.size() && std::isfinite(parsed),
            "invalid number for " + option + ": " + value);
    return parsed;
}

std::uint32_t parseUnsigned(const std::string& value,
                            const std::string& option) {
    std::size_t used = 0u;
    const unsigned long parsed = std::stoul(value, &used);
    require(used == value.size() &&
                parsed <= std::numeric_limits<std::uint32_t>::max(),
            "invalid unsigned integer for " + option + ": " + value);
    return static_cast<std::uint32_t>(parsed);
}

void printUsage() {
    std::cout
        << "usage: numi-matter-cardboard-probe --output DIR [options]\n"
        << "  --preset nagasawa2013|literature2009 (default literature2009)\n"
        << "  --upper-glue-gap-mm N --upper-bond-width-mm N  asymmetric glue (0 uses lower value)\n"
        << "  --without-self-contact  numerical control only for the finite-glue mesh\n"
        << "  --material liner=FILE --material medium=FILE\n"
        << "  --liner-material FILE --medium-material FILE\n"
        << "  --glue-material FILE --glue-gap-mm N --bond-width-mm N  finite glue bridges\n"
        << "  --without-medium       paired two-liner control geometry\n"
        << "  --steps N              ramp the end-grip rotation\n"
        << "  --hold-steps N --unload-steps N --relax-steps N  loading cycle phases\n"
        << "  --newton-iterations N  global nonlinear budget (1..64)\n"
        << "  --material-iterations N  local constitutive Newton budget (1..16)\n"
        << "  --fgmres-iterations N  Krylov-column budget (max 256)\n"
        << "  --bend-angle DEGREES   final right-grip rotation\n"
        << "  --dt SECONDS           frame timestep\n"
        << "  --nx N                 cross-section subdivisions per flute pitch\n"
        << "  --ny N                 subdivisions across the width\n"
        << "  --thickness-slices N   elements through each paper layer\n"
        << "  --length-mm N --width-mm N --height-mm N --pitch-mm N\n"
        << "  --liner-thickness-mm N --medium-thickness-mm N\n"
        << "  --compile-only         write package and initial OBJ without creating a Metal runtime\n";
}

void setMaterialPath(Arguments& arguments,
                     const std::string& assignment) {
    const std::size_t separator = assignment.find('=');
    require(separator != std::string::npos && separator > 0u &&
                separator + 1u < assignment.size(),
            "--material expects liner=FILE or medium=FILE");
    const std::string role = assignment.substr(0u, separator);
    const std::filesystem::path path = assignment.substr(separator + 1u);
    if (role == "liner") arguments.linerMaterial = path;
    else if (role == "medium") arguments.mediumMaterial = path;
    else require(false, "unknown material role in --material: " + role);
}

Arguments parseArguments(const int argc, const char* argv[]) {
    Arguments arguments;
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string_view(argv[i]) == "--preset") arguments.preset = argv[i + 1];
    arguments.linerMaterial =
        std::filesystem::path(NUMI_CARDBOARD_MATERIAL_DIR) /
        "nagasawa2013_liner_orthotropic.nmatter";
    arguments.mediumMaterial =
        std::filesystem::path(NUMI_CARDBOARD_MATERIAL_DIR) /
        "nagasawa2013_medium_orthotropic.nmatter";

    if (arguments.preset == "literature2009") {
        const auto root = std::filesystem::path(NUMI_CARDBOARD_MATERIAL_DIR);
        arguments.linerMaterial = root / "hajali2009_liner_hill_ideal.nmatter";
        arguments.mediumMaterial = root / "hajali2009_medium_hill_ideal.nmatter";
        arguments.glueMaterial = root / "starch2007_finite_glue_elastic.nmatter";
        arguments.pitchM = 0.0079; arguments.lengthM = 4.0 * arguments.pitchM;
        arguments.totalHeightM = 0.004210;
        arguments.linerThicknessM = 0.000277; arguments.mediumThicknessM = 0.000191;
        arguments.glueGapM = 0.000100; arguments.bondWidthM = 0.000800;
        arguments.upperGlueGapM = 0.000030; arguments.upperBondWidthM = 0.000600;
        arguments.fgmresIterations = 32u;
        arguments.localMaterialIterations = 16u;
        arguments.newtonIterations = 14u;
        arguments.nxPerPitch = 8u;
        arguments.ny = 1u;
        arguments.thicknessSlices = 1u;
        arguments.steps = 32u;
        arguments.holdSteps = 16u;
        arguments.unloadSteps = 32u;
        arguments.relaxSteps = 64u;
        arguments.timestepSeconds = 0.000025;
        arguments.bendAngleDegrees = 0.25;
    }
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--without-self-contact") { arguments.disableSelfContact = true; continue; }
        if (option == "--help" || option == "-h") {
            arguments.help = true;
            return arguments;
        }
        if (option == "--without-medium") {
            arguments.withoutMedium = true;
            continue;
        }
        if (option == "--compile-only") {
            arguments.compileOnly = true;
            continue;
        }
        require(index + 1 < argc, "missing value for " + option);
        const std::string value = argv[++index];
        if (option == "--output") arguments.output = value;
        else if (option == "--newton-iterations") arguments.newtonIterations = parseUnsigned(value, option);
        else if (option == "--material-iterations") arguments.localMaterialIterations = parseUnsigned(value, option);
        else if (option == "--preset") arguments.preset = value;
        else if (option == "--material") setMaterialPath(arguments, value);
        else if (option == "--liner-material") arguments.linerMaterial = value;
        else if (option == "--medium-material") arguments.mediumMaterial = value;
        else if (option == "--steps") arguments.steps = parseUnsigned(value, option);
        else if (option == "--upper-glue-gap-mm") arguments.upperGlueGapM = parseDouble(value, option) * 1e-3;
        else if (option == "--upper-bond-width-mm") arguments.upperBondWidthM = parseDouble(value, option) * 1e-3;
        else if (option == "--glue-material") arguments.glueMaterial = value;
        else if (option == "--glue-gap-mm") arguments.glueGapM = parseDouble(value, option) * 1e-3;
        else if (option == "--bond-width-mm") arguments.bondWidthM = parseDouble(value, option) * 1e-3;
        else if (option == "--hold-steps") arguments.holdSteps = parseUnsigned(value, option);
        else if (option == "--unload-steps") arguments.unloadSteps = parseUnsigned(value, option);
        else if (option == "--relax-steps") arguments.relaxSteps = parseUnsigned(value, option);
        else if (option == "--fgmres-iterations")
            arguments.fgmresIterations = parseUnsigned(value, option);
        else if (option == "--nx") arguments.nxPerPitch = parseUnsigned(value, option);
        else if (option == "--ny") arguments.ny = parseUnsigned(value, option);
        else if (option == "--thickness-slices")
            arguments.thicknessSlices = parseUnsigned(value, option);
        else if (option == "--bend-angle")
            arguments.bendAngleDegrees = parseDouble(value, option);
        else if (option == "--dt")
            arguments.timestepSeconds = parseDouble(value, option);
        else if (option == "--length-mm")
            arguments.lengthM = parseDouble(value, option) * 1.0e-3;
        else if (option == "--width-mm")
            arguments.widthM = parseDouble(value, option) * 1.0e-3;
        else if (option == "--height-mm")
            arguments.totalHeightM = parseDouble(value, option) * 1.0e-3;
        else if (option == "--pitch-mm")
            arguments.pitchM = parseDouble(value, option) * 1.0e-3;
        else if (option == "--liner-thickness-mm")
            arguments.linerThicknessM = parseDouble(value, option) * 1.0e-3;
        else if (option == "--medium-thickness-mm")
            arguments.mediumThicknessM = parseDouble(value, option) * 1.0e-3;
        else require(false, "unknown option: " + option);
    }

    require(arguments.glueMaterial.empty() == (arguments.glueGapM == 0.0 && arguments.bondWidthM == 0.0),
            "finite glue requires --glue-material, positive --glue-gap-mm and --bond-width-mm together");
    if (!arguments.glueMaterial.empty()) {
        require(!arguments.withoutMedium && arguments.glueGapM > 0 && arguments.bondWidthM > 0 &&
                arguments.bondWidthM < 0.45 * arguments.pitchM, "finite glue dimensions or medium are invalid");
    }
    require(arguments.newtonIterations > 0 && arguments.newtonIterations <= 64, "global Newton iterations must be 1..64");
    require(arguments.localMaterialIterations >= 1u && arguments.localMaterialIterations <= 16u, "material iterations must be 1..16");
    require(arguments.preset == "nagasawa2013" || arguments.preset == "literature2009",
            "unsupported preset");
    require(!arguments.glueMaterial.empty() || (arguments.upperGlueGapM == 0 && arguments.upperBondWidthM == 0), "upper glue dimensions require a glue material");
    require(arguments.upperGlueGapM >= 0 && arguments.upperBondWidthM >= 0, "upper glue dimensions must be nonnegative");
    require(!arguments.output.empty(), "--output DIR is required");
    require(arguments.steps > 0u && arguments.steps <= 10000u,
            "--steps must lie in [1, 10000]");
    require(arguments.holdSteps <= 10000u && arguments.unloadSteps <= 10000u &&
                arguments.relaxSteps <= 10000u, "cycle phase exceeds 10000 steps");
    require(arguments.relaxSteps == 0u || arguments.unloadSteps > 0u,
            "relaxation requires an unloading phase; no instantaneous release");
    require(arguments.fgmresIterations >= NM_MIXED_FGMRES_DEFAULT_RESTART &&
                arguments.fgmresIterations <= 256u,
            "--fgmres-iterations must lie between the compiled restart size and 256");
    require(arguments.nxPerPitch >= 8u && arguments.nxPerPitch <= 128u &&
                arguments.nxPerPitch % 2u == 0u,
            "--nx must be an even value in [8, 128]");
    require(arguments.ny >= 1u && arguments.ny <= 32u,
            "--ny must lie in [1, 32]");
    require(arguments.thicknessSlices >= 1u &&
                arguments.thicknessSlices <= 4u,
            "--thickness-slices must lie in [1, 4]");
    require(arguments.bendAngleDegrees >= 0.0 &&
                arguments.bendAngleDegrees <= 10.0,
            "--bend-angle must lie in [0, 10] degrees");
    require(arguments.timestepSeconds > 0.0 &&
                arguments.timestepSeconds <= 0.01,
            "--dt must lie in (0, 0.01] seconds");
    require(arguments.lengthM > 0.0 && arguments.widthM > 0.0 &&
                arguments.totalHeightM > 0.0 && arguments.pitchM > 0.0 &&
                arguments.linerThicknessM > 0.0 &&
                arguments.mediumThicknessM > 0.0,
            "all geometry dimensions must be positive");
    const double pitchCount = arguments.lengthM / arguments.pitchM;
    require(std::abs(pitchCount - std::round(pitchCount)) < 1.0e-9 &&
                pitchCount >= 1.0 && pitchCount <= 32.0,
            "length must be an integer number of pitches in [1, 32]");
    require(arguments.totalHeightM - 2.0 * arguments.linerThicknessM >
                arguments.mediumThicknessM,
            "board height leaves no positive corrugated-medium clearance");
    const double estimatedIntervals =
        pitchCount * static_cast<double>(arguments.nxPerPitch);
    const double estimatedTetrahedra = estimatedIntervals * 36.0 *
        static_cast<double>(arguments.ny) *
        static_cast<double>(arguments.thicknessSlices);
    require(estimatedTetrahedra <= 250000.0,
            "requested mesh exceeds the bounded 250000-tetrahedron probe limit");
    return arguments;
}

MaterialProgram readMaterial(const std::filesystem::path& path,
                             const std::string& role) {
    const ParseResult parsed = parseMatterLanguageFile(path);
    require(parsed.succeeded(), role + " material parse failed: " +
                diagnosticText(parsed.diagnostics));
    MaterialProgram result = parsed.material;
    require(std::ranges::find(result.supportedRepresentations,
                              Representation::fem) !=
                result.supportedRepresentations.end(),
            role + " material does not declare FEM support");
    require(std::ranges::none_of(result.parameters,
                                 [](const Parameter& parameter) {
                                     return parameter.identifiable;
                                 }),
            role + " material has identifiable parameters; this probe requires a frozen source file");
    const auto density = std::ranges::find_if(
        result.parameters,
        [](const Parameter& parameter) { return parameter.name == "density"; });
    require(density != result.parameters.end() &&
                std::isfinite(density->defaultValue) &&
                density->defaultValue > 0.0,
            role + " material must have a positive density parameter");
    return result;
}

double materialDensity(const MaterialProgram& material) {
    const auto density = std::ranges::find_if(
        material.parameters,
        [](const Parameter& parameter) { return parameter.name == "density"; });
    require(density != material.parameters.end(),
            "compiled paper material lost its density parameter");
    return density->defaultValue;
}

void requireSameInterface(const MaterialProgram& first,
                          const MaterialProgram& second) {
    require(first.staticFriction == second.staticFriction &&
                first.dynamicFriction == second.dynamicFriction &&
                first.restitution == second.restitution &&
                first.adhesion == second.adhesion,
            "regional paper materials must have exactly matching interface parameters");
}

struct PointKey {
    std::int64_t x = 0;
    std::int64_t z = 0;
    friend bool operator<(const PointKey& a, const PointKey& b) {
        return a.x < b.x || (a.x == b.x && a.z < b.z);
    }
};

class CrossSectionBuilder {
public:
    std::uint32_t point(const double x, const double z) {
        require(std::isfinite(x) && std::isfinite(z),
                "nonfinite cross-section point");
        constexpr double scale = 1.0e12;
        const PointKey key{
            static_cast<std::int64_t>(std::llround(x * scale)),
            static_cast<std::int64_t>(std::llround(z * scale)),
        };
        const auto found = indices_.find(key);
        if (found != indices_.end()) return found->second;
        const std::uint32_t index = static_cast<std::uint32_t>(points_.size());
        indices_.emplace(key, index);
        points_.push_back({x, z});
        return index;
    }

    void quad(const std::array<std::uint32_t, 4>& corners,
              const std::uint32_t materialIndex,
              const double frameAngle) {
        triangle({corners[0], corners[1], corners[2]}, materialIndex,
                 frameAngle);
        triangle({corners[0], corners[2], corners[3]}, materialIndex,
                 frameAngle);
    }

    [[nodiscard]] const std::vector<CrossSectionPoint>& points() const {
        return points_;
    }

    [[nodiscard]] const std::vector<TriangleCell>& triangles() const {
        return triangles_;
    }

public:
    void triangle(const std::array<std::uint32_t, 3> indices,
                  const std::uint32_t materialIndex,
                  const double frameAngle) {
        const CrossSectionPoint& a = points_.at(indices[0]);
        const CrossSectionPoint& b = points_.at(indices[1]);
        const CrossSectionPoint& c = points_.at(indices[2]);
        const double twiceArea = (b.x - a.x) * (c.z - a.z) -
            (b.z - a.z) * (c.x - a.x);
        require(std::isfinite(twiceArea) && std::abs(twiceArea) > 1.0e-15,
                "cardboard cross-section contains a zero-area triangle");
        triangles_.push_back({indices, materialIndex, frameAngle});
    }

private:
    std::map<PointKey, std::uint32_t> indices_;
    std::vector<CrossSectionPoint> points_;
    std::vector<TriangleCell> triangles_;
};

std::vector<double> fluteSampleXs(const Arguments& arguments) {
    const double step = arguments.pitchM /
        static_cast<double>(arguments.nxPerPitch);
    const std::uint32_t periods = static_cast<std::uint32_t>(std::llround(
        arguments.lengthM / arguments.pitchM));
    const std::uint64_t regularCount =
        static_cast<std::uint64_t>(periods) * arguments.nxPerPitch;
    std::vector<double> xs;
    xs.reserve(static_cast<std::size_t>(regularCount + periods * 2u + 1u));
    for (std::uint64_t index = 0u; index <= regularCount; ++index)
        xs.push_back(static_cast<double>(index) * step);
    for (std::uint32_t contact = 0u; contact <= periods * 2u; ++contact)
        xs.push_back(static_cast<double>(contact) * arguments.pitchM * 0.5);
    xs.push_back(arguments.lengthM);
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end(), [](const double a, const double b) {
                 return std::abs(a - b) <= 1.0e-12;
             }), xs.end());
    require(xs.size() >= 2u && std::abs(xs.front()) < 1.0e-12 &&
                std::abs(xs.back() - arguments.lengthM) < 1.0e-12,
            "corrugation sample span is incomplete");
    return xs;
}

struct WaveState {
    double center = 0.0;
    double slope = 0.0;
    bool valley = false;
    bool crest = false;
};

WaveState waveState(const Arguments& arguments, const double x) {
    const double gap = arguments.totalHeightM -
        2.0 * arguments.linerThicknessM;
    const double centerlineHeight = gap - arguments.mediumThicknessM;
    const double amplitude = 0.5 * centerlineHeight;
    const double midpoint = 0.5 * (
        arguments.linerThicknessM + arguments.mediumThicknessM * 0.5 +
        arguments.totalHeightM - arguments.linerThicknessM -
        arguments.mediumThicknessM * 0.5);
    const double halfPitch = 0.5 * arguments.pitchM;
    const double contactOrdinal = std::round(x / halfPitch);
    const double contactX = contactOrdinal * halfPitch;
    const bool atContact = std::abs(x - contactX) <= 1.0e-10;
    if (atContact) {
        const bool valley = (static_cast<std::int64_t>(contactOrdinal) % 2) == 0;
        return {
            midpoint + (valley ? -amplitude : amplitude),
            0.0,
            valley,
            !valley,
        };
    }
    const double phase = 2.0 * kPi * x / arguments.pitchM;
    return {
        midpoint - amplitude * std::cos(phase),
        amplitude * (2.0 * kPi / arguments.pitchM) * std::sin(phase),
        false,
        false,
    };
}

std::array<double, 4> frameQuaternion(const double tangentAngle) {
    // The local material axes map to flute tangent, board width, and the
    // right-handed in-plane normal. Quaternions are (x,y,z,w), material to
    // reference-world; a negative y rotation maps local x to (cos,0,sin).
    return {0.0, -std::sin(0.5 * tangentAngle), 0.0,
            std::cos(0.5 * tangentAngle)};
}

double signedSixVolume(const Vec3& a, const Vec3& b,
                       const Vec3& c, const Vec3& d) {
    const Vec3 u{b.x - a.x, b.y - a.y, b.z - a.z};
    const Vec3 v{c.x - a.x, c.y - a.y, c.z - a.z};
    const Vec3 w{d.x - a.x, d.y - a.y, d.z - a.z};
    return u.x * (v.y * w.z - v.z * w.y) -
        u.y * (v.x * w.z - v.z * w.x) +
        u.z * (v.x * w.y - v.y * w.x);
}

void addPrism(const TriangleCell& triangle,
              const std::uint32_t ySlice,
              const std::uint32_t pointsPerSlice,
              const std::vector<Vec3>& nodes,
              std::vector<TetMetadata>& tetrahedra) {
    std::array<std::uint32_t, 3> pointIds = triangle.points;
    std::sort(pointIds.begin(), pointIds.end());
    const std::uint32_t bottom0 = ySlice * pointsPerSlice + pointIds[0];
    const std::uint32_t bottom1 = ySlice * pointsPerSlice + pointIds[1];
    const std::uint32_t bottom2 = ySlice * pointsPerSlice + pointIds[2];
    const std::uint32_t top0 = bottom0 + pointsPerSlice;
    const std::uint32_t top1 = bottom1 + pointsPerSlice;
    const std::uint32_t top2 = bottom2 + pointsPerSlice;
    const std::array<std::array<std::uint32_t, 4>, 3> candidates{{
        {{bottom0, bottom1, bottom2, top2}},
        {{bottom0, bottom1, top1, top2}},
        {{bottom0, top0, top1, top2}},
    }};
    for (auto nodesForTet : candidates) {
        const double sixVolume = signedSixVolume(
            nodes[nodesForTet[0]], nodes[nodesForTet[1]],
            nodes[nodesForTet[2]], nodes[nodesForTet[3]]);
        require(std::isfinite(sixVolume) && std::abs(sixVolume) > 1.0e-20,
                "extruded cardboard mesh contains a zero-volume tetrahedron");
        if (sixVolume < 0.0) std::swap(nodesForTet[1], nodesForTet[2]);
        tetrahedra.push_back({nodesForTet, triangle.materialIndex,
                              triangle.frameAngle});
    }
}

MeshSource buildMesh(const Arguments& arguments,
                     const MaterialProgram& linerMaterial,
                     const MaterialProgram& mediumMaterial,
                     const std::string& linerDigest,
                     const std::string& mediumDigest,
                     const MaterialProgram* glueMaterial, const std::string& glueDigest) {
    requireSameInterface(linerMaterial, mediumMaterial);
    CrossSectionBuilder crossSection;
    if (glueMaterial == nullptr) {
        const std::vector<double> xs = fluteSampleXs(arguments);
        const double linerBottom = arguments.linerThicknessM;
        const double linerTop = arguments.totalHeightM -
            arguments.linerThicknessM;

        // Flat outer liners are separate finite-thickness continua. Their mesh
        // includes the exact flute-contact abscissae so contact nodes can be shared
        // without filling or flattening the intervening voids.
        for (std::size_t xIndex = 0u; xIndex + 1u < xs.size(); ++xIndex) {
            const double x0 = xs[xIndex];
            const double x1 = xs[xIndex + 1u];
            for (std::uint32_t slice = 0u;
                 slice < arguments.thicknessSlices; ++slice) {
                const double alpha0 = static_cast<double>(slice) /
                    static_cast<double>(arguments.thicknessSlices);
                const double alpha1 = static_cast<double>(slice + 1u) /
                    static_cast<double>(arguments.thicknessSlices);
                const double bottomZ0 = alpha0 * arguments.linerThicknessM;
                const double bottomZ1 = alpha1 * arguments.linerThicknessM;
                const double topZ0 = linerTop + alpha0 * arguments.linerThicknessM;
                const double topZ1 = linerTop + alpha1 * arguments.linerThicknessM;
                crossSection.quad({
                    crossSection.point(x0, bottomZ0),
                    crossSection.point(x1, bottomZ0),
                    crossSection.point(x1, bottomZ1),
                    crossSection.point(x0, bottomZ1),
                }, 0u, 0.0);
                crossSection.quad({
                    crossSection.point(x0, topZ0),
                    crossSection.point(x1, topZ0),
                    crossSection.point(x1, topZ1),
                    crossSection.point(x0, topZ1),
                }, 0u, 0.0);
            }
        }

        if (!arguments.withoutMedium) {
            const double halfThickness = 0.5 * arguments.mediumThicknessM;
            std::vector<CrossSectionPoint> lower(xs.size());
            std::vector<CrossSectionPoint> upper(xs.size());
            std::vector<double> slopes(xs.size());
            for (std::size_t index = 0u; index < xs.size(); ++index) {
                const double x = xs[index];
                const WaveState wave = waveState(arguments, x);
                const double normalScale = std::sqrt(1.0 + wave.slope * wave.slope);
                lower[index] = {
                    x + halfThickness * wave.slope / normalScale,
                    wave.center - halfThickness / normalScale,
                };
                upper[index] = {
                    x - halfThickness * wave.slope / normalScale,
                    wave.center + halfThickness / normalScale,
                };
                slopes[index] = wave.slope;
                if (wave.valley) {
                    lower[index] = {x, linerBottom};
                    upper[index] = {x, linerBottom + arguments.mediumThicknessM};
                } else if (wave.crest) {
                    upper[index] = {x, linerTop};
                    lower[index] = {x, linerTop - arguments.mediumThicknessM};
                }
            }
            for (std::size_t xIndex = 0u; xIndex + 1u < xs.size(); ++xIndex) {
                const double midpointX = 0.5 * (xs[xIndex] + xs[xIndex + 1u]);
                const double tangentAngle = std::atan(waveState(arguments, midpointX).slope);
                for (std::uint32_t slice = 0u;
                     slice < arguments.thicknessSlices; ++slice) {
                    const double alpha0 = static_cast<double>(slice) /
                        static_cast<double>(arguments.thicknessSlices);
                    const double alpha1 = static_cast<double>(slice + 1u) /
                        static_cast<double>(arguments.thicknessSlices);
                    const auto pointAt = [&](const std::size_t xIndex,
                                             const double alpha) {
                        return CrossSectionPoint{
                            lower[xIndex].x * (1.0 - alpha) + upper[xIndex].x * alpha,
                            lower[xIndex].z * (1.0 - alpha) + upper[xIndex].z * alpha,
                        };
                    };
                    const CrossSectionPoint p00 = pointAt(xIndex, alpha0);
                    const CrossSectionPoint p10 = pointAt(xIndex + 1u, alpha0);
                    const CrossSectionPoint p11 = pointAt(xIndex + 1u, alpha1);
                    const CrossSectionPoint p01 = pointAt(xIndex, alpha1);
                    crossSection.quad({
                        crossSection.point(p00.x, p00.z),
                        crossSection.point(p10.x, p10.z),
                        crossSection.point(p11.x, p11.z),
                        crossSection.point(p01.x, p01.z),
                    }, 1u, tangentAngle);
                }
            }
        }
    }

    MeshSource result;
    if (glueMaterial != nullptr) {
        requireSameInterface(linerMaterial, *glueMaterial);
        const auto glued = numi::cardboard::buildConformingGlueCrossSection({
            .length = arguments.lengthM, .pitch = arguments.pitchM,
            .caliper = arguments.totalHeightM, .linerThickness = arguments.linerThicknessM,
            .mediumThickness = arguments.mediumThicknessM, .glueMinimumGap = arguments.glueGapM,
            .bondWidth = arguments.bondWidthM,
            .upperGlueMinimumGap = arguments.upperGlueGapM, .upperBondWidth = arguments.upperBondWidthM,
            .nxPerPitch = arguments.nxPerPitch, .thicknessSlices = arguments.thicknessSlices});
        std::vector<std::uint32_t> remap;
        for (const auto& point : glued.points) remap.push_back(crossSection.point(point.x, point.z));
        for (const auto& cell : glued.triangles)
            crossSection.triangle({remap.at(cell.points[0]), remap.at(cell.points[1]), remap.at(cell.points[2])},
                static_cast<std::uint32_t>(cell.material), cell.frameAngle);
        result.glueFootprints = glued.glueFootprints;
    }
    const std::uint32_t pointsPerSlice =
        static_cast<std::uint32_t>(crossSection.points().size());
    require(pointsPerSlice > 0u, "cardboard source mesh is empty");
    result.restNodes.reserve(
        static_cast<std::size_t>(pointsPerSlice) * (arguments.ny + 1u));
    for (std::uint32_t ySlice = 0u; ySlice <= arguments.ny; ++ySlice) {
        const double y = arguments.widthM * static_cast<double>(ySlice) /
            static_cast<double>(arguments.ny);
        for (const CrossSectionPoint& point : crossSection.points())
            result.restNodes.push_back({point.x, y, point.z});
    }
    for (std::uint32_t ySlice = 0u; ySlice < arguments.ny; ++ySlice)
        for (const TriangleCell& triangle : crossSection.triangles())
            addPrism(triangle, ySlice, pointsPerSlice, result.restNodes,
                     result.tetrahedra);

    ObjectSource object;
    object.name = arguments.withoutMedium
        ? "cardboard_two_liner_bending_control"
        : "cardboard_explicit_sinusoidal_corrugated_strip";
    object.materialIndex = 0u;
    object.representation = Representation::fem;
    object.mixedFEM = false;
    object.adaptive = false;
    object.deformableContact = glueMaterial != nullptr && !arguments.disableSelfContact;
    object.deformableSelfContact = object.deformableContact;
    object.characteristicLength = std::min(
        arguments.linerThicknessM, arguments.mediumThicknessM) /
        static_cast<double>(arguments.thicknessSlices);
    object.femNodes.reserve(result.restNodes.size());
    for (const Vec3& point : result.restNodes)
        object.femNodes.push_back({point.x, point.y, point.z});
    object.tetrahedra.reserve(result.tetrahedra.size());
    result.materialIndices.reserve(result.tetrahedra.size());
    result.materialFrames.reserve(result.tetrahedra.size());
    for (const TetMetadata& tetrahedron : result.tetrahedra) {
        object.tetrahedra.push_back({tetrahedron.nodes});
        result.materialIndices.push_back(tetrahedron.materialIndex);
        result.materialFrames.push_back(frameQuaternion(tetrahedron.frameAngle));
    }

    const double gripTolerance = 1.0e-10;
    for (std::uint32_t node = 0u; node < result.restNodes.size(); ++node) {
        const double x = result.restNodes[node].x;
        if (std::abs(x) <= gripTolerance) result.leftGripNodes.push_back(node);
        if (std::abs(x - arguments.lengthM) <= gripTolerance)
            result.rightGripNodes.push_back(node);
    }
    require(!result.leftGripNodes.empty() && !result.rightGripNodes.empty(),
            "the authored grip boundaries contain no mesh nodes");
    object.femFixedNodes = result.leftGripNodes;
    object.femFixedNodes.insert(object.femFixedNodes.end(),
                                result.rightGripNodes.begin(),
                                result.rightGripNodes.end());
    std::sort(object.femFixedNodes.begin(), object.femFixedNodes.end());
    object.femFixedNodes.erase(
        std::unique(object.femFixedNodes.begin(), object.femFixedNodes.end()),
        object.femFixedNodes.end());

    result.world.frameTimestep = arguments.timestepSeconds;
    result.world.gravity = {0.0, 0.0, 0.0};
    if (glueMaterial != nullptr) result.world.contactSlop = 0.01 * std::min(arguments.mediumThicknessM, arguments.linerThicknessM);
    result.world.environmentCount = kEnvironmentCount;
    result.world.deterministic = true;
    result.world.mixedSolver.fgmresIterations = arguments.fgmresIterations;
    result.world.mixedSolver.newtonIterations = arguments.newtonIterations;
    result.world.materials = {linerMaterial, mediumMaterial};
    if (glueMaterial != nullptr) result.world.materials.push_back(*glueMaterial);
    if (!arguments.withoutMedium) {
        object.femMaterialIndices = result.materialIndices;
        std::ostringstream mapSource;
        mapSource << "numi.cardboard.layer-material-map.v1\n"
                  << "geometry=explicit_sinusoidal_approximation\n"
                  << "liner_sha256=" << linerDigest << '\n'
                  << "medium_sha256=" << mediumDigest << '\n'
                  << "glue_sha256=" << glueDigest << '\n'
                  << std::setprecision(17)
                  << "liner_density_kg_m3=" << materialDensity(linerMaterial) << '\n'
                  << "medium_density_kg_m3=" << materialDensity(mediumMaterial) << '\n';
        for (std::size_t index = 0u; index < result.materialIndices.size(); ++index)
            mapSource << index << ':' << result.materialIndices[index] << '\n';
        const Digest mapDigest = sha256(mapSource.str());
        object.femMaterialSourceIdentity = digestIdentity(mapDigest);
        result.materialMapDigest = hexDigest(mapDigest);

        std::ostringstream frameSource;
        frameSource << "numi.cardboard.flute-tangent-frame.v1\n"
                    << "source=analytic_sinusoidal_centerline\n"
                    << "axes=local-x:tangent,local-y:board-width,local-z:in-plane-normal\n"
                    << "quaternion=material-to-reference-world_xyzw\n"
                    << "liner_frame=identity\n"
                    << "liner_sha256=" << linerDigest << '\n'
                    << "medium_sha256=" << mediumDigest << '\n'
                  << "glue_sha256=" << glueDigest << '\n'
                    << std::hexfloat;
        for (const auto& frame : result.materialFrames)
            frameSource << frame[0] << ',' << frame[1] << ',' << frame[2] << ','
                        << frame[3] << '\n';
        const Digest frameDigest = sha256(frameSource.str());
        object.femMaterialFrameRotations = result.materialFrames;
        object.femMaterialFrameSourceIdentity = digestIdentity(frameDigest);
        result.frameMapDigest = hexDigest(frameDigest);
    }
    result.world.objects = {std::move(object)};
    return result;
}

double densityByMaterial(const WorldSource& source,
                         const std::uint32_t materialIndex) {
    require(materialIndex < source.materials.size(),
            "cell material index is outside source material array");
    return materialDensity(source.materials[materialIndex]);
}

std::string materialName(const WorldSource& source,
                         const std::uint32_t materialIndex) {
    require(materialIndex < source.materials.size(),
            "cell material index is outside source material array");
    return source.materials[materialIndex].name;
}

struct MaterialTotals {
    std::uint64_t tetrahedra = 0u;
    double volumeM3 = 0.0;
    double massKg = 0.0;
};

std::vector<MaterialTotals> calculateTotals(const CompiledWorld& world,
                                            const WorldSource& source) {
    std::vector<MaterialTotals> totals(source.materials.size());
    for (const NMTetrahedronGPU& tetrahedron : world.fem.tetrahedra) {
        const std::uint32_t materialIndex = tetrahedron.identity.x;
        require(materialIndex < totals.size(),
                "compiled tetrahedron material index is invalid");
        const double volume = tetrahedron.inverseRestRow0.w;
        require(std::isfinite(volume) && volume > 0.0,
                "compiled tetrahedron has nonpositive rest volume");
        MaterialTotals& total = totals[materialIndex];
        ++total.tetrahedra;
        total.volumeM3 += volume;
        total.massKg += volume * densityByMaterial(source, materialIndex);
    }
    return totals;
}

std::pair<Vec3, Vec3> nodeBounds(const CompiledWorld& world) {
    require(!world.fem.nodes.empty(), "compiled cardboard node array is empty");
    Vec3 minimum{INFINITY, INFINITY, INFINITY};
    Vec3 maximum{-INFINITY, -INFINITY, -INFINITY};
    for (const NMFEMNodeStateGPU& node : world.fem.nodes) {
        minimum.x = std::min(minimum.x, double(node.positionAndMass.x));
        minimum.y = std::min(minimum.y, double(node.positionAndMass.y));
        minimum.z = std::min(minimum.z, double(node.positionAndMass.z));
        maximum.x = std::max(maximum.x, double(node.positionAndMass.x));
        maximum.y = std::max(maximum.y, double(node.positionAndMass.y));
        maximum.z = std::max(maximum.z, double(node.positionAndMass.z));
    }
    return {minimum, maximum};
}

template <typename T>
id<MTLBuffer> sharedBuffer(id<MTLDevice> device,
                           const std::vector<T>& values,
                           const std::string& label) {
    require(!values.empty(), label + " buffer cannot be empty");
    id<MTLBuffer> result = [device newBufferWithBytes:values.data()
                                               length:values.size() * sizeof(T)
                                              options:MTLResourceStorageModeShared];
    require(result != nil, "failed to allocate " + label + " Metal buffer");
    return result;
}

void writeObj(const std::filesystem::path& path,
              const CompiledWorld& world,
              const std::vector<NMFEMNodeStateGPU>& nodes,
              const std::uint32_t environment) {
    const std::uint32_t nodeCount = world.dispatch.femNodeCount;
    require(nodes.size() >= static_cast<std::size_t>(environment + 1u) * nodeCount,
            "accepted FEM node snapshot is truncated");
    struct FaceRecord {
        std::array<std::uint32_t, 3> oriented{};
        std::uint32_t incidence = 0u;
    };
    std::map<std::array<std::uint32_t, 3>, FaceRecord> faces;
    for (const NMTetrahedronGPU& tetrahedron : world.fem.tetrahedra) {
        const std::array<std::uint32_t, 4> tet{
            tetrahedron.nodes.x, tetrahedron.nodes.y,
            tetrahedron.nodes.z, tetrahedron.nodes.w,
        };
        const std::array<std::array<std::uint32_t, 3>, 4> tetFaces{{
            {{tet[0], tet[1], tet[2]}},
            {{tet[0], tet[3], tet[1]}},
            {{tet[0], tet[2], tet[3]}},
            {{tet[1], tet[3], tet[2]}},
        }};
        for (const auto& oriented : tetFaces) {
            auto key = oriented;
            std::sort(key.begin(), key.end());
            FaceRecord& record = faces[key];
            if (record.incidence == 0u) record.oriented = oriented;
            ++record.incidence;
        }
    }
    std::ofstream output(path);
    require(output.good(), "cannot create OBJ: " + path.string());
    output << std::setprecision(9) << "# accepted Matter FEM state; environment="
           << environment << '\n';
    const std::size_t base = static_cast<std::size_t>(environment) * nodeCount;
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        const nm_float4 position = nodes[base + node].positionAndMass;
        output << "v " << position.x << ' ' << position.y << ' '
               << position.z << '\n';
    }
    for (const auto& [key, record] : faces) {
        (void)key;
        if (record.incidence != 1u) continue;
        output << "f " << record.oriented[0] + 1u << ' '
               << record.oriented[1] + 1u << ' '
               << record.oriented[2] + 1u << '\n';
    }
    require(output.good(), "failed while writing OBJ: " + path.string());
}

void writeInitialObj(const std::filesystem::path& path,
                     const CompiledWorld& world) {
    std::vector<NMFEMNodeStateGPU> nodes = world.fem.nodes;
    writeObj(path, world, nodes, 0u);
}

void writeMeshJson(const std::filesystem::path& path,
                   const Arguments& arguments,
                   const MeshSource& mesh) {
    const ObjectSource& object = mesh.world.objects.front();
    require(mesh.restNodes.size() == mesh.world.objects.front().femNodes.size(),
            "authored node export differs from source FEM node order");
    require(mesh.tetrahedra.size() == mesh.materialIndices.size() &&
                mesh.tetrahedra.size() == mesh.materialFrames.size(),
            "authored material/frame arrays differ from tetrahedron order");

    std::ofstream output(path);
    require(output.good(), "cannot create mesh audit JSON: " + path.string());
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"numi.cardboard.authored-mesh.v1\",\n"
           << "  \"units\": \"metres\",\n"
           << "  \"geometry_equation\": \"sinusoidal centerline; constant normal thickness; piecewise-linear sampled surfaces\",\n"
           << "  \"medium_present\": "
           << (arguments.withoutMedium ? "false" : "true") << ",\n"
           << "  \"material_densities_kg_m3\": [";
    for (std::size_t mi = 0; mi < mesh.world.materials.size(); ++mi) {
        if (mi) output << ',';
        output << materialDensity(mesh.world.materials[mi]);
    }
    output << "],\n  \"nodes_m\": [\n";
    for (std::size_t index = 0u; index < mesh.restNodes.size(); ++index) {
        const Vec3& node = mesh.restNodes[index];
        output << "    [" << node.x << ',' << node.y << ',' << node.z << ']'
               << (index + 1u == mesh.restNodes.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"tetrahedra\": [\n";
    for (std::size_t index = 0u; index < mesh.tetrahedra.size(); ++index) {
        const auto& nodes = mesh.tetrahedra[index].nodes;
        output << "    [" << nodes[0] << ',' << nodes[1] << ','
               << nodes[2] << ',' << nodes[3] << ']'
               << (index + 1u == mesh.tetrahedra.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"material_indices\": [\n";
    for (std::size_t index = 0u; index < mesh.materialIndices.size(); ++index)
        output << "    " << mesh.materialIndices[index]
               << (index + 1u == mesh.materialIndices.size() ? "\n" : ",\n");
    output << "  ],\n  \"material_frames_xyzw\": [\n";
    for (std::size_t index = 0u; index < mesh.materialFrames.size(); ++index) {
        const auto& frame = mesh.materialFrames[index];
        output << "    [" << frame[0] << ',' << frame[1] << ','
               << frame[2] << ',' << frame[3] << ']'
               << (index + 1u == mesh.materialFrames.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"fixed_nodes\": [\n";
    for (std::size_t index = 0u; index < object.femFixedNodes.size(); ++index)
        output << "    " << object.femFixedNodes[index]
               << (index + 1u == object.femFixedNodes.size() ? "\n" : ",\n");
    output << "  ],\n  \"left_grip_nodes\": [\n";
    for (std::size_t index = 0u; index < mesh.leftGripNodes.size(); ++index)
        output << "    " << mesh.leftGripNodes[index]
               << (index + 1u == mesh.leftGripNodes.size() ? "\n" : ",\n");
    output << "  ],\n  \"right_grip_nodes\": [\n";
    for (std::size_t index = 0u; index < mesh.rightGripNodes.size(); ++index)
        output << "    " << mesh.rightGripNodes[index]
               << (index + 1u == mesh.rightGripNodes.size() ? "\n" : ",\n");
    output << "  ]\n}\n";
    require(output.good(), "failed while writing authored mesh audit JSON");
}

void writePackageOrThrow(const CompileResult& compiled,
                         const std::filesystem::path& path) {
    std::string error;
    require(writePackage(compiled, path, &error),
            "cannot write compiled Matter package: " + error);
}

void writeManifest(const std::filesystem::path& path,
                   const Arguments& arguments,
                   const MeshSource& mesh,
                   const CompileResult& compiled,
                   const Digest& linerDigest,
                   const Digest& mediumDigest, const std::string& glueDigest) {
    const auto totals = calculateTotals(compiled.world, mesh.world);
    const auto [minimum, maximum] = nodeBounds(compiled.world);
    const double expectedMass = std::accumulate(
        totals.begin(), totals.end(), 0.0,
        [](const double sum, const MaterialTotals& total) {
            return sum + total.massKg;
        });
    double compiledNodalMass = 0.0;
    for (const NMFEMNodeStateGPU& node : compiled.world.fem.nodes)
        compiledNodalMass += node.positionAndMass.w;
    const double massRelativeError = std::abs(compiledNodalMass - expectedMass) /
        std::max(1.0e-20, expectedMass);
    require(massRelativeError < 5.0e-5,
            "compiled shared-node mass does not close against regional cell mass");

    std::ofstream output(path);
    require(output.good(), "cannot create manifest: " + path.string());
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"numi.cardboard.explicit-strip.v1\",\n"
           << "  \"owner\": \"Numi Matter FEM runtime\",\n"
           << "  \"preset\": \"" << jsonEscape(arguments.preset) << "\",\n"
           << "  \"geometry\": {\n"
           << "    \"source_dimensions\": \"" << (arguments.preset == "literature2009" ?
               "Cross-source recipe: Haj-Ali 2009 pitch and parametric glue zones; Popil 2017 separate sheet calipers and board caliper; coupon length/width numerical choices" :
               "Nagasawa 2013 nominal dimensions; sinusoidal approximation; CLI values below are authoritative") << "\",\n"
           << "    \"initial_condition\": \"assembled conditioned geometry with zero stress; corrugation forming, curing and moisture history are not simulated\",\n"
           << "    \"equation\": \"sinusoidal centerline with constant-normal-thickness sampled strip\",\n"
           << "    \"length_m\": " << arguments.lengthM << ",\n"
           << "    \"width_m\": " << arguments.widthM << ",\n"
           << "    \"total_height_m\": " << arguments.totalHeightM << ",\n"
           << "    \"pitch_m\": " << arguments.pitchM << ",\n"
           << "    \"liner_thickness_m\": " << arguments.linerThicknessM << ",\n"
           << "    \"medium_normal_thickness_m\": " << arguments.mediumThicknessM << ",\n"
           << "    \"medium_centerline_height_m\": "
           << (arguments.totalHeightM - 2.0 * arguments.linerThicknessM -
               arguments.mediumThicknessM - arguments.glueGapM - (arguments.upperGlueGapM > 0 ? arguments.upperGlueGapM : arguments.glueGapM)) << ",\n"
           << "    \"medium_present\": "
           << (arguments.withoutMedium ? "false" : "true") << ",\n"
           << "    \"glue_minimum_gap_m\": " << arguments.glueGapM << ",\n"
           << "    \"bond_width_m\": " << arguments.bondWidthM << ",\n"
           << "    \"upper_glue_gap_m\": " << (arguments.upperGlueGapM > 0 ? arguments.upperGlueGapM : arguments.glueGapM) << ",\n"
           << "    \"upper_bond_width_m\": " << (arguments.upperBondWidthM > 0 ? arguments.upperBondWidthM : arguments.bondWidthM) << ",\n"
           << "    \"contact_tie_assumption\": \""
           << (arguments.glueMaterial.empty() ? "legacy shared-node contact lines; no adhesive material" :
               "finite solid glue bridges; conforming bonded interfaces; no interface separation or cure law") << "\"\n"
           << "  },\n"
           << "  \"glue_footprints\": [\n";
    for (std::size_t i = 0; i < mesh.glueFootprints.size(); ++i) {
        const auto& footprint = mesh.glueFootprints[i];
        output << "    {\"upper\":" << (footprint.crest ? "true" : "false")
               << ",\"center_x_m\":" << footprint.centerX
               << ",\"requested_width_m\":" << footprint.requestedHorizontalWidth
               << ",\"actual_width_m\":" << footprint.actualHorizontalWidth
               << ",\"cross_section_area_m2\":" << footprint.crossSectionArea
               << "}" << (i + 1 == mesh.glueFootprints.size() ? "\n" : ",\n");
    }
    output << "  ],\n"
           << "  \"mesh\": {\n"
           << "    \"nodes\": " << compiled.world.fem.nodes.size() << ",\n"
           << "    \"tetrahedra\": " << compiled.world.fem.tetrahedra.size() << ",\n"
           << "    \"fixed_grip_nodes\": "
           << mesh.world.objects.front().femFixedNodes.size() << ",\n"
           << "    \"nx_per_pitch\": " << arguments.nxPerPitch << ",\n"
           << "    \"ny\": " << arguments.ny << ",\n"
           << "    \"thickness_slices\": " << arguments.thicknessSlices << ",\n"
           << "    \"free_nodes\": " << (compiled.world.fem.nodes.size() -
               mesh.world.objects.front().femFixedNodes.size()) << ",\n"
           << "    \"min_extent_m\": [" << minimum.x << ", " << minimum.y
           << ", " << minimum.z << "],\n"
           << "    \"max_extent_m\": [" << maximum.x << ", " << maximum.y
           << ", " << maximum.z << "],\n"
           << "    \"shared_node_total_mass_relative_error\": "
           << massRelativeError << "\n"
           << "  },\n"
           << "  \"materials\": {\n"
           << "    \"glue_path\": \"" << jsonEscape(arguments.glueMaterial.string()) << "\",\n"
           << "    \"glue_sha256\": \"" << glueDigest << "\",\n"
           << "    \"liner_path\": \"" << jsonEscape(arguments.linerMaterial.string()) << "\",\n"
           << "    \"liner_sha256\": \"" << hexDigest(linerDigest) << "\",\n"
           << "    \"medium_path\": \"" << jsonEscape(arguments.mediumMaterial.string()) << "\",\n"
           << "    \"medium_sha256\": \"" << hexDigest(mediumDigest) << "\",\n"
           << "    \"regional_map_sha256\": \"" << mesh.materialMapDigest << "\",\n"
           << "    \"frame_map_sha256\": \"" << mesh.frameMapDigest << "\",\n"
           << "    \"frame_source\": \"analytic local flute tangent; local x=tangent, y=width, z=in-plane normal\",\n"
           << "    \"cells\": [\n";
    for (std::size_t index = 0u; index < totals.size(); ++index) {
        const MaterialTotals& total = totals[index];
        output << "      {\"index\": " << index
               << ", \"name\": \"" << jsonEscape(materialName(mesh.world,
                                                                        static_cast<std::uint32_t>(index)))
               << "\", \"tetrahedra\": " << total.tetrahedra
               << ", \"volume_m3\": " << total.volumeM3
               << ", \"mass_kg\": " << total.massKg << "}"
               << (index + 1u == totals.size() ? "\n" : ",\n");
    }
    output << "    ]\n  },\n"
           << "  \"solver\": {\n"
           << "    \"backend\": \"implicit nonlinear Matter FEM on Apple Metal\",\n"
           << "    \"deformable_self_contact\": " << (mesh.world.objects.front().deformableSelfContact ? "true" : "false") << ",\n"
           << "    \"contact_slop_m\": " << mesh.world.contactSlop << ",\n"
           << "    \"dt_s\": " << arguments.timestepSeconds << ",\n"
           << "    \"bend_angle_deg\": " << arguments.bendAngleDegrees << ",\n"
           << "    \"steps\": " << arguments.steps << ",\n"
           << "    \"hold_steps\": " << arguments.holdSteps << ",\n"
           << "    \"unload_steps\": " << arguments.unloadSteps << ",\n"
           << "    \"relax_steps\": " << arguments.relaxSteps << ",\n"
           << "    \"local_material_newton_iterations\": " << arguments.localMaterialIterations << ",\n"
           << "    \"newton_iteration_budget\": "
           << compiled.world.mixedSolver.nonlinearIterations.x << ",\n"
           << "    \"fgmres_restart\": "
           << compiled.world.mixedSolver.nonlinearIterations.y << ",\n"
           << "    \"fgmres_iteration_budget\": "
           << compiled.world.mixedSolver.nonlinearIterations.z << ",\n"
           << "    \"line_search_steps\": "
           << compiled.world.mixedSolver.nonlinearIterations.w << ",\n"
           << "    \"relative_residual_tolerance\": "
           << compiled.world.mixedSolver.residualTolerances.x << ",\n"
           << "    \"volume_tolerance\": "
           << compiled.world.mixedSolver.residualTolerances.y << ",\n"
           << "    \"pressure_tolerance\": "
           << compiled.world.mixedSolver.residualTolerances.z << ",\n"
           << "    \"transport_tolerance\": "
           << compiled.world.mixedSolver.residualTolerances.w << ",\n"
           << "    \"maximum_rate_exponent\": 0,\n"
           << "    \"runtime_execution\": "
           << (arguments.compileOnly ? "false" : "true") << "\n"
           << "  },\n"
           << "  \"status_diagnostics_semantics\": \"CSV retains NMMatterStatusGPU.diagnostics x/y/z/w verbatim. Successful finalization writes minimum J, maximum stress, scheduler numerical.y, and maximum contact speed; scheduler numerical.y is not populated as a successful KKT residual in this probe path and may remain zero. Failure statuses replace the vector with the originating failure-site float4 payload (for example, code 10 uses relative residual, relative correction, volume residual, and pressure residual). Independent min_J and max_J are recomputed from accepted FEM positions; status_diagnostic_z must not be interpreted as the successful-state KKT residual.\",\n"
           << "  \"compiled_world_fingerprint\": "
           << compiled.world.fingerprint << ",\n"
           << "  \"evidence_boundary\": \"native software mechanics instrument only; analytic geometry approximation and published material inputs do not establish physical validation\"\n"
           << "}\n";
    require(output.good(), "failed while writing manifest: " + path.string());
}

void prepareOutput(const std::filesystem::path& output) {
    std::error_code error;
    if (std::filesystem::exists(output, error)) {
        require(!error && std::filesystem::is_directory(output),
                "--output exists and is not a directory");
        require(std::filesystem::directory_iterator(output) ==
                    std::filesystem::directory_iterator(),
                "--output must be a new or empty directory; existing evidence is preserved");
    } else {
        require(std::filesystem::create_directories(output, error) && !error,
                "cannot create output directory: " + output.string());
    }
}

std::string stepLabel(const std::uint32_t step) {
    std::ostringstream result;
    result << std::setw(6) << std::setfill('0') << step;
    return result.str();
}

struct LoadPoint { double angleDegrees; const char* phase; };

std::uint32_t totalSteps(const Arguments& a) {
    return a.steps + a.holdSteps + a.unloadSteps + a.relaxSteps;
}

LoadPoint loadingPoint(const Arguments& a, std::uint32_t step) {
    require(step < totalSteps(a), "loading step outside protocol");
    if (step < a.steps)
        return {a.bendAngleDegrees * static_cast<double>(step + 1u) / a.steps, "loading"};
    step -= a.steps;
    if (step < a.holdSteps) return {a.bendAngleDegrees, "hold"};
    step -= a.holdSteps;
    if (step < a.unloadSteps)
        return {a.bendAngleDegrees * (1.0 - static_cast<double>(step + 1u) / a.unloadSteps), "unloading"};
    return {0.0, "relaxation"};
}

struct Metrics {
    double minJ = INFINITY;
    double maxJ = -INFINITY;
    double maxDisplacementM = 0.0;
    double maxFreeDisplacementM = 0.0;
    double maxFreeSpeedMps = 0.0;
    double kineticEnergyJ = 0.0;
    double maxStateChange = 0.0;
};

std::array<double, 9> inverseRestMatrix(const NMTetrahedronGPU& tetrahedron) {
    return {
        tetrahedron.inverseRestRow0.x,
        tetrahedron.inverseRestRow0.y,
        tetrahedron.inverseRestRow0.z,
        tetrahedron.inverseRestRow1.x,
        tetrahedron.inverseRestRow1.y,
        tetrahedron.inverseRestRow1.z,
        tetrahedron.inverseRestRow2.x,
        tetrahedron.inverseRestRow2.y,
        tetrahedron.inverseRestRow2.z,
    };
}

Metrics stateMetrics(const CompiledWorld& world,
                     const RuntimeStateSnapshot& snapshot,
                     const std::uint32_t environment) {
    Metrics metrics;
    const std::uint32_t nodeCount = world.dispatch.femNodeCount;
    const std::size_t nodeBase = static_cast<std::size_t>(environment) * nodeCount;
    require(snapshot.femNodes.size() >= nodeBase + nodeCount,
            "accepted FEM node state snapshot is truncated");
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        const nm_float4 initial = world.fem.nodes[node].positionAndMass;
        const nm_float4 current = snapshot.femNodes[nodeBase + node].positionAndMass;
        const double dx = current.x - initial.x;
        const double dy = current.y - initial.y;
        const double dz = current.z - initial.z;
        const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        metrics.maxDisplacementM = std::max(metrics.maxDisplacementM, distance);
        const auto velocity = snapshot.femNodes[nodeBase + node].velocityAndInverseMass;
        const double speed2 = velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z;
        metrics.kineticEnergyJ += 0.5 * current.w * speed2;
        if (world.fem.nodes[node].restAndFixed.w == 0.0f) {
            metrics.maxFreeDisplacementM = std::max(metrics.maxFreeDisplacementM, distance);
            metrics.maxFreeSpeedMps = std::max(metrics.maxFreeSpeedMps, std::sqrt(speed2));
        }
    }

    for (std::size_t ti = 0; ti < world.fem.tetrahedra.size(); ++ti) {
        const NMTetrahedronGPU& tetrahedron = world.fem.tetrahedra[ti];
        const auto& material = world.materials.at(tetrahedron.identity.x);
        const auto stateBase = (environment * world.dispatch.tetrahedronCount + ti) * snapshot.materialStateStride;
        for (std::uint32_t state = 0; state < material.stateCount; ++state) {
            const double value = snapshot.femMaterialState.at(stateBase + state);
            require(std::isfinite(value), "nonfinite accepted material state");
            metrics.maxStateChange = std::max(metrics.maxStateChange,
                std::abs(value - world.stateInitials.at(material.stateInitialOffset + state)));
        }
        const std::array<std::uint32_t, 4> indices{
            tetrahedron.nodes.x, tetrahedron.nodes.y,
            tetrahedron.nodes.z, tetrahedron.nodes.w,
        };
        const nm_float4 p0 = snapshot.femNodes[nodeBase + indices[0]].positionAndMass;
        const nm_float4 p1 = snapshot.femNodes[nodeBase + indices[1]].positionAndMass;
        const nm_float4 p2 = snapshot.femNodes[nodeBase + indices[2]].positionAndMass;
        const nm_float4 p3 = snapshot.femNodes[nodeBase + indices[3]].positionAndMass;
        const std::array<double, 9> edge{
            p1.x - p0.x, p2.x - p0.x, p3.x - p0.x,
            p1.y - p0.y, p2.y - p0.y, p3.y - p0.y,
            p1.z - p0.z, p2.z - p0.z, p3.z - p0.z,
        };
        const auto inverseRest = inverseRestMatrix(tetrahedron);
        std::array<double, 9> deformation{};
        for (std::uint32_t row = 0u; row < 3u; ++row)
            for (std::uint32_t column = 0u; column < 3u; ++column)
                for (std::uint32_t k = 0u; k < 3u; ++k)
                    deformation[3u * row + column] +=
                        edge[3u * row + k] * inverseRest[3u * k + column];
        const double j =
            deformation[0] * (deformation[4] * deformation[8] -
                               deformation[5] * deformation[7]) -
            deformation[1] * (deformation[3] * deformation[8] -
                               deformation[5] * deformation[6]) +
            deformation[2] * (deformation[3] * deformation[7] -
                               deformation[4] * deformation[6]);
        require(std::isfinite(j), "nonfinite determinant in accepted FEM state");
        metrics.minJ = std::min(metrics.minJ, j);
        metrics.maxJ = std::max(metrics.maxJ, j);
    }
    return metrics;
}

Vec3 targetPosition(const Vec3& rest,
                    const bool rightGrip,
                    const double angleRadians,
                    const Arguments& arguments) {
    if (!rightGrip || angleRadians == 0.0) return rest;
    const Vec3 pivot{
        arguments.lengthM,
        0.5 * arguments.widthM,
        0.5 * arguments.totalHeightM,
    };
    const double rx = rest.x - pivot.x;
    const double rz = rest.z - pivot.z;
    const double cosine = std::cos(angleRadians);
    const double sine = std::sin(angleRadians);
    return {
        pivot.x + cosine * rx + sine * rz,
        rest.y,
        pivot.z - sine * rx + cosine * rz,
    };
}

void fillTargets(const Arguments& arguments,
                 const MeshSource& mesh,
                 const std::uint32_t step,
                 std::vector<nm_float4>& targets) {
    const std::uint32_t nodeCount =
        static_cast<std::uint32_t>(mesh.restNodes.size());
    targets.assign(static_cast<std::size_t>(kEnvironmentCount) * nodeCount,
                   nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
    const double angle = loadingPoint(arguments, step).angleDegrees * kPi / 180.0;
    std::vector<bool> left(nodeCount, false);
    std::vector<bool> right(nodeCount, false);
    for (const std::uint32_t node : mesh.leftGripNodes) left[node] = true;
    for (const std::uint32_t node : mesh.rightGripNodes) right[node] = true;
    for (std::uint32_t environment = 0u;
         environment < kEnvironmentCount; ++environment) {
        const double environmentAngle = environment == 0u ? angle : 0.0;
        const std::size_t base = static_cast<std::size_t>(environment) * nodeCount;
        for (std::uint32_t node = 0u; node < nodeCount; ++node) {
            if (!left[node] && !right[node]) continue;
            const Vec3 target = targetPosition(
                mesh.restNodes[node], right[node], environmentAngle, arguments);
            targets[base + node] = {
                static_cast<float>(target.x),
                static_cast<float>(target.y),
                static_cast<float>(target.z),
                1.0f,
            };
        }
    }
}

struct RunState {
    RuntimeStateSnapshot snapshot;
    std::vector<nm_float4> reactions;
};

class NativeRun {
public:
    explicit NativeRun(const CompiledWorld& world)
        : world_(world) {
        device_ = MTLCreateSystemDefaultDevice();
        require(device_ != nil, "Apple Metal device is unavailable");
        const std::string deviceName = [[device_ name] UTF8String];
        require(deviceName.find("Apple") != std::string::npos &&
                    deviceName.find("Paravirtual") == std::string::npos,
                "a physical Apple Metal device is required for the native run");
        queue_ = [device_ newCommandQueue];
        require(queue_ != nil, "cannot create Metal command queue");

        RuntimeConfiguration configuration;
        configuration.metallib = NUMI_CARDBOARD_METALLIB;
        configuration.environmentCount = kEnvironmentCount;
        configuration.captureEvents = true;
        configuration.captureDiagnostics = true;
        configuration.adaptiveTransfer = false;
        const RuntimeDiagnostics initialized = runtime_.initialize(
            world_, configuration);
        require(initialized.encoded,
                "Matter runtime initialization failed: " + initialized.message);
        nodeCount_ = world_.dispatch.femNodeCount;
        const std::vector<MRMetalWorldStatusGPU> initialStatuses =
            statusRecords();
        environmentStatuses_ = sharedBuffer(device_, initialStatuses,
                                            "environment status");
        const std::vector<nm_float4> initialTargets(
            static_cast<std::size_t>(kEnvironmentCount) * nodeCount_,
            nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
        targets_ = sharedBuffer(device_, initialTargets,
                                "FEM kinematic target");
        reactions_ = [device_ newBufferWithLength:
            static_cast<std::size_t>(kEnvironmentCount) * nodeCount_ *
                sizeof(nm_float4)
            options:MTLResourceStorageModeShared];
        require(reactions_ != nil,
                "cannot allocate FEM reaction readback buffer");
    }

    RunState step(const std::uint32_t index,
                  const std::vector<nm_float4>& targets) {
        require(targets.size() ==
                    static_cast<std::size_t>(kEnvironmentCount) * nodeCount_,
                "kinematic target count differs from compiled FEM state");
        std::memcpy(targets_.contents, targets.data(),
                    targets.size() * sizeof(nm_float4));
        auto* statusValues = static_cast<MRMetalWorldStatusGPU*>(
            environmentStatuses_.contents);
        for (std::uint32_t environment = 0u;
             environment < kEnvironmentCount; ++environment) {
            statusValues[environment] = {};
            statusValues[environment].environment = environment;
        }

        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        require(command != nil, "cannot create Metal command buffer");
        EncodeRequest request;
        request.commandBuffer = (__bridge void*)command;
        request.environmentStatuses = (__bridge void*)environmentStatuses_;
        request.femKinematicTargets = (__bridge void*)targets_;
        request.femKinematicTargetCount = kEnvironmentCount * nodeCount_;
        request.controlStep = index;
        request.physicsSubsteps = 1u;
        request.timestepSeconds = runtime_.timestepSeconds();
        request.runAdaptiveTransfer = false;
        request.phase = EncodePhase::preDynamics;
        const RuntimeDiagnostics pre = runtime_.encode(request);
        require(pre.encoded, "Matter preDynamics encode failed: " + pre.message);

        id<MTLBuffer> reactionSource =
            (__bridge id<MTLBuffer>)runtime_.femConstraintReactionBuffer();
        require(reactionSource != nil,
                "Matter FEM constraint reaction buffer is unavailable");
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        require(blit != nil, "cannot create reaction readback blit encoder");
        [blit copyFromBuffer:reactionSource
                sourceOffset:0u
                    toBuffer:reactions_
           destinationOffset:0u
                        size:static_cast<std::size_t>(kEnvironmentCount) *
                            nodeCount_ * sizeof(nm_float4)];
        [blit endEncoding];

        request.phase = EncodePhase::postCommit;
        const RuntimeDiagnostics post = runtime_.encode(request);
        require(post.encoded, "Matter postCommit encode failed: " + post.message);
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
                "Matter Metal command buffer failed");

        RunState result;
        result.snapshot = runtime_.snapshot();
        require(result.snapshot.available, result.snapshot.message);
        const auto* reactionData =
            static_cast<const nm_float4*>(reactions_.contents);
        result.reactions.assign(
            reactionData,
            reactionData + static_cast<std::size_t>(kEnvironmentCount) *
                nodeCount_);
        return result;
    }

    [[nodiscard]] const Runtime& runtime() const { return runtime_; }

private:
    std::vector<MRMetalWorldStatusGPU> statusRecords() const {
        std::vector<MRMetalWorldStatusGPU> result(kEnvironmentCount);
        for (std::uint32_t environment = 0u;
             environment < kEnvironmentCount; ++environment)
            result[environment].environment = environment;
        return result;
    }

    const CompiledWorld& world_;
    Runtime runtime_;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLBuffer> environmentStatuses_ = nil;
    id<MTLBuffer> targets_ = nil;
    id<MTLBuffer> reactions_ = nil;
    std::uint32_t nodeCount_ = 0u;
};

struct ReactionMetrics {
    Vec3 force{};
    double momentAboutY = 0.0;
};

ReactionMetrics reactionMetrics(const std::uint32_t environment,
                                const std::vector<nm_float4>& reactions,
                                const MeshSource& mesh,
                                const Arguments& arguments,
                                const double angleRadians) {
    const std::uint32_t nodeCount =
        static_cast<std::uint32_t>(mesh.restNodes.size());
    const std::size_t base = static_cast<std::size_t>(environment) * nodeCount;
    const Vec3 pivot{arguments.lengthM, 0.5 * arguments.widthM,
                     0.5 * arguments.totalHeightM};
    ReactionMetrics result;
    for (const std::uint32_t node : mesh.rightGripNodes) {
        const nm_float4 value = reactions.at(base + node);
        result.force.x += value.x;
        result.force.y += value.y;
        result.force.z += value.z;
        const Vec3 point = targetPosition(mesh.restNodes[node], true,
                                          angleRadians, arguments);
        const double rx = point.x - pivot.x;
        const double rz = point.z - pivot.z;
        result.momentAboutY += rz * value.x - rx * value.z;
    }
    return result;
}

void writeHeader(std::ofstream& output) {
    output << "step,time_s,environment,arm,target_angle_deg,status_code,"
              "object_index,failing_index,completed_microsteps,fgmres_iterations,"
              "status_diagnostic_x,status_diagnostic_y,status_diagnostic_z,"
              "status_diagnostic_w,min_J,max_J,"
              "max_node_displacement_m,reaction_x_N,reaction_y_N,reaction_z_N,"
              "reaction_moment_y_Nm,phase,max_free_displacement_m,max_free_speed_m_s,"
              "kinetic_energy_J,max_material_state_change,certificate_residual,"
              "certificate_correction,certificate_volume,certificate_pressure,certificate_raw_accepted_flag,step_accepted\n";
}

void writeObservation(std::ofstream& output,
                      const std::uint32_t step,
                      const double timestep,
                      const std::uint32_t environment,
                      const double targetAngleDegrees,
                      const NMMatterStatusGPU& status,
                      const Metrics& metrics,
                      const ReactionMetrics& reaction,
                      const char* phase, const NMSolverCertificateGPU& certificate, const bool stepAccepted) {
    output << step << ',' << std::setprecision(17)
           << static_cast<double>(step + 1u) * timestep << ','
           << environment << ','
           << (environment == 0u ? "bent" : "held_reference") << ','
           << targetAngleDegrees << ',' << status.code << ','
           << status.objectIndex << ',' << status.failingIndex << ','
           << status.completedMicrosteps << ',' << status.fgmresIterations << ','
           << status.diagnostics.x << ',' << status.diagnostics.y << ','
           << status.diagnostics.z << ',' << status.diagnostics.w << ','
           << metrics.minJ << ',' << metrics.maxJ
           << ',' << metrics.maxDisplacementM << ','
           << reaction.force.x << ',' << reaction.force.y << ','
           << reaction.force.z << ',' << reaction.momentAboutY << ',' << phase << ','
           << metrics.maxFreeDisplacementM << ',' << metrics.maxFreeSpeedMps << ','
           << metrics.kineticEnergyJ << ',' << metrics.maxStateChange << ','
           << certificate.nonlinear.x << ',' << certificate.nonlinear.y << ','
           << certificate.nonlinear.z << ',' << certificate.nonlinear.w << ','
           << certificate.validity.w << ',' << (stepAccepted ? 1 : 0) << '\n';
    output.flush();
    require(output.good(), "failed while writing per-step CSV observations");
}

void writeMaterialState(const std::filesystem::path& path,
                        const CompiledWorld& world, const WorldSource& source,
                        const RuntimeStateSnapshot& snapshot,
                        const std::uint32_t environment) {
    std::ofstream output(path);
    require(output.good(), "cannot create material-state export");
    output << std::setprecision(17)
           << "{\n  \"schema\": \"numi.cardboard.accepted-material-state.v1\",\n"
           << "  \"environment\": " << environment << ",\n"
           << "  \"status_code\": " << snapshot.statuses.at(environment).code << ",\n"
           << "  \"solver_certificate_raw_accepted_flag\": " << snapshot.solverCertificates.at(environment).validity.w << ",\n"
           << "  \"note\": \"Accepted material history. A failed step retains rolled-back state; incremental multipliers are not permanent-strain observables.\",\n"
           << "  \"materials\": [";
    for (std::size_t mi = 0; mi < source.materials.size(); ++mi) {
        if (mi) output << ',';
        const auto& material = source.materials[mi];
        output << "{\"index\":" << mi << ",\"name\":\"" << jsonEscape(material.name)
               << "\",\"state_names\":[";
        for (std::size_t si = 0; si < material.internalState.size(); ++si) {
            if (si) output << ',';
            output << '\"' << jsonEscape(material.internalState[si].name) << '\"';
        }
        output << "]}";
    }
    output << "],\n  \"tetrahedra\": [\n";
    for (std::size_t ti = 0; ti < world.fem.tetrahedra.size(); ++ti) {
        const auto& tet = world.fem.tetrahedra[ti];
        const auto& material = world.materials.at(tet.identity.x);
        const auto base = (environment * world.dispatch.tetrahedronCount + ti) * snapshot.materialStateStride;
        if (ti) output << ",\n";
        output << "    {\"index\":" << ti << ",\"material_index\":" << tet.identity.x << ",\"state\":[";
        for (std::uint32_t si = 0; si < material.stateCount; ++si) {
            if (si) output << ',';
            const auto value = snapshot.femMaterialState.at(base + si);
            require(std::isfinite(value), "nonfinite accepted material history");
            output << value;
        }
        output << "]}";
    }
    output << "\n  ]\n}\n";
    require(output.good(), "failed to export material history");
}

void writeSummaryJson(const std::filesystem::path& path,
                      const std::string& status,
                      const MeshSource& mesh,
                      const CompileResult& compiled,
                      const std::uint32_t acceptedSteps,
                      const std::string& failure = {}) {
    std::ofstream output(path);
    require(output.good(), "cannot create run summary: " + path.string());
    output << "{\n"
           << "  \"schema\": \"numi.cardboard.probe-result.v1\",\n"
           << "  \"status\": \"" << jsonEscape(status) << "\",\n"
           << "  \"accepted_steps\": " << acceptedSteps << ",\n"
           << "  \"compiled_world_fingerprint\": "
           << compiled.world.fingerprint << ",\n"
           << "  \"nodes\": " << compiled.world.fem.nodes.size() << ",\n"
           << "  \"tetrahedra\": " << compiled.world.fem.tetrahedra.size() << ",\n"
           << "  \"material_map_sha256\": \""
           << mesh.materialMapDigest << "\",\n"
           << "  \"frame_map_sha256\": \""
           << mesh.frameMapDigest << "\",\n"
           << "  \"failure\": \"" << jsonEscape(failure) << "\",\n"
           << "  \"physical_validation\": false\n"
           << "}\n";
    require(output.good(), "failed while writing run summary");
}

void printResult(const std::string& status,
                 const CompileResult& compiled,
                 const std::uint32_t acceptedSteps,
                 const std::string& failure = {}) {
    std::cout << "{\"schema\":\"numi.cardboard.probe-result.v1\","
              << "\"status\":\"" << jsonEscape(status) << "\","
              << "\"accepted_steps\":" << acceptedSteps << ','
              << "\"compiled_world_fingerprint\":"
              << compiled.world.fingerprint << ','
              << "\"nodes\":" << compiled.world.fem.nodes.size() << ','
              << "\"tetrahedra\":" << compiled.world.fem.tetrahedra.size() << ','
              << "\"failure\":\"" << jsonEscape(failure) << "\","
              << "\"physical_validation\":false}\n";
}

int execute(const Arguments& arguments) {
    const Digest linerDigest = sha256File(arguments.linerMaterial);
    const Digest mediumDigest = sha256File(arguments.mediumMaterial);
    MaterialProgram liner = readMaterial(arguments.linerMaterial, "liner");
    MaterialProgram medium = readMaterial(arguments.mediumMaterial, "medium");
    std::optional<MaterialProgram> glue;
    std::string glueDigest;
    if (!arguments.glueMaterial.empty()) {
        glue = readMaterial(arguments.glueMaterial, "glue");
        glueDigest = hexDigest(sha256File(arguments.glueMaterial));
    }
    MeshSource mesh = buildMesh(
        arguments, liner, medium, hexDigest(linerDigest), hexDigest(mediumDigest),
        glue ? &*glue : nullptr, glueDigest);

    CompileResult compiled = compileWorld(
        mesh.world, {.maximumRateExponent = 0u, .emitSpecializedMetal = false,
                     .localMaterialNewtonIterations = arguments.localMaterialIterations});
    require(compiled.succeeded(),
            "Matter cardboard source compile failed: " +
                diagnosticText(compiled.diagnostics));

    prepareOutput(arguments.output);
    writePackageOrThrow(compiled, arguments.output / "compiled.nmatterpack");
    writeInitialObj(arguments.output / "initial.obj", compiled.world);
    writeMeshJson(arguments.output / "mesh.json", arguments, mesh);
    writeManifest(arguments.output / "manifest.json", arguments, mesh,
                  compiled, linerDigest, mediumDigest, glueDigest);
    if (arguments.compileOnly) {
        writeSummaryJson(arguments.output / "result.json", "compile_only",
                         mesh, compiled, 0u);
        printResult("compile_only", compiled, 0u);
        return 0;
    }

    NativeRun run(compiled.world);
    std::ofstream observations(arguments.output / "observations.csv");
    require(observations.good(), "cannot create observations.csv");
    writeHeader(observations);
    bool failed = false;
    std::string failure;
    std::uint32_t acceptedSteps = 0u;
    for (std::uint32_t step = 0u; step < totalSteps(arguments); ++step) {
        std::vector<nm_float4> targets;
        fillTargets(arguments, mesh, step, targets);
        RunState current;
        try {
            current = run.step(step, targets);
        } catch (const std::exception& error) {
            failed = true;
            failure = "step " + std::to_string(step) + ": " + error.what();
            std::ofstream diagnostic(arguments.output / "failure.txt");
            diagnostic << failure << '\n';
            break;
        }
        require(current.snapshot.statuses.size() >= kEnvironmentCount,
                "Matter status readback is truncated");
        const LoadPoint load = loadingPoint(arguments, step);
        const double targetAngle = load.angleDegrees;
        const double targetRadians = targetAngle * kPi / 180.0;
        bool stepFailed = false;
        for (std::uint32_t environment = 0u;
             environment < kEnvironmentCount; ++environment) {
            const Metrics metrics = stateMetrics(
                compiled.world, current.snapshot, environment);
            const double environmentAngle = environment == 0u
                ? targetRadians : 0.0;
            const ReactionMetrics reaction = reactionMetrics(
                environment, current.reactions, mesh, arguments,
                environmentAngle);
            const NMMatterStatusGPU& status =
                current.snapshot.statuses[environment];
            const auto& certificate = current.snapshot.solverCertificates.at(environment);
            const bool accepted = status.code == NM_STATUS_SUCCESS && certificate.validity.w > 0.5f &&
                std::isfinite(certificate.nonlinear.x) && std::isfinite(certificate.nonlinear.y) &&
                std::isfinite(certificate.nonlinear.z) && std::isfinite(certificate.nonlinear.w) &&
                certificate.nonlinear.x <= compiled.world.mixedSolver.residualTolerances.x &&
                certificate.nonlinear.z <= compiled.world.mixedSolver.residualTolerances.y &&
                certificate.nonlinear.w <= compiled.world.mixedSolver.residualTolerances.z;
            writeObservation(observations, step, run.runtime().timestepSeconds(),
                             environment, environment == 0u ? targetAngle : 0.0,
                             status, metrics, reaction, load.phase,
                             certificate, accepted);
            const std::string arm = environment == 0u ? "bent" : "held_reference";
            writeObj(arguments.output /
                         ("accepted_step_" + stepLabel(step + 1u) + "_" +
                          arm + ".obj"),
                     compiled.world, current.snapshot.femNodes, environment);
            if (step + 1u == totalSteps(arguments) || !accepted ||
                std::string(loadingPoint(arguments, step + 1u).phase) != load.phase)
                writeMaterialState(arguments.output / ("material_state_" + stepLabel(step + 1u) + "_" + arm + ".json"),
                    compiled.world, mesh.world, current.snapshot, environment);
            stepFailed = stepFailed || !accepted;
        }
        if (stepFailed) {
            failed = true;
            failure = "Matter rejected step " + std::to_string(step) +
                "; status, accepted state OBJ, and CSV observations were retained";
            break;
        }
        ++acceptedSteps;
    }

    const std::string resultStatus = failed ? "failed" : "completed";
    writeSummaryJson(arguments.output / "result.json", resultStatus,
                     mesh, compiled, acceptedSteps, failure);
    printResult(resultStatus, compiled, acceptedSteps, failure);
    return failed ? 2 : 0;
}

} // namespace

int main(const int argc, const char* argv[]) {
    @autoreleasepool {
        Arguments arguments;
        try {
            arguments = parseArguments(argc, argv);
            if (arguments.help) {
                printUsage();
                return 0;
            }
            return execute(arguments);
        } catch (const std::exception& error) {
            const std::string message = error.what();
            std::cerr << "numi-matter-cardboard-probe=failed reason=\""
                      << message << "\"\n";
            if (!arguments.output.empty() &&
                std::filesystem::is_directory(arguments.output)) {
                std::ofstream failure(arguments.output / "failure.txt");
                if (failure.good()) failure << message << '\n';
            }
            return 1;
        }
    }
}

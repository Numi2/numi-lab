#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <CoreFoundation/CoreFoundation.h>
#include <mach/mach.h>

#include "cardboard_glue_mesh.hpp"
#include "cardboard_box_blank.hpp"
#include "cardboard_box_assembly_controller.hpp"
#include "cardboard_shipping_protocol.hpp"
#include "cardboard_shipping_driver.hpp"
#include "cardboard_shipping_memory.hpp"
#include "cardboard_shipping_snapshot.hpp"
#include "cardboard_reaction_observables.hpp"
#include "cardboard_shipping_closure_mesh.hpp"
#include "cardboard_shipping_crush.hpp"
#include "cardboard_shipping_capacity.hpp"
#include "cardboard_pin_adhesion_fixture.hpp"
#include "cardboard_starch_interface_mesh.hpp"
#include "cardboard_retained_matrix_damage_input.hpp"
#include "cardboard_paper_material_selection.hpp"
#include "cardboard_paper_projection_scaling.hpp"
#include "cardboard_tooling.hpp"
#ifdef NUMI_CARDBOARD_METAL_WORLD
#include "cardboard_shipping_rigid_driver.hpp"
#endif
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
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
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
    std::filesystem::path metallib;
    std::filesystem::path linerMaterial;
    std::filesystem::path mediumMaterial;
    bool linerMaterialExplicit = false;
    bool mediumMaterialExplicit = false;
    std::filesystem::path glueMaterial;
    double glueGapM = 0.0;
    double bondWidthM = 0.0;
    double upperGlueGapM = 0.0;
    double upperBondWidthM = 0.0;
    bool disableSelfContact = false;
    bool crease = false;
    bool prescribedToolPredictor = false;
    double prescribedToolBarrierStiffnessScale = 1.0;
    bool boxBlank = false;
    bool boxAssemblyNative = false;
    bool boxFixtureJoint = false;
    bool pinAdhesion = false;
    double pinMaximumOpeningM = 0.002;
    double pinWithdrawalDistanceM = 0.001;
    bool shippingBox = false;
    bool shippingPreflightOnly = false;
    bool compactImmutableSurfaces = false;
    std::uint32_t deformableContactCapacity = 0u;
    bool deformableContactCapacityProvided = false;
    bool shippingEngineeringAssumptions = false;
    bool shippingAssumedAllowanceProvided = false;
    std::uint32_t environmentCount = kEnvironmentCount;
    std::filesystem::path shippingCouponEvidence;
    std::filesystem::path starchInterfaceParameters;
    std::filesystem::path shippingHmaParameters;
    std::filesystem::path linerDamageParameters;
    std::filesystem::path mediumDamageParameters;
    std::filesystem::path shippingMeasurementFile;
    std::filesystem::path shippingRoofMeasurementFile;
    std::string shippingRoofMeasurementFileSha256;
    std::string shippingRoofMeasurementCheckpointPath;
    std::optional<numi::cardboard::ShippingOuterRoofMeasurement>
        shippingRoofMeasurementInput;
    std::string shippingRoofMeasurementLoadFailure;
    std::string shippingBlankGeometrySha256;
    std::array<double, 3> shippingImpactorHalfExtentsM{};
    std::string shippingImpactorGeometrySource = "authored_1.2x_nominal_outer_footprint";
    bool shippingImpactorHalfExtentsProvided = false;
    bool shippingImpactorSourceMeasurementValidated = false;
    bool shippingImpactorCompiledProxyMatchesMeasurement = false;
    bool shippingImpactorCurrentRoofMatchesCompiledProxy = false;
    std::uint32_t shippingImpactorProxyIndex = NM_INVALID_INDEX;
    std::filesystem::path shippingCapacityQualificationFile;
    std::string shippingCapacityQualificationSha256;
    std::optional<numi::cardboard::ShippingCapacityQualification>
        shippingCapacityQualification;
    std::string shippingCapacityQualificationLoadError;
    std::string shippingCouponEvidenceSha256;
    std::string starchInterfaceParametersSha256;
    std::string shippingHmaParametersSha256;
    std::string linerDamageParametersSha256;
    std::string mediumDamageParametersSha256;
    std::optional<numi::cardboard::retained_damage::ExplicitDamageSet>
        linerDamageSet;
    std::optional<numi::cardboard::retained_damage::ExplicitDamageSet>
        mediumDamageSet;
    bool shippingCouponDamageLawQualified = false;
    bool scalePaperProjectionResiduals = false;
    bool safeguardedPaperInitializer = false;
    bool compensatedPaperArithmetic = false;
    bool accuratePaperProjection = false;
    bool compensatedImplicitProjection = false;
    std::array<double, 13> linerProjectionScales{};
    std::array<double, 13> mediumProjectionScales{};
    std::array<double, 3> shippingInsideM{0.300, 0.200, 0.200};
    std::array<double, 3> shippingAssumedAllowanceM{};
    std::array<bool, 3> shippingAllowanceSet{};
    std::string shippingRunThroughStage = "complete";
    bool shippingStageOptionProvided = false;
    std::uint32_t shippingSettleSteps = 64u;
    bool shippingSettleStepsProvided = false;
    double shippingImpactTimestepSeconds = 0.0005;
    bool shippingImpactTimestepProvided = false;
    bool shippingImpactTimestepRefinement = false;
    double shippingGravityTransferTimestepSeconds = 0.0005;
    bool shippingGravityTransferTimestepProvided = false;
    double shippingScoreDepthRatio = 0.25;
    bool shippingScoreDepthRatioProvided = false;
    double shippingScoreToolSpeedMPerS = 50.0e-3;
    bool shippingScoreToolSpeedProvided = false;
    double boxFixtureGapM = 0.00005;
    std::uint32_t boxFixtureSteps = 64u;
    double boxAngleIncrementDegrees = 0.5;
    double boxShapeResidualBoundCalipers =
        numi::cardboard::kDefaultBoxShapeResidualBoundCalipers;
    double boxShapeFitCoverageMinimum =
        numi::cardboard::kDefaultBoxShapeFitCoverage;
    double boxPunchRadiusM = 0.0002;
    double boxScoreToolClearanceM = 0.00005;
    std::uint32_t boxScoreStepsPerLeg = 16u;
    std::uint32_t boxMaximumStepsPerAction = 100000u;
    numi::cardboard::FEFCO0201BlankConfig boxConfig;
    std::optional<numi::cardboard::ShippingBoxDesign> shippingDesign;
    std::optional<numi::cardboard::ShippingStarchLawParameters>
        shippingStarchLaw;
    std::optional<numi::cardboard::ShippingHmaLawParameters>
        shippingHmaLaw;
    bool referenceDisplacementGradient = false;
    double indentationM = 0.000020;
    double punchRadiusM = 0.000750;
    double toolClearanceM = 0.000010;
    double anvilClearanceM = 0.000010;
    std::string preset = "literature2009";
    bool withoutMedium = false;
    bool compileOnly = false;
    bool help = false;
    std::uint32_t steps = 8u;
    std::uint32_t holdSteps = 0u;
    std::uint32_t unloadSteps = 0u;
    std::uint32_t relaxSteps = 0u;
    std::uint32_t releaseSteps = 0u;
    std::uint32_t fgmresIterations = NM_MIXED_FGMRES_ITERATIONS;
    std::uint32_t fgmresRestart = NM_MIXED_FGMRES_DEFAULT_RESTART;
    bool fgmresRestartExplicit = false;
    std::uint32_t lineSearchSteps = 8u;
    FEMPreconditionerMode femPreconditioner =
        FEMPreconditionerMode::scalarDiagonal;
    std::uint32_t localMaterialIterations = 8u;
    std::uint32_t newtonIterations = NM_MIXED_NEWTON_ITERATIONS;
    std::uint32_t nxPerPitch = 16u;
    std::uint32_t ny = 4u;
    std::uint32_t thicknessSlices = 2u;
    double bendAngleDegrees = 1.0;
    double timestepSeconds = 1.0e-4;
    double relativeResidualTolerance = 1.0e-4;
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

using StarchInterfaceParameters =
    numi::cardboard::ShippingStarchLawParameters;

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
    // Stable constituent role, independent of any pooled/overlay material
    // index: 0=liner, 1=medium, 2=finite glue.
    std::vector<std::uint32_t> authoredRegionIds;
    std::string authoredRegionMapDigest;
    std::string materialMapDigest;
    std::string frameMapDigest;
    std::string damageOverlayMappingDigest;
    std::string damageOverlaySidecarSha256;
    std::uint64_t damageOverlayTetrahedronCount = 0u;
    std::uint32_t damageOverlayProgramCount = 0u;
    std::vector<numi::cardboard::retained_damage::ElementMapping>
        damageOverlayMappings;
    std::vector<std::uint32_t> damageOverlayMaterialIndices;
    std::vector<numi::cardboard::StarchPairedFaceRecord> starchPairedFaces;
    std::vector<NMCohesiveActivationCommandGPU> starchActivationCommands;
    std::optional<StarchInterfaceParameters> starchInterfaceParameters;
    std::vector<numi::cardboard::GlueFootprint> glueFootprints;
    std::optional<numi::cardboard::CreaseToolingGeometry> tooling;
    std::optional<numi::cardboard::FEFCO0201Blank> boxBlank;
    std::optional<numi::cardboard::BoxScoreToolingSet> boxScoreTooling;
    numi::cardboard::ShippingClosureMortarMesh hmaClosureMesh;
    std::optional<numi::cardboard::PinAdhesionFixtureGeometry> pinFixture;
    std::uint32_t shippingImpactorProxyIndex = NM_INVALID_INDEX;
};

struct Digest {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> bytes{};
};

std::string hexDigest(const Digest& digest);

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

void loadShippingCapacityQualification(Arguments& arguments) {
    if (arguments.shippingCapacityQualificationFile.empty()) return;
    try {
        require(std::filesystem::is_regular_file(
                    arguments.shippingCapacityQualificationFile),
                "capacity qualification file is not a regular file");
        std::ifstream input(arguments.shippingCapacityQualificationFile);
        require(input.good(), "cannot open shipping capacity qualification file");
        auto qualification =
            numi::cardboard::parseShippingCapacityQualification(
                input, arguments.shippingCapacityQualificationFile.string());
        const auto verifyReport = [&](const std::string& relativeName,
                                      const std::string& expectedSha) {
            const auto path = std::filesystem::weakly_canonical(
                arguments.shippingCapacityQualificationFile.parent_path() /
                relativeName);
            require(std::filesystem::is_regular_file(path),
                    "capacity qualification comparison report is missing: " + relativeName);
            require(hexDigest(sha256File(path)) == expectedSha,
                    "capacity qualification comparison report digest mismatch: " + relativeName);
        };
        verifyReport(qualification.meshReportFile,
                     qualification.meshReportSha256);
        verifyReport(qualification.timeReportFile,
                     qualification.timeReportSha256);
        arguments.shippingCapacityQualificationSha256 =
            hexDigest(sha256File(arguments.shippingCapacityQualificationFile));
        arguments.shippingCapacityQualification = std::move(qualification);
    } catch (const std::exception& error) {
        arguments.shippingCapacityQualificationLoadError = error.what();
        arguments.shippingCapacityQualification.reset();
        if (std::filesystem::is_regular_file(
                arguments.shippingCapacityQualificationFile)) {
            try {
                arguments.shippingCapacityQualificationSha256 = hexDigest(
                    sha256File(arguments.shippingCapacityQualificationFile));
            } catch (...) {
                // The parse error remains the primary admission diagnostic.
            }
        }
    }
}

std::string shippingSourceBindingDigest(const Arguments& arguments) {
    std::ostringstream binding;
    binding << "numi.cardboard.shipping-source-binding.v1\n"
            << "preset=literature2009\n"
            << "liner=" << (arguments.linerMaterial.empty() ? "none" :
                hexDigest(sha256File(arguments.linerMaterial))) << '\n'
            << "medium=" << (arguments.mediumMaterial.empty() ? "none" :
                hexDigest(sha256File(arguments.mediumMaterial))) << '\n'
            << "glue=" << (arguments.glueMaterial.empty() ? "none" :
                hexDigest(sha256File(arguments.glueMaterial))) << '\n'
            << "starch_parameters=" << arguments.starchInterfaceParametersSha256 << '\n'
            << "hma_parameters=" << arguments.shippingHmaParametersSha256 << '\n'
            << "liner_damage_parameters=" << arguments.linerDamageParametersSha256 << '\n'
            << "medium_damage_parameters=" << arguments.mediumDamageParametersSha256 << '\n'
            << "allowance_coupon=" << arguments.shippingCouponEvidenceSha256 << '\n'
            << "score_depth_ratio=" << std::setprecision(17)
            << arguments.shippingScoreDepthRatio << '\n'
            << "score_tool_speed_m_s=" << arguments.shippingScoreToolSpeedMPerS << '\n';
    return hexDigest(sha256(binding.str()));
}

std::string shippingBlankGeometryBindingDigest(const MeshSource& mesh) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    const auto updateBytes = [&](const void* data, std::uint64_t byteCount) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        while (byteCount > 0u) {
            const CC_LONG chunk = static_cast<CC_LONG>(std::min<std::uint64_t>(
                byteCount, std::numeric_limits<CC_LONG>::max()));
            CC_SHA256_Update(&context, bytes, chunk);
            bytes += chunk;
            byteCount -= chunk;
        }
    };
    const auto updateTag = [&](const std::string_view tag) {
        updateBytes(tag.data(), tag.size());
    };
    const auto updateCount = [&](const std::size_t count) {
        const std::uint64_t encoded = static_cast<std::uint64_t>(count);
        updateBytes(&encoded, sizeof(encoded));
    };
    updateTag("numi.cardboard.shipping-blank-geometry.v1\n");
    updateCount(mesh.restNodes.size());
    for (const Vec3& node : mesh.restNodes) {
        updateBytes(&node.x, sizeof(node.x));
        updateBytes(&node.y, sizeof(node.y));
        updateBytes(&node.z, sizeof(node.z));
    }
    updateCount(mesh.tetrahedra.size());
    for (const TetMetadata& tet : mesh.tetrahedra) {
        for (const std::uint32_t node : tet.nodes)
            updateBytes(&node, sizeof(node));
        updateBytes(&tet.materialIndex, sizeof(tet.materialIndex));
        updateBytes(&tet.frameAngle, sizeof(tet.frameAngle));
    }
    updateCount(mesh.materialFrames.size());
    for (const auto& frame : mesh.materialFrames)
        for (const double component : frame)
            updateBytes(&component, sizeof(component));
    updateCount(mesh.materialIndices.size());
    if (!mesh.materialIndices.empty())
        updateBytes(mesh.materialIndices.data(),
                    mesh.materialIndices.size() * sizeof(std::uint32_t));
    updateCount(mesh.authoredRegionIds.size());
    if (!mesh.authoredRegionIds.empty())
        updateBytes(mesh.authoredRegionIds.data(),
                    mesh.authoredRegionIds.size() * sizeof(std::uint32_t));
    Digest result;
    CC_SHA256_Final(result.bytes.data(), &context);
    return hexDigest(result);
}

std::string shippingJsonStringField(NSDictionary* const dictionary,
                                    NSString* const key,
                                    const std::string& label) {
    id value = dictionary[key];
    require(value != nil && [value isKindOfClass:[NSString class]],
            "shipping roof measurement field is missing or not text: " + label);
    const char* utf8 = [(NSString*)value UTF8String];
    require(utf8 != nullptr, "shipping roof measurement text is not UTF-8: " + label);
    return utf8;
}

double shippingJsonDoubleField(NSDictionary* const dictionary,
                               NSString* const key,
                               const std::string& label) {
    id value = dictionary[key];
    require(value != nil && [value isKindOfClass:[NSNumber class]],
            "shipping roof measurement field is missing or not numeric: " + label);
    const double result = [(NSNumber*)value doubleValue];
    require(std::isfinite(result),
            "shipping roof measurement field is nonfinite: " + label);
    return result;
}

std::uint64_t shippingJsonUnsignedField(NSDictionary* const dictionary,
                                        NSString* const key,
                                        const std::string& label) {
    id value = dictionary[key];
    require(value != nil && [value isKindOfClass:[NSNumber class]],
            "shipping roof measurement field is missing or not an integer: " + label);
    require(CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
            "shipping roof measurement integer may not be a JSON boolean: " + label);
    NSString* const encodedNumber = [(NSNumber*)value stringValue];
    const char* utf8 = [encodedNumber UTF8String];
    require(utf8 != nullptr && utf8[0] != '\0' && utf8[0] != '-',
            "shipping roof measurement integer is negative or not textual: " + label);
    std::size_t used = 0u;
    unsigned long long parsed = 0u;
    try {
        parsed = std::stoull(utf8, &used, 10);
    } catch (const std::exception&) {
        require(false, "shipping roof measurement integer is out of range: " + label);
    }
    require(used == std::strlen(utf8),
            "shipping roof measurement integer is fractional or exponent encoded: " + label);
    return static_cast<std::uint64_t>(parsed);
}

numi::cardboard::ShippingOuterRoofMeasurement loadShippingOuterRoofMeasurement(
    const std::filesystem::path& measurementPath) {
    namespace fs = std::filesystem;
    require(fs::is_regular_file(measurementPath),
            "shipping roof measurement is not a regular file");
    const std::uintmax_t fileBytes = fs::file_size(measurementPath);
    require(fileBytes > 0u && fileBytes <= 1024u * 1024u,
            "shipping roof measurement exceeds the 1 MiB input bound");
    std::ifstream input(measurementPath, std::ios::binary);
    require(input.good(), "cannot open shipping roof measurement");
    std::string content(static_cast<std::size_t>(fileBytes), '\0');
    input.read(content.data(), static_cast<std::streamsize>(content.size()));
    require(input.good() || input.eof(), "failed reading shipping roof measurement");
    require(static_cast<std::size_t>(input.gcount()) == content.size(),
            "shipping roof measurement was truncated while reading");
    NSData* const jsonData = [NSData dataWithBytes:content.data()
                                             length:content.size()];
    NSError* jsonError = nil;
    id parsed = [NSJSONSerialization JSONObjectWithData:jsonData
                                                options:NSJSONReadingFragmentsAllowed
                                                  error:&jsonError];
    require(parsed != nil && [parsed isKindOfClass:[NSDictionary class]],
            "shipping roof measurement JSON is invalid");
    NSDictionary* const root = (NSDictionary*)parsed;
    require(shippingJsonStringField(root, @"schema", "schema") ==
                "numi.cardboard.accepted-outer-roof.v2",
            "shipping roof input requires the source-bound accepted-outer-roof.v2 schema");
    const std::string status =
        shippingJsonStringField(root, @"status", "status");
    require(status == "measured_from_accepted_settled_checkpoint",
            "shipping roof input is not from an accepted settled assembly checkpoint");
    numi::cardboard::ShippingOuterRoofMeasurement result;
    result.acceptedRecordSchemaVerified = true;
    result.measuredFromAcceptedFixtureFreeSettledCheckpoint = true;
    result.coordinateUnits = shippingJsonStringField(root, @"units", "units");
    result.method = shippingJsonStringField(root, @"method", "method");
    result.acceptedCheckpointFile = shippingJsonStringField(
        root, @"accepted_fem_checkpoint_file", "accepted_fem_checkpoint_file");
    result.checkpointSha256 = shippingJsonStringField(
        root, @"accepted_fem_checkpoint_sha256", "accepted_fem_checkpoint_sha256");
    result.sourceBlankGeometrySha256 = shippingJsonStringField(
        root, @"source_blank_geometry_sha256", "source_blank_geometry_sha256");
    result.sourceShippingBindingSha256 = shippingJsonStringField(
        root, @"source_shipping_binding_sha256", "source_shipping_binding_sha256");
    result.sourcePhysicsFingerprint = shippingJsonUnsignedField(
        root, @"source_physics_fingerprint", "source_physics_fingerprint");
    result.compiledWorldFingerprint = shippingJsonUnsignedField(
        root, @"compiled_world_fingerprint", "compiled_world_fingerprint");
    NSDictionary* const bounds = (NSDictionary*)root[@"outer_roof_bounds"];
    require(bounds != nil && [bounds isKindOfClass:[NSDictionary class]],
            "shipping roof measurement outer_roof_bounds is missing");
    result.bounds.minimumX = shippingJsonDoubleField(bounds, @"min_x", "min_x");
    result.bounds.maximumX = shippingJsonDoubleField(bounds, @"max_x", "max_x");
    result.bounds.minimumZ = shippingJsonDoubleField(bounds, @"min_z", "min_z");
    result.bounds.maximumZ = shippingJsonDoubleField(bounds, @"max_z", "max_z");
    result.bounds.topY = shippingJsonDoubleField(bounds, @"top_y", "top_y");
    const std::uint64_t faceCount = shippingJsonUnsignedField(
        bounds, @"exterior_face_count", "exterior_face_count");
    require(faceCount > 0u && faceCount <= std::numeric_limits<std::uint32_t>::max(),
            "shipping roof measurement exterior face count is invalid");
    result.bounds.exteriorFaceCount = static_cast<std::uint32_t>(faceCount);
    result.bounds.valid = result.bounds.maximumX > result.bounds.minimumX &&
        result.bounds.maximumZ > result.bounds.minimumZ;

    NSArray* const extentValues = (NSArray*)root[@"compiled_impactor_half_extents_m"];
    require(extentValues != nil && [extentValues isKindOfClass:[NSArray class]] &&
                [extentValues count] == 3u,
            "shipping roof measurement must retain the compiled impactor extents");
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        id value = extentValues[axis];
        require(value != nil && [value isKindOfClass:[NSNumber class]],
                "shipping roof measurement impactor extent is not numeric");
        result.compiledImpactorHalfExtentsM[axis] = [(NSNumber*)value doubleValue];
    }
    const std::filesystem::path relativeCheckpoint(result.acceptedCheckpointFile);
    require(!relativeCheckpoint.empty() && !relativeCheckpoint.is_absolute(),
            "shipping roof checkpoint path must be relative to its measurement record");
    for (const auto& component : relativeCheckpoint)
        require(component != "..",
                "shipping roof checkpoint path may not escape the measurement directory");
    const fs::path checkpointPath = fs::weakly_canonical(
        measurementPath.parent_path() / relativeCheckpoint);
    require(fs::is_regular_file(checkpointPath),
            "source-bound shipping roof checkpoint file is missing");
    require(numi::cardboard::shippingRoofSha256Valid(result.checkpointSha256) &&
                hexDigest(sha256File(checkpointPath)) == result.checkpointSha256,
            "shipping roof checkpoint SHA-256 does not match its measured record");
    const auto checkpointIdentity =
        numi::cardboard::readShippingAcceptedSnapshotIdentity(checkpointPath);
    require(checkpointIdentity.available &&
                checkpointIdentity.sourcePhysicsFingerprint ==
                    result.sourcePhysicsFingerprint &&
                checkpointIdentity.deviceProgramFingerprint ==
                    result.compiledWorldFingerprint,
            "shipping roof checkpoint header fingerprints differ from the measured record");
    result.checkpointFileHashVerified = true;
    result.sourceMeasurementFileSha256 = hexDigest(sha256(content));
    require(numi::cardboard::shippingRoofSha256Valid(
                result.sourceBlankGeometrySha256) &&
                numi::cardboard::shippingRoofSha256Valid(
                    result.sourceShippingBindingSha256) && result.bounds.valid,
            "shipping roof measurement source binding or bounds are invalid");
    return result;
}

void bindShippingRoofMeasurementToCurrentSource(Arguments& arguments,
                                               const MeshSource& mesh) {
    if (arguments.shippingRoofMeasurementFile.empty()) return;
    try {
        arguments.shippingRoofMeasurementFile = std::filesystem::weakly_canonical(
            arguments.shippingRoofMeasurementFile);
        auto measurement = loadShippingOuterRoofMeasurement(
            arguments.shippingRoofMeasurementFile);
        arguments.shippingRoofMeasurementFileSha256 =
            measurement.sourceMeasurementFileSha256;
        arguments.shippingRoofMeasurementCheckpointPath =
            measurement.acceptedCheckpointFile;
        arguments.shippingBlankGeometrySha256 =
            shippingBlankGeometryBindingDigest(mesh);
        require(numi::cardboard::shippingRoofMeasurementMatchesCurrentSource(
                    measurement, arguments.shippingBlankGeometrySha256,
                    shippingSourceBindingDigest(arguments)),
                "source-bound shipping roof geometry/material/protocol identity does not match the current blank");
        require(arguments.shippingImpactorHalfExtentsProvided &&
                    numi::cardboard::shippingImpactorHalfExtentsMatchMeasuredRoof(
                        measurement, arguments.shippingImpactorHalfExtentsM),
                "explicit impactor X/Z half-extents do not equal 0.6 times the source-bound measured roof spans");
        arguments.shippingImpactorSourceMeasurementValidated = true;
        arguments.shippingRoofMeasurementInput = std::move(measurement);
        arguments.shippingImpactorGeometrySource =
            "source_bound_measured_roof_input_plus_explicit_half_extents";
    } catch (const std::exception& error) {
        arguments.shippingRoofMeasurementLoadFailure = error.what();
        throw std::runtime_error(
            "shipping impactor source-measurement admission failed: " +
            arguments.shippingRoofMeasurementLoadFailure);
    }
}

std::string writeShippingNominalCurveArtifact(
    const std::filesystem::path& outputDirectory,
    const std::vector<numi::cardboard::ShippingCompressionForceSample>& curve) {
    const auto path = outputDirectory / "shipping-compression-nominal-curve.csv";
    std::ofstream output(path);
    require(output.good(), "cannot create nominal compression curve artifact");
    output << "sample_index,stroke_m,force_N,contact_samples\n"
           << std::setprecision(17);
    for (const auto& sample : curve)
        output << sample.sampleIndex << ',' << sample.strokeM << ','
               << sample.forceN << ',' << sample.contactSamples << '\n';
    output.close();
    require(output.good(), "failed writing nominal compression curve artifact");
    return hexDigest(sha256File(path));
}

std::string hexDigest(const Digest& digest) {
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (const unsigned char byte : digest.bytes)
        result << std::setw(2) << static_cast<unsigned>(byte);
    return result.str();
}

std::string shippingScoreCommandFingerprint(
    const Arguments& arguments,
    const std::uint64_t sourcePhysicsFingerprint,
    const std::uint64_t deviceProgramFingerprint,
    const double boardCaliperM) {
    std::ostringstream identity;
    identity << std::setprecision(17)
             << "numi.cardboard.shipping-score-command.v1\n"
             << "source_physics_fingerprint=" << sourcePhysicsFingerprint << '\n'
             << "device_program_fingerprint=" << deviceProgramFingerprint << '\n'
             << "score_line_count=12\n"
             << "selected_depth_ratio=" << arguments.shippingScoreDepthRatio << '\n'
             << "board_caliper_m=" << boardCaliperM << '\n'
             << "tool_speed_m_per_s=" << arguments.shippingScoreToolSpeedMPerS << '\n'
             << "timestep_s=" << arguments.timestepSeconds << '\n'
             << "one_selected_depth_from_fresh_flat_blank=true\n"
             << "stroke=" << "accepted_approach_then_withdrawal" << '\n';
    return hexDigest(sha256(identity.str()));
}

template<class T>
std::string sha256Arena(const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (values.empty()) return hexDigest(sha256("", 0u));
    require(values.size() <= std::numeric_limits<std::size_t>::max() / sizeof(T),
            "shipping state arena byte count overflowed");
    return hexDigest(sha256(values.data(), values.size() * sizeof(T)));
}

[[maybe_unused]] std::map<std::string, std::string> shippingPersistentArenaHashes(
    const RuntimeStateSnapshot& snapshot) {
    return {
        {"fem_nodes", sha256Arena(snapshot.femNodes)},
        {"fem_fields", sha256Arena(snapshot.femFields)},
        {"fem_material_state", sha256Arena(snapshot.femMaterialState)},
        {"fem_topology_nodes", sha256Arena(snapshot.femTopologyNodes)},
        {"fem_topology_tetrahedra", sha256Arena(snapshot.femTopologyTetrahedra)},
        {"fem_topology_states", sha256Arena(snapshot.topologyStates)},
        {"cohesive_faces", sha256Arena(snapshot.cohesiveFaces)},
        {"puncture_channels", sha256Arena(snapshot.punctureChannels)},
        {"contact_histories", sha256Arena(snapshot.contactHistories)},
        {"deformable_contact_histories",
         sha256Arena(snapshot.deformableContactHistories)},
        {"human_support_histories", sha256Arena(snapshot.humanSupportHistories)},
        {"particle_material_state", sha256Arena(snapshot.particleMaterialState)},
    };
}

[[maybe_unused]] void writeShippingArenaHashes(std::ostream& output,
                              const std::map<std::string, std::string>& hashes) {
    output << '{';
    bool first = true;
    for (const auto& [name, digest] : hashes) {
        output << (first ? "" : ",") << "\n    \"" << name
               << "\": \"" << digest << '\"';
        first = false;
    }
    if (!first) output << '\n' << "  ";
    output << '}';
}

[[maybe_unused]] bool sameShippingSceneBodyState(const MRBodyStateGPU& a,
                                const MRBodyStateGPU& b) {
    const std::array<float, 28> valuesA{
        a.position.x, a.position.y, a.position.z, a.position.w,
        a.orientation.x, a.orientation.y, a.orientation.z, a.orientation.w,
        a.linearVelocityAndInverseMass.x, a.linearVelocityAndInverseMass.y,
        a.linearVelocityAndInverseMass.z, a.linearVelocityAndInverseMass.w,
        a.angularVelocity.x, a.angularVelocity.y, a.angularVelocity.z,
        a.angularVelocity.w,
        a.inverseInertiaWorldRow0.x, a.inverseInertiaWorldRow0.y,
        a.inverseInertiaWorldRow0.z, a.inverseInertiaWorldRow0.w,
        a.inverseInertiaWorldRow1.x, a.inverseInertiaWorldRow1.y,
        a.inverseInertiaWorldRow1.z, a.inverseInertiaWorldRow1.w,
        a.inverseInertiaWorldRow2.x, a.inverseInertiaWorldRow2.y,
        a.inverseInertiaWorldRow2.z, a.inverseInertiaWorldRow2.w};
    const std::array<float, 28> valuesB{
        b.position.x, b.position.y, b.position.z, b.position.w,
        b.orientation.x, b.orientation.y, b.orientation.z, b.orientation.w,
        b.linearVelocityAndInverseMass.x, b.linearVelocityAndInverseMass.y,
        b.linearVelocityAndInverseMass.z, b.linearVelocityAndInverseMass.w,
        b.angularVelocity.x, b.angularVelocity.y, b.angularVelocity.z,
        b.angularVelocity.w,
        b.inverseInertiaWorldRow0.x, b.inverseInertiaWorldRow0.y,
        b.inverseInertiaWorldRow0.z, b.inverseInertiaWorldRow0.w,
        b.inverseInertiaWorldRow1.x, b.inverseInertiaWorldRow1.y,
        b.inverseInertiaWorldRow1.z, b.inverseInertiaWorldRow1.w,
        b.inverseInertiaWorldRow2.x, b.inverseInertiaWorldRow2.y,
        b.inverseInertiaWorldRow2.z, b.inverseInertiaWorldRow2.w};
    return valuesA == valuesB && std::equal(
        std::begin(a.flagsAndIndices), std::end(a.flagsAndIndices),
        std::begin(b.flagsAndIndices));
}

#ifdef NUMI_CARDBOARD_METAL_WORLD
struct ShippingRigidSceneSidecar {
    std::filesystem::path binaryPath;
    std::filesystem::path metadataPath;
    std::string binarySha256;
    std::string sceneBodiesSha256;
};

template<class T>
void writeShippingRigidPod(std::ostream& output, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
    require(output.good(), "failed writing rigid scene checkpoint field");
}

template<class T>
T readShippingRigidPod(std::istream& input) {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    require(input.good(), "rigid scene checkpoint is truncated");
    return value;
}

std::filesystem::path writeShippingRigidSceneBinary(
    const std::filesystem::path& checkpointPath,
    const RuntimeStateSnapshot& snapshot,
    const std::uint64_t rigidWorldFingerprint,
    const double rigidMassKg,
    const std::array<double, 3>& gravityMPerS2,
    const std::vector<MRBodyStateGPU>& sceneBodies,
    const std::uint32_t activeTemporaryClampNodeCount,
    const double rigidContactSamples,
    const double supportFloorContactSamples,
    const double simulationTimeSeconds,
    const bool settled,
    const bool requireTemporaryToolContactFree = true) {
    static_assert(std::is_trivially_copyable_v<MRBodyStateGPU>);
    require(snapshot.available && !snapshot.statuses.empty() &&
                !snapshot.solverCertificates.empty() &&
                snapshot.statuses.front().code == NM_STATUS_SUCCESS &&
                snapshot.solverCertificates.front().validity.w > 0.5f,
            "rigid scene sidecar requires an accepted Matter snapshot");
    require(sceneBodies.size() >= 5u && rigidWorldFingerprint != 0u &&
                sceneBodies.back().flagsAndIndices[0] == MR_MOTION_DYNAMIC,
            "shipping rigid scene sidecar requires kinematic tools plus an appended finite-mass body");
    require(activeTemporaryClampNodeCount == 0u,
            "released rigid scene sidecar cannot retain temporary boundary clamps");
    const double forbiddenToolSamples =
        rigidContactSamples - supportFloorContactSamples;
    require(std::isfinite(rigidContactSamples) && rigidContactSamples >= 0.0 &&
                std::isfinite(supportFloorContactSamples) &&
                supportFloorContactSamples >= 0.0 &&
                forbiddenToolSamples >= -1.0e-6 &&
                (!requireTemporaryToolContactFree ||
                 forbiddenToolSamples <= 1.0e-6),
            "rigid scene sidecar violates its temporary-tool/support-floor contact gate");
    require(std::isfinite(rigidMassKg) && rigidMassKg > 0.0 &&
                std::isfinite(simulationTimeSeconds) && simulationTimeSeconds >= 0.0,
            "rigid scene sidecar mass or time is invalid");
    for (const double gravity : gravityMPerS2)
        require(std::isfinite(gravity), "rigid scene sidecar gravity is nonfinite");
    for (const auto& body : sceneBodies) {
        const std::array<float, 28> values{
            body.position.x, body.position.y, body.position.z, body.position.w,
            body.orientation.x, body.orientation.y, body.orientation.z, body.orientation.w,
            body.linearVelocityAndInverseMass.x, body.linearVelocityAndInverseMass.y,
            body.linearVelocityAndInverseMass.z, body.linearVelocityAndInverseMass.w,
            body.angularVelocity.x, body.angularVelocity.y, body.angularVelocity.z,
            body.angularVelocity.w,
            body.inverseInertiaWorldRow0.x, body.inverseInertiaWorldRow0.y,
            body.inverseInertiaWorldRow0.z, body.inverseInertiaWorldRow0.w,
            body.inverseInertiaWorldRow1.x, body.inverseInertiaWorldRow1.y,
            body.inverseInertiaWorldRow1.z, body.inverseInertiaWorldRow1.w,
            body.inverseInertiaWorldRow2.x, body.inverseInertiaWorldRow2.y,
            body.inverseInertiaWorldRow2.z, body.inverseInertiaWorldRow2.w};
        require(std::all_of(values.begin(), values.end(),
                    [](const float value) { return std::isfinite(value); }),
                "rigid scene sidecar contains nonfinite body state");
    }

    const Digest checkpointDigest = sha256File(checkpointPath);
    const std::filesystem::path binaryPath = checkpointPath.string() + ".rigid.nmsb";
    const std::filesystem::path partialPath = binaryPath.string() + ".partial";
    std::ofstream output(partialPath, std::ios::binary | std::ios::trunc);
    require(output.good(), "cannot create paired rigid scene checkpoint");
    constexpr std::array<char, 8> magic{'N','M','R','I','G','I','D','1'};
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    writeShippingRigidPod(output, std::uint32_t{2u});
    writeShippingRigidPod(output, std::uint32_t{0x01020304u});
    writeShippingRigidPod(output,
        static_cast<std::uint32_t>(sizeof(MRBodyStateGPU)));
    writeShippingRigidPod(output, static_cast<std::uint32_t>(sceneBodies.size()));
    writeShippingRigidPod(output, snapshot.sourcePhysicsFingerprint);
    writeShippingRigidPod(output, snapshot.deviceProgramFingerprint);
    writeShippingRigidPod(output, rigidWorldFingerprint);
    writeShippingRigidPod(output, snapshot.controlStep);
    writeShippingRigidPod(output, simulationTimeSeconds);
    writeShippingRigidPod(output, rigidMassKg);
    for (const double gravity : gravityMPerS2)
        writeShippingRigidPod(output, gravity);
    writeShippingRigidPod(output, rigidContactSamples);
    writeShippingRigidPod(output, supportFloorContactSamples);
    writeShippingRigidPod(output, activeTemporaryClampNodeCount);
    writeShippingRigidPod(output, static_cast<std::uint32_t>(settled ? 1u : 0u));
    writeShippingRigidPod(output,
        static_cast<std::uint32_t>(requireTemporaryToolContactFree ? 1u : 0u));
    output.write(reinterpret_cast<const char*>(checkpointDigest.bytes.data()),
        static_cast<std::streamsize>(checkpointDigest.bytes.size()));
    const Digest sceneDigest = sha256(sceneBodies.data(),
        sceneBodies.size() * sizeof(MRBodyStateGPU));
    output.write(reinterpret_cast<const char*>(sceneDigest.bytes.data()),
        static_cast<std::streamsize>(sceneDigest.bytes.size()));
    output.write(reinterpret_cast<const char*>(sceneBodies.data()),
        static_cast<std::streamsize>(sceneBodies.size() * sizeof(MRBodyStateGPU)));
    output.flush();
    require(output.good(), "failed flushing paired rigid scene checkpoint");
    output.close();
    require(!output.fail(), "failed closing paired rigid scene checkpoint");
    std::error_code error;
    std::filesystem::rename(partialPath, binaryPath, error);
    require(!error, "cannot publish paired rigid scene checkpoint: " + error.message());
    return binaryPath;
}

std::vector<MRBodyStateGPU> readShippingRigidSceneBinary(
    const std::filesystem::path& binaryPath,
    const std::filesystem::path& checkpointPath,
    const std::uint64_t sourcePhysicsFingerprint,
    const std::uint64_t deviceProgramFingerprint,
    const std::uint64_t rigidWorldFingerprint,
    const double rigidMassKg,
    const std::uint32_t expectedBodyCount,
    const std::uint32_t expectedControlStep,
    const bool requireSettled,
    const bool requireTemporaryToolContactFree = true) {
    std::ifstream input(binaryPath, std::ios::binary);
    require(input.good(), "cannot open paired rigid scene checkpoint");
    constexpr std::array<char, 8> magicExpected{'N','M','R','I','G','I','D','1'};
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    require(input.good() && magic == magicExpected,
            "paired rigid scene checkpoint has an unknown format");
    require(readShippingRigidPod<std::uint32_t>(input) == 2u &&
                readShippingRigidPod<std::uint32_t>(input) == 0x01020304u &&
                readShippingRigidPod<std::uint32_t>(input) == sizeof(MRBodyStateGPU),
            "paired rigid scene checkpoint ABI or byte order differs");
    const std::uint32_t bodyCount = readShippingRigidPod<std::uint32_t>(input);
    require(bodyCount == expectedBodyCount && bodyCount <= 64u,
            "paired rigid scene checkpoint body count differs");
    require(readShippingRigidPod<std::uint64_t>(input) == sourcePhysicsFingerprint &&
                readShippingRigidPod<std::uint64_t>(input) == deviceProgramFingerprint &&
                readShippingRigidPod<std::uint64_t>(input) == rigidWorldFingerprint,
            "paired rigid scene checkpoint source or world fingerprint differs");
    require(readShippingRigidPod<std::uint32_t>(input) == expectedControlStep,
            "paired rigid scene checkpoint accepted step differs");
    const double simulationTimeSeconds = readShippingRigidPod<double>(input);
    const double storedMassKg = readShippingRigidPod<double>(input);
    for (unsigned axis = 0u; axis < 3u; ++axis)
        require(std::isfinite(readShippingRigidPod<double>(input)),
                "paired rigid scene checkpoint has nonfinite gravity");
    const double rigidContactSamples = readShippingRigidPod<double>(input);
    const double supportFloorContactSamples =
        readShippingRigidPod<double>(input);
    const double forbiddenToolSamples =
        rigidContactSamples - supportFloorContactSamples;
    require(std::isfinite(rigidContactSamples) && rigidContactSamples >= 0.0 &&
                std::isfinite(supportFloorContactSamples) &&
                supportFloorContactSamples >= 0.0 &&
                forbiddenToolSamples >= -1.0e-6 &&
                (!requireTemporaryToolContactFree ||
                 forbiddenToolSamples <= 1.0e-6),
            "paired rigid scene checkpoint violates its temporary-tool/support-floor gate");
    require(readShippingRigidPod<std::uint32_t>(input) == 0u,
            "paired rigid scene checkpoint retained temporary clamps");
    const std::uint32_t settled = readShippingRigidPod<std::uint32_t>(input);
    const std::uint32_t contactFreeRequired =
        readShippingRigidPod<std::uint32_t>(input);
    require(settled <= 1u && (!requireSettled || settled == 1u) &&
                contactFreeRequired <= 1u &&
                (!requireTemporaryToolContactFree || contactFreeRequired == 1u) &&
                (!requireTemporaryToolContactFree ||
                 forbiddenToolSamples <= 1.0e-6) &&
                std::isfinite(simulationTimeSeconds) && simulationTimeSeconds >= 0.0 &&
                std::isfinite(storedMassKg) &&
                std::abs(storedMassKg - rigidMassKg) <=
                    2.0e-6 * std::max(1.0, rigidMassKg),
            "paired rigid scene checkpoint time, settle state, or mass differs");
    Digest storedCheckpointDigest;
    input.read(reinterpret_cast<char*>(storedCheckpointDigest.bytes.data()),
        static_cast<std::streamsize>(storedCheckpointDigest.bytes.size()));
    require(input.good() && storedCheckpointDigest.bytes == sha256File(checkpointPath).bytes,
            "paired rigid scene sidecar is bound to a different Matter checkpoint hash");
    Digest storedSceneDigest;
    input.read(reinterpret_cast<char*>(storedSceneDigest.bytes.data()),
        static_cast<std::streamsize>(storedSceneDigest.bytes.size()));
    std::vector<MRBodyStateGPU> bodies(bodyCount);
    input.read(reinterpret_cast<char*>(bodies.data()),
        static_cast<std::streamsize>(bodies.size() * sizeof(MRBodyStateGPU)));
    require(input.good(), "paired rigid scene checkpoint body state is truncated");
    require(storedSceneDigest.bytes ==
                sha256(bodies.data(), bodies.size() * sizeof(MRBodyStateGPU)).bytes,
            "paired rigid scene checkpoint body-state digest differs");
    require(input.peek() == std::char_traits<char>::eof(),
            "paired rigid scene checkpoint has trailing bytes");
    return bodies;
}

ShippingRigidSceneSidecar writeShippingRigidSceneSidecar(
    const std::filesystem::path& checkpointPath,
    const RuntimeStateSnapshot& snapshot,
    const std::uint64_t rigidWorldFingerprint,
    const double rigidMassKg,
    const std::array<double, 3>& gravityMPerS2,
    const std::vector<MRBodyStateGPU>& sceneBodies,
    const std::uint32_t activeTemporaryClampNodeCount,
    const double rigidContactSamples,
    const double supportFloorContactSamples,
    const double simulationTimeSeconds,
    const bool settled,
    const bool requireTemporaryToolContactFree = true) {
    ShippingRigidSceneSidecar result;
    result.binaryPath = writeShippingRigidSceneBinary(
        checkpointPath, snapshot, rigidWorldFingerprint, rigidMassKg,
        gravityMPerS2, sceneBodies, activeTemporaryClampNodeCount,
        rigidContactSamples, supportFloorContactSamples,
        simulationTimeSeconds, settled, requireTemporaryToolContactFree);
    result.binarySha256 = hexDigest(sha256File(result.binaryPath));
    result.sceneBodiesSha256 = sha256Arena(sceneBodies);
    result.metadataPath = result.binaryPath.string() + ".json";
    std::ofstream metadata(result.metadataPath, std::ios::trunc);
    require(metadata.good(), "cannot create paired rigid scene checkpoint metadata");
    const Digest checkpointDigest = sha256File(checkpointPath);
    metadata << std::setprecision(17)
        << "{\n  \"schema\": \"numi.cardboard.shipping-rigid-scene-sidecar.v2\",\n"
        << "  \"matter_checkpoint_file\": \""
        << jsonEscape(checkpointPath.filename().string()) << "\",\n"
        << "  \"matter_checkpoint_sha256\": \""
        << hexDigest(checkpointDigest) << "\",\n"
        << "  \"sidecar_binary_file\": \""
        << jsonEscape(result.binaryPath.filename().string()) << "\",\n"
        << "  \"sidecar_binary_sha256\": \"" << result.binarySha256 << "\",\n"
        << "  \"scene_bodies_sha256\": \"" << result.sceneBodiesSha256 << "\",\n"
        << "  \"matter_control_step\": " << snapshot.controlStep << ",\n"
        << "  \"source_physics_fingerprint\": " << snapshot.sourcePhysicsFingerprint << ",\n"
        << "  \"device_program_fingerprint\": " << snapshot.deviceProgramFingerprint << ",\n"
        << "  \"rigid_world_fingerprint\": " << rigidWorldFingerprint << ",\n"
        << "  \"accepted_control_step\": " << snapshot.controlStep << ",\n"
        << "  \"simulation_time_s\": " << simulationTimeSeconds << ",\n"
        << "  \"rigid_mass_kg\": " << rigidMassKg << ",\n"
        << "  \"rigid_gravity_m_s2\": [" << gravityMPerS2[0] << ','
        << gravityMPerS2[1] << ',' << gravityMPerS2[2] << "],\n"
        << "  \"active_temporary_grip_nodes\": "
        << activeTemporaryClampNodeCount << ",\n"
        << "  \"native_rigid_contact_samples_total\": "
        << rigidContactSamples << ",\n"
        << "  \"native_support_floor_contact_samples\": "
        << supportFloorContactSamples << ",\n"
        << "  \"native_temporary_tool_contact_samples\": "
        << rigidContactSamples - supportFloorContactSamples << ",\n"
        << "  \"matter_accepted_contact_history_sha256\": \""
        << sha256Arena(snapshot.contactHistories) << "\",\n"
        << "  \"matter_accepted_deformable_contact_history_sha256\": \""
        << sha256Arena(snapshot.deformableContactHistories) << "\",\n"
        << "  \"matter_contact_histories_serialized_and_hash_checked\": true,\n"
        << "  \"metalworld_contact_impulse_warm_start_enabled\": false,\n"
        << "  \"temporary_tool_contact_free_gate_required\": "
        << (requireTemporaryToolContactFree ? "true" : "false") << ",\n"
        << "  \"released_settled_gate_passed\": " << (settled ? "true" : "false") << ",\n"
        << "  \"continuum_self_contact_and_cohesive_state\": \"retained only in the bound Matter checkpoint; not required to be zero\",\n"
        << "  \"scene_body_count\": " << sceneBodies.size() << ",\n"
        << "  \"scene_bodies\": [\n";
    for (std::size_t index = 0u; index < sceneBodies.size(); ++index) {
        const auto& body = sceneBodies[index];
        metadata << "    {\"index\":" << index << ",\"position\":["
            << body.position.x << ',' << body.position.y << ',' << body.position.z
            << ',' << body.position.w << "],\"orientation_xyzw\":["
            << body.orientation.x << ',' << body.orientation.y << ','
            << body.orientation.z << ',' << body.orientation.w
            << "],\"linear_velocity_inverse_mass\":["
            << body.linearVelocityAndInverseMass.x << ','
            << body.linearVelocityAndInverseMass.y << ','
            << body.linearVelocityAndInverseMass.z << ','
            << body.linearVelocityAndInverseMass.w << "],\"angular_velocity\":["
            << body.angularVelocity.x << ',' << body.angularVelocity.y << ','
            << body.angularVelocity.z << ',' << body.angularVelocity.w
            << "],\"inverse_inertia_world_rows\":[["
            << body.inverseInertiaWorldRow0.x << ',' << body.inverseInertiaWorldRow0.y
            << ',' << body.inverseInertiaWorldRow0.z << ',' << body.inverseInertiaWorldRow0.w
            << "],[" << body.inverseInertiaWorldRow1.x << ','
            << body.inverseInertiaWorldRow1.y << ',' << body.inverseInertiaWorldRow1.z
            << ',' << body.inverseInertiaWorldRow1.w << "],["
            << body.inverseInertiaWorldRow2.x << ',' << body.inverseInertiaWorldRow2.y
            << ',' << body.inverseInertiaWorldRow2.z << ',' << body.inverseInertiaWorldRow2.w
            << "]],\"flags_and_indices\":["
            << body.flagsAndIndices[0] << ',' << body.flagsAndIndices[1] << ','
            << body.flagsAndIndices[2] << ',' << body.flagsAndIndices[3] << "]}"
            << (index + 1u == sceneBodies.size() ? "\n" : ",\n");
    }
    metadata << "  ],\n  \"native_matter_rigid_states_sha256\": \""
        << sha256Arena(snapshot.rigidStates) << "\",\n"
        << "  \"native_matter_rigid_generalized_candidate_sha256\": \""
        << sha256Arena(snapshot.rigidGeneralizedCandidate) << "\",\n"
        << "  \"matter_accepted_contact_history_sha256\": \""
        << sha256Arena(snapshot.contactHistories) << "\",\n"
        << "  \"matter_accepted_deformable_contact_history_sha256\": \""
        << sha256Arena(snapshot.deformableContactHistories) << "\",\n"
        << "  \"metalworld_contact_impulse_warm_start_enabled\": false,\n"
        << "  \"qualification\": false\n}\n";
    metadata.flush();
    require(metadata.good(), "failed writing paired rigid scene checkpoint metadata");
    return result;
}
#endif

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

std::array<double, 3> parseMillimetreTriple(
    const std::string& value, const std::string& option) {
    std::array<double, 3> result{};
    std::istringstream input(value);
    std::string token;
    for (std::size_t component = 0u; component < result.size(); ++component) {
        require(static_cast<bool>(std::getline(input, token, ',')) &&
                    !token.empty(),
                option + " requires exactly three comma-separated millimetre values");
        result[component] = parseDouble(token, option) * 1.0e-3;
    }
    require(!std::getline(input, token, ','),
            option + " requires exactly three comma-separated millimetre values");
    for (const double component : result)
        require(std::isfinite(component) && component > 0.0,
                option + " values must be finite and positive");
    return result;
}

void printUsage() {
    std::cout
        << "usage: numi-matter-cardboard-probe --output DIR [options]\n"
        << "  --preset nagasawa2013|literature2009 (default literature2009)\n"
        << "  --upper-glue-gap-mm N --upper-bond-width-mm N  asymmetric glue (0 uses lower value)\n"
        << "  --residual-tolerance N tighten nonlinear residual threshold (default 1e-4)\n"
        << "  --reference-displacement-gradient  opt in to immutable-reference FEM arithmetic\n"
        << "  --scaled-paper-residuals  opt in to component-scaled 13-state Hill projection residuals\n"
        << "  --safeguarded-paper-initializer  opt in to safeguarded scalar Hill initial guesses for liner and medium\n"
        << "  --compensated-paper-arithmetic  opt in to compensated float2 Hill arithmetic for liner and medium\n"
        << "  --prescribed-tool-predictor  native feasible initial guess and swept node/punch certificate\n"
        << "  --prescribed-tool-barrier-scale N  punch IPC stiffness multiplier [1,100] (default 1; study arms 1,2,4)\n"
        << "  --box-blank           FEFCO0201 flat blank native rest check; requires --bend-angle 0\n"
        << "  --box-assembly-native execute score-die contact and accepted-state wall-fold grips; adhesive action fails closed\n"
        << "  --box-fixture-joint externally clamp the seam and folded end edges; remains unbonded/unsealed\n"
        << "  --box-angle-increment-deg N --box-max-action-steps N --box-score-steps N\n"
        << "  --box-shape-residual-calipers N --box-shape-coverage-min N  accepted-shape completion gate\n"
        << "  --box-fixture-gap-mm N --box-fixture-steps N  seam stand-off and native grip subdivisions\n"
        << "  --box-punch-radius-mm N --box-score-clearance-mm N  numerical tooling choices\n"
        << "  --shipping-box        300x200x200 mm FEFCO 0201 shipping blank (one environment)\n"
        << "  --shipping-inside-length-mm N --shipping-inside-height-mm N --shipping-inside-width-mm N\n"
        << "  --shipping-coupon-evidence FILE  accepted hashed score/fold/release allowance record\n"
        << "  --shipping-capacity-qualification FILE  source-bound mesh/time numerical evidence required before any Fpeak mass is instantiated\n"
        << "  --shipping-roof-measurement FILE  prior accepted-settled roof record used to bind impactor geometry before compile\n"
        << "  --shipping-impactor-half-extents-mm X,Y,Z  explicit compiled impactor BOX half-extents (default authored 1.2x nominal footprint)\n"
        << "  --shipping-engineering-assumptions --shipping-allowance-x-mm N --shipping-allowance-y-mm N --shipping-allowance-z-mm N\n"
        << "  --shipping-preflight-only  initialize native arenas and sample memory; do not step\n"
        << "  --compact-immutable-surfaces  use the compiled exterior-face map for immutable shipping FEM contact\n"
        << "  --deformable-contact-capacity N  explicit shipping contact-slot cap; overflow still rejects/rolls back (default legacy 8 per tet)\n"
        << "  --shipping-through-stage scoring|walls|manufacturer-lap|flaps|complete\n"
        << "  --shipping-score-depth-ratio N  one score depth per fresh run [0.1, 0.4] (default 0.25)\n"
        << "  --shipping-score-speed-mm-s N  commanded punch speed; default 50 mm/s engineering assumption\n"
        << "  --shipping-settle-steps N  held boundary settle steps after the selected stage\n"
        << "  --shipping-impact-dt-us N  finite-mass drop step (default 500 us)\n"
        << "  --shipping-impact-dt-refinement  independent 500/250/125 us drops from the same released checkpoint\n"
        << "  --shipping-gravity-transfer-dt-us N  finite-mass body-weight-transfer step (default 500 us)\n"
        << "  --starch-interface-parameters FILE  explicit unvalidated mixed-mode cohesive inputs\n"
        << "  --shipping-hma-parameters FILE  required explicit HMA analogue/engineering law for lap and flap bonds\n"
        << "  --liner-damage-parameters FILE --medium-damage-parameters FILE  complete explicit retained-matrix feasible sets; no defaults\n"
        << "  --crease              rounded punch indentation; grips held at zero rotation\n"
        << "  --pin-adhesion        native six-capsule analogue pin-adhesion coupon; calibration remains unqualified\n"
        << "  --pin-opening-mm N --pin-withdrawal-mm N  engineering fixture travel (defaults 2 and 1 mm)\n"
        << "  --indent-mm N --punch-radius-mm N --tool-clearance-mm N --anvil-clearance-mm N\n"
        << "  --without-self-contact  numerical control only for the finite-glue mesh\n"
        << "  --material liner=FILE --material medium=FILE\n"
        << "  --liner-material FILE --medium-material FILE\n"
        << "    ordinary literature2009 defaults use source-fit bilinear tension hardening; retained-matrix overlays default to ideal Hill bases\n"
        << "  --glue-material FILE --glue-gap-mm N --bond-width-mm N  finite glue bridges\n"
        << "  --without-medium       paired two-liner control geometry\n"
        << "  --steps N              ramp the end-grip rotation\n"
        << "  --hold-steps N --unload-steps N --relax-steps N  loading cycle phases\n"
        << "  --release-steps N      release right grip after preceding phases; left remains fixed\n"
        << "  --newton-iterations N  global nonlinear budget (1..64)\n"
        << "  --material-iterations N  local constitutive Newton budget (1..16)\n"
        << "  --fgmres-iterations N  Krylov-column budget (max "
        << NM_MIXED_FGMRES_MAX_ITERATIONS << ")\n"
        << "  --fgmres-restart N     Krylov basis restart size [1, "
        << NM_MIXED_FGMRES_RESTART << "] (default "
        << NM_MIXED_FGMRES_DEFAULT_RESTART << ")\n"
        << "  --line-search-steps N  nonlinear/contact trial steps [1, 32] (default 8)\n"
        << "  --accurate-paper-projection  opt-in strict Hill state projection; requires compensated arithmetic (pair with --scaled-paper-residuals in the comparison study)\n"
        << "  --compensated-implicit-projection  opt in to compensated residual evaluation for retained-matrix generic implicit damage overlays\n"
        << "  --fgmres-preconditioner scalar|regional-tangent-diagonal|regional-galerkin|regional-galerkin-block|regional-galerkin-contact\n"
        << "    regional-galerkin modes require authored compact constituent regions; selectors 2/3 are contact-free. Contact selector 4 uses the enriched 32-mode RAP/block estimate and admits one immutable explicit FEM object (<=8192 nodes, <=32768 tets, <=4096 cohesive faces), one environment, no dynamic/generalized rigid or multiphysics rows, and only static/kinematic non-articulated contact proxies\n"
        << "  --bend-angle DEGREES   final right-grip rotation\n"
        << "  --dt SECONDS           frame timestep\n"
        << "  --nx N                 cross-section subdivisions per flute pitch\n"
        << "  --ny N                 subdivisions across the width\n"
        << "  --thickness-slices N   elements through each paper layer\n"
        << "  --length-mm N --width-mm N --height-mm N --pitch-mm N\n"
        << "  --liner-thickness-mm N --medium-thickness-mm N\n"
        << "  --compile-only         write package and initial OBJ without creating a Metal runtime\n";
    std::cout << "  --metallib FILE         explicit retained native Metal image for replay\n";
}

void setMaterialPath(Arguments& arguments,
                     const std::string& assignment) {
    const std::size_t separator = assignment.find('=');
    require(separator != std::string::npos && separator > 0u &&
                separator + 1u < assignment.size(),
            "--material expects liner=FILE or medium=FILE");
    const std::string role = assignment.substr(0u, separator);
    const std::filesystem::path path = assignment.substr(separator + 1u);
    if (role == "liner") {
        arguments.linerMaterial = path;
        arguments.linerMaterialExplicit = true;
    } else if (role == "medium") {
        arguments.mediumMaterial = path;
        arguments.mediumMaterialExplicit = true;
    }
    else require(false, "unknown material role in --material: " + role);
}

Arguments parseArguments(const int argc, const char* argv[]) {
    Arguments arguments;
    for (int i = 1; i < argc; ++i)
        if (std::string_view(argv[i]) == "--pin-adhesion")
            arguments.pinAdhesion = true;
#if defined(NUMI_CARDBOARD_BOX_NATIVE_DRIVER)
    if (!arguments.pinAdhesion) {
        arguments.boxAssemblyNative = true;
        arguments.boxBlank = true;
    }
#endif
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
    if (arguments.pinAdhesion) {
        // The article gives 50 x 160 mm coupon dimensions but does not assign
        // MD/CD or dimensions for the pins. The existing cross-source recipe's
        // 4.21 mm board caliper and flute/material inputs are retained explicitly.
        arguments.environmentCount = 1u;
        arguments.lengthM = 0.050;
        arguments.widthM = 0.160;
        arguments.nxPerPitch = 8u;
        arguments.ny = 8u;
        arguments.thicknessSlices = 1u;
        arguments.timestepSeconds = 0.005;
        arguments.bendAngleDegrees = 0.0;
        arguments.steps = 1u;
        arguments.holdSteps = 0u;
        arguments.unloadSteps = 0u;
        arguments.relaxSteps = 0u;
        arguments.releaseSteps = 0u;
    }
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--box-blank") { arguments.boxBlank = true; continue; }
        if (option == "--box-assembly-native") {
            arguments.boxAssemblyNative = true;
            arguments.boxBlank = true;
            continue;
        }
        if (option == "--box-fixture-joint") {
            arguments.boxAssemblyNative = true;
            arguments.boxBlank = true;
            arguments.boxFixtureJoint = true;
            continue;
        }
        if (option == "--shipping-box") {
            arguments.shippingBox = true;
            arguments.boxBlank = true;
            continue;
        }
        if (option == "--shipping-preflight-only") {
            arguments.shippingBox = true;
            arguments.shippingPreflightOnly = true;
            arguments.boxBlank = true;
            continue;
        }
        if (option == "--compact-immutable-surfaces") {
            arguments.compactImmutableSurfaces = true;
            continue;
        }
        if (option == "--shipping-engineering-assumptions") {
            arguments.shippingEngineeringAssumptions = true;
            continue;
        }
        if (option == "--prescribed-tool-predictor") { arguments.prescribedToolPredictor = true; continue; }
        if (option == "--crease") { arguments.crease = true; continue; }
        if (option == "--reference-displacement-gradient") { arguments.referenceDisplacementGradient = true; continue; }
        if (option == "--scaled-paper-residuals") { arguments.scalePaperProjectionResiduals = true; continue; }
        if (option == "--safeguarded-paper-initializer") {
            arguments.safeguardedPaperInitializer = true;
            continue;
        }
        if (option == "--compensated-paper-arithmetic") {
            arguments.compensatedPaperArithmetic = true;
            continue;
        }
        if (option == "--accurate-paper-projection") {
            arguments.accuratePaperProjection = true;
            continue;
        }
        if (option == "--compensated-implicit-projection") {
            arguments.compensatedImplicitProjection = true;
            continue;
        }
        if (option == "--shipping-impact-dt-refinement") {
            arguments.shippingImpactTimestepRefinement = true;
            continue;
        }
        if (option == "--pin-adhesion") { arguments.pinAdhesion = true; continue; }
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
        else if (option == "--metallib") arguments.metallib = value;
        else if (option == "--shipping-coupon-evidence")
            arguments.shippingCouponEvidence = value;
        else if (option == "--shipping-capacity-qualification")
            arguments.shippingCapacityQualificationFile = value;
        else if (option == "--shipping-roof-measurement")
            arguments.shippingRoofMeasurementFile = value;
        else if (option == "--shipping-impactor-half-extents-mm") {
            arguments.shippingImpactorHalfExtentsM =
                parseMillimetreTriple(value, option);
            arguments.shippingImpactorHalfExtentsProvided = true;
            arguments.shippingImpactorGeometrySource = "explicit_cli_half_extents";
        }
        else if (option == "--shipping-inside-length-mm")
            arguments.shippingInsideM[0] = parseDouble(value, option) * 1.0e-3;
        else if (option == "--shipping-inside-height-mm")
            arguments.shippingInsideM[1] = parseDouble(value, option) * 1.0e-3;
        else if (option == "--shipping-inside-width-mm")
            arguments.shippingInsideM[2] = parseDouble(value, option) * 1.0e-3;
        else if (option == "--shipping-allowance-x-mm") {
            arguments.shippingAssumedAllowanceM[0] = parseDouble(value, option) * 1.0e-3;
            arguments.shippingAllowanceSet[0] = true;
        }
        else if (option == "--shipping-allowance-y-mm") {
            arguments.shippingAssumedAllowanceM[1] = parseDouble(value, option) * 1.0e-3;
            arguments.shippingAllowanceSet[1] = true;
        }
        else if (option == "--shipping-allowance-z-mm") {
            arguments.shippingAssumedAllowanceM[2] = parseDouble(value, option) * 1.0e-3;
            arguments.shippingAllowanceSet[2] = true;
        }
        else if (option == "--shipping-through-stage") {
            arguments.shippingRunThroughStage = value;
            arguments.shippingStageOptionProvided = true;
        }
        else if (option == "--shipping-settle-steps") {
            arguments.shippingSettleSteps = parseUnsigned(value, option);
            arguments.shippingSettleStepsProvided = true;
        }
        else if (option == "--shipping-impact-dt-us") {
            arguments.shippingImpactTimestepSeconds =
                parseDouble(value, option) * 1.0e-6;
            arguments.shippingImpactTimestepProvided = true;
        }
        else if (option == "--shipping-gravity-transfer-dt-us") {
            arguments.shippingGravityTransferTimestepSeconds =
                parseDouble(value, option) * 1.0e-6;
            arguments.shippingGravityTransferTimestepProvided = true;
        }
        else if (option == "--shipping-score-depth-ratio") {
            arguments.shippingScoreDepthRatio = parseDouble(value, option);
            arguments.shippingScoreDepthRatioProvided = true;
        }
        else if (option == "--shipping-score-speed-mm-s") {
            arguments.shippingScoreToolSpeedMPerS =
                parseDouble(value, option) * 1.0e-3;
            arguments.shippingScoreToolSpeedProvided = true;
        }
        else if (option == "--pin-opening-mm")
            arguments.pinMaximumOpeningM = parseDouble(value, option) * 1.0e-3;
        else if (option == "--pin-withdrawal-mm")
            arguments.pinWithdrawalDistanceM = parseDouble(value, option) * 1.0e-3;
        else if (option == "--deformable-contact-capacity") {
            arguments.deformableContactCapacity = parseUnsigned(value, option);
            arguments.deformableContactCapacityProvided = true;
        }
        else if (option == "--starch-interface-parameters")
            arguments.starchInterfaceParameters = value;
        else if (option == "--shipping-hma-parameters")
            arguments.shippingHmaParameters = value;
        else if (option == "--liner-damage-parameters")
            arguments.linerDamageParameters = value;
        else if (option == "--medium-damage-parameters")
            arguments.mediumDamageParameters = value;
        else if (option == "--indent-mm") arguments.indentationM = parseDouble(value, option) * 1e-3;
        else if (option == "--punch-radius-mm") arguments.punchRadiusM = parseDouble(value, option) * 1e-3;
        else if (option == "--tool-clearance-mm") arguments.toolClearanceM = parseDouble(value, option) * 1e-3;
        else if (option == "--anvil-clearance-mm") arguments.anvilClearanceM = parseDouble(value, option) * 1e-3;
        else if (option == "--box-angle-increment-deg")
            arguments.boxAngleIncrementDegrees = parseDouble(value, option);
        else if (option == "--box-shape-residual-calipers")
            arguments.boxShapeResidualBoundCalipers = parseDouble(value, option);
        else if (option == "--box-shape-coverage-min")
            arguments.boxShapeFitCoverageMinimum = parseDouble(value, option);
        else if (option == "--box-max-action-steps")
            arguments.boxMaximumStepsPerAction = parseUnsigned(value, option);
        else if (option == "--box-score-steps")
            arguments.boxScoreStepsPerLeg = parseUnsigned(value, option);
        else if (option == "--box-punch-radius-mm")
            arguments.boxPunchRadiusM = parseDouble(value, option) * 1.0e-3;
        else if (option == "--box-score-clearance-mm")
            arguments.boxScoreToolClearanceM = parseDouble(value, option) * 1.0e-3;
        else if (option == "--box-fixture-gap-mm")
            arguments.boxFixtureGapM = parseDouble(value, option) * 1.0e-3;
        else if (option == "--box-fixture-steps")
            arguments.boxFixtureSteps = parseUnsigned(value, option);
        else if (option == "--residual-tolerance") arguments.relativeResidualTolerance = parseDouble(value, option);
        else if (option == "--prescribed-tool-barrier-scale")
            arguments.prescribedToolBarrierStiffnessScale = parseDouble(value, option);
        else if (option == "--newton-iterations") arguments.newtonIterations = parseUnsigned(value, option);
        else if (option == "--material-iterations") arguments.localMaterialIterations = parseUnsigned(value, option);
        else if (option == "--preset") arguments.preset = value;
        else if (option == "--material") setMaterialPath(arguments, value);
        else if (option == "--liner-material") {
            arguments.linerMaterial = value;
            arguments.linerMaterialExplicit = true;
        }
        else if (option == "--medium-material") {
            arguments.mediumMaterial = value;
            arguments.mediumMaterialExplicit = true;
        }
        else if (option == "--steps") arguments.steps = parseUnsigned(value, option);
        else if (option == "--upper-glue-gap-mm") arguments.upperGlueGapM = parseDouble(value, option) * 1e-3;
        else if (option == "--upper-bond-width-mm") arguments.upperBondWidthM = parseDouble(value, option) * 1e-3;
        else if (option == "--glue-material") arguments.glueMaterial = value;
        else if (option == "--glue-gap-mm") arguments.glueGapM = parseDouble(value, option) * 1e-3;
        else if (option == "--bond-width-mm") arguments.bondWidthM = parseDouble(value, option) * 1e-3;
        else if (option == "--hold-steps") arguments.holdSteps = parseUnsigned(value, option);
        else if (option == "--unload-steps") arguments.unloadSteps = parseUnsigned(value, option);
        else if (option == "--relax-steps") arguments.relaxSteps = parseUnsigned(value, option);
        else if (option == "--release-steps") arguments.releaseSteps = parseUnsigned(value, option);
        else if (option == "--fgmres-iterations")
            arguments.fgmresIterations = parseUnsigned(value, option);
        else if (option == "--fgmres-restart") {
            arguments.fgmresRestart = parseUnsigned(value, option);
            arguments.fgmresRestartExplicit = true;
        }
        else if (option == "--line-search-steps")
            arguments.lineSearchSteps = parseUnsigned(value, option);
        else if (option == "--fgmres-preconditioner") {
            if (value == "scalar")
                arguments.femPreconditioner =
                    FEMPreconditionerMode::scalarDiagonal;
            else if (value == "regional-tangent-diagonal")
                arguments.femPreconditioner =
                    FEMPreconditionerMode::regionalTangentDiagonal;
            else if (value == "regional-galerkin")
                arguments.femPreconditioner =
                    FEMPreconditionerMode::regionalGalerkin;
            else if (value == "regional-galerkin-block")
                arguments.femPreconditioner =
                    FEMPreconditionerMode::regionalGalerkinBlockJacobi;
            else if (value == "regional-galerkin-contact")
                arguments.femPreconditioner =
                    FEMPreconditionerMode::regionalGalerkinContact;
            else
                require(false, "--fgmres-preconditioner expects scalar or "
                               "regional-tangent-diagonal or regional-galerkin or regional-galerkin-block or regional-galerkin-contact");
        }
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

    const bool allShippingOffsets = std::all_of(
        arguments.shippingAllowanceSet.begin(),
        arguments.shippingAllowanceSet.end(), [](const bool value) { return value; });
    const bool anyShippingOffsets = std::any_of(
        arguments.shippingAllowanceSet.begin(),
        arguments.shippingAllowanceSet.end(), [](const bool value) { return value; });
    require(!anyShippingOffsets || allShippingOffsets,
            "shipping engineering allowance requires X, Y, and Z values");
    require(!arguments.shippingEngineeringAssumptions || allShippingOffsets,
            "shipping engineering assumptions require explicit X, Y, and Z allowances");
    require(arguments.shippingEngineeringAssumptions || !anyShippingOffsets,
            "explicit shipping allowances require --shipping-engineering-assumptions");
    require(arguments.shippingCouponEvidence.empty() ||
                !arguments.shippingEngineeringAssumptions,
            "choose accepted coupon evidence or engineering allowance assumptions, not both");
    const bool hasLinerDamageParameters =
        !arguments.linerDamageParameters.empty();
    const bool hasMediumDamageParameters =
        !arguments.mediumDamageParameters.empty();
    require(hasLinerDamageParameters == hasMediumDamageParameters,
            "retained-matrix damage requires explicit liner and medium parameter files together");
    const auto defaultPaths = numi::cardboard::paper_materials::selectPaths(
        NUMI_CARDBOARD_MATERIAL_DIR, arguments.preset,
        hasLinerDamageParameters,
        arguments.linerMaterialExplicit ? arguments.linerMaterial : std::filesystem::path{},
        arguments.mediumMaterialExplicit ? arguments.mediumMaterial : std::filesystem::path{});
    arguments.linerMaterial = defaultPaths.liner;
    arguments.mediumMaterial = defaultPaths.medium;
    require(!arguments.compensatedImplicitProjection ||
                (hasLinerDamageParameters &&
                 arguments.preset == "literature2009" &&
                 !arguments.withoutMedium),
            "compensated implicit projection requires explicit retained-matrix damage sets for both literature2009 constituents");
    require(!arguments.accuratePaperProjection ||
                (arguments.preset == "literature2009" &&
                 arguments.compensatedPaperArithmetic),
            "accurate paper projection requires the literature2009 recipe with --compensated-paper-arithmetic");
    require(!hasLinerDamageParameters ||
                (arguments.preset == "literature2009" && !arguments.withoutMedium),
            "retained-matrix damage overlays require the retained literature2009 liner and medium recipe");

    if (!arguments.starchInterfaceParameters.empty()) {
        std::ifstream parametersFile(arguments.starchInterfaceParameters);
        require(parametersFile.good(), "cannot open starch interface parameters: " +
                    arguments.starchInterfaceParameters.string());
        arguments.shippingStarchLaw =
            numi::cardboard::parseShippingStarchLawParameters(
                parametersFile, arguments.starchInterfaceParameters.string());
        arguments.starchInterfaceParametersSha256 = hexDigest(
            sha256File(arguments.starchInterfaceParameters));
    }
    if (!arguments.shippingHmaParameters.empty()) {
        std::ifstream parametersFile(arguments.shippingHmaParameters);
        require(parametersFile.good(), "cannot open shipping HMA parameters: " +
                    arguments.shippingHmaParameters.string());
        arguments.shippingHmaLaw =
            numi::cardboard::parseShippingHmaLawParameters(
                parametersFile, arguments.shippingHmaParameters.string());
        arguments.shippingHmaParametersSha256 = hexDigest(
            sha256File(arguments.shippingHmaParameters));
    }
    if (!arguments.linerDamageParameters.empty()) {
        std::ifstream inputFile(arguments.linerDamageParameters, std::ios::binary);
        require(inputFile.good(), "cannot open liner damage parameters: " +
                    arguments.linerDamageParameters.string());
        std::ostringstream contentStream;
        contentStream << inputFile.rdbuf();
        require(inputFile.good() || inputFile.eof(),
                "failed reading liner damage parameters: " +
                    arguments.linerDamageParameters.string());
        const std::string contents = contentStream.str();
        arguments.linerDamageParametersSha256 = hexDigest(sha256(contents));
        std::istringstream parametersFile(contents);
        arguments.linerDamageSet =
            numi::cardboard::retained_damage::parseExplicitDamageSet(
                parametersFile, arguments.linerDamageParametersSha256,
                arguments.linerDamageParameters.string());
    }
    if (!arguments.mediumDamageParameters.empty()) {
        std::ifstream inputFile(arguments.mediumDamageParameters, std::ios::binary);
        require(inputFile.good(), "cannot open medium damage parameters: " +
                    arguments.mediumDamageParameters.string());
        std::ostringstream contentStream;
        contentStream << inputFile.rdbuf();
        require(inputFile.good() || inputFile.eof(),
                "failed reading medium damage parameters: " +
                    arguments.mediumDamageParameters.string());
        const std::string contents = contentStream.str();
        arguments.mediumDamageParametersSha256 = hexDigest(sha256(contents));
        std::istringstream parametersFile(contents);
        arguments.mediumDamageSet =
            numi::cardboard::retained_damage::parseExplicitDamageSet(
                parametersFile, arguments.mediumDamageParametersSha256,
                arguments.mediumDamageParameters.string());
    }

    if (arguments.shippingBox) {
        require(arguments.preset == "literature2009",
                "shipping box currently requires the explicit literature2009 cross-source recipe");
        arguments.environmentCount = 1u;
        arguments.boxBlank = true;
        arguments.boxAssemblyNative = !arguments.shippingPreflightOnly;
        arguments.bendAngleDegrees = 0.0;
        arguments.referenceDisplacementGradient = true;
        arguments.boxPunchRadiusM = 1.0e-3;
        arguments.boxScoreToolClearanceM = 0.25 * 4.210e-3;
        numi::cardboard::ShippingBoxDesignDefaults defaults;
        defaults.inside = {arguments.shippingInsideM[0],
                           arguments.shippingInsideM[1],
                           arguments.shippingInsideM[2]};
        defaults.boardRecipeId = "literature2009";
        defaults.subdivisionsPerBand = 16u;

        std::optional<numi::cardboard::ShippingCouponDocument> coupon;
        if (!arguments.shippingCouponEvidence.empty()) {
            std::ifstream evidenceFile(arguments.shippingCouponEvidence);
            require(evidenceFile.good(), "cannot open shipping coupon evidence: " +
                        arguments.shippingCouponEvidence.string());
            coupon = numi::cardboard::parseShippingCouponDocument(
                evidenceFile, arguments.shippingCouponEvidence.string());
            arguments.shippingCouponDamageLawQualified =
                coupon->scoringDamageLawQualified;
            arguments.shippingCouponEvidenceSha256 = hexDigest(
                sha256File(arguments.shippingCouponEvidence));
            std::filesystem::path measurementPath = coupon->measurementFile;
            if (measurementPath.is_relative())
                measurementPath = arguments.shippingCouponEvidence.parent_path() /
                                  measurementPath;
            require(std::filesystem::is_regular_file(measurementPath),
                    "coupon evidence measurement_file is not a regular file");
            const std::string measuredSha = hexDigest(sha256File(measurementPath));
            require(measuredSha == coupon->coupon.measurementSha256,
                    "coupon evidence measurement SHA-256 does not match its file");
            require(coupon->coupon.boardRecipeId == defaults.boardRecipeId,
                    "coupon evidence recipe does not match the selected shipping recipe");
            arguments.shippingMeasurementFile = std::filesystem::weakly_canonical(
                measurementPath);
        }
        const std::array<double, 3>* assumed =
            arguments.shippingEngineeringAssumptions
                ? &arguments.shippingAssumedAllowanceM : nullptr;
        const auto selected = numi::cardboard::selectShippingDesign(
            defaults, coupon ? &*coupon : nullptr, assumed,
            arguments.shippingPreflightOnly || arguments.compileOnly);
        arguments.shippingDesign = selected.design;
        arguments.boxConfig = selected.design.blankConfig;
        if (!arguments.shippingImpactorHalfExtentsProvided) {
            arguments.shippingImpactorHalfExtentsM =
                numi::cardboard::shippingAuthoredNominalImpactorHalfExtents(
                    selected.design.blankConfig.panelLength,
                    selected.design.blankConfig.panelWidth,
                    0.025);
        }
        arguments.lengthM = selected.design.perimeterSpanM;
        arguments.widthM = selected.design.blankHeightM;
        arguments.totalHeightM = 4.210e-3;
        arguments.pitchM = 7.9e-3;
        arguments.linerThicknessM = 0.277e-3;
        arguments.mediumThicknessM = 0.191e-3;
        arguments.glueGapM = 0.100e-3;
        arguments.bondWidthM = 0.800e-3;
        arguments.upperGlueGapM = 0.030e-3;
        arguments.upperBondWidthM = 0.600e-3;
        arguments.nxPerPitch = 8u;
        arguments.thicknessSlices = 1u;
        arguments.fgmresIterations = 1024u;
        if (!arguments.fgmresRestartExplicit)
            arguments.fgmresRestart = NM_MIXED_FGMRES_RESTART;
        arguments.localMaterialIterations = 16u;
        arguments.newtonIterations = 14u;
        arguments.timestepSeconds = 25.0e-6;
        arguments.steps = 96u;
        arguments.holdSteps = 0u;
        arguments.unloadSteps = 0u;
        arguments.relaxSteps = 0u;
        arguments.releaseSteps = 0u;
        require(arguments.shippingRunThroughStage == "scoring" ||
                    arguments.shippingRunThroughStage == "walls" ||
                    arguments.shippingRunThroughStage == "manufacturer-lap" ||
                    arguments.shippingRunThroughStage == "flaps" ||
                    arguments.shippingRunThroughStage == "complete",
                "unsupported shipping stage cutoff");
        require(std::isfinite(arguments.shippingScoreDepthRatio) &&
                    arguments.shippingScoreDepthRatio >= 0.1 &&
                    arguments.shippingScoreDepthRatio <= 0.4,
                "shipping score depth ratio must lie in [0.1, 0.4] caliper");
        require(std::isfinite(arguments.shippingScoreToolSpeedMPerS) &&
                    arguments.shippingScoreToolSpeedMPerS > 0.0 &&
                    arguments.shippingScoreToolSpeedMPerS <= 1.0,
                "shipping score speed must lie in (0, 1000] mm/s");
        const bool includesAdhesiveOperations =
            arguments.shippingRunThroughStage == "manufacturer-lap" ||
            arguments.shippingRunThroughStage == "flaps" ||
            arguments.shippingRunThroughStage == "complete";
        require(!includesAdhesiveOperations || arguments.shippingHmaLaw.has_value() ||
                    arguments.shippingPreflightOnly || arguments.compileOnly,
                "manufacturer lap and flap closures require an explicit --shipping-hma-parameters file");
        require(arguments.shippingSettleSteps > 0u &&
                    arguments.shippingSettleSteps <= 4096u,
                "shipping settle steps must lie in [1, 4096]");
        require(std::isfinite(arguments.shippingImpactTimestepSeconds) &&
                    arguments.shippingImpactTimestepSeconds >= 1.0e-6 &&
                    arguments.shippingImpactTimestepSeconds <= 5.0e-3 &&
                    std::ceil(0.250 / arguments.shippingImpactTimestepSeconds) <=
                        1000000.0,
                "shipping impact timestep must lie in [1 us, 5 ms] with at most one million steps");
        require(!arguments.shippingImpactTimestepRefinement ||
                    (!arguments.shippingImpactTimestepProvided &&
                     arguments.shippingImpactTimestepSeconds == 0.0005),
                "the fixed 500/250/125 us impact-refinement series cannot be combined with --shipping-impact-dt-us");
        require(std::isfinite(arguments.shippingGravityTransferTimestepSeconds) &&
                    arguments.shippingGravityTransferTimestepSeconds >= 1.0e-6 &&
                    arguments.shippingGravityTransferTimestepSeconds <= 5.0e-3 &&
                    std::ceil(0.5 /
                        arguments.shippingGravityTransferTimestepSeconds) <=
                        1000000.0,
                "shipping gravity-transfer timestep must lie in [1 us, 5 ms] with at most one million ramp steps");
        require(arguments.shippingPreflightOnly || arguments.compileOnly ||
                    coupon.has_value() ||
                    arguments.shippingEngineeringAssumptions,
                "native shipping assembly requires accepted allowance coupon evidence or explicit engineering assumptions");
        require(!arguments.boxFixtureJoint,
                "the externally held mini-box fixture is not a manufacturer lap or shipping closure model");
        require(!arguments.shippingStarchLaw ||
                    (!arguments.glueMaterial.empty() && !arguments.withoutMedium),
                "starch cohesive parameters require finite glue and the medium region");
        if (arguments.shippingStarchLaw)
            arguments.starchInterfaceParameters =
                std::filesystem::weakly_canonical(arguments.starchInterfaceParameters);
    }

    if (arguments.boxAssemblyNative)
        arguments.bendAngleDegrees = 0.0;
    if (arguments.boxBlank) {
        require(arguments.preset == "literature2009" && !arguments.withoutMedium &&
                !arguments.crease && arguments.releaseSteps == 0u && arguments.bendAngleDegrees == 0.0,
                "box blank mode requires literature2009, medium, zero bend and no crease/release");
        if (!arguments.shippingBox) {
            arguments.boxConfig = {4.0 * arguments.pitchM, 3.0 * arguments.pitchM,
                4.0 * arguments.pitchM, arguments.pitchM, 0.2 * arguments.pitchM, arguments.ny};
            arguments.lengthM = 15.0 * arguments.pitchM;
            arguments.widthM = 7.0 * arguments.pitchM;
        }
        arguments.referenceDisplacementGradient = true;
    }

    require(!arguments.shippingScoreDepthRatioProvided || arguments.shippingBox,
            "--shipping-score-depth-ratio is only meaningful with --shipping-box");
    require(!arguments.shippingScoreToolSpeedProvided || arguments.shippingBox,
            "--shipping-score-speed-mm-s is only meaningful with --shipping-box");
    require(!arguments.compactImmutableSurfaces || arguments.shippingBox,
            "--compact-immutable-surfaces is currently supported only for the immutable shipping FEM");
    require(!arguments.deformableContactCapacityProvided || arguments.shippingBox,
            "--deformable-contact-capacity is currently supported only for the shipping FEM");
    require(!arguments.deformableContactCapacityProvided ||
                arguments.deformableContactCapacity > 0u,
            "--deformable-contact-capacity must be positive; omit it for the legacy derived capacity");
    require(arguments.shippingBox || arguments.shippingRoofMeasurementFile.empty(),
            "--shipping-roof-measurement requires --shipping-box");
    require(arguments.shippingBox || !arguments.shippingImpactorHalfExtentsProvided,
            "--shipping-impactor-half-extents-mm requires --shipping-box");
    if (arguments.shippingBox) {
        for (const double extent : arguments.shippingImpactorHalfExtentsM)
            require(std::isfinite(extent) && extent > 0.0,
                    "shipping impactor half-extents must be finite and positive");
        require(arguments.shippingRoofMeasurementFile.empty() ||
                    arguments.shippingImpactorHalfExtentsProvided,
                "a source-bound --shipping-roof-measurement requires explicit --shipping-impactor-half-extents-mm X,Y,Z");
    }

    if (arguments.pinAdhesion) {
        require(arguments.preset == "literature2009" &&
                    !arguments.shippingBox && !arguments.boxBlank &&
                    !arguments.boxAssemblyNative && !arguments.boxFixtureJoint &&
                    !arguments.crease && !arguments.withoutMedium,
                "pin adhesion requires the finite-glue literature2009 strip recipe and cannot be combined with box/crease modes");
        require(arguments.environmentCount == 1u,
                "pin adhesion requires exactly one coupon environment");
        require(std::abs(arguments.lengthM - 0.050) <= 1.0e-12 &&
                    std::abs(arguments.widthM - 0.160) <= 1.0e-12,
                "pin adhesion source coupon dimensions are fixed at 50 x 160 mm; MD/CD orientation remains unspecified");
        require(std::isfinite(arguments.totalHeightM) &&
                    arguments.totalHeightM > 0.0 &&
                    std::isfinite(arguments.pinMaximumOpeningM) &&
                    arguments.pinMaximumOpeningM > 0.0 &&
                    std::isfinite(arguments.pinWithdrawalDistanceM) &&
                    arguments.pinWithdrawalDistanceM > 0.0,
                "pin coupon caliper, opening, and withdrawal must be finite and positive");
        require(!arguments.disableSelfContact,
                "pin adhesion requires the authored exposed-surface contact path; --without-self-contact is not supported");
    }

    require(!arguments.prescribedToolPredictor || arguments.crease, "prescribed tool predictor requires crease mode");
    require(std::isfinite(arguments.prescribedToolBarrierStiffnessScale) &&
                arguments.prescribedToolBarrierStiffnessScale >= 1.0 &&
                arguments.prescribedToolBarrierStiffnessScale <= 100.0,
            "prescribed tool barrier scale must be finite and in [1, 100]");
    require(arguments.prescribedToolBarrierStiffnessScale == 1.0 ||
                arguments.prescribedToolPredictor,
            "nondefault prescribed tool barrier scale requires --prescribed-tool-predictor");
    if (arguments.crease) {
        require(arguments.releaseSteps == 0u, "crease mode does not release end grips");
        require(arguments.indentationM >= 0.0 && arguments.indentationM < arguments.totalHeightM &&
                arguments.punchRadiusM > 0.0 && arguments.toolClearanceM > 0.0 && arguments.anvilClearanceM > 0.0,
                "invalid tooling travel, radius or clearance");
        arguments.bendAngleDegrees = 0.0;
    }
    require(arguments.boxScoreStepsPerLeg > 0u &&
                arguments.boxScoreStepsPerLeg <= 4096u,
            "--box-score-steps must lie in [1, 4096]");
    require(arguments.boxMaximumStepsPerAction > 0u &&
                arguments.boxMaximumStepsPerAction <= 100000u,
            "--box-max-action-steps must lie in [1, 100000]");
    require(std::isfinite(arguments.boxAngleIncrementDegrees) &&
                arguments.boxAngleIncrementDegrees > 0.0 &&
                arguments.boxAngleIncrementDegrees <= 10.0,
            "--box-angle-increment-deg must lie in (0, 10]");
    require(std::isfinite(arguments.boxShapeResidualBoundCalipers) &&
                arguments.boxShapeResidualBoundCalipers > 0.0 &&
                arguments.boxShapeResidualBoundCalipers <= 10.0,
            "--box-shape-residual-calipers must lie in (0, 10]");
    require(std::isfinite(arguments.boxShapeFitCoverageMinimum) &&
                arguments.boxShapeFitCoverageMinimum > 0.0 &&
                arguments.boxShapeFitCoverageMinimum <= 1.0,
            "--box-shape-coverage-min must lie in (0, 1]");
    require(std::isfinite(arguments.boxPunchRadiusM) &&
                arguments.boxPunchRadiusM > 0.0 &&
                std::isfinite(arguments.boxScoreToolClearanceM) &&
                arguments.boxScoreToolClearanceM > 0.0,
            "box score tooling radius and clearance must be positive");
    require(std::isfinite(arguments.boxFixtureGapM) &&
                arguments.boxFixtureGapM > 0.0 &&
                arguments.boxFixtureGapM <= 0.01,
            "box fixture stand-off must be finite and in (0, 10] mm");
    require(arguments.boxFixtureSteps > 0u &&
                arguments.boxFixtureSteps <= 4096u,
            "--box-fixture-steps must lie in [1, 4096]");
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
                arguments.relaxSteps <= 10000u && arguments.releaseSteps <= 10000u, "cycle phase exceeds 10000 steps");
    require(arguments.relaxSteps == 0u || arguments.unloadSteps > 0u,
            "relaxation requires an unloading phase; no instantaneous release");
    require(arguments.fgmresIterations >= NM_MIXED_FGMRES_DEFAULT_RESTART &&
                arguments.fgmresIterations <= NM_MIXED_FGMRES_MAX_ITERATIONS,
            "--fgmres-iterations must lie between the compiled restart size and native operational ceiling");
    require(arguments.fgmresRestart >= 1u &&
                arguments.fgmresRestart <= NM_MIXED_FGMRES_RESTART,
            "--fgmres-restart must lie in [1, 16]");
    require(arguments.fgmresIterations >= arguments.fgmresRestart,
            "--fgmres-iterations must be at least --fgmres-restart");
    require(arguments.lineSearchSteps >= 1u &&
                arguments.lineSearchSteps <= 32u,
            "--line-search-steps must lie in [1, 32]");
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
    require(arguments.relativeResidualTolerance > 0.0 && arguments.relativeResidualTolerance <= 1.0e-4,
            "--residual-tolerance must lie in (0, 1e-4]; this probe permits tightening only");
    require(arguments.timestepSeconds > 0.0 &&
                arguments.timestepSeconds <= 0.01,
            "--dt must lie in (0, 0.01] seconds");
    require(arguments.lengthM > 0.0 && arguments.widthM > 0.0 &&
                arguments.totalHeightM > 0.0 && arguments.pitchM > 0.0 &&
                arguments.linerThicknessM > 0.0 &&
                arguments.mediumThicknessM > 0.0,
            "all geometry dimensions must be positive");
    if (!arguments.shippingBox && !arguments.pinAdhesion) {
        const double pitchCount = arguments.lengthM / arguments.pitchM;
        require(std::abs(pitchCount - std::round(pitchCount)) < 1.0e-9 &&
                    pitchCount >= 1.0 && pitchCount <= 32.0,
                "length must be an integer number of pitches in [1, 32]");
    }
    require(arguments.totalHeightM - 2.0 * arguments.linerThicknessM >
                arguments.mediumThicknessM,
            "board height leaves no positive corrugated-medium clearance");
    if (!arguments.shippingBox) {
        const double pitchCount = arguments.lengthM / arguments.pitchM;
        const double estimatedIntervals =
            pitchCount * static_cast<double>(arguments.nxPerPitch);
        const double estimatedTetrahedra = estimatedIntervals * 36.0 *
            static_cast<double>(arguments.ny) *
            static_cast<double>(arguments.thicknessSlices);
        require(estimatedTetrahedra <= 250000.0,
                "requested mesh exceeds the bounded 250000-tetrahedron probe limit");
    }
    require(arguments.shippingBox ||
                (arguments.shippingCouponEvidence.empty() &&
                 arguments.shippingCapacityQualificationFile.empty() &&
                 !arguments.shippingEngineeringAssumptions &&
                 !arguments.shippingPreflightOnly &&
                 !arguments.shippingStageOptionProvided &&
                 arguments.shippingHmaParameters.empty() &&
                 !arguments.shippingSettleStepsProvided),
            "shipping allowance, stage, or settle options require --shipping-box");
    require(!arguments.shippingStarchLaw ||
                (!arguments.glueMaterial.empty() &&
                 !arguments.withoutMedium &&
                 arguments.referenceDisplacementGradient),
            "native starch interfaces require finite glue, medium, and immutable reference-displacement FEM");
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
    result.world.compactImmutableFEMSurfaces =
        arguments.compactImmutableSurfaces;
    if (glueMaterial != nullptr) {
        requireSameInterface(linerMaterial, *glueMaterial);
        const auto glued = numi::cardboard::buildConformingGlueCrossSection({
            .length = arguments.lengthM, .pitch = arguments.pitchM,
            .caliper = arguments.totalHeightM, .linerThickness = arguments.linerThicknessM,
            .mediumThickness = arguments.mediumThicknessM, .glueMinimumGap = arguments.glueGapM,
            .bondWidth = arguments.bondWidthM,
            .upperGlueMinimumGap = arguments.upperGlueGapM, .upperBondWidth = arguments.upperBondWidthM,
            .nxPerPitch = arguments.nxPerPitch, .thicknessSlices = arguments.thicknessSlices,
            .allowPartialTerminalPitch = arguments.shippingBox ||
                                         arguments.pinAdhesion});
        std::vector<std::uint32_t> remap;
        for (const auto& point : glued.points) remap.push_back(crossSection.point(point.x, point.z));
        for (const auto& cell : glued.triangles)
            crossSection.triangle({remap.at(cell.points[0]), remap.at(cell.points[1]), remap.at(cell.points[2])},
                static_cast<std::uint32_t>(cell.material), cell.frameAngle);
        result.glueFootprints = glued.glueFootprints;
        if (arguments.boxBlank)
            result.boxBlank = numi::cardboard::buildFEFCO0201Blank(arguments.boxConfig, glued);
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

    if (result.boxBlank) {
        result.restNodes.clear();
        result.tetrahedra.clear();
        for (const auto& point : result.boxBlank->points)
            result.restNodes.push_back({point.x, point.y, point.z});
        for (const auto& tet : result.boxBlank->tetrahedra)
            result.tetrahedra.push_back({tet.nodes, static_cast<std::uint32_t>(tet.material), tet.frameAngle});
    }

    if (arguments.shippingStarchLaw) {
        require(glueMaterial != nullptr && !arguments.withoutMedium,
                "starch cohesive splitting requires the finite glue layer and medium");
        numi::cardboard::StarchInterfaceMesh source;
        source.referenceNodesM.reserve(result.restNodes.size());
        for (const Vec3& node : result.restNodes)
            source.referenceNodesM.push_back({node.x, node.y, node.z});
        source.tetrahedra.reserve(result.tetrahedra.size());
        for (const TetMetadata& tet : result.tetrahedra) {
            numi::cardboard::StarchInterfaceTet prepared;
            for (std::size_t slot = 0u; slot < 4u; ++slot)
                prepared.nodeIds[slot] = tet.nodes[slot];
            require(tet.materialIndex <= numi::cardboard::kStarchMaterialGlue,
                    "starch interface mesh has an unsupported cardboard region");
            prepared.materialIndex = static_cast<std::uint8_t>(tet.materialIndex);
            prepared.densityKgPerM3 = materialDensity(
                tet.materialIndex == 0u ? linerMaterial :
                tet.materialIndex == 1u ? mediumMaterial : *glueMaterial);
            const auto quaternion = frameQuaternion(tet.frameAngle);
            const double qx = quaternion[0];
            const double qy = quaternion[1];
            const double qz = quaternion[2];
            const double qw = quaternion[3];
            prepared.materialFrameToReference = {
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qw * qz),
                2.0 * (qx * qz + qw * qy),
                2.0 * (qx * qy + qw * qz),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qw * qx),
                2.0 * (qx * qz - qw * qy),
                2.0 * (qy * qz + qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            };
            source.tetrahedra.push_back(prepared);
        }
        const auto split = numi::cardboard::prepareStarchInterfaceMesh(source);
        require(!split.pairedFaces.empty(),
                "explicit starch law parameters found no conforming paper/glue faces");
        std::vector<double> originalFrameAngles;
        originalFrameAngles.reserve(result.tetrahedra.size());
        for (const TetMetadata& tet : result.tetrahedra)
            originalFrameAngles.push_back(tet.frameAngle);
        require(split.tetrahedra.size() == originalFrameAngles.size(),
                "starch node split changed tetrahedron ordering or count");
        result.restNodes.clear();
        result.restNodes.reserve(split.referenceNodesM.size());
        for (const auto point : split.referenceNodesM)
            result.restNodes.push_back({point.x, point.y, point.z});
        result.tetrahedra.clear();
        result.tetrahedra.reserve(split.tetrahedra.size());
        for (std::size_t index = 0u; index < split.tetrahedra.size(); ++index) {
            const auto& tet = split.tetrahedra[index];
            TetMetadata converted;
            for (std::size_t slot = 0u; slot < 4u; ++slot)
                converted.nodes[slot] = static_cast<std::uint32_t>(tet.nodeIds[slot]);
            converted.materialIndex = tet.materialIndex;
            converted.frameAngle = originalFrameAngles[index];
            result.tetrahedra.push_back(converted);
        }
        result.starchPairedFaces = split.pairedFaces;
        result.starchInterfaceParameters = *arguments.shippingStarchLaw;
    }

    ObjectSource object;
    object.name = arguments.pinAdhesion
        ? "cardboard_johnson_popil_analogue_pin_adhesion_coupon"
        : arguments.boxBlank ? "cardboard_fefco0201_unscored_flat_blank" : arguments.withoutMedium
        ? "cardboard_two_liner_bending_control"
        : "cardboard_explicit_sinusoidal_corrugated_strip";
    object.materialIndex = 0u;
    object.representation = Representation::fem;
    object.mixedFEM = false;
    object.adaptive = false;
    if (arguments.deformableContactCapacityProvided)
        object.femCapacity.deformableContacts =
            arguments.deformableContactCapacity;
    object.deformableContact = glueMaterial != nullptr && !arguments.disableSelfContact;
    object.deformableSelfContact = object.deformableContact &&
        !arguments.shippingStarchLaw.has_value();
    object.characteristicLength = std::min(
        arguments.linerThicknessM, arguments.mediumThicknessM) /
        static_cast<double>(arguments.thicknessSlices);
    object.femNodes.reserve(result.restNodes.size());
    for (const Vec3& point : result.restNodes)
        object.femNodes.push_back({point.x, point.y, point.z});
    object.tetrahedra.reserve(result.tetrahedra.size());
    result.materialIndices.reserve(result.tetrahedra.size());
    result.authoredRegionIds.reserve(result.tetrahedra.size());
    result.materialFrames.reserve(result.tetrahedra.size());
    for (const TetMetadata& tetrahedron : result.tetrahedra) {
        object.tetrahedra.push_back({tetrahedron.nodes});
        result.materialIndices.push_back(tetrahedron.materialIndex);
        require(tetrahedron.materialIndex <= 2u,
                "cardboard tetrahedron has an unsupported authored constituent role");
        result.authoredRegionIds.push_back(tetrahedron.materialIndex);
        result.materialFrames.push_back(frameQuaternion(tetrahedron.frameAngle));
    }
    require(result.authoredRegionIds.size() == object.tetrahedra.size(),
            "authored cardboard constituent map does not cover every tetrahedron");
    if (arguments.femPreconditioner ==
            FEMPreconditionerMode::regionalGalerkin ||
        arguments.femPreconditioner ==
            FEMPreconditionerMode::regionalGalerkinBlockJacobi ||
        arguments.femPreconditioner ==
            FEMPreconditionerMode::regionalGalerkinContact)
        object.femPreconditionerRegionIds = result.authoredRegionIds;
    std::ostringstream authoredRegionSource;
    authoredRegionSource << "numi.cardboard.authored-constituent-region-map.v1\n";
    for (std::size_t index = 0u; index < result.authoredRegionIds.size(); ++index)
        authoredRegionSource << index << ':' << result.authoredRegionIds[index] << '\n';
    result.authoredRegionMapDigest = hexDigest(sha256(authoredRegionSource.str()));

    if (arguments.referenceDisplacementGradient) {
        require(!arguments.withoutMedium, "reference-displacement arithmetic requires regional materials");
        object.femReferenceDisplacementGradient = true;
        object.femReferenceNodes = object.femNodes;
        std::ostringstream referenceSource;
        referenceSource << "numi.cardboard.stress-free-authored-reference.v1\n" << std::setprecision(17);
        for (const auto& node : object.femReferenceNodes)
            referenceSource << node[0] << ',' << node[1] << ',' << node[2] << '\n';
        object.femReferenceSourceIdentity = digestIdentity(sha256(referenceSource.str()));
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
    if (!arguments.boxBlank && !arguments.pinAdhesion) {
        object.femFixedNodes = result.leftGripNodes;
        object.femFixedNodes.insert(object.femFixedNodes.end(),
                                    result.rightGripNodes.begin(),
                                    result.rightGripNodes.end());
        std::sort(object.femFixedNodes.begin(), object.femFixedNodes.end());
        object.femFixedNodes.erase(
            std::unique(object.femFixedNodes.begin(), object.femFixedNodes.end()),
            object.femFixedNodes.end());
    }

    result.world.frameTimestep = arguments.timestepSeconds;
    // The shipping assembly and compression stages retain board self-weight
    // throughout their native history. The other coupon/strip protocols keep
    // their legacy zero-gravity controls.
    result.world.gravity = arguments.shippingBox
        ? std::array<double, 3>{0.0, -9.80665, 0.0}
        : std::array<double, 3>{0.0, 0.0, 0.0};
    if (glueMaterial != nullptr) result.world.contactSlop = 0.01 * std::min(arguments.mediumThicknessM, arguments.linerThicknessM);
    result.world.environmentCount = arguments.environmentCount;
    result.world.deterministic = true;
    result.world.mixedSolver.relativeResidual = arguments.relativeResidualTolerance;
    result.world.mixedSolver.fgmresIterations = arguments.fgmresIterations;
    result.world.mixedSolver.fgmresRestart = arguments.fgmresRestart;
    result.world.mixedSolver.lineSearchSteps = arguments.lineSearchSteps;
    result.world.mixedSolver.femPreconditioner = arguments.femPreconditioner;
    result.world.mixedSolver.newtonIterations = arguments.newtonIterations;
    result.world.materials = {linerMaterial, mediumMaterial};
    if (glueMaterial != nullptr) result.world.materials.push_back(*glueMaterial);
    if (arguments.shippingStarchLaw) {
        const auto& law = *arguments.shippingStarchLaw;
        CohesiveMaterialSource cohesive;
        cohesive.name = law.materialId;
        cohesive.normalPenaltyPaPerM = law.normalPenaltyPaPerM;
        cohesive.shearPenaltyPaPerM = law.shearPenaltyPaPerM;
        cohesive.compressionPenaltyPaPerM = law.compressionPenaltyPaPerM;
        cohesive.normalStrengthPa = law.normalStrengthPa;
        cohesive.shearStrengthPa = law.shearStrengthPa;
        cohesive.fractureEnergyModeIJPerM2 = law.fractureEnergyModeIJPerM2;
        cohesive.fractureEnergyModeIIJPerM2 = law.fractureEnergyModeIIJPerM2;
        cohesive.bkExponent = law.bkExponent;
        result.world.cohesiveMaterials.push_back(cohesive);
        object.femCohesiveMortarPatches.reserve(result.starchPairedFaces.size());
        std::uint32_t stableIdentifier = 1u;
        for (const auto& paired : result.starchPairedFaces) {
            require(paired.paperTetId <= std::numeric_limits<std::uint32_t>::max() &&
                        paired.glueTetId <= std::numeric_limits<std::uint32_t>::max(),
                    "starch paired tetrahedron index exceeds Matter's 32-bit source API");
            CohesiveMortarPatchSource patch;
            for (std::size_t slot = 0u; slot < 3u; ++slot) {
                require(paired.paperNodeIds[slot] <=
                            std::numeric_limits<std::uint32_t>::max() &&
                            paired.glueNodeIds[slot] <=
                            std::numeric_limits<std::uint32_t>::max(),
                        "starch paired node index exceeds Matter's 32-bit source API");
                patch.sideANodes[slot] = static_cast<std::uint32_t>(
                    paired.paperNodeIds[slot]);
                patch.sideBNodes[slot] = static_cast<std::uint32_t>(
                    paired.glueNodeIds[slot]);
                for (std::size_t coordinate = 0u; coordinate < 3u; ++coordinate) {
                    const double identityWeight = slot == coordinate ? 1.0 : 0.0;
                    patch.sideAVertexBarycentric[slot][coordinate] = identityWeight;
                    patch.sideBVertexBarycentric[slot][coordinate] = identityWeight;
                }
            }
            patch.firstTetrahedron =
                static_cast<std::uint32_t>(paired.paperTetId);
            patch.secondTetrahedron =
                static_cast<std::uint32_t>(paired.glueTetId);
            patch.materialIndex = paired.paperMaterialIndex;
            patch.stableIdentifier = stableIdentifier++;
            patch.cohesiveMaterialIndex = 0u;
            patch.activationGapToleranceM = law.activationGapToleranceM;
            object.femCohesiveMortarPatches.push_back(patch);
        }
        result.starchActivationCommands.reserve(
            static_cast<std::size_t>(arguments.environmentCount) *
            object.femCohesiveMortarPatches.size());
        for (std::uint32_t environment = 0u;
             environment < arguments.environmentCount; ++environment) {
            for (const auto& patch : object.femCohesiveMortarPatches) {
                NMCohesiveActivationCommandGPU command{};
                command.identity.x = environment;
                command.identity.y = 0u;
                command.identity.z = patch.stableIdentifier;
                command.identity.w = 0u;
                result.starchActivationCommands.push_back(command);
            }
        }
    }
    if (arguments.shippingHmaLaw) {
        require(arguments.shippingBox && result.boxBlank.has_value(),
                "HMA closure patches can only be attached to the shipping blank");
        const auto& law = *arguments.shippingHmaLaw;
        require(result.world.cohesiveMaterials.size() <
                    std::numeric_limits<std::uint32_t>::max(),
                "cohesive material table exceeds source index capacity");
        const std::uint32_t hmaMaterialIndex = static_cast<std::uint32_t>(
            result.world.cohesiveMaterials.size());
        CohesiveMaterialSource cohesive;
        cohesive.name = law.materialId;
        cohesive.normalPenaltyPaPerM = law.normalPenaltyPaPerM;
        cohesive.shearPenaltyPaPerM = law.shearPenaltyPaPerM;
        cohesive.compressionPenaltyPaPerM = law.compressionPenaltyPaPerM;
        cohesive.normalStrengthPa = law.normalStrengthPa;
        cohesive.shearStrengthPa = law.shearStrengthPa;
        cohesive.fractureEnergyModeIJPerM2 = law.fractureEnergyModeIJPerM2;
        cohesive.fractureEnergyModeIIJPerM2 = law.fractureEnergyModeIIJPerM2;
        cohesive.bkExponent = law.bkExponent;
        result.world.cohesiveMaterials.push_back(std::move(cohesive));

        std::vector<numi::cardboard::BoxBlankPoint> closureNodes;
        closureNodes.reserve(result.restNodes.size());
        for (const Vec3& point : result.restNodes)
            closureNodes.push_back({point.x, point.y, point.z});
        std::vector<numi::cardboard::ShippingClosureTet> closureTetrahedra;
        closureTetrahedra.reserve(result.tetrahedra.size());
        for (const TetMetadata& tet : result.tetrahedra)
            closureTetrahedra.push_back({tet.nodes, tet.materialIndex});
        const std::uint64_t firstStable64 =
            static_cast<std::uint64_t>(result.starchPairedFaces.size()) + 1u;
        require(firstStable64 <= std::numeric_limits<std::uint32_t>::max(),
                "starch interface IDs leave no range for shipping HMA patches");
        result.hmaClosureMesh =
            numi::cardboard::buildShippingClosureMortarMesh(
                *result.boxBlank, closureNodes, closureTetrahedra,
                /*hmaStripWidthM=*/0.010,
                law.activationGapToleranceM,
                hmaMaterialIndex,
                static_cast<std::uint32_t>(firstStable64));
        object.femCohesiveMortarPatches.reserve(
            object.femCohesiveMortarPatches.size() +
            result.hmaClosureMesh.patches.size());
        object.femCohesiveMortarPatches.insert(
            object.femCohesiveMortarPatches.end(),
            result.hmaClosureMesh.patches.begin(),
            result.hmaClosureMesh.patches.end());
    }
    if (!arguments.withoutMedium) {
        object.femMaterialIndices = result.materialIndices;
        std::ostringstream mapSource;
        mapSource << "numi.cardboard.layer-material-map.v1\n"
                  << "geometry=explicit_sinusoidal_approximation\n"
                  << "liner_sha256=" << linerDigest << '\n'
                  << "medium_sha256=" << mediumDigest << '\n'
                  << "glue_sha256=" << glueDigest << '\n'
                  << "authored_region_map_sha256="
                  << result.authoredRegionMapDigest << '\n'
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
    if (arguments.crease) {
        require(arguments.toolClearanceM > result.world.contactSlop &&
                arguments.anvilClearanceM > result.world.contactSlop,
                "tool and anvil clearances must exceed contact slop");
        // Only nodes on exposed tetrahedral faces may contact tooling. Shared
        // paper/glue interfaces and interior thickness nodes are excluded.
        std::map<std::array<std::uint32_t, 3>, std::uint32_t> faceCounts;
        for (const auto& cell : result.tetrahedra) {
            const auto& n = cell.nodes;
            for (auto face : std::array<std::array<std::uint32_t, 3>, 4>{{
                    {n[0], n[1], n[2]}, {n[0], n[1], n[3]},
                    {n[0], n[2], n[3]}, {n[1], n[2], n[3]}}}) {
                std::sort(face.begin(), face.end()); ++faceCounts[face];
            }
        }
        std::set<std::uint32_t> surfaceNodes;
        for (const auto& [face, count] : faceCounts) {
            require(count <= 2u, "nonmanifold tooling contact surface");
            if (count == 1u) surfaceNodes.insert(face.begin(), face.end());
        }
        require(!surfaceNodes.empty(), "empty tooling contact surface");
        object.femContactNodes.assign(surfaceNodes.begin(), surfaceNodes.end());
        numi::cardboard::CreaseToolingSpec spec;
        spec.boardLength = arguments.lengthM; spec.boardWidth = arguments.widthM;
        spec.boardCaliper = arguments.totalHeightM; spec.creaseX = 0.0;
        spec.supportSurfaceZ = -arguments.anvilClearanceM;
        spec.punchRadius = arguments.punchRadiusM; spec.punchEndOverhang = 0.002;
        spec.supportEndOverhang = 0.002; spec.supportThickness = 0.003;
        spec.initialClearance = arguments.toolClearanceM; spec.punchBodyIndex = 0u;
        auto geometry = numi::cardboard::makeCreaseTooling(spec);
        geometry.punch.prescribedEndPoseTranslation = arguments.prescribedToolPredictor;
        geometry.punch.prescribedBarrierStiffnessScale =
            arguments.prescribedToolBarrierStiffnessScale;
        geometry.support.localCenter[0] += 0.5 * arguments.lengthM;
        geometry.support.localCenter[1] += 0.5 * arguments.widthM;
        geometry.punchInitialPose.translation.x += 0.5 * arguments.lengthM;
        geometry.punchInitialPose.translation.y += 0.5 * arguments.widthM;
        geometry.punchInitialPose.translation.z += arguments.anvilClearanceM;
        result.world.rigidProxies = {geometry.punch, geometry.support};
        result.tooling = geometry;
    }
    if (arguments.pinAdhesion) {
        require(glueMaterial != nullptr && !arguments.withoutMedium,
                "pin adhesion requires the retained finite-glue literature2009 recipe");
        std::map<std::array<std::uint32_t, 3>, std::uint32_t> faceCounts;
        for (const auto& cell : result.tetrahedra) {
            const auto& n = cell.nodes;
            for (auto face : std::array<std::array<std::uint32_t, 3>, 4>{{
                    {n[0], n[1], n[2]}, {n[0], n[1], n[3]},
                    {n[0], n[2], n[3]}, {n[1], n[2], n[3]}}}) {
                std::sort(face.begin(), face.end());
                ++faceCounts[face];
            }
        }
        std::set<std::uint32_t> surfaceNodes;
        for (const auto& [face, count] : faceCounts) {
            require(count <= 2u, "nonmanifold pin coupon surface");
            if (count == 1u) surfaceNodes.insert(face.begin(), face.end());
        }
        require(!surfaceNodes.empty(), "pin coupon has no exposed finite-element surface");
        object.femContactNodes.assign(surfaceNodes.begin(), surfaceNodes.end());

        auto spec = numi::cardboard::pinAdhesionEngineeringPrototype(0u);
        spec.specimenWidthM = arguments.lengthM;
        spec.specimenLengthM = arguments.widthM;
        spec.boardCaliperM = arguments.totalHeightM;
        auto fixture = numi::cardboard::makePinAdhesionFixture(spec);
        for (auto& pin : fixture.pins) {
            pin.initialBodyTranslationM[0] += 0.5 * arguments.lengthM;
            pin.initialBodyTranslationM[1] += 0.5 * arguments.widthM;
            pin.initialBodyTranslationM[2] += 0.5 * arguments.totalHeightM;
            pin.proxy.sceneBodyIndex = NM_INVALID_INDEX;
            result.world.rigidProxies.push_back(pin.proxy);
        }
        result.pinFixture = std::move(fixture);
    }
    if (arguments.boxAssemblyNative) {
        require(result.boxBlank.has_value(),
                "native box assembly requires the authored FEFCO blank");
        require(object.deformableContact || arguments.disableSelfContact,
                "native box assembly requires the authored finite-glue contact surface or explicit numerical no-self-contact control");
        std::map<std::array<std::uint32_t, 3>, std::uint32_t> faceCounts;
        for (const auto& cell : result.tetrahedra) {
            const auto& n = cell.nodes;
            for (auto face : std::array<std::array<std::uint32_t, 3>, 4>{{
                    {n[0], n[1], n[2]}, {n[0], n[1], n[3]},
                    {n[0], n[2], n[3]}, {n[1], n[2], n[3]}}}) {
                std::sort(face.begin(), face.end());
                ++faceCounts[face];
            }
        }
        std::set<std::uint32_t> surfaceNodes;
        for (const auto& [face, count] : faceCounts) {
            require(count <= 2u, "nonmanifold box blank contact surface");
            if (count == 1u) surfaceNodes.insert(face.begin(), face.end());
        }
        require(!surfaceNodes.empty(), "box blank has no exposed contact nodes");
        object.femContactNodes.assign(surfaceNodes.begin(), surfaceNodes.end());
        const auto& blank = *result.boxBlank;
        const numi::cardboard::ShippingScoreToolDefaults shippingScoreDefaults;
        result.boxScoreTooling = arguments.shippingBox
            ? numi::cardboard::makeSequentialBoxScoreTooling(
                blank, arguments.boxPunchRadiusM,
                0.25 * arguments.pitchM,
                arguments.boxScoreToolClearanceM,
                arguments.boxScoreToolClearanceM, 0.001,
                0.25 * arguments.pitchM,
                shippingScoreDefaults.channelWidthM,
                shippingScoreDefaults.supportRailWidthM)
            : numi::cardboard::makeBoxScoreTooling(
                blank, arguments.boxPunchRadiusM, 0.25 * arguments.pitchM,
                arguments.boxScoreToolClearanceM,
                arguments.boxScoreToolClearanceM, 0.001,
                0.25 * arguments.pitchM, 0.0,
                shippingScoreDefaults.supportRailWidthM);
        result.world.rigidProxies = result.boxScoreTooling->proxies;
#ifdef NUMI_CARDBOARD_METAL_WORLD
        if (arguments.shippingBox) {
            numi::cardboard::ShippingRigidBodySpec shippingBody;
            shippingBody.massKg = 1.0;
            shippingBody.halfExtentM = arguments.shippingImpactorHalfExtentsM;
            shippingBody.initialCenterM = {100.0, 1.0, 100.0};
            shippingBody.gravityMPerS2 = {0.0, -9.80665, 0.0};
            shippingBody.timestepSeconds = arguments.timestepSeconds;
            require(result.boxScoreTooling->families.size() ==
                        numi::cardboard::kShippingScoreToolFamilyCount &&
                    result.boxScoreTooling->proxies.size() ==
                        numi::cardboard::kShippingScoreToolProxyCount,
                    "shipping tool factory diverged from its prebuild composition counts");
            const std::uint32_t scoreToolBodyCount = static_cast<std::uint32_t>(
                result.boxScoreTooling->families.size() *
                numi::cardboard::kShippingSceneBodiesPerScoreFamily);
            result.world.rigidProxies.push_back(
                numi::cardboard::makeShippingCompressionPlatenProxy(
                    {0.5 * blank.config.panelLength, 0.002,
                     0.5 * blank.config.panelWidth}, 0u,
                    scoreToolBodyCount));
            result.world.rigidProxies.push_back(
                numi::cardboard::makeShippingFloorProxy(
                    {0.5 * blank.perimeterSpan + 0.010,
                     0.5 * blank.blankHeight + 0.010}, 0.002, 0u,
                    scoreToolBodyCount + 1u));
            result.shippingImpactorProxyIndex = scoreToolBodyCount + 2u;
        result.world.rigidProxies.push_back(
                numi::cardboard::makeShippingImpactorProxy(
                    shippingBody, scoreToolBodyCount + 2u));
        require(result.world.rigidProxies.size() ==
                    numi::cardboard::kShippingRigidProxyCount,
                "shipping tool composition differs from its prebuild rigid-proxy count");
        }
#else
        require(!arguments.shippingBox,
                "shipping-box rigid proxies require the integrated MetalWorld build");
#endif
    }
    result.world.objects = {std::move(object)};
    return result;
}

void applyRetainedMatrixDamageOverlay(
    MeshSource& mesh, const MaterialProgram& linerMaterial,
    const MaterialProgram& mediumMaterial, const std::string& linerDigest,
    const std::string& mediumDigest, const std::string& glueDigest,
    const Arguments& arguments) {
    using namespace numi::cardboard::retained_damage;
    require(arguments.linerDamageSet.has_value() &&
                arguments.mediumDamageSet.has_value(),
            "retained-matrix overlay requires both explicit regional parameter sets");
    require(mesh.world.objects.size() == 1u &&
                mesh.world.objects.front().femMaterialIndices.size() ==
                    mesh.tetrahedra.size() &&
                mesh.materialIndices.size() == mesh.tetrahedra.size() &&
                mesh.restNodes.size() > 0u,
            "retained-matrix overlay source mesh/material map is incomplete");
    require(!arguments.scalePaperProjectionResiduals,
            "retained-matrix damage overlay cannot be combined with the source-only 13-row residual scaling experiment");

    using GeometryKey = std::tuple<std::uint32_t, std::array<double, 4>>;
    std::map<GeometryKey, std::size_t> geometryToRequest;
    std::vector<ElementRequest> uniqueRequests;
    std::vector<std::size_t> requestByTetrahedron(
        mesh.tetrahedra.size(), std::numeric_limits<std::size_t>::max());
    std::vector<std::array<double, 4>> modeLengthsByTetrahedron(
        mesh.tetrahedra.size());
    constexpr double inverseSqrtThree = 0.577350269189625764509148780501957456;
    const std::array<FractureVec3, 4> localNormals{{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {inverseSqrtThree, inverseSqrtThree, inverseSqrtThree}}};

    for (std::size_t element = 0u; element < mesh.tetrahedra.size(); ++element) {
        const std::uint32_t sourceMaterialIndex = mesh.materialIndices[element];
        if (sourceMaterialIndex == 2u) continue; // finite glue remains undamaged
        require(sourceMaterialIndex <= 1u,
                "retained-matrix damage encountered an unsupported source region");
        require(element <= std::numeric_limits<std::uint32_t>::max(),
                "tetrahedron index exceeds retained-matrix provenance capacity");
        const TetMetadata& tet = mesh.tetrahedra[element];
        numi::cardboard::FractureRotation3 materialToReference;
        const double cosine = std::cos(tet.frameAngle);
        const double sine = std::sin(tet.frameAngle);
        materialToReference.value = {
            cosine, 0.0, -sine,
            0.0, 1.0, 0.0,
            sine, 0.0, cosine};

        ElementRequest request;
        request.elementIndex = static_cast<std::uint32_t>(element);
        request.sourceMaterialIndex = sourceMaterialIndex;
        request.regionId = sourceMaterialIndex == 0u ? "liner" : "medium";
        const MaterialProgram& sourceMaterial = sourceMaterialIndex == 0u
            ? linerMaterial : mediumMaterial;
        request.sourceMatrix = sourceMaterial;
        request.sourceDensityKgPerM3 = materialDensity(sourceMaterial);
        request.sourceBinding = {
            sourceMaterial.name,
            sourceMaterialIndex == 0u ? linerDigest : mediumDigest,
            "retained-Haj-Ali-2009-source-material"};
        request.explicitSet = sourceMaterialIndex == 0u
            ? *arguments.linerDamageSet : *arguments.mediumDamageSet;
        for (std::size_t slot = 0u; slot < 4u; ++slot) {
            const std::uint32_t node = tet.nodes[slot];
            require(node < mesh.restNodes.size(),
                    "retained-matrix overlay tetrahedron has an invalid node index");
            const Vec3& point = mesh.restNodes[node];
            request.referenceVerticesM[slot] = {point.x, point.y, point.z};
            request.selectedMaterialPlanes[slot] = {
                localNormals[slot],
                numi::cardboard::FractureNormalFrame::ElementMaterial,
                0.0, materialToReference, 1.0e-12};
        }
        std::array<double, 4> modeLengths{};
        for (std::size_t plane = 0u; plane < modeLengths.size(); ++plane) {
            const auto geometry =
                numi::cardboard::computeTetCrackPlaneGeometry(
                    request.referenceVerticesM,
                    request.selectedMaterialPlanes[plane]);
            require(geometry.hasPositiveAreaCut(),
                    "retained-matrix mode plane has no positive-area cut");
            modeLengths[plane] = geometry.characteristicLengthM.value();
        }
        modeLengthsByTetrahedron[element] = modeLengths;
        const GeometryKey key{sourceMaterialIndex, modeLengths};
        auto [position, inserted] = geometryToRequest.emplace(
            key, uniqueRequests.size());
        if (inserted) uniqueRequests.push_back(std::move(request));
        requestByTetrahedron[element] = position->second;
    }
    require(!uniqueRequests.empty(),
            "retained-matrix damage found no liner or medium tetrahedra");

    const MeshProgramPool pool = buildMeshProgramPool(uniqueRequests);
    require(pool.overlayPrograms.size() == pool.overlayDensitiesKgPerM3.size() &&
                pool.overlayMaterialIndices.size() == uniqueRequests.size() &&
                pool.elementMappings.size() == uniqueRequests.size(),
            "retained-matrix builder returned inconsistent pool tables");
    const std::size_t baseMaterialCount = mesh.world.materials.size();
    require(baseMaterialCount <= std::numeric_limits<std::uint32_t>::max() &&
                pool.overlayPrograms.size() <=
                    std::numeric_limits<std::uint32_t>::max() - baseMaterialCount,
            "retained-matrix overlay exceeds source material index capacity");
    for (std::size_t program = 0u; program < pool.overlayPrograms.size(); ++program) {
        const double expectedDensity = materialDensity(
            pool.overlayPrograms[program]);
        require(pool.overlayDensitiesKgPerM3[program] == expectedDensity,
                "retained-matrix overlay changed its copied source density");
        mesh.world.materials.push_back(pool.overlayPrograms[program]);
    }
    auto& object = mesh.world.objects.front();
    mesh.damageOverlayTetrahedronCount = 0u;
    for (std::size_t element = 0u; element < mesh.tetrahedra.size(); ++element) {
        const std::size_t requestIndex = requestByTetrahedron[element];
        if (requestIndex == std::numeric_limits<std::size_t>::max()) continue;
        const std::uint32_t poolIndex = pool.overlayMaterialIndices[requestIndex];
        require(poolIndex < pool.overlayPrograms.size(),
                "retained-matrix material pool index is invalid");
        const auto materialIndex = static_cast<std::uint32_t>(
            baseMaterialCount + poolIndex);
        mesh.materialIndices[element] = materialIndex;
        mesh.tetrahedra[element].materialIndex = materialIndex;
        object.femMaterialIndices[element] = materialIndex;
        ++mesh.damageOverlayTetrahedronCount;
    }
    require(mesh.damageOverlayTetrahedronCount > 0u,
            "retained-matrix overlay remapped no substrate tetrahedra");

    for (auto& patch : object.femCohesiveMortarPatches) {
        if (patch.firstTetrahedron < mesh.materialIndices.size() &&
            mesh.materialIndices[patch.firstTetrahedron] >= baseMaterialCount)
            patch.materialIndex = mesh.materialIndices[patch.firstTetrahedron];
    }

    mesh.damageOverlayProgramCount = static_cast<std::uint32_t>(
        pool.overlayPrograms.size());
    mesh.damageOverlayMappings = pool.elementMappings;
    mesh.damageOverlayMaterialIndices.reserve(pool.elementMappings.size());
    for (std::size_t index = 0u; index < pool.elementMappings.size(); ++index)
        mesh.damageOverlayMaterialIndices.push_back(static_cast<std::uint32_t>(
            baseMaterialCount + pool.overlayMaterialIndices[index]));

    std::ostringstream mappingSource;
    mappingSource << "numi.cardboard.retained-matrix-damage-map.v1\n"
                  << "liner_parameter_file_sha256="
                  << arguments.linerDamageParametersSha256 << '\n'
                  << "medium_parameter_file_sha256="
                  << arguments.mediumDamageParametersSha256 << '\n'
                  << "liner_source_material_sha256=" << linerDigest << '\n'
                  << "medium_source_material_sha256=" << mediumDigest << '\n'
                  << "frame_map_sha256=" << mesh.frameMapDigest << '\n'
                  << "authored_region_map_sha256="
                  << mesh.authoredRegionMapDigest << '\n'
                  << "generic_compensated_implicit_projection="
                  << (arguments.compensatedImplicitProjection ? 1 : 0) << '\n'
                  << std::setprecision(17);
    for (std::size_t element = 0u; element < mesh.materialIndices.size(); ++element) {
        mappingSource << element << ':' << mesh.materialIndices[element];
        if (requestByTetrahedron[element] !=
            std::numeric_limits<std::size_t>::max()) {
            for (const double length : modeLengthsByTetrahedron[element])
                mappingSource << ':' << length;
        }
        mappingSource << '\n';
    }
    const Digest mappingDigest = sha256(mappingSource.str());
    mesh.damageOverlayMappingDigest = hexDigest(mappingDigest);

    std::ostringstream materialMapSource;
    materialMapSource << "numi.cardboard.layer-material-map.v2\n"
                      << "geometry=explicit_sinusoidal_approximation\n"
                      << "liner_sha256=" << linerDigest << '\n'
                      << "medium_sha256=" << mediumDigest << '\n'
                      << "glue_sha256=" << glueDigest << '\n'
                      << "liner_damage_parameters_sha256="
                      << arguments.linerDamageParametersSha256 << '\n'
                      << "medium_damage_parameters_sha256="
                      << arguments.mediumDamageParametersSha256 << '\n'
                      << "damage_overlay_mapping_sha256="
                      << mesh.damageOverlayMappingDigest << '\n'
                      << std::setprecision(17);
    for (std::size_t index = 0u; index < mesh.materialIndices.size(); ++index)
        materialMapSource << index << ':' << mesh.materialIndices[index] << '\n';
    const Digest materialMapDigest = sha256(materialMapSource.str());
    mesh.materialMapDigest = hexDigest(materialMapDigest);
    object.femMaterialSourceIdentity = digestIdentity(materialMapDigest);
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

double compiledSubstrateNodalMassKg(const CompiledWorld& world) {
    require(world.dispatch.femNodeCount > 0u &&
                world.fem.nodes.size() >= world.dispatch.femNodeCount,
            "compiled FEM nodal mass arena is truncated");
    double totalMassKg = 0.0;
    for (std::uint32_t nodeIndex = 0u;
         nodeIndex < world.dispatch.femNodeCount; ++nodeIndex) {
        const double mass = world.fem.nodes[nodeIndex].positionAndMass.w;
        require(std::isfinite(mass) && mass >= 0.0,
                "compiled FEM nodal mass is nonfinite or negative");
        totalMassKg += mass;
    }
    require(std::isfinite(totalMassKg) && totalMassKg > 0.0,
            "compiled FEM substrate has no finite positive nodal mass");
    return totalMassKg;
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

void writeBoxLayout(const std::filesystem::path& path, const MeshSource& mesh) {
    if (!mesh.boxBlank) return;
    const auto& blank = *mesh.boxBlank;
    std::ofstream output(path);
    require(output.good(), "cannot create box layout");
    const auto jsonIndexSemantics = [](const numi::cardboard::BoxAssemblyFeatureKind kind) {
        switch (kind) {
            case numi::cardboard::BoxAssemblyFeatureKind::panel:
                return "zero-based index into panels";
            case numi::cardboard::BoxAssemblyFeatureKind::scoreLine:
                return "zero-based index into score_lines";
            case numi::cardboard::BoxAssemblyFeatureKind::manufacturerJoint:
                return "0 identifies the manufacturer_joint object";
            case numi::cardboard::BoxAssemblyFeatureKind::boxEnd:
                return "zero-based index into box_ends: 0=bottom, 1=top";
        }
        return "unknown feature index semantics";
    };
    const auto rootScoreLine = std::find_if(
        blank.scoreLines.begin(), blank.scoreLines.end(), [](const auto& line) {
            return line.kind == numi::cardboard::BoxScoreKind::manufacturerJointRoot;
        });
    require(rootScoreLine != blank.scoreLines.end(),
            "box layout has no manufacturer-joint root score line");
    const auto rootScoreLineIndex = static_cast<std::size_t>(
        std::distance(blank.scoreLines.begin(), rootScoreLine));

    output << std::setprecision(17)
        << "{\n  \"schema\": \"numi.cardboard.fefco0201-layout.v2\",\n"
        << "  \"units\": \"metres\",\n  \"state\": \"flat_unscored_blank\",\n"
        << "  \"physical_assembly\": false,\n  \"dimensions\": {\"panel_length\": " << blank.config.panelLength
        << ", \"panel_width\": " << blank.config.panelWidth << ", \"wall_height\": " << blank.config.wallHeight
        << ", \"joint_width\": " << blank.config.manufacturerJointWidth << ", \"slot_kerf\": " << blank.config.slotKerf
        << ", \"board_caliper\": " << blank.boardCaliper
        << ", \"perimeter_span\": " << blank.perimeterSpan
        << ", \"blank_height\": " << blank.blankHeight
        << ", \"bottom_score_y\": " << blank.bottomScoreY
        << ", \"top_score_y\": " << blank.topScoreY
        << ", \"flap_depth\": " << blank.topFlapDepth << "},\n"
        << "  \"panels\": [\n";
    for (std::size_t i = 0; i < blank.panels.size(); ++i) {
        const auto& panel = blank.panels[i];
        output << "    {\"index\": " << panel.index
            << ", \"long_wall\": " << (panel.longWall ? "true" : "false")
            << ", \"bounds_kind\": \"axis_aligned_envelope\", \"bounds_m\": {\"x\": ["
            << panel.xStart << ',' << panel.xEnd << "], \"y\": ["
            << panel.yStart << ',' << panel.yEnd << "], \"z\": [0," << blank.boardCaliper << "]}}"
            << (i + 1 == blank.panels.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"manufacturer_joint\": {\"index\": 0, \"bounds_kind\": \"axis_aligned_envelope\", \"bounds_m\": {\"x\": [0,"
        << blank.config.manufacturerJointWidth << "], \"y\": [" << blank.bottomScoreY << ',' << blank.topScoreY
        << "], \"z\": [0," << blank.boardCaliper << "]}, \"root_score_line_index\": "
        << rootScoreLineIndex << "},\n  \"box_ends\": [\n";
    for (std::uint32_t endIndex = 0u; endIndex < 2u; ++endIndex) {
        const bool top = endIndex == 1u;
        const auto flapKind = top ? numi::cardboard::BoxScoreKind::topFlap
                                  : numi::cardboard::BoxScoreKind::bottomFlap;
        output << "    {\"index\": " << endIndex
            << ", \"end\": \"" << (top ? "top" : "bottom")
            << "\", \"bounds_kind\": \"axis_aligned_envelope\", \"bounds_m\": {\"x\": [" << blank.config.manufacturerJointWidth << ","
            << blank.perimeterSpan << "], \"y\": ["
            << (top ? blank.topScoreY : 0.0) << ','
            << (top ? blank.blankHeight : blank.bottomScoreY)
            << "], \"z\": [0," << blank.boardCaliper << "]}, \"panel_indices\": [";
        for (std::size_t panelIndex = 0; panelIndex < blank.panels.size(); ++panelIndex) {
            if (panelIndex) output << ',';
            output << blank.panels[panelIndex].index;
        }
        output << "], \"flap_score_line_indices\": [";
        bool firstFlap = true;
        for (std::size_t scoreIndex = 0; scoreIndex < blank.scoreLines.size(); ++scoreIndex) {
            const auto& score = blank.scoreLines[scoreIndex];
            if (score.kind != flapKind) continue;
            if (!firstFlap) output << ',';
            firstFlap = false;
            output << scoreIndex;
        }
        output << "]}" << (endIndex == 1u ? "\n" : ",\n");
    }
    output << "  ],\n  \"score_lines\": [\n";
    for (std::size_t i = 0; i < blank.scoreLines.size(); ++i) {
        const auto& score = blank.scoreLines[i];
        output << "    {\"kind\": " << static_cast<unsigned>(score.kind) << ", \"panel\": " << score.panelIndex
            << ", \"first\": [" << score.first.x << ',' << score.first.y << ',' << score.first.z
            << "], \"second\": [" << score.second.x << ',' << score.second.y << ',' << score.second.z
            << "], \"physically_scored\": false}" << (i + 1 == blank.scoreLines.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"assembly_actions\": [\n";
    for (std::size_t i = 0; i < blank.assembly.actions.size(); ++i) {
        const auto& action = blank.assembly.actions[i];
        output << "    {\"stage\": \"" << jsonEscape(action.stage) << "\", \"kind\": " << static_cast<unsigned>(action.kind)
            << ", \"feature\": {\"type\": \"" << boxAssemblyFeatureKindName(action.feature.kind)
            << "\", \"index\": " << action.feature.index
            << ", \"index_semantics\": \"" << jsonIndexSemantics(action.feature.kind) << "\"}"
            << ", \"hinge_score_line_index\": ";
        if (action.hingeScoreLineIndex == numi::cardboard::kNoBoxScoreLineIndex)
            output << "null";
        else
            output << action.hingeScoreLineIndex;
        output << ", \"target_angle_deg\": " << action.targetAngleDegrees
            << ", \"executed\": false}" << (i + 1 == blank.assembly.actions.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
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

void writeRetainedMatrixDamageOverlayJson(
    const std::filesystem::path& path,
    const Arguments& arguments,
    MeshSource& mesh) {
    require(arguments.linerDamageSet && arguments.mediumDamageSet &&
                !mesh.damageOverlayMappings.empty() &&
                mesh.damageOverlayMappings.size() ==
                    mesh.damageOverlayMaterialIndices.size(),
            "retained-matrix sidecar is missing source sets or pool mappings");
    std::ofstream output(path);
    require(output.good(), "cannot create retained-matrix damage sidecar: " +
                path.string());
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"numi.cardboard.retained-matrix-overlay-pool.v1\",\n"
           << "  \"evidence_class\": \"explicit feasible-set candidate; not calibrated to the shipping corrugated grade\",\n"
           << "  \"input_sets\": {\n"
           << "    \"liner\": {\"path\": \""
           << jsonEscape(arguments.linerDamageParameters.string())
           << "\", \"sha256\": \""
           << arguments.linerDamageParametersSha256
           << "\", \"set_id\": \""
           << jsonEscape(arguments.linerDamageSet->setId)
           << "\", \"evidence_class\": \""
           << jsonEscape(arguments.linerDamageSet->evidenceClass) << "\"},\n"
           << "    \"medium\": {\"path\": \""
           << jsonEscape(arguments.mediumDamageParameters.string())
           << "\", \"sha256\": \""
           << arguments.mediumDamageParametersSha256
           << "\", \"set_id\": \""
           << jsonEscape(arguments.mediumDamageSet->setId)
           << "\", \"evidence_class\": \""
           << jsonEscape(arguments.mediumDamageSet->evidenceClass) << "\"}\n"
           << "  },\n"
           << "  \"source_scope\": {\"liner_sha256\": \""
           << hexDigest(sha256File(arguments.linerMaterial))
           << "\", \"medium_sha256\": \""
           << hexDigest(sha256File(arguments.mediumMaterial))
           << "\", \"preserved_source_density_and_Hill_program\": true,"
              " \"generic_compensated_implicit_projection\": "
           << (arguments.compensatedImplicitProjection ? "true" : "false")
           << ", \"glue_overlayed\": false},\n"
           << "  \"mapping\": {\"sha256\": \""
           << mesh.damageOverlayMappingDigest
           << "\", \"representative_geometry_profile_count\": "
           << mesh.damageOverlayMappings.size()
           << ", \"unique_overlay_program_count\": "
           << mesh.damageOverlayProgramCount
           << ", \"substrate_tetrahedra_remapped\": "
           << mesh.damageOverlayTetrahedronCount
           << ", \"tetrahedron_to_material_map\": \"mesh.json material_indices; full map bound by mapping hash\"},\n"
           << "  \"programs\": [\n";
    for (std::size_t index = 0u;
         index < mesh.damageOverlayMappings.size(); ++index) {
        const auto& mapping = mesh.damageOverlayMappings[index];
        const auto& set = mapping.sourceMaterialIndex == 0u
            ? *arguments.linerDamageSet : *arguments.mediumDamageSet;
        output << "    {\"representative_tetrahedron\": "
               << mapping.elementIndex
               << ", \"source_material_index\": "
               << mapping.sourceMaterialIndex
               << ", \"source_material_id\": \""
               << jsonEscape(mapping.sourceMaterialId)
               << "\", \"source_material_sha256\": \""
               << mapping.sourceMaterialSha256
               << "\", \"overlay_material_index\": "
               << mesh.damageOverlayMaterialIndices[index]
               << ", \"overlay_pool_index\": "
               << mapping.overlayPoolIndex
               << ", \"reference_volume_m3\": "
               << mapping.referenceVolumeM3
               << ", \"source_density_kg_m3\": "
               << mapping.sourceDensityKgPerM3
               << ", \"damage_set_id\": \""
               << jsonEscape(mapping.explicitSetId)
               << "\", \"damage_set_sha256\": \""
               << mapping.explicitSetSha256
               << "\", \"evidence_class\": \""
               << jsonEscape(mapping.evidenceClass)
               << "\", \"damage_cap\": " << mapping.damageCap
               << ", \"smoothing_strain_width\": "
               << mapping.smoothStrainWidth
               << ", \"hardening_pa_11_22_33_23_13_12\": [";
        for (std::size_t mode = 0u;
             mode < mapping.plasticHardeningPa.size(); ++mode) {
            if (mode) output << ',';
            output << mapping.plasticHardeningPa[mode];
        }
        output << "], \"plane_area_m2\": [";
        for (std::size_t plane = 0u;
             plane < mapping.planeGeometry.size(); ++plane) {
            if (plane) output << ',';
            output << mapping.planeGeometry[plane].referenceAreaM2;
        }
        output << "], \"characteristic_length_m\": [";
        for (std::size_t plane = 0u;
             plane < mapping.modeLengthM.size(); ++plane) {
            if (plane) output << ',';
            output << mapping.modeLengthM[plane];
        }
        output << "], \"mode_onset_strain\": [";
        for (std::size_t mode = 0u;
             mode < mapping.modeOnsetStrain.size(); ++mode) {
            if (mode) output << ',';
            output << mapping.modeOnsetStrain[mode];
        }
        output << "], \"mode_softening_strain\": [";
        for (std::size_t mode = 0u;
             mode < mapping.modeSofteningStrain.size(); ++mode) {
            if (mode) output << ',';
            output << mapping.modeSofteningStrain[mode];
        }
        output << "], \"strength_pa\": [";
        for (std::size_t mode = 0u;
             mode < mapping.fractureModes.size(); ++mode) {
            if (mode) output << ',';
            output << mapping.fractureModes[mode].strengthPa;
        }
        output << "], \"fracture_energy_j_m2\": [";
        for (std::size_t mode = 0u;
             mode < mapping.fractureModes.size(); ++mode) {
            if (mode) output << ',';
            output << mapping.fractureModes[mode].fractureEnergyJPerM2;
        }
        output << "], \"rate_time_seconds\": [";
        for (std::size_t mode = 0u;
             mode < mapping.fractureModes.size(); ++mode) {
            if (mode) output << ',';
            output << mapping.fractureModes[mode].rateTimeConstantSeconds;
        }
        output << "], \"densification_modulus_pa\": [";
        for (std::size_t axis = 0u;
             axis < mapping.densificationModulusPa.size(); ++axis) {
            if (axis) output << ',';
            output << mapping.densificationModulusPa[axis];
        }
        output << "], \"densification_onset_strain\": [";
        for (std::size_t axis = 0u;
             axis < mapping.densificationOnsetStrain.size(); ++axis) {
            if (axis) output << ',';
            output << mapping.densificationOnsetStrain[axis];
        }
        output << "], \"densification_rate_time_seconds\": [";
        for (std::size_t axis = 0u;
             axis < mapping.densificationRateTimeConstantSeconds.size(); ++axis) {
            if (axis) output << ',';
            output << mapping.densificationRateTimeConstantSeconds[axis];
        }
        output << "], \"hardening_binding\": {\"source_id\": \""
               << jsonEscape(mapping.hardeningBinding.sourceId)
               << "\", \"dataset_id\": \""
               << jsonEscape(mapping.hardeningBinding.datasetId)
               << "\", \"dataset_sha256\": \""
               << mapping.hardeningBinding.datasetSha256
               << "\", \"mapping\": \""
               << jsonEscape(mapping.hardeningBinding.mapping)
               << "\"}, \"fracture_binding\": {\"set_id\": \""
               << jsonEscape(mapping.fractureBinding.setId)
               << "\", \"sha256\": \"" << mapping.fractureBinding.sha256
               << "\", \"evidence_class\": \""
               << jsonEscape(mapping.fractureBinding.evidenceClass)
               << "\"}, \"rate_binding\": {\"set_id\": \""
               << jsonEscape(mapping.rateBinding.setId)
               << "\", \"sha256\": \"" << mapping.rateBinding.sha256
               << "\", \"evidence_class\": \""
               << jsonEscape(mapping.rateBinding.evidenceClass)
               << "\"}, \"densification_binding\": {\"set_id\": \""
               << jsonEscape(mapping.densificationBinding.setId)
               << "\", \"sha256\": \""
               << mapping.densificationBinding.sha256
               << "\", \"evidence_class\": \""
               << jsonEscape(mapping.densificationBinding.evidenceClass)
               << "\"}, \"parameter_input_sha256\": \""
               << set.sha256 << "\"}"
               << (index + 1u == mesh.damageOverlayMappings.size()
                       ? "\n" : ",\n");
    }
    output << "  ],\n"
           << "  \"qualification\": {\"source_grade_calibrated\": false, "
              "\"shipping_physics_qualified\": false}\n}\n";
    output.flush();
    require(output.good(), "failed writing retained-matrix damage sidecar");
    output.close();
    mesh.damageOverlaySidecarSha256 = hexDigest(sha256File(path));
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
    const double effectiveToolBarrierScale = static_cast<double>(
        static_cast<float>(arguments.prescribedToolBarrierStiffnessScale));
    const std::string scoreCommandSha256 = arguments.shippingBox
        ? shippingScoreCommandFingerprint(
              arguments, compiled.world.physicsFingerprint,
              compiled.world.fingerprint, arguments.totalHeightM)
        : std::string{};

    std::ofstream output(path);
    require(output.good(), "cannot create manifest: " + path.string());
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"numi.cardboard.explicit-strip.v1\",\n"
           << "  \"specimen\": \"" << (arguments.pinAdhesion ? "pin_adhesion_coupon_analogue" : arguments.boxBlank ? "fefco0201_flat_blank" : "corrugated_strip") << "\",\n"
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
           << "    \"liner_path_explicit\": " << (arguments.linerMaterialExplicit ? "true" : "false") << ",\n"
           << "    \"medium_path\": \"" << jsonEscape(arguments.mediumMaterial.string()) << "\",\n"
           << "    \"medium_sha256\": \"" << hexDigest(mediumDigest) << "\",\n"
           << "    \"medium_path_explicit\": " << (arguments.mediumMaterialExplicit ? "true" : "false") << ",\n"
           << "    \"regional_map_sha256\": \"" << mesh.materialMapDigest << "\",\n"
           << "    \"authored_region_map_sha256\": \""
           << mesh.authoredRegionMapDigest << "\",\n"
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
           << "  \"tooling\": {\"enabled\": " << (arguments.crease ? "true" : "false")
           << ", \"mode\": \"rounded capsule on finite backing box\", \"indentation_m\": " << arguments.indentationM
           << ", \"native_feasible_predictor\": " << (arguments.prescribedToolPredictor ? "true" : "false")
           << ", \"prescribed_tool_barrier_stiffness_scale\": " << effectiveToolBarrierScale;
    if (arguments.prescribedToolBarrierStiffnessScale != effectiveToolBarrierScale)
        output << ", \"prescribed_tool_barrier_stiffness_scale_requested\": "
               << arguments.prescribedToolBarrierStiffnessScale;
    output << ", \"barrier_scale_role\": \"numerical conditioning sensitivity on prescribed punch only; not a material calibration; activation distance and collision floors unchanged\""
           << ", \"punch_radius_m\": " << arguments.punchRadiusM
           << ", \"initial_nose_clearance_m\": " << arguments.toolClearanceM
           << ", \"initial_anvil_clearance_m\": " << arguments.anvilClearanceM
           << ", \"crease_x_m\": " << 0.5 * arguments.lengthM
           << ", \"timing\": \"prescribed end pose and consistent step velocity in preDynamics and postCommit; independent end gap audit\"},\n"
           << "  \"shipping_score_command\": {\"enabled\": "
           << (arguments.shippingBox ? "true" : "false")
           << ", \"selected_depth_ratio_of_caliper\": "
           << arguments.shippingScoreDepthRatio
           << ", \"tool_speed_m_per_s\": "
           << arguments.shippingScoreToolSpeedMPerS
           << ", \"tool_speed_evidence_class\": \"engineering_assumption_unqualified\""
           << ", \"execution\": \"one selected depth per invocation, approach then retract; each invocation constructs a fresh flat blank\""
           << ", \"source_command_sha256\": \"" << scoreCommandSha256
           << "\"},\n"
           << "  \"solver\": {\n"
           << "    \"backend\": \"implicit nonlinear Matter FEM on Apple Metal\",\n"
           << "    \"deformable_self_contact\": " << (mesh.world.objects.front().deformableSelfContact ? "true" : "false") << ",\n"
           << "    \"compact_immutable_surface_map\": "
           << (arguments.compactImmutableSurfaces ? "true" : "false") << ",\n"
           << "    \"compiled_surface_face_count\": "
           << compiled.world.dispatch.surfaceFaceCount << ",\n"
           << "    \"deformable_contact_capacity_requested\": "
           << (arguments.deformableContactCapacityProvided
                   ? std::to_string(arguments.deformableContactCapacity) : "null") << ",\n"
           << "    \"deformable_contact_capacity_compiled\": "
           << compiled.world.dispatch.deformableContactCapacity << ",\n"
           << "    \"contact_slop_m\": " << mesh.world.contactSlop << ",\n"
           << "    \"deformation_gradient\": \"" << (arguments.referenceDisplacementGradient ? "persistent_reference_displacement" : "absolute_position") << "\",\n"
           << "    \"dt_s\": " << arguments.timestepSeconds << ",\n"
           << "    \"bend_angle_deg\": " << arguments.bendAngleDegrees << ",\n"
           << "    \"steps\": " << arguments.steps << ",\n"
           << "    \"hold_steps\": " << arguments.holdSteps << ",\n"
           << "    \"unload_steps\": " << arguments.unloadSteps << ",\n"
           << "    \"relax_steps\": " << arguments.relaxSteps << ",\n"
           << "    \"release_steps\": " << arguments.releaseSteps << ",\n"
           << "    \"release_policy\": \"right grip free; left grip remains fixed; physical state preserved\",\n"
           << "    \"local_material_newton_iterations\": " << arguments.localMaterialIterations << ",\n"
           << "    \"newton_iteration_budget\": "
           << compiled.world.mixedSolver.nonlinearIterations.x << ",\n"
           << "    \"fgmres_restart\": "
           << compiled.world.mixedSolver.nonlinearIterations.y << ",\n"
           << "    \"fgmres_iteration_budget\": "
           << compiled.world.mixedSolver.nonlinearIterations.z << ",\n"
           << "    \"fgmres_preconditioner\": \""
           << (compiled.world.mixedSolver.executionBudgets.z == 4u
                   ? "regional_galerkin_contact_rap"
                   : compiled.world.mixedSolver.executionBudgets.z == 3u
                   ? "regional_galerkin_block_jacobi"
                   : compiled.world.mixedSolver.executionBudgets.z == 2u
                   ? "regional_galerkin_rap"
                   : compiled.world.mixedSolver.executionBudgets.z == 1u
                       ? "regional_tangent_fem_diagonal" : "scalar_diagonal")
           << "\",\n"
           << "    \"fgmres_preconditioner_rebuild_policy\": \""
           << (compiled.world.mixedSolver.executionBudgets.z >= 1u
                   ? "once_per_newton_linearization"
                   : "legacy_each_krylov_restart_cycle")
           << "\",\n"
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
           << "    \"gravity_m_s2\": ["
           << mesh.world.gravity[0] << ',' << mesh.world.gravity[1] << ','
           << mesh.world.gravity[2] << "],\n"
           << "    \"maximum_rate_exponent\": 0,\n"
           << "    \"runtime_execution\": "
           << (arguments.compileOnly ? "false" : "true") << "\n"
           << "  },\n"
           << "  \"paper_projection_residual_scaling\": {\"enabled\": "
           << (arguments.scalePaperProjectionResiduals ? "true" : "false")
           << ", \"scope\": \"left-scales the exact 13 implicit Hill state residuals only; no energy, yield, history, or global tolerance change\", \"liner_row_scales\": ";
    const auto writeProjectionScales = [&](const std::array<double, 13>& scales) {
        if (!arguments.scalePaperProjectionResiduals) {
            output << "null";
            return;
        }
        output << '[';
        for (std::size_t index = 0u; index < scales.size(); ++index) {
            if (index != 0u) output << ',';
            output << scales[index];
        }
        output << ']';
    };
    writeProjectionScales(arguments.linerProjectionScales);
    output << ", \"medium_row_scales\": ";
    writeProjectionScales(arguments.mediumProjectionScales);
    output << "},\n"
           << "  \"paper_hill_accurate_projection\": {\"enabled\": "
           << (arguments.accuratePaperProjection ? "true" : "false")
           << ", \"scope\": \"opt-in tighter local implicit-state projection tolerance for source-ordered 13-state Hill materials only; requires compensated arithmetic; global solver tolerances and constitutive equations unchanged\", \"liner_and_medium_only\": true},\n"
           << "  \"generic_compensated_implicit_projection\": {\"enabled\": "
           << (arguments.compensatedImplicitProjection ? "true" : "false")
           << ", \"scope\": \"opt-in generic implicit projection policy applied only to generated retained-matrix liner/medium overlays; authored source Hill materials and finite glue are unchanged\", \"overlay_material_indices\": [";
    for (std::size_t index = 0u;
         index < mesh.damageOverlayMaterialIndices.size(); ++index)
        output << (index == 0u ? "" : ",")
               << mesh.damageOverlayMaterialIndices[index];
    output << "]},\n"
           << "  \"corrugator_starch_interface\": {\"enabled\": "
           << (mesh.starchInterfaceParameters ? "true" : "false")
           << ", \"paired_face_count\": " << mesh.starchPairedFaces.size()
           << ", \"parameter_file\": \""
           << jsonEscape(arguments.starchInterfaceParameters.string())
           << "\", \"parameter_file_sha256\": \""
           << arguments.starchInterfaceParametersSha256
           << "\", \"source_class\": \""
           << (mesh.starchInterfaceParameters
                   ? jsonEscape(mesh.starchInterfaceParameters->sourceClass)
                   : std::string("not_provided"))
           << "\", \"activation_event\": \""
           << (mesh.starchInterfaceParameters
                   ? jsonEscape(mesh.starchInterfaceParameters->activationEvent)
                   : std::string("not_requested"))
           << "\", \"activation_step_role\": \"after an accepted zero-displacement reference step; activation request is transactional and consumes no scoring stroke\""
           << ", \"activation_gap_tolerance_m\": "
           << (mesh.starchInterfaceParameters
                   ? mesh.starchInterfaceParameters->activationGapToleranceM : 0.0)
           << ", \"material_grade_calibrated\": false, \"closure_hma_parameter_file_supplied\": "
           << (arguments.shippingHmaLaw ? "true" : "false") << "},\n"
           << "  \"retained_matrix_damage_overlay\": {\"enabled\": "
           << (mesh.damageOverlayProgramCount > 0u ? "true" : "false")
           << ", \"candidate_only\": true, \"shipping_physics_qualified\": false, \"source_grade_calibrated\": false, \"liner_parameter_file\": \""
           << jsonEscape(arguments.linerDamageParameters.string())
           << "\", \"liner_parameter_file_sha256\": \""
           << arguments.linerDamageParametersSha256
           << "\", \"liner_set_id\": \""
           << (arguments.linerDamageSet
                   ? jsonEscape(arguments.linerDamageSet->setId) : std::string{})
           << "\", \"liner_evidence_class\": \""
           << (arguments.linerDamageSet
                   ? jsonEscape(arguments.linerDamageSet->evidenceClass)
                   : std::string{})
           << "\", \"medium_parameter_file\": \""
           << jsonEscape(arguments.mediumDamageParameters.string())
           << "\", \"medium_parameter_file_sha256\": \""
           << arguments.mediumDamageParametersSha256
           << "\", \"medium_set_id\": \""
           << (arguments.mediumDamageSet
                   ? jsonEscape(arguments.mediumDamageSet->setId) : std::string{})
           << "\", \"medium_evidence_class\": \""
           << (arguments.mediumDamageSet
                   ? jsonEscape(arguments.mediumDamageSet->evidenceClass)
                   : std::string{})
           << "\", \"representative_program_count\": "
           << mesh.damageOverlayProgramCount
           << ", \"substrate_tetrahedra_remapped\": "
           << mesh.damageOverlayTetrahedronCount
           << ", \"mapping_sha256\": \""
           << mesh.damageOverlayMappingDigest
           << "\", \"sidecar\": \""
           << (mesh.damageOverlayProgramCount > 0u
                   ? "damage-overlay-program-pool.json" : "")
           << "\", \"sidecar_sha256\": \""
           << mesh.damageOverlaySidecarSha256
           << "\", \"parameter_contract\": \"all values are explicit source-bound feasible-set inputs; no default or corrugated-grade calibration is inferred\"},\n"
           << "  \"shipping_hma_closure\": {\"enabled\": "
           << (arguments.shippingHmaLaw ? "true" : "false")
           << ", \"parameter_file\": \""
           << jsonEscape(arguments.shippingHmaParameters.string())
           << "\", \"parameter_file_sha256\": \""
           << arguments.shippingHmaParametersSha256
           << "\", \"source_class\": \""
           << (arguments.shippingHmaLaw
                   ? jsonEscape(arguments.shippingHmaLaw->sourceClass)
                   : std::string("not_provided"))
           << "\", \"source_citation\": \""
           << (arguments.shippingHmaLaw
                   ? jsonEscape(arguments.shippingHmaLaw->sourceCitation)
                   : std::string{})
           << "\", \"transfer_uncertainty_fraction\": "
           << (arguments.shippingHmaLaw
                   ? arguments.shippingHmaLaw->transferUncertaintyFraction : 0.0)
           << ", \"mortar_patch_count\": " << mesh.hmaClosureMesh.patches.size()
           << ", \"patch_group_count\": " << mesh.hmaClosureMesh.groups.size()
           << ", \"source_grade_calibrated\": false},\n"
           ;
    if (arguments.pinAdhesion) {
        const auto sourceRoot = std::filesystem::path(
            NUMI_CARDBOARD_MATERIAL_DIR).parent_path() /
            "examples/cardboard/source-data";
        const auto profile = sourceRoot / "pin_adhesion_fixture_profile.json";
        const auto article = sourceRoot / "johnson_popil_2015_pin_protocol_record.md";
        const auto forcePlot = sourceRoot / "johnson_popil_2016_pin_force_displacement_digitized.csv";
        const auto strengthPlot = sourceRoot / "johnson_popil_2016_pin_strength_summary_digitized.csv";
        require(std::filesystem::is_regular_file(profile) &&
                    std::filesystem::is_regular_file(article) &&
                    std::filesystem::is_regular_file(forcePlot) &&
                    std::filesystem::is_regular_file(strengthPlot),
                "pin-adhesion source profile or cited source artifacts are unavailable");
        const auto& fixture = *mesh.pinFixture;
        output << "  \"pin_adhesion\": {\"enabled\": true,\n"
               << "    \"source\": {\"profile_path\": \""
               << jsonEscape(profile.string()) << "\", \"profile_sha256\": \""
               << hexDigest(sha256File(profile))
               << "\", \"article_protocol_path\": \""
               << jsonEscape(article.string()) << "\", \"article_protocol_sha256\": \""
               << hexDigest(sha256File(article))
               << "\", \"presentation_force_plot_path\": \""
               << jsonEscape(forcePlot.string()) << "\", \"presentation_force_plot_sha256\": \""
               << hexDigest(sha256File(forcePlot))
               << "\", \"presentation_strength_plot_path\": \""
               << jsonEscape(strengthPlot.string()) << "\", \"presentation_strength_plot_sha256\": \""
               << hexDigest(sha256File(strengthPlot)) << "},\n"
               << "    \"published_protocol\": {\"specimen_width_m\": "
               << fixture.spec.specimenWidthM << ", \"specimen_length_m\": "
               << fixture.spec.specimenLengthM
               << ", \"relative_platen_rate_m_per_s\": "
               << fixture.spec.relativePlatenRateMPerSecond
               << ", \"pin_count\": 6, \"remaining_board_pin_ordinals_1_based\": [1,3,5], \"selected_facing_pin_ordinals_1_based\": [2,4,6], \"selected_facing\": \"bottom liner\", \"md_cd_orientation\": \"unreported by source; axes are caller-authored\"},\n"
               << "    \"engineering_fixture\": {\"dimension_basis\": \"engineering_prototype\", \"board_caliper_m\": "
               << fixture.spec.boardCaliperM << ", \"pin_radius_m\": "
               << fixture.spec.pinRadiusM << ", \"pin_pitch_across_width_m\": "
               << fixture.spec.pinPitchAcrossWidthM << ", \"pin_end_overhang_m\": "
               << fixture.spec.pinEndOverhangM << ", \"initial_surface_clearance_m\": "
               << fixture.spec.initialClearanceM << ", \"maximum_relative_opening_m\": "
               << arguments.pinMaximumOpeningM << ", \"withdrawal_distance_m\": "
               << arguments.pinWithdrawalDistanceM << ", \"symmetric_motion_split\": true, \"flute_orientation\": \"profile axis aligned to coupon width by explicit modeling choice\"},\n"
               << "    \"instrument_convention\": \"absolute summed signed z impulse on selected-facing pins divided by step dt; opposite platen reactions are not summed; rejected attempts are excluded\",\n"
               << "    \"work_convention\": \"trapezoidal integral over accepted relative-opening, withdrawal, and release samples only; no fitted displacement origin\",\n"
               << "    \"source_limits\": \"article and presentation digitization are distinct; pointwise correspondence and plotted displacement origin are unresolved; source does not report paper/core failure locus\",\n"
               << "    \"recipe_transfer\": \"Haj-Ali liner/NSSC and starch inputs plus Popil caliper are cross-source analogues, not the Johnson-Popil pin specimen grade\",\n"
               << "    \"mixed_interface_enabled\": "
               << (mesh.starchInterfaceParameters ? "true" : "false")
               << ", \"calibration_qualified\": false, \"physical_validation\": false, \"source_failure_locus_available\": false, \"native_damage_classification\": \"stable interface ID, raw state and flags only; no paper/core inference\"},\n";
    } else {
        output << "  \"pin_adhesion\": {\"enabled\": false},\n";
    }
    output << "  \"status_diagnostics_semantics\": \"CSV retains NMMatterStatusGPU.diagnostics x/y/z/w verbatim. Successful finalization writes minimum J, maximum stress, scheduler numerical.y, and maximum contact speed; scheduler numerical.y is not populated as a successful KKT residual in this probe path and may remain zero. Failure statuses replace the vector with the originating failure-site float4 payload (for example, code 10 uses relative residual, relative correction, volume residual, and pressure residual). Independent min_J and max_J are recomputed from accepted native FEM kinematics (stored displacement for opted-in elements, positions for legacy); status_diagnostic_z must not be interpreted as the successful-state KKT residual.\",\n"
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
    return a.steps + a.holdSteps + a.unloadSteps + a.relaxSteps + a.releaseSteps;
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
    step -= a.unloadSteps;
    if (step < a.relaxSteps) return {0.0, "relaxation"};
    return {0.0, "release"};
}

struct Metrics {
    double minJ = INFINITY;
    double maxJ = -INFINITY;
    double maxDisplacementM = 0.0;
    double maxFreeDisplacementM = 0.0;
    double maxFreeSpeedMps = 0.0;
    double kineticEnergyJ = 0.0;
    double maxStateChange = 0.0;
    std::uint32_t currentFixedNodes = 0u;
};

[[maybe_unused]] double femKineticEnergy(const RuntimeStateSnapshot& snapshot,
                        const std::uint32_t nodeCount,
                        const std::uint32_t environment) {
    if (nodeCount == 0u || environment >
            std::numeric_limits<std::size_t>::max() / nodeCount)
        throw std::invalid_argument("FEM kinetic-energy node range is invalid");
    const std::size_t base = static_cast<std::size_t>(environment) * nodeCount;
    if (snapshot.femNodes.size() < base + nodeCount)
        throw std::invalid_argument("FEM kinetic-energy snapshot is truncated");
    double energy = 0.0;
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        const auto& state = snapshot.femNodes[base + node];
        const double mass = state.positionAndMass.w;
        const double vx = state.velocityAndInverseMass.x;
        const double vy = state.velocityAndInverseMass.y;
        const double vz = state.velocityAndInverseMass.z;
        if (!(std::isfinite(mass) && mass >= 0.0 &&
              std::isfinite(vx) && std::isfinite(vy) && std::isfinite(vz)))
            throw std::invalid_argument("FEM kinetic-energy state is nonfinite or has negative mass");
        energy += 0.5 * mass * (vx * vx + vy * vy + vz * vz);
    }
    if (!std::isfinite(energy))
        throw std::overflow_error("FEM kinetic-energy sum is nonfinite");
    return energy;
}

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
        if (snapshot.femNodes[nodeBase + node].restAndFixed.w == 0.0f) {
            metrics.maxFreeDisplacementM = std::max(metrics.maxFreeDisplacementM, distance);
            metrics.maxFreeSpeedMps = std::max(metrics.maxFreeSpeedMps, std::sqrt(speed2));
        } else { ++metrics.currentFixedNodes; }
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
        const bool displacementMode = (tetrahedron.identity.w & NM_TETRAHEDRON_REFERENCE_DISPLACEMENT_GRADIENT) != 0u;
        const auto coordinate = [&](const unsigned index) {
            const auto& node = snapshot.femNodes[nodeBase + indices[index]];
            return displacementMode ? node.referenceDisplacementAndMode : node.positionAndMass;
        };
        const nm_float4 p0 = coordinate(0), p1 = coordinate(1), p2 = coordinate(2), p3 = coordinate(3);
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
        if (displacementMode) { deformation[0] += 1.0; deformation[4] += 1.0; deformation[8] += 1.0; }
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
    targets.assign(static_cast<std::size_t>(arguments.environmentCount) * nodeCount,
                   nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
    const double angle = loadingPoint(arguments, step).angleDegrees * kPi / 180.0;
    std::vector<bool> left(nodeCount, false);
    std::vector<bool> right(nodeCount, false);
    for (const std::uint32_t node : mesh.leftGripNodes) left[node] = true;
    for (const std::uint32_t node : mesh.rightGripNodes) right[node] = true;
    for (std::uint32_t environment = 0u;
         environment < arguments.environmentCount; ++environment) {
        const double environmentAngle = environment == 0u ? angle : 0.0;
        const std::size_t base = static_cast<std::size_t>(environment) * nodeCount;
        for (std::uint32_t node = 0u; node < nodeCount; ++node) {
            if (!left[node] && !right[node]) continue;
            if (right[node] && std::string_view(loadingPoint(arguments, step).phase) == "release") {
                targets[base + node] = {0.0f, 0.0f, 0.0f, 2.0f};
                continue;
            }
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

double indentationAtEnd(const Arguments& a, std::uint32_t step) {
    if (step < a.steps) return a.indentationM * (step + 1u) / a.steps;
    step -= a.steps;
    if (step < a.holdSteps) return a.indentationM;
    step -= a.holdSteps;
    if (step < a.unloadSteps) return a.indentationM * (1.0 - double(step + 1u) / a.unloadSteps);
    return 0.0;
}

std::vector<MRBodyStateGPU> toolingBodies(const Arguments& a, const MeshSource& mesh,
                                        const std::uint32_t step) {
    if (!mesh.tooling) return {};
    std::vector<MRBodyStateGPU> bodies(a.environmentCount);
    const double start = step == 0u ? 0.0 : indentationAtEnd(a, step - 1u);
    const double end = indentationAtEnd(a, step);
    const auto pose = mesh.tooling->punchInitialPose;
    for (std::uint32_t e = 0u; e < a.environmentCount; ++e) {
        bodies[e].position = {float(pose.translation.x), float(pose.translation.y),
            float(pose.translation.z - (e == 0u ? start : 0.0)), 0.0f};
        bodies[e].orientation = {0.0f, 0.0f, 0.0f, 1.0f};
        bodies[e].flagsAndIndices[0] = MR_MOTION_KINEMATIC;
        bodies[e].linearVelocityAndInverseMass = {0.0f, 0.0f,
            float(e == 0u ? -(end - start) / a.timestepSeconds : 0.0), 0.0f};
    }
    return bodies;
}

struct RunState {
    RuntimeStateSnapshot snapshot;
    std::vector<nm_float4> reactions;
    std::vector<nm_float4> contactReactions;
    std::vector<nm_float4> residuals;
    std::array<double, 3> platenContactImpulseNs{};
    std::array<double, 3> impactorContactImpulseNs{};
    double platenContactSamples = 0.0;
    double impactorContactSamples = 0.0;
    double rigidContactSamples = 0.0;
    double supportFloorContactSamples = 0.0;
    double forbiddenRigidToolContactSamples = 0.0;
    double platenMeanForceY = 0.0;
    double impactorMeanForceY = 0.0;
};

class NativeRun {
public:
    explicit NativeRun(const CompiledWorld& world,
                       const std::filesystem::path& metallib,
                       const bool shippingMemoryGate = false)
        : world_(world), environmentCount_(world.dispatch.environmentCount),
          shippingMemoryGate_(shippingMemoryGate) {
        require(environmentCount_ > 0u,
                "compiled Matter world has no environments");
        device_ = MTLCreateSystemDefaultDevice();
        require(device_ != nil, "Apple Metal device is unavailable");
        const std::string deviceName = [[device_ name] UTF8String];
        require(deviceName.find("Apple") != std::string::npos &&
                    deviceName.find("Paravirtual") == std::string::npos,
                "a physical Apple Metal device is required for the native run");
        queue_ = [device_ newCommandQueue];
        require(queue_ != nil, "cannot create Metal command queue");

        RuntimeConfiguration configuration;
        configuration.metallib = metallib.empty()
            ? std::filesystem::path(NUMI_CARDBOARD_METALLIB) : metallib;
        configuration.environmentCount = environmentCount_;
        configuration.captureEvents = true;
        configuration.captureDiagnostics = true;
        configuration.adaptiveTransfer = false;
        initialized_ = runtime_.initialize(
            world_, configuration);
        require(initialized_.encoded,
                "Matter runtime initialization failed: " + initialized_.message);
        nodeCount_ = world_.dispatch.femNodeCount;
        const std::vector<MRMetalWorldStatusGPU> initialStatuses =
            statusRecords();
        environmentStatuses_ = sharedBuffer(device_, initialStatuses,
                                            "environment status");
        const std::vector<nm_float4> initialTargets(
            static_cast<std::size_t>(environmentCount_) * nodeCount_,
            nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
        targets_ = sharedBuffer(device_, initialTargets,
                                "FEM kinematic target");
        reactions_ = [device_ newBufferWithLength:
            static_cast<std::size_t>(environmentCount_) * nodeCount_ *
                sizeof(nm_float4)
            options:MTLResourceStorageModeShared];
        contactReactions_ = [device_ newBufferWithLength:
            static_cast<std::size_t>(environmentCount_) * nodeCount_ *
                sizeof(nm_float4)
            options:MTLResourceStorageModeShared];
        residuals_ = [device_ newBufferWithLength:
            static_cast<std::size_t>(environmentCount_) * nodeCount_ * sizeof(nm_float4)
            options:MTLResourceStorageModeShared];
        require(residuals_ != nil, "cannot allocate FEM residual readback buffer");
        require(reactions_ != nil,
                "cannot allocate FEM reaction readback buffer");
        require(contactReactions_ != nil,
                "cannot allocate FEM contact-reaction readback buffer");
#ifdef NUMI_CARDBOARD_METAL_WORLD
        if (shippingMemoryGate_) {
            std::uint32_t bodyCount = 0u;
            for (const NMRigidProxyGPU& proxy : world_.contact.rigidProxies)
                if (proxy.bodyIndex != NM_INVALID_INDEX)
                    bodyCount = std::max(bodyCount, proxy.bodyIndex + 1u);
            require(bodyCount > 0u,
                    "shipping rigid world has no compiled body-index range");
            shippingRigidBodyCount_ = bodyCount;
            const std::vector<MRABABodyWrenchGPU> zeroWrenches(bodyCount);
            shippingExternalWrenches_ = sharedBuffer(
                device_, zeroWrenches,
                "shipping transaction-scoped rigid external wrench");
        }
#endif
        lastMemoryObservation_ = memoryObservation();
    }

#ifdef NUMI_CARDBOARD_METAL_WORLD
    void enableShippingRigidWorld(
        const std::span<const MRBodyStateGPU> initialToolBodies,
        const numi::cardboard::ShippingRigidBodySpec& bodySpec) {
        require(shippingMemoryGate_,
                "finite-mass shipping coupling is available only for a shipping run");
        require(environmentCount_ == 1u,
                "shipping rigid-world bridge currently requires one environment");
        auto bundle = numi::cardboard::makeShippingRigidWorld(
            initialToolBodies, bodySpec);
        const std::string geometryMismatch =
            numi::cardboard::shippingRigidBranchGeometryMismatchReason(
                world_.contact.rigidProxies, bundle, bodySpec);
        require(geometryMismatch.empty(), geometryMismatch);
        shippingRigidRunner_ = std::make_unique<
            numi::cardboard::ShippingRigidWorldRunner>(
                std::move(bundle), runtime_, runtime_.timestepSeconds(), world_);
    }

    void restoreShippingCheckpoint(const RuntimeStateSnapshot& checkpoint) {
        const auto restored = runtime_.restore(checkpoint);
        require(restored.encoded,
                "shipping branch could not restore its exact accepted FEM checkpoint: " +
                    restored.message);
    }

    void setShippingFreeBodyState(
        const std::array<double, 3>& center,
        const std::array<double, 3>& linearVelocity = {}) {
        require(shippingRigidRunner_ != nullptr,
                "shipping finite-mass runner is unavailable");
        shippingRigidRunner_->setFreeBodyState(
            center, {0.0, 0.0, 0.0, 1.0}, linearVelocity);
    }

    void restoreShippingRigidSceneBodies(
        const std::span<const MRBodyStateGPU> sceneBodies) {
        require(shippingRigidRunner_ != nullptr,
                "shipping finite-mass runner is unavailable");
        shippingRigidRunner_->restoreAcceptedSceneBodies(sceneBodies);
    }
#endif

    RunState step(const std::uint32_t index,
                  const std::vector<nm_float4>& targets,
                  const std::vector<MRBodyStateGPU>& bodies = {},
                  const std::vector<NMCohesiveActivationCommandGPU>&
                      cohesiveActivationCommands = {},
                  const double timestepOverrideSeconds = 0.0,
                  const bool captureRigidContactEvidence = false,
                  const std::optional<double> shippingExternalForceYNewtons =
                      std::nullopt) {
        (void)captureRigidContactEvidence;
#ifndef NUMI_CARDBOARD_METAL_WORLD
        (void)shippingExternalForceYNewtons;
#endif
        const double stepTimestepSeconds = timestepOverrideSeconds > 0.0
            ? timestepOverrideSeconds : runtime_.timestepSeconds();
        require(std::isfinite(stepTimestepSeconds) && stepTimestepSeconds > 0.0,
                "native step timestep must be finite and positive");
        if (shippingMemoryGate_) {
            refreshShippingMemoryObservation();
            numi::cardboard::enforceShippingPeakMemoryLimit(lastMemoryObservation_);
        }
        require(targets.size() ==
                    static_cast<std::size_t>(environmentCount_) * nodeCount_,
                "kinematic target count differs from compiled FEM state");
        std::memcpy(targets_.contents, targets.data(),
                    targets.size() * sizeof(nm_float4));
        auto* statusValues = static_cast<MRMetalWorldStatusGPU*>(
            environmentStatuses_.contents);
        for (std::uint32_t environment = 0u;
             environment < environmentCount_; ++environment) {
            statusValues[environment] = {};
            statusValues[environment].environment = environment;
        }

        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        require(command != nil, "cannot create Metal command buffer");
        EncodeRequest request;
        request.commandBuffer = (__bridge void*)command;
        request.environmentStatuses = (__bridge void*)environmentStatuses_;
        request.femKinematicTargets = (__bridge void*)targets_;
        request.femKinematicTargetCount = environmentCount_ * nodeCount_;
        id<MTLBuffer> activationBuffer = nil;
        if (!cohesiveActivationCommands.empty()) {
            require(cohesiveActivationCommands.size() <=
                        std::numeric_limits<std::uint32_t>::max(),
                    "cohesive activation command count exceeds native ABI capacity");
            activationBuffer = sharedBuffer(
                device_, cohesiveActivationCommands,
                "accepted-state cardboard starch activation commands");
            request.cohesiveActivationCommands = (__bridge void*)activationBuffer;
            request.cohesiveActivationCommandCount =
                static_cast<std::uint32_t>(cohesiveActivationCommands.size());
            request.cohesiveActivationCommandStride =
                sizeof(NMCohesiveActivationCommandGPU);
        }
        id<MTLBuffer> toolBodies = nil;
        if (!bodies.empty()) {
            require(bodies.size() % environmentCount_ == 0u,
                    "tool body arena is not environment-major");
            const std::uint32_t bodyCount = static_cast<std::uint32_t>(
                bodies.size() / environmentCount_);
            require(bodyCount > 0u, "tool body arena is empty");
            // Backward Euler contact must be evaluated at the prescribed
            // endpoint, just like FEM candidate positions. The velocity remains
            // the finite difference of the prescribed trajectory. Supplying a
            // start pose here solves a different obstacle and leaves endpoint
            // penetration for postCommit to reject.
            auto endpoints = bodies;
            for (auto& body : endpoints) {
                body.position.x += stepTimestepSeconds * body.linearVelocityAndInverseMass.x;
                body.position.y += stepTimestepSeconds * body.linearVelocityAndInverseMass.y;
                body.position.z += stepTimestepSeconds * body.linearVelocityAndInverseMass.z;
            }
            toolBodies = sharedBuffer(device_, endpoints, "prescribed end-pose punch bodies");
            request.rigid.currentBodies = (__bridge void*)toolBodies;
            request.rigid.currentBodyCount = bodyCount;
            request.rigid.currentBodyStride = bodyCount;
        }
#ifdef NUMI_CARDBOARD_METAL_WORLD
        if (shippingRigidRunner_ != nullptr) {
            require(bodies.size() ==
                        shippingRigidRunner_->toolSceneBodyCount(),
                    "shipping rigid bridge requires one endpoint for each compiled tool body");
            // `bodies` follows the same caller contract as the direct Matter
            // path above: every entry is the accepted start pose plus its
            // prescribed velocity.  Derive the endpoint once here and keep
            // the velocity supplied by the caller; treating the start pose as
            // an endpoint silently froze all six reusable score-tool bodies.
            std::vector<MRBodyStateGPU> sceneTargets = bodies;
            const auto& acceptedScenes = shippingRigidRunner_->sceneBodies();
            require(acceptedScenes.size() == sceneTargets.size() + 1u,
                    "shipping rigid scene checkpoint has an unexpected body count");
            const float dt = static_cast<float>(stepTimestepSeconds);
            require(std::isfinite(dt) && dt > 0.0f,
                    "shipping rigid bridge timestep is invalid");
            const auto nearStartCoordinate = [](const float requested,
                                                const float accepted) {
                const float tolerance = 4.0f *
                    std::numeric_limits<float>::epsilon() *
                    std::max({1.0f, std::abs(requested), std::abs(accepted)});
                return std::isfinite(requested) && std::isfinite(accepted) &&
                    std::abs(requested - accepted) <= tolerance;
            };
            for (std::size_t bodyIndex = 0u;
                 bodyIndex < sceneTargets.size(); ++bodyIndex) {
                auto& target = sceneTargets[bodyIndex];
                const auto& start = acceptedScenes[bodyIndex];
                require(target.flagsAndIndices[0] == MR_MOTION_KINEMATIC,
                        "shipping tool endpoint is not kinematic");
                require(nearStartCoordinate(target.position.x, start.position.x) &&
                            nearStartCoordinate(target.position.y, start.position.y) &&
                            nearStartCoordinate(target.position.z, start.position.z),
                        "shipping tool command does not begin at its accepted scene pose");
                require(std::abs(target.orientation.x - start.orientation.x) <= 1.0e-6f &&
                            std::abs(target.orientation.y - start.orientation.y) <= 1.0e-6f &&
                            std::abs(target.orientation.z - start.orientation.z) <= 1.0e-6f &&
                            std::abs(target.orientation.w - start.orientation.w) <= 1.0e-6f,
                        "shipping shared score tool attempted an unsupported body rotation");
                const float vx = target.linearVelocityAndInverseMass.x;
                const float vy = target.linearVelocityAndInverseMass.y;
                const float vz = target.linearVelocityAndInverseMass.z;
                require(std::isfinite(vx) && std::isfinite(vy) &&
                            std::isfinite(vz) &&
                            std::abs(target.angularVelocity.x) <= 1.0e-7f &&
                            std::abs(target.angularVelocity.y) <= 1.0e-7f &&
                            std::abs(target.angularVelocity.z) <= 1.0e-7f,
                        "shipping kinematic tool command has invalid velocity or rotation");
                target.position.x = start.position.x + dt * vx;
                target.position.y = start.position.y + dt * vy;
                target.position.z = start.position.z + dt * vz;
                target.linearVelocityAndInverseMass = {vx, vy, vz, 0.0f};
                target.orientation = start.orientation;
                target.angularVelocity = {0.0f, 0.0f, 0.0f, 0.0f};
            }
            require(shippingExternalWrenches_ != nil,
                    "shipping external wrench buffer was not initialized");
            const std::uint32_t bodyCount = shippingRigidBodyCount_;
            const std::uint32_t freeBodyIndex =
                shippingRigidRunner_->freeBodyIndex();
            require(bodyCount == freeBodyIndex + 1u &&
                        freeBodyIndex < bodyCount,
                    "shipping wrench/body index layout is inconsistent");
            auto* wrenches = static_cast<MRABABodyWrenchGPU*>(
                shippingExternalWrenches_.contents);
            std::fill(wrenches, wrenches + bodyCount,
                      MRABABodyWrenchGPU{});
            const auto gravity = shippingRigidRunner_->gravityMPerS2();
            const auto forceTerms =
                numi::cardboard::makeShippingBodyWrenchTerms(
                    shippingRigidRunner_->freeBodyMassKg(), gravity[1],
                    shippingExternalForceYNewtons.value_or(0.0));
            // OwnsFreeBodyCandidate transfers the complete generalized RHS
            // to Matter. Include body weight here; MetalWorld deliberately
            // does not apply its gravity a second time to this candidate.
            wrenches[freeBodyIndex] = forceTerms.totalWrench;
            void* externalWrenchBuffer =
                (__bridge void*)shippingExternalWrenches_;
            const std::uint32_t externalWrenchCount = bodyCount;
            const std::uint32_t externalWrenchStride = bodyCount;
            const auto coupled = shippingRigidRunner_->step(
                sceneTargets, (__bridge void*)targets_, environmentCount_ * nodeCount_,
                (__bridge void*)activationBuffer,
                static_cast<std::uint32_t>(cohesiveActivationCommands.size()),
                externalWrenchBuffer, externalWrenchCount,
                externalWrenchStride, index,
                static_cast<float>(stepTimestepSeconds),
                false);
            require(coupled.diagnostics.succeeded(),
                    "MetalWorld/Matter coupled shipping step failed: " +
                        coupled.diagnostics.message);
            require(coupled.matterAcceptance.accepted,
                    "embedded Matter shipping step was not accepted: " +
                        coupled.matterAcceptance.reason);
            id<MTLCommandBuffer> copyCommand = [queue_ commandBuffer];
            require(copyCommand != nil,
                    "cannot create coupled reaction readback command buffer");
            id<MTLBuffer> reactionSource =
                (__bridge id<MTLBuffer>)runtime_.femConstraintReactionBuffer();
            id<MTLBuffer> contactReactionSource =
                (__bridge id<MTLBuffer>)runtime_.femConstraintContactForceBuffer();
            id<MTLBuffer> residualSource =
                (__bridge id<MTLBuffer>)runtime_.femMechanicalResidualBuffer();
            require(reactionSource != nil && contactReactionSource != nil &&
                        residualSource != nil,
                    "coupled Matter reaction buffers are unavailable");
            id<MTLBlitCommandEncoder> copy = [copyCommand blitCommandEncoder];
            require(copy != nil, "cannot encode coupled reaction readback");
            const std::size_t readbackBytes =
                static_cast<std::size_t>(environmentCount_) * nodeCount_ *
                sizeof(nm_float4);
            [copy copyFromBuffer:reactionSource sourceOffset:0u toBuffer:reactions_
                destinationOffset:0u size:readbackBytes];
            [copy copyFromBuffer:contactReactionSource sourceOffset:0u
                toBuffer:contactReactions_ destinationOffset:0u size:readbackBytes];
            [copy copyFromBuffer:residualSource sourceOffset:0u toBuffer:residuals_
                destinationOffset:0u size:readbackBytes];
            [copy endEncoding];
            [copyCommand commit];
            [copyCommand waitUntilCompleted];
            require(copyCommand.status == MTLCommandBufferStatusCompleted,
                    "coupled reaction readback command failed");
            RunState result;
            result.snapshot = runtime_.snapshot();
            require(result.snapshot.available, result.snapshot.message);
            const std::span<const NMRigidReactionGPU> nativeReactions(
                result.snapshot.reactions);
            const std::span<const NMRigidProxyGPU> nativeProxies(
                world_.contact.rigidProxies);
            const auto platenReaction =
                numi::cardboard::accumulateShippingNativeBodyReaction(
                    nativeReactions, nativeProxies, environmentCount_, 0u,
                    shippingRigidRunner_->compressionPlatenBodyIndex());
            const auto impactorReaction =
                numi::cardboard::accumulateShippingNativeBodyReaction(
                    nativeReactions, nativeProxies, environmentCount_, 0u,
                    shippingRigidRunner_->freeBodyIndex());
            result.platenContactImpulseNs[1] = platenReaction.impulseNs[1];
            result.impactorContactImpulseNs[1] = impactorReaction.impulseNs[1];
            result.platenContactSamples = platenReaction.contactSamples;
            result.impactorContactSamples = impactorReaction.contactSamples;
            result.platenMeanForceY =
                numi::cardboard::shippingReactionForceYFromImpulse(
                    platenReaction, stepTimestepSeconds);
            result.impactorMeanForceY =
                numi::cardboard::shippingReactionForceYFromImpulse(
                    impactorReaction, stepTimestepSeconds);
            const auto contactGate =
                numi::cardboard::classifyShippingRigidContactSamples(
                    nativeReactions, nativeProxies, environmentCount_,
                    shippingRigidRunner_->supportFloorBodyIndex());
            result.rigidContactSamples = contactGate.totalSamples;
            result.supportFloorContactSamples =
                contactGate.supportFloorSamples;
            result.forbiddenRigidToolContactSamples =
                contactGate.forbiddenToolSamples;
            const auto* reactionData = static_cast<const nm_float4*>(reactions_.contents);
            result.reactions.assign(reactionData, reactionData +
                static_cast<std::size_t>(environmentCount_) * nodeCount_);
            const auto* contactReactionData =
                static_cast<const nm_float4*>(contactReactions_.contents);
            result.contactReactions.assign(contactReactionData,
                contactReactionData +
                    static_cast<std::size_t>(environmentCount_) * nodeCount_);
            const auto* residualData = static_cast<const nm_float4*>(residuals_.contents);
            result.residuals.assign(residualData, residualData +
                static_cast<std::size_t>(environmentCount_) * nodeCount_);
            lastMemoryObservation_ = memoryObservation();
            return result;
        }
#endif
        request.controlStep = index;
        request.physicsSubsteps = 1u;
        request.timestepSeconds = static_cast<float>(stepTimestepSeconds);
        request.runAdaptiveTransfer = false;
        request.phase = EncodePhase::preDynamics;
        const RuntimeDiagnostics pre = runtime_.encode(request);
        require(pre.encoded, "Matter preDynamics encode failed: " + pre.message);

        id<MTLBuffer> reactionSource =
            (__bridge id<MTLBuffer>)runtime_.femConstraintReactionBuffer();
        id<MTLBuffer> contactReactionSource =
            (__bridge id<MTLBuffer>)runtime_.femConstraintContactForceBuffer();
        require(reactionSource != nil && contactReactionSource != nil,
                "Matter FEM constraint/contact reaction buffers are unavailable");
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        require(blit != nil, "cannot create reaction readback blit encoder");
        [blit copyFromBuffer:reactionSource
                sourceOffset:0u
                    toBuffer:reactions_
           destinationOffset:0u
                        size:static_cast<std::size_t>(environmentCount_) *
                            nodeCount_ * sizeof(nm_float4)];
        [blit copyFromBuffer:contactReactionSource
                sourceOffset:0u
                    toBuffer:contactReactions_
           destinationOffset:0u
                        size:static_cast<std::size_t>(environmentCount_) *
                            nodeCount_ * sizeof(nm_float4)];
        id<MTLBuffer> residualSource = (__bridge id<MTLBuffer>)runtime_.femMechanicalResidualBuffer();
        require(residualSource != nil, "Matter FEM residual buffer is unavailable");
        [blit copyFromBuffer:residualSource sourceOffset:0u toBuffer:residuals_
            destinationOffset:0u size:static_cast<std::size_t>(environmentCount_) * nodeCount_ * sizeof(nm_float4)];
        [blit endEncoding];

        // The same immutable prescribed endpoint is both solved and certified.
        // These tools have no free rigid unknown; native FEM owns all response.
        request.phase = EncodePhase::postCommit;
        // Cohesive activation is an explicit preDynamics event only. Replaying
        // it after commit could latch a second reference frame or turn a
        // rejected endpoint into an apparent activation.
        request.cohesiveActivationCommands = nullptr;
        request.cohesiveActivationCommandCount = 0u;
        request.cohesiveActivationCommandStride = 0u;
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
            reactionData + static_cast<std::size_t>(environmentCount_) *
                nodeCount_);
        const auto* contactReactionData =
            static_cast<const nm_float4*>(contactReactions_.contents);
        result.contactReactions.assign(
            contactReactionData,
            contactReactionData + static_cast<std::size_t>(environmentCount_) *
                nodeCount_);
        const auto* residualData = static_cast<const nm_float4*>(residuals_.contents);
        result.residuals.assign(residualData, residualData + static_cast<std::size_t>(environmentCount_) * nodeCount_);
        lastMemoryObservation_ = memoryObservation();
        return result;
    }

    [[nodiscard]] const Runtime& runtime() const { return runtime_; }

#ifdef NUMI_CARDBOARD_METAL_WORLD
    [[nodiscard]] const std::vector<MRBodyStateGPU>& shippingRigidSceneBodies() const noexcept {
        return shippingRigidRunner_ != nullptr
            ? shippingRigidRunner_->sceneBodies() : emptyShippingSceneBodies_;
    }
    [[nodiscard]] double shippingRigidFreeBodyMassKg() const noexcept {
        return shippingRigidRunner_ != nullptr
            ? shippingRigidRunner_->freeBodyMassKg() : 0.0;
    }
    [[nodiscard]] std::uint32_t shippingRigidFreeBodyIndex() const noexcept {
        return shippingRigidRunner_ != nullptr
            ? shippingRigidRunner_->freeBodyIndex() : NM_INVALID_INDEX;
    }
    [[nodiscard]] std::uint32_t shippingRigidToolSceneBodyCount() const noexcept {
        return shippingRigidRunner_ != nullptr
            ? shippingRigidRunner_->toolSceneBodyCount() : 0u;
    }
    [[nodiscard]] std::uint32_t shippingRigidPlatenBodyIndex() const noexcept {
        return shippingRigidRunner_ != nullptr
            ? shippingRigidRunner_->compressionPlatenBodyIndex() : NM_INVALID_INDEX;
    }
    [[nodiscard]] std::uint32_t shippingRigidFloorBodyIndex() const noexcept {
        return shippingRigidRunner_ != nullptr
            ? shippingRigidRunner_->supportFloorBodyIndex() : NM_INVALID_INDEX;
    }
    [[nodiscard]] std::uint64_t shippingRigidWorldFingerprint() const noexcept {
        return shippingRigidRunner_ != nullptr
            ? shippingRigidRunner_->worldFingerprint() : 0u;
    }
    [[nodiscard]] std::array<double, 3> shippingRigidGravityMPerS2() const noexcept {
        return shippingRigidRunner_ != nullptr
            ? shippingRigidRunner_->gravityMPerS2()
            : std::array<double, 3>{};
    }
#endif

    [[nodiscard]] const numi::cardboard::ShippingRuntimeMemoryObservation& lastMemoryObservation() const {
        return lastMemoryObservation_;
    }

    [[nodiscard]] bool shippingMemoryWithinLimit() const {
        return !shippingMemoryGate_ ||
            numi::cardboard::shippingObservedPeakBytes(lastMemoryObservation_) <=
                numi::cardboard::kShippingPeakMemoryLimitBytes;
    }

    void refreshShippingMemoryObservation() {
        if (shippingMemoryGate_) lastMemoryObservation_ = memoryObservation();
    }

    void writeMemoryObservation(const std::filesystem::path& path,
                                const std::uint32_t step,
                                const std::string_view stage) const {
        if (!shippingMemoryGate_) return;
        std::ofstream output(path, std::ios::app);
        require(output.good(), "cannot append shipping memory observations");
        const auto& memory = lastMemoryObservation_;
        output << step << ',' << stage << ',' << memory.runtimeResidentBytes << ','
               << memory.metalCurrentAllocatedBytes << ','
               << memory.metalRecommendedWorkingSetBytes << ','
               << (memory.hostResidentSampleAvailable ? memory.hostResidentBytes : 0u)
               << ',' << (memory.hostResidentSampleAvailable ? 1 : 0) << ','
               << numi::cardboard::shippingObservedPeakBytes(memory) << ','
               << (shippingMemoryWithinLimit() ? 1 : 0) << '\n';
        output.flush();
    }

private:
    numi::cardboard::ShippingRuntimeMemoryObservation memoryObservation() const {
        numi::cardboard::ShippingRuntimeMemoryObservation result;
        result.runtimeResidentBytes = initialized_.residentBytes;
        result.metalCurrentAllocatedBytes = device_.currentAllocatedSize;
        result.metalRecommendedWorkingSetBytes =
            device_.recommendedMaxWorkingSetSize;
        task_vm_info_data_t vmInfo{};
        mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
        const kern_return_t status = task_info(
            mach_task_self(), TASK_VM_INFO,
            reinterpret_cast<task_info_t>(&vmInfo), &count);
        if (status == KERN_SUCCESS && vmInfo.phys_footprint > 0u) {
            result.hostResidentBytes = vmInfo.phys_footprint;
            result.hostResidentSampleAvailable = true;
        }
        return result;
    }

    std::vector<MRMetalWorldStatusGPU> statusRecords() const {
        std::vector<MRMetalWorldStatusGPU> result(environmentCount_);
        for (std::uint32_t environment = 0u;
             environment < environmentCount_; ++environment)
            result[environment].environment = environment;
        return result;
    }

    const CompiledWorld& world_;
    std::uint32_t environmentCount_ = 0u;
    bool shippingMemoryGate_ = false;
    RuntimeDiagnostics initialized_{};
    numi::cardboard::ShippingRuntimeMemoryObservation lastMemoryObservation_{};
    Runtime runtime_;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLBuffer> environmentStatuses_ = nil;
    id<MTLBuffer> targets_ = nil;
    id<MTLBuffer> reactions_ = nil;
    id<MTLBuffer> contactReactions_ = nil;
    id<MTLBuffer> residuals_ = nil;
    std::uint32_t nodeCount_ = 0u;
#ifdef NUMI_CARDBOARD_METAL_WORLD
    id<MTLBuffer> shippingExternalWrenches_ = nil;
    std::uint32_t shippingRigidBodyCount_ = 0u;
    std::vector<MRBodyStateGPU> emptyShippingSceneBodies_;
    std::unique_ptr<numi::cardboard::ShippingRigidWorldRunner>
        shippingRigidRunner_;
#endif
};

numi::cardboard::ShippingStaticMemoryLowerBound shippingStaticMemoryLowerBound(
    const CompiledWorld& world) {
    return numi::cardboard::estimateShippingStaticMemoryLowerBound(
        world.dispatch.femNodeCount, world.dispatch.tetrahedronCount,
        world.dispatch.environmentCount,
        std::clamp(world.mixedSolver.nonlinearIterations.y, 1u,
                   numi::cardboard::kShippingFgmresRestartCeiling));
}

template<class Vector>
std::uint64_t shippingVectorCapacityBytes(const Vector& values) {
    return numi::cardboard::shippingMemoryCheckedMultiply(
        static_cast<std::uint64_t>(values.capacity()),
        static_cast<std::uint64_t>(sizeof(typename Vector::value_type)));
}

template<class World>
std::uint64_t shippingCompiledWorldStorageBytes(const World& world) {
    using numi::cardboard::shippingMemoryCheckedAdd;
    std::uint64_t total = 0u;
    const auto add = [&](const auto& vector) {
        total = shippingMemoryCheckedAdd(total,
            shippingVectorCapacityBytes(vector));
    };
    add(world.constitutive); add(world.materials); add(world.cohesiveMaterials);
    add(world.mixedMaterials); add(world.parameters); add(world.stateInitials);
    add(world.instructions); add(world.scalarPrograms); add(world.objects);
    add(world.mpm.particles); add(world.mpm.nodes); add(world.mpm.grids);
    add(world.mpm.blocks); add(world.mpm.blockLookup); add(world.mpm.stencils);
    add(world.mpm.nodeIncidence); add(world.mpm.nodeRanges);
    add(world.fem.nodes); add(world.fem.tetrahedra); add(world.fem.surfaceFaces);
    add(world.fem.nodeIncidence); add(world.fem.nodeRanges); add(world.fem.capacities);
    add(world.fem.fields); add(world.fem.fieldBoundaries); add(world.fem.topologyNodes);
    add(world.fem.cohesiveFaces); add(world.fem.mutationCommands);
    add(world.fem.punctureChannels); add(world.fem.humanAttachments);
    add(world.contact.rigidProxies); add(world.contact.pairs);
    add(world.contact.nodeIncidence); add(world.contact.nodeRanges);
    add(world.contact.rigidIncidence); add(world.contact.rigidRanges);
    add(world.vascular.species); add(world.vascular.compartments);
    add(world.vascular.connections); add(world.vascular.tissues);
    add(world.vascular.exchanges); add(world.vascular.tissueBindings);
    add(world.vascular.unknowns); add(world.vascular.cavities);
    add(world.vascular.cavityFaces); add(world.vascular.compartmentCavity);
    add(world.vascular.cavityNodeIncidence); add(world.vascular.cavityNodeRanges);
    add(world.vascular.connectionIncidence); add(world.vascular.connectionRanges);
    add(world.vascular.bloodExchangeIncidence); add(world.vascular.bloodExchangeRanges);
    add(world.vascular.tissueExchangeIncidence); add(world.vascular.tissueExchangeRanges);
    add(world.vascular.names); add(world.adaptive); add(world.schedulers);
    add(world.identification); add(world.learnedMaterials); add(world.learnedLayers);
    add(world.learnedWeights);
    // A 25% allowance covers nested names and small vector/string allocations
    // in authored constitutive and source records not represented by these
    // outer vector capacities.
    return numi::cardboard::shippingMemoryCheckedAdd(
        total, total / 4u);
}

std::uint64_t shippingSourceMeshStorageBytes(const MeshSource& mesh) {
    using numi::cardboard::shippingMemoryCheckedAdd;
    std::uint64_t total = 0u;
    const auto add = [&](const auto& vector) {
        total = shippingMemoryCheckedAdd(total,
            shippingVectorCapacityBytes(vector));
    };
    add(mesh.tetrahedra); add(mesh.leftGripNodes); add(mesh.rightGripNodes);
    add(mesh.restNodes); add(mesh.materialFrames); add(mesh.materialIndices);
    add(mesh.authoredRegionIds); add(mesh.damageOverlayMaterialIndices);
    add(mesh.damageOverlayMappings); add(mesh.starchPairedFaces);
    add(mesh.starchActivationCommands); add(mesh.glueFootprints);
    add(mesh.world.materials);
    add(mesh.world.cohesiveMaterials); add(mesh.world.objects);
    add(mesh.world.rigidProxies);
    for (const auto& object : mesh.world.objects) {
        add(object.particles); add(object.femNodes); add(object.femReferenceNodes);
        add(object.femFixedNodes); add(object.femHumanAttachments);
        add(object.femContactNodes); add(object.tetrahedra);
        add(object.femCohesiveInterfaces); add(object.femCohesiveMortarPatches);
        add(object.femMaterialIndices); add(object.femPreconditionerRegionIds);
        add(object.femMaterialFrameRotations); add(object.fieldBoundaries);
        add(object.mutationCommands);
    }
    // The source mesh, authored world and compiled world coexist during the
    // upload transaction. Double the raw vector capacity to cover metadata,
    // nested parameter strings, and temporary builder vectors.
    return numi::cardboard::shippingMemoryCheckedMultiply(total, 2u);
}

numi::cardboard::ShippingMemoryBudgetInput shippingMemoryBudgetInput(
    const Arguments& arguments,
    const MeshSource& mesh,
    const CompileResult& compiled) {
    using namespace numi::cardboard;
    const auto& dispatch = compiled.world.dispatch;
    ShippingMemoryBudgetInput input;
    input.nodeCount = dispatch.femNodeCount;
    input.tetrahedronCount = dispatch.tetrahedronCount;
    input.cohesiveFaceCount = dispatch.cohesiveFaceCount;
    input.contactPairCount = dispatch.contactPairCount;
    input.activeContactCapacity = std::min<std::uint64_t>(
        dispatch.reservedMixed1, dispatch.contactPairCount);
    input.deformableContactCapacity = dispatch.deformableContactCapacity;
    input.compactImmutableSurfaceMap =
        (dispatch.flags & NM_MATTER_STATIC_FEM_SURFACE_MAP) != 0u;
    input.surfaceFaceCount = dispatch.surfaceFaceCount;
    input.gridNodeCount = dispatch.gridNodeCount;
    input.rigidGeneralizedCapacity = dispatch.rigidGeneralizedCapacity;
    input.rigidProxyCount = dispatch.rigidProxyCount;
    for (const auto& proxy : compiled.world.contact.rigidProxies) {
        if (proxy.bodyIndex != NM_INVALID_INDEX)
            input.rigidBodyCount = std::max<std::uint64_t>(
                input.rigidBodyCount,
                static_cast<std::uint64_t>(proxy.bodyIndex) + 1u);
    }
    input.materialStateStride = dispatch.materialStateStride;
    input.environmentCount = dispatch.environmentCount;
    input.fgmresRestart = std::clamp<std::uint64_t>(
        compiled.world.mixedSolver.nonlinearIterations.y, 1u,
        kShippingFgmresRestartCeiling);
    input.regionalGalerkinEnabled =
        compiled.world.mixedSolver.executionBudgets.z == 2u ||
        compiled.world.mixedSolver.executionBudgets.z == 3u ||
        compiled.world.mixedSolver.executionBudgets.z == 4u;
    input.regionalGalerkinBlockFineEnabled =
        compiled.world.mixedSolver.executionBudgets.z == 3u ||
        compiled.world.mixedSolver.executionBudgets.z == 4u;
    input.regionalGalerkinDiagnosticsCapture = input.regionalGalerkinEnabled;
    if (input.regionalGalerkinEnabled) {
        std::uint32_t maximumRegion = 0u;
        for (const auto& tetrahedron : compiled.world.fem.tetrahedra) {
            const auto region = (tetrahedron.identity.w &
                NM_TETRAHEDRON_PRECONDITIONER_REGION_MASK) >>
                NM_TETRAHEDRON_PRECONDITIONER_REGION_SHIFT;
            maximumRegion = std::max(maximumRegion, region);
        }
        input.compactPreconditionerRegionCount =
            compiled.world.fem.tetrahedra.empty() ? 0u :
                static_cast<std::uint64_t>(maximumRegion) + 1u;
    }
    const auto present = [](const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr && value[0] != '\0';
    };
    input.rigidContactArmijoEnabled = !arguments.shippingBox &&
        (present("NM_FEM_CAPSULE_RESIDUAL_ARMIJO_ROOT") ||
         present("NM_FEM_CAPSULE_BOX_RESIDUAL_ARMIJO_ROOT") ||
         present("NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT"));
    input.fullContactArmijoEnabled = !arguments.shippingBox &&
        present("NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT");
    input.cohesiveArmijoEnabled = !arguments.shippingBox &&
        (present("NM_FEM_COHESIVE_RESIDUAL_ARMIJO_ROOT") ||
         present("NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT"));
    input.fullContactArmijoRequired = arguments.shippingBox;
    input.hostSourceMeshBytes = shippingSourceMeshStorageBytes(mesh);
    input.hostCompiledWorldBytes = shippingCompiledWorldStorageBytes(compiled.world);
    input.hostGeneratedMetalBytes = compiled.generatedMetal.capacity();
    input.uploadStagingBytes = input.hostCompiledWorldBytes;
    input.retainedCompleteCheckpointCount = arguments.shippingBox ? 2u : 0u;
    input.estimatedSnapshotPayloadBytes =
        estimateShippingSnapshotPayloadBytes(input);
    return input;
}

numi::cardboard::ShippingMemoryBudget shippingPrebuildMemoryBudget(
    const Arguments& arguments,
    const numi::cardboard::ShippingPrebuildMemoryScreen& screen) {
    using namespace numi::cardboard;
    ShippingPrebuildArenaOptions options;
    options.deformableContactCapacity =
        arguments.deformableContactCapacityProvided
            ? arguments.deformableContactCapacity : 0u;
    options.cohesiveFaceUpperBound =
        (arguments.shippingStarchLaw || arguments.shippingHmaLaw)
            ? shippingMemoryCheckedMultiply(
                  screen.nominalTetrahedronCount, 1u) / 5u : 0u;
    options.rigidProxyCount = arguments.shippingBox
        ? kShippingRigidProxyCount : 0u;
    options.rigidBodyCount = arguments.shippingBox
        ? kShippingRigidBodyCount : 0u;
    options.fgmresRestart = arguments.fgmresRestart;
    options.compactImmutableSurfaceMap = arguments.compactImmutableSurfaces;
    options.requireFullContactResidualDescent = arguments.shippingBox;
    auto budget = estimateShippingPrebuildMemoryBudget(screen, options);
    if (arguments.femPreconditioner ==
            FEMPreconditionerMode::regionalGalerkin ||
        arguments.femPreconditioner ==
            FEMPreconditionerMode::regionalGalerkinBlockJacobi ||
        arguments.femPreconditioner ==
            FEMPreconditionerMode::regionalGalerkinContact) {
        ShippingMemoryBudgetInput rap;
        rap.nodeCount = screen.nominalNodeCount;
        rap.tetrahedronCount = screen.nominalTetrahedronCount;
        rap.deformableContactCapacity = options.deformableContactCapacity == 0u
            ? shippingMemoryCheckedMultiply(rap.tetrahedronCount, 8u)
            : options.deformableContactCapacity;
        rap.compactPreconditionerRegionCount = 3u;
        rap.fgmresRestart = options.fgmresRestart;
        rap.regionalGalerkinEnabled = true;
        rap.regionalGalerkinDiagnosticsCapture = true;
        rap.regionalGalerkinBlockFineEnabled =
            arguments.femPreconditioner ==
                FEMPreconditionerMode::regionalGalerkinBlockJacobi ||
            arguments.femPreconditioner ==
                FEMPreconditionerMode::regionalGalerkinContact;
        const auto rapBudget = estimateShippingMemoryBudget(rap);
        budget.regionalGalerkinBytes = rapBudget.regionalGalerkinBytes;
        budget.regionalGalerkinBlockFineBytes =
            rapBudget.regionalGalerkinBlockFineBytes;
        budget.regionalGalerkinDiagnosticsBytes =
            rapBudget.regionalGalerkinDiagnosticsBytes;
        budget.estimatedMetalPeakBytes = shippingMemoryCheckedAdd(
            budget.estimatedMetalPeakBytes,
            shippingMemoryCheckedAdd(budget.regionalGalerkinBytes,
                                     budget.regionalGalerkinDiagnosticsBytes));
        budget.estimatedPeakDomainBytes = std::max(
            budget.estimatedMetalPeakBytes, budget.estimatedHostPeakBytes);
        budget.withinLimit =
            budget.estimatedPeakDomainBytes <= budget.limitBytes;
    }
    return budget;
}

struct ShippingResidualDescentAdmission {
    bool required = true;
    bool runtimeShapeAvailable = false;
    bool privateArenaAvailable = false;
    bool memoryLimitAvailable = false;
    bool physicsExecutionAdmitted = false;
    // Solver qualification is a separate evidence question. A supported
    // analogue run may execute with an unqualified verdict; the admission
    // gate only proves that the selected production path and its arenas are
    // actually available for the authored world.
    bool solverPathQualified = false;
    std::vector<std::string> blockers;
};

ShippingResidualDescentAdmission shippingResidualDescentAdmission(
    const CompiledWorld& world,
    const numi::cardboard::ShippingMemoryBudget& budget) {
    ShippingResidualDescentAdmission result;
    const auto add = [&](const std::string& reason) {
        result.blockers.push_back(reason);
    };
    const auto& dispatch = world.dispatch;
    if (dispatch.environmentCount > 4u)
        add("environment count exceeds full-contact Armijo cap 4");
    if (dispatch.femNodeCount >
        numi::cardboard::kShippingFullContactArmijoNodeLimit)
        add("FEM node count exceeds full-contact Armijo runtime cap 8192");
    if (dispatch.tetrahedronCount > 32768u)
        add("tetrahedron count exceeds full-contact Armijo cap 32768");
    if (dispatch.deformableContactCapacity > 131072u)
        add("deformable-contact capacity exceeds full-contact Armijo cap 131072");
    if (world.contact.pairs.empty() || world.contact.pairs.size() > 65536u)
        add("rigid contact-pair count is outside the full-contact Armijo range 1..65536");
    if (world.contact.rigidProxies.size() < 2u ||
        world.contact.rigidProxies.size() > 16u)
        add("rigid proxy count is outside the full-contact Armijo range 2..16");
    if (world.fem.cohesiveFaces.empty() ||
        world.fem.cohesiveFaces.size() > 4096u)
        add("full-contact Armijo requires 1..4096 cohesive faces");

    if (world.objects.size() != 1u) {
        add("native full-contact Armijo supports exactly one FEM object");
    } else {
        const auto& object = world.objects.front();
        constexpr std::uint32_t forbidden =
            NM_OBJECT_ADAPTIVE | NM_OBJECT_IDENTIFIABLE |
            NM_OBJECT_MIXED_FEM | NM_OBJECT_MULTIPHYSICS |
            NM_OBJECT_MUTABLE_TOPOLOGY;
        if (object.representation != NM_REPRESENTATION_FEM ||
            (object.flags & forbidden) != 0u)
            add("native full-contact Armijo excludes mixed/adaptive/multiphysics/mutable FEM objects");
        if ((object.flags & (NM_OBJECT_DISABLE_SELF_CONTACT |
                             NM_OBJECT_DISABLE_DEFORMABLE_CONTACT)) != 0u)
            add("native full-contact Armijo requires deformable self-contact enabled");
    }
    if ((dispatch.flags & NM_MATTER_CONTACT) == 0u ||
        dispatch.particleCount != 0u || dispatch.mpmGridCount != 0u ||
        dispatch.mpmBlockCount != 0u || dispatch.mpmActiveNodeCapacity != 0u)
        add("native full-contact Armijo requires contact-enabled FEM without MPM");
    if (dispatch.fieldBoundaryCount != 0u ||
        dispatch.femHumanAttachmentCount != 0u ||
        dispatch.learnedWeightCount != 0u || dispatch.mutationCommandCount != 0u ||
        !world.vascular.unknowns.empty() || !world.vascular.cavities.empty() ||
        !world.vascular.cavityFaces.empty())
        add("native full-contact Armijo excludes field, Human, learned, mutation, and vascular coupling");
    if (dispatch.rigidQCapacity != 0u ||
        (dispatch.rigidGeneralizedCapacity != 0u &&
         dispatch.rigidGeneralizedCapacity != 6u))
        add("native full-contact Armijo supports no rigid unknowns or one six-DOF free body");

    std::uint32_t capsuleCount = 0u;
    std::uint32_t boxCount = 0u;
    std::uint32_t freeBodyCount = 0u;
    for (const auto& proxy : world.contact.rigidProxies) {
        if (proxy.adaptiveObjectIndex != NM_INVALID_INDEX)
            add("native full-contact Armijo excludes adaptive rigid proxies");
        if (proxy.shapeKind == NM_RIGID_CAPSULE) {
            ++capsuleCount;
            if (proxy.flags != NM_RIGID_PRESCRIBED_TRANSLATION ||
                proxy.bodyIndex == NM_INVALID_INDEX)
                add("native full-contact Armijo requires endpoint-prescribed capsule tools");
        } else if (proxy.shapeKind == NM_RIGID_BOX) {
            ++boxCount;
            const bool staticBox = proxy.flags == 0u &&
                proxy.sceneBodyIndex == NM_INVALID_INDEX;
            const bool freeBox = proxy.flags == NM_RIGID_DYNAMIC &&
                proxy.generalizedFreeBodyIndex == 0u &&
                proxy.bodyIndex != NM_INVALID_INDEX &&
                proxy.sceneBodyIndex != NM_INVALID_INDEX;
            if (freeBox) ++freeBodyCount;
            if (!staticBox && !freeBox)
                add("native full-contact Armijo supports only static boxes or one free dynamic box");
        } else {
            add("native full-contact Armijo supports only capsule and box proxies");
        }
    }
    if (capsuleCount == 0u || boxCount == 0u)
        add("native full-contact Armijo requires both capsule and box tools");
    if (freeBodyCount > 1u ||
        ((dispatch.rigidGeneralizedCapacity == 6u) != (freeBodyCount == 1u)))
        add("native full-contact Armijo free-body proxy layout does not match its six-DOF arena");
    bool capsulePair = false;
    bool boxPair = false;
    for (const auto& pair : world.contact.pairs) {
        if (pair.objectIndex != 0u || pair.continuumNode >= dispatch.femNodeCount ||
            pair.rigidProxy >= world.contact.rigidProxies.size()) {
            add("native full-contact Armijo found a non-FEM or out-of-range contact pair");
            continue;
        }
        const auto shape = world.contact.rigidProxies[pair.rigidProxy].shapeKind;
        capsulePair = capsulePair || shape == NM_RIGID_CAPSULE;
        boxPair = boxPair || shape == NM_RIGID_BOX;
    }
    if (!capsulePair || !boxPair)
        add("native full-contact Armijo requires authored contact pairs for both tool shapes");

    result.runtimeShapeAvailable = result.blockers.empty();
    result.privateArenaAvailable = budget.requiredResidualDescentArenaAdmitted;
    result.memoryLimitAvailable = budget.estimatedPeakDomainBytes <= budget.limitBytes;
    if (!result.privateArenaAvailable)
        add(budget.requiredResidualDescentArenaFailure.empty()
                ? "required private residual-descent arenas exceed runtime caps"
                : budget.requiredResidualDescentArenaFailure);
    if (!result.memoryLimitAvailable)
        add("required arena envelope exceeds the 16 GiB shipping budget");
    result.physicsExecutionAdmitted = result.runtimeShapeAvailable &&
        result.privateArenaAvailable && result.memoryLimitAvailable;
    return result;
}

void writeShippingResidualDescentAdmission(
    const std::filesystem::path& path,
    const CompiledWorld& world,
    const numi::cardboard::ShippingMemoryBudget& budget,
    const ShippingResidualDescentAdmission& admission) {
    std::ofstream output(path);
    require(output.good(), "cannot create shipping residual-descent admission report");
    output << "{\n  \"schema\": \"numi.cardboard.shipping-residual-descent-admission.v1\",\n"
           << "  \"evidence_class\": \"compiled runtime-scope and arena admission; not solver qualification\",\n"
           << "  \"required_for_shipping_physics\": true,\n"
           << "  \"required_on_every_physics_transaction\": true,\n"
           << "  \"runtime_selector_policy\": \"NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT=all\",\n"
           << "  \"runtime_shape_available\": "
           << (admission.runtimeShapeAvailable ? "true" : "false") << ",\n"
           << "  \"private_arena_caps_available\": "
           << (admission.privateArenaAvailable ? "true" : "false") << ",\n"
           << "  \"memory_limit_available\": "
           << (admission.memoryLimitAvailable ? "true" : "false") << ",\n"
           << "  \"physics_execution_admitted\": "
           << (admission.physicsExecutionAdmitted ? "true" : "false") << ",\n"
           << "  \"shipping_path_qualified\": "
           << (admission.solverPathQualified ? "true" : "false") << ",\n"
           << "  \"verdict_note\": \"execution admission does not qualify solver accuracy or physical assembly; supported analogue runs remain unqualified until evidence gates pass\",\n"
           << "  \"actual_private_arenas_allocated\": false,\n"
           << "  \"compiled_counts\": {\"nodes\": " << world.dispatch.femNodeCount
           << ", \"tetrahedra\": " << world.dispatch.tetrahedronCount
           << ", \"deformable_contact_capacity\": "
           << world.dispatch.deformableContactCapacity
           << ", \"rigid_contact_pairs\": " << world.contact.pairs.size()
           << ", \"rigid_proxies\": " << world.contact.rigidProxies.size()
           << ", \"cohesive_faces\": " << world.fem.cohesiveFaces.size() << "},\n"
           << "  \"private_arena_estimate_bytes\": {\"full_contact\": "
           << budget.requiredFullContactArmijoBytes
           << ", \"rigid_contact\": "
           << budget.requiredRigidContactArmijoBytes
           << ", \"cohesive\": "
           << budget.requiredCohesiveArmijoBytes << "},\n"
           << "  \"blockers\": [";
    for (std::size_t i = 0u; i < admission.blockers.size(); ++i) {
        if (i) output << ',';
        output << '\"' << jsonEscape(admission.blockers[i]) << '\"';
    }
    output << "]\n}\n";
    require(output.good(), "failed writing shipping residual-descent admission report");
}

bool shippingAllowanceHasMeasuredCouponEvidence(
    const numi::cardboard::ShippingBoxDesign& design) {
    const auto evidenceClass = design.allowanceCoupon.evidenceClass;
    return design.allowanceCoupon.accepted &&
        (evidenceClass == numi::cardboard::CouponEvidenceClass::physicalAcceptedCoupon ||
         evidenceClass == numi::cardboard::CouponEvidenceClass::nativeAcceptedCoupon);
}

void writeShippingPrebuildMemoryScreen(
    const std::filesystem::path& path,
    const Arguments& arguments,
    const numi::cardboard::ShippingPrebuildMemoryScreen& estimate,
    const numi::cardboard::ShippingMemoryBudget& budget) {
    require(arguments.shippingDesign.has_value(),
            "shipping prebuild screen requires a derived design");
    std::ofstream output(path);
    require(output.good(), "cannot create shipping prebuild memory screen");
    const std::uint32_t preconditionerSelector =
        static_cast<std::uint32_t>(arguments.femPreconditioner);
    const bool regionalPreconditioner = preconditionerSelector >= 2u;
    const bool enrichedBlockPreconditioner = preconditionerSelector == 3u ||
        preconditionerSelector == 4u;
    const std::uint32_t assumedRegionalModes =
        enrichedBlockPreconditioner ? 32u :
        (preconditionerSelector == 2u ? 18u : 0u);
    output << std::setprecision(17)
           << "{\n  \"schema\": \"numi.cardboard.shipping-prebuild-memory-screen.v1\",\n"
           << "  \"evidence_class\": \"pre-mesh planning estimate; not exact geometry or native memory admission\",\n"
           << "  \"finished_inside_dimensions_m\": ["
           << arguments.shippingDesign->defaults.inside.lengthX << ','
           << arguments.shippingDesign->defaults.inside.heightY << ','
           << arguments.shippingDesign->defaults.inside.widthZ << "],\n"
           << "  \"blank_perimeter_m\": "
           << arguments.shippingDesign->perimeterSpanM << ",\n"
           << "  \"pitch_m\": 0.0079,\n"
           << "  \"y_intervals\": " << estimate.yIntervals << ",\n"
           << "  \"thickness_slices\": " << estimate.thicknessSlices << ",\n"
           << "  \"fgmres_preconditioner\": {\"selector\": "
           << preconditionerSelector << ", \"regional_galerkin_enabled\": "
           << (regionalPreconditioner ? "true" : "false")
           << ", \"assumed_coarse_modes\": " << assumedRegionalModes
           << ", \"enriched_block_fine_estimate_enabled\": "
           << (enrichedBlockPreconditioner ? "true" : "false") << "},\n"
           << "  \"reference_pitch_intervals\": "
           << estimate.referencePitchIntervals << ",\n"
           << "  \"requested_pitch_intervals\": "
           << estimate.requestedPitchIntervals << ",\n"
           << "  \"cardinality_margin\": " << estimate.cardinalityMargin << ",\n"
           << "  \"reference_y_intervals\": "
           << numi::cardboard::kShippingReferenceYIntervals << ",\n"
           << "  \"requested_y_intervals\": " << estimate.yIntervals << ",\n"
           << "  \"reference_geometry_counts\": {\"nodes\": "
           << numi::cardboard::kShippingReferenceNodes
           << ", \"tetrahedra\": "
           << numi::cardboard::kShippingReferenceTetrahedra << "},\n"
           << "  \"nominal_cardinality\": {\"nodes\": "
           << estimate.nominalNodeCount << ", \"tetrahedra\": "
           << estimate.nominalTetrahedronCount << "},\n"
           << "  \"estimated_cardinality\": {\"nodes\": "
           << estimate.estimatedNodeCount << ", \"tetrahedra\": "
           << estimate.estimatedTetrahedronCount << "},\n"
           << "  \"contact_allocation_options\": {\"compact_immutable_surface_map_requested\": "
           << (arguments.compactImmutableSurfaces ? "true" : "false")
           << ", \"prebuild_surface_face_upper_bound\": "
           << (arguments.compactImmutableSurfaces
                   ? numi::cardboard::shippingMemoryCheckedMultiply(
                         estimate.nominalTetrahedronCount, 4u)
                   : 0u)
           << ", \"deformable_contact_capacity_requested\": "
           << (arguments.deformableContactCapacityProvided
                   ? std::to_string(arguments.deformableContactCapacity) : "null")
           << ", \"prebuild_deformable_contact_capacity\": "
           << (arguments.deformableContactCapacityProvided
                   ? arguments.deformableContactCapacity
                   : numi::cardboard::shippingMemoryCheckedMultiply(
                         estimate.nominalTetrahedronCount, 8u))
           << ", \"rigid_proxy_count_bound\": "
           << (arguments.shippingBox
                   ? numi::cardboard::kShippingRigidProxyCount : 0u)
           << ", \"rigid_body_count_bound\": "
           << (arguments.shippingBox
                   ? numi::cardboard::kShippingRigidBodyCount : 0u)
           << ", \"contact_pair_count_upper_bound\": "
           << (arguments.shippingBox
                   ? numi::cardboard::shippingMemoryCheckedMultiply(
                         estimate.nominalNodeCount,
                         numi::cardboard::kShippingRigidProxyCount)
                   : 0u) << "},\n"
           << "  \"required_residual_descent\": {\"required_for_shipping_physics\": "
           << (arguments.shippingBox ? "true" : "false")
           << ", \"runtime_private_arenas_allocated\": false"
           << ", \"private_full_contact_bytes\": "
           << budget.requiredFullContactArmijoBytes
           << ", \"private_rigid_contact_bytes\": "
           << budget.requiredRigidContactArmijoBytes
           << ", \"private_cohesive_bytes\": "
           << budget.requiredCohesiveArmijoBytes
           << ", \"runtime_arena_caps_available\": "
           << (budget.requiredResidualDescentArenaAdmitted ? "true" : "false")
           << ", \"shipping_path_qualified\": false, \"blocker\": \""
           << jsonEscape(budget.requiredResidualDescentArenaFailure)
           << "\"},\n"
           << "  \"estimated_static_arena_bytes\": {\"fem_node_states\": "
           << estimate.staticArenaScreen.femNodeStatesBytes
           << ", \"tetrahedra\": " << estimate.staticArenaScreen.tetrahedraBytes
           << ", \"fgmres_vectors\": " << estimate.staticArenaScreen.fgmresBasisBytes
           << ", \"sum\": " << estimate.staticArenaScreen.totalBytes << "},\n"
           << "  \"limit_bytes\": "
           << numi::cardboard::kShippingPeakMemoryLimitBytes << ",\n"
           << "  \"screen_result\": \""
           << (budget.withinLimit && estimate.staticArenaScreen.totalBytes <=
                   numi::cardboard::kShippingPeakMemoryLimitBytes
               ? "continue_to_exact_mesh_compile" : "reject_before_mesh_build") << "\",\n"
           << "  \"estimated_peak_domain_bytes\": "
           << budget.estimatedPeakDomainBytes << ",\n"
           << "  \"estimated_metal_peak_bytes\": "
           << budget.estimatedMetalPeakBytes << ",\n"
           << "  \"estimated_host_peak_bytes\": "
           << budget.estimatedHostPeakBytes << ",\n"
           << "  \"budget_components_bytes\": {\"measured_base_and_contact_envelope\": "
           << budget.measuredBaseAndContactEnvelopeBytes << ", \"cohesive_faces\": "
           << budget.cohesiveRuntimeBytes << ", \"persistent_contact_surface\": "
           << budget.persistentContactSurfaceBytes
           << ", \"persistent_contact_slots\": "
           << budget.persistentContactSlotBytes
           << ", \"persistent_contact_auxiliary\": "
           << budget.persistentContactAuxiliaryBytes
           << ", \"rigid_contact_runtime\": "
           << budget.rigidContactRuntimeBytes
           << ", \"constraint_contact_force_readback\": "
           << budget.constraintContactForceReadbackBytes
           << ", \"regional_galerkin\": "
           << budget.regionalGalerkinBytes << ", \"upload_staging\": "
           << budget.uploadStagingBytes << ", \"host_mesh_compile\": "
           << budget.hostMeshAndCompileBytes << ", \"host_checkpoints\": "
           << budget.hostCheckpointBytes << "},\n"
           << "  \"estimate_method\": \"Pre-allocation gate uses nominal cardinality scaled from the compiled reference shipping mesh. A separate 25%-padded cardinality screen is reported as stress context. The nominal arena estimate uses the exact legacy or explicit deformable-contact capacity, a conservative 4*T surface upper bound, the 12 reusable score/assembly rigid proxies with every FEM node treated contact-eligible (unless preflight-only omits tools), a cohesive reserve only when a cohesive law is requested, measured case032 noncontact runtime remainder scaled by FEM/tet/Krylov floor, optional RAP/private arenas, upload staging and host/checkpoint reserves. After compile, exact dispatch capacities and compact surface counts replace these bounds\",\n"
           << "  \"exact_compiled_arena_report_still_required\": true,\n"
           << "  \"native_runtime_observation_still_required\": true,\n"
           << "  \"qualification_allowed\": false\n}\n";
    require(output.good(), "failed writing shipping prebuild screen");
}

void writeShippingMemoryReport(
    const std::filesystem::path& path,
    const Arguments& arguments,
    const MeshSource& mesh,
    const CompiledWorld& world,
    const numi::cardboard::ShippingStaticMemoryLowerBound& lowerBound,
    const numi::cardboard::ShippingRuntimeMemoryObservation* observation,
    const numi::cardboard::ShippingMemoryBudget* budget = nullptr) {
    require(arguments.shippingDesign.has_value(),
            "shipping memory report requires a derived design");
    std::ofstream output(path);
    require(output.good(), "cannot create shipping memory report");
    output << std::setprecision(17)
           << "{\n  \"schema\": \"numi.cardboard.shipping-memory-preflight.v1\",\n"
           << "  \"evidence_class\": \"native allocation accounting; not a physical qualification\",\n"
           << "  \"phase\": \""
           << (observation ? "runtime_initialized_no_step_or_native_run" :
                              "compiled_world_static_lower_bound_only")
           << "\",\n  \"compiled_world_fingerprint\": " << world.fingerprint
           << ",\n  \"physics_fingerprint\": " << world.physicsFingerprint
           << ",\n  \"environment_count\": " << lowerBound.environmentCount
           << ",\n  \"mesh\": {\"nodes\": " << world.dispatch.femNodeCount
           << ", \"tetrahedra\": " << world.dispatch.tetrahedronCount
           << ", \"cohesive_faces\": " << world.dispatch.cohesiveFaceCount
           << ", \"compiled_surface_faces\": " << world.dispatch.surfaceFaceCount
           << ", \"compact_immutable_surface_map\": "
           << ((world.dispatch.flags & NM_MATTER_STATIC_FEM_SURFACE_MAP) != 0u
                   ? "true" : "false")
           << ", \"deformable_contact_capacity\": "
           << world.dispatch.deformableContactCapacity << "},\n"
           << "  \"source_bindings\": {\n"
           << "    \"liner_sha256\": \"" << hexDigest(sha256File(arguments.linerMaterial)) << "\",\n"
           << "    \"medium_sha256\": \"" << hexDigest(sha256File(arguments.mediumMaterial)) << "\",\n"
           << "    \"glue_sha256\": \""
           << (arguments.glueMaterial.empty() ? std::string{} :
               hexDigest(sha256File(arguments.glueMaterial))) << "\",\n"
           << "    \"material_map_sha256\": \"" << mesh.materialMapDigest << "\",\n"
           << "    \"authored_region_map_sha256\": \""
           << mesh.authoredRegionMapDigest << "\",\n"
           << "    \"material_frame_map_sha256\": \"" << mesh.frameMapDigest << "\",\n"
           << "    \"coupon_evidence_sha256\": \"" << arguments.shippingCouponEvidenceSha256 << "\",\n"
           << "    \"capacity_qualification_file_sha256\": \""
           << arguments.shippingCapacityQualificationSha256 << "\",\n"
           << "    \"coupon_measurement_sha256\": \""
           << (shippingAllowanceHasMeasuredCouponEvidence(*arguments.shippingDesign)
                   ? arguments.shippingDesign->allowanceCoupon.measurementSha256
                   : std::string{}) << "\",\n"
           << "    \"starch_parameters_sha256\": \""
           << arguments.starchInterfaceParametersSha256 << "\",\n"
           << "    \"shipping_hma_parameters_sha256\": \""
           << arguments.shippingHmaParametersSha256 << "\"\n  },\n"
           << "  \"static_lower_bound_bytes\": {\n"
           << "    \"fem_node_states\": " << lowerBound.femNodeStatesBytes << ",\n"
           << "    \"tetrahedra\": " << lowerBound.tetrahedraBytes << ",\n"
           << "    \"fgmres_vectors\": " << lowerBound.fgmresBasisBytes << ",\n"
           << "    \"sum\": " << lowerBound.totalBytes << ",\n"
           << "    \"scope\": \"lower bound only; excludes other runtime arenas, contact, checkpoints, command captures, host mesh, compiler and package buffers\"\n  },\n"
           << "  \"peak_limit_bytes\": "
           << numi::cardboard::kShippingPeakMemoryLimitBytes << ",\n"
           << "  \"static_lower_bound_within_limit\": "
           << (lowerBound.totalBytes <=
                   numi::cardboard::kShippingPeakMemoryLimitBytes ? "true" : "false");
    if (budget) {
        output << ",\n  \"estimated_arena_budget\": {\n"
               << "    \"evidence_class\": \"checked pre-allocation envelope; baseline factor rounded above case032 measured resident-to-static-floor ratio, optional arenas use runtime caps\",\n"
               << "    \"measured_base_and_contact_envelope_bytes\": "
               << budget->measuredBaseAndContactEnvelopeBytes << ",\n"
               << "    \"static_fem_solver_floor_bytes\": "
               << budget->staticFEMSolverFloorBytes << ",\n"
               << "    \"measured_noncontact_runtime_envelope_bytes\": "
               << budget->measuredNoncontactRuntimeEnvelopeBytes << ",\n"
               << "    \"constraint_contact_force_readback_bytes\": "
               << budget->constraintContactForceReadbackBytes << ",\n"
               << "    \"persistent_contact_surface_bytes\": "
               << budget->persistentContactSurfaceBytes << ",\n"
               << "    \"persistent_contact_slot_bytes\": "
               << budget->persistentContactSlotBytes << ",\n"
               << "    \"persistent_contact_auxiliary_bytes\": "
               << budget->persistentContactAuxiliaryBytes << ",\n"
               << "    \"rigid_contact_runtime_bytes\": "
               << budget->rigidContactRuntimeBytes << ",\n"
               << "    \"cohesive_runtime_bytes\": "
               << budget->cohesiveRuntimeBytes << ",\n"
               << "    \"regional_galerkin_bytes\": "
               << budget->regionalGalerkinBytes << ",\n"
               << "    \"regional_galerkin_block_fine_bytes\": "
               << budget->regionalGalerkinBlockFineBytes << ",\n"
               << "    \"regional_galerkin_diagnostics_readback_bytes\": "
               << budget->regionalGalerkinDiagnosticsBytes << ",\n"
               << "    \"rigid_contact_armijo_bytes\": "
               << budget->rigidContactArmijoBytes << ",\n"
               << "    \"full_contact_armijo_bytes\": "
               << budget->fullContactArmijoBytes << ",\n"
               << "    \"required_full_contact_armijo_bytes\": "
               << budget->requiredFullContactArmijoBytes << ",\n"
               << "    \"cohesive_armijo_bytes\": "
               << budget->cohesiveArmijoBytes << ",\n"
               << "    \"required_rigid_contact_armijo_bytes\": "
               << budget->requiredRigidContactArmijoBytes << ",\n"
               << "    \"required_cohesive_armijo_bytes\": "
               << budget->requiredCohesiveArmijoBytes << ",\n"
               << "    \"required_residual_descent_arena_caps_admitted\": "
               << (budget->requiredResidualDescentArenaAdmitted
                       ? "true" : "false") << ",\n"
               << "    \"required_residual_descent_arena_failure\": \""
               << jsonEscape(budget->requiredResidualDescentArenaFailure)
               << "\",\n"
               << "    \"upload_staging_reserve_bytes\": "
               << budget->uploadStagingBytes << ",\n"
               << "    \"snapshot_readback_reserve_bytes\": "
               << budget->snapshotReadbackBytes << ",\n"
               << "    \"estimated_metal_peak_bytes_including_staging\": "
               << budget->estimatedMetalPeakBytes << ",\n"
               << "    \"host_mesh_and_compile_bytes\": "
               << budget->hostMeshAndCompileBytes << ",\n"
               << "    \"host_checkpoint_payload_bytes\": "
               << budget->hostCheckpointBytes << ",\n"
               << "    \"estimated_host_peak_bytes\": "
               << budget->estimatedHostPeakBytes << ",\n"
               << "    \"estimated_peak_domain_bytes\": "
               << budget->estimatedPeakDomainBytes << ",\n"
               << "    \"limit_bytes\": " << budget->limitBytes << ",\n"
               << "    \"admission_passed_before_native_initialization\": "
               << (budget->withinLimit ? "true" : "false") << ",\n"
               << "    \"measured_domains_overlap_on_unified_memory\": true\n  }";
    } else {
        output << ",\n  \"estimated_arena_budget\": null";
    }
    if (observation) {
        output << ",\n  \"native_observation_bytes\": {\n"
               << "    \"runtime_resident_bytes\": " << observation->runtimeResidentBytes << ",\n"
               << "    \"metal_current_allocated_bytes\": " << observation->metalCurrentAllocatedBytes << ",\n"
               << "    \"metal_recommended_max_working_set_bytes\": " << observation->metalRecommendedWorkingSetBytes << ",\n"
               << "    \"host_phys_footprint_bytes\": "
               << (observation->hostResidentSampleAvailable
                       ? std::to_string(observation->hostResidentBytes) : "null") << ",\n"
               << "    \"largest_measured_domain_bytes\": "
               << numi::cardboard::shippingObservedPeakBytes(*observation) << ",\n"
               << "    \"within_16gib_componentwise_limit\": "
               << (numi::cardboard::shippingObservedPeakBytes(*observation) <=
                       numi::cardboard::kShippingPeakMemoryLimitBytes ? "true" : "false")
               << ",\n    \"host_sampler_available\": "
               << (observation->hostResidentSampleAvailable ? "true" : "false") << ",\n"
               << "    \"counters_may_overlap\": true,\n"
               << "    \"recommended_working_set_is_advisory\": true\n  }";
    } else {
        output << ",\n  \"runtime_observation\": null,\n"
               << "  \"native_initialize_required_before_memory_admission\": true";
    }
    output << ",\n  \"qualification_allowed\": false,\n"
           << "  \"shipping_stage_enabled\": "
           << (arguments.shippingPreflightOnly || arguments.compileOnly
                   ? "false" : "true") << "\n}\n";
    require(output.good(), "failed writing shipping memory report");
}

void writeShippingDesignReport(const std::filesystem::path& path,
                               const Arguments& arguments,
                               const MeshSource& mesh,
                               const CompiledWorld& world,
                               const numi::cardboard::ShippingStaticMemoryLowerBound& memory) {
    require(arguments.shippingDesign.has_value(),
            "shipping design report requires a derived design");
    const auto& design = *arguments.shippingDesign;
    const auto closure = numi::cardboard::makeShippingClosurePlan(design);
    const auto depths = numi::cardboard::shippingScoreDepthsM(
        numi::cardboard::ShippingScoreToolDefaults{},
        mesh.boxBlank ? mesh.boxBlank->boardCaliper : arguments.totalHeightM);
    const auto section = numi::cardboard::makeShippingCrossSectionConfig(design);
    const auto pitchCapacity = numi::cardboard::auditShippingCrossSectionCapacity(
        design, section);
    const double boardCaliperM = mesh.boxBlank
        ? mesh.boxBlank->boardCaliper : arguments.totalHeightM;
    const std::string scoreCommandSha256 = shippingScoreCommandFingerprint(
        arguments, world.physicsFingerprint, world.fingerprint, boardCaliperM);
    std::ofstream output(path);
    require(output.good(), "cannot create shipping design report");
    output << std::setprecision(17)
           << "{\n  \"schema\": \"numi.cardboard.shipping-design.v1\",\n"
           << "  \"status\": \"flat_dieline_native_protocol_inputs\",\n"
           << "  \"finished_inside_dimensions_m\": [" << design.defaults.inside.lengthX << ','
           << design.defaults.inside.heightY << ',' << design.defaults.inside.widthZ << "],\n"
           << "  \"score_center_span_minus_inside_m_xyz\": ["
           << design.allowanceCoupon.scoreCenterSpanMinusInsideM[0] << ','
           << design.allowanceCoupon.scoreCenterSpanMinusInsideM[1] << ','
           << design.allowanceCoupon.scoreCenterSpanMinusInsideM[2] << "],\n"
           << "  \"allowance_disposition\": \""
           << (shippingAllowanceHasMeasuredCouponEvidence(design)
                   ? (design.allowanceFromPhysicalCoupon ? "physical_coupon_allowance" :
                      "native_coupon_allowance")
                   : (arguments.shippingEngineeringAssumptions
                          ? "explicit_engineering_assumption_unqualified"
                          : "zero_offset_lower_bound_only")) << "\",\n"
           << "  \"coupon_id\": \"" << jsonEscape(design.allowanceCoupon.couponId) << "\",\n"
           << "  \"coupon_source_run_id\": \"" << jsonEscape(design.allowanceCoupon.sourceRunId) << "\",\n"
           << "  \"coupon_measurement_sha256\": "
           << (shippingAllowanceHasMeasuredCouponEvidence(design)
                   ? "\"" + design.allowanceCoupon.measurementSha256 + "\""
                   : "null") << ",\n"
           << "  \"coupon_evidence_file_sha256\": \""
           << arguments.shippingCouponEvidenceSha256 << "\",\n"
           << "  \"coupon_evidence_file\": \""
           << jsonEscape(arguments.shippingCouponEvidence.string()) << "\",\n"
           << "  \"coupon_measurement_file\": \""
           << jsonEscape(arguments.shippingMeasurementFile.string()) << "\",\n"
           << "  \"allowance_gates\": {\"complete_score_fold_release\": "
           << (shippingAllowanceHasMeasuredCouponEvidence(design) ? "true" : "false")
           << ", \"scoring_damage_law_qualified\": "
           << (arguments.shippingCouponDamageLawQualified ? "true" : "false")
           << "},\n"
           << "  \"blank\": {\"panel_length_m\": " << design.blankConfig.panelLength
           << ", \"panel_width_m\": " << design.blankConfig.panelWidth
           << ", \"wall_height_m\": " << design.blankConfig.wallHeight
           << ", \"manufacturer_lap_m\": " << design.blankConfig.manufacturerJointWidth
           << ", \"slot_kerf_m\": " << design.blankConfig.slotKerf
           << ", \"perimeter_span_m\": " << design.perimeterSpanM
           << ", \"blank_height_m\": " << design.blankHeightM
           << ", \"flap_depth_m\": " << design.flapDepthM << "},\n"
           << "  \"board_recipe\": {\"id\": \"literature2009_cross_source\", "
           << "\"profile\": \"sinusoidal_fallback\", \"pitch_m\": " << section.pitch
           << ", \"caliper_m\": " << section.caliper
           << ", \"liner_thickness_m\": " << section.linerThickness
           << ", \"medium_thickness_m\": " << section.mediumThickness
           << ", \"lower_glue_gap_m\": " << section.glueMinimumGap
           << ", \"lower_bond_width_m\": " << section.bondWidth
           << ", \"upper_glue_gap_m\": " << section.upperGlueMinimumGap
           << ", \"safeguarded_paper_hill_initializer\": "
           << (arguments.safeguardedPaperInitializer ? "true" : "false")
           << ", \"paper_compensated_arithmetic\": "
           << (arguments.compensatedPaperArithmetic ? "true" : "false")
           << ", \"paper_hill_accurate_projection\": "
           << (arguments.accuratePaperProjection ? "true" : "false")
           << ", \"upper_bond_width_m\": " << section.upperBondWidth
           << ", \"partial_terminal_pitch\": true, \"pitch_intervals\": "
           << pitchCapacity.clippedPitchIntervals << "},\n"
           << "  \"score_tool_defaults\": {\"nose_radius_m\": 0.001, \"channel_width_m\": 0.006, "
           << "\"support_rail_width_m\": 0.010, "
           << "\"clearance_calipers\": 0.25, \"depth_study_candidate_m\": ["
           << depths[0] << ',' << depths[1] << ',' << depths[2] << ',' << depths[3]
           << "], \"selected_depth_ratio_of_caliper\": "
           << arguments.shippingScoreDepthRatio
           << ", \"selected_depth_m\": "
           << arguments.shippingScoreDepthRatio * boardCaliperM
           << ", \"depth_series_execution\": \"one selected ratio per invocation from a freshly constructed flat blank; candidate ratios are separate runs, never cumulative passes\""
           << ", \"tool_speed_m_per_s\": "
           << arguments.shippingScoreToolSpeedMPerS
           << ", \"tool_speed_source_class\": \"engineering_assumption_unqualified\""
           << ", \"score_command_sha256\": \"" << scoreCommandSha256
           << "\", \"channel_tool_geometry_available\": "
           << (mesh.boxScoreTooling &&
                       mesh.boxScoreTooling->channelWidth > 0.0
                   ? "true" : "false")
           << ", \"channel_tool_geometry_qualified\": false, \"crease_damage_law_qualified\": false},\n"
           << "  \"closure_plan\": {\"adhesive\": \"hot_melt\", \"patch_count\": "
           << closure.patches.size() << ", \"overlap_area_m2\": " << closure.overlapAreaM2
           << ", \"adhesive_footprint_area_m2\": " << closure.adhesiveFootprintAreaM2
           << ", \"manufacturer_lap_width_m\": 0.025, \"flap_patch_layout\": "
           << "\"major flaps bond to underlying minor flaps; major flaps meet at an unbonded butt line\", "
           << "\"tape\": false, \"adhesive_grade_or_law_calibrated\": false, "
           << "\"authored_mortar_patch_groups\": [";
    for (std::size_t index = 0u; index < mesh.hmaClosureMesh.groups.size(); ++index) {
        if (index) output << ',';
        const auto& group = mesh.hmaClosureMesh.groups[index];
        output << "{\"id\":\"" << jsonEscape(group.patchId)
               << "\",\"patches\":" << group.stableIdentifiers.size()
               << ",\"expected_area_m2\":" << group.expectedAdhesiveAreaM2
               << ",\"prepared_area_m2\":" << group.preparedAdhesiveAreaM2 << '}';
    }
    output << "]},\n"
           << "  \"hma_parameter_provenance\": {\"file_sha256\": \""
           << arguments.shippingHmaParametersSha256 << "\", \"source_class\": \""
           << (arguments.shippingHmaLaw
                   ? jsonEscape(arguments.shippingHmaLaw->sourceClass)
                   : std::string("not_provided"))
           << "\", \"source_citation\": \""
           << (arguments.shippingHmaLaw
                   ? jsonEscape(arguments.shippingHmaLaw->sourceCitation)
                   : std::string{})
           << "\", \"transfer_uncertainty_fraction\": "
           << (arguments.shippingHmaLaw
                   ? arguments.shippingHmaLaw->transferUncertaintyFraction : 0.0)
           << ", \"calibrated_corrugated_closure\": false},\n"
           << "  \"starch_interface\": {\"paired_face_count\": " << mesh.starchPairedFaces.size()
           << ", \"law_parameters_sha256\": \"" << arguments.starchInterfaceParametersSha256
           << "\", \"source_class\": \""
           << (mesh.starchInterfaceParameters
                   ? mesh.starchInterfaceParameters->sourceClass : "not_provided")
           << "\", \"qualified_starch_calibration\": false, \"self_contact_after_debonding\": false},\n"
           << "  \"mesh\": {\"nodes\": " << world.dispatch.femNodeCount
           << ", \"tetrahedra\": " << world.dispatch.tetrahedronCount
           << ", \"cohesive_faces\": " << world.dispatch.cohesiveFaceCount
           << ", \"static_lower_bound_bytes\": " << memory.totalBytes << "},\n"
           << "  \"accepted_stage_gates\": {\"scoring\": false, \"folded_walls\": false, "
           << "\"manufacturer_lap\": false, \"flaps\": false, \"released_settled_assembly\": false},\n"
           << "  \"qualification\": {\"native_process_complete\": false, "
           << "\"physical_box_qualified\": false, \"source_grade_calibrated\": false}\n}\n";
    require(output.good(), "failed writing shipping design report");
}

std::filesystem::path writeShippingAcceptedCheckpoint(
    const std::filesystem::path& outputDirectory,
    const RuntimeStateSnapshot& snapshot,
    const CompiledWorld& world,
    const Arguments& arguments,
    const std::string& stage,
    const std::uint32_t nativeStep,
    const std::uint32_t assemblyActionIndex,
    const double scoreDieHeightM,
    const double simulationTimeSeconds,
    const std::string& branchIdentity = "linear-run") {
    require(arguments.shippingBox,
            "shipping checkpoint requested for a non-shipping specimen");
    const auto& substrateGravity = world.dispatch.gravityAndTimestep;
    require(substrateGravity.x == 0.0f &&
                substrateGravity.y == static_cast<float>(-9.80665) &&
                substrateGravity.z == 0.0f,
            "shipping stage checkpoint does not carry the authored substrate self-weight dispatch");
    require(snapshot.available,
            "shipping checkpoint requires an available completion-boundary snapshot");
    require(snapshot.sourcePhysicsFingerprint == world.physicsFingerprint &&
                snapshot.deviceProgramFingerprint == world.fingerprint,
            "shipping checkpoint does not match the compiled Matter source/program");
    require(std::isfinite(simulationTimeSeconds) && simulationTimeSeconds >= 0.0,
            "shipping checkpoint simulation time is invalid");
    const double substrateMassKg = compiledSubstrateNodalMassKg(world);
    const auto checkpointDirectory = outputDirectory / "checkpoints";
    std::error_code error;
    std::filesystem::create_directories(checkpointDirectory, error);
    require(!error, "cannot create shipping checkpoints directory: " + error.message());
    const std::string stem = "accepted-step-" + stepLabel(nativeStep) + "-" +
        stage;
    const auto snapshotPath = checkpointDirectory / (stem + ".nmcb");
    const auto metadataPath = checkpointDirectory / (stem + ".json");
    numi::cardboard::writeShippingAcceptedSnapshot(
        snapshotPath, snapshot, world, arguments.environmentCount, true);
    require(!snapshot.statuses.empty() && !snapshot.solverCertificates.empty(),
            "shipping frame index requires a native status and solver certificate");
    const auto& frameStatus = snapshot.statuses.front();
    const auto& frameCertificate = snapshot.solverCertificates.front();
    const bool finiteCertificate =
        std::isfinite(frameCertificate.nonlinear.x) &&
        std::isfinite(frameCertificate.nonlinear.y) &&
        std::isfinite(frameCertificate.nonlinear.z) &&
        std::isfinite(frameCertificate.nonlinear.w);
    const bool nativeAccepted = nativeStep > 0u &&
        frameStatus.code == NM_STATUS_SUCCESS && finiteCertificate &&
        frameCertificate.validity.w > 0.5f &&
        frameCertificate.nonlinear.x <= world.mixedSolver.residualTolerances.x &&
        frameCertificate.nonlinear.z <= world.mixedSolver.residualTolerances.y &&
        frameCertificate.nonlinear.w <= world.mixedSolver.residualTolerances.z;
    if (nativeStep > 0u)
        require(nativeAccepted,
                "shipping geometry frame lacks an accepted native status/certificate");

    const auto framesDirectory = outputDirectory / "shipping-frames";
    std::filesystem::create_directories(framesDirectory, error);
    require(!error,
            "cannot create shipping accepted-frame geometry directory: " + error.message());
    const auto geometryPath = framesDirectory / (stem + ".obj");
    writeObj(geometryPath, world, snapshot.femNodes, 0u);
    const auto frameIndexPath = outputDirectory / "shipping-accepted-frames.csv";
    const bool writeFrameHeader = !std::filesystem::exists(frameIndexPath);
    std::ofstream frameIndex(frameIndexPath, std::ios::app);
    require(frameIndex.good(), "cannot append shipping accepted-frame index");
    if (writeFrameHeader)
        frameIndex << "stage,simulation_time_s,geometry_path,native_step_index,native_accepted,status_code,certificate_valid,residual_x,relative_correction_y,volume_residual_z,pressure_residual_w,checkpoint_identity,checkpoint_sha256,branch_identity,source_physics_fingerprint,device_program_fingerprint,score_command_sha256\n";
    frameIndex << std::setprecision(17)
        << stage << ',' << simulationTimeSeconds << ','
        << "shipping-frames/" << geometryPath.filename().string() << ','
        << snapshot.controlStep << ',' << (nativeAccepted ? 1 : 0) << ','
        << frameStatus.code << ',' << frameCertificate.validity.w << ','
        << frameCertificate.nonlinear.x << ','
        << frameCertificate.nonlinear.y << ','
        << frameCertificate.nonlinear.z << ','
        << frameCertificate.nonlinear.w << ',' << snapshotPath.filename().string()
        << ',' << hexDigest(sha256File(snapshotPath)) << ',' << branchIdentity
        << ',' << snapshot.sourcePhysicsFingerprint << ','
        << snapshot.deviceProgramFingerprint << ','
        << shippingScoreCommandFingerprint(
               arguments, world.physicsFingerprint, world.fingerprint,
               arguments.totalHeightM) << '\n';
    frameIndex.flush();
    require(frameIndex.good(), "failed writing shipping accepted-frame index row");
    std::ofstream metadata(metadataPath);
    require(metadata.good(), "cannot create shipping checkpoint metadata");
    metadata << std::setprecision(17)
             << "{\n  \"schema\": \"numi.cardboard.accepted-shipping-checkpoint.v1\",\n"
             << "  \"checkpoint_file\": \"" << jsonEscape(snapshotPath.filename().string()) << "\",\n"
             << "  \"checkpoint_sha256\": \"" << hexDigest(sha256File(snapshotPath)) << "\",\n"
             << "  \"stage\": \"" << jsonEscape(stage) << "\",\n"
             << "  \"native_step_index\": " << nativeStep << ",\n"
             << "  \"simulation_time_s\": " << simulationTimeSeconds << ",\n"
             << "  \"branch_identity\": \"" << jsonEscape(branchIdentity) << "\",\n"
             << "  \"geometry_frame\": \"shipping-frames/"
             << jsonEscape(geometryPath.filename().string()) << "\",\n"
             << "  \"native_accepted\": " << (nativeAccepted ? "true" : "false") << ",\n"
             << "  \"native_status_code\": " << frameStatus.code << ",\n"
             << "  \"native_certificate_validity\": "
             << frameCertificate.validity.w << ",\n"
             << "  \"accepted_control_step\": " << snapshot.controlStep << ",\n"
             << "  \"accepted_physics_substep\": " << snapshot.physicsSubstep << ",\n"
             << "  \"assembly_action_index\": ";
    if (assemblyActionIndex == numi::cardboard::kNoBoxScoreLineIndex)
        metadata << "null";
    else
        metadata << assemblyActionIndex;
    metadata << ",\n  \"score_die_height_m\": " << scoreDieHeightM << ",\n"
             << "  \"source_physics_fingerprint\": " << snapshot.sourcePhysicsFingerprint << ",\n"
             << "  \"device_program_fingerprint\": " << snapshot.deviceProgramFingerprint << ",\n"
             << "  \"shipping_score_command_sha256\": \""
             << shippingScoreCommandFingerprint(
                    arguments, world.physicsFingerprint, world.fingerprint,
                    arguments.totalHeightM) << "\",\n"
             << "  \"shipping_score_depth_ratio_of_caliper\": "
             << arguments.shippingScoreDepthRatio << ",\n"
             << "  \"shipping_score_speed_m_per_s\": "
             << arguments.shippingScoreToolSpeedMPerS << ",\n"
             << "  \"substrate_self_weight\": {\"enabled\": "
             << (arguments.shippingBox ? "true" : "false")
             << ", \"gravity_m_s2\": ["
             << world.dispatch.gravityAndTimestep.x << ','
             << world.dispatch.gravityAndTimestep.y << ','
             << world.dispatch.gravityAndTimestep.z
             << "], \"compiled_nodal_mass_total_kg\": " << substrateMassKg
             << ", \"substrate_weight_force_n\": [0,"
             << substrateMassKg * world.dispatch.gravityAndTimestep.y
             << ",0], \"native_fem_rhs\": \"nodal_mass_times_compiled_gravity\", "
                "\"gravity_is_immutable_across_stage_branches\": true, "
                "\"stage_scope\": \"all accepted shipping FEM stages, including rigid-body mass initialization\"},\n"
             << "  \"environment_count\": " << arguments.environmentCount << ",\n"
             << "  \"starch_interface_parameters_sha256\": \""
             << arguments.starchInterfaceParametersSha256 << "\",\n"
             << "  \"shipping_hma_parameters_sha256\": \""
             << arguments.shippingHmaParametersSha256 << "\",\n"
             << "  \"coupon_evidence_sha256\": \""
             << arguments.shippingCouponEvidenceSha256 << "\",\n"
             << "  \"continuation_scope\": \"same compiled Matter world and ABI; runtime restore must still pass native fingerprint admission\",\n"
             << "  \"qualification\": false\n}\n";
    require(metadata.good(), "failed writing shipping checkpoint metadata");
    return snapshotPath;
}

struct ReactionMetrics {
    Vec3 mechanicalConstraintForce{};
    Vec3 contactForce{};
    Vec3 force{};
    double mechanicalConstraintMomentAboutY = 0.0;
    double contactMomentAboutY = 0.0;
    double momentAboutY = 0.0;
};

ReactionMetrics reactionMetrics(const std::uint32_t environment,
                                const std::vector<nm_float4>& reactions,
                                const std::vector<nm_float4>& contactReactions,
                                const RuntimeStateSnapshot& acceptedState,
                                const MeshSource& mesh) {
    const std::uint32_t nodeCount =
        static_cast<std::uint32_t>(mesh.restNodes.size());
    const std::size_t base = static_cast<std::size_t>(environment) * nodeCount;
    require(reactions.size() >= base + nodeCount &&
                contactReactions.size() >= base + nodeCount &&
                acceptedState.femNodes.size() >= base + nodeCount,
            "accepted grip reaction readback is truncated");
    std::vector<std::uint32_t> selectedNodes(mesh.rightGripNodes.begin(),
                                              mesh.rightGripNodes.end());
    const auto wrench = numi::cardboard::summarizeConstraintReactions(
        std::span<const NMFEMNodeStateGPU>(acceptedState.femNodes),
        std::span<const nm_float4>(reactions),
        std::span<const nm_float4>(contactReactions), base, nodeCount,
        selectedNodes);
    ReactionMetrics result;
    result.mechanicalConstraintForce = {wrench.mechanicalConstraintForce.x,
        wrench.mechanicalConstraintForce.y, wrench.mechanicalConstraintForce.z};
    result.contactForce = {wrench.contactForce.x,
        wrench.contactForce.y, wrench.contactForce.z};
    result.force = {wrench.totalForce.x, wrench.totalForce.y,
                    wrench.totalForce.z};
    result.mechanicalConstraintMomentAboutY = wrench.mechanicalConstraintMoment.y;
    result.contactMomentAboutY = wrench.contactMoment.y;
    result.momentAboutY = wrench.totalMoment.y;
    return result;
}

double measuredGripAngle(const MeshSource& mesh, const RuntimeStateSnapshot& snapshot,
                         const std::uint32_t environment) {
    double meanZ = 0.0;
    for (const auto node : mesh.rightGripNodes) meanZ += mesh.restNodes[node].z;
    meanZ /= mesh.rightGripNodes.size();
    double x = 0.0, z = 0.0;
    const auto base = environment * mesh.restNodes.size();
    for (const auto node : mesh.rightGripNodes) {
        const double weight = mesh.restNodes[node].z - meanZ;
        const auto point = snapshot.femNodes.at(base + node).positionAndMass;
        x += weight * point.x; z += weight * point.z;
    }
    return std::atan2(x, z) * 180.0 / kPi;
}

void writeHeader(std::ofstream& output) {
    output << "step,time_s,environment,arm,target_angle_deg,status_code,"
              "object_index,failing_index,completed_microsteps,fgmres_iterations,"
              "status_diagnostic_x,status_diagnostic_y,status_diagnostic_z,"
              "status_diagnostic_w,min_J,max_J,"
              "max_node_displacement_m,reaction_x_N,reaction_y_N,reaction_z_N,"
              "reaction_moment_y_Nm,mechanical_constraint_reaction_x_N,"
              "mechanical_constraint_reaction_y_N,mechanical_constraint_reaction_z_N,"
              "contact_reaction_x_N,contact_reaction_y_N,contact_reaction_z_N,"
              "mechanical_constraint_moment_y_Nm,contact_moment_y_Nm,"
              "phase,max_free_displacement_m,max_free_speed_m_s,"
              "kinetic_energy_J,max_material_state_change,certificate_residual,"
              "certificate_correction,certificate_volume,certificate_pressure,certificate_raw_accepted_flag,step_accepted,right_grip_constrained,measured_right_grip_angle_deg,current_fixed_nodes,free_force_imbalance_l2_N,free_force_imbalance_max_N\n";
}

void writeObservation(std::ofstream& output,
                      const std::uint32_t step,
                      const double timestep,
                      const std::uint32_t environment,
                      const double targetAngleDegrees,
                      const NMMatterStatusGPU& status,
                      const Metrics& metrics,
                      const ReactionMetrics& reaction,
                      const char* phase, const NMSolverCertificateGPU& certificate, const bool stepAccepted,
                      const double measuredAngle, const bool crease, const double forceL2, const double forceMax) {
    output << step << ',' << std::setprecision(17)
           << static_cast<double>(step + 1u) * timestep << ','
           << environment << ','
           << (crease ? (environment == 0u ? "indented" : "stationary_tool_reference") : (environment == 0u ? "bent" : "held_reference")) << ','
           << targetAngleDegrees << ',' << status.code << ','
           << status.objectIndex << ',' << status.failingIndex << ','
           << status.completedMicrosteps << ',' << status.fgmresIterations << ','
           << status.diagnostics.x << ',' << status.diagnostics.y << ','
           << status.diagnostics.z << ',' << status.diagnostics.w << ','
           << metrics.minJ << ',' << metrics.maxJ
           << ',' << metrics.maxDisplacementM;
    if (stepAccepted) {
        output << ',' << reaction.force.x << ',' << reaction.force.y << ','
               << reaction.force.z << ',' << reaction.momentAboutY << ','
               << reaction.mechanicalConstraintForce.x << ','
               << reaction.mechanicalConstraintForce.y << ','
               << reaction.mechanicalConstraintForce.z << ','
               << reaction.contactForce.x << ',' << reaction.contactForce.y << ','
               << reaction.contactForce.z << ','
               << reaction.mechanicalConstraintMomentAboutY << ','
               << reaction.contactMomentAboutY;
    } else {
        // Rejected candidates are not reaction evidence. Keep the twelve
        // reaction fields empty while retaining the failed status/certificate.
        output << std::string(12u, ',');
    }
    output << ',' << phase << ','
           << metrics.maxFreeDisplacementM << ',' << metrics.maxFreeSpeedMps << ','
           << metrics.kineticEnergyJ << ',' << metrics.maxStateChange << ','
           << certificate.nonlinear.x << ',' << certificate.nonlinear.y << ','
           << certificate.nonlinear.z << ',' << certificate.nonlinear.w << ','
           << certificate.validity.w << ',' << (stepAccepted ? 1 : 0) << ','
           << (std::string_view(phase) == "release" ? 0 : 1) << ',' << measuredAngle << ','
           << metrics.currentFixedNodes << ',' << forceL2 << ',' << forceMax << '\n';
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
    output << "\n  ],\n  \"native_node_state\": [\n";
    for (std::size_t ni = 0; ni < world.dispatch.femNodeCount; ++ni) {
        const auto& node = snapshot.femNodes.at(environment * world.dispatch.femNodeCount + ni);
        if (ni) output << ",\n";
        output << "    {\"position_m\": [" << node.positionAndMass.x << ',' << node.positionAndMass.y << ',' << node.positionAndMass.z
            << "], \"velocity_m_s\": [" << node.velocityAndInverseMass.x << ',' << node.velocityAndInverseMass.y << ',' << node.velocityAndInverseMass.z
            << "], \"reference_displacement_m\": [" << node.referenceDisplacementAndMode.x << ',' << node.referenceDisplacementAndMode.y << ',' << node.referenceDisplacementAndMode.z
            << "], \"displacement_mode\": " << node.referenceDisplacementAndMode.w << ", \"constraint_tag\": " << node.restAndFixed.w << '}';
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
           << "  \"authored_region_map_sha256\": \""
           << mesh.authoredRegionMapDigest << "\",\n"
           << "  \"frame_map_sha256\": \""
           << mesh.frameMapDigest << "\",\n"
           << "  \"damage_overlay_mapping_sha256\": \""
           << mesh.damageOverlayMappingDigest << "\",\n"
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

struct BoxScoreAngleObservation {
    bool valid = false;
    double foldTurnDegrees = std::numeric_limits<double>::quiet_NaN();
    double interiorAngleDegrees = std::numeric_limits<double>::quiet_NaN();
    double targetFoldDegrees = 0.0;
    double relativeTargetFoldDegrees = std::numeric_limits<double>::quiet_NaN();
    std::uint32_t sampleCount = 0u;
};

double wrapBoxDegrees(double value) {
    value = std::fmod(value + 180.0, 360.0);
    if (value < 0.0) value += 360.0;
    return value - 180.0;
}

std::array<BoxScoreAngleObservation, 12> boxScoreAngles(
    const MeshSource& mesh,
    const std::vector<NMFEMNodeStateGPU>& actionStartNodes,
    const std::vector<NMFEMNodeStateGPU>& acceptedNodes,
    const std::uint32_t environment,
    const std::uint32_t nodeCount) {
    require(mesh.boxBlank && mesh.boxBlank->scoreLines.size() == 12u,
            "score-angle observation requires all authored score lines");
    const auto& blank = *mesh.boxBlank;
    std::array<BoxScoreAngleObservation, 12> result{};
    std::array<numi::cardboard::BoxWallFit, 4> wallFits{};
    for (std::uint32_t panel = 0u; panel < 4u; ++panel)
        wallFits[panel] = numi::cardboard::measureBoxWallInteriorFit(
            blank, blank.points, acceptedNodes, environment, nodeCount, panel);
    const auto tabFit = numi::cardboard::measureBoxTabInteriorFit(
        blank, blank.points, acceptedNodes, environment, nodeCount);
    for (std::uint32_t scoreIndex = 0u; scoreIndex < blank.scoreLines.size(); ++scoreIndex) {
        const auto& line = blank.scoreLines[scoreIndex];
        auto& observation = result[scoreIndex];
        observation.targetFoldDegrees = line.targetFoldDegrees;
        if (line.kind == numi::cardboard::BoxScoreKind::manufacturerJointRoot) {
            if (!wallFits[0].valid || !tabFit.valid) continue;
            observation.relativeTargetFoldDegrees =
                numi::cardboard::boxWallRelativeTargetTurnDegrees(0u);
            observation.foldTurnDegrees = wrapBoxDegrees(
                wallFits[0].angleDegrees - tabFit.angleDegrees);
            observation.sampleCount = std::min(wallFits[0].sampleCount, tabFit.sampleCount);
            observation.valid = true;
        } else if (line.kind == numi::cardboard::BoxScoreKind::bodyCorner) {
            const std::uint32_t left = line.panelIndex;
            const std::uint32_t right = left + 1u;
            if (right >= wallFits.size() || !wallFits[left].valid || !wallFits[right].valid)
                continue;
            observation.relativeTargetFoldDegrees =
                numi::cardboard::boxWallRelativeTargetTurnDegrees(right);
            observation.foldTurnDegrees = wrapBoxDegrees(
                wallFits[right].angleDegrees - wallFits[left].angleDegrees);
            observation.sampleCount = std::min(wallFits[left].sampleCount,
                                               wallFits[right].sampleCount);
            observation.valid = true;
        } else {
            const auto flapAction = std::find_if(
                blank.assembly.actions.begin(), blank.assembly.actions.end(),
                [scoreIndex](const auto& candidate) {
                    return candidate.kind ==
                               numi::cardboard::BoxAssemblyAction::Kind::foldFlap &&
                           candidate.feature.index == scoreIndex;
                });
            if (flapAction != blank.assembly.actions.end())
                observation.relativeTargetFoldDegrees = flapAction->targetAngleDegrees;
            const auto turn = numi::cardboard::measureBoxFlapInteriorTurn(
                blank, blank.points, actionStartNodes, acceptedNodes, environment,
                nodeCount, scoreIndex);
            if (!turn.valid) continue;
            observation.foldTurnDegrees = turn.signedTurnDegrees;
            observation.sampleCount = turn.sampleCount;
            observation.valid = true;
        }
        if (observation.valid)
            observation.interiorAngleDegrees = std::clamp(
                180.0 - std::abs(observation.foldTurnDegrees), 0.0, 180.0);
    }
    return result;
}

struct BoxSpringbackObservation {
    bool valid = false;
    double heldTurnDegrees = std::numeric_limits<double>::quiet_NaN();
    double releasedTurnDegrees = std::numeric_limits<double>::quiet_NaN();
    double springbackTurnDegrees = std::numeric_limits<double>::quiet_NaN();
    double heldFitRmsM = std::numeric_limits<double>::quiet_NaN();
    double releasedFitRmsM = std::numeric_limits<double>::quiet_NaN();
    double fitBoundM = std::numeric_limits<double>::quiet_NaN();
    double heldFitCoverage = std::numeric_limits<double>::quiet_NaN();
    double releasedFitCoverage = std::numeric_limits<double>::quiet_NaN();
    std::uint32_t sampleCount = 0u;
    bool heldFitPassed = false;
    bool releasedFitPassed = false;
};

BoxSpringbackObservation boxSpringbackObservation(
    const MeshSource& mesh,
    const std::vector<NMFEMNodeStateGPU>& flatReferenceNodes,
    const std::vector<NMFEMNodeStateGPU>& heldNodes,
    const std::vector<NMFEMNodeStateGPU>& releasedNodes,
    const std::uint32_t environment,
    const std::uint32_t nodeCount,
    const std::uint32_t scoreIndex,
    const double residualBoundCalipers,
    const double minimumCoverage) {
    require(mesh.boxBlank && scoreIndex < mesh.boxBlank->scoreLines.size(),
            "springback observation requires an authored score line");
    const auto& blank = *mesh.boxBlank;
    const auto& line = blank.scoreLines[scoreIndex];
    BoxSpringbackObservation result;
    const double commonBoundM = residualBoundCalipers * blank.boardCaliper;
    const auto passes = [&](const double rms, const double coverage,
                            const double bound) {
        return std::isfinite(rms) && std::isfinite(coverage) &&
            std::isfinite(bound) && bound > 0.0 && rms <= bound &&
            coverage >= minimumCoverage;
    };
    if (line.kind == numi::cardboard::BoxScoreKind::bottomFlap ||
        line.kind == numi::cardboard::BoxScoreKind::topFlap) {
        const auto held = numi::cardboard::measureBoxFlapInteriorTurn(
            blank, blank.points, flatReferenceNodes, heldNodes,
            environment, nodeCount, scoreIndex, residualBoundCalipers);
        const auto released = numi::cardboard::measureBoxFlapInteriorTurn(
            blank, blank.points, flatReferenceNodes, releasedNodes,
            environment, nodeCount, scoreIndex, residualBoundCalipers);
        if (!held.valid || !released.valid) return result;
        result.heldTurnDegrees = held.signedTurnDegrees;
        result.releasedTurnDegrees = released.signedTurnDegrees;
        result.heldFitRmsM = held.fitResidualRms;
        result.releasedFitRmsM = released.fitResidualRms;
        result.fitBoundM = std::max(held.residualBound, released.residualBound);
        result.heldFitCoverage = held.fitCoverage;
        result.releasedFitCoverage = released.fitCoverage;
        result.sampleCount = std::min(held.sampleCount, released.sampleCount);
        result.heldFitPassed = passes(held.fitResidualRms,
            held.fitCoverage, held.residualBound);
        result.releasedFitPassed = passes(released.fitResidualRms,
            released.fitCoverage, released.residualBound);
    } else {
        struct RelativeTurn {
            bool valid = false;
            double degrees = std::numeric_limits<double>::quiet_NaN();
            double rmsM = std::numeric_limits<double>::quiet_NaN();
            double coverage = std::numeric_limits<double>::quiet_NaN();
            std::uint32_t count = 0u;
        };
        const auto relativeTurn = [&](const std::vector<NMFEMNodeStateGPU>& nodes) {
            RelativeTurn turn;
            std::vector<numi::cardboard::BoxWallFit> wallFits;
            wallFits.reserve(blank.panels.size());
            for (std::uint32_t panel = 0u; panel < blank.panels.size(); ++panel)
                wallFits.push_back(numi::cardboard::measureBoxWallInteriorFit(
                    blank, blank.points, nodes, environment, nodeCount, panel,
                    residualBoundCalipers));
            const auto tabFit = numi::cardboard::measureBoxTabInteriorFit(
                blank, blank.points, nodes, environment, nodeCount,
                residualBoundCalipers);
            double firstAngle = 0.0, secondAngle = 0.0;
            double rms = 0.0, coverage = 1.0;
            std::uint32_t samples = 0u;
            if (line.kind == numi::cardboard::BoxScoreKind::manufacturerJointRoot) {
                if (wallFits.empty() || !wallFits[0].valid || !tabFit.valid)
                    return turn;
                firstAngle = tabFit.angleDegrees;
                secondAngle = wallFits[0].angleDegrees;
                rms = std::max(wallFits[0].fitResidualRms,
                               tabFit.fitResidualRms);
                coverage = std::min(wallFits[0].fitCoverage,
                                    tabFit.fitCoverage);
                samples = std::min(wallFits[0].sampleCount,
                                  tabFit.sampleCount);
            } else if (line.kind == numi::cardboard::BoxScoreKind::bodyCorner) {
                const std::uint32_t left = line.panelIndex;
                const std::uint32_t right = left + 1u;
                if (right >= wallFits.size() || !wallFits[left].valid ||
                    !wallFits[right].valid) return turn;
                firstAngle = wallFits[left].angleDegrees;
                secondAngle = wallFits[right].angleDegrees;
                rms = std::max(wallFits[left].fitResidualRms,
                               wallFits[right].fitResidualRms);
                coverage = std::min(wallFits[left].fitCoverage,
                                    wallFits[right].fitCoverage);
                samples = std::min(wallFits[left].sampleCount,
                                  wallFits[right].sampleCount);
            } else {
                return turn;
            }
            turn.valid = true;
            turn.degrees = wrapBoxDegrees(secondAngle - firstAngle);
            turn.rmsM = rms;
            turn.coverage = coverage;
            turn.count = samples;
            return turn;
        };
        const auto held = relativeTurn(heldNodes);
        const auto released = relativeTurn(releasedNodes);
        if (!held.valid || !released.valid) return result;
        result.heldTurnDegrees = held.degrees;
        result.releasedTurnDegrees = released.degrees;
        result.heldFitRmsM = held.rmsM;
        result.releasedFitRmsM = released.rmsM;
        result.fitBoundM = commonBoundM;
        result.heldFitCoverage = held.coverage;
        result.releasedFitCoverage = released.coverage;
        result.sampleCount = std::min(held.count, released.count);
        result.heldFitPassed = passes(held.rmsM, held.coverage, commonBoundM);
        result.releasedFitPassed = passes(
            released.rmsM, released.coverage, commonBoundM);
    }
    result.springbackTurnDegrees = wrapBoxDegrees(
        result.releasedTurnDegrees - result.heldTurnDegrees);
    result.valid = result.heldFitPassed && result.releasedFitPassed &&
        std::isfinite(result.springbackTurnDegrees);
    return result;
}

bool boxEnvironmentAccepted(const CompiledWorld& world,
                            const RuntimeStateSnapshot& snapshot,
                            const std::uint32_t environment) {
    if (environment >= snapshot.statuses.size() ||
        environment >= snapshot.solverCertificates.size()) return false;
    const auto& status = snapshot.statuses[environment];
    const auto& certificate = snapshot.solverCertificates[environment];
    return status.code == NM_STATUS_SUCCESS && certificate.validity.w > 0.5f &&
        std::isfinite(certificate.nonlinear.x) &&
        std::isfinite(certificate.nonlinear.y) &&
        std::isfinite(certificate.nonlinear.z) &&
        std::isfinite(certificate.nonlinear.w) &&
        certificate.nonlinear.x <= world.mixedSolver.residualTolerances.x &&
        certificate.nonlinear.z <= world.mixedSolver.residualTolerances.y &&
        certificate.nonlinear.w <= world.mixedSolver.residualTolerances.z;
}

struct ShippingHistoryAreaObservables {
    std::string paperStatus = "not_enabled";
    double paperAreaWeightedDamageFraction =
        std::numeric_limits<double>::quiet_NaN();
    double paperReferencePlaneAreaM2 =
        std::numeric_limits<double>::quiet_NaN();
    std::uint64_t paperOverlayTetrahedra = 0u;
    std::string starchStatus = "not_enabled";
    double starchAreaWeightedDamageFraction =
        std::numeric_limits<double>::quiet_NaN();
    double starchReferenceAreaM2 =
        std::numeric_limits<double>::quiet_NaN();
    std::uint64_t starchActiveFaces = 0u;
    double starchFullyFailedAreaM2 = 0.0;
    double starchFullyFailedAreaFraction =
        std::numeric_limits<double>::quiet_NaN();
    std::uint64_t starchFullyFailedFaces = 0u;
    std::string starchFullyFailedStatus = "not_enabled";
    std::string hmaStatus = "not_enabled";
    double hmaAreaWeightedDamageFraction =
        std::numeric_limits<double>::quiet_NaN();
    double hmaReferenceAreaM2 =
        std::numeric_limits<double>::quiet_NaN();
    std::uint64_t hmaActiveFaces = 0u;
    double hmaFullyFailedAreaM2 = 0.0;
    double hmaFullyFailedAreaFraction =
        std::numeric_limits<double>::quiet_NaN();
    std::uint64_t hmaFullyFailedFaces = 0u;
    std::string hmaFullyFailedStatus = "not_enabled";
};

ShippingHistoryAreaObservables shippingHistoryAreaObservables(
    const CompiledWorld& world, const MeshSource& mesh,
    const RuntimeStateSnapshot& snapshot, const std::uint32_t environment) {
    ShippingHistoryAreaObservables result;
    const std::uint32_t tetrahedronCount = world.dispatch.tetrahedronCount;
    const std::uint32_t faceCount = world.dispatch.cohesiveFaceCount;
    const std::uint32_t stride = snapshot.materialStateStride;
    if (environment >= world.dispatch.environmentCount || stride == 0u)
        return result;

    if (!mesh.damageOverlayMappings.empty()) {
        result.paperStatus = "unavailable";
        if (mesh.damageOverlayMappings.size() !=
                mesh.damageOverlayMaterialIndices.size() ||
            snapshot.femMaterialState.size() <
                static_cast<std::size_t>(world.dispatch.environmentCount) *
                    tetrahedronCount * stride)
            return result;
        double weightedDamage = 0.0;
        double referenceArea = 0.0;
        bool valid = true;
        std::map<std::uint32_t, std::array<std::uint32_t, 8>> stateIndicesByMaterial;
        for (std::size_t mappingIndex = 0u;
             mappingIndex < mesh.damageOverlayMappings.size(); ++mappingIndex) {
            const auto& mapping = mesh.damageOverlayMappings[mappingIndex];
            const std::uint32_t tetIndex = mapping.elementIndex;
            if (tetIndex >= world.fem.tetrahedra.size() ||
                tetIndex >= tetrahedronCount ||
                mesh.damageOverlayMaterialIndices[mappingIndex] >=
                    world.materials.size() ||
                world.fem.tetrahedra[tetIndex].identity.x !=
                    mesh.damageOverlayMaterialIndices[mappingIndex]) {
                valid = false;
                break;
            }
            const auto& material = world.constitutive.at(
                mesh.damageOverlayMaterialIndices[mappingIndex]);
            auto stateIndicesIt = stateIndicesByMaterial.find(
                mesh.damageOverlayMaterialIndices[mappingIndex]);
            if (stateIndicesIt == stateIndicesByMaterial.end()) {
                std::array<std::uint32_t, 8> stateIndices{};
                for (std::uint32_t mode = 0u; mode < stateIndices.size(); ++mode) {
                    const std::string name = "overlay_fracture_excess_" +
                        std::to_string(mode);
                    const auto state = std::find_if(
                        material.material.internalState.begin(),
                        material.material.internalState.end(), [&](const auto& entry) {
                            return entry.name == name;
                        });
                    if (state == material.material.internalState.end()) {
                        valid = false;
                        break;
                    }
                    stateIndices[mode] = static_cast<std::uint32_t>(
                        state - material.material.internalState.begin());
                }
                if (!valid) break;
                stateIndicesIt = stateIndicesByMaterial.emplace(
                    mesh.damageOverlayMaterialIndices[mappingIndex],
                    stateIndices).first;
            }
            const auto& stateIndices = stateIndicesIt->second;
            const std::size_t stateBase =
                (static_cast<std::size_t>(environment) * tetrahedronCount +
                 tetIndex) * stride;
            for (std::uint32_t plane = 0u; plane < 4u; ++plane) {
                const double area = mapping.planeGeometry[plane].referenceAreaM2;
                if (!(std::isfinite(area) && area > 0.0)) {
                    valid = false;
                    break;
                }
                double planeSurvival = 1.0;
                for (std::uint32_t localMode = 0u; localMode < 2u; ++localMode) {
                    const std::uint32_t mode = 2u * plane + localMode;
                    const std::size_t stateIndex = stateIndices[mode];
                    if (stateIndex >= stride) {
                        valid = false;
                        break;
                    }
                    const double excess = snapshot.femMaterialState.at(
                        stateBase + stateIndex);
                    const double onset = mapping.modeOnsetStrain[mode];
                    const double softening = mapping.modeSofteningStrain[mode];
                    if (!(std::isfinite(excess) && excess >= 0.0 &&
                          std::isfinite(onset) && onset > 0.0 &&
                          std::isfinite(softening) && softening > 0.0)) {
                        valid = false;
                        break;
                    }
                    const double survival = onset / (onset + excess) *
                        std::exp(-excess / softening);
                    if (!(std::isfinite(survival) && survival >= 0.0 &&
                          survival <= 1.0)) {
                        valid = false;
                        break;
                    }
                    planeSurvival *= survival;
                }
                if (!valid) break;
                const double damage = mapping.damageCap * (1.0 - planeSurvival);
                weightedDamage += area * damage;
                referenceArea += area;
            }
            if (!valid) break;
            ++result.paperOverlayTetrahedra;
        }
        if (valid && referenceArea > 0.0 && std::isfinite(weightedDamage)) {
            result.paperStatus = "available_model_area_weighted_damage_index";
            result.paperAreaWeightedDamageFraction = weightedDamage / referenceArea;
            result.paperReferencePlaneAreaM2 = referenceArea;
        } else {
            result.paperStatus = "unavailable_invalid_or_incomplete_accepted_history";
            result.paperOverlayTetrahedra = 0u;
        }
    }

    const auto starchFaceCount = mesh.starchPairedFaces.size();
    std::set<std::uint32_t> hmaStableIds;
    for (const auto& patch : mesh.hmaClosureMesh.patches)
        hmaStableIds.insert(patch.stableIdentifier);
    const bool starchEnabled = starchFaceCount != 0u;
    const bool hmaEnabled = !hmaStableIds.empty();
    result.starchStatus = starchEnabled ? "no_active_faces_yet" : "not_enabled";
    result.hmaStatus = hmaEnabled ? "no_active_faces_yet" : "not_enabled";
    result.starchFullyFailedStatus = starchEnabled ? "no_active_faces_yet" : "not_enabled";
    result.hmaFullyFailedStatus = hmaEnabled ? "no_active_faces_yet" : "not_enabled";
    if (snapshot.cohesiveFaces.size() <
            static_cast<std::size_t>(world.dispatch.environmentCount) * faceCount)
        return result;
    double starchWeightedDamage = 0.0;
    double starchArea = 0.0;
    double hmaWeightedDamage = 0.0;
    double hmaArea = 0.0;
    bool validFaces = true;
    const std::size_t faceBase = static_cast<std::size_t>(environment) * faceCount;
    for (std::uint32_t faceIndex = 0u; faceIndex < faceCount; ++faceIndex) {
        const auto& face = snapshot.cohesiveFaces.at(faceBase + faceIndex);
        if (face.lawIdentity.x != NM_COHESIVE_LAW_MIXED_MODE_BILINEAR)
            continue;
        const std::uint32_t stableId = face.adjacency.z;
        const bool isStarch = stableId != 0u &&
            stableId <= starchFaceCount;
        const bool isHma = hmaStableIds.contains(stableId);
        if (!isStarch && !isHma) continue;
        if ((face.adjacency.w & NM_TOPOLOGY_COHESIVE_DORMANT) != 0u)
            continue;
        const double area = face.geometry.w;
        const double damage = face.state.x;
        if (!(std::isfinite(area) && area > 0.0 &&
              std::isfinite(damage) && damage >= 0.0 && damage <= 1.0)) {
            validFaces = false;
            break;
        }
        if (isStarch) {
            starchArea += area;
            starchWeightedDamage += area * damage;
            ++result.starchActiveFaces;
            if (damage >= 1.0) {
                result.starchFullyFailedAreaM2 += area;
                ++result.starchFullyFailedFaces;
            }
        } else {
            hmaArea += area;
            hmaWeightedDamage += area * damage;
            ++result.hmaActiveFaces;
            if (damage >= 1.0) {
                result.hmaFullyFailedAreaM2 += area;
                ++result.hmaFullyFailedFaces;
            }
        }
    }
    if (!validFaces) {
        if (starchEnabled) result.starchStatus = "unavailable_invalid_accepted_history";
        if (hmaEnabled) result.hmaStatus = "unavailable_invalid_accepted_history";
        if (starchEnabled) result.starchFullyFailedStatus = "unavailable_invalid_accepted_history";
        if (hmaEnabled) result.hmaFullyFailedStatus = "unavailable_invalid_accepted_history";
        return result;
    }
    if (starchEnabled && starchArea > 0.0) {
        result.starchStatus = "available_area_weighted_cohesive_damage_not_geometric_open_area";
        result.starchAreaWeightedDamageFraction = starchWeightedDamage / starchArea;
        result.starchReferenceAreaM2 = starchArea;
        result.starchFullyFailedAreaFraction =
            result.starchFullyFailedAreaM2 / starchArea;
        result.starchFullyFailedStatus =
            "available_zero_tensile_shear_capacity_fraction_not_geometric_open_area";
    }
    if (hmaEnabled && hmaArea > 0.0) {
        result.hmaStatus = "available_area_weighted_cohesive_damage_not_geometric_open_area";
        result.hmaAreaWeightedDamageFraction = hmaWeightedDamage / hmaArea;
        result.hmaReferenceAreaM2 = hmaArea;
        result.hmaFullyFailedAreaFraction = result.hmaFullyFailedAreaM2 / hmaArea;
        result.hmaFullyFailedStatus =
            "available_zero_tensile_shear_capacity_fraction_not_geometric_open_area";
    }
    return result;
}

void writeShippingHistoryAreaHeader(std::ostream& output) {
    output << "native_step,simulation_time_s,dt_s,stage,accepted,"
              "paper_status,paper_area_weighted_damage_fraction,paper_reference_plane_area_m2,paper_overlay_tetrahedra,"
              "starch_status,starch_area_weighted_cohesive_damage_fraction,starch_active_reference_area_m2,starch_active_faces,"
              "starch_fully_failed_status,starch_fully_failed_cohesive_area_m2,"
              "starch_fully_failed_cohesive_area_fraction,starch_fully_failed_faces,"
              "hma_status,hma_area_weighted_cohesive_damage_fraction,hma_active_reference_area_m2,hma_active_faces,"
              "hma_fully_failed_status,hma_fully_failed_cohesive_area_m2,"
              "hma_fully_failed_cohesive_area_fraction,hma_fully_failed_faces\n";
}

void writeShippingHistoryAreaRow(
    std::ostream& output, const std::uint32_t nativeStep,
    const double simulationTimeSeconds, const double timestepSeconds,
    const std::string& stage, const bool accepted,
    const ShippingHistoryAreaObservables& values) {
    output << nativeStep << ',' << std::setprecision(17)
        << simulationTimeSeconds << ',' << timestepSeconds << ','
        << stage << ',' << (accepted ? 1 : 0) << ','
        << values.paperStatus << ',';
    if (std::isfinite(values.paperAreaWeightedDamageFraction))
        output << values.paperAreaWeightedDamageFraction;
    output << ',';
    if (std::isfinite(values.paperReferencePlaneAreaM2))
        output << values.paperReferencePlaneAreaM2;
    output << ',' << values.paperOverlayTetrahedra << ','
        << values.starchStatus << ',';
    if (std::isfinite(values.starchAreaWeightedDamageFraction))
        output << values.starchAreaWeightedDamageFraction;
    output << ',';
    if (std::isfinite(values.starchReferenceAreaM2))
        output << values.starchReferenceAreaM2;
    output << ',' << values.starchActiveFaces << ','
        << values.starchFullyFailedStatus << ',';
    if (std::isfinite(values.starchFullyFailedAreaM2))
        output << values.starchFullyFailedAreaM2;
    output << ',';
    if (std::isfinite(values.starchFullyFailedAreaFraction))
        output << values.starchFullyFailedAreaFraction;
    output << ',' << values.starchFullyFailedFaces << ','
        << values.hmaStatus << ',';
    if (std::isfinite(values.hmaAreaWeightedDamageFraction))
        output << values.hmaAreaWeightedDamageFraction;
    output << ',';
    if (std::isfinite(values.hmaReferenceAreaM2))
        output << values.hmaReferenceAreaM2;
    output << ',' << values.hmaActiveFaces << ','
        << values.hmaFullyFailedStatus << ',';
    if (std::isfinite(values.hmaFullyFailedAreaM2))
        output << values.hmaFullyFailedAreaM2;
    output << ',';
    if (std::isfinite(values.hmaFullyFailedAreaFraction))
        output << values.hmaFullyFailedAreaFraction;
    output << ',' << values.hmaFullyFailedFaces << '\n';
}

struct BoxNativeActionRecord {
    std::string status = "not_attempted";
    std::string diagnostic;
    std::uint32_t firstNativeStep = NM_INVALID_INDEX;
    std::uint32_t lastNativeStep = NM_INVALID_INDEX;
    std::uint32_t acceptedNativeSteps = 0u;
    bool angleValid = false;
    double observedFoldTurnDegrees = 0.0;
    double observedInteriorAngleDegrees = 0.0;
    bool shapeFitObservable = false;
    double shapeFitResidualRmsM = 0.0;
    double shapeFitResidualBoundM = 0.0;
    double shapeFitCoverage = 0.0;
    bool shapeFitWithinBound = false;
    bool shapeFitCompletionRejected = false;
};

struct BoxFixtureJointPair {
    std::uint32_t tabNode = 0u;
    std::uint32_t seamNode = 0u;
    numi::cardboard::BoxVec3 tabStart{};
    numi::cardboard::BoxVec3 tabEnd{};
    numi::cardboard::BoxVec3 seamEndpoint{};
};

void writeBoxActionCsv(const std::filesystem::path& path,
                       const numi::cardboard::FEFCO0201Blank& blank,
                       const std::vector<BoxNativeActionRecord>& rows) {
    std::ofstream output(path, std::ios::trunc);
    require(output.good(), "cannot create box action ledger");
    output << "action_index,stage,kind,feature_type,feature_index,source_target_angle_deg,relative_target_angle_deg,status,first_native_step,last_native_step,accepted_native_steps,observed_fold_turn_deg,observed_interior_angle_deg,physical_bond_law_required,shape_fit_residual_rms_m,shape_fit_residual_bound_m,shape_fit_coverage,shape_fit_within_bound,shape_fit_completion_rejected,diagnostic\n"
           << std::setprecision(17);
    for (std::size_t index = 0u; index < blank.assembly.actions.size(); ++index) {
        const auto& action = blank.assembly.actions[index];
        const auto& row = rows.at(index);
        const bool bond = action.kind == numi::cardboard::BoxAssemblyAction::Kind::bondManufacturerJoint ||
            action.kind == numi::cardboard::BoxAssemblyAction::Kind::closeEndWithTapeOrAdhesive;
        const auto csv = [](const std::string& value) {
            std::string escaped = "\"";
            for (const char c : value) {
                if (c == '"') escaped += "\"\"";
                else escaped += c;
            }
            return escaped + "\"";
        };
        output << index << ',' << csv(action.stage) << ','
            << static_cast<unsigned>(action.kind) << ','
            << csv(numi::cardboard::boxAssemblyFeatureKindName(action.feature.kind)) << ','
            << action.feature.index << ',' << action.targetAngleDegrees << ',';
        if (action.kind == numi::cardboard::BoxAssemblyAction::Kind::erectWallPanel)
            output << numi::cardboard::boxWallRelativeTargetTurnDegrees(
                action.feature.index);
        else if (action.kind == numi::cardboard::BoxAssemblyAction::Kind::foldFlap)
            output << action.targetAngleDegrees;
        output << ',' << csv(row.status) << ',' << row.firstNativeStep << ','
            << row.lastNativeStep << ',' << row.acceptedNativeSteps << ',';
        if (row.angleValid)
            output << row.observedFoldTurnDegrees << ',' << row.observedInteriorAngleDegrees;
        else output << ",";
        output << ',' << (bond ? 1 : 0) << ',';
        if (row.shapeFitObservable)
            output << row.shapeFitResidualRmsM << ',' << row.shapeFitResidualBoundM
                   << ',' << row.shapeFitCoverage << ','
                   << (row.shapeFitWithinBound ? 1 : 0) << ','
                   << (row.shapeFitCompletionRejected ? 1 : 0);
        else
            output << ",,,,";
        output << ',' << csv(row.diagnostic) << '\n';
    }
    require(output.good(), "failed writing box action ledger");
}

void writeBoxGripPatchCsv(
    const std::filesystem::path& path,
    const numi::cardboard::FEFCO0201Blank& blank,
    const std::vector<numi::cardboard::BoxFeatureGrip>& grips) {
    std::ofstream output(path, std::ios::trunc);
    require(output.good(), "cannot create box grip reference-patch ledger");
    output << "action_index,stage,panel_index,score_line_index,patch_index,plane_axis,plane_coordinate_m,min_x_m,max_x_m,min_y_m,max_y_m,node_count,node_indices,interpretation\n"
           << std::setprecision(17);
    for (const auto& grip : grips) {
        const auto& action = blank.assembly.actions.at(grip.assemblyActionIndex);
        for (std::size_t patchIndex = 0u;
             patchIndex < grip.referencePatches.size(); ++patchIndex) {
            const auto& patch = grip.referencePatches[patchIndex];
            output << grip.assemblyActionIndex << ",\"" << action.stage << "\","
                << grip.panelIndex << ',' << grip.scoreLineIndex << ','
                << patchIndex << ',' << patch.planeAxis << ','
                << patch.planeCoordinate << ',' << patch.minimumX << ','
                << patch.maximumX << ',' << patch.minimumY << ','
                << patch.maximumY << ',' << patch.nodeCount << ",\"";
            for (std::size_t node = 0u; node < patch.nodeIndices.size(); ++node) {
                if (node != 0u) output << ';';
                output << patch.nodeIndices[node];
            }
            output << "\",\"temporary kinematic clamp reference patch on compiled exterior boundary nodes; not rigid-body contact\"\n";
        }
    }
    require(output.good(), "failed writing box grip reference-patch ledger");
}

int executeBoxAssemblyNative(
    Arguments& arguments, const MeshSource& mesh,
    const CompileResult& compiled,
    const numi::cardboard::ShippingMemoryBudget* shippingArenaBudget = nullptr) {
    require(mesh.boxBlank && mesh.boxScoreTooling,
            "native FEFCO box tooling was not compiled");
    const auto& blank = *mesh.boxBlank;
    const auto& tooling = *mesh.boxScoreTooling;
    const std::uint32_t scoreToolSceneBodyCount = tooling.sequentialFamilies
        ? static_cast<std::uint32_t>(tooling.families.size() *
              numi::cardboard::kShippingSceneBodiesPerScoreFamily) : 2u;
#ifdef NUMI_CARDBOARD_METAL_WORLD
    const std::uint32_t shippingPlatenBodyIndex = scoreToolSceneBodyCount;
    const std::uint32_t shippingFloorBodyIndex = scoreToolSceneBodyCount + 1u;
    const std::uint32_t shippingToolSceneBodyCount = scoreToolSceneBodyCount + 2u;
    const std::uint32_t shippingFreeBodyIndex = shippingToolSceneBodyCount;
    const std::uint32_t shippingSceneBodyCount = shippingToolSceneBodyCount + 1u;
    if (arguments.shippingBox)
        require(tooling.sequentialFamilies &&
                    tooling.families.size() ==
                        numi::cardboard::kShippingScoreToolFamilyCount &&
                    tooling.proxies.size() ==
                        numi::cardboard::kShippingScoreToolProxyCount &&
                    scoreToolSceneBodyCount ==
                        numi::cardboard::kShippingScoreToolSceneBodyCount &&
                    shippingSceneBodyCount ==
                        numi::cardboard::kShippingRigidBodyCount &&
                    compiled.world.dispatch.rigidProxyCount ==
                        numi::cardboard::kShippingRigidProxyCount,
                "compiled shipping tools/body slots diverged from shared prebuild counts");
#endif
    const double compiledSubstrateMassKg = arguments.shippingBox
        ? compiledSubstrateNodalMassKg(compiled.world) : 0.0;
    const std::uint32_t nodeCount = compiled.world.dispatch.femNodeCount;
    require(blank.scoreLines.size() == 12u && blank.assembly.actions.size() == 15u,
            "native box driver requires the complete 12-score/15-action source layout");

    std::ofstream stepOutput(arguments.output / "box-native-steps.csv");
    std::ofstream angleOutput(arguments.output / "box-score-angles.csv");
    std::ofstream gripReactionOutput;
    std::ofstream springbackOutput;
    require(stepOutput.good() && angleOutput.good(),
            "cannot create native box observation files");
    stepOutput << "native_step,stage,action_index,accepted,status_env0,status_env1,fgmres_env0,fgmres_env1,certificate_valid_env0,certificate_residual_env0,certificate_correction_env0,certificate_volume_env0,certificate_pressure_env0,simulation_time_s,dt_s,error\n";
    angleOutput << "native_step,stage,action_index,environment,step_accepted,score_line_index,score_kind,fit_valid,fold_turn_deg,interior_angle_deg,source_target_fold_deg,relative_target_fold_deg,sample_count\n"
                << std::setprecision(17);
    if (arguments.shippingBox) {
        gripReactionOutput.open(arguments.output / "box-grip-reactions.csv");
        springbackOutput.open(arguments.output / "box-score-springback.csv");
        require(gripReactionOutput.good() && springbackOutput.good(),
                "cannot create accepted-grip-reaction/springback ledgers");
        gripReactionOutput
            << "native_step,time_s,stage,assembly_action_index,score_line_index,environment,step_accepted,reaction_valid,active_grip_node_count,pivot_x_m,pivot_y_m,pivot_z_m,mechanical_constraint_force_x_N,mechanical_constraint_force_y_N,mechanical_constraint_force_z_N,contact_force_x_N,contact_force_y_N,contact_force_z_N,total_force_x_N,total_force_y_N,total_force_z_N,mechanical_constraint_moment_x_Nm,mechanical_constraint_moment_y_Nm,mechanical_constraint_moment_z_Nm,contact_moment_x_Nm,contact_moment_y_Nm,contact_moment_z_Nm,total_moment_x_Nm,total_moment_y_Nm,total_moment_z_Nm\n"
            << std::setprecision(17);
        springbackOutput
            << "environment,score_line_index,score_kind,held_native_step,held_time_s,released_native_step,released_time_s,released_settled,held_turn_deg,released_turn_deg,springback_turn_deg,held_fit_rms_m,released_fit_rms_m,fit_bound_m,held_fit_coverage,released_fit_coverage,sample_count,held_fit_pass,released_fit_pass,springback_observable,reason\n"
            << std::setprecision(17);
    }
    std::vector<BoxNativeActionRecord> actionRows(blank.assembly.actions.size());
    writeBoxActionCsv(arguments.output / "box-actions.csv", blank, actionRows);

    std::ofstream shippingContactLedger;
    std::ofstream shippingCompressionLedger;
    std::ofstream shippingGravityTransferLedger;
    std::ofstream shippingImpactLedger;
    std::ofstream shippingImpactDynamicsLedger;
    std::ofstream shippingHistoryAreaLedger;
    [[maybe_unused]] const auto appendShippingVector3 = [](
        std::ostream& stream, const std::array<double, 3>& vector) {
        stream << vector[0] << ',' << vector[1] << ',' << vector[2];
    };
    NativeRun run(compiled.world, arguments.metallib, arguments.shippingBox);

    if (arguments.shippingBox) {
        std::ofstream memory(arguments.output / "memory-observations.csv");
        require(memory.good(), "cannot create shipping memory observations");
        memory << "native_step,stage,runtime_resident_bytes,metal_current_allocated_bytes,metal_recommended_working_set_bytes,host_phys_footprint_bytes,host_sample_available,largest_measured_domain_bytes,within_16gib\n";
        memory.close();
        run.writeMemoryObservation(arguments.output / "memory-observations.csv",
                                   0u, "initialized");
        const auto staticMemory = shippingStaticMemoryLowerBound(compiled.world);
        const auto arenaBudget = numi::cardboard::estimateShippingMemoryBudget(
            shippingMemoryBudgetInput(arguments, mesh, compiled));
        writeShippingMemoryReport(
            arguments.output / "shipping-memory-native-initialized.json",
            arguments, mesh, compiled.world, staticMemory,
            &run.lastMemoryObservation(), &arenaBudget);
        shippingContactLedger.open(
            arguments.output / "shipping-contact-observations.csv");
        shippingCompressionLedger.open(
            arguments.output / "shipping-compression.csv");
        shippingGravityTransferLedger.open(
            arguments.output / "shipping-gravity-transfer.csv");
        shippingImpactLedger.open(
            arguments.output / "shipping-impact-cases.csv");
        shippingImpactDynamicsLedger.open(
            arguments.output / "shipping-impact-dynamics.csv");
        shippingHistoryAreaLedger.open(
            arguments.output / "shipping-history-observables.csv");
        require(shippingContactLedger.good() && shippingCompressionLedger.good() &&
                    shippingGravityTransferLedger.good() &&
                    shippingImpactLedger.good() &&
                    shippingImpactDynamicsLedger.good() &&
                    shippingHistoryAreaLedger.good(),
                "cannot create shipping compression/impact ledgers");
        shippingContactLedger
            << "native_step,stage,event_id,accepted,dt_s,rigid_contact_samples,platen_reaction_samples,impactor_reaction_samples,platen_impulse_y_Ns,platen_mean_force_y_N,impactor_impulse_y_Ns,impactor_mean_force_y_N,impactor_y_m,impactor_vy_m_s\n"
            << std::setprecision(17);
        shippingCompressionLedger
            << "native_step,phase,accepted,stroke_m,stroke_fraction,macro_observation_duration_s,control_step_count,control_step_dt_min_s,control_step_dt_max_s,platen_reaction_samples,platen_impulse_y_Ns,platen_mean_force_y_N,candidate_peak_force_N,qualified_fpeak_N,cumulative_platen_work_J,kinetic_energy_J,kinetic_to_work_ratio,peak_detector_drop_fraction\n"
            << std::setprecision(17);
        shippingGravityTransferLedger
            << "case_id,stage,native_step,case_step,time_s,dt_s,accepted,support_fraction,actuator_force_y_n,actuator_impulse_y_ns,body_y_m,body_velocity_x_m_s,body_velocity_y_m_s,body_velocity_z_m_s,momentum_x_kg_m_s,momentum_y_kg_m_s,momentum_z_kg_m_s,angular_momentum_x_kg_m2_s,angular_momentum_y_kg_m2_s,angular_momentum_z_kg_m2_s,delta_momentum_x_kg_m_s,delta_momentum_y_kg_m_s,delta_momentum_z_kg_m_s,gravity_impulse_x_ns,gravity_impulse_y_ns,gravity_impulse_z_ns,actuator_impulse_x_ns,actuator_impulse_y_ns,actuator_impulse_z_ns,native_contact_impulse_x_ns,native_contact_impulse_y_ns,native_contact_impulse_z_ns,native_contact_force_x_n,native_contact_force_y_n,native_contact_force_z_n,native_contact_samples,momentum_residual_x_ns,momentum_residual_y_ns,momentum_residual_z_ns,momentum_residual_normalized,actuator_work_j,gravity_work_j,rigid_translational_ke_j,rigid_rotational_ke_j,gravity_potential_relative_j,mechanical_energy_relative_j,fem_ke_env0_j,requested_gravity_force_y_n,requested_total_external_force_y_n,requested_total_external_impulse_y_ns\n"
            << std::setprecision(17);
        shippingImpactLedger
            << "case_id,mass_ratio_fpeak_over_g,mass_kg,location,offset_z_m,drop_height_m,impact_timestep_s,planned_steps,initial_potential_energy_j,source_checkpoint_sha256,mass_matched_seed_checkpoint,steps_accepted,first_contact_step,max_contact_force_N,max_speed_m_s,status\n"
            << std::setprecision(17);
        shippingImpactDynamicsLedger
            << "case_id,native_step,case_step,time_s,dt_s,accepted,body_y_m,body_vx_m_s,body_vy_m_s,body_vz_m_s,body_wx_rad_s,body_wy_rad_s,body_wz_rad_s,momentum_x_kg_m_s,momentum_y_kg_m_s,momentum_z_kg_m_s,angular_momentum_x_kg_m2_s,angular_momentum_y_kg_m2_s,angular_momentum_z_kg_m2_s,rigid_translational_ke_j,rigid_rotational_ke_j,gravity_potential_relative_j,rigid_mechanical_energy_relative_j,fem_ke_env0_j,matter_impulse_x_ns,matter_impulse_y_ns,matter_impulse_z_ns,matter_mean_force_y_n,matter_contact_samples,rigid_contact_samples\n"
            << std::setprecision(17);
        writeShippingHistoryAreaHeader(shippingHistoryAreaLedger);
    }
    RuntimeStateSnapshot currentSnapshot = run.runtime().snapshot();
    require(currentSnapshot.available, currentSnapshot.message);
    require(currentSnapshot.femNodes.size() >=
                static_cast<std::size_t>(arguments.environmentCount) * nodeCount,
            "native box initial FEM snapshot is truncated");
    const auto initialNodes = currentSnapshot.femNodes;
    auto activeActionStartNodes = initialNodes;
    const double parkHeight = blank.boardCaliper +
        2.0 * tooling.punchRadius + tooling.toolClearance;
    const double contactHeight = blank.boardCaliper + tooling.punchRadius;
    require(parkHeight > blank.boardCaliper + 2.0 * tooling.punchRadius,
            "box score die park position is invalid");

    std::uint32_t nextNativeStep = 0u;
    std::uint32_t acceptedNativeSteps = 0u;
    double shippingSimulationTimeSeconds = 0.0;
    double boxSimulationTimeSeconds = 0.0;
    double lastAcceptedTimestepSeconds = 0.0;
    std::uint32_t lastAcceptedNativeStep = NM_INVALID_INDEX;
    std::uint64_t shippingScoreApproachAcceptedSteps = 0u;
    std::uint64_t shippingScoreRetractAcceptedSteps = 0u;
    double shippingScoreEffectiveSpeedMinMPerS =
        std::numeric_limits<double>::infinity();
    double shippingScoreEffectiveSpeedMaxMPerS = 0.0;
    std::vector<NMCohesiveActivationCommandGPU> pendingStarchActivationCommands;
    std::vector<NMCohesiveActivationCommandGPU> pendingHmaActivationCommands;
    std::string pendingHmaActivationEvent;
    std::set<std::string> acceptedHmaActivationEvents;
    std::ofstream adhesiveActivationOutput;
    if (arguments.shippingBox) {
        adhesiveActivationOutput.open(arguments.output / "box-adhesive-activation.csv");
        require(adhesiveActivationOutput.good(),
                "cannot create shipping adhesive activation ledger");
        adhesiveActivationOutput
            << "event,patch_id,stable_identifier,accepted_endpoint_gap_m,activation_gap_tolerance_m,gap_gate_passed,activation_step_accepted\n"
            << std::setprecision(17);
    }
    bool starchInitialActivationAccepted = false;
    std::uint32_t starchInitialActivationStep = NM_INVALID_INDEX;
    std::string stopReason;
    bool nativeFailure = false;
    bool shapeFitRejection = false;
    bool shippingCompressionAttempted = false;
    bool shippingCompressionCompleted = false;
    bool shippingCompressionBothBranchesReachedPeak = false;
    bool shippingCompressionHalfSpeedCompleted = false;
    bool shippingDropCasesAttempted = false;
    bool shippingDropCasesCompleted = false;
    std::uint32_t shippingDropCasesAccepted = 0u;
    std::uint32_t shippingDropCasesPlanned =
        arguments.shippingImpactTimestepRefinement ? 18u : 6u;
    double shippingCompressionPeakForceN = 0.0;
    double shippingCompressionHalfSpeedPeakForceN = 0.0;
    double shippingCompressionHalfSpeedPeakStrokeM = 0.0;
    double shippingCompressionHalfSpeedRelativePeakDifference = 0.0;
    bool shippingCompressionRateSensitivityEvaluated = false;
    bool shippingCompressionRateSensitivityPassed = false;
    bool shippingCompressionNumericallyQualified = false;
    std::string shippingCompressionNumericalQualificationStatus = "not_evaluated";
    std::string shippingCompressionNumericalQualificationReason;
    std::string shippingCompressionNominalCurveSha256;
    std::uint64_t shippingCompressionNominalCurveSampleCount = 0u;
    bool shippingGravityTransferAttempted = false;
    bool shippingGravityTransferCompleted = false;
    std::uint32_t shippingGravityTransferCasesAccepted = 0u;
    double shippingGravityTransferMaximumMomentumResidualNs = 0.0;
    double shippingGravityTransferMaximumMomentumResidualNormalized = 0.0;
    double shippingCompressionCommonStrokeM = 0.0;
    double shippingCompressionCommonStrokeNominalWorkJ = 0.0;
    double shippingCompressionCommonStrokeHalfSpeedWorkJ = 0.0;
    double shippingCompressionRelativeWorkDifference = 0.0;
    double shippingCompressionMaximumNormalizedCurveDifference = 0.0;
    double shippingCompressionRmsNormalizedCurveDifference = 0.0;
    double shippingCompressionNominalWorkAtPeakJ = 0.0;
    double shippingCompressionHalfSpeedWorkAtPeakJ = 0.0;
    double shippingCompressionNominalKineticAtPeakJ = 0.0;
    double shippingCompressionHalfSpeedKineticAtPeakJ = 0.0;
    double shippingCompressionNominalKineticWorkRatioAtPeak = 0.0;
    double shippingCompressionHalfSpeedKineticWorkRatioAtPeak = 0.0;
    double shippingCompressionStrokeM = 0.0;
    double shippingCompressionPeakStrokeM = 0.0;
    std::uint64_t shippingCompressionPeakSampleIndex = 0u;
    double shippingFloorPlacementClearanceM = 0.0;
    std::uint32_t shippingFloorPlacementSteps = 0u;
    [[maybe_unused]] std::array<double, 3> lastPlatenContactImpulseNs{};
    [[maybe_unused]] std::array<double, 3> lastImpactorContactImpulseNs{};
    [[maybe_unused]] double lastPlatenContactSamples = 0.0;
    [[maybe_unused]] double lastImpactorContactSamples = 0.0;
    [[maybe_unused]] double lastRigidContactSamples = 0.0;
    [[maybe_unused]] double lastSupportFloorContactSamples = 0.0;
    [[maybe_unused]] double lastForbiddenRigidToolContactSamples = 0.0;
    [[maybe_unused]] double lastPlatenMeanForceY = 0.0;
    [[maybe_unused]] double lastImpactorMeanForceY = 0.0;
    if (arguments.shippingBox) run.refreshShippingMemoryObservation();
    if (arguments.shippingBox && !run.shippingMemoryWithinLimit()) {
        nativeFailure = true;
        stopReason = "initialized native runtime/Metal/host memory exceeded 16 GiB; no shipping physics step was attempted";
    }
    if (arguments.shippingBox && !nativeFailure)
        writeShippingAcceptedCheckpoint(
            arguments.output, currentSnapshot, compiled.world, arguments,
            "flat-initial", 0u, numi::cardboard::kNoBoxScoreLineIndex,
            parkHeight, shippingSimulationTimeSeconds);
    const auto& exposedBoundaryNodeIndices =
        mesh.world.objects.at(0u).femContactNodes;
    const std::set<std::uint32_t> compiledExteriorNodeSet(
        exposedBoundaryNodeIndices.begin(), exposedBoundaryNodeIndices.end());
    const auto boxFeatureGrips = numi::cardboard::makeBoxFeatureGrips(
        blank, blank.points, exposedBoundaryNodeIndices);
    writeBoxGripPatchCsv(arguments.output / "box-grip-patches.csv",
                         blank, boxFeatureGrips);
    const auto tabAnchorNodes = numi::cardboard::makeBoxTabBodyAnchorNodes(
        blank, blank.points, exposedBoundaryNodeIndices);
    bool tabAnchorActive = false;
    bool tabAnchorReleasedBeforeJointFixture = false;
    bool scoreToolsWithdrawn = false;
    const double withdrawnToolHeight = blank.config.wallHeight +
        2.0 * blank.config.panelWidth + 0.050;

    const auto zeroTargets = [&] {
        return std::vector<nm_float4>(
            static_cast<std::size_t>(arguments.environmentCount) * nodeCount,
            nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
    };
#ifdef NUMI_CARDBOARD_METAL_WORLD
    double shippingToolCenterX = 0.5 * blank.perimeterSpan;
    double shippingToolCenterZ = 0.5 * blank.blankHeight;
    double shippingCompressionPlatenY = 1.0;
    double shippingFloorCenterY = -0.050;
#endif
    const auto parkedBodies = [&] {
        std::vector<MRBodyStateGPU> bodies;
        bool haveAcceptedShippingScene = false;
#ifdef NUMI_CARDBOARD_METAL_WORLD
        if (arguments.shippingBox) {
            require(tooling.sequentialFamilies &&
                        tooling.families.size() ==
                            numi::cardboard::kShippingScoreToolFamilyCount &&
                        tooling.proxies.size() ==
                            numi::cardboard::kShippingScoreToolProxyCount &&
                        scoreToolSceneBodyCount ==
                            numi::cardboard::kShippingScoreToolSceneBodyCount,
                    "shipping scoring requires exactly three reusable tool families");
            const auto& acceptedScene = run.shippingRigidSceneBodies();
            if (!acceptedScene.empty()) {
                haveAcceptedShippingScene = true;
                const std::size_t toolBodyCount = scoreToolSceneBodyCount +
                    2u;
                require(toolBodyCount + 1u ==
                            numi::cardboard::kShippingRigidBodyCount,
                        "shipping scene body slots differ from its prebuild rigid-body count");
                require(acceptedScene.size() == toolBodyCount + 1u &&
                            run.shippingRigidToolSceneBodyCount() == toolBodyCount,
                        "shipping accepted scene does not match its compiled shared-tool layout");
                bodies.assign(acceptedScene.begin(),
                              acceptedScene.begin() +
                                  static_cast<std::ptrdiff_t>(toolBodyCount));
                // Every transaction begins from accepted kinematic poses.
                // Clear prior-step velocities; only an explicit command may
                // move a tool in the next transaction.
                for (MRBodyStateGPU& body : bodies) {
                    body.linearVelocityAndInverseMass.x = 0.0f;
                    body.linearVelocityAndInverseMass.y = 0.0f;
                    body.linearVelocityAndInverseMass.z = 0.0f;
                    body.angularVelocity = {0.0f, 0.0f, 0.0f, 0.0f};
                }
            } else {
                bodies = numi::cardboard::makeBoxToolBodyStates(
                    tooling, arguments.environmentCount, parkHeight);
                auto compressionPlaten =
                    numi::cardboard::makeShippingKinematicSceneBody(
                        {shippingToolCenterX, shippingCompressionPlatenY,
                         shippingToolCenterZ});
                auto shippingFloor =
                    numi::cardboard::makeShippingKinematicSceneBody(
                        {shippingToolCenterX, shippingFloorCenterY,
                         shippingToolCenterZ},
                        numi::cardboard::shippingSupportFloorOrientationXYWZ());
                bodies.push_back(compressionPlaten);
                bodies.push_back(shippingFloor);
            }
        } else {
            bodies = numi::cardboard::makeBoxToolBodyStates(
                tooling, arguments.environmentCount, parkHeight);
        }
#else
        bodies = numi::cardboard::makeBoxToolBodyStates(
            tooling, arguments.environmentCount, parkHeight);
#endif
        if (scoreToolsWithdrawn && !haveAcceptedShippingScene) {
            for (std::uint32_t environment = 0u;
                 environment < arguments.environmentCount; ++environment) {
                const std::size_t bodyCount = scoreToolSceneBodyCount +
                    (arguments.shippingBox ? 2u : 0u);
                const std::size_t base =
                    static_cast<std::size_t>(environment) * bodyCount;
                if (tooling.sequentialFamilies) {
                    for (const auto& family : tooling.families) {
                        bodies.at(base + family.scoreBodyIndex).position.z =
                            static_cast<float>(withdrawnToolHeight);
                        bodies.at(base + family.supportBodyIndex).position.z = -2.0f;
                    }
                } else {
                    bodies.at(base + tooling.scoreBodyIndex).position.z =
                        static_cast<float>(withdrawnToolHeight);
                    bodies.at(base + tooling.supportBodyIndex).position.z = -2.0f;
                }
            }
        }
        return bodies;
    };
    const auto writeBoxGripReactionRows = [&] (
        const std::uint32_t nativeStep,
        const std::string& stage,
        const bool accepted,
        const RunState* state
    ) {
        if (!arguments.shippingBox) return;
        const double timeSeconds = boxSimulationTimeSeconds;
        for (const auto& grip : boxFeatureGrips) {
            for (std::uint32_t environment = 0u;
                 environment < arguments.environmentCount; ++environment) {
                std::vector<std::uint32_t> activeNodes;
                if (accepted && state != nullptr) {
                    require(state->snapshot.femNodes.size() >=
                                static_cast<std::size_t>(arguments.environmentCount) * nodeCount &&
                                state->reactions.size() >=
                                static_cast<std::size_t>(arguments.environmentCount) * nodeCount &&
                                state->contactReactions.size() >=
                                static_cast<std::size_t>(arguments.environmentCount) * nodeCount,
                            "accepted shipping grip reaction readback is truncated");
                    const std::size_t base =
                        static_cast<std::size_t>(environment) * nodeCount;
                    for (const std::uint32_t local : grip.nodeIndices) {
                        if (local >= nodeCount)
                            throw std::out_of_range(
                                "shipping grip node exceeds the compiled FEM mesh");
                        if (state->snapshot.femNodes[base + local].restAndFixed.w == 3.0f)
                            activeNodes.push_back(local);
                    }
                }
                numi::cardboard::ConstraintReactionWrench wrench;
                const bool reactionValid = accepted && state != nullptr &&
                    !activeNodes.empty();
                if (reactionValid) {
                    const std::size_t base =
                        static_cast<std::size_t>(environment) * nodeCount;
                    wrench = numi::cardboard::summarizeConstraintReactions(
                        state->snapshot.femNodes,
                        state->reactions,
                        state->contactReactions,
                        base, nodeCount, activeNodes);
                }
                gripReactionOutput << nativeStep << ',' << timeSeconds << ','
                    << stage << ',' << grip.assemblyActionIndex << ',';
                if (grip.scoreLineIndex ==
                    numi::cardboard::kNoBoxScoreLineIndex)
                    gripReactionOutput << -1;
                else gripReactionOutput << grip.scoreLineIndex;
                gripReactionOutput << ',' << environment << ','
                    << (accepted ? 1 : 0) << ','
                    << (reactionValid ? 1 : 0) << ','
                    << (reactionValid ? activeNodes.size() : 0u) << ',';
                if (reactionValid)
                    gripReactionOutput << wrench.pivot.x << ','
                        << wrench.pivot.y << ',' << wrench.pivot.z;
                else gripReactionOutput << ",,";
                gripReactionOutput << ',';
                if (reactionValid)
                    gripReactionOutput << wrench.mechanicalConstraintForce.x << ','
                        << wrench.mechanicalConstraintForce.y << ','
                        << wrench.mechanicalConstraintForce.z << ','
                        << wrench.contactForce.x << ','
                        << wrench.contactForce.y << ','
                        << wrench.contactForce.z << ','
                        << wrench.totalForce.x << ','
                        << wrench.totalForce.y << ','
                        << wrench.totalForce.z << ','
                        << wrench.mechanicalConstraintMoment.x << ','
                        << wrench.mechanicalConstraintMoment.y << ','
                        << wrench.mechanicalConstraintMoment.z << ','
                        << wrench.contactMoment.x << ','
                        << wrench.contactMoment.y << ','
                        << wrench.contactMoment.z << ','
                        << wrench.totalMoment.x << ','
                        << wrench.totalMoment.y << ','
                        << wrench.totalMoment.z;
                else gripReactionOutput << std::string(18u, ',');
                gripReactionOutput << '\n';
            }
        }
        gripReactionOutput.flush();
        require(gripReactionOutput.good(),
                "failed while writing accepted shipping grip reactions");
    };
#ifdef NUMI_CARDBOARD_METAL_WORLD
    if (arguments.shippingBox && !arguments.shippingPreflightOnly) {
        const auto initialBodies = parkedBodies();
        require(initialBodies.size() == scoreToolSceneBodyCount + 2u,
                "shipping tool initialization omitted a family, platen, or floor body");
        const std::vector<MRBodyStateGPU> initialToolBodies = initialBodies;
        numi::cardboard::ShippingRigidBodySpec rigidSpec;
        rigidSpec.massKg = 1.0;
        require(arguments.shippingDesign.has_value(),
                "shipping finite-mass tool requires the shipping design dimensions");
        rigidSpec.halfExtentM = arguments.shippingImpactorHalfExtentsM;
        rigidSpec.initialCenterM = {100.0, 1.0, 100.0};
        // Gravity is switched on only for the finite-mass drop branches. The
        // parked body is not a load during scoring, assembly, or quasistatic
        // platen compression.
        rigidSpec.gravityMPerS2 = {0.0, 0.0, 0.0};
        rigidSpec.timestepSeconds = run.runtime().timestepSeconds();
        run.enableShippingRigidWorld(initialToolBodies, rigidSpec);
    }
#endif
    auto attemptNativeStep = [&](const std::string& stage,
                                 const std::uint32_t actionIndex,
                                 const std::vector<nm_float4>& targets,
                                 const std::vector<MRBodyStateGPU>& bodies,
                                 const double timestepOverrideSeconds = 0.0,
                                 const bool captureRigidContactEvidence = false,
                                 const std::string& eventId = std::string{},
                                 const std::optional<double>
                                     shippingExternalForceYNewtons =
                                         std::nullopt) {
#ifndef NUMI_CARDBOARD_METAL_WORLD
        (void)captureRigidContactEvidence;
        (void)eventId;
        (void)shippingExternalForceYNewtons;
#endif
        const std::uint32_t stepIndex = nextNativeStep++;
        std::string error;
        std::optional<RunState> state;
        const bool submitsStarchActivation =
            !pendingStarchActivationCommands.empty();
        const bool submitsHmaActivation =
            !pendingHmaActivationCommands.empty();
        const std::string submittedHmaEvent = pendingHmaActivationEvent;
        std::vector<NMCohesiveActivationCommandGPU> activationCommands =
            pendingStarchActivationCommands;
        activationCommands.insert(activationCommands.end(),
            pendingHmaActivationCommands.begin(),
            pendingHmaActivationCommands.end());
        try {
            state = run.step(stepIndex, targets, bodies,
                             activationCommands, timestepOverrideSeconds,
                             captureRigidContactEvidence,
                             shippingExternalForceYNewtons);
            currentSnapshot = state->snapshot;
            lastPlatenContactImpulseNs = state->platenContactImpulseNs;
            lastImpactorContactImpulseNs = state->impactorContactImpulseNs;
            lastPlatenContactSamples = state->platenContactSamples;
            lastImpactorContactSamples = state->impactorContactSamples;
            lastRigidContactSamples = state->rigidContactSamples;
            lastSupportFloorContactSamples =
                state->supportFloorContactSamples;
            lastForbiddenRigidToolContactSamples =
                state->forbiddenRigidToolContactSamples;
            lastPlatenMeanForceY = state->platenMeanForceY;
            lastImpactorMeanForceY = state->impactorMeanForceY;
        } catch (const std::exception& caught) {
            error = caught.what();
            nativeFailure = true;
            currentSnapshot = run.runtime().snapshot();
        }
        const bool accepted = !error.empty() ? false :
            boxEnvironmentAccepted(compiled.world, currentSnapshot, 0u) &&
            (arguments.environmentCount == 1u ||
             boxEnvironmentAccepted(compiled.world, currentSnapshot, 1u));
        const double attemptedTimestep = timestepOverrideSeconds > 0.0
            ? timestepOverrideSeconds : run.runtime().timestepSeconds();
        if (accepted) {
            ++acceptedNativeSteps;
            require(std::isfinite(attemptedTimestep) && attemptedTimestep > 0.0,
                    "accepted box step has an invalid simulation duration");
            boxSimulationTimeSeconds += attemptedTimestep;
            lastAcceptedTimestepSeconds = attemptedTimestep;
            lastAcceptedNativeStep = stepIndex;
            if (arguments.shippingBox) {
                shippingSimulationTimeSeconds = boxSimulationTimeSeconds;
            }
            if (submitsStarchActivation) {
                starchInitialActivationAccepted = true;
                starchInitialActivationStep = stepIndex;
                pendingStarchActivationCommands.clear();
            }
            if (submitsHmaActivation) {
                acceptedHmaActivationEvents.insert(submittedHmaEvent);
                pendingHmaActivationCommands.clear();
                pendingHmaActivationEvent.clear();
            }
        }
        auto codeFor = [&](const std::uint32_t environment) {
            return environment < currentSnapshot.statuses.size()
                ? currentSnapshot.statuses[environment].code : NM_STATUS_INVALID_DISPATCH;
        };
        const auto& cert0 = currentSnapshot.solverCertificates.size() > 0u
            ? currentSnapshot.solverCertificates[0u] : NMSolverCertificateGPU{};
        const auto& status0 = currentSnapshot.statuses.size() > 0u
            ? currentSnapshot.statuses[0u] : NMMatterStatusGPU{};
        const auto& status1 = currentSnapshot.statuses.size() > 1u
            ? currentSnapshot.statuses[1u] : NMMatterStatusGPU{};
        stepOutput << stepIndex << ',' << stage << ',';
        if (actionIndex == numi::cardboard::kNoBoxScoreLineIndex) stepOutput << -1;
        else stepOutput << actionIndex;
        stepOutput << ',' << (accepted ? 1 : 0) << ',' << codeFor(0u) << ',';
        if (arguments.environmentCount > 1u)
            stepOutput << codeFor(1u) << ',' << status0.fgmresIterations << ','
                << status1.fgmresIterations;
        else stepOutput << "NA," << status0.fgmresIterations << ",NA";
        stepOutput << ',' << cert0.validity.w << ','
            << cert0.nonlinear.x << ',' << cert0.nonlinear.y << ','
            << cert0.nonlinear.z << ',' << cert0.nonlinear.w << ','
            << boxSimulationTimeSeconds << ',' << attemptedTimestep << ',' << '"';
        for (const char c : error) stepOutput << (c == '"' ? '\'' : c);
        stepOutput << "\"\n";

        if (arguments.shippingBox && accepted &&
            (stepIndex == 0u || stepIndex % 32u == 0u)) {
            const auto observables = shippingHistoryAreaObservables(
                compiled.world, mesh, currentSnapshot, 0u);
            writeShippingHistoryAreaRow(shippingHistoryAreaLedger, stepIndex,
                boxSimulationTimeSeconds, attemptedTimestep, stage, true,
                observables);
            shippingHistoryAreaLedger.flush();
        }

        if (currentSnapshot.femNodes.size() >=
            static_cast<std::size_t>(arguments.environmentCount) * nodeCount) {
            for (std::uint32_t environment = 0u;
                 environment < arguments.environmentCount; ++environment) {
                const auto angles = boxScoreAngles(
                    mesh, activeActionStartNodes, currentSnapshot.femNodes,
                    environment, nodeCount);
                for (std::uint32_t scoreIndex = 0u; scoreIndex < angles.size(); ++scoreIndex) {
                    const auto& angle = angles[scoreIndex];
                    angleOutput << stepIndex << ',' << stage << ',';
                    if (actionIndex == numi::cardboard::kNoBoxScoreLineIndex)
                        angleOutput << -1;
                    else angleOutput << actionIndex;
                    angleOutput << ',' << environment << ',' << (accepted ? 1 : 0)
                        << ',' << scoreIndex << ','
                        << static_cast<unsigned>(blank.scoreLines[scoreIndex].kind)
                        << ',' << (angle.valid ? 1 : 0) << ',';
                    if (angle.valid)
                        angleOutput << angle.foldTurnDegrees << ','
                                    << angle.interiorAngleDegrees;
                    else angleOutput << ",";
                    angleOutput << ',' << angle.targetFoldDegrees << ',';
                    if (std::isfinite(angle.relativeTargetFoldDegrees))
                        angleOutput << angle.relativeTargetFoldDegrees;
                    angleOutput << ',' << angle.sampleCount << '\n';
                }
            }
            if (nextNativeStep == 1u || !accepted) {
                writeObj(arguments.output / ("box-state-step-" + stepLabel(stepIndex + 1u) + ".obj"),
                         compiled.world, currentSnapshot.femNodes, 0u);
            }
        }
        writeBoxGripReactionRows(
            stepIndex, stage, accepted, state ? &*state : nullptr);
        stepOutput.flush();
        angleOutput.flush();
        if (arguments.shippingBox) {
            run.writeMemoryObservation(arguments.output / "memory-observations.csv",
                                       stepIndex, stage);
            if (!run.shippingMemoryWithinLimit()) {
                nativeFailure = true;
                stopReason = "measured shipping runtime/Metal/host memory exceeded 16 GiB after this step";
            }
        }
        if (!error.empty()) stopReason = "native exception: " + error;
        else if (!accepted)
            stopReason = "Matter rejected native step " + std::to_string(stepIndex);
#ifdef NUMI_CARDBOARD_METAL_WORLD
        if (arguments.shippingBox && captureRigidContactEvidence) {
            const double dt = timestepOverrideSeconds > 0.0
                ? timestepOverrideSeconds : run.runtime().timestepSeconds();
            const auto& sceneBodies = run.shippingRigidSceneBodies();
            shippingContactLedger << stepIndex << ',' << stage << ','
                << eventId << ',' << (accepted ? 1 : 0) << ',' << dt << ','
                << lastRigidContactSamples << ','
                << lastPlatenContactSamples << ',' << lastImpactorContactSamples << ','
                << lastPlatenContactImpulseNs[1] << ',' << lastPlatenMeanForceY << ','
                << lastImpactorContactImpulseNs[1] << ',' << lastImpactorMeanForceY << ',';
            if (sceneBodies.size() > shippingFreeBodyIndex)
                shippingContactLedger
                    << sceneBodies[shippingFreeBodyIndex].position.y << ','
                    << sceneBodies[shippingFreeBodyIndex].linearVelocityAndInverseMass.y;
            else shippingContactLedger << ',';
            shippingContactLedger << '\n';
            shippingContactLedger.flush();
        }
#endif
        return accepted && !nativeFailure;
    };

    const auto activateHmaPatchGroups = [&](const std::vector<std::string>& patchIds,
                                             const std::string& eventName,
                                             const std::uint32_t actionIndex) {
        require(arguments.shippingHmaLaw.has_value() &&
                    !mesh.hmaClosureMesh.patches.empty(),
                "shipping HMA activation has no explicit compiled law/patch table");
        require(boxEnvironmentAccepted(compiled.world, currentSnapshot, 0u),
                "HMA activation requires an already accepted endpoint");
        std::vector<NMCohesiveActivationCommandGPU> commands;
        struct GapRow {
            std::string patchId;
            std::uint32_t stableIdentifier = 0u;
            double gapM = 0.0;
            double toleranceM = 0.0;
        };
        std::vector<GapRow> rows;
        std::set<std::uint32_t> uniqueIds;
        for (const std::string& patchId : patchIds) {
            const auto group = std::find_if(
                mesh.hmaClosureMesh.groups.begin(),
                mesh.hmaClosureMesh.groups.end(),
                [&](const auto& candidate) {
                    return candidate.patchId == patchId;
                });
            require(group != mesh.hmaClosureMesh.groups.end(),
                    "shipping HMA activation names an absent authored patch group: " +
                        patchId);
            for (const std::uint32_t stableId : group->stableIdentifiers) {
                if (acceptedHmaActivationEvents.contains(eventName)) continue;
                require(uniqueIds.insert(stableId).second,
                        "shipping HMA activation repeats a stable patch ID");
                const auto patch = std::find_if(
                    mesh.hmaClosureMesh.patches.begin(),
                    mesh.hmaClosureMesh.patches.end(),
                    [stableId](const auto& candidate) {
                        return candidate.stableIdentifier == stableId;
                    });
                require(patch != mesh.hmaClosureMesh.patches.end(),
                        "shipping HMA activation cannot resolve a mortar patch");
                const double gapM =
                    numi::cardboard::shippingMortarAcceptedMaximumGapM(
                        *patch, currentSnapshot.femNodes, 0u, nodeCount);
                const double toleranceM =
                    arguments.shippingHmaLaw->activationGapToleranceM;
                rows.push_back({patchId, stableId, gapM, toleranceM});
                if (!std::isfinite(gapM) || gapM > toleranceM) {
                    for (const auto& row : rows)
                        adhesiveActivationOutput << eventName << ','
                            << row.patchId << ',' << row.stableIdentifier << ','
                            << row.gapM << ',' << row.toleranceM << ",0,0\n";
                    adhesiveActivationOutput.flush();
                    stopReason = "accepted endpoint missed the explicit HMA activation gap for " +
                        patchId + ": measured " + std::to_string(gapM) +
                        " m, allowed " + std::to_string(toleranceM) + " m";
                    return false;
                }
                NMCohesiveActivationCommandGPU command{};
                command.identity.x = 0u;
                command.identity.y = 0u;
                command.identity.z = stableId;
                command.identity.w = 0u;
                commands.push_back(command);
            }
        }
        if (commands.empty()) {
            adhesiveActivationOutput.flush();
            return true;
        }
        pendingHmaActivationCommands = std::move(commands);
        pendingHmaActivationEvent = eventName;
        const bool accepted = attemptNativeStep(
            "hma_activate_" + eventName, actionIndex,
            zeroTargets(), parkedBodies());
        for (const auto& row : rows)
            adhesiveActivationOutput << eventName << ',' << row.patchId << ','
                << row.stableIdentifier << ',' << row.gapM << ','
                << row.toleranceM << ",1," << (accepted ? 1 : 0) << '\n';
        adhesiveActivationOutput.flush();
        return accepted;
    };

    if (!nativeFailure && !mesh.starchActivationCommands.empty()) {
        // First accept the assembled reference geometry without a load or
        // activation request. The next preDynamics transaction applies the
        // activation command against that already accepted endpoint.
        if (!attemptNativeStep("initial_zero_displacement_accept",
                numi::cardboard::kNoBoxScoreLineIndex,
                zeroTargets(), parkedBodies())) {
            nativeFailure = true;
            if (stopReason.empty())
                stopReason = "Matter rejected the zero-displacement starch setup endpoint";
        }
        if (!nativeFailure) {
            pendingStarchActivationCommands = mesh.starchActivationCommands;
            if (!attemptNativeStep("initial_starch_activation",
                    numi::cardboard::kNoBoxScoreLineIndex,
                    zeroTargets(), parkedBodies())) {
                nativeFailure = true;
                if (stopReason.empty())
                    stopReason = "Matter rejected the transactional initial starch activation endpoint";
            }
        }
        if (starchInitialActivationAccepted && arguments.shippingBox) {
            writeShippingAcceptedCheckpoint(
                arguments.output, currentSnapshot, compiled.world, arguments,
                "starch-activated-initial", nextNativeStep,
                numi::cardboard::kNoBoxScoreLineIndex, parkHeight,
                shippingSimulationTimeSeconds);
        }
    }

    // Hold the manufacturer's tab body in its authored accepted pose while
    // the wall turns are measured and folded. Only exposed tetrahedral
    // boundary nodes in the tab body are clamped; no interior coordinates are
    // written. The tab's free edge is part of this clamp, so fixture mode
    // releases it only after all four wall boundary clamps have been retained.
    {
        auto anchorTargets = zeroTargets();
        for (const std::uint32_t node : tabAnchorNodes) {
            const auto& accepted = initialNodes.at(node);
            require(accepted.restAndFixed.w == 0.0f,
                    "authored tab anchor node is not initially free");
            numi::cardboard::setBoxGripTarget(
                anchorTargets, nodeCount, 0u, node,
                numi::cardboard::boxAcceptedPosition(accepted), 3.0f);
        }
        if (attemptNativeStep("tab_body_anchor_acquire",
                numi::cardboard::kNoBoxScoreLineIndex,
                anchorTargets, parkedBodies())) {
            tabAnchorActive = true;
        } else {
            nativeFailure = true;
        }
    }

    // The shipping tool pack contains one punch/rail family per distinct
    // fixed local orientation and span. Each authored line is scored in its
    // own accepted transaction sequence by translating the selected family;
    // no tool is rebound to a different proxy index.
    double dieHeight = parkHeight;
    std::uint32_t activeScoreBodyIndex = tooling.scoreBodyIndex;
    std::uint32_t activeSupportBodyIndex = tooling.supportBodyIndex;
    numi::cardboard::BoxVec3 activeScoreCenter = tooling.scoreBodyPosition;
    const std::uint32_t bodiesPerEnvironment = scoreToolSceneBodyCount +
        (arguments.shippingBox ? 2u : 0u);
    const auto moveShippingToolBody = [&] (
        const std::uint32_t bodyIndex,
        const numi::cardboard::BoxVec3 destination,
        const std::uint32_t scoreLineIndex,
        const std::string& stage,
        const double requestedSpeedMPerS = 0.0,
        const double requestedMaximumTimestepSeconds = 0.0,
        numi::cardboard::ShippingConstantSpeedMoveSummary* timingSummary = nullptr
    ) {
#ifdef NUMI_CARDBOARD_METAL_WORLD
        require(arguments.shippingBox &&
                    bodyIndex < scoreToolSceneBodyCount + 2u,
                "shipping score relocation selected an invalid compiled tool body");
        const double maximumDt = requestedMaximumTimestepSeconds > 0.0
            ? requestedMaximumTimestepSeconds
            : run.runtime().timestepSeconds();
        const double speed = requestedSpeedMPerS > 0.0
            ? requestedSpeedMPerS : arguments.shippingScoreToolSpeedMPerS;
        require(std::isfinite(maximumDt) && maximumDt > 0.0 &&
                    std::isfinite(speed) && speed > 0.0,
                "shipping tool translation requires a finite positive rate and timestep");
        auto initialScene = run.shippingRigidSceneBodies();
        require(initialScene.size() == bodiesPerEnvironment + 1u,
                "shipping score relocation requires an accepted scene checkpoint");
        const auto initial = initialScene.at(bodyIndex).position;
        const numi::cardboard::BoxVec3 start{
            initial.x, initial.y, initial.z};
        const numi::cardboard::BoxVec3 displacement = destination - start;
        const double distance = numi::cardboard::boxLength(displacement);
        if (distance <= 4.0 * std::numeric_limits<float>::epsilon() *
                            std::max(1.0, numi::cardboard::boxLength(destination))) {
            if (timingSummary != nullptr) *timingSummary = {};
            return true;
        }
        const auto schedule = numi::cardboard::makeShippingConstantSpeedMoveSchedule(
            distance, speed, maximumDt,
            arguments.boxMaximumStepsPerAction);
        if (timingSummary != nullptr) {
            *timingSummary = {
                static_cast<std::uint32_t>(schedule.substeps.size()),
                schedule.totalDurationSeconds,
                schedule.minimumUsedTimestepSeconds,
                schedule.maximumUsedTimestepSeconds};
        }
        for (const auto& substep : schedule.substeps) {
            const auto& acceptedScene = run.shippingRigidSceneBodies();
            require(acceptedScene.size() == bodiesPerEnvironment + 1u,
                    "shipping score relocation lost its accepted scene");
            const auto& acceptedBody = acceptedScene.at(bodyIndex);
            const double fraction = substep.completedFraction;
            const numi::cardboard::BoxVec3 endpoint{
                start.x + fraction * displacement.x,
                start.y + fraction * displacement.y,
                start.z + fraction * displacement.z};
            const double stepDt = substep.timestepSeconds;
            auto bodies = parkedBodies();
            numi::cardboard::setBoxToolBodyCommand(
                bodies, bodiesPerEnvironment, 0u, bodyIndex,
                {acceptedBody.position.x, acceptedBody.position.y,
                 acceptedBody.position.z},
                {(endpoint.x - acceptedBody.position.x) / stepDt,
                 (endpoint.y - acceptedBody.position.y) / stepDt,
                 (endpoint.z - acceptedBody.position.z) / stepDt});
            if (!attemptNativeStep(stage, scoreLineIndex,
                    zeroTargets(), bodies, stepDt))
                return false;
        }
        return true;
#else
        (void)bodyIndex;
        (void)destination;
        (void)scoreLineIndex;
        (void)stage;
        (void)requestedSpeedMPerS;
        (void)requestedMaximumTimestepSeconds;
        (void)timingSummary;
        return false;
#endif
    };
    const auto performScoreLeg = [&](const double destination,
                                     const std::string& stage,
                                     const std::uint32_t scoreLineIndex) {
        const double startHeight = dieHeight;
        const double travelM = std::abs(destination - startHeight);
        std::uint32_t legSteps = arguments.boxScoreStepsPerLeg;
        if (arguments.shippingBox) {
            legSteps = numi::cardboard::makeShippingScoreLegSchedule(
                travelM, arguments.shippingScoreToolSpeedMPerS,
                run.runtime().timestepSeconds(),
                arguments.boxMaximumStepsPerAction).stepCount;
        }
        const bool scoreApproach = stage.find("approach") != std::string::npos;
        const double effectiveSpeedMPerS = travelM /
            (static_cast<double>(legSteps) * run.runtime().timestepSeconds());
        for (std::uint32_t legStep = 1u;
             legStep <= legSteps; ++legStep) {
            const double endHeight = startHeight +
                (destination - startHeight) *
                static_cast<double>(legStep) / legSteps;
            auto bodies = parkedBodies();
            const double velocity = (endHeight - dieHeight) /
                run.runtime().timestepSeconds();
            numi::cardboard::setBoxToolBodyCommand(
                bodies, bodiesPerEnvironment, 0u, activeScoreBodyIndex,
                {activeScoreCenter.x, activeScoreCenter.y, dieHeight},
                {0.0, 0.0, velocity});
            if (!attemptNativeStep(stage, scoreLineIndex,
                                   zeroTargets(), bodies)) return false;
            dieHeight = endHeight;
            if (arguments.shippingBox) {
                if (scoreApproach) ++shippingScoreApproachAcceptedSteps;
                else ++shippingScoreRetractAcceptedSteps;
                shippingScoreEffectiveSpeedMinMPerS = std::min(
                    shippingScoreEffectiveSpeedMinMPerS, effectiveSpeedMPerS);
                shippingScoreEffectiveSpeedMaxMPerS = std::max(
                    shippingScoreEffectiveSpeedMaxMPerS, effectiveSpeedMPerS);
            }
        }
        return true;
    };
    if (!nativeFailure && arguments.shippingBox) {
        const double selectedDepthM =
            arguments.shippingScoreDepthRatio * blank.boardCaliper;
        const double targetHeight = blank.boardCaliper +
            tooling.punchRadius - selectedDepthM;
        for (std::uint32_t scoreLineIndex = 0u;
             scoreLineIndex < blank.scoreLines.size() && !nativeFailure;
             ++scoreLineIndex) {
            const std::uint32_t familyIndex =
                numi::cardboard::boxSequentialFamilyForScoreLine(
                    tooling, scoreLineIndex);
            const auto& family = tooling.families.at(familyIndex);
            const auto center = numi::cardboard::boxScoreLineCenter(
                blank, scoreLineIndex);
            activeScoreBodyIndex = family.scoreBodyIndex;
            activeSupportBodyIndex = family.supportBodyIndex;
            activeScoreCenter = center;
            dieHeight = parkHeight;
            if (!moveShippingToolBody(activeScoreBodyIndex,
                    {center.x, center.y, parkHeight}, scoreLineIndex,
                    "score_line_punch_reposition")) {
                nativeFailure = true;
                break;
            }
            // The paired rails stay below the substrate while traversing the
            // blank, then rise only at the selected crease to establish its
            // local channel support.
            if (!moveShippingToolBody(activeSupportBodyIndex,
                    {center.x, center.y, -2.0}, scoreLineIndex,
                    "score_line_support_reposition")) {
                nativeFailure = true;
                break;
            }
            if (!moveShippingToolBody(activeSupportBodyIndex,
                    {center.x, center.y, 0.0}, scoreLineIndex,
                    "score_line_support_engage")) {
                nativeFailure = true;
                break;
            }
            if (!performScoreLeg(targetHeight,
                    "score_die_selected_depth_approach", scoreLineIndex) ||
                !performScoreLeg(parkHeight,
                    "score_die_selected_depth_retract", scoreLineIndex)) {
                nativeFailure = true;
                break;
            }
            dieHeight = parkHeight;
            if (!moveShippingToolBody(activeSupportBodyIndex,
                    {center.x, center.y, -2.0}, scoreLineIndex,
                    "score_line_support_withdraw")) {
                nativeFailure = true;
                break;
            }
        }
    } else if (!nativeFailure) {
        if (!performScoreLeg(contactHeight,
                                    "score_die_contact_approach",
                                    numi::cardboard::kNoBoxScoreLineIndex)) {
            nativeFailure = true;
        } else if (!performScoreLeg(parkHeight,
                                    "score_die_contact_retract",
                                    numi::cardboard::kNoBoxScoreLineIndex)) {
            nativeFailure = true;
        }
    }
    if (arguments.shippingBox && !nativeFailure)
        writeShippingAcceptedCheckpoint(
            arguments.output, currentSnapshot, compiled.world, arguments,
            "score-pass", nextNativeStep,
            numi::cardboard::kNoBoxScoreLineIndex, dieHeight,
            shippingSimulationTimeSeconds);
    if (arguments.shippingBox && !nativeFailure) {
        for (const auto& family : tooling.families) {
            if (!moveShippingToolBody(family.scoreBodyIndex,
                    {family.referenceCenter.x, family.referenceCenter.y,
                     withdrawnToolHeight},
                    numi::cardboard::kNoBoxScoreLineIndex,
                    "accepted_score_tool_withdrawal")) {
                nativeFailure = true;
                if (stopReason.empty())
                    stopReason = "native runtime rejected score-tool withdrawal";
                break;
            }
        }
        if (!nativeFailure) {
            dieHeight = withdrawnToolHeight;
            scoreToolsWithdrawn = true;
            writeShippingAcceptedCheckpoint(
                arguments.output, currentSnapshot, compiled.world, arguments,
                "score-tools-withdrawn", nextNativeStep,
                numi::cardboard::kNoBoxScoreLineIndex, dieHeight,
                shippingSimulationTimeSeconds);
        }
    }

    bool blockedByBondLaw = false;
    const auto makeFixtureJointTargets = [&] {
        using numi::cardboard::BoxVec3;
        const auto tab = numi::cardboard::makeBoxManufacturerJointEdgeNodes(
            blank, blank.points, true);
        const auto seam = numi::cardboard::makeBoxManufacturerJointEdgeNodes(
            blank, blank.points, false);
        require(!tab.empty() && tab.size() == seam.size(),
                "fixture seam edges have different or empty boundary-node counts");
        require(std::all_of(tab.begin(), tab.end(), [&](const std::uint32_t node) {
                    return compiledExteriorNodeSet.contains(node);
                }) &&
                std::all_of(seam.begin(), seam.end(), [&](const std::uint32_t node) {
                    return compiledExteriorNodeSet.contains(node);
                }),
                "fixture seam action selected a node outside the compiled exterior boundary map");
        const double tolerance = 2.0e-10 *
            std::max({blank.perimeterSpan, blank.blankHeight, blank.boardCaliper});
        const auto byRestYZ = [&](const std::uint32_t left,
                                  const std::uint32_t right) {
            const auto& a = blank.points.at(left);
            const auto& b = blank.points.at(right);
            if (a.y != b.y) return a.y < b.y;
            if (a.z != b.z) return a.z < b.z;
            return left < right;
        };
        auto orderedTab = tab;
        auto orderedSeam = seam;
        std::sort(orderedTab.begin(), orderedTab.end(), byRestYZ);
        std::sort(orderedSeam.begin(), orderedSeam.end(), byRestYZ);

        BoxVec3 seamCentroid{};
        for (const std::uint32_t node : orderedSeam) {
            const BoxVec3 p = numi::cardboard::boxAcceptedPosition(
                currentSnapshot.femNodes.at(node));
            seamCentroid = seamCentroid + p;
        }
        seamCentroid = (1.0 / orderedSeam.size()) * seamCentroid;
        BoxVec3 wallCentroid{};
        std::uint32_t wallCount = 0u;
        for (std::uint32_t node = 0u; node < nodeCount; ++node) {
            const auto& rest = blank.points[node];
            if (rest.x <= blank.config.manufacturerJointWidth + tolerance ||
                rest.x >= blank.perimeterSpan - tolerance ||
                rest.y <= blank.bottomScoreY + tolerance ||
                rest.y >= blank.topScoreY - tolerance)
                continue;
            wallCentroid = wallCentroid +
                numi::cardboard::boxAcceptedPosition(
                    currentSnapshot.femNodes.at(node));
            ++wallCount;
        }
        require(wallCount > 0u,
                "fixture seam cannot derive a center from accepted wall nodes");
        wallCentroid = (1.0 / wallCount) * wallCentroid;
        BoxVec3 outward = seamCentroid - wallCentroid;
        outward.y = 0.0;
        outward = numi::cardboard::boxNormalized(outward);

        std::vector<BoxFixtureJointPair> pairs;
        pairs.reserve(orderedTab.size());
        for (std::size_t index = 0u; index < orderedTab.size(); ++index) {
            const auto& tabRest = blank.points.at(orderedTab[index]);
            const auto& seamRest = blank.points.at(orderedSeam[index]);
            require(std::abs(tabRest.y - seamRest.y) <= tolerance &&
                        std::abs(tabRest.z - seamRest.z) <= tolerance,
                    "fixture seam edge nodes do not have matching authored y/z coordinates");
            const BoxVec3 seamPosition = numi::cardboard::boxAcceptedPosition(
                currentSnapshot.femNodes.at(orderedSeam[index]));
            const BoxVec3 tabStart = numi::cardboard::boxAcceptedPosition(
                currentSnapshot.femNodes.at(orderedTab[index]));
            const BoxVec3 tabTarget = seamPosition +
                arguments.boxFixtureGapM * outward;
            pairs.push_back({orderedTab[index], orderedSeam[index],
                             tabStart, tabTarget, seamPosition});
        }
        return pairs;
    };
    const auto makeFixtureEndTargets = [&](const bool topEnd) {
        auto targets = zeroTargets();
        std::set<std::uint32_t> clampedNodes;
        for (const auto& grip : boxFeatureGrips) {
            if (grip.scoreLineIndex >= blank.scoreLines.size()) continue;
            const auto kind = blank.scoreLines[grip.scoreLineIndex].kind;
            if ((kind == numi::cardboard::BoxScoreKind::topFlap) != topEnd)
                continue;
            for (const std::uint32_t node : grip.nodeIndices)
                clampedNodes.insert(node);
        }
        require(!clampedNodes.empty(),
                "fixture end closure has no authored flap free-edge nodes");
        for (const std::uint32_t node : clampedNodes) {
            const auto& accepted = currentSnapshot.femNodes.at(node);
            require(accepted.restAndFixed.w == 0.0f ||
                        accepted.restAndFixed.w == 3.0f,
                    "fixture end edge contains a nonfree authored constraint");
            numi::cardboard::setBoxGripTarget(
                targets, nodeCount, 0u, node,
                numi::cardboard::boxAcceptedPosition(accepted), 3.0f);
        }
        return std::pair<std::vector<nm_float4>, std::size_t>{
            std::move(targets), clampedNodes.size()};
    };
    const std::uint32_t shippingActionLimit = !arguments.shippingBox ?
        static_cast<std::uint32_t>(blank.assembly.actions.size()) :
        arguments.shippingRunThroughStage == "scoring" ? 0u :
        arguments.shippingRunThroughStage == "walls" ? 4u :
        arguments.shippingRunThroughStage == "manufacturer-lap" ? 5u :
        static_cast<std::uint32_t>(blank.assembly.actions.size());
    if (!nativeFailure) {
        for (std::uint32_t actionIndex = 0u;
             actionIndex < shippingActionLimit; ++actionIndex) {
            const auto& action = blank.assembly.actions[actionIndex];
            auto& row = actionRows[actionIndex];
            if (arguments.boxFixtureJoint &&
                action.kind == numi::cardboard::BoxAssemblyAction::Kind::bondManufacturerJoint &&
                tabAnchorActive) {
                // Wall grips are persistent external supports. Confirm each
                // distal edge remains clamped before releasing the tab's free
                // edge, which the seam fixture will move in the next stage.
                for (std::uint32_t wallAction = 0u; wallAction < 4u; ++wallAction) {
                    const auto grip = std::find_if(boxFeatureGrips.begin(), boxFeatureGrips.end(),
                        [wallAction](const auto& candidate) {
                            return candidate.assemblyActionIndex == wallAction;
                        });
                    require(grip != boxFeatureGrips.end() && !grip->nodeIndices.empty(),
                            "retained wall fixture grip is missing");
                    for (const std::uint32_t node : grip->nodeIndices)
                        require(currentSnapshot.femNodes.at(node)
                                    .restAndFixed.w == 3.0f,
                                "completed wall boundary grip was not retained");
                }

                auto releaseTargets = zeroTargets();
                for (const std::uint32_t node : tabAnchorNodes) {
                    const auto& accepted = currentSnapshot.femNodes.at(node);
                    require(accepted.restAndFixed.w == 3.0f,
                            "tab body anchor was not active before fixture release");
                    numi::cardboard::setBoxGripTarget(
                        releaseTargets, nodeCount, 0u, node,
                        numi::cardboard::boxAcceptedPosition(accepted), 2.0f);
                }
                const std::uint32_t nativeStep = nextNativeStep;
                if (!attemptNativeStep("tab_body_anchor_release_before_joint_fixture",
                        actionIndex, releaseTargets, parkedBodies())) {
                    row.status = "native_rejected";
                    row.diagnostic = stopReason;
                    nativeFailure = true;
                    break;
                }
                row.firstNativeStep = nativeStep;
                row.lastNativeStep = nativeStep;
                ++row.acceptedNativeSteps;
                for (const std::uint32_t node : tabAnchorNodes)
                    require(currentSnapshot.femNodes.at(node)
                                .restAndFixed.w == 0.0f,
                            "released tab anchor did not return to free state");
                for (std::uint32_t wallAction = 0u; wallAction < 4u; ++wallAction) {
                    const auto grip = std::find_if(boxFeatureGrips.begin(), boxFeatureGrips.end(),
                        [wallAction](const auto& candidate) {
                            return candidate.assemblyActionIndex == wallAction;
                        });
                    require(grip != boxFeatureGrips.end() &&
                                !grip->nodeIndices.empty(),
                            "retained wall fixture grip is missing");
                    for (const std::uint32_t node : grip->nodeIndices)
                        require(currentSnapshot.femNodes.at(node)
                                    .restAndFixed.w == 3.0f,
                                "tab anchor release disturbed a retained wall clamp");
                }
                tabAnchorActive = false;
                tabAnchorReleasedBeforeJointFixture = true;
            }
            if (nativeFailure) break;
            activeActionStartNodes = currentSnapshot.femNodes;
            const bool manufacturerBondAction =
                action.kind == numi::cardboard::BoxAssemblyAction::Kind::bondManufacturerJoint;
            const bool endBondAction =
                action.kind == numi::cardboard::BoxAssemblyAction::Kind::closeEndWithTapeOrAdhesive;
            if (arguments.shippingBox && arguments.shippingHmaLaw &&
                (manufacturerBondAction || endBondAction)) {
                row.firstNativeStep = nextNativeStep;
                if (manufacturerBondAction) {
                    require(tabAnchorActive,
                            "manufacturer tab fold lost its external boundary clamp");
                    const auto tabStart = currentSnapshot.femNodes;
                    std::vector<std::array<double,3>> foldedTargets;
                    foldedTargets.reserve(tabAnchorNodes.size());
                    for (const std::uint32_t node : tabAnchorNodes) {
                        require(tabStart.at(node).restAndFixed.w == 3.0f,
                                "manufacturer tab boundary node is not clamped for folding");
                        foldedTargets.push_back(
                            numi::cardboard::shippingClosureTabFoldTarget(
                                blank, blank.points.at(node)));
                    }
                    const std::uint32_t foldSteps =
                        std::max(1u, arguments.boxFixtureSteps);
                    bool foldAccepted = true;
                    for (std::uint32_t foldStep = 1u;
                         foldStep <= foldSteps; ++foldStep) {
                        const double fraction =
                            static_cast<double>(foldStep) / foldSteps;
                        auto targets = zeroTargets();
                        for (std::size_t index = 0u;
                             index < tabAnchorNodes.size(); ++index) {
                            const std::uint32_t node = tabAnchorNodes[index];
                            const auto start = numi::cardboard::boxAcceptedPosition(
                                tabStart.at(node));
                            const auto& end = foldedTargets[index];
                            const numi::cardboard::BoxVec3 target{
                                start.x + fraction * (end[0] - start.x),
                                start.y + fraction * (end[1] - start.y),
                                start.z + fraction * (end[2] - start.z)};
                            numi::cardboard::setBoxGripTarget(
                                targets, nodeCount, 0u, node, target, 3.0f);
                        }
                        const std::uint32_t nativeStep = nextNativeStep;
                        if (!attemptNativeStep("manufacturer_tab_boundary_fold",
                                actionIndex, targets, parkedBodies())) {
                            row.status = "native_rejected";
                            row.diagnostic = stopReason;
                            nativeFailure = true;
                            foldAccepted = false;
                            break;
                        }
                        row.lastNativeStep = nativeStep;
                        ++row.acceptedNativeSteps;
                    }
                    if (!foldAccepted) break;
                    if (!activateHmaPatchGroups(
                            {"manufacturer-hma-lap"},
                            "accepted_manufacturer_lap_overlap", actionIndex)) {
                        row.status = "hma_activation_gap_rejected";
                        row.diagnostic = stopReason;
                        nativeFailure = true;
                        break;
                    }
                    row.lastNativeStep = nextNativeStep - 1u;
                    ++row.acceptedNativeSteps; // accepted transactional activation
                    row.status = "hma_lap_activated_unvalidated";
                } else {
                    const bool topEnd = action.feature.index == 1u;
                    const std::string prefix = topEnd ? "top-" : "bottom-";
                    std::vector<std::string> patchIds;
                    for (const auto& group : mesh.hmaClosureMesh.groups)
                        if (group.patchId.starts_with(prefix))
                            patchIds.push_back(group.patchId);
                    require(patchIds.size() == 4u,
                            "accepted flap closure does not have four authored HMA overlap groups");
                    const std::string event = topEnd
                        ? "accepted_top_flap_overlaps"
                        : "accepted_bottom_flap_overlaps";
                    if (!activateHmaPatchGroups(patchIds, event, actionIndex)) {
                        row.status = "hma_activation_gap_rejected";
                        row.diagnostic = stopReason;
                        nativeFailure = true;
                        break;
                    }
                    row.firstNativeStep = nextNativeStep - 1u;
                    row.lastNativeStep = nextNativeStep - 1u;
                    row.acceptedNativeSteps = 1u;
                    row.status = "flap_hma_activated_unvalidated";
                }
                row.diagnostic = "named accepted-overlap activation using explicit unvalidated HMA inputs; source_class=" +
                    arguments.shippingHmaLaw->sourceClass + "; citation=" +
                    arguments.shippingHmaLaw->sourceCitation +
                    "; transfer_uncertainty_fraction=" +
                    std::to_string(arguments.shippingHmaLaw->transferUncertaintyFraction) +
                    "; this is not a calibrated corrugated-board closure";
                writeBoxActionCsv(arguments.output / "box-actions.csv",
                                  blank, actionRows);
                writeObj(arguments.output /
                             ("box-action-" + stepLabel(actionIndex + 1u) +
                              "-accepted.obj"),
                         compiled.world, currentSnapshot.femNodes, 0u);
                writeShippingAcceptedCheckpoint(
                    arguments.output, currentSnapshot, compiled.world, arguments,
                    manufacturerBondAction ? "manufacturer-lap-activated" :
                        (action.feature.index == 1u
                            ? "top-flap-overlaps-activated"
                            : "bottom-flap-overlaps-activated"),
                    nextNativeStep, actionIndex, dieHeight,
                    shippingSimulationTimeSeconds);
                continue;
            }
            if (action.kind == numi::cardboard::BoxAssemblyAction::Kind::bondManufacturerJoint ||
                action.kind == numi::cardboard::BoxAssemblyAction::Kind::closeEndWithTapeOrAdhesive) {
                if (arguments.boxFixtureJoint) {
                    const bool manufacturerJoint =
                        action.kind == numi::cardboard::BoxAssemblyAction::Kind::bondManufacturerJoint;
                    const bool topEnd = action.feature.index == 1u;
                    std::size_t fixtureNodeCount = 0u;
                    if (manufacturerJoint) {
                        const auto pairs = makeFixtureJointTargets();
                        fixtureNodeCount = 2u * pairs.size();
                        if (row.firstNativeStep == NM_INVALID_INDEX)
                            row.firstNativeStep = nextNativeStep;
                        for (std::uint32_t fixtureStep = 0u;
                             fixtureStep < arguments.boxFixtureSteps;
                             ++fixtureStep) {
                            const double fraction =
                                static_cast<double>(fixtureStep + 1u) /
                                arguments.boxFixtureSteps;
                            auto targets = zeroTargets();
                            for (const auto& pair : pairs) {
                                const auto tabTarget = pair.tabStart + fraction *
                                    (pair.tabEnd - pair.tabStart);
                                numi::cardboard::setBoxGripTarget(
                                    targets, nodeCount, 0u, pair.seamNode,
                                    pair.seamEndpoint, 3.0f);
                                numi::cardboard::setBoxGripTarget(
                                    targets, nodeCount, 0u, pair.tabNode,
                                    tabTarget, 3.0f);
                            }
                            const std::uint32_t nativeStep = nextNativeStep;
                            const bool accepted = attemptNativeStep(
                                "external_joint_fixture_move", actionIndex,
                                targets, parkedBodies());
                            row.lastNativeStep = nativeStep;
                            if (!accepted) {
                                row.status = "native_rejected";
                                row.diagnostic = stopReason;
                                nativeFailure = true;
                                break;
                            }
                            ++row.acceptedNativeSteps;
                        }
                    } else {
                        auto fixture = makeFixtureEndTargets(topEnd);
                        fixtureNodeCount = fixture.second;
                        const std::uint32_t nativeStep = nextNativeStep;
                        const bool accepted = attemptNativeStep(
                            topEnd ? "external_top_clamp_acquire"
                                   : "external_bottom_clamp_acquire",
                            actionIndex, fixture.first, parkedBodies());
                        if (row.firstNativeStep == NM_INVALID_INDEX)
                            row.firstNativeStep = nativeStep;
                        row.lastNativeStep = nativeStep;
                        if (!accepted) {
                            row.status = "native_rejected";
                            row.diagnostic = stopReason;
                            nativeFailure = true;
                        } else {
                            row.acceptedNativeSteps = 1u;
                        }
                    }
                    if (nativeFailure) break;
                    row.status = manufacturerJoint
                        ? "externally_held_unbonded"
                        : "externally_clamped_unsealed";
                    const auto gaps = numi::cardboard::measureBoxClosureGapDiagnostics(
                        blank, blank.points, currentSnapshot.femNodes, 0u,
                        nodeCount, topEnd);
                    std::ostringstream diagnostic;
                    diagnostic << std::setprecision(9)
                        << "external boundary fixture retained " << fixtureNodeCount
                        << " authored edge nodes; no adhesive/cohesive traction-separation law; "
                        << "physical_assembly_without_fixtures=false, sealed=false, "
                        << "adhesive_calibrated=false; nearest-node diagnostics only: "
                        << "manufacturer_joint_m="
                        << (gaps.manufacturerJoint.valid ? gaps.manufacturerJoint.distance : -1.0)
                        << ", short_flap_overlap_m="
                        << (gaps.shortFlapOverlap.valid ? gaps.shortFlapOverlap.distance : -1.0)
                        << ", long_flap_overlap_m="
                        << (gaps.longFlapOverlap.valid ? gaps.longFlapOverlap.distance : -1.0);
                    if (manufacturerJoint)
                        diagnostic << ", prescribed_fixture_standoff_m="
                                   << arguments.boxFixtureGapM
                                   << ", fixture_native_substeps="
                                   << arguments.boxFixtureSteps;
                    row.diagnostic = diagnostic.str();
                    writeBoxActionCsv(arguments.output / "box-actions.csv",
                                      blank, actionRows);
                    writeObj(arguments.output /
                                 ("box-action-" + stepLabel(actionIndex + 1u) +
                                  "-accepted.obj"),
                             compiled.world, currentSnapshot.femNodes, 0u);
                    continue;
                }
                row.status = "blocked_missing_cohesive_law";
                const auto gaps = numi::cardboard::measureBoxClosureGapDiagnostics(
                    blank, blank.points, currentSnapshot.femNodes, 0u, nodeCount,
                    action.feature.index == 1u);
                std::ostringstream diagnostic;
                diagnostic << std::setprecision(9)
                    << "source action requires adhesive/cohesive traction-separation coupling; "
                    << "nearest-node diagnostics only (not continuous contact): manufacturer_joint_m="
                    << (gaps.manufacturerJoint.valid ? gaps.manufacturerJoint.distance : -1.0)
                    << ", short_flap_overlap_m="
                    << (gaps.shortFlapOverlap.valid ? gaps.shortFlapOverlap.distance : -1.0)
                    << ", long_flap_overlap_m="
                    << (gaps.longFlapOverlap.valid ? gaps.longFlapOverlap.distance : -1.0);
                row.diagnostic = diagnostic.str();
                blockedByBondLaw = true;
                stopReason = row.diagnostic;
                break;
            }

            const RuntimeStateSnapshot actionStart = currentSnapshot;
            bool completed = false;
            for (std::uint32_t localStep = 0u;
                 localStep < arguments.boxMaximumStepsPerAction; ++localStep) {
                const auto plan = numi::cardboard::planBoxAssemblyActionStep(
                    blank, blank.points, exposedBoundaryNodeIndices,
                    currentSnapshot.femNodes,
                    actionStart.femNodes, 0u, nodeCount, actionIndex,
                    arguments.boxAngleIncrementDegrees, 0.5,
                    arguments.boxShapeResidualBoundCalipers,
                    arguments.boxShapeFitCoverageMinimum);
                if (plan.angleObservable) {
                    row.shapeFitObservable = true;
                    row.shapeFitResidualRmsM = plan.fitResidualRms;
                    row.shapeFitResidualBoundM = plan.shapeFitResidualBound;
                    row.shapeFitCoverage = plan.fitCoverage;
                    row.shapeFitWithinBound = plan.shapeFitWithinBound;
                }
                if (plan.shapeFitRejected) {
                    row.shapeFitCompletionRejected = true;
                    std::ostringstream diagnostic;
                    diagnostic << std::setprecision(9)
                        << "target angle reached but accepted interior shape fit failed engineering gate: residual_m="
                        << plan.fitResidualRms
                        << ", residual_bound_m=" << plan.shapeFitResidualBound
                        << ", coverage=" << plan.fitCoverage
                        << ", coverage_min=" << plan.shapeFitCoverageMinimum;
                    row.status = "shape_fit_rejected";
                    row.diagnostic = diagnostic.str();
                    shapeFitRejection = true;
                    nativeFailure = true;
                    stopReason = row.diagnostic;
                    break;
                }
                if (!plan.valid || plan.physicalBondLawRequired) {
                    row.status = "controller_invalid";
                    row.diagnostic = "accepted-state boundary-action planner returned an invalid or bond-dependent step";
                    nativeFailure = true;
                    stopReason = row.diagnostic;
                    break;
                }
                if (plan.angleObservable) {
                    const std::uint32_t hingeScore =
                        action.kind == numi::cardboard::BoxAssemblyAction::Kind::foldFlap
                            ? action.feature.index : action.hingeScoreLineIndex;
                    if (hingeScore < blank.scoreLines.size()) {
                        const auto scoreObservations = boxScoreAngles(
                            mesh, actionStart.femNodes, currentSnapshot.femNodes, 0u,
                            nodeCount);
                        const auto& measured = scoreObservations[hingeScore];
                        row.angleValid = measured.valid;
                        row.observedFoldTurnDegrees = measured.foldTurnDegrees;
                        row.observedInteriorAngleDegrees = measured.interiorAngleDegrees;
                    }
                }
                if (plan.phase == numi::cardboard::BoxAssemblyActionStep::Phase::acceptedContactOnly) {
                    if (action.kind == numi::cardboard::BoxAssemblyAction::Kind::erectWallPanel) {
                        row.status = "completed_relative_turn_clamp_retained";
                        row.diagnostic =
                            "accepted relative turn is within tolerance; external wall boundary clamp retained for later folds, tube, and seam; no crease damage or adhesive law";
                    } else {
                        row.status = "completed_from_accepted_state";
                        row.diagnostic = "interior fit is within target tolerance and no temporary flap grip remains";
                    }
                    completed = true;
                    break;
                }
                auto targets = zeroTargets();
                for (const auto& target : plan.nodeTargets)
                    numi::cardboard::setBoxGripTarget(
                        targets, nodeCount, 0u, target.node, target.endpoint,
                        target.command);
                const std::uint32_t nativeStep = nextNativeStep;
                const bool accepted = attemptNativeStep(
                    plan.phase == numi::cardboard::BoxAssemblyActionStep::Phase::releaseGrip
                        ? "wall_grip_release" : "wall_grip_move",
                    actionIndex, targets, parkedBodies());
                if (row.firstNativeStep == NM_INVALID_INDEX)
                    row.firstNativeStep = nativeStep;
                row.lastNativeStep = nativeStep;
                if (accepted) ++row.acceptedNativeSteps;
                const std::uint32_t hingeScore =
                    action.kind == numi::cardboard::BoxAssemblyAction::Kind::foldFlap
                        ? action.feature.index : action.hingeScoreLineIndex;
                if (hingeScore < blank.scoreLines.size()) {
                    const auto scoreObservations = boxScoreAngles(
                        mesh, actionStart.femNodes, currentSnapshot.femNodes, 0u,
                        nodeCount);
                    const auto& measured = scoreObservations[hingeScore];
                    row.angleValid = measured.valid;
                    row.observedFoldTurnDegrees = measured.foldTurnDegrees;
                    row.observedInteriorAngleDegrees = measured.interiorAngleDegrees;
                }
                if (!accepted) {
                    row.status = "native_rejected";
                    row.diagnostic = stopReason;
                    nativeFailure = true;
                    break;
                }
            }
            if (!nativeFailure && !completed) {
                row.status = "action_step_limit";
                row.diagnostic = "accepted boundary grip did not reach its interior-fit target before the authored native-step limit";
                nativeFailure = true;
                stopReason = row.diagnostic;
            }
            if (nativeFailure) break;
            writeBoxActionCsv(arguments.output / "box-actions.csv", blank, actionRows);
            writeObj(arguments.output / ("box-action-" + stepLabel(actionIndex + 1u) + "-accepted.obj"),
                     compiled.world, currentSnapshot.femNodes, 0u);
            if (arguments.shippingBox)
                writeShippingAcceptedCheckpoint(
                    arguments.output, currentSnapshot, compiled.world, arguments,
                    "assembly-action-" + stepLabel(actionIndex + 1u),
                    nextNativeStep, actionIndex, dieHeight,
                    shippingSimulationTimeSeconds);
        }
    }

    const bool stageCutoffReached = arguments.shippingBox &&
        shippingActionLimit < blank.assembly.actions.size() &&
        !nativeFailure && !blockedByBondLaw;
    if (stageCutoffReached) {
        for (std::uint32_t index = shippingActionLimit;
             index < actionRows.size(); ++index) {
            actionRows[index].status = "not_attempted_after_stage_cutoff";
            actionRows[index].diagnostic =
                "explicit --shipping-through-stage cutoff; no downstream assembly action was attempted";
        }
    }

    std::uint32_t acceptedSettleSteps = 0u;
    if (stageCutoffReached && arguments.shippingSettleSteps > 0u) {
        for (std::uint32_t settleStep = 0u;
             settleStep < arguments.shippingSettleSteps; ++settleStep) {
            if (!attemptNativeStep("held_boundary_settle",
                    numi::cardboard::kNoBoxScoreLineIndex,
                    zeroTargets(), parkedBodies())) {
                nativeFailure = true;
                break;
            }
            ++acceptedSettleSteps;
        }
        if (acceptedSettleSteps > 0u)
            writeShippingAcceptedCheckpoint(
                arguments.output, currentSnapshot, compiled.world, arguments,
                "held-boundary-settle", nextNativeStep,
                numi::cardboard::kNoBoxScoreLineIndex, dieHeight,
                shippingSimulationTimeSeconds);
        if (nativeFailure)
            stopReason = "native rejection during held-boundary settle; this is not a released assembly state";
    }

    if (blockedByBondLaw) {
        bool afterBlock = false;
        for (std::size_t index = 0u; index < actionRows.size(); ++index) {
            if (actionRows[index].status == "blocked_missing_cohesive_law") {
                afterBlock = true;
                continue;
            }
            if (afterBlock && actionRows[index].status == "not_attempted") {
                actionRows[index].status = "not_attempted_after_bond_law_block";
                actionRows[index].diagnostic = "sequence stops at the first source-described adhesive operation";
            }
        }
    } else if (shapeFitRejection) {
        for (auto& row : actionRows)
            if (row.status == "not_attempted") {
                row.status = "not_attempted_after_shape_fit_rejection";
                row.diagnostic = stopReason;
            }
    } else if (nativeFailure) {
        for (auto& row : actionRows)
            if (row.status == "not_attempted") {
                row.status = "not_attempted_after_native_failure";
                row.diagnostic = stopReason;
            }
    } else if (stageCutoffReached) {
        // The rows beyond the selected stage were explicitly marked above.
    }
    std::uint32_t releasedClampNodeCount = 0u;
    std::uint32_t remainingActiveTemporaryClampNodeCount = 0u;
    std::uint32_t remainingActivePrescribedNodeCount = 0u;
    bool releasedClampZeroActiveGatePassed = false;
    std::uint32_t releasedSettleSteps = 0u;
    double releasedAssemblyMaximumSpeedMPerS = std::numeric_limits<double>::infinity();
    bool releasedAssemblySettled = false;
    std::vector<NMFEMNodeStateGPU> heldBeforeClampReleaseNodes;
    bool heldBeforeClampReleaseAvailable = false;
    std::uint32_t heldBeforeClampReleaseStep = NM_INVALID_INDEX;
    double heldBeforeClampReleaseTimeSeconds =
        std::numeric_limits<double>::quiet_NaN();
    bool springbackRowsWritten = false;
    std::uint32_t springbackRowsEmitted = 0u;
    std::uint32_t springbackObservableRows = 0u;
    double releasedAssemblyRigidContactSamplesTotal = 0.0;
    double releasedAssemblySupportFloorContactSamples = 0.0;
    double releasedAssemblyForbiddenToolContactSamples = 0.0;
    [[maybe_unused]] std::filesystem::path releasedAssemblyCheckpointPath;
    [[maybe_unused]] std::filesystem::path releasedAssemblyRigidSidecarPath;
    [[maybe_unused]] std::filesystem::path releasedAssemblyRigidSidecarMetadataPath;
    [[maybe_unused]] std::string releasedAssemblyRigidSidecarSha256;
    [[maybe_unused]] std::filesystem::path shippingSupportRigidSidecarPath;
    [[maybe_unused]] std::filesystem::path shippingSupportRigidSidecarMetadataPath;
    [[maybe_unused]] std::string shippingSupportRigidSidecarSha256;
    [[maybe_unused]] bool shippingSupportRigidSidecarRestoreVerified = false;
    [[maybe_unused]] std::uint64_t shippingSupportRigidWorldFingerprint = 0u;
    [[maybe_unused]] double shippingSupportRigidMassKg = 0.0;
    [[maybe_unused]] std::filesystem::path shippingCrushAssemblyCheckpointPath;
    std::optional<numi::cardboard::ShippingOuterRoofMeasurement>
        shippingOuterRoofMeasurement;
    std::optional<numi::cardboard::ShippingImpactorFootprintPlan>
        shippingImpactorFootprintPlan;
    std::string shippingOuterRoofMeasurementFailure;
    const bool fullShippingActionSequence =
        arguments.shippingBox &&
        shippingActionLimit == blank.assembly.actions.size() &&
        !nativeFailure && !blockedByBondLaw && !shapeFitRejection;
    const auto writeSpringbackRows = [&](const bool settled,
                                         const char* unavailableReason) {
        if (!arguments.shippingBox || springbackRowsWritten) return;
        const bool acceptedEndpoints = settled && !nativeFailure &&
            releasedSettleSteps > 0u && heldBeforeClampReleaseAvailable &&
            releasedClampZeroActiveGatePassed &&
            remainingActiveTemporaryClampNodeCount == 0u &&
            remainingActivePrescribedNodeCount == 0u &&
            currentSnapshot.femNodes.size() >=
                static_cast<std::size_t>(arguments.environmentCount) * nodeCount &&
            boxEnvironmentAccepted(compiled.world, currentSnapshot, 0u) &&
            (arguments.environmentCount == 1u ||
             boxEnvironmentAccepted(compiled.world, currentSnapshot, 1u));
        for (std::uint32_t environment = 0u;
             environment < arguments.environmentCount; ++environment) {
            for (std::uint32_t scoreIndex = 0u;
                 scoreIndex < blank.scoreLines.size(); ++scoreIndex) {
                BoxSpringbackObservation observation;
                if (acceptedEndpoints) {
                    try {
                        observation = boxSpringbackObservation(
                            mesh, initialNodes, heldBeforeClampReleaseNodes,
                            currentSnapshot.femNodes, environment, nodeCount,
                            scoreIndex, arguments.boxShapeResidualBoundCalipers,
                            arguments.boxShapeFitCoverageMinimum);
                    } catch (const std::exception&) {
                        observation = {};
                    }
                }
                const bool valid = acceptedEndpoints && observation.valid;
                springbackOutput << environment << ',' << scoreIndex << ','
                    << static_cast<unsigned>(blank.scoreLines[scoreIndex].kind) << ',';
                if (heldBeforeClampReleaseAvailable)
                    springbackOutput << heldBeforeClampReleaseStep;
                springbackOutput << ',';
                if (std::isfinite(heldBeforeClampReleaseTimeSeconds))
                    springbackOutput << heldBeforeClampReleaseTimeSeconds;
                springbackOutput << ',';
                if (lastAcceptedNativeStep != NM_INVALID_INDEX)
                    springbackOutput << lastAcceptedNativeStep;
                springbackOutput << ',';
                if (std::isfinite(shippingSimulationTimeSeconds))
                    springbackOutput << shippingSimulationTimeSeconds;
                springbackOutput << ',' << (settled ? 1 : 0) << ',';
                if (valid) springbackOutput << observation.heldTurnDegrees;
                springbackOutput << ',';
                if (valid) springbackOutput << observation.releasedTurnDegrees;
                springbackOutput << ',';
                if (valid) springbackOutput << observation.springbackTurnDegrees;
                springbackOutput << ',';
                if (std::isfinite(observation.heldFitRmsM))
                    springbackOutput << observation.heldFitRmsM;
                springbackOutput << ',';
                if (std::isfinite(observation.releasedFitRmsM))
                    springbackOutput << observation.releasedFitRmsM;
                springbackOutput << ',';
                if (std::isfinite(observation.fitBoundM))
                    springbackOutput << observation.fitBoundM;
                springbackOutput << ',';
                if (std::isfinite(observation.heldFitCoverage))
                    springbackOutput << observation.heldFitCoverage;
                springbackOutput << ',';
                if (std::isfinite(observation.releasedFitCoverage))
                    springbackOutput << observation.releasedFitCoverage;
                springbackOutput << ',' << observation.sampleCount << ','
                    << (observation.heldFitPassed ? 1 : 0) << ','
                    << (observation.releasedFitPassed ? 1 : 0) << ','
                    << (valid ? 1 : 0) << ','
                    << (valid ? "qualified_endpoint_shape_fits" :
                        acceptedEndpoints ? "shape_fit_or_coverage_gate_failed" :
                        unavailableReason)
                    << '\n';
                ++springbackRowsEmitted;
                if (valid) ++springbackObservableRows;
            }
        }
        springbackOutput.flush();
        require(springbackOutput.good(),
                "failed while writing held-to-released springback observations");
        springbackRowsWritten = true;
    };
    if (fullShippingActionSequence) {
        require(acceptedHmaActivationEvents.contains(
                    "accepted_manufacturer_lap_overlap") &&
                acceptedHmaActivationEvents.contains(
                    "accepted_bottom_flap_overlaps") &&
                acceptedHmaActivationEvents.contains(
                    "accepted_top_flap_overlaps"),
                "complete shipping sequence lacks an accepted named HMA overlap event");
#ifdef NUMI_CARDBOARD_METAL_WORLD
        // Put the real floor support under the accepted box while boundary
        // grips still hold the assembly. The support remains in the native
        // scene through release/settle; gravity is never disabled or the FEM
        // reference/history reset to manufacture a settled pose.
        const auto preReleaseBounds =
            numi::cardboard::measureShippingAcceptedBounds(
                currentSnapshot.femNodes, nodeCount, 0u);
        constexpr double supportHalfThickness = 0.002;
        shippingToolCenterX = 0.5 * (preReleaseBounds.minimum.x +
                                     preReleaseBounds.maximum.x);
        shippingToolCenterZ = 0.5 * (preReleaseBounds.minimum.z +
                                     preReleaseBounds.maximum.z);
        const auto floorPlacement =
            numi::cardboard::makeShippingNonpenetratingFloorPlacement(
                preReleaseBounds.minimum.y, supportHalfThickness,
                mesh.world.contactSlop);
        shippingFloorCenterY = floorPlacement.centerY;
        shippingFloorPlacementClearanceM = floorPlacement.positiveClearanceM;
        const auto& acceptedFloorScene = run.shippingRigidSceneBodies();
        require(acceptedFloorScene.size() == shippingSceneBodyCount,
                "pre-release ground placement requires the complete accepted scene");
        const double floorStartX = acceptedFloorScene[shippingFloorBodyIndex].position.x;
        const double floorStartZ = acceptedFloorScene[shippingFloorBodyIndex].position.z;
        const double floorStartY = acceptedFloorScene[shippingFloorBodyIndex].position.y;
        const double targetFloorY = shippingFloorCenterY;
        const double supportTravel = std::sqrt(
            std::pow(shippingToolCenterX - floorStartX, 2.0) +
            std::pow(shippingToolCenterZ - floorStartZ, 2.0) +
            std::pow(targetFloorY - floorStartY, 2.0));
        shippingFloorPlacementSteps = std::max(1u,
            static_cast<std::uint32_t>(std::ceil(supportTravel / 0.010)));
        bool floorPlacementAccepted = true;
        for (std::uint32_t step = 1u;
             step <= shippingFloorPlacementSteps; ++step) {
            const double fraction = static_cast<double>(step) /
                shippingFloorPlacementSteps;
            const double nextCenterX = floorStartX + fraction *
                (0.5 * (preReleaseBounds.minimum.x +
                        preReleaseBounds.maximum.x) - floorStartX);
            const double nextCenterZ = floorStartZ + fraction *
                (0.5 * (preReleaseBounds.minimum.z +
                        preReleaseBounds.maximum.z) - floorStartZ);
            const double nextFloorCenterY = floorStartY + fraction *
                (targetFloorY - floorStartY);
            if (!moveShippingToolBody(shippingFloorBodyIndex,
                    {nextCenterX, nextFloorCenterY, nextCenterZ},
                    numi::cardboard::kNoBoxScoreLineIndex,
                    "place_ground_support_before_release", 0.5)) {
                floorPlacementAccepted = false;
                nativeFailure = true;
                stopReason = "native runtime rejected nonpenetrating ground-support placement before clamp release";
                break;
            }
            shippingToolCenterX = nextCenterX;
            shippingToolCenterZ = nextCenterZ;
            shippingFloorCenterY = nextFloorCenterY;
        }
        if (floorPlacementAccepted &&
            !attemptNativeStep("ground_support_contact_gate_hold",
                numi::cardboard::kNoBoxScoreLineIndex,
                zeroTargets(), parkedBodies(), 0.0, true,
                "ground_support_before_release")) {
            floorPlacementAccepted = false;
            nativeFailure = true;
            stopReason = "native runtime rejected the stationary pre-release ground-support endpoint";
        }
        if (floorPlacementAccepted && !nativeFailure &&
            lastForbiddenRigidToolContactSamples > 1.0e-6) {
            floorPlacementAccepted = false;
            nativeFailure = true;
            stopReason = "score/anvil/platen/free-body contact remained while placing the shipping ground support";
        }
        if (!floorPlacementAccepted) releasedAssemblySettled = false;
#endif
        if (nativeFailure) {
            // Preserve the accepted assembly prefix and do not run a release
            // or settle stage after a rejected support-tool placement.
        } else {
        heldBeforeClampReleaseAvailable =
            currentSnapshot.femNodes.size() >=
                static_cast<std::size_t>(arguments.environmentCount) * nodeCount &&
            boxEnvironmentAccepted(compiled.world, currentSnapshot, 0u) &&
            (arguments.environmentCount == 1u ||
             boxEnvironmentAccepted(compiled.world, currentSnapshot, 1u));
        if (heldBeforeClampReleaseAvailable) {
            const std::uint32_t activeHeldGrips =
                numi::cardboard::countActiveBoxTemporaryGripNodes(
                    currentSnapshot.femNodes, arguments.environmentCount,
                    nodeCount);
            heldBeforeClampReleaseAvailable = activeHeldGrips > 0u;
        }
        if (heldBeforeClampReleaseAvailable) {
            heldBeforeClampReleaseNodes = currentSnapshot.femNodes;
            heldBeforeClampReleaseStep = lastAcceptedNativeStep;
            heldBeforeClampReleaseTimeSeconds = shippingSimulationTimeSeconds;
        }
        auto releaseTargets = zeroTargets();
        for (std::uint32_t node = 0u; node < nodeCount; ++node) {
            const auto& accepted = currentSnapshot.femNodes.at(node);
            if (accepted.restAndFixed.w != 3.0f) continue;
            numi::cardboard::setBoxGripTarget(
                releaseTargets, nodeCount, 0u, node,
                numi::cardboard::boxAcceptedPosition(accepted), 2.0f);
            ++releasedClampNodeCount;
        }
        require(releasedClampNodeCount > 0u,
                "full shipping assembly has no temporary external grips to release");
        if (!attemptNativeStep("release_all_temporary_boundary_clamps",
                numi::cardboard::kNoBoxScoreLineIndex,
                releaseTargets, parkedBodies())) {
            nativeFailure = true;
            stopReason = "native runtime rejected release of temporary assembly boundary clamps";
        } else {
            try {
                remainingActiveTemporaryClampNodeCount =
                    numi::cardboard::countActiveBoxTemporaryGripNodes(
                        currentSnapshot.femNodes, arguments.environmentCount,
                        nodeCount);
                releasedClampZeroActiveGatePassed =
                    remainingActiveTemporaryClampNodeCount == 0u;
            } catch (const std::exception& error) {
                stopReason = std::string("cannot verify released clamp state: ") +
                    error.what();
            }
            try {
                remainingActivePrescribedNodeCount =
                    numi::cardboard::countActiveBoxPrescribedNodes(
                        currentSnapshot.femNodes, arguments.environmentCount,
                        nodeCount);
            } catch (const std::exception& error) {
                nativeFailure = true;
                stopReason = std::string("cannot verify released prescribed-node state: ") +
                    error.what();
            }
            if (!releasedClampZeroActiveGatePassed) {
                nativeFailure = true;
                if (stopReason.empty())
                    stopReason = "accepted release endpoint retained one or more temporary grip tags";
            } else if (remainingActivePrescribedNodeCount != 0u) {
                nativeFailure = true;
                stopReason = "released shipping blank retained a prescribed FEM node; no permanent node constraints are declared";
            } else {
                tabAnchorActive = false;
                writeShippingAcceptedCheckpoint(
                    arguments.output, currentSnapshot, compiled.world, arguments,
                    "released-clamps", nextNativeStep,
                    numi::cardboard::kNoBoxScoreLineIndex, dieHeight,
                    shippingSimulationTimeSeconds);
            }
        }
        if (!nativeFailure) {
            for (std::uint32_t step = 0u;
                 step < arguments.shippingSettleSteps; ++step) {
                if (!attemptNativeStep("released_assembly_settle",
                        numi::cardboard::kNoBoxScoreLineIndex,
                        zeroTargets(), parkedBodies())) {
                    nativeFailure = true;
                    stopReason = "native runtime rejected released-assembly settling step";
                    break;
                }
                ++releasedSettleSteps;
            }
        }
        if (releasedSettleSteps > 0u) {
            try {
                remainingActiveTemporaryClampNodeCount =
                    numi::cardboard::countActiveBoxTemporaryGripNodes(
                        currentSnapshot.femNodes, arguments.environmentCount,
                        nodeCount);
            } catch (const std::exception& error) {
                nativeFailure = true;
                stopReason = std::string("cannot verify settled clamp state: ") +
                    error.what();
            }
            if (remainingActiveTemporaryClampNodeCount != 0u) {
                nativeFailure = true;
                stopReason = "settled released assembly reacquired temporary grip tags";
            }
            try {
                remainingActivePrescribedNodeCount =
                    numi::cardboard::countActiveBoxPrescribedNodes(
                        currentSnapshot.femNodes, arguments.environmentCount,
                        nodeCount);
            } catch (const std::exception& error) {
                nativeFailure = true;
                stopReason = std::string("cannot verify settled prescribed-node state: ") +
                    error.what();
            }
            if (remainingActivePrescribedNodeCount != 0u) {
                nativeFailure = true;
                stopReason = "settled released shipping blank retained a prescribed FEM node";
            }
            releasedAssemblyMaximumSpeedMPerS = 0.0;
            for (std::uint32_t node = 0u; node < nodeCount; ++node) {
                const auto& velocity = currentSnapshot.femNodes.at(node)
                    .velocityAndInverseMass;
                const double speed = std::sqrt(
                    static_cast<double>(velocity.x) * velocity.x +
                    static_cast<double>(velocity.y) * velocity.y +
                    static_cast<double>(velocity.z) * velocity.z);
                if (!std::isfinite(speed)) {
                    releasedAssemblyMaximumSpeedMPerS =
                        std::numeric_limits<double>::infinity();
                    break;
                }
                releasedAssemblyMaximumSpeedMPerS =
                    std::max(releasedAssemblyMaximumSpeedMPerS, speed);
            }
            releasedAssemblySettled =
                !nativeFailure &&
                releasedClampZeroActiveGatePassed &&
                remainingActiveTemporaryClampNodeCount == 0u &&
                releasedAssemblyMaximumSpeedMPerS <= 1.0e-4;
            if (releasedAssemblySettled &&
                lastForbiddenRigidToolContactSamples > 1.0e-6) {
                nativeFailure = true;
                releasedAssemblySettled = false;
                stopReason = "settled release endpoint retains temporary score/anvil/platen/free-body contact";
            }
            if (!nativeFailure && releasedAssemblySettled &&
                releasedClampZeroActiveGatePassed &&
                remainingActiveTemporaryClampNodeCount == 0u) {
                releasedAssemblyRigidContactSamplesTotal =
                    lastRigidContactSamples;
                releasedAssemblySupportFloorContactSamples =
                    lastSupportFloorContactSamples;
                releasedAssemblyForbiddenToolContactSamples =
                    lastForbiddenRigidToolContactSamples;
                releasedAssemblyCheckpointPath = writeShippingAcceptedCheckpoint(
                    arguments.output, currentSnapshot, compiled.world, arguments,
                    "released-settled-assembly", nextNativeStep,
                    numi::cardboard::kNoBoxScoreLineIndex, dieHeight,
                    shippingSimulationTimeSeconds);
#ifdef NUMI_CARDBOARD_METAL_WORLD
                require(currentSnapshot.rigidStates.size() ==
                            compiled.world.contact.rigidProxies.size() *
                                arguments.environmentCount &&
                            currentSnapshot.rigidGeneralizedCandidate.size() ==
                            compiled.world.dispatch.rigidGeneralizedCapacity *
                                arguments.environmentCount,
                        "released assembly snapshot lacks full Matter rigid state for its paired scene sidecar");
                const auto sidecar = writeShippingRigidSceneSidecar(
                    releasedAssemblyCheckpointPath, currentSnapshot,
                    run.shippingRigidWorldFingerprint(),
                    run.shippingRigidFreeBodyMassKg(),
                    run.shippingRigidGravityMPerS2(),
                    run.shippingRigidSceneBodies(),
                    remainingActiveTemporaryClampNodeCount,
                    lastRigidContactSamples,
                    lastSupportFloorContactSamples,
                    shippingSimulationTimeSeconds, true);
                releasedAssemblyRigidSidecarPath = sidecar.binaryPath;
                releasedAssemblyRigidSidecarMetadataPath = sidecar.metadataPath;
                releasedAssemblyRigidSidecarSha256 = sidecar.binarySha256;
#else
                nativeFailure = true;
                releasedAssemblySettled = false;
                stopReason = "released assembly cannot be paired with native rigid scene state in this build";
#endif
            }
            if (!releasedAssemblySettled && !nativeFailure)
                stopReason = "accepted released assembly remained above the explicit 1e-4 m/s settle speed gate";
        }
        }
    }
    if (fullShippingActionSequence) {
        const char* reason = releasedAssemblySettled
            ? "" : !heldBeforeClampReleaseAvailable
                ? "accepted_held_endpoint_unavailable"
                : releasedSettleSteps == 0u
                    ? "release_settle_not_completed"
                    : nativeFailure
                        ? "release_transaction_or_settle_failed"
                        : "released_endpoint_failed_settle_or_constraint_gate";
        writeSpringbackRows(releasedAssemblySettled, reason);
    }

    // The completed assembly checkpoint is the only seed for the subsequent
    // platen and impact protocol. Keep the two tests as separate accepted
    // branches: compression measures Fpeak, while each impact mass starts from
    // the same uncompressed, released FEM snapshot.
#ifdef NUMI_CARDBOARD_METAL_WORLD
    if (arguments.shippingBox && fullShippingActionSequence &&
        releasedAssemblySettled && !nativeFailure) {
        require(!releasedAssemblyCheckpointPath.empty(),
                "settled shipping assembly did not publish an accepted checkpoint");
        const auto bounds = numi::cardboard::measureShippingAcceptedBounds(
            currentSnapshot.femNodes, nodeCount, 0u);
        bool supportPositioningAccepted = true;
        std::vector<MRBodyStateGPU> shippingSupportSceneBodies;
        if (!attemptNativeStep("released_ground_support_contact_gate_hold",
                numi::cardboard::kNoBoxScoreLineIndex,
                zeroTargets(), parkedBodies(), 0.0, true,
                "crush_checkpoint_ground_support_gate")) {
            supportPositioningAccepted = false;
            nativeFailure = true;
            stopReason = "native runtime rejected the stationary ground-support endpoint";
        }
        if (supportPositioningAccepted && !nativeFailure &&
            lastForbiddenRigidToolContactSamples > 1.0e-6) {
            supportPositioningAccepted = false;
            nativeFailure = true;
            stopReason = "support-ready checkpoint retains temporary score/anvil/platen/free-body contacts";
        }
        if (supportPositioningAccepted && !nativeFailure &&
            lastSupportFloorContactSamples <= 0.0) {
            supportPositioningAccepted = false;
            nativeFailure = true;
            stopReason = "released shipping assembly has no native contact reaction from its declared ground support";
        }
        if (supportPositioningAccepted) {
            shippingCrushAssemblyCheckpointPath =
                writeShippingAcceptedCheckpoint(
                    arguments.output, currentSnapshot, compiled.world, arguments,
                    "released-settled-crush-support-ready", nextNativeStep,
                    numi::cardboard::kNoBoxScoreLineIndex, dieHeight,
                    shippingSimulationTimeSeconds);
            try {
                numi::cardboard::ShippingOuterRoofMeasurement measurement;
                measurement.bounds =
                    numi::cardboard::measureShippingOuterRoofBounds(
                        currentSnapshot.femNodes, nodeCount,
                        compiled.world.fem.tetrahedra,
                        compiled.world.fem.surfaceFaces,
                        blank.boardCaliper, 0u, 0u);
                measurement.measuredFromAcceptedFixtureFreeSettledCheckpoint =
                    releasedAssemblySettled && releasedClampZeroActiveGatePassed &&
                    remainingActiveTemporaryClampNodeCount == 0u &&
                    lastForbiddenRigidToolContactSamples <= 1.0e-6;
                measurement.checkpointSha256 =
                    hexDigest(sha256File(shippingCrushAssemblyCheckpointPath));
                measurement.sourcePhysicsFingerprint =
                    compiled.world.physicsFingerprint;
                measurement.compiledWorldFingerprint = compiled.world.fingerprint;
                measurement.sourceBlankGeometrySha256 =
                    shippingBlankGeometryBindingDigest(mesh);
                measurement.sourceShippingBindingSha256 =
                    shippingSourceBindingDigest(arguments);
                measurement.acceptedCheckpointFile =
                    shippingCrushAssemblyCheckpointPath
                        .lexically_relative(arguments.output).generic_string();
                measurement.compiledImpactorHalfExtentsM =
                    arguments.shippingImpactorHalfExtentsM;
                measurement.checkpointFileHashVerified = true;
                measurement.acceptedRecordSchemaVerified = true;
                arguments.shippingImpactorCurrentRoofMatchesCompiledProxy =
                    arguments.shippingImpactorCompiledProxyMatchesMeasurement &&
                    arguments.shippingImpactorSourceMeasurementValidated &&
                    numi::cardboard::shippingImpactorHalfExtentsMatchMeasuredRoof(
                        measurement, arguments.shippingImpactorHalfExtentsM);
                shippingImpactorFootprintPlan =
                    numi::cardboard::makeShippingImpactorFootprintPlan(
                        measurement, measurement.checkpointSha256,
                        compiled.world.physicsFingerprint,
                        compiled.world.fingerprint);
                shippingOuterRoofMeasurement = measurement;
                std::ofstream roofOutput(
                    arguments.output / "shipping-outer-roof-measurement.json");
                require(roofOutput.good(),
                        "cannot create accepted outer-roof measurement record");
                const auto& roof = measurement.bounds;
                const auto& footprint = *shippingImpactorFootprintPlan;
                roofOutput << std::setprecision(17)
                    << "{\n  \"schema\": \"numi.cardboard.accepted-outer-roof.v2\",\n"
                    << "  \"status\": \"measured_from_accepted_settled_checkpoint\",\n"
                    << "  \"units\": \"m\",\n"
                    << "  \"method\": \"" << jsonEscape(measurement.method) << "\",\n"
                    << "  \"accepted_fem_checkpoint_file\": \""
                    << jsonEscape(measurement.acceptedCheckpointFile)
                    << "\",\n  \"accepted_fem_checkpoint_sha256\": \""
                    << measurement.checkpointSha256 << "\",\n"
                    << "  \"source_blank_geometry_sha256\": \""
                    << measurement.sourceBlankGeometrySha256 << "\",\n"
                    << "  \"source_shipping_binding_sha256\": \""
                    << measurement.sourceShippingBindingSha256 << "\",\n"
                    << "  \"source_physics_fingerprint\": "
                    << measurement.sourcePhysicsFingerprint << ",\n"
                    << "  \"compiled_world_fingerprint\": "
                    << measurement.compiledWorldFingerprint << ",\n"
                    << "  \"compiled_impactor_half_extents_m\": ["
                    << measurement.compiledImpactorHalfExtentsM[0] << ','
                    << measurement.compiledImpactorHalfExtentsM[1] << ','
                    << measurement.compiledImpactorHalfExtentsM[2] << "],\n"
                    << "  \"outer_roof_bounds\": {\"min_x\":" << roof.minimumX
                    << ",\"max_x\":" << roof.maximumX
                    << ",\"min_z\":" << roof.minimumZ
                    << ",\"max_z\":" << roof.maximumZ
                    << ",\"top_y\":" << roof.topY
                    << ",\"exterior_face_count\":" << roof.exteriorFaceCount
                    << "},\n  \"impactor_full_plan_x_m\": "
                    << footprint.fullLengthX
                    << ",\n  \"impactor_full_plan_z_m\": "
                    << footprint.fullWidthZ
                    << ",\n  \"impactor_center_x_m\": " << footprint.centerX
                    << ",\n  \"impactor_center_z_m\": " << footprint.centerZ
                    << ",\n  \"quarter_width_offset_z_m\": "
                    << footprint.quarterWidthOffsetZ
                    << ",\n  \"footprint_basis\": \""
                    << jsonEscape(footprint.basis)
                    << "\",\n  \"impact_contact_qualification\": false,\n"
                    << "  \"geometry_claim\": \"measured outer roof; source-run compiled impactor extents are recorded separately and do not imply that this footprint was covered\"\n}\n";
                require(roofOutput.good(),
                        "failed writing accepted outer-roof measurement record");
            } catch (const std::exception& error) {
                shippingOuterRoofMeasurement.reset();
                shippingImpactorFootprintPlan.reset();
                shippingOuterRoofMeasurementFailure = error.what();
            }
        }

        if (supportPositioningAccepted && !nativeFailure) {
            // Bind the external rigid tool poses and native rigid projection to
            // the accepted FEM checkpoint for exact branch reconstruction.
            shippingSupportSceneBodies = run.shippingRigidSceneBodies();
            require(shippingSupportSceneBodies.size() == shippingSceneBodyCount,
                    "shipping impact checkpoint needs all tools and its accepted finite-mass scene body");
            require(currentSnapshot.rigidStates.size() ==
                        compiled.world.contact.rigidProxies.size() *
                            arguments.environmentCount,
                    "shipping impact checkpoint lacks the Matter rigid-proxy state arena");
            require(currentSnapshot.rigidGeneralizedCandidate.size() ==
                        compiled.world.dispatch.rigidGeneralizedCapacity *
                            arguments.environmentCount,
                    "shipping impact checkpoint lacks the native free-body generalized candidate");
            require(lastForbiddenRigidToolContactSamples <= 1.0e-6 &&
                        lastSupportFloorContactSamples > 0.0,
                    "shipping impact checkpoint must retain only its supporting-floor contact");
            const auto checkpointGravity = run.shippingRigidGravityMPerS2();
            shippingSupportRigidWorldFingerprint =
                run.shippingRigidWorldFingerprint();
            shippingSupportRigidMassKg = run.shippingRigidFreeBodyMassKg();
            const auto pairedSidecar = writeShippingRigidSceneSidecar(
                shippingCrushAssemblyCheckpointPath, currentSnapshot,
                shippingSupportRigidWorldFingerprint,
                shippingSupportRigidMassKg, checkpointGravity,
                shippingSupportSceneBodies,
                remainingActiveTemporaryClampNodeCount,
                lastRigidContactSamples,
                lastSupportFloorContactSamples,
                shippingSimulationTimeSeconds, false);
            shippingSupportRigidSidecarPath = pairedSidecar.binaryPath;
            shippingSupportRigidSidecarMetadataPath = pairedSidecar.metadataPath;
            shippingSupportRigidSidecarSha256 = pairedSidecar.binarySha256;
            const auto serializedSupportBodies = readShippingRigidSceneBinary(
                shippingSupportRigidSidecarPath,
                shippingCrushAssemblyCheckpointPath,
                currentSnapshot.sourcePhysicsFingerprint,
                currentSnapshot.deviceProgramFingerprint,
                run.shippingRigidWorldFingerprint(),
                run.shippingRigidFreeBodyMassKg(),
                static_cast<std::uint32_t>(shippingSupportSceneBodies.size()),
                currentSnapshot.controlStep, false);
            require(std::equal(serializedSupportBodies.begin(),
                        serializedSupportBodies.end(),
                        shippingSupportSceneBodies.begin(),
                        sameShippingSceneBodyState),
                    "support-ready sidecar round trip changed the accepted MetalWorld scene bodies");
            shippingSupportSceneBodies = serializedSupportBodies;
            std::ofstream rigidCheckpoint(
                arguments.output / "shipping-rigid-accepted-checkpoint.json");
            require(rigidCheckpoint.good(),
                    "cannot create the shipping rigid checkpoint sidecar");
            rigidCheckpoint << std::setprecision(17)
                << "{\n  \"schema\": \"numi.cardboard.shipping-rigid-checkpoint.v1\",\n"
                << "  \"fem_checkpoint_file\": \""
                << jsonEscape(shippingCrushAssemblyCheckpointPath.filename().string())
                << "\",\n  \"fem_checkpoint_sha256\": \""
                << hexDigest(sha256File(shippingCrushAssemblyCheckpointPath))
                << "\",\n  \"paired_scene_sidecar_file\": \""
                << jsonEscape(shippingSupportRigidSidecarPath.filename().string())
                << "\",\n  \"paired_scene_sidecar_sha256\": \""
                << shippingSupportRigidSidecarSha256
                << "\",\n  \"paired_scene_bodies_sha256\": \""
                << pairedSidecar.sceneBodiesSha256
                << "\",\n  \"rigid_world_fingerprint\": "
                << run.shippingRigidWorldFingerprint()
                << ",\n  \"rigid_mass_kg\": "
                << run.shippingRigidFreeBodyMassKg()
                << ",\n  \"rigid_gravity_m_s2\": ["
                << checkpointGravity[0] << ',' << checkpointGravity[1] << ','
                << checkpointGravity[2] << "],\n"
                << "  \"rigid_generalized_coordinates\": {\"nq\": 0, \"nv\": 0},\n"
                << "  \"scene_bodies\": [\n";
            for (std::size_t index = 0u;
                 index < shippingSupportSceneBodies.size(); ++index) {
                const auto& body = shippingSupportSceneBodies[index];
                rigidCheckpoint << "    {\"index\":" << index
                    << ",\"position_m\":[" << body.position.x << ','
                    << body.position.y << ',' << body.position.z
                    << "],\"orientation_xyzw\":[" << body.orientation.x << ','
                    << body.orientation.y << ',' << body.orientation.z << ','
                    << body.orientation.w << "],\"linear_velocity_inverse_mass\":["
                    << body.linearVelocityAndInverseMass.x << ','
                    << body.linearVelocityAndInverseMass.y << ','
                    << body.linearVelocityAndInverseMass.z << ','
                    << body.linearVelocityAndInverseMass.w
                    << "],\"angular_velocity\":[" << body.angularVelocity.x << ','
                    << body.angularVelocity.y << ',' << body.angularVelocity.z << ','
                    << body.angularVelocity.w << "],\"flags_and_indices\":["
                    << body.flagsAndIndices[0] << ',' << body.flagsAndIndices[1] << ','
                    << body.flagsAndIndices[2] << ',' << body.flagsAndIndices[3] << "]}"
                    << (index + 1u == shippingSupportSceneBodies.size()
                        ? "\n" : ",\n");
            }
            const auto write4 = [&](const char* key, const nm_float4& value) {
                rigidCheckpoint << "\"" << key << "\":[" << value.x << ','
                    << value.y << ',' << value.z << ',' << value.w << ']';
            };
            rigidCheckpoint << "  ],\n  \"matter_rigid_proxy_states\": [\n";
            for (std::size_t index = 0u;
                 index < currentSnapshot.rigidStates.size(); ++index) {
                const auto& state = currentSnapshot.rigidStates[index];
                rigidCheckpoint << "    {";
                write4("center_and_radius", state.centerAndRadius);
                rigidCheckpoint << ','; write4("extent", state.extent);
                rigidCheckpoint << ','; write4("orientation", state.orientation);
                rigidCheckpoint << ',';
                write4("linear_velocity_inverse_mass",
                       state.linearVelocityAndInverseMass);
                rigidCheckpoint << ','; write4("angular_velocity", state.angularVelocity);
                rigidCheckpoint << ','; write4("body_center", state.bodyCenter);
                rigidCheckpoint << ',';
                write4("inverse_inertia_row_0", state.inverseInertiaRow0);
                rigidCheckpoint << ',';
                write4("inverse_inertia_row_1", state.inverseInertiaRow1);
                rigidCheckpoint << ',';
                write4("inverse_inertia_row_2", state.inverseInertiaRow2);
                rigidCheckpoint << '}'
                    << (index + 1u == currentSnapshot.rigidStates.size()
                        ? "\n" : ",\n");
            }
            rigidCheckpoint << "  ],\n  \"matter_rigid_generalized_candidate\": [";
            for (std::size_t index = 0u;
                 index < currentSnapshot.rigidGeneralizedCandidate.size(); ++index)
                rigidCheckpoint << (index == 0u ? "" : ",")
                    << currentSnapshot.rigidGeneralizedCandidate[index];
            rigidCheckpoint << "],\n  \"native_rigid_contact_samples\": "
                << lastRigidContactSamples << ",\n"
                << "  \"native_support_floor_contact_samples\": "
                << lastSupportFloorContactSamples << ",\n"
                << "  \"native_temporary_tool_contact_samples\": "
                << lastForbiddenRigidToolContactSamples << ",\n"
                << "  \"matter_contact_history_restore\": {\"serialized_in_matter_checkpoint\": true, \"accepted_history_sha256\": \""
                << sha256Arena(currentSnapshot.contactHistories)
                << "\", \"deformable_history_sha256\": \""
                << sha256Arena(currentSnapshot.deformableContactHistories)
                << "\", \"branch_restore_hash_checked\": true},\n"
                << "  \"metalworld_contact_impulse_warm_start\": {\"enabled\": false, \"scope\": \"MetalWorld rigid impulse warm start only; distinct from serialized Matter accepted contact histories\"},\n"
                << "  \"release_contact_gate\": \"support-floor contact permitted; temporary rigid tools must have zero contact samples\",\n"
                << "  \"checkpoint_contains_complete_rigid_sidecar\": true,\n"
                << "  \"drop_branches_restore_same_matter_snapshot_then_apply_mass_pose_and_gravity_interventions\": true,\n"
                << "  \"qualification\": false\n}\n";
            require(rigidCheckpoint.good(),
                    "failed writing shipping rigid checkpoint sidecar");

            struct CompressionPassOutcome {
                bool complete = false;
                double peakForceN = 0.0;
                double peakStrokeM = 0.0;
                double finalStrokeM = 0.0;
                double totalWorkJ = 0.0;
                double workAtPeakJ = 0.0;
                double kineticEnergyAtPeakJ = 0.0;
                double kineticWorkRatioAtPeak = 0.0;
                std::uint64_t peakSampleIndex = 0u;
                std::vector<numi::cardboard::ShippingCompressionForceSample> forceCurve;
                std::vector<double> workByCurveSampleJ;
                std::vector<double> kineticByCurveSampleJ;
            };
            const double compressionSupportReadyY = shippingCompressionPlatenY;
            const auto runCompressionPass = [&](const double speedMmPerMinute,
                                                const std::string& branchName) {
                CompressionPassOutcome outcome;
                shippingCompressionAttempted = true;
                currentSnapshot =
                    numi::cardboard::readShippingAcceptedSnapshot(
                        shippingCrushAssemblyCheckpointPath,
                        compiled.world.physicsFingerprint,
                        compiled.world.fingerprint);
                const auto expectedMatterArenaHashes =
                    shippingPersistentArenaHashes(currentSnapshot);
                const std::string expectedRigidStatesHash =
                    sha256Arena(currentSnapshot.rigidStates);
                const std::string expectedGeneralizedHash =
                    sha256Arena(currentSnapshot.rigidGeneralizedCandidate);
                run.restoreShippingCheckpoint(currentSnapshot);
                const auto sidecarSceneBodies = readShippingRigidSceneBinary(
                    shippingSupportRigidSidecarPath,
                    shippingCrushAssemblyCheckpointPath,
                    currentSnapshot.sourcePhysicsFingerprint,
                    currentSnapshot.deviceProgramFingerprint,
                    shippingSupportRigidWorldFingerprint,
                    shippingSupportRigidMassKg,
                    static_cast<std::uint32_t>(shippingSupportSceneBodies.size()),
                    currentSnapshot.controlStep,
                    false);
                require(std::equal(sidecarSceneBodies.begin(),
                            sidecarSceneBodies.end(),
                            shippingSupportSceneBodies.begin(),
                            sameShippingSceneBodyState),
                        "support-ready rigid sidecar differs from the captured branch seed");
                run.restoreShippingRigidSceneBodies(sidecarSceneBodies);
                const auto restoredSceneBodies = run.shippingRigidSceneBodies();
                require(restoredSceneBodies.size() == sidecarSceneBodies.size() &&
                            std::equal(restoredSceneBodies.begin(),
                                restoredSceneBodies.end(), sidecarSceneBodies.begin(),
                                sameShippingSceneBodyState),
                        "support-ready rigid sidecar failed exact scene-body restore");
                shippingSupportRigidSidecarRestoreVerified = true;
                currentSnapshot = run.runtime().snapshot();
                require(currentSnapshot.available,
                        "compression branch did not restore its accepted support-ready checkpoint");
                require(shippingPersistentArenaHashes(currentSnapshot) ==
                            expectedMatterArenaHashes &&
                            sha256Arena(currentSnapshot.rigidStates) ==
                                expectedRigidStatesHash &&
                            sha256Arena(currentSnapshot.rigidGeneralizedCandidate) ==
                                expectedGeneralizedHash,
                        "compression restore changed the paired Matter/FEM/material/contact checkpoint state");
                shippingCompressionPlatenY = compressionSupportReadyY;
                shippingCompressionStrokeM = 0.0;
                shippingSimulationTimeSeconds = 0.0;

                numi::cardboard::ShippingCompressionProtocol protocol;
                const double platenSpeedMPerS = speedMmPerMinute * 1.0e-3 / 60.0;
                constexpr double displacementIncrementM = 0.25e-3;
                protocol.platenSpeedMPerS = platenSpeedMPerS;
                protocol.incrementM = displacementIncrementM;
                protocol.targetStrokeFractionOfInsideHeight = 0.70;
                const double platenHalfHeight = 0.002;
                const double firstContactClearance = 0.005;
                const double platenApproachY = bounds.maximum.y +
                    platenHalfHeight + firstContactClearance;
                const double platenStartY = shippingCompressionPlatenY;
                const std::uint32_t platenPositionSteps = std::max(1u,
                    static_cast<std::uint32_t>(std::ceil(
                        std::abs(platenApproachY - platenStartY) / 0.010)));
                bool platenPositioningAccepted = true;
                for (std::uint32_t step = 1u; step <= platenPositionSteps; ++step) {
                    const double fraction = static_cast<double>(step) /
                        platenPositionSteps;
                    const double nextY = platenStartY +
                        fraction * (platenApproachY - platenStartY);
                    const auto& scene = run.shippingRigidSceneBodies();
                    require(scene.size() == shippingSceneBodyCount,
                            "compression platen positioning lost the accepted scene");
                    const auto& platen = scene[shippingPlatenBodyIndex];
                    if (!moveShippingToolBody(shippingPlatenBodyIndex,
                            {platen.position.x, nextY, platen.position.z},
                            numi::cardboard::kNoBoxScoreLineIndex,
                            "compression_platen_positioning_" + branchName,
                            0.5)) {
                        platenPositioningAccepted = false;
                        nativeFailure = true;
                        stopReason = "native runtime rejected placement of the compression platen above the accepted assembly in " + branchName;
                        break;
                    }
                    shippingCompressionPlatenY = nextY;
                }
                if (platenPositioningAccepted &&
                    !attemptNativeStep(
                        "compression_platen_positioning_hold_" + branchName,
                        numi::cardboard::kNoBoxScoreLineIndex,
                        zeroTargets(), parkedBodies())) {
                    platenPositioningAccepted = false;
                    nativeFailure = true;
                    stopReason = "native runtime rejected the stationary compression-platen endpoint in " + branchName;
                }
                if (!platenPositioningAccepted || nativeFailure) return outcome;

                bool firstContactAccepted = false;
                const double firstContactIncrementM = displacementIncrementM;
                const double compressionTargetStrokeM =
                    arguments.shippingDesign->defaults.inside.heightY *
                    protocol.targetStrokeFractionOfInsideHeight;
                require(firstContactIncrementM < compressionTargetStrokeM,
                        "first contact increment consumes the planned compression stroke");
                numi::cardboard::ShippingFirstCompressionPeakDetector peakDetector(
                    protocol.requiredPeakDropFraction,
                    protocol.consecutiveDeclinesForPeak);
                outcome.forceCurve.push_back({0u, 0.0, 0.0, 0.0});
                outcome.workByCurveSampleJ.push_back(0.0);
                double preContactKineticEnergyJ = 0.0;
                for (std::uint32_t approach = 0u;
                     approach < 100u; ++approach) {
                    preContactKineticEnergyJ = femKineticEnergy(
                        currentSnapshot, nodeCount, 0u);
                    const double nextY = shippingCompressionPlatenY -
                        displacementIncrementM;
                    const auto& scene = run.shippingRigidSceneBodies();
                    require(scene.size() == shippingSceneBodyCount,
                            "compression contact approach lost the accepted scene");
                    const auto& platen = scene[shippingPlatenBodyIndex];
                    numi::cardboard::ShippingConstantSpeedMoveSummary moveTiming;
                    if (!moveShippingToolBody(shippingPlatenBodyIndex,
                            {platen.position.x, nextY, platen.position.z},
                            numi::cardboard::kNoBoxScoreLineIndex,
                            "compression_contact_approach_" + branchName,
                            platenSpeedMPerS, 0.0, &moveTiming)) {
                        nativeFailure = true;
                        stopReason = "native runtime rejected the low-rate compression platen contact approach in " + branchName;
                        break;
                    }
                    shippingCompressionPlatenY = nextY;
                    const double forceY = std::abs(lastPlatenMeanForceY);
                    const double kineticEnergyJ =
                        femKineticEnergy(currentSnapshot, nodeCount, 0u);
                    if (lastPlatenContactSamples > 0.0) {
                        firstContactAccepted = true;
                        shippingCompressionStrokeM = firstContactIncrementM;
                        outcome.totalWorkJ = 0.5 * forceY * firstContactIncrementM;
                        outcome.forceCurve.push_back({
                            1u, shippingCompressionStrokeM, forceY,
                            lastPlatenContactSamples});
                        outcome.workByCurveSampleJ.push_back(outcome.totalWorkJ);
                        outcome.kineticByCurveSampleJ.push_back(
                            preContactKineticEnergyJ);
                        outcome.kineticByCurveSampleJ.push_back(kineticEnergyJ);
                        peakDetector.observeAccepted(
                            shippingCompressionStrokeM, forceY,
                            lastPlatenContactSamples);
                        const double kineticToWorkRatio = outcome.totalWorkJ > 0.0
                            ? kineticEnergyJ / outcome.totalWorkJ
                            : std::numeric_limits<double>::quiet_NaN();
                        shippingCompressionLedger << (nextNativeStep > 1u
                                ? nextNativeStep - 2u : 0u)
                            << ',' << branchName << "_contact_origin,1";
                        for (std::uint32_t zeroColumn = 0u;
                             zeroColumn < 12u; ++zeroColumn)
                            shippingCompressionLedger << ",0";
                        shippingCompressionLedger << ','
                            << preContactKineticEnergyJ << ",,0\n";
                        shippingCompressionLedger << nextNativeStep - 1u
                            << ',' << branchName << "_contact_approach,1,"
                            << shippingCompressionStrokeM << ','
                            << shippingCompressionStrokeM /
                                arguments.shippingDesign->defaults.inside.heightY << ','
                            << moveTiming.observationDurationSeconds << ','
                            << moveTiming.acceptedStepCount << ','
                            << moveTiming.minimumControlTimestepSeconds << ','
                            << moveTiming.maximumControlTimestepSeconds << ','
                            << lastPlatenContactSamples << ','
                            << lastPlatenContactImpulseNs[1] << ',' << forceY
                            << ",0,0," << outcome.totalWorkJ << ','
                            << kineticEnergyJ << ',';
                        if (std::isfinite(kineticToWorkRatio))
                            shippingCompressionLedger << kineticToWorkRatio;
                        shippingCompressionLedger << ",0\n";
                    } else {
                        shippingCompressionLedger << nextNativeStep - 1u
                            << ',' << branchName << "_contact_approach,1,0,0,"
                            << moveTiming.observationDurationSeconds << ','
                            << moveTiming.acceptedStepCount << ','
                            << moveTiming.minimumControlTimestepSeconds << ','
                            << moveTiming.maximumControlTimestepSeconds << ','
                            << lastPlatenContactSamples << ','
                            << lastPlatenContactImpulseNs[1] << ',' << forceY
                            << ",0,0,0," << kineticEnergyJ << ",,0\n";
                    }
                    shippingCompressionLedger.flush();
                    if (firstContactAccepted) {
                        break;
                    }
                }
                if (!nativeFailure && !firstContactAccepted) {
                    nativeFailure = true;
                    stopReason = "compression platen did not produce an accepted native contact within the 25 mm approach bound in " + branchName;
                }
                if (!firstContactAccepted || nativeFailure) return outcome;

                auto remainingProtocol = protocol;
                remainingProtocol.targetStrokeFractionOfInsideHeight =
                    (compressionTargetStrokeM - firstContactIncrementM) /
                    arguments.shippingDesign->defaults.inside.heightY;
                const auto schedule =
                    numi::cardboard::makeShippingCompressionSchedule(
                        arguments.shippingDesign->defaults.inside.heightY,
                        remainingProtocol);
                bool compressionAccepted = true;
                bool peakQualified = false;
                for (const auto& increment : schedule) {
                    const double nextY = shippingCompressionPlatenY -
                        increment.displacementM;
                    const auto& scene = run.shippingRigidSceneBodies();
                    require(scene.size() == shippingSceneBodyCount,
                            "slow compression lost the accepted rigid scene");
                    const auto& platen = scene[shippingPlatenBodyIndex];
                    const double compressionSpeed = increment.displacementM /
                        increment.timestepSeconds;
                    numi::cardboard::ShippingConstantSpeedMoveSummary moveTiming;
                    if (!moveShippingToolBody(shippingPlatenBodyIndex,
                            {platen.position.x, nextY, platen.position.z},
                            numi::cardboard::kNoBoxScoreLineIndex,
                            "slow_compression_" + branchName,
                            compressionSpeed, 0.0, &moveTiming)) {
                        compressionAccepted = false;
                        nativeFailure = true;
                        stopReason = "native runtime rejected the released-box compression branch before its planned 70 percent inside-height stroke in " + branchName;
                        break;
                    }
                    shippingCompressionPlatenY = nextY;
                    shippingCompressionStrokeM += increment.displacementM;
                    if (!(lastPlatenContactSamples > 0.0)) {
                        compressionAccepted = false;
                        nativeFailure = true;
                        stopReason = "native compression platen lost FEM contact before a qualified Fpeak in " + branchName;
                        break;
                    }
                    const double forceY = std::abs(lastPlatenMeanForceY);
                    const double previousForceN = outcome.forceCurve.back().forceN;
                    outcome.totalWorkJ += 0.5 *
                        (previousForceN + forceY) * increment.displacementM;
                    const double kineticEnergyJ =
                        stateMetrics(compiled.world, currentSnapshot, 0u).kineticEnergyJ;
                    outcome.forceCurve.push_back({
                        static_cast<std::uint64_t>(outcome.forceCurve.size()),
                        shippingCompressionStrokeM, forceY,
                        lastPlatenContactSamples});
                    outcome.workByCurveSampleJ.push_back(outcome.totalWorkJ);
                    outcome.kineticByCurveSampleJ.push_back(kineticEnergyJ);
                    const auto qualified = peakDetector.observeAccepted(
                        shippingCompressionStrokeM, forceY,
                        lastPlatenContactSamples);
                    const double detectorDropFraction =
                        peakDetector.hasCandidate() &&
                            peakDetector.candidate().forceN > 0.0
                        ? std::max(0.0, 1.0 - forceY /
                            peakDetector.candidate().forceN)
                        : 0.0;
                    const double kineticToWorkRatio = outcome.totalWorkJ > 0.0
                        ? kineticEnergyJ / outcome.totalWorkJ
                        : std::numeric_limits<double>::quiet_NaN();
                    const bool peakJustQualified = qualified.has_value() &&
                        !peakQualified;
                    shippingCompressionLedger << nextNativeStep - 1u
                        << ',' << branchName << "_slow_compression,1,"
                        << shippingCompressionStrokeM << ','
                        << shippingCompressionStrokeM /
                            arguments.shippingDesign->defaults.inside.heightY << ','
                        << moveTiming.observationDurationSeconds << ','
                        << moveTiming.acceptedStepCount << ','
                        << moveTiming.minimumControlTimestepSeconds << ','
                        << moveTiming.maximumControlTimestepSeconds << ','
                        << lastPlatenContactSamples << ','
                        << lastPlatenContactImpulseNs[1] << ',' << forceY << ','
                        << (peakDetector.hasCandidate()
                            ? peakDetector.candidate().forceN : 0.0) << ','
                        << (peakJustQualified ? qualified->peak.forceN : 0.0) << ','
                        << outcome.totalWorkJ << ',' << kineticEnergyJ << ',';
                    if (std::isfinite(kineticToWorkRatio))
                        shippingCompressionLedger << kineticToWorkRatio;
                    shippingCompressionLedger << ',' << detectorDropFraction << '\n';
                    shippingCompressionLedger.flush();
                    if (peakJustQualified) {
                        peakQualified = true;
                        outcome.peakForceN = qualified->peak.forceN;
                        outcome.peakStrokeM = qualified->peak.strokeM;
                        outcome.peakSampleIndex = qualified->peak.sampleIndex;
                        const auto peakCurve = std::lower_bound(
                            outcome.forceCurve.begin(), outcome.forceCurve.end(),
                            qualified->peak.strokeM,
                            [](const auto& sample, const double stroke) {
                                return sample.strokeM < stroke;
                            });
                        const std::size_t peakCurveIndex = static_cast<std::size_t>(
                            peakCurve - outcome.forceCurve.begin());
                        require(peakCurveIndex < outcome.workByCurveSampleJ.size() &&
                                    peakCurveIndex < outcome.kineticByCurveSampleJ.size() &&
                                    peakCurve != outcome.forceCurve.end() &&
                                    peakCurve->strokeM == qualified->peak.strokeM,
                                "compression peak does not map to its accepted work/kinetic trace");
                        outcome.workAtPeakJ =
                            outcome.workByCurveSampleJ[peakCurveIndex];
                        outcome.kineticEnergyAtPeakJ =
                            outcome.kineticByCurveSampleJ[peakCurveIndex];
                        outcome.kineticWorkRatioAtPeak = outcome.workAtPeakJ > 0.0
                            ? outcome.kineticEnergyAtPeakJ / outcome.workAtPeakJ
                            : std::numeric_limits<double>::quiet_NaN();
                    }
                }
                outcome.finalStrokeM = shippingCompressionStrokeM;
                outcome.totalWorkJ = outcome.workByCurveSampleJ.empty()
                    ? 0.0 : outcome.workByCurveSampleJ.back();
                outcome.complete = compressionAccepted &&
                    !nativeFailure && peakQualified;
                if (outcome.complete) {
                    (void)writeShippingAcceptedCheckpoint(
                        arguments.output, currentSnapshot, compiled.world,
                        arguments,
                        "slow-compression-" + branchName + "-first-local-peak-qualified",
                        nextNativeStep,
                        numi::cardboard::kNoBoxScoreLineIndex, dieHeight,
                        shippingSimulationTimeSeconds,
                        "compression-from-" + branchName + "/" +
                            hexDigest(sha256File(
                                shippingCrushAssemblyCheckpointPath)));
                } else if (compressionAccepted && !nativeFailure) {
                    stopReason = "no first local compression peak met the two-decline and one-percent-drop rule within the 70 percent stroke bound in " + branchName;
                }
                return outcome;
            };

            const CompressionPassOutcome nominalCompression =
                runCompressionPass(12.5, "nominal_12_5_mm_min");
            CompressionPassOutcome halfSpeedCompression;
            if (!nativeFailure)
                halfSpeedCompression =
                    runCompressionPass(6.25, "half_speed_6_25_mm_min");
            shippingCompressionPeakForceN = nominalCompression.peakForceN;
            shippingCompressionPeakStrokeM = nominalCompression.peakStrokeM;
            shippingCompressionPeakSampleIndex =
                nominalCompression.peakSampleIndex;
            shippingCompressionHalfSpeedPeakForceN =
                halfSpeedCompression.peakForceN;
            shippingCompressionHalfSpeedPeakStrokeM =
                halfSpeedCompression.peakStrokeM;
            shippingCompressionHalfSpeedCompleted =
                halfSpeedCompression.complete;
            const bool bothPeakCurvesComplete = nominalCompression.complete &&
                halfSpeedCompression.complete;
            shippingCompressionBothBranchesReachedPeak = bothPeakCurvesComplete;
            shippingCompressionCompleted = false;
            shippingCompressionStrokeM = nominalCompression.finalStrokeM;
            shippingCompressionNominalWorkAtPeakJ =
                nominalCompression.workAtPeakJ;
            shippingCompressionHalfSpeedWorkAtPeakJ =
                halfSpeedCompression.workAtPeakJ;
            shippingCompressionNominalKineticAtPeakJ =
                nominalCompression.kineticEnergyAtPeakJ;
            shippingCompressionHalfSpeedKineticAtPeakJ =
                halfSpeedCompression.kineticEnergyAtPeakJ;
            shippingCompressionNominalKineticWorkRatioAtPeak =
                nominalCompression.kineticWorkRatioAtPeak;
            shippingCompressionHalfSpeedKineticWorkRatioAtPeak =
                halfSpeedCompression.kineticWorkRatioAtPeak;
            if (shippingCompressionPeakForceN > 0.0 &&
                shippingCompressionHalfSpeedPeakForceN > 0.0) {
                shippingCompressionHalfSpeedRelativePeakDifference =
                    numi::cardboard::shippingCompressionRelativePeakDifference(
                        shippingCompressionPeakForceN,
                        shippingCompressionHalfSpeedPeakForceN);
            }
            if (bothPeakCurvesComplete && !nativeFailure) {
                const auto sensitivity =
                    numi::cardboard::compareShippingCompressionCurves(
                        nominalCompression.forceCurve,
                        halfSpeedCompression.forceCurve,
                        shippingCompressionPeakForceN);
                shippingCompressionRateSensitivityEvaluated = true;
                shippingCompressionCommonStrokeM = sensitivity.commonStrokeM;
                shippingCompressionCommonStrokeNominalWorkJ =
                    sensitivity.nominalWorkJ;
                shippingCompressionCommonStrokeHalfSpeedWorkJ =
                    sensitivity.halfSpeedWorkJ;
                shippingCompressionRelativeWorkDifference =
                    sensitivity.relativeWorkDifference;
                shippingCompressionMaximumNormalizedCurveDifference =
                    sensitivity.maximumForceDifferenceNormalizedByNominalPeak;
                shippingCompressionRmsNormalizedCurveDifference =
                    sensitivity.rmsForceDifferenceNormalizedByNominalCurve;
                numi::cardboard::ShippingCompressionProtocol sensitivityProtocol;
                const bool peakGatePassed =
                    numi::cardboard::shippingCompressionPeakEquivalent(
                        shippingCompressionPeakForceN,
                        shippingCompressionHalfSpeedPeakForceN,
                        sensitivityProtocol.peakForceEquivalenceToleranceFraction);
                shippingCompressionRateSensitivityPassed = peakGatePassed &&
                    numi::cardboard::shippingCompressionSensitivityPassed(
                        sensitivity,
                        sensitivityProtocol.workEquivalenceToleranceFraction,
                        sensitivityProtocol.normalizedFullCurveToleranceFraction);
                shippingCompressionCompleted =
                    shippingCompressionRateSensitivityPassed;
            }
            if (!shippingCompressionCompleted && !nativeFailure &&
                stopReason.empty()) {
                stopReason = !bothPeakCurvesComplete
                    ? "nominal-speed and half-speed compression passes must each reach the first local peak on their shared accepted assembly checkpoint"
                    : "nominal-versus-half-speed compression failed the two-percent peak/work or five-percent normalized full-curve gate; impact masses were not instantiated";
            }

            if (!nominalCompression.forceCurve.empty()) {
                shippingCompressionNominalCurveSha256 =
                    writeShippingNominalCurveArtifact(
                        arguments.output, nominalCompression.forceCurve);
                shippingCompressionNominalCurveSampleCount =
                    nominalCompression.forceCurve.size();
            }
            if (arguments.shippingCapacityQualificationLoadError.size() > 0u) {
                shippingCompressionNumericalQualificationStatus = "invalid_input";
                shippingCompressionNumericalQualificationReason =
                    arguments.shippingCapacityQualificationLoadError;
            } else if (!arguments.shippingCapacityQualification.has_value()) {
                shippingCompressionNumericalQualificationStatus = "missing_required_evidence";
                shippingCompressionNumericalQualificationReason =
                    "mesh and timestep convergence evidence was not supplied; speed sensitivity alone does not qualify Fpeak";
            } else {
                const double terminalFraction =
                    nominalCompression.finalStrokeM /
                    arguments.shippingDesign->defaults.inside.heightY;
                const numi::cardboard::ShippingCapacityIdentity identity{
                    .boardRecipeId = "literature2009",
                    .sourcePhysicsFingerprint = compiled.world.physicsFingerprint,
                    .deviceProgramFingerprint = compiled.world.fingerprint,
                    .compiledWorldFingerprint = compiled.world.fingerprint,
                    .sourceBindingSha256 = shippingSourceBindingDigest(arguments),
                    .assembledMatterCheckpointSha256 =
                        std::filesystem::is_regular_file(
                            shippingCrushAssemblyCheckpointPath)
                        ? hexDigest(sha256File(shippingCrushAssemblyCheckpointPath))
                        : std::string{},
                    .assembledRigidSidecarSha256 = shippingSupportRigidSidecarSha256,
                    .nominalCurveSha256 = shippingCompressionNominalCurveSha256,
                    .nominalPeakSampleIndex = nominalCompression.peakSampleIndex,
                    .nominalAcceptedSampleCount = nominalCompression.forceCurve.size(),
                    .nominalTerminalStrokeFraction = terminalFraction,
                    .nominalFullStrokeCompleted = nominalCompression.complete &&
                        std::abs(terminalFraction - 0.70) <=
                            numi::cardboard::kShippingCapacityEndpointTolerance,
                    .nominalRateSensitivityPassed =
                        shippingCompressionRateSensitivityPassed};
                const auto qualification =
                    numi::cardboard::validateShippingCapacityQualification(
                        *arguments.shippingCapacityQualification, identity);
                shippingCompressionNumericallyQualified = qualification.qualified;
                shippingCompressionNumericalQualificationStatus =
                    qualification.qualified ? "qualified_for_numerical_capacity_only"
                                            : "rejected_binding_or_convergence_evidence";
                std::ostringstream reasons;
                for (std::size_t i = 0u; i < qualification.reasons.size(); ++i) {
                    if (i) reasons << ';';
                    reasons << qualification.reasons[i];
                }
                shippingCompressionNumericalQualificationReason = reasons.str();
            }
            if (!shippingCompressionNumericallyQualified && !nativeFailure &&
                stopReason.empty()) {
                stopReason = shippingCompressionCompleted
                    ? "compression histories completed, but Fpeak numerical capacity is not qualified by source-bound mesh and timestep evidence; no heavy-object masses were instantiated"
                    : "compression did not complete; no numerical capacity admission was evaluated";
            }
        }

        const bool shippingImpactorGeometryAdmitted =
            arguments.shippingImpactorSourceMeasurementValidated &&
            arguments.shippingImpactorCompiledProxyMatchesMeasurement &&
            arguments.shippingImpactorCurrentRoofMatchesCompiledProxy;
        if (shippingCompressionCompleted && shippingCompressionNumericallyQualified &&
            (!shippingImpactorFootprintPlan.has_value() ||
             !shippingImpactorGeometryAdmitted) &&
            !nativeFailure && stopReason.empty()) {
            stopReason = "Fpeak was numerically admitted, but heavy-load scenarios were blocked because the compiled BOX does not have a source-bound footprint that matches the accepted settled roof; a later run must provide --shipping-roof-measurement and matching explicit --shipping-impactor-half-extents-mm. Current roof match=" +
                (arguments.shippingImpactorCurrentRoofMatchesCompiledProxy
                    ? std::string("true") : std::string("false")) +
                (shippingOuterRoofMeasurementFailure.empty()
                    ? std::string{} : "; measurement error: " +
                        shippingOuterRoofMeasurementFailure);
        }

        if (shippingCompressionCompleted && shippingCompressionNumericallyQualified &&
            shippingImpactorFootprintPlan.has_value() &&
            shippingImpactorGeometryAdmitted &&
            !nativeFailure &&
            shippingCompressionPeakForceN > 0.0 &&
            std::filesystem::is_regular_file(
                shippingCrushAssemblyCheckpointPath)) {
            shippingDropCasesAttempted = true;
            const auto& impactorFootprint = *shippingImpactorFootprintPlan;
            constexpr double gravityMPerS2 = 9.80665;
            constexpr double dropHeightM = 0.100;
            constexpr double impactDurationSeconds = 0.250;
            const double impactorHalfHeightM =
                arguments.shippingImpactorHalfExtentsM[1];
            constexpr std::array<double, 3> ratios{0.5, 1.0, 1.5};
            const std::array<std::pair<std::string, double>, 2> locations{{
                {"center", 0.0},
                {"positive_quarter_width", 0.25}}};
            // Gravity-transfer compression is distinct from a 100 mm drop.
            // Each mass branch starts from the same accepted assembly snapshot,
            // at a nonpenetrating roof pose, with its own matched mass/inertia.
            // Gravity remains on while an external body wrench decreases from
            // body weight to zero; a separate zero-wrench transaction releases
            // the actuator before the post-release observation window.
            shippingGravityTransferAttempted = true;
            numi::cardboard::ShippingGravityTransferProtocol transferProtocol;
            transferProtocol.timestepSeconds =
                arguments.shippingGravityTransferTimestepSeconds;
            const std::uint32_t transferSteps =
                numi::cardboard::shippingGravityTransferStepCount(
                    transferProtocol);
            const std::uint32_t releaseObservationSteps = std::max(1u,
                static_cast<std::uint32_t>(std::ceil(
                    transferProtocol.postReleaseObservationSeconds /
                    transferProtocol.timestepSeconds)));
            const double transferDt = transferProtocol.timestepSeconds;
            const double actualTransferDurationSeconds =
                static_cast<double>(transferSteps) * transferDt;
            const double actualReleaseObservationSeconds =
                static_cast<double>(releaseObservationSteps) * transferDt;
            const double impactorHalfHeightForTransfer =
                arguments.shippingImpactorHalfExtentsM[1];
            for (std::size_t ratioIndex = 0u;
                 ratioIndex < ratios.size() && !nativeFailure; ++ratioIndex) {
                const double mass = numi::cardboard::shippingImpactMassFromPeakForce(
                    shippingCompressionPeakForceN, ratios[ratioIndex],
                    gravityMPerS2);
                const std::string caseId = "ratio-" +
                    std::to_string(static_cast<unsigned>(ratioIndex + 1u)) +
                    "-gravity-transfer";
                currentSnapshot = numi::cardboard::readShippingAcceptedSnapshot(
                    shippingCrushAssemblyCheckpointPath,
                    compiled.world.physicsFingerprint,
                    compiled.world.fingerprint);
                const auto sourceSceneBodies = readShippingRigidSceneBinary(
                    shippingSupportRigidSidecarPath,
                    shippingCrushAssemblyCheckpointPath,
                    currentSnapshot.sourcePhysicsFingerprint,
                    currentSnapshot.deviceProgramFingerprint,
                    shippingSupportRigidWorldFingerprint,
                    shippingSupportRigidMassKg,
                    static_cast<std::uint32_t>(shippingSupportSceneBodies.size()),
                    currentSnapshot.controlStep, false);
                require(std::equal(sourceSceneBodies.begin(), sourceSceneBodies.end(),
                            shippingSupportSceneBodies.begin(),
                            sameShippingSceneBodyState),
                        "gravity-transfer branch source sidecar differs from released support-ready state");
                const auto inheritedHashes =
                    shippingPersistentArenaHashes(currentSnapshot);
                require(currentSnapshot.rigidStates.size() ==
                            compiled.world.contact.rigidProxies.size() &&
                            currentSnapshot.rigidGeneralizedCandidate.size() ==
                            compiled.world.dispatch.rigidGeneralizedCapacity,
                        "gravity-transfer branch source checkpoint omits native rigid state");
                std::fill(currentSnapshot.rigidStates.begin(),
                          currentSnapshot.rigidStates.end(), NMRigidStateGPU{});
                std::fill(currentSnapshot.rigidGeneralizedCandidate.begin(),
                          currentSnapshot.rigidGeneralizedCandidate.end(), 0.0f);
                std::fill(currentSnapshot.reactions.begin(),
                          currentSnapshot.reactions.end(), NMRigidReactionGPU{});
                run.restoreShippingCheckpoint(currentSnapshot);
                const auto preSeedHashes = shippingPersistentArenaHashes(
                    run.runtime().snapshot());
                require(preSeedHashes == inheritedHashes,
                        "gravity-transfer branch changed inherited assembly FEM/history before rigid initialization");

                numi::cardboard::ShippingRigidBodySpec transferSpec;
                transferSpec.massKg = mass;
                transferSpec.halfExtentM = arguments.shippingImpactorHalfExtentsM;
                const double centerX = impactorFootprint.centerX;
                const double centerZ = impactorFootprint.centerZ;
                // Round the tangent pose upward if FP32 cannot represent the
                // exact geometric touch without a sub-ULP overlap. This is a
                // coordinate-rounding allowance, not an engineered clearance.
                float tangentCenterY = static_cast<float>(
                    impactorFootprint.roofY + impactorHalfHeightForTransfer);
                if (static_cast<double>(tangentCenterY) -
                        impactorHalfHeightForTransfer < impactorFootprint.roofY)
                    tangentCenterY = std::nextafter(
                        tangentCenterY,
                        std::numeric_limits<float>::infinity());
                const double initialRoofGapM =
                    static_cast<double>(tangentCenterY) -
                    impactorHalfHeightForTransfer - impactorFootprint.roofY;
                const double tangentRoundingBoundM = 2.0 *
                    std::numeric_limits<float>::epsilon() *
                    std::max(1.0, std::abs(impactorFootprint.roofY));
                require(initialRoofGapM >= 0.0 &&
                            initialRoofGapM <= tangentRoundingBoundM,
                        "gravity-transfer body could not be placed at a nonpenetrating tangent roof pose");
                transferSpec.initialCenterM = {
                    centerX,
                    tangentCenterY,
                    centerZ};
                transferSpec.gravityMPerS2 = {0.0, -gravityMPerS2, 0.0};
                transferSpec.timestepSeconds = transferDt;
                require(sourceSceneBodies.size() == shippingSceneBodyCount,
                        "gravity-transfer source sidecar has the wrong scene-body count");
                const std::vector<MRBodyStateGPU> transferTools(
                    sourceSceneBodies.begin(),
                    sourceSceneBodies.begin() +
                        static_cast<std::ptrdiff_t>(shippingToolSceneBodyCount));
                run.enableShippingRigidWorld(transferTools, transferSpec);
                const std::uint64_t transferWorldFingerprint =
                    run.shippingRigidWorldFingerprint();
                require(transferWorldFingerprint != 0u &&
                            std::abs(run.shippingRigidFreeBodyMassKg() - mass) <=
                                2.0e-6 * std::max(1.0, mass),
                        "gravity-transfer branch world does not match its declared mass identity");
                shippingCompressionPlatenY =
                    shippingSupportSceneBodies[shippingPlatenBodyIndex].position.y;
                shippingToolCenterX = shippingSupportSceneBodies[shippingFloorBodyIndex].position.x;
                shippingToolCenterZ = shippingSupportSceneBodies[shippingFloorBodyIndex].position.z;
                shippingFloorCenterY = shippingSupportSceneBodies[shippingFloorBodyIndex].position.y;
                const double fullSupport =
                    numi::cardboard::shippingGravityTransferSupportForceY(
                        mass, -gravityMPerS2, 1.0);
                shippingSimulationTimeSeconds = 0.0;
                bool contactObserved = false;
                double actuatorWorkJ = 0.0;
                double gravityWorkJ = 0.0;
                double caseMaximumMomentumResidualNs = 0.0;
                double caseMaximumMomentumResidualNormalized = 0.0;
                const auto recordMomentumBalance = [&](const auto& balance) {
                    const double residualNs = std::sqrt(std::inner_product(
                        balance.residualNs.begin(), balance.residualNs.end(),
                        balance.residualNs.begin(), 0.0));
                    caseMaximumMomentumResidualNs = std::max(
                        caseMaximumMomentumResidualNs, residualNs);
                    caseMaximumMomentumResidualNormalized = std::max(
                        caseMaximumMomentumResidualNormalized,
                        balance.normalizedResidual);
                    shippingGravityTransferMaximumMomentumResidualNs = std::max(
                        shippingGravityTransferMaximumMomentumResidualNs,
                        residualNs);
                    shippingGravityTransferMaximumMomentumResidualNormalized =
                        std::max(
                            shippingGravityTransferMaximumMomentumResidualNormalized,
                            balance.normalizedResidual);
                };
                std::uint32_t acceptedTransferSteps = 0u;
                const auto seedBefore = run.shippingRigidSceneBodies();
                require(seedBefore.size() == shippingSceneBodyCount &&
                            std::abs(seedBefore[shippingFreeBodyIndex].position.y -
                                transferSpec.initialCenterM[1]) <= 2.0e-7,
                        "gravity-transfer seed did not start at its nonpenetrating tangent roof pose");
                const bool seedAccepted = attemptNativeStep(
                    "gravity_transfer_mass_pose_seed",
                    numi::cardboard::kNoBoxScoreLineIndex,
                    zeroTargets(), parkedBodies(), transferDt, true,
                    caseId + "/mass-matched-gravity-seed", fullSupport);
                if (!seedAccepted) {
                    nativeFailure = true;
                    stopReason = "native gravity-transfer zero-displacement seed was rejected for " + caseId;
                    shippingGravityTransferLedger << caseId
                        << ",mass_pose_seed," << (nextNativeStep - 1u)
                        << ",0,0," << transferDt
                        << ",false,1," << fullSupport << ','
                        << fullSupport * transferDt << ','
                        << transferSpec.initialCenterM[1] << ",0,0,0,"
                        << "0,0,0,0,0,0,0,0,0,"
                        << "0," << -mass * gravityMPerS2 * transferDt << ",0,"
                        << "0," << fullSupport * transferDt << ",0,"
                        << "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
                        << -mass * gravityMPerS2 << ",0,0\n";
                    shippingGravityTransferLedger.flush();
                    break;
                }
                currentSnapshot = run.runtime().snapshot();
                require(currentSnapshot.available &&
                            shippingPersistentArenaHashes(currentSnapshot) ==
                                inheritedHashes,
                        "accepted gravity-transfer seed changed the inherited FEM/material/interface state");
                const auto acceptedSeedScene = run.shippingRigidSceneBodies();
                require(acceptedSeedScene.size() == shippingSceneBodyCount &&
                            std::abs(mass * acceptedSeedScene[shippingFreeBodyIndex]
                                .linearVelocityAndInverseMass.w - 1.0) <= 2.0e-5 &&
                            std::abs(acceptedSeedScene[shippingFreeBodyIndex].position.y -
                                transferSpec.initialCenterM[1]) <= 2.0e-6 &&
                            std::abs(acceptedSeedScene[shippingFreeBodyIndex]
                                .linearVelocityAndInverseMass.y) <= 2.0e-6f,
                        "accepted gravity-transfer seed changed its mass, tangent pose, or resting velocity");
                {
                    const auto beforeDynamics =
                        numi::cardboard::observeShippingRigidDynamics(
                            seedBefore[shippingFreeBodyIndex], {0.0, -gravityMPerS2, 0.0},
                            transferSpec.initialCenterM);
                    const auto afterDynamics =
                        numi::cardboard::observeShippingRigidDynamics(
                            acceptedSeedScene[shippingFreeBodyIndex], {0.0, -gravityMPerS2, 0.0},
                            transferSpec.initialCenterM);
                    const std::array<double, 3> zeroDisplacement{
                        static_cast<double>(acceptedSeedScene[shippingFreeBodyIndex].position.x) -
                            seedBefore[shippingFreeBodyIndex].position.x,
                        static_cast<double>(acceptedSeedScene[shippingFreeBodyIndex].position.y) -
                            seedBefore[shippingFreeBodyIndex].position.y,
                        static_cast<double>(acceptedSeedScene[shippingFreeBodyIndex].position.z) -
                            seedBefore[shippingFreeBodyIndex].position.z};
                    const auto seedBalance =
                        numi::cardboard::shippingLinearMomentumBalance(
                            beforeDynamics.linearMomentumKgMPerS,
                            afterDynamics.linearMomentumKgMPerS,
                            {0.0, -gravityMPerS2, 0.0}, mass,
                            {0.0, fullSupport, 0.0},
                            lastImpactorContactImpulseNs, transferDt);
                    recordMomentumBalance(seedBalance);
                    const double seedActuatorWork =
                        numi::cardboard::shippingActuatorWorkJ(
                            {0.0, fullSupport, 0.0}, zeroDisplacement);
                    const double seedGravityWork = mass * -gravityMPerS2 *
                        zeroDisplacement[1];
                    actuatorWorkJ += seedActuatorWork;
                    gravityWorkJ += seedGravityWork;
                    contactObserved = lastImpactorContactSamples > 0.0;
                    shippingGravityTransferLedger << caseId
                        << ",mass_pose_seed," << (nextNativeStep - 1u)
                        << ",0,0," << transferDt << ",true,1,"
                        << fullSupport << ',' << fullSupport * transferDt << ','
                        << acceptedSeedScene[shippingFreeBodyIndex].position.y << ',';
                    appendShippingVector3(shippingGravityTransferLedger,
                        {acceptedSeedScene[shippingFreeBodyIndex].linearVelocityAndInverseMass.x,
                         acceptedSeedScene[shippingFreeBodyIndex].linearVelocityAndInverseMass.y,
                         acceptedSeedScene[shippingFreeBodyIndex].linearVelocityAndInverseMass.z});
                    shippingGravityTransferLedger << ',';
                    appendShippingVector3(shippingGravityTransferLedger,
                        afterDynamics.linearMomentumKgMPerS);
                    shippingGravityTransferLedger << ',';
                    appendShippingVector3(shippingGravityTransferLedger,
                        afterDynamics.angularMomentumKgM2PerS);
                    shippingGravityTransferLedger << ',';
                    appendShippingVector3(shippingGravityTransferLedger,
                        seedBalance.deltaMomentumKgMPerS);
                    shippingGravityTransferLedger << ',';
                    appendShippingVector3(shippingGravityTransferLedger,
                        seedBalance.gravityImpulseNs);
                    shippingGravityTransferLedger << ',';
                    appendShippingVector3(shippingGravityTransferLedger,
                        seedBalance.actuatorImpulseNs);
                    shippingGravityTransferLedger << ',';
                    appendShippingVector3(shippingGravityTransferLedger,
                        lastImpactorContactImpulseNs);
                    shippingGravityTransferLedger << ',';
                    const std::array<double, 3> seedContactForce{
                        lastImpactorContactImpulseNs[0] / transferDt,
                        lastImpactorContactImpulseNs[1] / transferDt,
                        lastImpactorContactImpulseNs[2] / transferDt};
                    appendShippingVector3(shippingGravityTransferLedger,
                        seedContactForce);
                    shippingGravityTransferLedger << ','
                        << lastImpactorContactSamples << ',';
                    appendShippingVector3(shippingGravityTransferLedger,
                        seedBalance.residualNs);
                    shippingGravityTransferLedger << ','
                        << seedBalance.normalizedResidual << ','
                        << seedActuatorWork << ',' << seedGravityWork << ','
                        << afterDynamics.translationalKineticEnergyJ << ','
                        << afterDynamics.rotationalKineticEnergyJ << ','
                        << afterDynamics.gravitationalPotentialRelativeJ << ','
                        << afterDynamics.mechanicalEnergyRelativeJ() << ','
                        << femKineticEnergy(currentSnapshot, nodeCount, 0u) << ','
                        << mass * -gravityMPerS2 << ','
                        << mass * -gravityMPerS2 + fullSupport << ','
                        << (mass * -gravityMPerS2 + fullSupport) * transferDt
                        << '\n';
                    shippingGravityTransferLedger.flush();
                    require(shippingGravityTransferLedger.good(),
                            "failed writing accepted gravity-transfer mass-matched seed row");
                }
                const auto gravitySeedPath =
                    writeShippingAcceptedCheckpoint(
                        arguments.output, currentSnapshot, compiled.world,
                        arguments, "gravity-transfer-" + caseId + "-seed",
                        nextNativeStep,
                        numi::cardboard::kNoBoxScoreLineIndex, dieHeight,
                        shippingSimulationTimeSeconds,
                        "gravity-transfer-from-" + hexDigest(sha256File(
                            shippingCrushAssemblyCheckpointPath)) + "/" + caseId);
                const auto gravitySeedRigidSidecar =
                    writeShippingRigidSceneSidecar(
                        gravitySeedPath, currentSnapshot,
                        transferWorldFingerprint,
                        run.shippingRigidFreeBodyMassKg(),
                        run.shippingRigidGravityMPerS2(),
                        run.shippingRigidSceneBodies(), 0u,
                        lastRigidContactSamples,
                        lastSupportFloorContactSamples,
                        shippingSimulationTimeSeconds, false, false);

                const auto recordTransferStep = [&](
                    const std::string& stage,
                    const std::uint32_t localStep,
                    const double localTime,
                    const double dt,
                    const double supportFraction,
                    const double forceY,
                    const MRBodyStateGPU& before,
                    const bool accepted) {
                    const auto& afterScene = run.shippingRigidSceneBodies();
                    require(afterScene.size() == shippingSceneBodyCount,
                            "gravity-transfer step omitted its dynamic scene body");
                    const auto& body = afterScene[shippingFreeBodyIndex];
                    const auto beforeDynamics =
                        numi::cardboard::observeShippingRigidDynamics(
                            before, {0.0, -gravityMPerS2, 0.0},
                            transferSpec.initialCenterM);
                    const auto dynamics =
                        numi::cardboard::observeShippingRigidDynamics(
                            body, {0.0, -gravityMPerS2, 0.0},
                            transferSpec.initialCenterM);
                    const std::array<double, 3> displacement{
                        static_cast<double>(body.position.x) - before.position.x,
                        static_cast<double>(body.position.y) - before.position.y,
                        static_cast<double>(body.position.z) - before.position.z};
                    const double currentActuatorWork =
                        accepted ? numi::cardboard::shippingActuatorWorkJ(
                            {0.0, forceY, 0.0}, displacement) : 0.0;
                    const double currentGravityWork = accepted
                        ? mass * -gravityMPerS2 * displacement[1] : 0.0;
                    const auto balance =
                        numi::cardboard::shippingLinearMomentumBalance(
                            beforeDynamics.linearMomentumKgMPerS,
                            dynamics.linearMomentumKgMPerS,
                            {0.0, -gravityMPerS2, 0.0}, mass,
                            {0.0, forceY, 0.0},
                            accepted ? lastImpactorContactImpulseNs
                                     : std::array<double, 3>{}, dt);
                    if (accepted) {
                        actuatorWorkJ += currentActuatorWork;
                        gravityWorkJ += currentGravityWork;
                        recordMomentumBalance(balance);
                        contactObserved = contactObserved ||
                            lastImpactorContactSamples > 0.0;
                    }
                    shippingGravityTransferLedger << caseId << ',' << stage << ','
                        << (nextNativeStep == 0u ? 0u : nextNativeStep - 1u)
                        << ',' << localStep << ',' << localTime << ',' << dt << ','
                        << (accepted ? 1 : 0) << ',' << supportFraction << ','
                        << forceY << ',' << forceY * dt << ',' << body.position.y << ','
                        ;
                    appendShippingVector3(shippingGravityTransferLedger,
                        {body.linearVelocityAndInverseMass.x,
                         body.linearVelocityAndInverseMass.y,
                         body.linearVelocityAndInverseMass.z});
                    shippingGravityTransferLedger << ',';
                    appendShippingVector3(shippingGravityTransferLedger,
                        dynamics.linearMomentumKgMPerS);
                    shippingGravityTransferLedger << ',';
                    appendShippingVector3(shippingGravityTransferLedger,
                        dynamics.angularMomentumKgM2PerS);
                    shippingGravityTransferLedger << ',';
                    appendShippingVector3(shippingGravityTransferLedger,
                        balance.deltaMomentumKgMPerS);
                    shippingGravityTransferLedger << ',';
                    appendShippingVector3(shippingGravityTransferLedger,
                        balance.gravityImpulseNs);
                    shippingGravityTransferLedger << ',';
                    appendShippingVector3(shippingGravityTransferLedger,
                        balance.actuatorImpulseNs);
                    shippingGravityTransferLedger << ',';
                    const std::array<double, 3> contactImpulse = accepted
                        ? lastImpactorContactImpulseNs
                        : std::array<double, 3>{};
                    appendShippingVector3(shippingGravityTransferLedger,
                        contactImpulse);
                    shippingGravityTransferLedger << ',';
                    const std::array<double, 3> contactForce{
                        contactImpulse[0] / dt, contactImpulse[1] / dt,
                        contactImpulse[2] / dt};
                    appendShippingVector3(shippingGravityTransferLedger,
                        contactForce);
                    shippingGravityTransferLedger << ','
                        << (accepted ? lastImpactorContactSamples : 0.0) << ',';
                    appendShippingVector3(shippingGravityTransferLedger,
                        balance.residualNs);
                    shippingGravityTransferLedger << ','
                        << balance.normalizedResidual << ','
                        << currentActuatorWork << ',' << currentGravityWork << ','
                        << dynamics.translationalKineticEnergyJ << ','
                        << dynamics.rotationalKineticEnergyJ << ','
                        << dynamics.gravitationalPotentialRelativeJ << ','
                        << dynamics.mechanicalEnergyRelativeJ() << ','
                        << femKineticEnergy(currentSnapshot, nodeCount, 0u) << ','
                        << mass * -gravityMPerS2 << ','
                        << mass * -gravityMPerS2 + forceY << ','
                        << (mass * -gravityMPerS2 + forceY) * dt << '\n';
                    shippingGravityTransferLedger.flush();
                    require(shippingGravityTransferLedger.good(),
                            "failed writing shipping gravity-transfer ledger");
                };
                for (std::uint32_t step = 0u; step < transferSteps; ++step) {
                    const double fraction =
                        numi::cardboard::shippingGravityTransferSupportFraction(
                            step, transferSteps);
                    const double supportForce =
                        numi::cardboard::shippingGravityTransferSupportForceY(
                            mass, -gravityMPerS2, fraction);
                    const auto& scene = run.shippingRigidSceneBodies();
                    const MRBodyStateGPU before = scene.at(4u);
                    const bool accepted = attemptNativeStep(
                        "gravity_weight_transfer",
                        numi::cardboard::kNoBoxScoreLineIndex,
                        zeroTargets(), parkedBodies(), transferDt, true,
                        caseId, supportForce);
                    recordTransferStep("weight_transfer", step,
                        (static_cast<double>(step) + 0.5) * transferDt,
                        transferDt, fraction, supportForce, before, accepted);
                    if (!accepted) {
                        nativeFailure = true;
                        stopReason = "native gravity-weight transfer rejected case " + caseId;
                        break;
                    }
                    ++acceptedTransferSteps;
                }
                if (nativeFailure) break;
                // Release is a separate accepted transaction with an explicit
                // zero-force input, followed by continued full gravity.
                {
                    const auto& scene = run.shippingRigidSceneBodies();
                    const MRBodyStateGPU before = scene.at(4u);
                    const bool accepted = attemptNativeStep(
                        "gravity_support_actuator_release",
                        numi::cardboard::kNoBoxScoreLineIndex,
                        zeroTargets(), parkedBodies(), transferDt, true,
                        caseId + "/zero-support-release", 0.0);
                    recordTransferStep("actuator_release", transferSteps,
                        actualTransferDurationSeconds, transferDt, 0.0,
                        0.0, before, accepted);
                    if (!accepted) {
                        nativeFailure = true;
                        stopReason = "native zero-force actuator release rejected case " + caseId;
                        break;
                    }
                }
                for (std::uint32_t step = 0u;
                     step < releaseObservationSteps && !nativeFailure; ++step) {
                    const auto& scene = run.shippingRigidSceneBodies();
                    const MRBodyStateGPU before = scene.at(4u);
                    const bool accepted = attemptNativeStep(
                        "gravity_transfer_post_release_observation",
                        numi::cardboard::kNoBoxScoreLineIndex,
                        zeroTargets(), parkedBodies(), transferDt, true,
                        caseId + "/post-release", 0.0);
                    recordTransferStep("post_release", step,
                        actualTransferDurationSeconds +
                            transferDt * (static_cast<double>(step) + 1.0),
                        transferDt, 0.0, 0.0, before, accepted);
                    if (!accepted) {
                        nativeFailure = true;
                        stopReason = "native post-release observation rejected case " + caseId;
                        break;
                    }
                }
                if (nativeFailure) break;
                const bool caseComplete =
                    acceptedTransferSteps == transferSteps && contactObserved;
                std::filesystem::path releasedTransferCheckpointPath;
                if (caseComplete) {
                    ++shippingGravityTransferCasesAccepted;
                    releasedTransferCheckpointPath = writeShippingAcceptedCheckpoint(
                        arguments.output, currentSnapshot, compiled.world,
                        arguments, "gravity-transfer-" + caseId + "-released-end",
                        nextNativeStep,
                        numi::cardboard::kNoBoxScoreLineIndex, dieHeight,
                        shippingSimulationTimeSeconds,
                        "gravity-transfer-seed-" +
                            hexDigest(sha256File(gravitySeedPath)) + "/" + caseId);
                }
                std::ofstream transferSummary(arguments.output /
                    ("shipping-gravity-transfer-" + caseId + ".json"));
                require(transferSummary.good(),
                        "cannot create gravity-transfer branch summary");
                transferSummary << std::setprecision(17)
                    << "{\n  \"case_id\": \"" << caseId << "\",\n"
                    << "  \"source_released_assembly_checkpoint_sha256\": \""
                    << hexDigest(sha256File(shippingCrushAssemblyCheckpointPath))
                    << "\",\n"
                    << "  \"source_rigid_scene_sidecar_sha256\": \""
                    << shippingSupportRigidSidecarSha256 << "\",\n"
                    << "  \"mass_pose_seed_checkpoint_file\": \""
                    << jsonEscape(gravitySeedPath.filename().string()) << "\",\n"
                    << "  \"mass_pose_seed_checkpoint_sha256\": \""
                    << hexDigest(sha256File(gravitySeedPath)) << "\",\n"
                    << "  \"mass_pose_seed_rigid_scene_sidecar_file\": \""
                    << jsonEscape(gravitySeedRigidSidecar.binaryPath.filename().string())
                    << "\",\n"
                    << "  \"mass_pose_seed_rigid_scene_sidecar_sha256\": \""
                    << gravitySeedRigidSidecar.binarySha256 << "\",\n"
                    << "  \"branch_rigid_world_fingerprint\": "
                    << transferWorldFingerprint << ",\n"
                    << "  \"branch_initial_center_m\": ["
                    << transferSpec.initialCenterM[0] << ','
                    << transferSpec.initialCenterM[1] << ','
                    << transferSpec.initialCenterM[2] << "],\n"
                    << "  \"released_checkpoint_file\": "
                    << (caseComplete
                        ? "\"" + jsonEscape(releasedTransferCheckpointPath.filename().string()) + "\""
                        : "null") << ",\n"
                    << "  \"released_checkpoint_sha256\": "
                    << (caseComplete
                        ? "\"" + hexDigest(sha256File(releasedTransferCheckpointPath)) + "\""
                        : "null") << ",\n"
                    << "  \"mass_kg\": " << mass << ",\n"
                    << "  \"mass_ratio_fpeak_over_g\": " << ratios[ratioIndex] << ",\n"
                    << "  \"finite_mass_body_gravity_m_s2\": [0,-"
                    << gravityMPerS2 << ",0],\n"
                    << "  \"substrate_self_weight_gravity_m_s2\": ["
                    << compiled.world.dispatch.gravityAndTimestep.x << ','
                    << compiled.world.dispatch.gravityAndTimestep.y << ','
                    << compiled.world.dispatch.gravityAndTimestep.z << "],\n"
                    << "  \"substrate_compiled_nodal_mass_kg\": "
                    << compiledSubstrateMassKg << ",\n"
                    << "  \"substrate_weight_force_n\": [0,"
                    << compiledSubstrateMassKg *
                        compiled.world.dispatch.gravityAndTimestep.y
                    << ",0],\n"
                    << "  \"substrate_gravity_force_law\": \"nodal_mass_times_compiled_gravity_in_FEM_residual\",\n"
                    << "  \"support_ramp_requested_duration_s\": "
                    << transferProtocol.durationSeconds << ",\n"
                    << "  \"support_ramp_actual_duration_s\": "
                    << actualTransferDurationSeconds << ",\n"
                    << "  \"support_ramp_timestep_s\": " << transferDt << ",\n"
                    << "  \"support_ramp_steps_planned\": " << transferSteps << ",\n"
                    << "  \"mass_pose_seed_support_force_n\": " << fullSupport << ",\n"
                    << "  \"ramp_first_force_n\": "
                    << numi::cardboard::shippingGravityTransferSupportForceY(
                        mass, -gravityMPerS2,
                        numi::cardboard::shippingGravityTransferSupportFraction(
                            0u, transferSteps)) << ",\n"
                    << "  \"ramp_last_force_n_before_release\": "
                    << numi::cardboard::shippingGravityTransferSupportForceY(
                        mass, -gravityMPerS2,
                        numi::cardboard::shippingGravityTransferSupportFraction(
                            transferSteps - 1u, transferSteps)) << ",\n"
                    << "  \"release_force_n\": 0,\n"
                    << "  \"post_release_steps\": " << releaseObservationSteps << ",\n"
                    << "  \"post_release_actual_observation_s\": "
                    << actualReleaseObservationSeconds << ",\n"
                    << "  \"initial_roof_gap_m\": " << initialRoofGapM << ",\n"
                    << "  \"tangent_pose_fp32_rounding_bound_m\": "
                    << tangentRoundingBoundM << ",\n"
                    << "  \"accepted_weight_transfer_steps\": "
                    << acceptedTransferSteps << ",\n"
                    << "  \"native_contact_observed\": "
                    << (contactObserved ? "true" : "false") << ",\n"
                    << "  \"actuator_work_j_signed\": " << actuatorWorkJ << ",\n"
                    << "  \"gravity_work_j\": " << gravityWorkJ << ",\n"
                    << "  \"maximum_linear_momentum_balance_residual_ns\": "
                    << caseMaximumMomentumResidualNs << ",\n"
                    << "  \"maximum_normalized_linear_momentum_balance_residual\": "
                    << caseMaximumMomentumResidualNormalized << ",\n"
                    << "  \"accepted_case_complete\": "
                    << (caseComplete ? "true" : "false") << ",\n"
                    << "  \"qualification\": false\n}\n";
                transferSummary.flush();
                require(transferSummary.good(),
                        "failed writing gravity-transfer branch summary");
                shippingGravityTransferLedger.flush();
            }
            shippingGravityTransferCompleted = !nativeFailure &&
                shippingGravityTransferCasesAccepted == ratios.size();
            std::vector<double> impactTimesteps{
                arguments.shippingImpactTimestepSeconds};
            if (arguments.shippingImpactTimestepRefinement)
                impactTimesteps = {0.0005, 0.00025, 0.000125};
            struct ShippingImpactCase {
                std::size_t ratioIndex;
                std::string locationName;
                double offsetZ;
                double timestepSeconds;
            };
            std::vector<ShippingImpactCase> impactCases;
            for (const double timestep : impactTimesteps)
                for (std::size_t ratioIndex = 0u;
                     ratioIndex < ratios.size(); ++ratioIndex)
                    for (const auto& [locationName, offsetZ] : locations)
                        impactCases.push_back({ratioIndex, locationName,
                                               offsetZ, timestep});
            const std::uint32_t expectedImpactCaseCount =
                static_cast<std::uint32_t>(impactCases.size());
            shippingDropCasesPlanned = expectedImpactCaseCount;
            for (const auto& impactCase : impactCases) {
                if (nativeFailure) break;
                const std::size_t ratioIndex = impactCase.ratioIndex;
                const std::string& locationName = impactCase.locationName;
                const double offsetZ = impactCase.offsetZ;
                const double impactTimestepSeconds = impactCase.timestepSeconds;
                const std::uint32_t impactSteps = static_cast<std::uint32_t>(
                    std::ceil(impactDurationSeconds / impactTimestepSeconds));
                const double mass = numi::cardboard::shippingImpactMassFromPeakForce(
                    shippingCompressionPeakForceN, ratios[ratioIndex],
                    gravityMPerS2);
                const std::string caseId = "ratio-" +
                        std::to_string(static_cast<unsigned>(ratioIndex + 1u)) +
                        "-" + locationName + "-dt-" +
                        std::to_string(static_cast<unsigned long long>(
                            std::llround(impactTimestepSeconds * 1.0e6))) +
                        "us";
                    currentSnapshot =
                        numi::cardboard::readShippingAcceptedSnapshot(
                            shippingCrushAssemblyCheckpointPath,
                            compiled.world.physicsFingerprint,
                            compiled.world.fingerprint);
                    const auto sourceSceneBodies = readShippingRigidSceneBinary(
                        shippingSupportRigidSidecarPath,
                        shippingCrushAssemblyCheckpointPath,
                        currentSnapshot.sourcePhysicsFingerprint,
                        currentSnapshot.deviceProgramFingerprint,
                        shippingSupportRigidWorldFingerprint,
                        shippingSupportRigidMassKg,
                        static_cast<std::uint32_t>(shippingSupportSceneBodies.size()),
                        currentSnapshot.controlStep, false);
                    require(std::equal(sourceSceneBodies.begin(), sourceSceneBodies.end(),
                                shippingSupportSceneBodies.begin(),
                                sameShippingSceneBodyState),
                            "impact branch source sidecar differs from released support-ready state");
                    const auto inheritedAssemblyArenaHashes =
                        shippingPersistentArenaHashes(currentSnapshot);
                    require(currentSnapshot.rigidStates.size() ==
                                compiled.world.contact.rigidProxies.size() *
                                    arguments.environmentCount &&
                                currentSnapshot.rigidGeneralizedCandidate.size() ==
                                    compiled.world.dispatch.rigidGeneralizedCapacity *
                                        arguments.environmentCount,
                            "shipping impact source checkpoint has incomplete rigid state");
                    // The released assembly checkpoint carries its original
                    // parked 1 kg projection. Do not reuse that rigid history
                    // for a different branch mass: reset only the rigid
                    // projection to Matter's all-zero uninitialized sentinel,
                    // while preserving the accepted FEM/material/interface
                    // histories. The first accepted zero-rigid-body-gravity
                    // transaction below initializes every proxy from this
                    // branch's exact scene bodies and mass/inertia; the FEM
                    // substrate gravity remains the compiled shipping value.
                    std::fill(currentSnapshot.rigidStates.begin(),
                              currentSnapshot.rigidStates.end(), NMRigidStateGPU{});
                    std::fill(currentSnapshot.rigidGeneralizedCandidate.begin(),
                              currentSnapshot.rigidGeneralizedCandidate.end(), 0.0f);
                    std::fill(currentSnapshot.reactions.begin(),
                              currentSnapshot.reactions.end(),
                              NMRigidReactionGPU{});
                    run.restoreShippingCheckpoint(currentSnapshot);
                    currentSnapshot = run.runtime().snapshot();
                    require(currentSnapshot.available,
                            "restored shipping impact checkpoint is unavailable");
                    const auto preSeedArenaHashes =
                        shippingPersistentArenaHashes(currentSnapshot);
                    require(preSeedArenaHashes == inheritedAssemblyArenaHashes,
                            "impact branch restore changed inherited FEM/history arenas before zero-rigid-body-gravity initialization");
                    shippingCompressionPlatenY = 1.0;
                    const auto startCenter =
                        numi::cardboard::shippingImpactStartCenter(
                            impactorFootprint, impactorHalfHeightM,
                            dropHeightM, offsetZ);
                    numi::cardboard::ShippingRigidBodySpec impactSpec;
                    impactSpec.massKg = mass;
                    impactSpec.halfExtentM = arguments.shippingImpactorHalfExtentsM;
                    impactSpec.initialCenterM = startCenter;
                    impactSpec.gravityMPerS2 = {0.0, 0.0, 0.0};
                    impactSpec.timestepSeconds = impactTimestepSeconds;
                    require(sourceSceneBodies.size() == shippingSceneBodyCount,
                            "shipping impact branch lost a scene body");
                    const std::vector<MRBodyStateGPU> seedToolVector(
                        sourceSceneBodies.begin(),
                        sourceSceneBodies.begin() +
                            static_cast<std::ptrdiff_t>(shippingToolSceneBodyCount));
                    // Each mass/location branch first establishes an accepted
                    // Matter rigid projection with its own exact body mass,
                    // inertia, pose, and zero velocity. Rigid-body gravity is
                    // enabled only after this mass-matched rigid-body seed has been
                    // checkpointed. The FEM substrate keeps its compiled
                    // shipping gravity throughout both transactions.
                    run.enableShippingRigidWorld(seedToolVector, impactSpec);
                    shippingSimulationTimeSeconds = 0.0;
                    const bool seedAccepted = attemptNativeStep(
                        "impact_mass_pose_zero_rigid_body_gravity_initialization",
                        numi::cardboard::kNoBoxScoreLineIndex,
                        zeroTargets(), seedToolVector,
                        impactTimestepSeconds, true, caseId + "/mass-pose-seed");
                    if (!seedAccepted) {
                        nativeFailure = true;
                        stopReason = "native zero-rigid-body-gravity mass-matched seed rejected for " +
                            caseId;
                        shippingImpactLedger << caseId << ',' << ratios[ratioIndex]
                            << ',' << mass << ',' << locationName << ',' << offsetZ
                            << ',' << dropHeightM << ',' << mass * gravityMPerS2 *
                                dropHeightM << ','
                            << hexDigest(sha256File(shippingCrushAssemblyCheckpointPath))
                            << ",,0,,0,0,mass_matched_seed_rejected\n";
                        shippingImpactLedger.flush();
                        break;
                    }
                    const std::vector<MRBodyStateGPU> acceptedSeedScene =
                        run.shippingRigidSceneBodies();
                    require(acceptedSeedScene.size() == shippingSceneBodyCount,
                            "accepted impact seed omitted its finite-mass scene body");
                    const auto& acceptedSeedBody = acceptedSeedScene[shippingFreeBodyIndex];
                    const double inertiaX = mass * (
                        impactSpec.halfExtentM[1] * impactSpec.halfExtentM[1] +
                        impactSpec.halfExtentM[2] * impactSpec.halfExtentM[2]) / 3.0;
                    const double inertiaY = mass * (
                        impactSpec.halfExtentM[0] * impactSpec.halfExtentM[0] +
                        impactSpec.halfExtentM[2] * impactSpec.halfExtentM[2]) / 3.0;
                    const double inertiaZ = mass * (
                        impactSpec.halfExtentM[0] * impactSpec.halfExtentM[0] +
                        impactSpec.halfExtentM[1] * impactSpec.halfExtentM[1]) / 3.0;
                    require(std::abs(mass *
                                acceptedSeedBody.linearVelocityAndInverseMass.w - 1.0) <=
                                2.0e-5 &&
                                std::abs(acceptedSeedBody.orientation.x) < 1.0e-6f &&
                                std::abs(acceptedSeedBody.orientation.y) < 1.0e-6f &&
                                std::abs(acceptedSeedBody.orientation.z) < 1.0e-6f &&
                                std::abs(acceptedSeedBody.orientation.w - 1.0f) < 1.0e-6f &&
                                std::abs(acceptedSeedBody.inverseInertiaWorldRow0.x * inertiaX - 1.0) <= 3.0e-5 &&
                                std::abs(acceptedSeedBody.inverseInertiaWorldRow1.y * inertiaY - 1.0) <= 3.0e-5 &&
                                std::abs(acceptedSeedBody.inverseInertiaWorldRow2.z * inertiaZ - 1.0) <= 3.0e-5,
                            "accepted impact seed mass or inertia differs from its branch properties");
                    for (std::size_t axis = 0u; axis < 3u; ++axis) {
                        const std::array<float, 3> position{
                            acceptedSeedBody.position.x,
                            acceptedSeedBody.position.y,
                            acceptedSeedBody.position.z};
                        const double expected = startCenter[axis];
                        const double tolerance = 8.0 *
                            std::numeric_limits<float>::epsilon() *
                            std::max(1.0, std::abs(expected));
                        require(std::abs(static_cast<double>(position[axis]) -
                                             expected) <= tolerance,
                                "zero-rigid-body-gravity impact seed changed the prescribed start pose");
                    }
                    require(std::abs(acceptedSeedBody.linearVelocityAndInverseMass.x) < 1.0e-7f &&
                                std::abs(acceptedSeedBody.linearVelocityAndInverseMass.y) < 1.0e-7f &&
                                std::abs(acceptedSeedBody.linearVelocityAndInverseMass.z) < 1.0e-7f &&
                                std::abs(acceptedSeedBody.angularVelocity.x) < 1.0e-7f &&
                                std::abs(acceptedSeedBody.angularVelocity.y) < 1.0e-7f &&
                                std::abs(acceptedSeedBody.angularVelocity.z) < 1.0e-7f,
                            "zero-rigid-body-gravity impact seed acquired a nonzero velocity");
                    require(lastImpactorContactSamples == 0.0,
                            "impact seed is not contact-free at its prescribed start pose");
                    require(currentSnapshot.rigidStates.size() ==
                                compiled.world.contact.rigidProxies.size() *
                                    arguments.environmentCount &&
                                currentSnapshot.rigidGeneralizedCandidate.size() ==
                                    compiled.world.dispatch.rigidGeneralizedCapacity *
                                        arguments.environmentCount,
                            "accepted impact seed lost a rigid or generalized state arena");
                    const auto postSeedArenaHashes =
                        shippingPersistentArenaHashes(currentSnapshot);
                    require(postSeedArenaHashes == inheritedAssemblyArenaHashes,
                            "accepted zero-rigid-body-gravity initialization changed inherited FEM/history arena bytes");
                    std::size_t impactProxyIndex =
                        std::numeric_limits<std::size_t>::max();
                    std::uint32_t impactProxyMatches = 0u;
                    for (std::size_t proxyIndex = 0u;
                         proxyIndex < compiled.world.contact.rigidProxies.size();
                         ++proxyIndex) {
                        const auto& proxy =
                            compiled.world.contact.rigidProxies[proxyIndex];
                        if (proxy.bodyIndex == 4u &&
                            (proxy.flags & NM_RIGID_DYNAMIC) != 0u) {
                            impactProxyIndex = proxyIndex;
                            ++impactProxyMatches;
                        }
                    }
                    require(impactProxyMatches == 1u &&
                                arguments.environmentCount == 1u &&
                                impactProxyIndex < currentSnapshot.rigidStates.size(),
                            "accepted impact seed must have one auditable native free-body proxy");
                    const auto projectionError =
                        numi::cardboard::compareShippingImpactProxyProjection(
                            compiled.world.contact.rigidProxies[impactProxyIndex],
                            currentSnapshot.rigidStates[impactProxyIndex],
                            acceptedSeedBody);
                    const double proxyProjectionTolerance = 8.0 *
                        std::numeric_limits<float>::epsilon() *
                        std::max({1.0, std::abs(static_cast<double>(mass)),
                            std::abs(static_cast<double>(acceptedSeedBody.position.x)),
                            std::abs(static_cast<double>(acceptedSeedBody.position.y)),
                            std::abs(static_cast<double>(acceptedSeedBody.position.z)),
                            std::abs(static_cast<double>(acceptedSeedBody.inverseInertiaWorldRow0.x)),
                            std::abs(static_cast<double>(acceptedSeedBody.inverseInertiaWorldRow1.y)),
                            std::abs(static_cast<double>(acceptedSeedBody.inverseInertiaWorldRow2.z))});
                    require(std::isfinite(projectionError.maximum()) &&
                                projectionError.maximum() <= proxyProjectionTolerance,
                            "accepted Matter rigid-proxy snapshot disagrees with its mass-matched scene body");
                    double maximumGeneralizedMagnitude = 0.0;
                    for (const float value :
                         currentSnapshot.rigidGeneralizedCandidate) {
                        require(std::isfinite(value),
                                "accepted impact seed generalized state is nonfinite");
                        maximumGeneralizedMagnitude = std::max(
                            maximumGeneralizedMagnitude,
                            std::abs(static_cast<double>(value)));
                    }
                    require(maximumGeneralizedMagnitude <= 1.0e-7,
                            "zero-rigid-body-gravity accepted impact seed unexpectedly changed its free generalized velocity");
                    const auto impactSeedCheckpointPath =
                        writeShippingAcceptedCheckpoint(
                            arguments.output, currentSnapshot, compiled.world,
                            arguments,
                            "impact-" + caseId + "-mass-matched-zero-rigid-body-gravity-seed",
                            nextNativeStep,
                            numi::cardboard::kNoBoxScoreLineIndex, dieHeight,
                            shippingSimulationTimeSeconds,
                            "impact-seed/" + caseId + "/mass-kg-" +
                                std::to_string(mass));
                    const auto impactSeedRigidSidecar =
                        writeShippingRigidSceneSidecar(
                            impactSeedCheckpointPath, currentSnapshot,
                            run.shippingRigidWorldFingerprint(),
                            run.shippingRigidFreeBodyMassKg(),
                            run.shippingRigidGravityMPerS2(),
                            run.shippingRigidSceneBodies(), 0u,
                            lastRigidContactSamples,
                            lastSupportFloorContactSamples,
                            shippingSimulationTimeSeconds, false);
                    const auto seedSidecarPath = arguments.output / "checkpoints" /
                        ("impact-" + caseId + "-rigid-matched-seed.json");
                    std::ofstream seedSidecar(seedSidecarPath);
                    require(seedSidecar.good(),
                            "cannot create mass-matched impact seed sidecar");
                    seedSidecar << std::setprecision(17)
                        << "{\n  \"schema\": \"numi.cardboard.shipping-impact-seed.v1\",\n"
                        << "  \"case_id\": \"" << caseId << "\",\n"
                        << "  \"matter_checkpoint_file\": \""
                        << jsonEscape(impactSeedCheckpointPath.filename().string())
                        << "\",\n  \"matter_checkpoint_sha256\": \""
                        << hexDigest(sha256File(impactSeedCheckpointPath))
                        << "\",\n  \"released_assembly_checkpoint_sha256\": \""
                        << hexDigest(sha256File(shippingCrushAssemblyCheckpointPath))
                        << "\",\n  \"released_assembly_rigid_sidecar_sha256\": \""
                        << shippingSupportRigidSidecarSha256
                        << "\",\n  \"branch_mass_kg\": " << mass
                        << ",\n  \"branch_mass_ratio_fpeak_over_g\": "
                        << ratios[ratioIndex]
                        << ",\n  \"branch_initial_center_m\": ["
                        << startCenter[0] << ',' << startCenter[1] << ','
                        << startCenter[2] << "],\n  \"branch_half_extent_m\": ["
                        << impactSpec.halfExtentM[0] << ','
                        << impactSpec.halfExtentM[1] << ','
                        << impactSpec.halfExtentM[2]
                        << "],\n  \"branch_initial_linear_velocity_m_s\": [0,0,0]"
                        << ",\n  \"mass_pose_seed_rigid_scene_sidecar_file\": \""
                        << jsonEscape(impactSeedRigidSidecar.binaryPath.filename().string())
                        << "\",\n  \"mass_pose_seed_rigid_scene_sidecar_sha256\": \""
                        << impactSeedRigidSidecar.binarySha256
                        << "\",\n  \"native_step_index\": " << nextNativeStep - 1u
                        << ",\n  \"source_physics_fingerprint\": "
                        << currentSnapshot.sourcePhysicsFingerprint
                        << ",\n  \"device_program_fingerprint\": "
                        << currentSnapshot.deviceProgramFingerprint
                        << ",\n  \"rigid_world_fingerprint_zero_rigid_body_gravity\": "
                        << run.shippingRigidWorldFingerprint()
                        << ",\n  \"rigid_projection_reinitialized_from_source_checkpoint\": true"
                        << ",\n  \"inherited_state_arena_hashes_before_zero_rigid_body_gravity_initialization\": ";
                    writeShippingArenaHashes(seedSidecar, preSeedArenaHashes);
                    seedSidecar << ",\n  \"inherited_state_arena_hashes_after_zero_rigid_body_gravity_initialization\": ";
                    writeShippingArenaHashes(seedSidecar, postSeedArenaHashes);
                    seedSidecar << ",\n  \"inherited_fem_history_bytes_unchanged\": true"
                        << ",\n  \"matter_rigid_projection_scene_match\": {\n"
                        << "    \"proxy_index\": " << impactProxyIndex << ",\n"
                        << "    \"projection_tolerance_max_abs\": "
                        << proxyProjectionTolerance << ",\n"
                        << "    \"max_abs_error\": " << projectionError.maximum() << ",\n"
                        << "    \"center_and_radius_error\": "
                        << projectionError.centerAndRadius << ",\n"
                        << "    \"extent_error\": " << projectionError.extent << ",\n"
                        << "    \"orientation_error\": " << projectionError.orientation << ",\n"
                        << "    \"linear_velocity_inverse_mass_error\": "
                        << projectionError.linearVelocityAndInverseMass << ",\n"
                        << "    \"angular_velocity_error\": "
                        << projectionError.angularVelocity << ",\n"
                        << "    \"body_center_error\": " << projectionError.bodyCenter << ",\n"
                        << "    \"inverse_inertia_rows_error\": "
                        << projectionError.inverseInertiaRows << ",\n"
                        << "    \"pass\": true\n  }"
                        << ",\n  \"rigid_generalized_coordinates\": {\"nq\":0,\"nv\":0}"
                        << ",\n  \"rigid_generalized_candidate_max_abs\": "
                        << maximumGeneralizedMagnitude
                        << ",\n  \"rigid_mass_kg\": " << mass
                        << ",\n  \"mass_ratio_fpeak_over_g\": " << ratios[ratioIndex]
                        << ",\n  \"half_extent_m\": ["
                        << impactSpec.halfExtentM[0] << ','
                        << impactSpec.halfExtentM[1] << ','
                        << impactSpec.halfExtentM[2]
                        << "],\n  \"inertia_diagonal_kg_m2\": ["
                        << inertiaX << ',' << inertiaY << ',' << inertiaZ
                        << "],\n  \"accepted_center_m\": ["
                        << acceptedSeedBody.position.x << ','
                        << acceptedSeedBody.position.y << ','
                        << acceptedSeedBody.position.z
                        << "],\n  \"accepted_linear_velocity_m_s\": ["
                        << acceptedSeedBody.linearVelocityAndInverseMass.x << ','
                        << acceptedSeedBody.linearVelocityAndInverseMass.y << ','
                        << acceptedSeedBody.linearVelocityAndInverseMass.z
                        << "],\n  \"rigid_body_initialization_gravity_m_s2\": [0,0,0],\n"
                        << "  \"rigid_body_subsequent_drop_gravity_m_s2\": [0,-"
                        << gravityMPerS2 << ",0],\n"
                        << "  \"substrate_self_weight_gravity_m_s2\": ["
                        << compiled.world.dispatch.gravityAndTimestep.x << ','
                        << compiled.world.dispatch.gravityAndTimestep.y << ','
                        << compiled.world.dispatch.gravityAndTimestep.z << "],\n"
                        << "  \"substrate_compiled_nodal_mass_kg\": "
                        << compiledSubstrateMassKg << ",\n"
                        << "  \"substrate_weight_force_n\": [0,"
                        << compiledSubstrateMassKg *
                            compiled.world.dispatch.gravityAndTimestep.y
                        << ",0],\n"
                        << "  \"scene_bodies\": [\n";
                    for (std::size_t bodyIndex = 0u;
                         bodyIndex < acceptedSeedScene.size(); ++bodyIndex) {
                        const auto& body = acceptedSeedScene[bodyIndex];
                        seedSidecar << "    {\"index\":" << bodyIndex
                            << ",\"position_m\":[" << body.position.x << ','
                            << body.position.y << ',' << body.position.z
                            << "],\"orientation_xyzw\":[" << body.orientation.x << ','
                            << body.orientation.y << ',' << body.orientation.z << ','
                            << body.orientation.w
                            << "],\"linear_velocity_inverse_mass\":["
                            << body.linearVelocityAndInverseMass.x << ','
                            << body.linearVelocityAndInverseMass.y << ','
                            << body.linearVelocityAndInverseMass.z << ','
                            << body.linearVelocityAndInverseMass.w
                            << "],\"angular_velocity\":["
                            << body.angularVelocity.x << ',' << body.angularVelocity.y << ','
                            << body.angularVelocity.z << ',' << body.angularVelocity.w
                            << "],\"inverse_inertia_world_rows\":[["
                            << body.inverseInertiaWorldRow0.x << ','
                            << body.inverseInertiaWorldRow0.y << ','
                            << body.inverseInertiaWorldRow0.z << ','
                            << body.inverseInertiaWorldRow0.w << "],["
                            << body.inverseInertiaWorldRow1.x << ','
                            << body.inverseInertiaWorldRow1.y << ','
                            << body.inverseInertiaWorldRow1.z << ','
                            << body.inverseInertiaWorldRow1.w << "],["
                            << body.inverseInertiaWorldRow2.x << ','
                            << body.inverseInertiaWorldRow2.y << ','
                            << body.inverseInertiaWorldRow2.z << ','
                            << body.inverseInertiaWorldRow2.w << "]],\"flags_and_indices\":["
                            << body.flagsAndIndices[0] << ','
                            << body.flagsAndIndices[1] << ','
                            << body.flagsAndIndices[2] << ','
                            << body.flagsAndIndices[3] << "]}"
                            << (bodyIndex + 1u == acceptedSeedScene.size()
                                ? "\n" : ",\n");
                    }
                    seedSidecar << "  ],\n  \"matter_rigid_proxy_state_count\": "
                        << currentSnapshot.rigidStates.size()
                        << ",\n  \"matter_rigid_proxy_state_sha256\": \""
                        << sha256Arena(currentSnapshot.rigidStates)
                        << "\",\n  \"matter_rigid_proxy_states\": [\n";
                    const auto writeSeedState4 = [&seedSidecar](
                        const char* key, const nm_float4& value) {
                        require(std::isfinite(value.x) && std::isfinite(value.y) &&
                                    std::isfinite(value.z) && std::isfinite(value.w),
                                "accepted rigid-proxy snapshot contains a nonfinite field");
                        seedSidecar << "\"" << key << "\":[" << value.x << ','
                            << value.y << ',' << value.z << ',' << value.w << ']';
                    };
                    for (std::size_t index = 0u;
                         index < currentSnapshot.rigidStates.size(); ++index) {
                        const auto& state = currentSnapshot.rigidStates[index];
                        seedSidecar << "    {";
                        writeSeedState4("center_and_radius", state.centerAndRadius);
                        seedSidecar << ',';
                        writeSeedState4("extent", state.extent);
                        seedSidecar << ',';
                        writeSeedState4("orientation", state.orientation);
                        seedSidecar << ',';
                        writeSeedState4("linear_velocity_inverse_mass",
                            state.linearVelocityAndInverseMass);
                        seedSidecar << ',';
                        writeSeedState4("angular_velocity", state.angularVelocity);
                        seedSidecar << ',';
                        writeSeedState4("body_center", state.bodyCenter);
                        seedSidecar << ',';
                        writeSeedState4("inverse_inertia_row_0",
                            state.inverseInertiaRow0);
                        seedSidecar << ',';
                        writeSeedState4("inverse_inertia_row_1",
                            state.inverseInertiaRow1);
                        seedSidecar << ',';
                        writeSeedState4("inverse_inertia_row_2",
                            state.inverseInertiaRow2);
                        seedSidecar << "}"
                            << (index + 1u == currentSnapshot.rigidStates.size()
                                ? "\n" : ",\n");
                    }
                    seedSidecar << "  ],\n  \"matter_rigid_generalized_candidate_count\": "
                        << currentSnapshot.rigidGeneralizedCandidate.size()
                        << ",\n  \"matter_rigid_generalized_candidate_sha256\": \""
                        << sha256Arena(currentSnapshot.rigidGeneralizedCandidate)
                        << "\",\n  \"matter_rigid_generalized_candidate\": [";
                    for (std::size_t index = 0u;
                         index < currentSnapshot.rigidGeneralizedCandidate.size();
                         ++index)
                        seedSidecar << (index == 0u ? "" : ",")
                            << currentSnapshot.rigidGeneralizedCandidate[index];
                    seedSidecar << "],\n  \"scene_bodies_sha256\": \""
                        << sha256Arena(acceptedSeedScene) << "\",\n"
                        << "  \"rigid_state_arenas_complete\": true";

                    // The gravity-on world is a new MetalWorld branch with the
                    // same body properties. Restore the accepted body state
                    // projected under this branch's mass and inertia before
                    // beginning the 100 mm drop.
                    impactSpec.gravityMPerS2 = {0.0, -gravityMPerS2, 0.0};
                    run.enableShippingRigidWorld(seedToolVector, impactSpec);
                    run.restoreShippingRigidSceneBodies(acceptedSeedScene);
                    const auto gravityRestoredScene =
                        run.shippingRigidSceneBodies();
                    const auto appliedGravity = run.shippingRigidGravityMPerS2();
                    require(gravityRestoredScene.size() == acceptedSeedScene.size() &&
                                std::equal(gravityRestoredScene.begin(),
                                    gravityRestoredScene.end(),
                                    acceptedSeedScene.begin(),
                                    sameShippingSceneBodyState) &&
                                std::abs(run.shippingRigidFreeBodyMassKg() - mass) <=
                                2.0e-6 * std::max(1.0, mass) &&
                                std::abs(appliedGravity[1] + gravityMPerS2) <=
                                    2.0e-6 * gravityMPerS2,
                            "gravity-on impact branch did not restore the exact accepted seed bodies, mass, and gravity");
                    seedSidecar << ",\n  \"gravity_on_world_fingerprint\": "
                        << run.shippingRigidWorldFingerprint()
                        << ",\n  \"gravity_on_mass_kg\": "
                        << run.shippingRigidFreeBodyMassKg()
                        << ",\n  \"gravity_branch_restored_all_scene_body_fields_exactly\": true"
                        << ",\n  \"matter_checkpoint_contains_all_native_state_arenas\": true,\n"
                        << "  \"qualification\": false\n}\n";
                    seedSidecar.flush();
                    require(seedSidecar.good(),
                            "failed writing mass-matched impact seed sidecar");
                    seedSidecar.close();
                    require(!seedSidecar.fail(),
                            "failed closing mass-matched impact seed sidecar");
                    shippingSimulationTimeSeconds = 0.0;
                    std::uint32_t caseAcceptedSteps = 0u;
                    std::uint32_t firstContactStep = NM_INVALID_INDEX;
                    double maximumForceN = 0.0;
                    double maximumSpeedMPerS = 0.0;
                    bool caseAccepted = true;
                    for (std::uint32_t step = 0u; step < impactSteps; ++step) {
                        if (!attemptNativeStep("finite_mass_gravity_drop",
                                numi::cardboard::kNoBoxScoreLineIndex,
                                zeroTargets(), parkedBodies(),
                                impactTimestepSeconds, true, caseId)) {
                            caseAccepted = false;
                            nativeFailure = true;
                            stopReason = "native finite-mass drop rejected case " +
                                caseId + " at its first rejected step";
                            break;
                        }
                        ++caseAcceptedSteps;
                        if (firstContactStep == NM_INVALID_INDEX &&
                            lastImpactorContactSamples > 0.0) {
                            firstContactStep = nextNativeStep - 1u;
                            (void)writeShippingAcceptedCheckpoint(
                                arguments.output, currentSnapshot, compiled.world,
                                arguments, "impact-" + caseId + "-first-contact",
                                nextNativeStep,
                                numi::cardboard::kNoBoxScoreLineIndex, dieHeight,
                                shippingSimulationTimeSeconds,
                                "drop-from-" + hexDigest(sha256File(
                                    shippingCrushAssemblyCheckpointPath)) + "/" +
                                    caseId);
                        }
                        maximumForceN = std::max(
                            maximumForceN, std::abs(lastImpactorMeanForceY));
                        const auto& sceneBodies = run.shippingRigidSceneBodies();
                        if (sceneBodies.size() > shippingFreeBodyIndex) {
                            const auto& body = sceneBodies[shippingFreeBodyIndex];
                            const auto& velocity = body.linearVelocityAndInverseMass;
                            const double speed = std::sqrt(
                                static_cast<double>(velocity.x) * velocity.x +
                                static_cast<double>(velocity.y) * velocity.y +
                                static_cast<double>(velocity.z) * velocity.z);
                            maximumSpeedMPerS = std::max(
                                maximumSpeedMPerS, speed);
                            const auto dynamics =
                                numi::cardboard::observeShippingRigidDynamics(
                                    body, {0.0, -gravityMPerS2, 0.0},
                                    startCenter);
                            require(std::abs(dynamics.massKg - mass) <=
                                        2.0e-6 * std::max(1.0, mass),
                                    "impact dynamics ledger observed a mass mismatch");
                            const auto& linear = body.linearVelocityAndInverseMass;
                            const auto& angular = body.angularVelocity;
                            const auto& momentum = dynamics.linearMomentumKgMPerS;
                            const auto& angularMomentum =
                                dynamics.angularMomentumKgM2PerS;
                            shippingImpactDynamicsLedger
                                << caseId << ',' << (nextNativeStep - 1u) << ','
                                << caseAcceptedSteps << ','
                                << caseAcceptedSteps * impactTimestepSeconds << ','
                                << impactTimestepSeconds << ",true,"
                                << body.position.y << ',' << linear.x << ','
                                << linear.y << ',' << linear.z << ','
                                << angular.x << ',' << angular.y << ','
                                << angular.z << ','
                                << momentum[0] << ',' << momentum[1] << ','
                                << momentum[2] << ',' << angularMomentum[0] << ','
                                << angularMomentum[1] << ',' << angularMomentum[2] << ','
                                << dynamics.translationalKineticEnergyJ << ','
                                << dynamics.rotationalKineticEnergyJ << ','
                                << dynamics.gravitationalPotentialRelativeJ << ','
                                << dynamics.mechanicalEnergyRelativeJ() << ','
                                << femKineticEnergy(currentSnapshot, nodeCount, 0u)
                                << ',' << lastImpactorContactImpulseNs[0] << ','
                                << lastImpactorContactImpulseNs[1] << ','
                                << lastImpactorContactImpulseNs[2] << ','
                                << lastImpactorMeanForceY << ','
                                << lastImpactorContactSamples << ','
                                << lastRigidContactSamples << '\n';
                            shippingImpactDynamicsLedger.flush();
                            require(shippingImpactDynamicsLedger.good(),
                                    "failed writing accepted shipping impact dynamics row");
                        }
                    }
                    const bool contactAccepted =
                        firstContactStep != NM_INVALID_INDEX;
                    const bool caseComplete = caseAccepted &&
                        caseAcceptedSteps == impactSteps && contactAccepted;
                    if (caseComplete) {
                        (void)writeShippingAcceptedCheckpoint(
                            arguments.output, currentSnapshot, compiled.world,
                            arguments, "impact-" + caseId + "-accepted-end",
                            nextNativeStep,
                            numi::cardboard::kNoBoxScoreLineIndex, dieHeight,
                            shippingSimulationTimeSeconds,
                            "drop-from-" + hexDigest(sha256File(
                                shippingCrushAssemblyCheckpointPath)) + "/" +
                                caseId);
                    }
                    shippingImpactLedger << caseId << ',' << ratios[ratioIndex]
                        << ',' << mass << ',' << locationName << ',' << offsetZ
                        << ',' << dropHeightM << ',' << impactTimestepSeconds << ','
                        << impactSteps << ',' << mass * gravityMPerS2 *
                            dropHeightM << ','
                        << hexDigest(sha256File(shippingCrushAssemblyCheckpointPath))
                        << ',' << seedSidecarPath.filename().string()
                        << ',' << caseAcceptedSteps << ',';
                    if (contactAccepted)
                        shippingImpactLedger << firstContactStep;
                    shippingImpactLedger << ',' << maximumForceN << ','
                        << maximumSpeedMPerS << ','
                        << (caseComplete ? "accepted_native_contact_complete" :
                            contactAccepted ? "accepted_contact_incomplete" :
                                               "no_native_impactor_contact")
                        << '\n';
                    shippingImpactLedger.flush();
                    if (caseComplete) ++shippingDropCasesAccepted;
                    else if (!nativeFailure) {
                        nativeFailure = true;
                        stopReason = "finite-mass impact case " + caseId +
                            " did not reach an accepted contact within its 250 ms observation window";
                    }
                if (nativeFailure) break;
            }
            shippingDropCasesCompleted = !nativeFailure &&
                shippingDropCasesAccepted == expectedImpactCaseCount;
        } else if (shippingCompressionCompleted && !nativeFailure) {
            stopReason = shippingCompressionNumericallyQualified
                ? "slow compression has no usable positive measured native platen Fpeak; no impact masses were instantiated"
                : "slow compression histories completed but no source-bound mesh/time numerical-capacity qualification was admitted; no impact or gravity-transfer masses were instantiated";
        }
    }
#else
    if (arguments.shippingBox && fullShippingActionSequence &&
        releasedAssemblySettled && !nativeFailure) {
        nativeFailure = true;
        stopReason = "shipping compression/drop requires the integrated MetalWorld finite-mass rigid bridge";
    }
#endif

    writeBoxActionCsv(arguments.output / "box-actions.csv", blank, actionRows);
    writeObj(arguments.output / "box-native-final-accepted-env0.obj",
             compiled.world, currentSnapshot.femNodes, 0u);
    if (arguments.environmentCount > 1u)
        writeObj(arguments.output / "box-native-final-reference-env1.obj",
                 compiled.world, currentSnapshot.femNodes, 1u);
    std::ofstream result(arguments.output / "box-native-result.json");
    if (arguments.shippingBox && !springbackRowsWritten)
        writeSpringbackRows(false, "release_protocol_not_reached");
    const std::string finalStatus = shapeFitRejection ? "shape_fit_rejected" :
        nativeFailure ? "failed_native" :
        blockedByBondLaw ? "blocked_missing_cohesive_law" :
        stageCutoffReached ? "stage_cutoff_held_boundary_state" :
        (arguments.shippingBox && fullShippingActionSequence)
            ? (shippingDropCasesCompleted && shippingGravityTransferCompleted
                ? "completed_native_gravity_transfer_compression_and_drop_unqualified"
                : shippingDropCasesCompleted
                ? "completed_native_drop_but_gravity_transfer_incomplete_unqualified"
                : shippingGravityTransferCompleted
                ? "completed_native_gravity_transfer_and_compression_only_unqualified"
                : shippingCompressionCompleted
                ? "completed_native_compression_only_unqualified"
                : shippingCompressionAttempted
                ? "compression_peak_not_qualified"
                : releasedAssemblySettled
                ? "completed_released_assembly_unqualified"
                : "released_assembly_unsettled") :
        arguments.boxFixtureJoint ? "completed_fixture_held_unbonded" :
        "completed_boundary_actions";
    const bool shippingSimulationProcessComplete = arguments.shippingBox &&
        fullShippingActionSequence && releasedAssemblySettled && !nativeFailure;
    if (arguments.shippingBox && lastAcceptedNativeStep != NM_INVALID_INDEX) {
        const auto observables = shippingHistoryAreaObservables(
            compiled.world, mesh, currentSnapshot, 0u);
        writeShippingHistoryAreaRow(shippingHistoryAreaLedger,
            lastAcceptedNativeStep, boxSimulationTimeSeconds,
            lastAcceptedTimestepSeconds, "final_accepted_state", true,
            observables);
        shippingHistoryAreaLedger.flush();
    }
    result << "{\n  \"schema\": \"numi.cardboard.fefco0201-native-execution.v1\",\n"
           << "  \"status\": \"" << finalStatus << "\",\n"
           << "  \"compiled_world_fingerprint\": " << compiled.world.fingerprint << ",\n"
           << "  \"source_physics_fingerprint\": " << compiled.world.physicsFingerprint << ",\n"
           << "  \"device_program_fingerprint\": " << compiled.world.fingerprint << ",\n"
           << "  \"substrate_self_weight\": {\"enabled\": "
           << (arguments.shippingBox ? "true" : "false")
           << ", \"gravity_m_s2\": ["
           << compiled.world.dispatch.gravityAndTimestep.x << ','
           << compiled.world.dispatch.gravityAndTimestep.y << ','
           << compiled.world.dispatch.gravityAndTimestep.z
           << "], \"compiled_nodal_mass_total_kg\": "
           << compiledSubstrateMassKg
           << ", \"substrate_weight_force_n\": [0,"
           << compiledSubstrateMassKg *
                compiled.world.dispatch.gravityAndTimestep.y
           << ",0], \"native_fem_rhs\": \"nodal_mass_times_compiled_gravity\", "
              "\"gravity_is_immutable_across_stage_branches\": true, "
              "\"stage_gravity_switches\": false, "
              "\"stage_scope\": \"score, assembly, release-settle, compression, support transfer, rigid-body mass initialization and drop\"},\n"
           << "  \"safeguarded_paper_hill_initializer\": "
           << (arguments.safeguardedPaperInitializer ? "true" : "false") << ",\n"
           << "  \"paper_compensated_arithmetic\": "
           << (arguments.compensatedPaperArithmetic ? "true" : "false") << ",\n"
           << "  \"paper_hill_accurate_projection\": "
           << (arguments.accuratePaperProjection ? "true" : "false") << ",\n"
           << "  \"native_steps_attempted\": " << nextNativeStep << ",\n"
           << "  \"native_steps_accepted\": " << acceptedNativeSteps << ",\n"
           << "  \"starch_initial_activation_requested\": "
           << (!mesh.starchActivationCommands.empty() ? "true" : "false") << ",\n"
           << "  \"starch_initial_activation_accepted\": "
           << (starchInitialActivationAccepted ? "true" : "false") << ",\n"
           << "  \"starch_initial_activation_native_step\": ";
    if (starchInitialActivationAccepted)
        result << starchInitialActivationStep;
    else
        result << "null";
    result << ",\n"
           << "  \"shipping_through_stage\": \""
           << (arguments.shippingBox ? arguments.shippingRunThroughStage : "not_shipping") << "\",\n"
           << "  \"shipping_stage_cutoff_reached\": "
           << (stageCutoffReached ? "true" : "false") << ",\n"
           << "  \"held_boundary_settle_steps_accepted\": " << acceptedSettleSteps << ",\n"
           << "  \"held_boundary_settle_is_released_assembly\": false,\n"
           << "  \"score_tool_and_anvil_withdrawn_before_folding\": "
           << (scoreToolsWithdrawn ? "true" : "false") << ",\n"
           << "  \"released_clamp_node_count\": " << releasedClampNodeCount << ",\n"
           << "  \"released_clamp_zero_active_gate_passed\": "
           << (releasedClampZeroActiveGatePassed ? "true" : "false") << ",\n"
           << "  \"remaining_active_temporary_clamp_node_count\": "
           << remainingActiveTemporaryClampNodeCount << ",\n"
           << "  \"released_settle_steps_accepted\": " << releasedSettleSteps << ",\n"
           << "  \"released_assembly_max_nodal_speed_m_s\": ";
    if (std::isfinite(releasedAssemblyMaximumSpeedMPerS))
        result << releasedAssemblyMaximumSpeedMPerS;
    else
        result << "null";
    result << ",\n"
           << "  \"released_assembly_speed_gate_m_s\": 0.0001,\n"
           << "  \"released_assembly_settled\": "
           << (releasedAssemblySettled ? "true" : "false") << ",\n"
           << "  \"springback_observation_file\": "
           << (arguments.shippingBox ? "\"box-score-springback.csv\"" : "null")
           << ",\n"
           << "  \"springback_observation_rows\": "
           << springbackRowsEmitted
           << ",\n"
           << "  \"springback_observable_rows\": "
           << springbackObservableRows << ",\n"
           << "  \"springback_observation_contract\": \"accepted held endpoint immediately before clamp release to accepted released endpoint after settle; score-shape RMS and coverage gates apply at both endpoints; floor support remains active\",\n"
           << "  \"grip_reaction_file\": "
           << (arguments.shippingBox ? "\"box-grip-reactions.csv\"" : "null")
           << ",\n"
           << "  \"grip_reaction_contract\": \"mechanical prescribed-row action includes continuum, cohesive, external, gravity and inertia terms; rigid/deformable contact is a separate additive support-force channel; moments use accepted tag-3 positions; Human tag-2 attachment reactions remain separate\",\n"
           << "  \"released_assembly_matter_checkpoint_file\": ";
    if (!releasedAssemblyCheckpointPath.empty())
        result << "\"checkpoints/"
               << jsonEscape(releasedAssemblyCheckpointPath.filename().string())
               << "\"";
    else
        result << "null";
    result << ",\n  \"released_assembly_matter_checkpoint_sha256\": ";
    if (!releasedAssemblyCheckpointPath.empty())
        result << "\"" << hexDigest(sha256File(releasedAssemblyCheckpointPath)) << "\"";
    else
        result << "null";
    result << ",\n  \"released_assembly_rigid_scene_sidecar_file\": ";
    if (!releasedAssemblyRigidSidecarPath.empty())
        result << "\"checkpoints/"
               << jsonEscape(releasedAssemblyRigidSidecarPath.filename().string())
               << "\"";
    else
        result << "null";
    result << ",\n  \"released_assembly_rigid_scene_sidecar_sha256\": ";
    if (!releasedAssemblyRigidSidecarSha256.empty())
        result << "\"" << releasedAssemblyRigidSidecarSha256 << "\"";
    else
        result << "null";
    result << ",\n  \"released_assembly_rigid_scene_sidecar_metadata_file\": ";
    if (!releasedAssemblyRigidSidecarMetadataPath.empty())
        result << "\"checkpoints/"
               << jsonEscape(releasedAssemblyRigidSidecarMetadataPath.filename().string())
               << "\"";
    else
        result << "null";
    result << ",\n  \"released_assembly_temporary_tool_contact_free\": "
           << (releasedAssemblySettled &&
                       releasedAssemblyForbiddenToolContactSamples <= 1.0e-6
                   ? "true" : "false") << ",\n"
           << "  \"released_assembly_rigid_contact_samples_total\": "
           << releasedAssemblyRigidContactSamplesTotal << ",\n"
           << "  \"released_assembly_ground_support_contact_samples\": "
           << releasedAssemblySupportFloorContactSamples << ",\n"
           << "  \"released_assembly_forbidden_temporary_tool_contact_samples\": "
           << releasedAssemblyForbiddenToolContactSamples << ",\n"
           << "  \"shipping_support_floor_placement_clearance_m\": "
           << shippingFloorPlacementClearanceM << ",\n"
           << "  \"shipping_support_floor_placement_accepted_steps\": "
           << shippingFloorPlacementSteps << ",\n"
           << "  \"shipping_support_floor_contact_body_index\": 3,\n"
           << "  \"released_shipping_blank_active_prescribed_node_count\": "
           << remainingActivePrescribedNodeCount << ",\n"
           << "  \"shipping_support_ready_rigid_scene_sidecar_file\": ";
    if (!shippingSupportRigidSidecarPath.empty())
        result << "\"checkpoints/"
               << jsonEscape(shippingSupportRigidSidecarPath.filename().string())
               << "\"";
    else
        result << "null";
    result << ",\n  \"shipping_support_ready_rigid_scene_sidecar_sha256\": ";
    if (!shippingSupportRigidSidecarSha256.empty())
        result << "\"" << shippingSupportRigidSidecarSha256 << "\"";
    else
        result << "null";
    result << ",\n  \"shipping_support_ready_rigid_scene_sidecar_metadata_file\": ";
    if (!shippingSupportRigidSidecarMetadataPath.empty())
        result << "\"checkpoints/"
               << jsonEscape(shippingSupportRigidSidecarMetadataPath.filename().string())
               << "\"";
    else
        result << "null";
    result << ",\n  \"shipping_support_ready_rigid_sidecar_restore_verified\": "
           << (shippingSupportRigidSidecarRestoreVerified ? "true" : "false")
           << ",\n"
           << "  \"released_assembly_simulation_process_complete\": "
           << (shippingSimulationProcessComplete ? "true" : "false") << ",\n"
           << "  \"released_assembly_simulation_gate_passed\": false,\n"
           << "  \"released_assembly_simulation_gate_status\": \"not_evaluated_no_measured_inside_dimensions\",\n"
           << "  \"source_grade_calibration_gate_passed\": false,\n"
           << "  \"physical_shipping_box_validation_passed\": false,\n"
           << "  \"released_assembly_qualification\": false,\n"
           << "  \"native_slow_compression_attempted\": "
           << (shippingCompressionAttempted ? "true" : "false") << ",\n"
           << "  \"native_slow_compression_completed\": "
           << (shippingCompressionCompleted ? "true" : "false") << ",\n"
           << "  \"native_compression_branches_reached_first_peak\": "
           << (shippingCompressionBothBranchesReachedPeak ? "true" : "false") << ",\n"
           << "  \"native_compression_rate_sensitivity_gate_passed\": ";
    if (shippingCompressionRateSensitivityEvaluated)
        result << (shippingCompressionRateSensitivityPassed ? "true" : "false");
    else
        result << "null";
    result << ",\n"
           << "  \"native_half_speed_compression_completed\": "
           << (shippingCompressionHalfSpeedCompleted ? "true" : "false") << ",\n"
           << "  \"native_compression_numerical_capacity_qualification_status\": \""
           << jsonEscape(shippingCompressionNumericalQualificationStatus) << "\",\n"
           << "  \"native_compression_numerical_capacity_qualification_passed\": "
           << (shippingCompressionNumericallyQualified ? "true" : "false") << ",\n"
           << "  \"native_compression_numerical_capacity_qualification_reason\": \""
           << jsonEscape(shippingCompressionNumericalQualificationReason) << "\",\n"
           << "  \"native_compression_numerical_capacity_qualification_scope\": \"Fpeak mass sizing only; not source-grade or physical-box qualification\",\n"
           << "  \"shipping_capacity_qualification_file\": \""
           << jsonEscape(arguments.shippingCapacityQualificationFile.string()) << "\",\n"
           << "  \"shipping_capacity_qualification_file_sha256\": \""
           << arguments.shippingCapacityQualificationSha256 << "\",\n"
           << "  \"shipping_source_material_binding_sha256\": \""
           << (arguments.shippingBox ? shippingSourceBindingDigest(arguments) : "") << "\",\n"
           << "  \"assembled_checkpoint_sha256_for_capacity_binding\": \""
           << (!shippingCrushAssemblyCheckpointPath.empty() &&
                   std::filesystem::is_regular_file(shippingCrushAssemblyCheckpointPath)
               ? hexDigest(sha256File(shippingCrushAssemblyCheckpointPath)) : "")
           << "\",\n"
           << "  \"assembled_rigid_sidecar_sha256_for_capacity_binding\": \""
           << shippingSupportRigidSidecarSha256 << "\",\n"
           << "  \"nominal_compression_curve_sha256\": \""
           << shippingCompressionNominalCurveSha256 << "\",\n"
           << "  \"nominal_compression_curve_samples\": "
           << shippingCompressionNominalCurveSampleCount << ",\n"
           << "  \"nominal_compression_curve_file\": \"shipping-compression-nominal-curve.csv\",\n"
           << "  \"compression_branches_restore_same_support_ready_checkpoint\": true,\n"
           << "  \"compression_stroke_m\": " << shippingCompressionStrokeM << ",\n"
           << "  \"compression_stroke_fraction_of_inside_height\": "
           << (arguments.shippingDesign
                ? shippingCompressionStrokeM /
                    arguments.shippingDesign->defaults.inside.heightY : 0.0)
           << ",\n"
           << "  \"compression_platen_speed_m_per_s\": "
           << (12.5e-3 / 60.0) << ",\n"
           << "  \"compression_half_speed_platen_speed_m_per_s\": "
           << (6.25e-3 / 60.0) << ",\n"
           << "  \"compression_peak_stroke_m\": ";
    if (shippingCompressionPeakForceN > 0.0)
        result << shippingCompressionPeakStrokeM;
    else
        result << "null";
    result << ",\n  \"compression_peak_sample_index\": ";
    if (shippingCompressionPeakForceN > 0.0)
        result << shippingCompressionPeakSampleIndex;
    else
        result << "null";
    result << ",\n  \"compression_peak_detection\": \"first_rising_local_peak_then_two_consecutive_declines_and_at_least_1_percent_drop\",\n"
           << "  \"compression_force_curve_origin\": \"zero-force sample at the accepted endpoint before the first contact-crossing increment; that accepted 0.25 mm increment is the first measured stroke sample\",\n"
           << "  \"compression_max_stroke_fraction\": 0.70,\n"
           << "  \"compression_peak_contact_force_n\": ";
    if (shippingCompressionPeakForceN > 0.0)
        result << shippingCompressionPeakForceN;
    else
        result << "null";
    result << ",\n  \"compression_half_speed_peak_force_n\": ";
    if (shippingCompressionHalfSpeedPeakForceN > 0.0)
        result << shippingCompressionHalfSpeedPeakForceN;
    else
        result << "null";
    result << ",\n  \"compression_half_speed_peak_stroke_m\": ";
    if (shippingCompressionHalfSpeedPeakForceN > 0.0)
        result << shippingCompressionHalfSpeedPeakStrokeM;
    else
        result << "null";
    result << ",\n"
           << "  \"compression_half_speed_relative_peak_force_difference\": ";
    if (shippingCompressionPeakForceN > 0.0 &&
        shippingCompressionHalfSpeedPeakForceN > 0.0)
        result << shippingCompressionHalfSpeedRelativePeakDifference;
    else
        result << "null";
    result << ",\n"
           << "  \"compression_rate_sensitivity\": {\n"
           << "    \"comparison_evaluated\": "
           << (shippingCompressionRateSensitivityEvaluated ? "true" : "false") << ",\n"
           << "    \"same_support_ready_checkpoint\": true,\n"
           << "    \"peak_force_relative_difference\": ";
    if (shippingCompressionPeakForceN > 0.0 &&
        shippingCompressionHalfSpeedPeakForceN > 0.0)
        result << shippingCompressionHalfSpeedRelativePeakDifference;
    else
        result << "null";
    result << ",\n    \"peak_force_tolerance_fraction\": 0.02,\n"
           << "    \"peak_force_gate_passed\": ";
    if (shippingCompressionRateSensitivityEvaluated)
        result << (shippingCompressionHalfSpeedRelativePeakDifference <= 0.02
            ? "true" : "false");
    else
        result << "null";
    result << ",\n    \"common_stroke_m\": ";
    if (shippingCompressionRateSensitivityEvaluated)
        result << shippingCompressionCommonStrokeM;
    else
        result << "null";
    result << ",\n    \"nominal_work_over_common_stroke_j\": ";
    if (shippingCompressionRateSensitivityEvaluated)
        result << shippingCompressionCommonStrokeNominalWorkJ;
    else
        result << "null";
    result << ",\n    \"half_speed_work_over_common_stroke_j\": ";
    if (shippingCompressionRateSensitivityEvaluated)
        result << shippingCompressionCommonStrokeHalfSpeedWorkJ;
    else
        result << "null";
    result << ",\n    \"relative_work_difference\": ";
    if (shippingCompressionRateSensitivityEvaluated)
        result << shippingCompressionRelativeWorkDifference;
    else
        result << "null";
    result << ",\n    \"work_tolerance_fraction\": 0.02,\n"
           << "    \"maximum_force_difference_over_nominal_peak\": ";
    if (shippingCompressionRateSensitivityEvaluated)
        result << shippingCompressionMaximumNormalizedCurveDifference;
    else
        result << "null";
    result << ",\n    \"normalized_rms_curve_difference\": ";
    if (shippingCompressionRateSensitivityEvaluated)
        result << shippingCompressionRmsNormalizedCurveDifference;
    else
        result << "null";
    result << ",\n    \"maximum_normalized_full_curve_tolerance_fraction\": 0.05,\n"
           << "    \"nominal_work_at_peak_j\": ";
    if (shippingCompressionRateSensitivityEvaluated)
        result << shippingCompressionNominalWorkAtPeakJ;
    else
        result << "null";
    result << ",\n    \"half_speed_work_at_peak_j\": ";
    if (shippingCompressionRateSensitivityEvaluated)
        result << shippingCompressionHalfSpeedWorkAtPeakJ;
    else
        result << "null";
    result << ",\n    \"nominal_kinetic_energy_at_peak_j\": ";
    if (shippingCompressionRateSensitivityEvaluated)
        result << shippingCompressionNominalKineticAtPeakJ;
    else
        result << "null";
    result << ",\n    \"half_speed_kinetic_energy_at_peak_j\": ";
    if (shippingCompressionRateSensitivityEvaluated)
        result << shippingCompressionHalfSpeedKineticAtPeakJ;
    else
        result << "null";
    result << ",\n    \"nominal_kinetic_to_work_ratio_at_peak\": ";
    if (shippingCompressionRateSensitivityEvaluated &&
        std::isfinite(shippingCompressionNominalKineticWorkRatioAtPeak))
        result << shippingCompressionNominalKineticWorkRatioAtPeak;
    else
        result << "null";
    result << ",\n    \"half_speed_kinetic_to_work_ratio_at_peak\": ";
    if (shippingCompressionRateSensitivityEvaluated &&
        std::isfinite(shippingCompressionHalfSpeedKineticWorkRatioAtPeak))
        result << shippingCompressionHalfSpeedKineticWorkRatioAtPeak;
    else
        result << "null";
    result << ",\n    \"passed\": ";
    if (shippingCompressionRateSensitivityEvaluated)
        result << (shippingCompressionRateSensitivityPassed ? "true" : "false");
    else
        result << "null";
    result << "\n  },\n"
           << "  \"compression_force_curve_file\": \"shipping-compression.csv\",\n"
           << "  \"history_observables_file\": \"shipping-history-observables.csv\",\n"
           << "  \"paper_damage_observable_definition\": \"Each explicit retained-matrix overlay tetrahedron contributes its four authored reference crack-plane cut areas. For each plane, damage is damageCap*(1-survivalNormal*survivalShear), where survival(mode)=onset/(onset+acceptedExcess)*exp(-acceptedExcess/softening); the reported index is reference-area-weighted. This is a constitutive damage index, not a measured or binary geometric crack area.\",\n"
           << "  \"cohesive_degradation_observable_definition\": \"For active accepted mixed-mode starch/HMA faces, the reported fraction is sum(referenceFaceArea*acceptedFaceDamage)/sum(referenceFaceArea). It is an area-weighted cohesive degradation index, not geometric open/debond area; dormant faces are excluded until accepted activation.\",\n"
           << "  \"compression_work_method\": \"trapezoidal integral from a zero-force contact origin through the first accepted contact-crossing increment and subsequent accepted platen stroke\",\n"
           << "  \"compression_kinetic_energy_method\": \"sum 0.5 times accepted FEM nodal mass times squared velocity from RuntimeStateSnapshot\",\n"
           << "  \"drop_mass_basis\": \"nominal 12.5 mm/min measured first-local-peak Fpeak, only after speed sensitivity and separately supplied source-bound mesh/time numerical-capacity evidence pass\",\n"
           << "  \"shipping_impact_timestep_s\": "
           << arguments.shippingImpactTimestepSeconds << ",\n"
           << "  \"shipping_impact_dt_refinement_requested\": "
           << (arguments.shippingImpactTimestepRefinement ? "true" : "false") << ",\n"
           << "  \"shipping_impact_timestep_refinement_cli\": "
           << "\"--shipping-impact-dt-refinement runs 500, 250, and 125 us branches; every mass/location/dt case restores the same accepted released assembly checkpoint\",\n"
           << "  \"shipping_impact_timestep_levels_s\": [";
    if (arguments.shippingImpactTimestepRefinement)
        result << "0.0005,0.00025,0.000125";
    else
        result << arguments.shippingImpactTimestepSeconds;
    result << "],\n"
           << "  \"shipping_impact_dynamics_file\": \"shipping-impact-dynamics.csv\",\n"
           << "  \"shipping_impact_energy_basis\": \"rigid translational and rotational kinetic energy plus gravity-relative potential; FEM kinetic energy and native Matter reaction impulse/mean force are separate columns\",\n"
           << "  \"compression_speed_comparison_is_timestep_refinement\": false,\n"
           << "  \"compression_force_source\": \"Matter RuntimeStateSnapshot.reactions mapped by compiled rigid proxy bodyIndex; sum impulse_y and divide by accepted control-step dt\",\n"
           << "  \"native_rigid_contact_samples_last_step\": "
           << lastRigidContactSamples << ",\n"
           << "  \"native_gravity_transfer_attempted\": "
           << (shippingGravityTransferAttempted ? "true" : "false") << ",\n"
           << "  \"native_gravity_transfer_completed_cases\": "
           << shippingGravityTransferCasesAccepted << ",\n"
           << "  \"native_gravity_transfer_all_cases_completed\": "
           << (shippingGravityTransferCompleted ? "true" : "false") << ",\n"
           << "  \"gravity_transfer_timestep_s\": "
           << arguments.shippingGravityTransferTimestepSeconds << ",\n"
           << "  \"gravity_transfer_ramp_requested_duration_s\": 0.5,\n"
           << "  \"gravity_transfer_ramp_actual_duration_s\": "
           << std::ceil(0.5 / arguments.shippingGravityTransferTimestepSeconds) *
                arguments.shippingGravityTransferTimestepSeconds << ",\n"
           << "  \"gravity_transfer_post_release_observation_requested_s\": 0.25,\n"
           << "  \"gravity_transfer_post_release_observation_actual_s\": "
           << std::max(1.0, std::ceil(0.25 /
                arguments.shippingGravityTransferTimestepSeconds)) *
                arguments.shippingGravityTransferTimestepSeconds << ",\n"
           << "  \"gravity_transfer_dynamics_file\": \"shipping-gravity-transfer.csv\",\n"
           << "  \"gravity_transfer_maximum_linear_momentum_residual_ns\": "
           << shippingGravityTransferMaximumMomentumResidualNs << ",\n"
           << "  \"gravity_transfer_maximum_normalized_momentum_residual\": "
           << shippingGravityTransferMaximumMomentumResidualNormalized << ",\n"
           << "  \"gravity_transfer_actuator_policy\": \"full gravity remains enabled; bodyIndex4 upward force is midpoint-ramped from mg to near zero, then a separate accepted zero-wrench release transaction and post-release observation run\",\n"
           << "  \"gravity_transfer_qualification\": false,\n"
           << "  \"native_finite_mass_drop_attempted\": "
           << (shippingDropCasesAttempted ? "true" : "false") << ",\n"
           << "  \"native_finite_mass_drop_completed_cases\": "
           << shippingDropCasesAccepted << ",\n"
           << "  \"native_finite_mass_drop_planned_cases\": "
           << shippingDropCasesPlanned << ",\n"
           << "  \"native_finite_mass_drop_supported\": "
           << (shippingDropCasesCompleted ? "true" : "false") << ",\n"
           << "  \"shipping_drop_protocol_executable\": "
           << (shippingDropCasesCompleted ? "true" : "false") << ",\n"
           << "  \"shipping_drop_blocker\": \""
           << (shippingDropCasesCompleted ? std::string{} : jsonEscape(stopReason))
           << "\",\n"
           << "  \"native_crush_and_drop_qualification\": false,\n"
           << "  \"shipping_outer_roof_measurement_status\": \""
           << (shippingImpactorFootprintPlan
                ? "measured_from_accepted_settled_checkpoint"
                : jsonEscape(shippingOuterRoofMeasurementFailure.empty()
                    ? std::string("unavailable")
                    : shippingOuterRoofMeasurementFailure))
           << "\",\n"
           << "  \"shipping_outer_roof_measurement_file\": \""
           << (shippingImpactorFootprintPlan
                ? "shipping-outer-roof-measurement.json" : "") << "\",\n"
           << "  \"shipping_outer_roof_input_file\": \""
           << jsonEscape(arguments.shippingRoofMeasurementFile.string()) << "\",\n"
           << "  \"shipping_outer_roof_input_sha256\": \""
           << arguments.shippingRoofMeasurementFileSha256 << "\",\n"
           << "  \"shipping_blank_geometry_binding_sha256\": \""
           << arguments.shippingBlankGeometrySha256 << "\",\n"
           << "  \"shipping_source_binding_sha256\": \""
           << (arguments.shippingBox ? shippingSourceBindingDigest(arguments) : "")
           << "\",\n"
           << "  \"shipping_impactor_half_extents_m\": ["
           << arguments.shippingImpactorHalfExtentsM[0] << ','
           << arguments.shippingImpactorHalfExtentsM[1] << ','
           << arguments.shippingImpactorHalfExtentsM[2] << "],\n"
           << "  \"shipping_impactor_geometry_source\": \""
           << jsonEscape(arguments.shippingImpactorGeometrySource) << "\",\n"
           << "  \"shipping_impactor_source_measurement_validated\": "
           << (arguments.shippingImpactorSourceMeasurementValidated ? "true" : "false")
           << ",\n  \"shipping_impactor_compiled_proxy_matches_explicit_geometry\": "
           << (arguments.shippingImpactorCompiledProxyMatchesMeasurement ? "true" : "false")
           << ",\n  \"shipping_impactor_current_roof_matches_compiled_proxy\": "
           << (arguments.shippingImpactorCurrentRoofMatchesCompiledProxy ? "true" : "false")
           << ",\n"
           << "  \"shipping_outer_roof_checkpoint_sha256\": \""
           << (shippingImpactorFootprintPlan
                ? shippingImpactorFootprintPlan->sourceCheckpointSha256 : "")
           << "\",\n"
           << "  \"shipping_outer_roof_bounds_m\": ";
    if (shippingOuterRoofMeasurement) {
        const auto& roof = shippingOuterRoofMeasurement->bounds;
        result << "{\"min_x\":" << roof.minimumX
               << ",\"max_x\":" << roof.maximumX
               << ",\"min_z\":" << roof.minimumZ
               << ",\"max_z\":" << roof.maximumZ
               << ",\"top_y\":" << roof.topY
               << ",\"exterior_face_count\":" << roof.exteriorFaceCount << '}';
    } else {
        result << "null";
    }
    result << ",\n  \"shipping_impactor_plan_length_x_m\": ";
    if (shippingImpactorFootprintPlan)
        result << shippingImpactorFootprintPlan->fullLengthX;
    else
        result << "null";
    result << ",\n  \"shipping_impactor_plan_width_z_m\": ";
    if (shippingImpactorFootprintPlan)
        result << shippingImpactorFootprintPlan->fullWidthZ;
    else
        result << "null";
    result << ",\n  \"shipping_impactor_footprint_basis\": \""
           << (shippingImpactorFootprintPlan
                ? "1.2 times measured accepted outer-roof x/z bounds; center and quarter-width offset use the same measured bounds"
                : "no internal-dimension fallback; requires accepted outer-roof measurement")
           << "\",\n  \"shipping_impactor_actual_outer_roof_coverage_verified\": "
           << (arguments.shippingImpactorSourceMeasurementValidated &&
                       arguments.shippingImpactorCompiledProxyMatchesMeasurement &&
                       arguments.shippingImpactorCurrentRoofMatchesCompiledProxy
                   ? "true" : "false")
           << ",\n  \"shipping_impactor_coverage_is_contact_qualified\": false,\n"
           << "  \"score_line_count\": " << blank.scoreLines.size() << ",\n"
           << "  \"assembly_action_count\": " << blank.assembly.actions.size() << ",\n"
           << "  \"score_tool_radius_m\": " << tooling.punchRadius << ",\n"
           << "  \"box_grip_reference_patch_csv\": \"box-grip-patches.csv\",\n"
           << "  \"box_grip_node_source\": \"compiled exposed femContactNodes; temporary kinematic targets are restricted to the exterior-node map\",\n"
           << "  \"wall_grip_patch_policy\": \"finite x/y patch inside each wall panel on exterior z=0 and z=caliper liner faces; score edges and interior flute/glue nodes are excluded\",\n"
           << "  \"score_tool_clearance_m\": " << tooling.toolClearance << ",\n"
           << "  \"score_approach_steps_per_leg\": " << arguments.boxScoreStepsPerLeg << ",\n"
           << "  \"shipping_score_depth_ratio_of_caliper\": "
           << (arguments.shippingBox ? arguments.shippingScoreDepthRatio : 0.0)
           << ",\n"
           << "  \"shipping_score_selected_depth_m\": "
           << (arguments.shippingBox
                   ? arguments.shippingScoreDepthRatio * blank.boardCaliper : 0.0)
           << ",\n"
           << "  \"shipping_score_tool_speed_requested_m_per_s\": "
           << (arguments.shippingBox ? arguments.shippingScoreToolSpeedMPerS : 0.0)
           << ",\n"
           << "  \"shipping_score_tool_speed_evidence_class\": \"engineering_assumption_unqualified\",\n"
           << "  \"shipping_score_approach_steps_accepted\": "
           << shippingScoreApproachAcceptedSteps << ",\n"
           << "  \"shipping_score_retract_steps_accepted\": "
           << shippingScoreRetractAcceptedSteps << ",\n"
           << "  \"shipping_score_effective_speed_range_m_per_s\": [";
    if (std::isfinite(shippingScoreEffectiveSpeedMinMPerS))
        result << shippingScoreEffectiveSpeedMinMPerS;
    else
        result << "null";
    result << ',';
    if (shippingScoreEffectiveSpeedMaxMPerS > 0.0)
        result << shippingScoreEffectiveSpeedMaxMPerS;
    else
        result << "null";
    result << "],\n"
           << "  \"shipping_score_command_sha256\": \""
           << (arguments.shippingBox
                   ? shippingScoreCommandFingerprint(
                         arguments, compiled.world.physicsFingerprint,
                         compiled.world.fingerprint, blank.boardCaliper)
                   : std::string{}) << "\",\n"
           << "  \"shipping_score_case_semantics\": \"one selected depth, one approach and one withdrawal across the 12-axis tool pack; each depth study invocation starts from the independently constructed flat blank\",\n"
           << "  \"maximum_fold_increment_deg\": " << arguments.boxAngleIncrementDegrees << ",\n"
           << "  \"shape_fit_residual_bound_calipers\": "
           << arguments.boxShapeResidualBoundCalipers << ",\n"
           << "  \"shape_fit_coverage_minimum\": "
           << arguments.boxShapeFitCoverageMinimum << ",\n"
           << "  \"shape_fit_bound_is_published_tolerance\": false,\n"
           << "  \"score_contact_is_damage\": false,\n"
           << "  \"source_scored_lines_are_physical\": false,\n"
           << "  \"manufacturer_joint_cohesive_law_present\": "
           << (arguments.shippingHmaLaw ? "true" : "false") << ",\n"
           << "  \"manufacturer_joint_lap_activated\": "
           << (acceptedHmaActivationEvents.contains(
                    "accepted_manufacturer_lap_overlap") ? "true" : "false") << ",\n"
           << "  \"bottom_flap_overlaps_activated\": "
           << (acceptedHmaActivationEvents.contains(
                    "accepted_bottom_flap_overlaps") ? "true" : "false") << ",\n"
           << "  \"top_flap_overlaps_activated\": "
           << (acceptedHmaActivationEvents.contains(
                    "accepted_top_flap_overlaps") ? "true" : "false") << ",\n"
           << "  \"hma_calibrated_for_corrugated_recipe\": false,\n"
           << "  \"fixture_mode\": " << (arguments.boxFixtureJoint ? "true" : "false") << ",\n"
           << "  \"externally_held\": " << (currentSnapshot.femNodes.size() >= nodeCount && std::any_of(
                currentSnapshot.femNodes.begin(),
                currentSnapshot.femNodes.begin() + nodeCount,
                [](const auto& node) { return node.restAndFixed.w == 3.0f; })
                ? "true" : "false") << ",\n"
           << "  \"initial_tab_body_anchor_node_count\": " << tabAnchorNodes.size() << ",\n"
           << "  \"initial_tab_body_anchor_active_at_end\": " << (tabAnchorActive ? "true" : "false") << ",\n"
           << "  \"tab_anchor_released_after_wall_clamps_before_joint_fixture\": "
           << (tabAnchorReleasedBeforeJointFixture ? "true" : "false") << ",\n"
           << "  \"wall_boundary_clamps_retained_through_joint_fixture\": "
           << (arguments.boxFixtureJoint && tabAnchorReleasedBeforeJointFixture ? "true" : "false") << ",\n"
           << "  \"self_contact_enabled\": "
           << (mesh.world.objects.front().deformableSelfContact ? "true" : "false") << ",\n"
           << "  \"numerical_control_without_deformable_contact\": "
           << (arguments.disableSelfContact ? "true" : "false") << ",\n"
           << "  \"physical_assembly_without_fixtures\": false,\n"
           << "  \"sealed\": false,\n"
           << "  \"adhesive_calibrated\": false,\n"
           << "  \"physical_validation\": false,\n"
           << "  \"fixture_standoff_m\": "
           << (arguments.boxFixtureJoint ? arguments.boxFixtureGapM : 0.0) << ",\n"
           << "  \"fixture_joint_native_substeps\": "
           << (arguments.boxFixtureJoint ? arguments.boxFixtureSteps : 0u) << ",\n"
           << "  \"stop_reason\": \"" << jsonEscape(stopReason) << "\"\n}\n";
    require(result.good(), "failed writing native box result JSON");
    if (arguments.shippingBox) {
        const auto staticMemory = shippingStaticMemoryLowerBound(compiled.world);
        writeShippingMemoryReport(
            arguments.output / "shipping-memory-native-final.json",
            arguments, mesh, compiled.world, staticMemory,
            &run.lastMemoryObservation(), shippingArenaBudget
                ? &*shippingArenaBudget : nullptr);
    }
    const std::string status = finalStatus;
    writeSummaryJson(arguments.output / "result.json", status, mesh, compiled,
                     acceptedNativeSteps, stopReason);
    printResult(status, compiled, acceptedNativeSteps, stopReason);
    if (shapeFitRejection) return 4;
    if (nativeFailure) return 2;
    if (blockedByBondLaw) return 3;
    return 0;
}

struct PinAdhesionTraceRow {
    std::uint64_t attemptedStep = 0u;
    std::uint64_t acceptedStep = 0u;
    std::string phase;
    std::uint32_t phaseStep = 0u;
    double elapsedSeconds = 0.0;
    double relativeOpeningM = 0.0;
    double commandedRelativeSpeedMPerS = 0.0;
    double remainingBoardForceN = 0.0;
    double selectedFacingForceN = 0.0;
    double instrumentLoadMagnitudeN = 0.0;
    double remainingBoardContactSamples = 0.0;
    double selectedFacingContactSamples = 0.0;
    std::int32_t statusCode = -1;
    double certificateResidual = NAN;
    double certificateCorrection = NAN;
    double certificateVolumeResidual = NAN;
    double certificatePressureResidual = NAN;
    bool certificateValid = false;
    bool accepted = false;
};

bool pinNativeStepAccepted(const RuntimeStateSnapshot& snapshot,
                           const CompiledWorld& compiled) {
    if (!snapshot.available || snapshot.statuses.size() != 1u ||
        snapshot.solverCertificates.size() != 1u)
        return false;
    const auto& status = snapshot.statuses.front();
    const auto& certificate = snapshot.solverCertificates.front();
    return status.code == NM_STATUS_SUCCESS && certificate.validity.w > 0.5f &&
        std::isfinite(certificate.nonlinear.x) &&
        std::isfinite(certificate.nonlinear.y) &&
        std::isfinite(certificate.nonlinear.z) &&
        std::isfinite(certificate.nonlinear.w) &&
        certificate.nonlinear.x <= compiled.mixedSolver.residualTolerances.x &&
        certificate.nonlinear.z <= compiled.mixedSolver.residualTolerances.y &&
        certificate.nonlinear.w <= compiled.mixedSolver.residualTolerances.z;
}

std::vector<MRBodyStateGPU> pinBodiesForInterval(
    const numi::cardboard::PinAdhesionFixtureGeometry& fixture,
    const double startRelativeOpeningM,
    const double endRelativeOpeningM,
    const double timestepSeconds) {
    require(std::isfinite(timestepSeconds) && timestepSeconds > 0.0,
            "pin body interval timestep must be finite and positive");
    std::vector<MRBodyStateGPU> bodies;
    bodies.reserve(fixture.pins.size());
    for (const auto& pin : fixture.pins) {
        const auto start = numi::cardboard::pinBodyTranslationAtSeparation(
            fixture, pin, startRelativeOpeningM);
        const auto end = numi::cardboard::pinBodyTranslationAtSeparation(
            fixture, pin, endRelativeOpeningM);
        MRBodyStateGPU body{};
        body.position = {static_cast<float>(start[0]),
                         static_cast<float>(start[1]),
                         static_cast<float>(start[2]), 0.0f};
        body.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
        body.flagsAndIndices[0] = MR_MOTION_KINEMATIC;
        body.flagsAndIndices[2] = MR_INVALID_INDEX;
        body.linearVelocityAndInverseMass = {
            0.0f, 0.0f,
            static_cast<float>((end[2] - start[2]) / timestepSeconds), 0.0f};
        bodies.push_back(body);
    }
    return bodies;
}

struct PinGroupReaction {
    double signedImpulseZNs = 0.0;
    double contactSamples = 0.0;
};

std::pair<PinGroupReaction, PinGroupReaction> measurePinGroupReactions(
    const RuntimeStateSnapshot& snapshot,
    const CompiledWorld& compiled,
    const numi::cardboard::PinAdhesionFixtureGeometry& fixture,
    const double timestepSeconds) {
    require(std::isfinite(timestepSeconds) && timestepSeconds > 0.0,
            "pin reaction conversion requires a positive timestep");
    const auto& proxies = compiled.contact.rigidProxies;
    require(proxies.size() == fixture.pins.size() &&
                snapshot.reactions.size() == proxies.size(),
            "pin reaction/proxy state does not match the six authored bars");
    PinGroupReaction remaining;
    PinGroupReaction selected;
    std::array<bool, 6> seen{};
    for (std::size_t index = 0u; index < fixture.pins.size(); ++index) {
        const auto bodyIndex = fixture.pins[index].proxy.bodyIndex;
        require(bodyIndex < seen.size() && !seen[bodyIndex],
                "pin fixture has a duplicate or unsupported global body index");
        seen[bodyIndex] = true;
        require(proxies[index].bodyIndex == bodyIndex,
                "compiled pin proxy ordering differs from the authored platen groups");
        const auto& impulse = snapshot.reactions[index].impulseAndCount;
        for (const float value : {impulse.x, impulse.y, impulse.z, impulse.w})
            require(std::isfinite(value),
                    "native pin reaction contains a nonfinite impulse or contact count");
        require(impulse.w >= 0.0f,
                "native pin reaction has a negative contact count");
        auto& group = fixture.pins[index].group ==
                numi::cardboard::PinPlatenGroup::remainingBoard
            ? remaining : selected;
        group.signedImpulseZNs += impulse.z;
        group.contactSamples += impulse.w;
    }
    require(std::all_of(seen.begin(), seen.end(), [](const bool value) { return value; }),
            "pin fixture body indices are not the expected zero-based six-body set");
    return {remaining, selected};
}

void writePinTraceHeader(std::ostream& output) {
    output << "attemptedStep,acceptedStep,phase,phaseStep,elapsedS,relativeOpeningM,"
              "commandedRelativeSpeedMPerS,remainingBoardForceN,selectedFacingForceN,"
              "instrumentLoadMagnitudeN,remainingBoardContactSamples,"
              "selectedFacingContactSamples,statusCode,certificateValid,"
              "certificateResidual,certificateCorrection,certificateVolumeResidual,"
              "certificatePressureResidual,accepted\n";
}

void writePinTraceRow(std::ostream& output, const PinAdhesionTraceRow& row) {
    output << row.attemptedStep << ',';
    if (row.accepted) output << row.acceptedStep;
    output << ',' << row.phase << ',' << row.phaseStep << ','
           << std::setprecision(17) << row.elapsedSeconds << ','
           << row.relativeOpeningM << ',' << row.commandedRelativeSpeedMPerS << ',';
    if (row.accepted) {
        output << row.remainingBoardForceN << ',' << row.selectedFacingForceN << ','
               << row.instrumentLoadMagnitudeN << ','
               << row.remainingBoardContactSamples << ','
               << row.selectedFacingContactSamples;
    } else {
        output << ",,,,";
    }
    output << ',' << row.statusCode << ',' << (row.certificateValid ? 1 : 0) << ',';
    if (std::isfinite(row.certificateResidual)) output << row.certificateResidual;
    output << ',';
    if (std::isfinite(row.certificateCorrection)) output << row.certificateCorrection;
    output << ',';
    if (std::isfinite(row.certificateVolumeResidual)) output << row.certificateVolumeResidual;
    output << ',';
    if (std::isfinite(row.certificatePressureResidual)) output << row.certificatePressureResidual;
    output << ',' << (row.accepted ? 1 : 0) << '\n';
}

void writePinCheckpoint(const std::filesystem::path& path,
                        const RuntimeStateSnapshot& snapshot,
                        const std::uint64_t sourceFingerprint,
                        const std::uint64_t programFingerprint,
                        const std::string_view role) {
    std::filesystem::create_directories(path.parent_path());
    numi::cardboard::writeShippingAcceptedSnapshot(path, snapshot);
    std::ofstream metadata(path.string() + ".json");
    require(metadata.good(), "cannot write pin accepted-checkpoint metadata");
    metadata << "{\n  \"schema\": \"numi.cardboard.pin-adhesion-checkpoint.v1\",\n"
            << "  \"role\": \"" << jsonEscape(role) << "\",\n"
            << "  \"snapshot_file\": \"" << jsonEscape(path.filename().string()) << "\",\n"
            << "  \"snapshot_sha256\": \"" << hexDigest(sha256File(path)) << "\",\n"
            << "  \"source_physics_fingerprint\": " << sourceFingerprint << ",\n"
            << "  \"device_program_fingerprint\": " << programFingerprint << ",\n"
            << "  \"accepted_control_step\": " << snapshot.controlStep << ",\n"
            << "  \"native_certificate_bound\": true, \"qualification\": false\n}\n";
    require(metadata.good(), "failed writing pin checkpoint metadata");
}

int executePinAdhesion(const Arguments& arguments,
                       const MeshSource& mesh,
                       const CompileResult& compiled) {
    require(mesh.pinFixture.has_value(),
            "pin mode has no authored capsule fixture geometry");
    const auto& fixture = *mesh.pinFixture;
    const auto schedule = numi::cardboard::makePinAdhesionProtocolSchedule(
        fixture, arguments.timestepSeconds, arguments.pinMaximumOpeningM,
        arguments.pinWithdrawalDistanceM);
    require(schedule.points.size() <= 200000u,
            "pin schedule exceeds its fixed bounded accepted-step count");
    NativeRun run(compiled.world, arguments.metallib);
    const std::size_t nodeCount = mesh.restNodes.size();
    const std::vector<nm_float4> targets(
        nodeCount, nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
    std::ofstream trace(arguments.output / "pin-adhesion-trace.csv");
    require(trace.good(), "cannot create pin adhesion trace");
    writePinTraceHeader(trace);

    std::vector<PinAdhesionTraceRow> rows;
    rows.reserve(schedule.points.size() + 2u);
    std::vector<numi::cardboard::PinAdhesionAcceptedSample> workSamples;
    workSamples.reserve(schedule.points.size() + 1u);
    std::array<std::uint64_t, 4> phaseExpected{};
    std::array<std::uint64_t, 4> phaseAccepted{};
    for (const auto& point : schedule.points)
        ++phaseExpected[static_cast<std::size_t>(point.phase)];
    std::map<std::uint32_t, NMCohesiveFaceGPU> peakFaces;
    std::map<std::uint32_t, double> peakFaceDamage;
    std::vector<std::string> checkpointFiles;
    std::uint64_t acceptedSteps = 0u;
    std::uint64_t attemptedSteps = 0u;
    bool failed = false;
    std::string failure;
    bool initialAccepted = false;
    bool activationAccepted = !mesh.starchInterfaceParameters.has_value();
    std::uint32_t nativeControlStep = 0u;
    auto appendSnapshotRow = [&](const RunState& state,
                                 const std::string& phase,
                                 const std::uint32_t phaseStep,
                                 const double elapsedSeconds,
                                 const double relativeOpeningM,
                                 const double commandedSpeed,
                                 const bool workEligible,
                                 const bool permitCheckpoint,
                                 const std::string_view checkpointRole) {
        PinAdhesionTraceRow row;
        row.attemptedStep = attemptedSteps++;
        row.phase = phase;
        row.phaseStep = phaseStep;
        row.elapsedSeconds = elapsedSeconds;
        row.relativeOpeningM = relativeOpeningM;
        row.commandedRelativeSpeedMPerS = commandedSpeed;
        const bool hasStatus = state.snapshot.statuses.size() == 1u &&
            state.snapshot.solverCertificates.size() == 1u;
        if (hasStatus) {
            row.statusCode = state.snapshot.statuses.front().code;
            const auto& certificate = state.snapshot.solverCertificates.front();
            row.certificateValid = certificate.validity.w > 0.5f;
            row.certificateResidual = certificate.nonlinear.x;
            row.certificateCorrection = certificate.nonlinear.y;
            row.certificateVolumeResidual = certificate.nonlinear.z;
            row.certificatePressureResidual = certificate.nonlinear.w;
        }
        row.accepted = hasStatus && pinNativeStepAccepted(state.snapshot, compiled.world);
        if (row.accepted) {
            const auto [remaining, selected] = measurePinGroupReactions(
                state.snapshot, compiled.world, fixture, arguments.timestepSeconds);
            row.acceptedStep = acceptedSteps;
            row.remainingBoardForceN = std::abs(
                remaining.signedImpulseZNs / arguments.timestepSeconds);
            row.selectedFacingForceN = std::abs(
                selected.signedImpulseZNs / arguments.timestepSeconds);
            row.instrumentLoadMagnitudeN = row.selectedFacingForceN;
            row.remainingBoardContactSamples = remaining.contactSamples;
            row.selectedFacingContactSamples = selected.contactSamples;
            ++acceptedSteps;
            for (const auto& face : state.snapshot.cohesiveFaces) {
                if (face.adjacency.y != 0u) continue;
                const std::uint32_t id = face.adjacency.z;
                const double damage = face.state.x;
                if (!std::isfinite(damage))
                    throw std::runtime_error("cohesive face has nonfinite accepted damage");
                if (!peakFaceDamage.contains(id) || damage > peakFaceDamage.at(id)) {
                    peakFaceDamage[id] = damage;
                    peakFaces[id] = face;
                }
            }
            if (workEligible)
                workSamples.push_back({row.acceptedStep,
                    row.relativeOpeningM, row.instrumentLoadMagnitudeN, true});
            if (permitCheckpoint) {
                const auto checkpointPath = arguments.output / "checkpoints" /
                    (std::string(checkpointRole) + ".nmcb");
                writePinCheckpoint(checkpointPath, state.snapshot,
                    compiled.world.physicsFingerprint, compiled.world.fingerprint,
                    checkpointRole);
                checkpointFiles.push_back(checkpointPath.filename().string());
                const auto objPath = arguments.output /
                    ("pin-accepted-" + std::string(checkpointRole) + ".obj");
                writeObj(objPath, compiled.world, state.snapshot.femNodes, 0u);
            }
        }
        writePinTraceRow(trace, row);
        trace.flush();
        require(trace.good(), "failed writing native pin trace row");
        rows.push_back(row);
        return row.accepted;
    };

    const double initialOpening = 2.0 * fixture.spec.initialClearanceM;
    const auto initialBodies = pinBodiesForInterval(
        fixture, initialOpening, initialOpening, arguments.timestepSeconds);
    try {
        const RunState seed = run.step(nativeControlStep++, targets, initialBodies);
        initialAccepted = appendSnapshotRow(seed, "initial_accepted_state", 0u,
            0.0, initialOpening, 0.0, false, true, "initial-accepted-state");
        if (!initialAccepted) {
            failed = true;
            failure = "initial zero-motion pin specimen state was rejected or uncertified";
        }
    } catch (const std::exception& error) {
        failed = true;
        failure = std::string("initial zero-motion pin specimen step failed: ") + error.what();
        PinAdhesionTraceRow row;
        row.attemptedStep = attemptedSteps++;
        row.phase = "initial_accepted_state";
        writePinTraceRow(trace, row);
        rows.push_back(row);
    }

    if (!failed && mesh.starchInterfaceParameters) {
        try {
            const RunState activation = run.step(
                nativeControlStep++, targets, initialBodies,
                mesh.starchActivationCommands);
            activationAccepted = appendSnapshotRow(
                activation, "initial_accepted_state_cohesive_activation", 0u,
                0.0, initialOpening, 0.0, false, true,
                "cohesive-activation-accepted-state");
            if (!activationAccepted) {
                failed = true;
                failure = "transactional initial_accepted_state cohesive activation failed or was rejected";
            }
        } catch (const std::exception& error) {
            failed = true;
            failure = std::string("pin cohesive activation transaction failed: ") + error.what();
            PinAdhesionTraceRow row;
            row.attemptedStep = attemptedSteps++;
            row.phase = "initial_accepted_state_cohesive_activation";
            writePinTraceRow(trace, row);
            rows.push_back(row);
        }
    }

    std::array<std::uint32_t, 4> phaseSeen{};
    std::optional<numi::cardboard::PinAdhesionPhase> lastPhase;
    for (const auto& point : schedule.points) {
        if (failed) break;
        const auto phaseIndex = static_cast<std::size_t>(point.phase);
        const bool isLastInPhase =
            point.phaseStep + 1u == phaseExpected[phaseIndex];
        const bool isApproachContactEndpoint =
            point.phase == numi::cardboard::PinAdhesionPhase::platenApproach &&
            point.endRelativeOpeningM == 0.0;
        const bool checkpointAtBoundary = isLastInPhase;
        std::string phaseLabel =
            numi::cardboard::pinAdhesionPhaseName(point.phase);
        if (lastPhase && *lastPhase != point.phase && phaseSeen[
                static_cast<std::size_t>(*lastPhase)] ==
                phaseExpected[static_cast<std::size_t>(*lastPhase)]) {
            lastPhase = point.phase;
        } else if (!lastPhase) {
            lastPhase = point.phase;
        }
        const auto bodies = pinBodiesForInterval(
            fixture, point.startRelativeOpeningM,
            point.endRelativeOpeningM, arguments.timestepSeconds);
        try {
            const RunState state = run.step(nativeControlStep++, targets, bodies);
            const bool workEligible =
                point.phase == numi::cardboard::PinAdhesionPhase::relativeOpening ||
                point.phase == numi::cardboard::PinAdhesionPhase::platenWithdrawal ||
                point.phase == numi::cardboard::PinAdhesionPhase::releasedHold;
            const bool accepted = appendSnapshotRow(
                state, phaseLabel, point.phaseStep, point.elapsedSeconds,
                point.endRelativeOpeningM,
                point.effectiveRelativeSpeedMPerSecond, workEligible,
                checkpointAtBoundary,
                std::string("phase-") + phaseLabel + "-accepted");
            if (accepted) {
                ++phaseAccepted[phaseIndex];
                if (isApproachContactEndpoint)
                    workSamples.push_back({acceptedSteps - 1u, 0.0,
                        rows.back().instrumentLoadMagnitudeN, true});
            } else {
                failed = true;
                failure = "native pin step " + std::to_string(point.controlStep) +
                    " was rejected or lacked an accepted solver certificate";
            }
        } catch (const std::exception& error) {
            failed = true;
            failure = "native pin step " + std::to_string(point.controlStep) +
                " failed: " + error.what();
            PinAdhesionTraceRow row;
            row.attemptedStep = attemptedSteps++;
            row.phase = phaseLabel;
            row.phaseStep = point.phaseStep;
            row.elapsedSeconds = point.elapsedSeconds;
            row.relativeOpeningM = point.endRelativeOpeningM;
            row.commandedRelativeSpeedMPerS =
                point.effectiveRelativeSpeedMPerSecond;
            writePinTraceRow(trace, row);
            rows.push_back(row);
        }
    }
    trace.close();

    std::ofstream failureLocus(arguments.output / "pin-cohesive-failure-locus.csv");
    require(failureLocus.good(), "cannot create pin interface-state output");
    failureLocus << "stableInterfaceId,lawKind,cohesiveMaterialIndex,rawFlags,"
                    "maximumAcceptedDamage,maximumAcceptedOpeningM,"
                    "tractionAtMaximumDamagePa,fractureWorkAtMaximumDamageJPerM2\n"
                 << std::setprecision(17);
    for (const auto& [identity, face] : peakFaces) {
        failureLocus << identity << ',' << face.lawIdentity.x << ','
                     << face.lawIdentity.y << ',' << face.adjacency.w << ','
                     << peakFaceDamage.at(identity) << ',' << face.state.y << ','
                     << face.state.z << ',' << face.state.w << '\n';
    }
    failureLocus.close();
    require(failureLocus.good(), "failed writing pin interface-state output");

    std::optional<numi::cardboard::PinAdhesionTraceMetrics> metrics;
    if (workSamples.size() >= 2u) {
        metrics = numi::cardboard::summarizePinAdhesionTrace(
            fixture, workSamples, false);
    }
    const std::array<const char*, 4> phases{
        "platen_approach", "relative_opening", "platen_withdrawal", "released_hold"};
    bool allPhasesComplete = true;
    for (std::size_t i = 0u; i < phases.size(); ++i)
        allPhasesComplete = allPhasesComplete && phaseExpected[i] == phaseAccepted[i];
    const bool completed = !failed && initialAccepted && activationAccepted &&
        allPhasesComplete && rows.size() >= schedule.points.size() + 1u;

    std::ofstream history(arguments.output / "pin-adhesion-history.json");
    require(history.good(), "cannot create pin history sidecar");
    history << std::setprecision(17)
            << "{\n  \"schema\": \"numi.cardboard.pin-adhesion-native-history.v1\",\n"
            << "  \"status\": \"" << (completed ? "completed" : "failed_prefix") << "\",\n"
            << "  \"terminal_reason\": \"" << jsonEscape(failure) << "\",\n"
            << "  \"manifest_sha256\": \"" << hexDigest(sha256File(
                    arguments.output / "manifest.json")) << "\",\n"
            << "  \"source_window_complete\": false,\n"
            << "  \"source_window_note\": \"Published test displacement origin and article-to-presentation point correspondence remain unresolved.\",\n"
            << "  \"condition_id\": \"native-forward-engineering-prototype\",\n"
            << "  \"native_source_physics_fingerprint\": " << compiled.world.physicsFingerprint << ",\n"
            << "  \"native_device_program_fingerprint\": " << compiled.world.fingerprint << ",\n"
            << "  \"protocol_parameters\": {\"specimen_width_m\": "
            << fixture.spec.specimenWidthM << ", \"specimen_length_m\": "
            << fixture.spec.specimenLengthM << ", \"board_caliper_m\": "
            << fixture.spec.boardCaliperM << ", \"relative_platen_rate_m_per_s\": "
            << fixture.spec.relativePlatenRateMPerSecond
            << ", \"timestep_s\": " << arguments.timestepSeconds
            << ", \"maximum_relative_opening_m\": "
            << arguments.pinMaximumOpeningM << ", \"withdrawal_distance_m\": "
            << arguments.pinWithdrawalDistanceM << ", \"schedule_step_count\": "
            << schedule.points.size() << ", \"schedule_duration_s\": "
            << schedule.totalDurationSeconds << ", \"pin_count\": "
            << fixture.pins.size() << ", \"pin_radius_m\": "
            << fixture.spec.pinRadiusM << ", \"pin_pitch_m\": "
            << fixture.spec.pinPitchAcrossWidthM << ", \"pin_end_overhang_m\": "
            << fixture.spec.pinEndOverhangM << ", \"initial_clearance_m\": "
            << fixture.spec.initialClearanceM << "},\n"
            << "  \"material_inputs\": {\"liner_path\": \""
            << jsonEscape(arguments.linerMaterial.string()) << "\", \"liner_sha256\": \""
            << hexDigest(sha256File(arguments.linerMaterial))
            << "\", \"medium_path\": \""
            << jsonEscape(arguments.mediumMaterial.string()) << "\", \"medium_sha256\": \""
            << hexDigest(sha256File(arguments.mediumMaterial))
            << "\", \"glue_path\": \""
            << jsonEscape(arguments.glueMaterial.string()) << "\", \"glue_sha256\": \""
            << hexDigest(sha256File(arguments.glueMaterial))
            << "\", \"starch_parameter_path\": \""
            << jsonEscape(arguments.starchInterfaceParameters.string())
            << "\", \"starch_parameter_sha256\": \""
            << arguments.starchInterfaceParametersSha256 << "\"},\n"
            << "  \"trace_csv_sha256\": \"" << hexDigest(sha256File(
                    arguments.output / "pin-adhesion-trace.csv")) << "\",\n"
            << "  \"phase_completion\": {\n"
            << "    \"initial_accepted_state\": " << (initialAccepted ? "true" : "false") << ",\n"
            << "    \"cohesive_activation\": " << (activationAccepted ? "true" : "false") << ",\n";
    for (std::size_t i = 0u; i < phases.size(); ++i)
        history << "    \"" << phases[i] << "\": "
                << (phaseExpected[i] == phaseAccepted[i] ? "true" : "false")
                << (i + 1u == phases.size() ? "\n" : ",\n");
    history << "  },\n  \"completed_phases\": [";
    bool firstPhase = true;
    for (std::size_t i = 0u; i < phases.size(); ++i) {
        if (phaseExpected[i] != phaseAccepted[i]) continue;
        if (!firstPhase) history << ',';
        history << '"' << phases[i] << '"';
        firstPhase = false;
    }
    history << "],\n  \"phase_counts\": {";
    for (std::size_t i = 0u; i < phases.size(); ++i) {
        if (i) history << ',';
        history << "\"" << phases[i] << "\": {\"accepted\": "
                << phaseAccepted[i] << ", \"planned\": "
                << phaseExpected[i] << '}';
    }
    history << "},\n  \"samples\": [\n";
    for (std::size_t i = 0u; i < rows.size(); ++i) {
        const auto& row = rows[i];
        history << "    {\"attemptedStep\": " << row.attemptedStep
                << ", \"acceptedStep\": ";
        if (row.accepted) history << row.acceptedStep;
        else history << "null";
        history << ", \"phase\": \"" << jsonEscape(row.phase)
                << "\", \"accepted\": " << (row.accepted ? "true" : "false")
                << ", \"elapsedS\": " << row.elapsedSeconds
                << ", \"relativeOpeningM\": " << row.relativeOpeningM
                << ", \"commandedRelativeSpeedMPerS\": "
                << row.commandedRelativeSpeedMPerS
                << ", \"instrumentLoadMagnitudeN\": "
                << (row.accepted ? row.instrumentLoadMagnitudeN : 0.0)
                << ", \"statusCode\": " << row.statusCode
                << ", \"certificateValid\": "
                << (row.certificateValid ? "true" : "false") << "}"
                << (i + 1u == rows.size() ? "\n" : ",\n");
    }
    history << "  ],\n  \"metrics\": {\"available\": "
            << (metrics.has_value() ? "true" : "false")
            << ", \"acceptedOpeningSamples\": "
            << (metrics ? metrics->acceptedPointCount : 0u)
            << ", \"peakMeasuredLoadN\": "
            << (metrics ? metrics->peakMeasuredLoadN : 0.0)
            << ", \"peakAcceptedStep\": "
            << (metrics ? metrics->peakAcceptedStep : 0u)
            << ", \"integratedWorkNm\": "
            << (metrics ? metrics->integratedWorkNm : 0.0)
            << ", \"integratedWorkNmm\": "
            << (metrics ? metrics->integratedWorkNmm : 0.0)
            << ", \"pinStrengthNPerMm\": "
            << (metrics ? metrics->pinStrengthNPerMm : 0.0)
            << ", \"sourceWindowComplete\": false},\n"
            << "  \"calibration_qualified\": false, \"physical_validation\": false, \"source_failure_locus_available\": false,\n"
            << "  \"failure_locus_file\": \"pin-cohesive-failure-locus.csv\", \"failure_locus_sha256\": \""
            << hexDigest(sha256File(arguments.output / "pin-cohesive-failure-locus.csv"))
            << "\", \"failure_locus_raw_interface_count\": " << peakFaces.size() << ",\n"
            << "  \"accepted_checkpoint_files\": [";
    for (std::size_t i = 0u; i < checkpointFiles.size(); ++i) {
        if (i) history << ',';
        history << '"' << jsonEscape(checkpointFiles[i]) << '"';
    }
    history << "]\n}\n";
    require(history.good(), "failed writing pin native history sidecar");

    writeSummaryJson(arguments.output / "result.json",
        completed ? "completed" : "failed", mesh, compiled,
        static_cast<std::uint32_t>(acceptedSteps), failure);
    printResult(completed ? "completed" : "failed", compiled,
                static_cast<std::uint32_t>(acceptedSteps), failure);
    return completed ? 0 : 2;
}

int execute(Arguments arguments) {
    if (arguments.shippingBox) {
        require(arguments.shippingDesign.has_value(),
                "shipping design was not resolved before mesh construction");
        prepareOutput(arguments.output);
        const std::uint32_t shippingYIntervals =
            3u * arguments.boxConfig.subdivisionsPerBand;
        const auto prebuildScreen =
            numi::cardboard::screenShippingPrebuildMemory(
                arguments.shippingDesign->perimeterSpanM, 0.0079, 1u,
                arguments.fgmresRestart, shippingYIntervals,
                arguments.thicknessSlices);
        const auto prebuildBudget = shippingPrebuildMemoryBudget(
            arguments, prebuildScreen);
        writeShippingPrebuildMemoryScreen(
            arguments.output / "shipping-memory-prebuild.json",
            arguments, prebuildScreen, prebuildBudget);
        const bool shippingPhysicsRequested = arguments.boxAssemblyNative &&
            !arguments.compileOnly;
        if (shippingPhysicsRequested &&
            (!prebuildBudget.withinLimit ||
            prebuildScreen.staticArenaScreen.totalBytes >
                numi::cardboard::kShippingPeakMemoryLimitBytes)) {
            const bool descentUnavailable =
                !prebuildBudget.requiredResidualDescentArenaAdmitted;
            const std::string reason = descentUnavailable
                ? "required full-contact residual descent exceeds current native caps at pre-mesh shipping cardinality; full mesh and physics were not attempted"
                : "pre-mesh shipping cardinality screen exceeds 16 GiB planning limit; full mesh construction was not attempted";
            std::ofstream failure(arguments.output / "failure.txt");
            failure << reason << '\n';
            std::ofstream result(arguments.output / "result.json");
            result << "{\n  \"schema\": \"numi.cardboard.prebuild-screen-result.v1\",\n"
                   << "  \"status\": \""
                   << (descentUnavailable
                           ? "required_residual_descent_unavailable_prebuild"
                           : "prebuild_estimate_exceeds_memory_limit")
                   << "\",\n"
                   << "  \"mesh_built\": false,\n  \"native_runtime_initialized\": false,\n"
                   << "  \"required_residual_descent\": true,\n"
                   << "  \"runtime_arena_caps_admitted\": "
                   << (prebuildBudget.requiredResidualDescentArenaAdmitted
                           ? "true" : "false") << ",\n"
                   << "  \"private_full_contact_bytes\": "
                   << prebuildBudget.requiredFullContactArmijoBytes << ",\n"
                   << "  \"private_rigid_contact_bytes\": "
                   << prebuildBudget.requiredRigidContactArmijoBytes << ",\n"
                   << "  \"private_cohesive_bytes\": "
                   << prebuildBudget.requiredCohesiveArmijoBytes << ",\n"
                   << "  \"estimated_static_arena_bytes\": "
                   << prebuildScreen.staticArenaScreen.totalBytes << ",\n"
                   << "  \"reason\": \"" << jsonEscape(reason) << "\",\n"
                   << "  \"qualification_allowed\": false\n}\n";
            return 2;
        }
    }
    const Digest linerDigest = sha256File(arguments.linerMaterial);
    const Digest mediumDigest = sha256File(arguments.mediumMaterial);
    MaterialProgram liner = readMaterial(arguments.linerMaterial, "liner");
    MaterialProgram medium = readMaterial(arguments.mediumMaterial, "medium");
    if (arguments.linerDamageSet && arguments.mediumDamageSet) {
        require(!numi::cardboard::paper_materials::isBilinearFitIdentity(liner.name) &&
                    !numi::cardboard::paper_materials::isBilinearFitIdentity(medium.name),
                "retained-matrix damage overlays cannot use bilinear tension-fit assets because the generated overlay already supplies hardening");
    }
    if (arguments.safeguardedPaperInitializer) {
        liner.paperHillScalarInitializer = true;
        medium.paperHillScalarInitializer = true;
    }
    if (arguments.compensatedPaperArithmetic) {
        liner.paperCompensatedArithmetic = true;
        medium.paperCompensatedArithmetic = true;
    }
    if (arguments.accuratePaperProjection) {
        liner.paperHillAccurateProjection = true;
        medium.paperHillAccurateProjection = true;
    }
    if (arguments.scalePaperProjectionResiduals) {
        arguments.linerProjectionScales =
            numi::cardboard::paperProjectionRowScales(liner);
        arguments.mediumProjectionScales =
            numi::cardboard::paperProjectionRowScales(medium);
        liner = numi::cardboard::scalePaperProjectionResiduals(std::move(liner));
        medium = numi::cardboard::scalePaperProjectionResiduals(std::move(medium));
    }
    std::optional<MaterialProgram> glue;
    std::string glueDigest;
    if (!arguments.glueMaterial.empty()) {
        glue = readMaterial(arguments.glueMaterial, "glue");
        glueDigest = hexDigest(sha256File(arguments.glueMaterial));
        if (arguments.compensatedPaperArithmetic)
            glue->paperCompensatedArithmetic = true;
    }
    MeshSource mesh = buildMesh(
        arguments, liner, medium, hexDigest(linerDigest), hexDigest(mediumDigest),
        glue ? &*glue : nullptr, glueDigest);
    if (arguments.shippingBox) {
        arguments.shippingBlankGeometrySha256 =
            shippingBlankGeometryBindingDigest(mesh);
        bindShippingRoofMeasurementToCurrentSource(arguments, mesh);
    }
    if (arguments.linerDamageSet && arguments.mediumDamageSet) {
        applyRetainedMatrixDamageOverlay(
            mesh, liner, medium, hexDigest(linerDigest),
            hexDigest(mediumDigest), glueDigest, arguments);
    }
    if (arguments.compensatedImplicitProjection) {
        require(!mesh.damageOverlayMaterialIndices.empty(),
                "compensated implicit projection has no generated retained-matrix overlays");
        for (const std::uint32_t materialIndex :
             mesh.damageOverlayMaterialIndices) {
            require(materialIndex < mesh.world.materials.size(),
                    "retained-matrix compensated-projection material index is outside the compiled material table");
            mesh.world.materials[materialIndex]
                .genericCompensatedImplicitProjection = true;
        }
    }

    CompileResult compiled = compileWorld(
        mesh.world, {.maximumRateExponent = 0u, .emitSpecializedMetal = false,
                     .localMaterialNewtonIterations = arguments.localMaterialIterations});
    require(compiled.succeeded(),
            "Matter cardboard source compile failed: " +
                diagnosticText(compiled.diagnostics));
    if (arguments.shippingBox) {
        require(mesh.shippingImpactorProxyIndex <
                    compiled.world.contact.rigidProxies.size(),
                "compiled shipping world omitted the explicit impactor proxy");
        const auto& impactorProxy = compiled.world.contact.rigidProxies[
            mesh.shippingImpactorProxyIndex];
        const std::string mismatch =
            numi::cardboard::shippingImpactorProxyMismatchReason(
                impactorProxy, mesh.shippingImpactorProxyIndex,
                arguments.shippingImpactorHalfExtentsM);
        require(mismatch.empty(), mismatch);
        arguments.shippingImpactorProxyIndex = mesh.shippingImpactorProxyIndex;
        arguments.shippingImpactorCompiledProxyMatchesMeasurement = true;
    }
    double compiledSubstrateMassKg = 0.0;
    if (arguments.shippingBox) {
        const auto& gravity = compiled.world.dispatch.gravityAndTimestep;
        require(gravity.x == 0.0f &&
                    gravity.y == static_cast<float>(-9.80665) &&
                    gravity.z == 0.0f,
                "shipping FEM dispatch must compile full board self-weight before any native stage");
        compiledSubstrateMassKg = compiledSubstrateNodalMassKg(compiled.world);
        double tetConstituentMassKg = 0.0;
        for (const MaterialTotals& total : calculateTotals(
                 compiled.world, mesh.world))
            tetConstituentMassKg += total.massKg;
        require(std::isfinite(compiledSubstrateMassKg) &&
                    compiledSubstrateMassKg > 0.0 &&
                    std::isfinite(tetConstituentMassKg) &&
                    tetConstituentMassKg > 0.0 &&
                    std::abs(compiledSubstrateMassKg - tetConstituentMassKg) <=
                        2.0e-5 * std::max(1.0, tetConstituentMassKg),
                "shipping FEM nodal gravity mass does not match the authored tetrahedron constituent masses");
    }

    if (!arguments.shippingBox) prepareOutput(arguments.output);
    std::optional<numi::cardboard::ShippingStaticMemoryLowerBound>
        shippingStaticLowerBound;
    std::optional<numi::cardboard::ShippingMemoryBudget>
        shippingArenaBudget;
    if (arguments.shippingBox) {
        shippingStaticLowerBound = shippingStaticMemoryLowerBound(compiled.world);
        shippingArenaBudget = numi::cardboard::estimateShippingMemoryBudget(
            shippingMemoryBudgetInput(arguments, mesh, compiled));
        writeShippingMemoryReport(
            arguments.output / "shipping-memory-preflight.json",
            arguments, mesh, compiled.world, *shippingStaticLowerBound, nullptr,
            &*shippingArenaBudget);
        writeShippingDesignReport(
            arguments.output / "shipping-design.json", arguments, mesh,
            compiled.world, *shippingStaticLowerBound);
        const auto descentAdmission = shippingResidualDescentAdmission(
            compiled.world, *shippingArenaBudget);
        auto shippingDescentAdmission = descentAdmission;
        if (arguments.boxAssemblyNative &&
            shippingDescentAdmission.physicsExecutionAdmitted) {
            constexpr const char* incompatibleSelectors[] = {
                "NM_FEM_RESIDUAL_ARMIJO_ROOT",
                "NM_FEM_COHESIVE_RESIDUAL_ARMIJO_ROOT",
                "NM_FEM_CAPSULE_RESIDUAL_ARMIJO_ROOT",
                "NM_FEM_CAPSULE_BOX_RESIDUAL_ARMIJO_ROOT"};
            for (const char* name : incompatibleSelectors) {
                if (std::getenv(name) != nullptr) {
                    shippingDescentAdmission.physicsExecutionAdmitted = false;
                    shippingDescentAdmission.blockers.emplace_back(
                        std::string("shipping full-contact descent conflicts with ") + name);
                }
            }
            if (shippingDescentAdmission.physicsExecutionAdmitted &&
                ::setenv("NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT", "all", 1) != 0) {
                shippingDescentAdmission.physicsExecutionAdmitted = false;
                shippingDescentAdmission.blockers.emplace_back(
                    "could not select full-contact residual descent for every shipping physics transaction");
            }
        }
        writeShippingResidualDescentAdmission(
            arguments.output / "shipping-residual-descent-admission.json",
            compiled.world, *shippingArenaBudget, shippingDescentAdmission);
        if (arguments.boxAssemblyNative &&
            !shippingDescentAdmission.physicsExecutionAdmitted) {
            const std::string reason =
                "shipping residual-descent runtime scope, required arenas, memory, or selector is unavailable; native physics was not initialized";
            std::ofstream failure(arguments.output / "failure.txt");
            failure << reason << '\n';
            writeSummaryJson(arguments.output / "result.json",
                "required_residual_descent_unavailable", mesh, compiled, 0u,
                reason);
            return 2;
        }
        if (arguments.shippingPreflightOnly &&
            !shippingDescentAdmission.physicsExecutionAdmitted) {
            const std::string reason =
                "shipping preflight found unsupported residual-descent scope or arenas; native runtime was not initialized";
            std::ofstream failure(arguments.output / "failure.txt");
            failure << reason << '\n';
            writeSummaryJson(arguments.output / "result.json",
                "shipping_preflight_unavailable_no_native_init", mesh,
                compiled, 0u, reason);
            return 2;
        }
        if (shippingStaticLowerBound->totalBytes >
                numi::cardboard::kShippingPeakMemoryLimitBytes &&
            !arguments.compileOnly) {
            const std::string reason =
                "static known-arena lower bound exceeds the 16 GiB shipping limit; native runtime was not initialized";
            std::ofstream failure(arguments.output / "failure.txt");
            failure << reason << '\n';
            writeSummaryJson(arguments.output / "result.json",
                             "static_memory_preflight_rejected", mesh,
                             compiled, 0u, reason);
            return 2;
        }
        if (!shippingArenaBudget->withinLimit && !arguments.compileOnly) {
            const std::string reason =
                "compiled shipping arena envelope exceeds 16 GiB; native runtime was not initialized";
            std::ofstream failure(arguments.output / "failure.txt");
            failure << reason << '\n';
            writeSummaryJson(arguments.output / "result.json",
                             "arena_memory_preflight_rejected", mesh,
                             compiled, 0u, reason);
            return 2;
        }
    }
    writePackageOrThrow(compiled, arguments.output / "compiled.nmatterpack");
    writeInitialObj(arguments.output / "initial.obj", compiled.world);
    writeMeshJson(arguments.output / "mesh.json", arguments, mesh);
    if (mesh.damageOverlayProgramCount > 0u) {
        writeRetainedMatrixDamageOverlayJson(
            arguments.output / "damage-overlay-program-pool.json",
            arguments, mesh);
    }
    writeBoxLayout(arguments.output / "box-layout.json", mesh);
    writeManifest(arguments.output / "manifest.json", arguments, mesh,
                  compiled, linerDigest, mediumDigest, glueDigest);
    if (arguments.compileOnly) {
        writeSummaryJson(arguments.output / "result.json", "compile_only",
                         mesh, compiled, 0u);
        printResult("compile_only", compiled, 0u);
        return 0;
    }

    if (arguments.shippingPreflightOnly) {
        NativeRun run(compiled.world, arguments.metallib, true);
        {
            std::ofstream memory(arguments.output / "memory-observations.csv");
            require(memory.good(), "cannot create shipping memory observations");
            memory << "native_step,stage,runtime_resident_bytes,metal_current_allocated_bytes,metal_recommended_working_set_bytes,host_phys_footprint_bytes,host_sample_available,largest_measured_domain_bytes,within_16gib\n";
        }
        writeShippingMemoryReport(
            arguments.output / "shipping-memory-native-preflight.json",
            arguments, mesh, compiled.world, *shippingStaticLowerBound,
            &run.lastMemoryObservation(), &*shippingArenaBudget);
        run.writeMemoryObservation(arguments.output / "memory-observations.csv",
                                   0u, "initialized_no_step");
        const bool withinLimit = run.shippingMemoryWithinLimit();
        const std::string status = withinLimit
            ? "native_preflight_only_no_physics_step"
            : "native_memory_preflight_rejected";
        writeSummaryJson(arguments.output / "result.json", status, mesh,
                         compiled, 0u,
                         withinLimit ? std::string{} :
                            "initialized memory observations exceed 16 GiB");
        return withinLimit ? 0 : 2;
    }

    if (arguments.pinAdhesion)
        return executePinAdhesion(arguments, mesh, compiled);

    if (arguments.boxAssemblyNative)
        return executeBoxAssemblyNative(
            arguments, mesh, compiled,
            shippingArenaBudget ? &*shippingArenaBudget : nullptr);

    NativeRun run(compiled.world, arguments.metallib, arguments.shippingBox);
    // These are the last Newton linearization, including after rollback. They
    // are diagnostic conditioning data, never accepted physical geometry.
    const auto writeRejectedRegionalConditioning = [&](const std::uint32_t step,
            const RuntimeStateSnapshot& snapshot) {
        if (!snapshot.femRegionalGalerkinDiagnosticsAvailable) return;
        const auto directory = arguments.output /
            ("rejected-conditioning-" + stepLabel(step + 1u));
        std::filesystem::create_directories(directory);
        std::ofstream metadata(directory / "manifest.json");
        require(metadata.good(), "cannot write rejected conditioning metadata");
        metadata << "{\n  \"schema\": \"numi.cardboard.regional-conditioning.v1\",\n"
            << "  \"accepted_physical_state\": false,\n"
            << "  \"control_step\": " << step << ",\n"
            << "  \"source_physics_fingerprint\": \""
            << snapshot.sourcePhysicsFingerprint << "\",\n"
            << "  \"device_program_fingerprint\": \""
            << snapshot.deviceProgramFingerprint << "\",\n"
            << "  \"environment_count\": " << arguments.environmentCount << ",\n"
            << "  \"fem_node_count\": " << compiled.world.dispatch.femNodeCount << ",\n"
            << "  \"mode_count\": " << snapshot.femRegionalGalerkinModeCount << ",\n"
            << "  \"byte_order\": \"native_little_endian\",\n"
            << "  \"layout\": \"P mode-node; effective_P/AP environment-mode-node; RAP pivoted LU environment-row-column; block_inverse environment-node-row\",\n"
            << "  \"files\": [\n";
        bool first = true;
        const auto writeArena = [&](const char* name, const auto& values) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            const auto filename = std::string(name) + ".bin";
            std::ofstream arena(directory / filename, std::ios::binary);
            require(arena.good(), "cannot write rejected conditioning arena");
            const std::size_t bytes = values.size() * sizeof(Value);
            if (bytes != 0u) arena.write(reinterpret_cast<const char*>(values.data()), bytes);
            require(arena.good(), "rejected conditioning arena write failed");
            if (!first) metadata << ",\n";
            first = false;
            metadata << "    {\"name\": \"" << filename << "\", \"count\": "
                << values.size() << ", \"stride_bytes\": " << sizeof(Value)
                << ", \"bytes\": " << bytes << ", \"sha256\": \""
                << sha256Arena(values) << "\"}";
        };
        writeArena("candidate_nodes", snapshot.femRegionalGalerkinLinearizationNodes);
        writeArena("P", snapshot.femRegionalGalerkinBasisP);
        writeArena("effective_P", snapshot.femRegionalGalerkinEffectiveBasisP);
        writeArena("AP", snapshot.femRegionalGalerkinOperatorAP);
        writeArena("pivoted_RAP", snapshot.femRegionalGalerkinPivotedRAP);
        writeArena("pivots", snapshot.femRegionalGalerkinPivots);
        writeArena("factor_valid", snapshot.femRegionalGalerkinFactorValid);
        writeArena("node_block_inverse", snapshot.femRegionalGalerkinNodeBlockInverse);
        metadata << "\n  ]\n}\n";
        require(metadata.good(), "rejected conditioning metadata write failed");
    };
    std::ofstream observations(arguments.output / "observations.csv");
    require(observations.good(), "cannot create observations.csv");
    writeHeader(observations);
    std::ofstream toolObservations;
    if (arguments.crease) {
        toolObservations.open(arguments.output / "tool-observations.csv");
        require(toolObservations.good(), "cannot create tool observations");
        toolObservations << "step,environment,commanded_end_travel_m,commanded_speed_m_s,punch_force_z_N,punch_contact_count,min_end_punch_node_gap_m,min_anvil_node_gap_m,step_accepted\n";
        toolObservations << std::setprecision(17);
    }
    bool failed = false;
    std::string failure;
    std::uint32_t acceptedSteps = 0u;
    for (std::uint32_t step = 0u; step < totalSteps(arguments); ++step) {
        std::vector<nm_float4> targets;
        fillTargets(arguments, mesh, step, targets);
        RunState current;
        try {
            current = run.step(step, targets, toolingBodies(arguments, mesh, step));
        } catch (const std::exception& error) {
            failed = true;
            failure = "step " + std::to_string(step) + ": " + error.what();
            std::ofstream diagnostic(arguments.output / "failure.txt");
            diagnostic << failure << '\n';
            break;
        }
        require(current.snapshot.statuses.size() >= arguments.environmentCount,
                "Matter status readback is truncated");
        const LoadPoint load = loadingPoint(arguments, step);
        const double targetAngle = load.angleDegrees;
        bool stepFailed = false;
        bool toolEndPoseFailed = false;
        for (std::uint32_t environment = 0u;
             environment < arguments.environmentCount; ++environment) {
            const Metrics metrics = stateMetrics(
                compiled.world, current.snapshot, environment);
            const NMMatterStatusGPU& status =
                current.snapshot.statuses[environment];
            const auto& certificate = current.snapshot.solverCertificates.at(environment);
            const bool accepted = status.code == NM_STATUS_SUCCESS && certificate.validity.w > 0.5f &&
                std::isfinite(certificate.nonlinear.x) && std::isfinite(certificate.nonlinear.y) &&
                std::isfinite(certificate.nonlinear.z) && std::isfinite(certificate.nonlinear.w) &&
                certificate.nonlinear.x <= compiled.world.mixedSolver.residualTolerances.x &&
                certificate.nonlinear.z <= compiled.world.mixedSolver.residualTolerances.y &&
                certificate.nonlinear.w <= compiled.world.mixedSolver.residualTolerances.z;
            ReactionMetrics reaction;
            if (accepted) {
                reaction = reactionMetrics(
                    environment, current.reactions, current.contactReactions,
                    current.snapshot, mesh);
            }
            double forceNorm2 = 0.0, forceMax = 0.0;
            for (std::size_t n = 0; n < mesh.restNodes.size(); ++n) {
                const auto i = environment * mesh.restNodes.size() + n;
                if (current.snapshot.femNodes.at(i).restAndFixed.w != 0.0f) continue;
                const auto r = current.residuals.at(i);
                const double norm2 = (double(r.x)*r.x + double(r.y)*r.y + double(r.z)*r.z) /
                    (run.runtime().timestepSeconds() * run.runtime().timestepSeconds());
                forceNorm2 += norm2; forceMax = std::max(forceMax, std::sqrt(norm2));
            }
            writeObservation(observations, step, run.runtime().timestepSeconds(),
                             environment, environment == 0u ? targetAngle : 0.0,
                             status, metrics, reaction, load.phase,
                             certificate, accepted, measuredGripAngle(mesh, current.snapshot, environment), arguments.crease, std::sqrt(forceNorm2), forceMax);
            const std::string arm = arguments.crease ? (environment == 0u ? "indented" : "stationary_tool_reference") : (environment == 0u ? "bent" : "held_reference");
            writeObj(arguments.output /
                         ("accepted_step_" + stepLabel(step + 1u) + "_" +
                          arm + ".obj"),
                     compiled.world, current.snapshot.femNodes, environment);
            if (step + 1u == totalSteps(arguments) || !accepted ||
                std::string(loadingPoint(arguments, step + 1u).phase) != load.phase)
                writeMaterialState(arguments.output / ("material_state_" + stepLabel(step + 1u) + "_" + arm + ".json"),
                    compiled.world, mesh.world, current.snapshot, environment);
            if (mesh.tooling) {
                const auto& geometry = *mesh.tooling;
                auto endPose = geometry.punchInitialPose;
                const double travel = environment == 0u ? indentationAtEnd(arguments, step) : 0.0;
                const double previous = environment == 0u && step > 0u ? indentationAtEnd(arguments, step - 1u) : 0.0;
                endPose.translation.z -= travel;
                double minPunch = INFINITY, minAnvil = INFINITY, impulseZ = 0.0;
                std::uint32_t contacts = 0u;
                const auto nodeBase = environment * mesh.restNodes.size();
                for (std::size_t n = 0; n < mesh.restNodes.size(); ++n) {
                    const auto x = current.snapshot.femNodes.at(nodeBase + n).positionAndMass;
                    const numi::cardboard::ToolVec3 point{x.x, x.y, x.z};
                    minPunch = std::min(minPunch, numi::cardboard::capsuleContactWitness(geometry.punch, endPose, point).separation);
                    minAnvil = std::min(minAnvil, numi::cardboard::boxContactWitness(geometry.support, {}, point).separation);
                }
                require(current.snapshot.contactSamples.size() % arguments.environmentCount == 0u, "contact sample arena shape");
                const auto sampleStride = current.snapshot.contactSamples.size() / arguments.environmentCount;
                for (std::size_t i = 0; i < sampleStride; ++i) {
                    const auto& sample = current.snapshot.contactSamples[environment * sampleStride + i];
                    if (sample.identity.y == 0u && (sample.identity.w & NM_CONTACT_VALID) != 0u) {
                        ++contacts; impulseZ += sample.impulseAndNormal.z;
                    }
                }
                // Independently audit the prescribed endpoint used by both
                // the native Newton solve and the postCommit certificate.
                const double roundoff = 8.0 * std::numeric_limits<float>::epsilon() *
                    std::max({arguments.lengthM, arguments.widthM, arguments.totalHeightM, arguments.punchRadiusM});
                if (accepted && (!std::isfinite(minPunch) || !std::isfinite(minAnvil) ||
                                 minPunch < -roundoff || minAnvil < -roundoff)) toolEndPoseFailed = true;
                toolObservations << step << ',' << environment << ',' << travel << ','
                    << -(travel - previous) / arguments.timestepSeconds << ',' << impulseZ / run.runtime().timestepSeconds()
                    << ',' << contacts << ',' << minPunch << ',' << minAnvil << ',' << (accepted ? 1 : 0) << '\n';
            }
            stepFailed = stepFailed || !accepted;
        }
        if (stepFailed) {
            writeRejectedRegionalConditioning(step, current.snapshot);
            failed = true;
            failure = "Matter rejected step " + std::to_string(step) +
                "; status, accepted state OBJ, and CSV observations were retained";
            break;
        }
        ++acceptedSteps;
        if (toolEndPoseFailed) {
            for (std::uint32_t e = 0u; e < arguments.environmentCount; ++e) {
                const std::string arm = e == 0u ? "indented" : "stationary_tool_reference";
                writeMaterialState(arguments.output / ("material_state_" + stepLabel(step + 1u) + "_" + arm + ".json"),
                    compiled.world, mesh.world, current.snapshot, e);
            }
            failed = true;
            failure = "native step " + std::to_string(step) +
                " accepted, but commanded end-pose tooling gap audit failed; "
                "native accepted state retained, tool trajectory not qualified";
            std::ofstream diagnostic(arguments.output / "failure.txt");
            diagnostic << failure << '\n';
            break;
        }
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
            if (!arguments.shippingCapacityQualificationFile.empty())
                loadShippingCapacityQualification(arguments);
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

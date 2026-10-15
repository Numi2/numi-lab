#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>

#include "cardboard_tooling.hpp"
#include "cardboard_paper_projection_scaling.hpp"
#include "cardboard_shipping_snapshot.hpp"
#if defined(NUMI_CARDBOARD_METAL_WORLD)
#include "cardboard_shipping_rigid_driver.hpp"
#endif
#include "metalrobo/engine_types.h"
#include "numi/matter/matter.hpp"
#include "numi/matter/sparse_private_contact_admission.hpp"
#include "numi/matter/full_contact_armijo_admission.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <unistd.h>

#ifndef NUMI_MATTER_METALLIB
#define NUMI_MATTER_METALLIB ""
#endif

namespace {
using namespace numi::matter;
using namespace numi::cardboard;

constexpr std::uint32_t kEnvironmentCount = 1u;
constexpr std::uint32_t kPunchBody = 0u;
constexpr std::uint32_t kAnvilBody = 1u;
constexpr std::uint32_t kFaceId = 17u;
constexpr double kMm = 1.0e-3;
constexpr double kPadLength = 4.0 * kMm;
constexpr double kPadWidth = 4.0 * kMm;
constexpr double kPadThickness = 2.0 * kMm;
constexpr double kPunchRadius = 0.5 * kMm;
constexpr double kInitialClearance = 30.0e-6;
constexpr double kSupportClearance = 20.0e-6;
constexpr double kToolSpeed = 0.2;
constexpr double kActivationGap = 25.0e-6;
constexpr double kFreeBodyMortarGap = 5.0e-6;
constexpr double kActivationTolerance = 100.0e-6;
unsigned checks = 0u;
std::uint64_t nativeFEMTransactionsExecuted = 0u;
bool extendedBudget = false;
bool accuratePaper = false;
bool regionalContactPreconditioner = false;
std::uint32_t authoredNewtonIterations = 0u;

std::string diagnosticStateOutputPath;

std::string indexedDiagnosticStem(const std::uint64_t index) {
    std::ostringstream stem;
    stem << "transaction-" << std::setw(6) << std::setfill('0') << index;
    return stem.str();
}

void writeRawDiagnosticSnapshot(const std::filesystem::path& path,
                                const RuntimeStateSnapshot& snapshot) {
    std::ofstream output(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!output) throw std::runtime_error("could not create raw diagnostic snapshot");
    shipping_snapshot_detail::writeSnapshotPayload(output, snapshot);
    // The shipping payload predates the optional regional-Galerkin readback
    // fields in RuntimeStateSnapshot. Append those fields under an explicit
    // suffix marker so this raw diagnostic remains complete without becoming
    // an accepted .nmcb checkpoint or changing that checkpoint format.
    constexpr std::array<char, 8> diagnosticsTag{
        'N', 'M', 'R', 'A', 'P', 'D', 'I', '1'};
    output.write(diagnosticsTag.data(),
                 static_cast<std::streamsize>(diagnosticsTag.size()));
    shipping_snapshot_detail::writePod(output,
        static_cast<std::uint8_t>(snapshot.available));
    shipping_snapshot_detail::writePod(output,
        static_cast<std::uint8_t>(snapshot.femRegionalGalerkinDiagnosticsAvailable));
    shipping_snapshot_detail::writePod(output,
        snapshot.femRegionalGalerkinModeCount);
    shipping_snapshot_detail::writeString(output,
        snapshot.femRegionalGalerkinDiagnosticsMessage);
    shipping_snapshot_detail::writeVector(output,
        snapshot.femRegionalGalerkinLinearizationNodes);
    shipping_snapshot_detail::writeVector(output,
        snapshot.femRegionalGalerkinBasisP);
    shipping_snapshot_detail::writeVector(output,
        snapshot.femRegionalGalerkinEffectiveBasisP);
    shipping_snapshot_detail::writeVector(output,
        snapshot.femRegionalGalerkinOperatorAP);
    shipping_snapshot_detail::writeVector(output,
        snapshot.femRegionalGalerkinPivotedRAP);
    shipping_snapshot_detail::writeVector(output,
        snapshot.femRegionalGalerkinPivots);
    shipping_snapshot_detail::writeVector(output,
        snapshot.femRegionalGalerkinFactorValid);
    shipping_snapshot_detail::writeVector(output,
        snapshot.femRegionalGalerkinNodeBlockInverse);
    output.flush();
    if (!output) throw std::runtime_error("raw diagnostic snapshot write failed");
}

template<class T>
void writeDiagnosticVectorCount(std::ostream& output,
                                const char* name,
                                const std::vector<T>& values,
                                bool& first) {
    if (!first) output << ',';
    first = false;
    output << '"' << name << "\":" << values.size();
}

template<class T>
void writeDiagnosticVectorHash(std::ostream& output,
                               const char* name,
                               const std::vector<T>& values,
                               bool& first) {
    const std::size_t bytes = values.size() * sizeof(T);
    if (bytes > std::numeric_limits<CC_LONG>::max())
        throw std::runtime_error("fixture vector exceeds SHA-256 input bound");
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    if (CC_SHA256(values.data(), static_cast<CC_LONG>(bytes), digest.data()) == nullptr)
        throw std::runtime_error("fixture vector SHA-256 failed");
    if (!first) output << ',';
    first = false;
    constexpr char hex[] = "0123456789abcdef";
    output << '"' << name << "\":\"";
    for (const unsigned char value : digest)
        output << hex[value >> 4u] << hex[value & 15u];
    output << '"';
}

void writeDiagnosticJsonFloat(std::ostream& output, const float value) {
    if (std::isfinite(value))
        output << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    else
        output << "null";
}

void writeDiagnosticStateSummary(std::ostream& output,
                                 const RuntimeStateSnapshot& snapshot) {
    output << "{\"available\":" << (snapshot.available ? "true" : "false")
           << ",\"control_step\":" << snapshot.controlStep
           << ",\"physics_substep\":" << snapshot.physicsSubstep
           << ",\"allocation_generation\":" << snapshot.allocationGeneration
           << ",\"material_state_stride\":" << snapshot.materialStateStride
           << ",\"statuses\":[";
    for (std::size_t index = 0u; index < snapshot.statuses.size(); ++index) {
        if (index != 0u) output << ',';
        const auto& status = snapshot.statuses[index];
        output << "{\"code\":" << status.code
               << ",\"environment\":" << status.environment
               << ",\"object_index\":" << status.objectIndex
               << ",\"failing_index\":" << status.failingIndex
               << ",\"completed_microsteps\":" << status.completedMicrosteps
               << ",\"fgmres_iterations\":" << status.fgmresIterations
               << ",\"contact_count\":" << status.contactCount
               << ",\"event_count\":" << status.eventCount
               << ",\"diagnostics\":[";
        writeDiagnosticJsonFloat(output, status.diagnostics.x);
        output << ',';
        writeDiagnosticJsonFloat(output, status.diagnostics.y);
        output << ',';
        writeDiagnosticJsonFloat(output, status.diagnostics.z);
        output << ',';
        writeDiagnosticJsonFloat(output, status.diagnostics.w);
        output << "]}";
    }
    output << "],\"solver_certificates\":[";
    for (std::size_t index = 0u; index < snapshot.solverCertificates.size(); ++index) {
        if (index != 0u) output << ',';
        const auto& certificate = snapshot.solverCertificates[index];
        output << "{\"nonlinear\":[";
        writeDiagnosticJsonFloat(output, certificate.nonlinear.x);
        output << ',';
        writeDiagnosticJsonFloat(output, certificate.nonlinear.y);
        output << ',';
        writeDiagnosticJsonFloat(output, certificate.nonlinear.z);
        output << ',';
        writeDiagnosticJsonFloat(output, certificate.nonlinear.w);
        output << "],\"contact\":[";
        writeDiagnosticJsonFloat(output, certificate.contact.x);
        output << ',';
        writeDiagnosticJsonFloat(output, certificate.contact.y);
        output << ',';
        writeDiagnosticJsonFloat(output, certificate.contact.z);
        output << ',';
        writeDiagnosticJsonFloat(output, certificate.contact.w);
        output << "],\"transport\":[";
        writeDiagnosticJsonFloat(output, certificate.transport.x);
        output << ',';
        writeDiagnosticJsonFloat(output, certificate.transport.y);
        output << ',';
        writeDiagnosticJsonFloat(output, certificate.transport.z);
        output << ',';
        writeDiagnosticJsonFloat(output, certificate.transport.w);
        output << "],\"validity\":[";
        writeDiagnosticJsonFloat(output, certificate.validity.x);
        output << ',';
        writeDiagnosticJsonFloat(output, certificate.validity.y);
        output << ',';
        writeDiagnosticJsonFloat(output, certificate.validity.z);
        output << ',';
        writeDiagnosticJsonFloat(output, certificate.validity.w);
        output << "]}";
    }
    output << "],\"vector_counts\":{";
    bool first = true;
#define WRITE_DIAGNOSTIC_COUNT(field) \
    writeDiagnosticVectorCount(output, #field, snapshot.field, first)
    WRITE_DIAGNOSTIC_COUNT(sutureProxyEdges);
    WRITE_DIAGNOSTIC_COUNT(particles);
    WRITE_DIAGNOSTIC_COUNT(femNodes);
    WRITE_DIAGNOSTIC_COUNT(femFields);
    WRITE_DIAGNOSTIC_COUNT(vascularState);
    WRITE_DIAGNOSTIC_COUNT(vascularClock);
    WRITE_DIAGNOSTIC_COUNT(femTopologyNodes);
    WRITE_DIAGNOSTIC_COUNT(femTopologyTetrahedra);
    WRITE_DIAGNOSTIC_COUNT(cohesiveFaces);
    WRITE_DIAGNOSTIC_COUNT(punctureChannels);
    WRITE_DIAGNOSTIC_COUNT(topologyStates);
    WRITE_DIAGNOSTIC_COUNT(statuses);
    WRITE_DIAGNOSTIC_COUNT(solverCertificates);
    WRITE_DIAGNOSTIC_COUNT(mpmActiveNodeIndices);
    WRITE_DIAGNOSTIC_COUNT(mpmNodeToActive);
    WRITE_DIAGNOSTIC_COUNT(mpmActiveNodeCounts);
    WRITE_DIAGNOSTIC_COUNT(rigidGeneralizedCandidate);
    WRITE_DIAGNOSTIC_COUNT(learnedWeights);
    WRITE_DIAGNOSTIC_COUNT(adaptive);
    WRITE_DIAGNOSTIC_COUNT(schedulers);
    WRITE_DIAGNOSTIC_COUNT(reactions);
    WRITE_DIAGNOSTIC_COUNT(rigidStates);
    WRITE_DIAGNOSTIC_COUNT(contactSamples);
    WRITE_DIAGNOSTIC_COUNT(deformableContactFailures);
    WRITE_DIAGNOSTIC_COUNT(contactHistories);
    WRITE_DIAGNOSTIC_COUNT(humanSupportHistories);
    WRITE_DIAGNOSTIC_COUNT(humanSupportConsequences);
    WRITE_DIAGNOSTIC_COUNT(deformableContactHistories);
    WRITE_DIAGNOSTIC_COUNT(particleMaterialState);
    WRITE_DIAGNOSTIC_COUNT(femMaterialState);
    WRITE_DIAGNOSTIC_COUNT(identification);
    WRITE_DIAGNOSTIC_COUNT(environmentParameters);
#undef WRITE_DIAGNOSTIC_COUNT
    writeDiagnosticVectorCount(output, "fem_regional_galerkin_linearization_nodes",
        snapshot.femRegionalGalerkinLinearizationNodes, first);
    writeDiagnosticVectorCount(output, "fem_regional_galerkin_basis_p",
        snapshot.femRegionalGalerkinBasisP, first);
    writeDiagnosticVectorCount(output, "fem_regional_galerkin_effective_basis_p",
        snapshot.femRegionalGalerkinEffectiveBasisP, first);
    writeDiagnosticVectorCount(output, "fem_regional_galerkin_operator_ap",
        snapshot.femRegionalGalerkinOperatorAP, first);
    writeDiagnosticVectorCount(output, "fem_regional_galerkin_pivoted_rap",
        snapshot.femRegionalGalerkinPivotedRAP, first);
    writeDiagnosticVectorCount(output, "fem_regional_galerkin_pivots",
        snapshot.femRegionalGalerkinPivots, first);
    writeDiagnosticVectorCount(output, "fem_regional_galerkin_factor_valid",
        snapshot.femRegionalGalerkinFactorValid, first);
    writeDiagnosticVectorCount(output, "fem_regional_galerkin_node_block_inverse",
        snapshot.femRegionalGalerkinNodeBlockInverse, first);
    output << "},\"accepted_state_vector_sha256\":{";
    first = true;
#define WRITE_DIAGNOSTIC_PHYSICAL_HASH(field) \
    writeDiagnosticVectorHash(output, #field, snapshot.field, first)
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(particles);
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(femNodes);
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(femFields);
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(vascularState);
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(vascularClock);
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(femTopologyNodes);
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(femTopologyTetrahedra);
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(cohesiveFaces);
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(punctureChannels);
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(topologyStates);
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(rigidStates);
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(contactHistories);
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(humanSupportHistories);
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(humanSupportConsequences);
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(deformableContactHistories);
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(particleMaterialState);
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(femMaterialState);
    WRITE_DIAGNOSTIC_PHYSICAL_HASH(environmentParameters);
#undef WRITE_DIAGNOSTIC_PHYSICAL_HASH
    output << "},\"fem_regional_galerkin_diagnostics_available\":"
           << (snapshot.femRegionalGalerkinDiagnosticsAvailable ? "true" : "false")
           << ",\"fem_regional_galerkin_mode_count\":"
           << snapshot.femRegionalGalerkinModeCount << '}';
}

bool diagnosticSnapshotAccepted(const RuntimeStateSnapshot& snapshot) {
    return snapshot.available && !snapshot.statuses.empty() &&
        snapshot.statuses.size() == snapshot.solverCertificates.size() &&
        std::all_of(snapshot.statuses.begin(), snapshot.statuses.end(),
            [](const NMMatterStatusGPU& status) {
                return status.code == NM_STATUS_SUCCESS;
            }) &&
        std::all_of(snapshot.solverCertificates.begin(),
            snapshot.solverCertificates.end(),
            [](const NMSolverCertificateGPU& certificate) {
                return certificate.validity.w > 0.5f;
            });
}

class DiagnosticStateOutput {
public:
    explicit DiagnosticStateOutput(const std::string& path) : directory_(path) {
        if (path.empty() || !std::filesystem::create_directory(directory_))
            throw std::runtime_error(
                "diagnostic state output must name a new, unused directory");
    }

    std::uint64_t begin(const std::uint32_t requestedControlStep,
                        const RuntimeStateSnapshot& before) {
        const std::uint64_t index = nextIndex_++;
        const std::string stem = indexedDiagnosticStem(index);
        writeRawDiagnosticSnapshot(directory_ / (stem + "-before.bin"), before);
        beforeControlStep_ = requestedControlStep;
        beforeIndex_ = index;
        beforeSummary_.str(std::string{});
        beforeSummary_.clear();
        writeDiagnosticStateSummary(beforeSummary_, before);
        return index;
    }

    void finish(const std::uint64_t index,
                const RuntimeStateSnapshot* after,
                const char* completionStage) {
        if (index != beforeIndex_)
            throw std::runtime_error("diagnostic transaction index is inconsistent");
        const std::string stem = indexedDiagnosticStem(index);
        if (after != nullptr)
            writeRawDiagnosticSnapshot(directory_ / (stem + "-after.bin"), *after);
        const auto metadataPath = directory_ / (stem + ".json");
        std::ofstream metadata(metadataPath, std::ios::out | std::ios::trunc);
        if (!metadata) throw std::runtime_error("could not create diagnostic metadata");
        metadata << "{\"schema\":\"numi.cardboard.full_contact_state_attempt.v1\""
                 << ",\"attempt_index\":" << index
                 << ",\"requested_control_step\":" << beforeControlStep_
                 << ",\"completion_stage\":\"" << completionStage << '\"'
                 << ",\"before_file\":\"" << stem << "-before.bin\""
                 << ",\"after_present\":" << (after != nullptr ? "true" : "false")
                 << ",\"accepted\":"
                 << (after != nullptr && diagnosticSnapshotAccepted(*after)
                        ? "true" : "false")
                 << ",\"before\":" << beforeSummary_.str();
        if (after != nullptr) {
            std::ostringstream afterSummary;
            writeDiagnosticStateSummary(afterSummary, *after);
            metadata << ",\"after_file\":\"" << stem << "-after.bin\""
                     << ",\"after\":" << afterSummary.str();
        }
        metadata << "}\n";
        metadata.flush();
        if (!metadata) throw std::runtime_error("diagnostic metadata write failed");
    }

private:
    std::filesystem::path directory_;
    std::uint64_t nextIndex_ = 0u;
    std::uint64_t beforeIndex_ = std::numeric_limits<std::uint64_t>::max();
    std::uint32_t beforeControlStep_ = 0u;
    std::ostringstream beforeSummary_;
};

std::unique_ptr<DiagnosticStateOutput> diagnosticStateOutput;

void require(const bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

struct BarrierReference {
    double energy = 0.0;
    double forceMagnitude = 0.0;
    double radialHessian = 0.0;
};

BarrierReference barrierReference(const double separation,
                                  const double thickness,
                                  const double stiffness) {
    if (!(thickness > 0.0) || !(stiffness > 0.0) ||
        separation >= thickness) return {};
    const double floor = 1.0e-4 * thickness;
    const double distance = std::max(separation, floor);
    const double distance2 = distance * distance;
    const double thickness2 = thickness * thickness;
    const double offset = distance2 - thickness2;
    const double logarithm = std::log(distance2 / thickness2);
    const double energyAtDistance = std::max(
        -stiffness * offset * offset * logarithm, 0.0);
    const double firstSquared = -stiffness *
        (2.0 * offset * logarithm + offset * offset / distance2);
    const double secondSquared = -stiffness *
        (2.0 * logarithm + 4.0 * offset / distance2 -
         offset * offset / (distance2 * distance2));
    const double energyDerivative = 2.0 * distance * firstSquared;
    const double radialHessian = 4.0 * distance2 * secondSquared +
        2.0 * firstSquared;
    const double energy = separation >= floor
        ? energyAtDistance
        : energyAtDistance + energyDerivative * (separation - floor);
    return {
        std::max(energy, 0.0),
        std::max(-energyDerivative, 0.0),
        separation < floor ? 0.0 : std::max(radialHessian, 0.0),
    };
}

void verifyBarrierRegularization() {
    constexpr double thickness = 1.0e-5;
    constexpr double stiffness = 1.16491e12;
    constexpr double floor = 1.0e-4 * thickness;
    const double h = 1.0e-7 * thickness;
    for (const double separation : {
             -0.25 * floor, 0.0, 0.5 * floor, floor,
             2.0 * floor, 0.25 * thickness, 0.99 * thickness}) {
        // At the C1 join, use a smaller symmetric increment so the first
        // derivative check is not dominated by curvature on the analytic
        // side. The second derivative is intentionally discontinuous there.
        const double localH = separation == floor ? 1.0e-6 * floor : h;
        const BarrierReference center = barrierReference(
            separation, thickness, stiffness);
        const BarrierReference minus = barrierReference(
            separation - localH, thickness, stiffness);
        const BarrierReference plus = barrierReference(
            separation + localH, thickness, stiffness);
        const double energyDerivative =
            (plus.energy - minus.energy) / (2.0 * localH);
        const double derivativeScale = std::max(
            std::abs(center.forceMagnitude), 1.0e-18);
        require(std::isfinite(center.energy) && center.energy >= 0.0 &&
                std::isfinite(center.forceMagnitude) &&
                std::isfinite(center.radialHessian),
            "regularized IPC barrier returned a nonfinite value");
        require(std::abs(energyDerivative + center.forceMagnitude) <=
                    2.0e-5 * derivativeScale + 1.0e-16,
            "regularized IPC barrier force is not the negative energy derivative");
        if (separation > floor) {
            const double energyHessian =
                (plus.energy - 2.0 * center.energy + minus.energy) /
                (localH * localH);
            const double hessianScale = std::max(
                std::abs(center.radialHessian), 1.0e-10);
            require(std::abs(energyHessian - center.radialHessian) <=
                        2.0e-3 * hessianScale + 1.0e-3,
                "regularized IPC barrier Hessian disagrees with the potential");
        }
        if (separation < floor) {
            require(minus.forceMagnitude == center.forceMagnitude &&
                    plus.forceMagnitude == center.forceMagnitude &&
                    center.radialHessian == 0.0,
                "sub-floor continuation did not retain constant force and zero radial curvature");
        }
    }
    const BarrierReference atFloor = barrierReference(
        floor, thickness, stiffness);
    const BarrierReference belowZero = barrierReference(
        0.0, thickness, stiffness);
    const BarrierReference outside = barrierReference(
        1.01 * thickness, thickness, stiffness);
    require(belowZero.forceMagnitude > 0.0 &&
            belowZero.radialHessian == 0.0 &&
            belowZero.energy > atFloor.energy,
        "sub-floor continuation did not preserve finite overlap repulsion");
    require(outside.energy == 0.0 && outside.forceMagnitude == 0.0 &&
            outside.radialHessian == 0.0,
        "unloaded positive-separation IPC branch is not exactly zero");
    std::cout << "ipc_barrier_regularization=pass"
              << " c1_floor_m=" << floor
              << " below_floor_repulsion=true outside_support_exact_zero=true\n";
}

void verifyArmijoRequiresResolvedDescent() {
    // This case comes from retained full-contact trace 035. At this tiny
    // positive alpha, the sufficient-decrease product rounds back to base in
    // FP32. The old `trial <= bound` predicate therefore admitted no change.
    constexpr float base = 3.43035972e-5f;
    constexpr float alpha = 1.6975e-8f;
    constexpr float c1 = 1.0e-4f;
    const float bound = (1.0f - c1 * alpha) * base;
    require(bound == base,
        "Armijo rounding regression input no longer rounds its bound to base");
    const float unchangedTrial = base;
    const bool oldPredicate = unchangedTrial <= bound;
    const bool resolvedPredicate = alpha > 0.0f && unchangedTrial < base &&
        unchangedTrial <= bound;
    require(oldPredicate && !resolvedPredicate,
        "positive-alpha Armijo must reject a rounded no-descent trial");
    const float resolvedLower = std::nextafter(base, 0.0f);
    require(resolvedLower < base && resolvedLower <= bound,
        "strict descent should still admit a representably lower residual");
    std::cout << "armijo_fp32_rounded_bound=pass base=" << base
              << " alpha=" << alpha << " bound_equals_base=true"
              << " unchanged_rejected=true representable_descent_admitted=true\n";
}

template<class T>
id<MTLBuffer> makeBuffer(id<MTLDevice> device, const std::vector<T>& values) {
    const T zero{};
    const void* data = values.empty() ? static_cast<const void*>(&zero)
                                      : static_cast<const void*>(values.data());
    return [device newBufferWithBytes:data
        length:std::max<std::size_t>(1u, values.size()) * sizeof(T)
        options:MTLResourceStorageModeShared];
}

class StderrCapture {
public:
    StderrCapture() : file_(std::tmpfile()), saved_(::dup(STDERR_FILENO)) {
        require(file_ != nullptr && saved_ >= 0, "stderr capture setup failed");
        require(::dup2(::fileno(file_), STDERR_FILENO) >= 0,
                "stderr capture redirect failed");
    }
    StderrCapture(const StderrCapture&) = delete;
    StderrCapture& operator=(const StderrCapture&) = delete;
    ~StderrCapture() {
        if (saved_ >= 0) {
            std::fflush(stderr);
            (void)::dup2(saved_, STDERR_FILENO);
            (void)::close(saved_);
        }
        if (file_ != nullptr) std::fclose(file_);
    }
    std::string finish() {
        std::fflush(stderr);
        require(::dup2(saved_, STDERR_FILENO) >= 0,
                "stderr capture restore failed");
        (void)::close(saved_);
        saved_ = -1;
        std::rewind(file_);
        std::string text;
        char bytes[4096];
        for (;;) {
            const std::size_t count = std::fread(bytes, 1u, sizeof(bytes), file_);
            text.append(bytes, count);
            if (count != sizeof(bytes)) break;
        }
        return text;
    }
private:
    FILE* file_ = nullptr;
    int saved_ = -1;
};

struct MeshAndWorld {
    WorldSource world;
    std::vector<std::array<double, 3>> nodes;
    CreaseToolingGeometry tooling;
    std::array<std::uint32_t, 9> lowerInterfaceNodes{};
    std::array<std::uint32_t, 9> upperInterfaceNodes{};
    double interfaceZ = 0.0;
    double interfaceGapM = 0.0;
    std::uint32_t deformableCandidateUpperBound = 0u;
    std::uint32_t mortarPrimitiveA = NM_INVALID_INDEX;
    std::uint32_t mortarPrimitiveB = NM_INVALID_INDEX;
};

std::array<std::uint64_t, 4> fixtureSourceIdentity(
    const std::string& canonicalSource) {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    if (canonicalSource.size() > std::numeric_limits<CC_LONG>::max() ||
        CC_SHA256(canonicalSource.data(),
            static_cast<CC_LONG>(canonicalSource.size()), digest.data()) == nullptr)
        throw std::runtime_error("fixture source identity SHA-256 failed");
    std::array<std::uint64_t, 4> result{};
    std::memcpy(result.data(), digest.data(), digest.size());
    return result;
}

std::uint32_t oppositeCornerForFace(
    const std::array<std::uint32_t, 4>& tetrahedron,
    const std::array<std::uint32_t, 3>& face) {
    auto sortedFace = face;
    std::sort(sortedFace.begin(), sortedFace.end());
    for (std::uint32_t opposite = 0u; opposite < 4u; ++opposite) {
        std::array<std::uint32_t, 3> candidate{};
        std::uint32_t cursor = 0u;
        for (std::uint32_t local = 0u; local < 4u; ++local)
            if (local != opposite) candidate[cursor++] = tetrahedron[local];
        std::sort(candidate.begin(), candidate.end());
        if (candidate == sortedFace) return opposite;
    }
    throw std::runtime_error("mortar parent triangle is not a tetrahedron face");
}

std::uint32_t node(const unsigned i, const unsigned j, const unsigned k) {
    return static_cast<std::uint32_t>(i + 3u * j + 9u * k);
}

double signedTetrahedronVolume6(
    const std::array<std::array<double, 3>, 4>& points) {
    const std::array<double, 3> a{
        points[1][0] - points[0][0],
        points[1][1] - points[0][1],
        points[1][2] - points[0][2],
    };
    const std::array<double, 3> b{
        points[2][0] - points[0][0],
        points[2][1] - points[0][1],
        points[2][2] - points[0][2],
    };
    const std::array<double, 3> c{
        points[3][0] - points[0][0],
        points[3][1] - points[0][1],
        points[3][2] - points[0][2],
    };
    return a[0] * (b[1] * c[2] - b[2] * c[1]) -
           a[1] * (b[0] * c[2] - b[2] * c[0]) +
           a[2] * (b[0] * c[1] - b[1] * c[0]);
}

using Point3 = std::array<double, 3>;
using TetrahedronNodes = std::array<std::uint32_t, 4>;

Point3 subtract(const Point3& lhs, const Point3& rhs) {
    return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

Point3 cross(const Point3& lhs, const Point3& rhs) {
    return {
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    };
}

double dot(const Point3& lhs, const Point3& rhs) {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

bool tetrahedraHavePositiveSATOverlap(
    const TetrahedronNodes& lhs,
    const TetrahedronNodes& rhs,
    const std::vector<Point3>& points,
    double& minimumAxisOverlap) {
    constexpr std::array<std::array<unsigned, 3>, 4> faces = {{
        {{1u, 2u, 3u}}, {{0u, 3u, 2u}},
        {{0u, 1u, 3u}}, {{0u, 2u, 1u}},
    }};
    constexpr std::array<std::array<unsigned, 2>, 6> edges = {{
        {{0u, 1u}}, {{0u, 2u}}, {{0u, 3u}},
        {{1u, 2u}}, {{1u, 3u}}, {{2u, 3u}},
    }};
    std::vector<Point3> axes;
    axes.reserve(44u);
    const auto addFaceAxes = [&](const TetrahedronNodes& tetrahedron) {
        for (const auto& face : faces) {
            const Point3 first = subtract(
                points[tetrahedron[face[1]]], points[tetrahedron[face[0]]]);
            const Point3 second = subtract(
                points[tetrahedron[face[2]]], points[tetrahedron[face[0]]]);
            axes.push_back(cross(first, second));
        }
    };
    addFaceAxes(lhs);
    addFaceAxes(rhs);
    for (const auto& lhsEdge : edges) {
        const Point3 lhsDirection = subtract(
            points[lhs[lhsEdge[1]]], points[lhs[lhsEdge[0]]]);
        for (const auto& rhsEdge : edges) {
            const Point3 rhsDirection = subtract(
                points[rhs[rhsEdge[1]]], points[rhs[rhsEdge[0]]]);
            axes.push_back(cross(lhsDirection, rhsDirection));
        }
    }

    minimumAxisOverlap = std::numeric_limits<double>::infinity();
    constexpr double kDegenerateAxisSquared = 1.0e-28;
    constexpr double kStrictOverlapTolerance = 1.0e-12;
    for (Point3 axis : axes) {
        const double normSquared = dot(axis, axis);
        if (!(normSquared > kDegenerateAxisSquared)) continue;
        const double inverseNorm = 1.0 / std::sqrt(normSquared);
        for (double& component : axis) component *= inverseNorm;

        double lhsMinimum = std::numeric_limits<double>::infinity();
        double lhsMaximum = -std::numeric_limits<double>::infinity();
        double rhsMinimum = std::numeric_limits<double>::infinity();
        double rhsMaximum = -std::numeric_limits<double>::infinity();
        for (unsigned corner = 0u; corner < 4u; ++corner) {
            const double lhsProjection = dot(points[lhs[corner]], axis);
            const double rhsProjection = dot(points[rhs[corner]], axis);
            lhsMinimum = std::min(lhsMinimum, lhsProjection);
            lhsMaximum = std::max(lhsMaximum, lhsProjection);
            rhsMinimum = std::min(rhsMinimum, rhsProjection);
            rhsMaximum = std::max(rhsMaximum, rhsProjection);
        }
        const double overlap = std::min(lhsMaximum, rhsMaximum) -
                               std::max(lhsMinimum, rhsMinimum);
        if (overlap <= kStrictOverlapTolerance) return false;
        minimumAxisOverlap = std::min(minimumAxisOverlap, overlap);
    }
    return std::isfinite(minimumAxisOverlap);
}

void verifyLegacyDetachedTetGeometryRejected() {
    std::vector<Point3> points;
    points.reserve(30u);
    for (unsigned k = 0u; k < 3u; ++k)
        for (unsigned j = 0u; j < 3u; ++j)
            for (unsigned i = 0u; i < 3u; ++i)
                points.push_back({
                    -0.5 * kPadLength + kPadLength * double(i) / 2.0,
                    -0.5 * kPadWidth + kPadWidth * double(j) / 2.0,
                    kPadThickness * double(k) / 2.0,
                });

    constexpr std::array<std::array<unsigned, 4>, 6> split = {{
        {{0u, 1u, 2u, 6u}}, {{0u, 2u, 3u, 6u}},
        {{0u, 3u, 7u, 6u}}, {{0u, 7u, 4u, 6u}},
        {{0u, 4u, 5u, 6u}}, {{0u, 5u, 1u, 6u}},
    }};
    std::vector<TetrahedronNodes> tetrahedra;
    tetrahedra.reserve(48u);
    for (unsigned k = 0u; k < 2u; ++k) {
        for (unsigned j = 0u; j < 2u; ++j) {
            for (unsigned i = 0u; i < 2u; ++i) {
                const std::array<std::uint32_t, 8> corners = {
                    node(i, j, k), node(i + 1u, j, k),
                    node(i + 1u, j + 1u, k), node(i, j + 1u, k),
                    node(i, j, k + 1u), node(i + 1u, j, k + 1u),
                    node(i + 1u, j + 1u, k + 1u), node(i, j + 1u, k + 1u),
                };
                for (const auto& local : split)
                    tetrahedra.push_back({
                        corners[local[0]], corners[local[1]],
                        corners[local[2]], corners[local[3]],
                    });
            }
        }
    }
    require(tetrahedra.size() == 48u && tetrahedra[24u] ==
                TetrahedronNodes{9u, 10u, 13u, 22u} &&
            tetrahedra[29u] == TetrahedronNodes{9u, 19u, 10u, 22u},
            "legacy detached-tet negative control topology changed unexpectedly");

    // Reproduce the old one-face edit: only the selected upper tet's copy of
    // [9,10,22] was moved, leaving adjacent tetrahedra on the original plane.
    constexpr double kLegacyGap = 5.0e-6;
    std::array<std::uint32_t, 27> duplicate{};
    duplicate.fill(NM_INVALID_INDEX);
    for (const std::uint32_t original : {9u, 10u, 22u}) {
        duplicate[original] = static_cast<std::uint32_t>(points.size());
        Point3 moved = points.at(original);
        moved[2] += kLegacyGap;
        points.push_back(moved);
    }
    for (std::uint32_t& index : tetrahedra[29u])
        if (duplicate[index] != NM_INVALID_INDEX) index = duplicate[index];

    const std::array<std::array<std::uint32_t, 2>, 1> witnessPairs = {{{28u, 29u}}};
    double minimumOverlap = std::numeric_limits<double>::infinity();
    bool foundPositiveOverlap = false;
    for (const auto& pair : witnessPairs) {
        double pairOverlap = 0.0;
        if (tetrahedraHavePositiveSATOverlap(
                tetrahedra[pair[0]], tetrahedra[pair[1]], points, pairOverlap)) {
            foundPositiveOverlap = true;
            minimumOverlap = std::min(minimumOverlap, pairOverlap);
        }
    }
    require(foundPositiveOverlap && minimumOverlap > 1.0e-8,
            "legacy one-face detached-tet overlap negative control was not detected");
    std::cout << "legacy_one_face_detached_tet_overlap=detected"
              << " witness_tets=28,29 minimum_sat_overlap_m="
              << minimumOverlap << '\n';
}

MeshAndWorld makeWorld(const MaterialProgram& bulk,
                       const bool forceDeformableOverflow = false,
                       const bool partialMortarPatch = false,
                       const double initialMortarGap = kActivationGap,
                       const bool highSparsePairDomain = false) {
    MeshAndWorld result;
    WorldSource& source = result.world;
    source.environmentCount = kEnvironmentCount;
    source.frameTimestep = 1.0e-4;
    source.gravity = {0.0, 0.0, 0.0};
    source.contactSlop = 10.0e-6;
    source.materials = {bulk};
    source.mixedSolver.newtonIterations = authoredNewtonIterations != 0u
        ? authoredNewtonIterations : (extendedBudget ? 32u : 16u);
    source.mixedSolver.fgmresIterations = extendedBudget ? 256u : 64u;
    source.mixedSolver.fgmresRestart = extendedBudget ? 16u :
        source.mixedSolver.fgmresRestart;
    source.mixedSolver.lineSearchSteps = extendedBudget ? 24u :
        source.mixedSolver.lineSearchSteps;
    source.mixedSolver.relativeResidual = 1.0e-7;
    if (regionalContactPreconditioner)
        source.mixedSolver.femPreconditioner =
            FEMPreconditionerMode::regionalGalerkinContact;
    source.cohesiveMaterials = {{
        .name = "synthetic_full_contact_armijo_interface_fixture_only",
        .normalPenaltyPaPerM = 1.0e9,
        .shearPenaltyPaPerM = 8.0e8,
        .compressionPenaltyPaPerM = 2.0e9,
        .normalStrengthPa = 1.0e6,
        .shearStrengthPa = 8.0e5,
        .fractureEnergyModeIJPerM2 = 1000.0,
        .fractureEnergyModeIIJPerM2 = 2000.0,
        .bkExponent = 1.5,
    }};

    ObjectSource pad;
    pad.name = "synthetic_free_fem_full_contact_armijo_fixture";
    pad.representation = Representation::fem;
    pad.materialIndex = 0u;
    pad.mixedFEM = false;
    pad.adaptive = false;
    pad.deformableContact = true;
    pad.deformableSelfContact = true;
    pad.characteristicLength = 1.0 * kMm;
    if (forceDeformableOverflow) pad.femCapacity.deformableContacts = 1u;

    for (unsigned k = 0u; k < 3u; ++k) {
        const double z = kPadThickness * double(k) / 2.0;
        for (unsigned j = 0u; j < 3u; ++j) {
            const double y = -0.5 * kPadWidth + kPadWidth * double(j) / 2.0;
            for (unsigned i = 0u; i < 3u; ++i) {
                const double x = -0.5 * kPadLength + kPadLength * double(i) / 2.0;
                pad.femNodes.push_back({x, y, z});
                if (i == 0u || i == 2u || j == 0u || j == 2u ||
                    k == 0u || k == 2u)
                    pad.femContactNodes.push_back(node(i, j, k));
            }
        }
    }
    for (unsigned k = 0u; k < 2u; ++k) {
        for (unsigned j = 0u; j < 2u; ++j) {
            for (unsigned i = 0u; i < 2u; ++i) {
                const std::array<std::uint32_t, 8> c = {
                    node(i, j, k), node(i + 1u, j, k),
                    node(i + 1u, j + 1u, k), node(i, j + 1u, k),
                    node(i, j, k + 1u), node(i + 1u, j, k + 1u),
                    node(i + 1u, j + 1u, k + 1u), node(i, j + 1u, k + 1u),
                };
                constexpr std::array<std::array<unsigned, 4>, 6> split = {{
                    {{0u, 1u, 2u, 6u}}, {{0u, 2u, 3u, 6u}},
                    {{0u, 3u, 7u, 6u}}, {{0u, 7u, 4u, 6u}},
                    {{0u, 4u, 5u, 6u}}, {{0u, 5u, 1u, 6u}},
                }};
                for (const auto& local : split)
                    pad.tetrahedra.push_back({{
                        c[local[0]], c[local[1]], c[local[2]], c[local[3]]
                    }});
            }
        }
    }
    require(pad.femNodes.size() == 27u && pad.tetrahedra.size() == 48u,
            "structured two-layer fixture source dimensions changed unexpectedly");
    // Candidate generation can visit every unordered pair of authored FEM
    // face slots. Reserve that complete conservative bound instead of the
    // generic 8*tet active-manifold estimate, which is too small for this
    // intentionally self-contacting split-pad source. The explicit overflow
    // fixture retains its one-slot limit as a deliberate negative control.
    const std::uint64_t authoredFaceSlots =
        4u * static_cast<std::uint64_t>(pad.tetrahedra.size());
    const std::uint64_t candidateUpperBound = authoredFaceSlots > 1u
        ? authoredFaceSlots * (authoredFaceSlots - 1u) / 2u : 0u;
    require(candidateUpperBound > 0u &&
            candidateUpperBound <= std::numeric_limits<std::uint32_t>::max(),
            "split-pad FEM face-pair capacity exceeds the native index range");
    result.deformableCandidateUpperBound =
        static_cast<std::uint32_t>(candidateUpperBound);
    if (!forceDeformableOverflow)
        pad.femCapacity.deformableContacts =
            result.deformableCandidateUpperBound;
    require(forceDeformableOverflow
                ? pad.femCapacity.deformableContacts == 1u
                : pad.femCapacity.deformableContacts >=
                    result.deformableCandidateUpperBound,
            "split-pad deformable-contact capacity does not cover every authored face pair");

    // Clamp four lower perimeter nodes while leaving a substantial free FEM
    // interior and all tool-facing surface nodes in the nonlinear solve.
    for (const auto& [i, j] : std::array<std::array<unsigned, 2>, 4>{{
             {{0u, 0u}}, {{2u, 0u}}, {{0u, 2u}}, {{2u, 2u}},
         }})
        pad.femFixedNodes.push_back(node(i, j, 0u));

    // Split the two z-layer pads along their complete shared plane. The upper
    // half receives duplicate interface vertices and a rigid +gap translation
    // of its interface and top vertices, preserving each tetrahedron's shape
    // while preventing detached-tet overlap with the lower half.
    using Face = std::array<std::uint32_t, 3>;
    struct InterfaceFaceUse {
        std::uint32_t tetrahedron = NM_INVALID_INDEX;
        bool upperHalf = false;
    };
    constexpr std::uint32_t kLowerLayerTetCount = 24u;
    constexpr std::uint32_t kUpperLayerTetBegin = kLowerLayerTetCount;
    constexpr double kInterfaceTolerance = 1.0e-12;
    const double interfaceZ = 0.5 * kPadThickness;
    require(std::isfinite(initialMortarGap) && initialMortarGap > 0.0,
            "split-pad fixture requires a finite positive interface gap");
    std::map<Face, std::vector<InterfaceFaceUse>> interfaceFaces;
    for (std::uint32_t tetIndex = 0u;
         tetIndex < pad.tetrahedra.size(); ++tetIndex) {
        const auto& tet = pad.tetrahedra[tetIndex].nodes;
        double minimumZ = std::numeric_limits<double>::infinity();
        double maximumZ = -std::numeric_limits<double>::infinity();
        for (const std::uint32_t index : tet) {
            minimumZ = std::min(minimumZ, pad.femNodes.at(index)[2]);
            maximumZ = std::max(maximumZ, pad.femNodes.at(index)[2]);
        }
        const bool lowerHalf = maximumZ <= interfaceZ + kInterfaceTolerance;
        const bool upperHalf = minimumZ >= interfaceZ - kInterfaceTolerance;
        require(lowerHalf != upperHalf,
                "authored tetrahedron crosses the intended z-layer split");
        for (unsigned opposite = 0u; opposite < 4u; ++opposite) {
            Face face{};
            unsigned cursor = 0u;
            for (unsigned local = 0u; local < 4u; ++local)
                if (local != opposite) face[cursor++] = tet[local];
            const bool onInterface = std::all_of(face.begin(), face.end(),
                [&](const std::uint32_t index) {
                    return std::abs(pad.femNodes.at(index)[2] - interfaceZ) <=
                        kInterfaceTolerance;
                });
            if (!onInterface) continue;
            std::sort(face.begin(), face.end());
            interfaceFaces[face].push_back({tetIndex, upperHalf});
        }
    }
    require(interfaceFaces.size() == 8u,
            "two-half pad must expose exactly eight matching interface triangles");
    for (const auto& entry : interfaceFaces) {
        const auto& uses = entry.second;
        require(uses.size() == 2u && uses[0].upperHalf != uses[1].upperHalf,
                "each z-split triangle must have one lower and one upper parent tetrahedron");
    }

    Face sideA{node(0u, 0u, 1u), node(1u, 0u, 1u), node(1u, 1u, 1u)};
    std::sort(sideA.begin(), sideA.end());
    const auto selectedFace = interfaceFaces.find(sideA);
    require(selectedFace != interfaceFaces.end() &&
            selectedFace->second.size() == 2u,
            "selected cohesive triangle is not shared across both z-layer halves");
    std::uint32_t firstTet = NM_INVALID_INDEX;
    std::uint32_t secondTet = NM_INVALID_INDEX;
    for (const InterfaceFaceUse& use : selectedFace->second) {
        if (use.upperHalf) secondTet = use.tetrahedron;
        else firstTet = use.tetrahedron;
    }
    require(firstTet < kLowerLayerTetCount &&
            secondTet >= kUpperLayerTetBegin &&
            secondTet < pad.tetrahedra.size(),
            "selected cohesive triangle has invalid lower/upper parent identities");

    std::array<std::uint32_t, 27> upperNodeMap{};
    for (std::uint32_t index = 0u; index < upperNodeMap.size(); ++index)
        upperNodeMap[index] = index;
    std::uint32_t interfaceCursor = 0u;
    for (unsigned j = 0u; j < 3u; ++j) {
        for (unsigned i = 0u; i < 3u; ++i) {
            const std::uint32_t original = node(i, j, 1u);
            const std::uint32_t duplicate =
                static_cast<std::uint32_t>(pad.femNodes.size());
            auto point = pad.femNodes.at(original);
            point[2] += initialMortarGap;
            pad.femNodes.push_back(point);
            result.lowerInterfaceNodes[interfaceCursor] = original;
            result.upperInterfaceNodes[interfaceCursor] = duplicate;
            upperNodeMap[original] = duplicate;
            pad.femContactNodes.push_back(original);
            pad.femContactNodes.push_back(duplicate);
            ++interfaceCursor;
        }
    }
    result.interfaceZ = interfaceZ;
    result.interfaceGapM = initialMortarGap;

    for (unsigned j = 0u; j < 3u; ++j)
        for (unsigned i = 0u; i < 3u; ++i)
            pad.femNodes.at(node(i, j, 2u))[2] += initialMortarGap;

    std::uint32_t remappedUpperTetCount = 0u;
    for (std::uint32_t tetIndex = 0u;
         tetIndex < pad.tetrahedra.size(); ++tetIndex) {
        auto& tet = pad.tetrahedra[tetIndex].nodes;
        // The structured source writes all lower-layer cells first and all
        // upper-layer cells second; verify this compact index contract below.
        const bool upperLayer = tetIndex >= kUpperLayerTetBegin;
        if (!upperLayer) continue;
        for (std::uint32_t& index : tet) {
            require(index < upperNodeMap.size(),
                    "upper-layer tetrahedron references a non-authored node");
            index = upperNodeMap[index];
        }
        ++remappedUpperTetCount;
    }
    require(remappedUpperTetCount == 24u,
            "z split did not remap every upper-layer tetrahedron");

    std::sort(pad.femContactNodes.begin(), pad.femContactNodes.end());
    pad.femContactNodes.erase(std::unique(pad.femContactNodes.begin(),
        pad.femContactNodes.end()), pad.femContactNodes.end());
    require(pad.femContactNodes.size() <= pad.femNodes.size() &&
            std::all_of(pad.femContactNodes.begin(), pad.femContactNodes.end(),
                [&](const std::uint32_t index) {
                    return index < pad.femNodes.size();
                }),
            "split-pad contact surface contains an invalid node index");

    // Check positive element orientation and the complete physical gap. A
    // strict z separation between all lower/upper tetrahedra rules out any
    // volume overlap or coincident unpaired boundary across the split.
    const double upperInterfaceZ = interfaceZ + initialMortarGap;
    for (std::uint32_t tetIndex = 0u;
         tetIndex < pad.tetrahedra.size(); ++tetIndex) {
        const auto& tet = pad.tetrahedra[tetIndex].nodes;
        std::array<std::array<double, 3>, 4> points{};
        double minimumZ = std::numeric_limits<double>::infinity();
        double maximumZ = -std::numeric_limits<double>::infinity();
        for (unsigned local = 0u; local < 4u; ++local) {
            points[local] = pad.femNodes.at(tet[local]);
            minimumZ = std::min(minimumZ, points[local][2]);
            maximumZ = std::max(maximumZ, points[local][2]);
        }
        const double volume6 = signedTetrahedronVolume6(points);
        require(std::isfinite(volume6) && volume6 > 0.0,
                "split-pad tetrahedron is inverted or degenerate");
        if (tetIndex < kLowerLayerTetCount)
            require(maximumZ <= interfaceZ + kInterfaceTolerance,
                    "lower-pad tetrahedron intrudes through the interface plane");
        else
            require(minimumZ >= upperInterfaceZ - kInterfaceTolerance,
                    "upper-pad tetrahedron intrudes into the interface gap");
    }
    for (std::size_t index = 0u;
         index < result.lowerInterfaceNodes.size(); ++index) {
        const auto& lower = pad.femNodes.at(result.lowerInterfaceNodes[index]);
        const auto& upper = pad.femNodes.at(result.upperInterfaceNodes[index]);
        require(std::abs((upper[2] - lower[2]) - initialMortarGap) <=
                    kInterfaceTolerance &&
                lower[0] == upper[0] && lower[1] == upper[1],
                "duplicated interface vertices do not preserve the authored uniform gap");
    }

    std::array<std::uint32_t, 3> sideB{};
    for (unsigned local = 0u; local < 3u; ++local)
        sideB[local] = upperNodeMap[sideA[local]];
    CohesiveMortarPatchSource interface;
    interface.sideANodes = sideA;
    interface.sideBNodes = sideB;
    interface.sideAVertexBarycentric = partialMortarPatch
        ? std::array<std::array<double, 3>, 3>{{
            {{0.7, 0.2, 0.1}}, {{0.2, 0.7, 0.1}}, {{0.2, 0.1, 0.7}}
          }}
        : std::array<std::array<double, 3>, 3>{{
            {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}
          }};
    interface.sideBVertexBarycentric = interface.sideAVertexBarycentric;
    interface.firstTetrahedron = firstTet;
    interface.secondTetrahedron = secondTet;
    interface.materialIndex = 0u;
    interface.stableIdentifier = kFaceId;
    interface.cohesiveMaterialIndex = 0u;
    interface.activationGapToleranceM = kActivationTolerance;
    pad.femCohesiveMortarPatches.push_back(interface);

    if (highSparsePairDomain) {
        // Keep this construction just above the legacy dense pair limit while
        // remaining inside the full-contact fixture's node/tet/proxy bounds.
        // The extra tetrahedra are isolated, fixed, and far from every tool;
        // they enlarge the canonical FEM/tool pair domain without adding
        // active mechanics to the bounded sparse-index admission test.
        constexpr std::uint32_t kHighPairProxyCount = 9u;
        constexpr std::uint32_t kHighPairTarget =
            detail::kDensePrivateContactPairLimit + 1u;
        const std::size_t targetContactNodes =
            (static_cast<std::size_t>(kHighPairTarget) +
             kHighPairProxyCount - 1u) / kHighPairProxyCount;
        require(source.rigidProxies.size() == 0u,
                "high-pair proxy sources must be appended before tooling creation");
        require(pad.femContactNodes.size() < targetContactNodes,
                "high-pair baseline already exceeds its authored target");
        const std::size_t additionalContactNodes =
            targetContactNodes - pad.femContactNodes.size();
        const std::size_t additionalTetrahedra =
            (additionalContactNodes + 3u) / 4u;
        require(pad.femNodes.size() + additionalTetrahedra * 4u <= 8192u &&
                pad.tetrahedra.size() + additionalTetrahedra <= 32768u,
                "high-pair sparse fixture exceeds the bounded FEM dimensions");
        constexpr double kRemoteNodeBaseX = 0.25;
        constexpr double kRemoteNodeY = 0.3;
        constexpr double kRemoteNodeZ = 0.4;
        constexpr double kRemoteTetSize = 0.0005;
        constexpr double kRemoteTetSpacing = 0.001;
        for (std::size_t tetrahedron = 0u;
             tetrahedron < additionalTetrahedra; ++tetrahedron) {
            const double x = kRemoteNodeBaseX +
                static_cast<double>(tetrahedron) * kRemoteTetSpacing;
            const std::uint32_t firstNode =
                static_cast<std::uint32_t>(pad.femNodes.size());
            pad.femNodes.push_back({x, kRemoteNodeY, kRemoteNodeZ});
            pad.femNodes.push_back({x + kRemoteTetSize, kRemoteNodeY,
                                    kRemoteNodeZ});
            pad.femNodes.push_back({x, kRemoteNodeY + kRemoteTetSize,
                                    kRemoteNodeZ});
            pad.femNodes.push_back({x, kRemoteNodeY,
                                    kRemoteNodeZ + kRemoteTetSize});
            pad.tetrahedra.push_back({{
                firstNode, firstNode + 1u, firstNode + 2u, firstNode + 3u
            }});
            for (std::uint32_t local = 0u; local < 4u; ++local) {
                pad.femContactNodes.push_back(firstNode + local);
                pad.femFixedNodes.push_back(firstNode + local);
            }
        }
        std::sort(pad.femContactNodes.begin(), pad.femContactNodes.end());
        pad.femContactNodes.erase(std::unique(pad.femContactNodes.begin(),
            pad.femContactNodes.end()), pad.femContactNodes.end());
        require(pad.femContactNodes.size() >= targetContactNodes,
                "high-pair source did not reach the intended contact-node count");
        require(pad.femContactNodes.size() * kHighPairProxyCount >
                    detail::kDensePrivateContactPairLimit,
                "high-pair source did not exceed the legacy dense pair domain");
        pad.femCapacity.deformableContacts = 131072u;
    }

    // The two split material subdomains begin stress-free in the authored
    // reference configuration. The initial physical gap is part of those
    // two source meshes; activation later latches only the local interface
    // reference from an accepted endpoint and never rewrites bulk rest data.
    pad.femReferenceDisplacementGradient = true;
    pad.femReferenceNodes = pad.femNodes;
    pad.femMaterialIndices.assign(pad.tetrahedra.size(), 0u);
    if (regionalContactPreconditioner)
        // The source has one authored bulk material region. Reuse its compact
        // all-zero tetrahedron map as the preconditioner region labeling.
        pad.femPreconditionerRegionIds = pad.femMaterialIndices;
    std::ostringstream referenceSource;
    referenceSource << "synthetic.full-contact.reference.v1\n" << std::hexfloat;
    for (const auto& point : pad.femReferenceNodes)
        referenceSource << point[0] << ',' << point[1] << ',' << point[2] << '\n';
    pad.femReferenceSourceIdentity =
        fixtureSourceIdentity(referenceSource.str());
    std::ostringstream materialSource;
    materialSource << "synthetic.full-contact.regional-map.v1\n";
    for (const std::uint32_t materialIndex : pad.femMaterialIndices)
        materialSource << materialIndex << '\n';
    pad.femMaterialSourceIdentity =
        fixtureSourceIdentity(materialSource.str());
    result.mortarPrimitiveA = 4u * firstTet + oppositeCornerForFace(
        pad.tetrahedra[firstTet].nodes, sideA);
    result.mortarPrimitiveB = 4u * secondTet + oppositeCornerForFace(
        pad.tetrahedra[secondTet].nodes, sideB);
    result.nodes = pad.femNodes;

    CreaseToolingSpec spec;
    spec.boardLength = kPadLength;
    spec.boardWidth = kPadWidth;
    spec.boardCaliper = kPadThickness;
    spec.creaseX = 0.0;
    spec.supportSurfaceZ = -kSupportClearance;
    spec.punchRadius = kPunchRadius;
    spec.punchEndOverhang = 0.1 * kMm;
    spec.supportEndOverhang = 0.25 * kMm;
    spec.supportThickness = 1.0 * kMm;
    spec.initialClearance = kInitialClearance;
    spec.punchBodyIndex = kPunchBody;
    spec.contactMaterialIndex = 0u;
    double highestFEMNodeZ = -std::numeric_limits<double>::infinity();
    for (const auto& point : pad.femNodes)
        highestFEMNodeZ = std::max(highestFEMNodeZ, point[2]);
    require(std::isfinite(highestFEMNodeZ),
            "synthetic full-contact fixture has no finite FEM surface height");
    // Keep the unrelated crease punch above every FEM node, including the
    // lifted upper half, by the declared initial tool clearance. This leaves
    // all authored contact floors and mortar activation tolerances unchanged.
    spec.initialClearance = std::max(spec.initialClearance,
        highestFEMNodeZ + kInitialClearance -
            spec.supportSurfaceZ - spec.boardCaliper);
    require(spec.supportSurfaceZ + spec.boardCaliper +
                spec.initialClearance - highestFEMNodeZ >=
                    kInitialClearance - 1.0e-12,
            "synthetic full-contact punch does not preserve its declared clearance above all FEM nodes");
    result.tooling = makeCreaseTooling(spec);
    result.tooling.punch.prescribedEndPoseTranslation = true;
    result.tooling.support.bodyIndex = kAnvilBody;
    source.rigidProxies = {result.tooling.punch, result.tooling.support};
    if (highSparsePairDomain) {
        for (std::uint32_t proxyIndex = 2u; proxyIndex < 9u; ++proxyIndex) {
            RigidProxySource remote;
            remote.shape = NM_RIGID_BOX;
            remote.localCenter = {
                -1.0 + 0.05 * static_cast<double>(proxyIndex - 2u),
                1.0,
                1.0,
            };
            remote.localExtent = {0.001, 0.001, 0.001};
            remote.materialIndex = 0u;
            source.rigidProxies.push_back(remote);
        }
    }
    source.objects.push_back(std::move(pad));
    return result;
}

MaterialProgram readMaterial(const std::string& path) {
    const ParseResult parsed = parseMatterFile(path);
    require(parsed.succeeded(), "bulk material parse failed: " + path);
    return parsed.material;
}

CompiledWorld compile(const WorldSource& source) {
    CompileOptions options;
    options.maximumRateExponent = 0u;
    options.emitSpecializedMetal = false;
    options.localMaterialNewtonIterations = 16u;
    const CompileResult result = compileWorld(source, options);
    if (!result.succeeded()) {
        std::string errors;
        for (const Diagnostic& diagnostic : result.diagnostics)
            if (diagnostic.severity == Diagnostic::Severity::error)
                errors += diagnostic.message + "; ";
        throw std::runtime_error("full-contact fixture compile failed: " + errors);
    }
    return result.world;
}

void verifyAuthoredLineSearchBudget(const MaterialProgram& bulk) {
    std::array<std::uint64_t, 3> fingerprints{};
    std::size_t fingerprintIndex = 0u;
    for (const std::uint32_t steps : {1u, 8u, 32u}) {
        MeshAndWorld source = makeWorld(bulk);
        source.world.mixedSolver.lineSearchSteps = steps;
        const CompiledWorld compiled = compile(source.world);
        require(compiled.mixedSolver.nonlinearIterations.w == steps,
                "compiled solver did not retain the authored line-search budget");
        fingerprints[fingerprintIndex++] = compiled.fingerprint;
    }
    require(fingerprints[0] != fingerprints[1] &&
                fingerprints[1] != fingerprints[2] &&
                fingerprints[0] != fingerprints[2],
            "compiled identity did not bind the authored line-search budget");
    for (const std::uint32_t steps : {0u, 33u}) {
        MeshAndWorld source = makeWorld(bulk);
        source.world.mixedSolver.lineSearchSteps = steps;
        const CompileResult compiled = compileWorld(source.world);
        require(!compiled.succeeded(),
                "compiler accepted a zero or over-cap line-search budget");
    }
    std::cout << "authored_line_search_budget=pass accepted=1,8,32 rejected=0,33\n";
}

void verifyPrescribedBoxCompilerAdmission(const MaterialProgram& bulk) {
    MeshAndWorld moving = makeWorld(bulk);
    moving.world.rigidProxies.at(1u).prescribedEndPoseTranslation = true;
    const CompiledWorld compiled = compile(moving.world);
    require(compiled.contact.rigidProxies.size() == 2u &&
            compiled.contact.rigidProxies[1].shapeKind == NM_RIGID_BOX &&
            compiled.contact.rigidProxies[1].flags ==
                NM_RIGID_PRESCRIBED_TRANSLATION &&
            compiled.contact.rigidProxies[1].reserved2 == 0u,
            "compiler did not preserve fixed-orientation prescribed BOX translation");

    moving.world.rigidProxies[1].prescribedBarrierStiffnessScale = 2.0;
    CompileOptions options;
    options.maximumRateExponent = 0u;
    options.emitSpecializedMetal = false;
    options.localMaterialNewtonIterations = 16u;
    const CompileResult scaledBox = compileWorld(moving.world, options);
    require(!scaledBox.succeeded(),
            "compiler admitted unsupported non-unit BOX barrier scaling");

    moving = makeWorld(bulk);
    moving.world.rigidProxies[1].prescribedEndPoseTranslation = true;
    moving.world.rigidProxies[1].dynamic = true;
    moving.world.rigidProxies[1].sceneBodyIndex = 1u;
    const CompileResult dynamicBox = compileWorld(moving.world, options);
    require(!dynamicBox.succeeded(),
            "compiler admitted a free dynamic BOX as endpoint-prescribed tooling");
    std::cout << "prescribed_box_compiler_admission=pass"
              << " nonunit_barrier_and_dynamic_box_rejected=true\n";
}

MRBodyStateGPU bodyAt(const ToolPose pose, const float velocityZ) {
    MRBodyStateGPU result{};
    result.position = {static_cast<float>(pose.translation.x),
        static_cast<float>(pose.translation.y), static_cast<float>(pose.translation.z), 0.0f};
    result.orientation = {static_cast<float>(pose.orientation.x),
        static_cast<float>(pose.orientation.y), static_cast<float>(pose.orientation.z),
        static_cast<float>(pose.orientation.w)};
    result.linearVelocityAndInverseMass = {0.0f, 0.0f, velocityZ, 0.0f};
    result.angularVelocity = {0.0f, 0.0f, 0.0f, 0.0f};
    result.inverseInertiaWorldRow0 = {0.0f, 0.0f, 0.0f, 0.0f};
    result.inverseInertiaWorldRow1 = {0.0f, 0.0f, 0.0f, 0.0f};
    result.inverseInertiaWorldRow2 = {0.0f, 0.0f, 0.0f, 0.0f};
    result.flagsAndIndices[0] = MR_MOTION_KINEMATIC;
    return result;
}

struct TrialRecord {
    std::uint32_t iteration = 0u;
    double base = NAN;
    double trial = NAN;
    double alpha = NAN;
    double selected = NAN;
    bool accepted = false;
};

std::vector<TrialRecord> traceRecords(const std::string& text) {
    const std::regex expression(
        R"("newton_iteration":([0-9]+).*"base_l2":([^,]+),"trial_l2":([^,]+),"trial_alpha":([^,]+),"accepted":(true|false).*"selected_alpha":([^,]+))");
    std::vector<TrialRecord> records;
    std::smatch match;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.find("fem_residual_armijo_iteration=") == std::string::npos)
            continue;
        if (!std::regex_search(line, match, expression)) continue;
        try {
            records.push_back({
                static_cast<std::uint32_t>(std::stoul(match[1].str())),
                std::stod(match[2].str()), std::stod(match[3].str()),
                std::stod(match[4].str()), std::stod(match[6].str()),
                match[5].str() == "true",
            });
        } catch (...) {
            // Null or malformed telemetry is not admitted as a trial record.
        }
    }
    return records;
}

void emitSelectedDeformableContactCapture(const std::string& trace) {
    constexpr const char* capturePrefix = "fem_deformable_contact_capture=";
    constexpr const char* skippedPrefix =
        "fem_deformable_contact_capture_skipped=";
    std::istringstream lines(trace);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.rfind(capturePrefix, 0u) == 0u ||
            line.rfind(skippedPrefix, 0u) == 0u) {
            std::fwrite(line.data(), 1u, line.size(), stderr);
            std::fputc('\n', stderr);
        }
    }
    std::fflush(stderr);
}

void retainFixtureTrace(const char* stage, const std::string& trace) {
    if (diagnosticStateOutputPath.empty()) return;
    const auto path = std::filesystem::path(diagnosticStateOutputPath) /
        (std::string("fixture-") + stage + ".log");
    if (std::filesystem::exists(path))
        throw std::runtime_error("refusing to replace a retained fixture trace");
    std::ofstream output(path, std::ios::binary | std::ios::out);
    if (!output) throw std::runtime_error("could not retain fixture trace");
    output.write(trace.data(), static_cast<std::streamsize>(trace.size()));
    output.flush();
    if (!output.good()) throw std::runtime_error("incomplete retained fixture trace");
}

struct StepResult {
    RuntimeStateSnapshot snapshot;
    std::string trace;
};

struct AcceptedRollbackSequence {
    RuntimeStateSnapshot initial;
    StepResult activation;
    StepResult loaded;
    StepResult rejected;
};

class ScopedEnvironmentValue {
public:
    ScopedEnvironmentValue(const char* name, const char* value)
        : name_(name) {
        if (const char* previous = std::getenv(name_.c_str())) {
            hadPrevious_ = true;
            previous_ = previous;
        }
        if (value != nullptr) setenv(name_.c_str(), value, 1);
        else unsetenv(name_.c_str());
    }

    ~ScopedEnvironmentValue() {
        if (hadPrevious_) setenv(name_.c_str(), previous_.c_str(), 1);
        else unsetenv(name_.c_str());
    }

    ScopedEnvironmentValue(const ScopedEnvironmentValue&) = delete;
    ScopedEnvironmentValue& operator=(const ScopedEnvironmentValue&) = delete;

private:
    std::string name_;
    std::string previous_;
    bool hadPrevious_ = false;
};

struct Fixture {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLBuffer> statuses = nil;
    id<MTLBuffer> bodyBuffer = nil;
    Runtime runtime;
    CompiledWorld world;
    std::vector<MRBodyStateGPU> bodies;
    std::uint32_t nodeCount = 0u;
    bool translateSupportBox = false;
    bool sweepSupportBox = false;
    bool stationaryPunch = false;

    Fixture(const MeshAndWorld& source, const bool overflow = false,
            const bool prescribeSupportBoxTranslation = false,
            const bool sweepPrescribedSupportBox = false,
            const bool parkPunch = false,
            const bool aliasProductionSurfaceScratch = false,
            const bool aliasCohesiveCandidateScratch = false)
        : world(compile(source.world)) {
        translateSupportBox = prescribeSupportBoxTranslation;
        sweepSupportBox = sweepPrescribedSupportBox;
        stationaryPunch = parkPunch;
        setenv("NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT", "all", 1);
        [[maybe_unused]] const ScopedEnvironmentValue surfaceScratchAlias(
            "NM_FEM_FULL_CONTACT_REUSE_SELF_SURFACE_SCRATCH",
            aliasProductionSurfaceScratch ? "1" : nullptr);
        [[maybe_unused]] const ScopedEnvironmentValue cohesiveScratchAlias(
            "NM_FEM_FULL_CONTACT_REUSE_COHESIVE_SCRATCH",
            aliasCohesiveCandidateScratch ? "1" : nullptr);
        unsetenv("NM_FEM_RESIDUAL_ARMIJO_ROOT");
        unsetenv("NM_FEM_CAPSULE_RESIDUAL_ARMIJO_ROOT");
        unsetenv("NM_FEM_CAPSULE_BOX_RESIDUAL_ARMIJO_ROOT");
        unsetenv("NM_FEM_COHESIVE_RESIDUAL_ARMIJO_ROOT");
        setenv("NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT", "0", 1);
        device = MTLCreateSystemDefaultDevice();
        require(device != nil, "native Metal device unavailable");
        require([[device name] rangeOfString:@"Apple"].location != NSNotFound &&
                [[device name] rangeOfString:@"Paravirtual"].location == NSNotFound,
            "a physical Apple Metal device is required");
        queue = [device newCommandQueue];
        require(queue != nil, "Metal command queue unavailable");
        RuntimeConfiguration config;
        config.metallib = NUMI_MATTER_METALLIB;
        config.environmentCount = world.dispatch.environmentCount;
        config.captureEvents = false;
        config.captureDiagnostics = true;
        config.adaptiveTransfer = false;
        const RuntimeDiagnostics initialized = runtime.initialize(world, config);
        require(initialized.encoded, "runtime initialize failed: " + initialized.message);
        nodeCount = world.dispatch.femNodeCount;
        const CreaseToolingGeometry& tooling = source.tooling;
        MRBodyStateGPU supportBody =
            bodyAt({{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 1.0}}, 0.0f);
        if (world.dispatch.rigidGeneralizedCapacity == 6u) {
            supportBody.linearVelocityAndInverseMass.w = 1.0f;
            supportBody.inverseInertiaWorldRow0.x = 1.0f;
            supportBody.inverseInertiaWorldRow1.y = 1.0f;
            supportBody.inverseInertiaWorldRow2.z = 1.0f;
            supportBody.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
        }
        if (translateSupportBox)
            supportBody.linearVelocityAndInverseMass.x = 0.1f;
        if (sweepSupportBox)
            supportBody.linearVelocityAndInverseMass.z = 20.0f;
        bodies = {
            bodyAt(tooling.punchInitialPose,
                (sweepSupportBox || stationaryPunch)
                    ? 0.0f : static_cast<float>(-kToolSpeed)),
            supportBody,
        };
        bodyBuffer = makeBuffer(device, bodies);
        statuses = [device newBufferWithLength:
            world.dispatch.environmentCount * sizeof(MRMetalWorldStatusGPU)
            options:MTLResourceStorageModeShared];
        require(bodyBuffer != nil && statuses != nil,
                "fixture body or status buffer unavailable");
        require(world.dispatch.contactPairCount > 0u &&
                world.dispatch.deformableContactCapacity > 0u &&
                world.dispatch.cohesiveFaceCount == 1u &&
                world.dispatch.rigidProxyCount == 2u,
                "compiled fixture lacks contact, mortar, or rigid source rows");
        require(world.contact.rigidProxies.size() == 2u &&
                world.contact.rigidProxies[0].shapeKind == NM_RIGID_CAPSULE &&
                world.contact.rigidProxies[1].shapeKind == NM_RIGID_BOX,
                "fixture must compile one prescribed capsule and one static BOX");
        bool hasCapsulePair = false;
        bool hasBoxPair = false;
        for (const NMContactPairGPU& pair : world.contact.pairs) {
            require(pair.objectIndex == 0u && pair.rigidProxy < 2u,
                    "rigid pair escaped the single FEM / two tool owners");
            hasCapsulePair = hasCapsulePair ||
                world.contact.rigidProxies[pair.rigidProxy].shapeKind == NM_RIGID_CAPSULE;
            hasBoxPair = hasBoxPair ||
                world.contact.rigidProxies[pair.rigidProxy].shapeKind == NM_RIGID_BOX;
        }
        require(hasCapsulePair && hasBoxPair,
                "fixture must compile authored FEM pairs for both tool shapes");
        require((world.objects[0].flags & NM_OBJECT_DISABLE_SELF_CONTACT) == 0u &&
                (world.objects[0].flags & NM_OBJECT_DISABLE_DEFORMABLE_CONTACT) == 0u,
                "compiled fixture did not enable deformable self-contact");
        if (overflow)
            require(world.dispatch.deformableContactCapacity == 1u,
                "overflow case did not bind its deliberately bounded contact capacity");
    }

    RuntimeStateSnapshot snapshot() {
        const RuntimeStateSnapshot value = runtime.snapshot();
        require(value.available, "runtime snapshot failed: " + value.message);
        return value;
    }

    StepResult step(const std::uint32_t controlStep,
                    const bool rejectPostCommit = false,
                    const bool commandActivation = false,
                    const bool mismatchSupportVelocity = false,
                    const bool mismatchSupportOrientation = false) {
        RuntimeStateSnapshot before;
        if (diagnosticStateOutput) {
            before = runtime.snapshot();
            if (!before.available)
                throw std::runtime_error(
                    "diagnostic before-state snapshot failed: " + before.message);
        }
        const std::uint64_t diagnosticIndex = diagnosticStateOutput
            ? diagnosticStateOutput->begin(controlStep, before) : 0u;
        auto endpoint = bodies;
        RuntimeStateSnapshot result;
        const char* stage = "prepare";
        try {
        endpoint[kPunchBody].position.z +=
            endpoint[kPunchBody].linearVelocityAndInverseMass.z *
            static_cast<float>(runtime.timestepSeconds());
        if (translateSupportBox)
            endpoint[kAnvilBody].position.x +=
                endpoint[kAnvilBody].linearVelocityAndInverseMass.x *
                static_cast<float>(runtime.timestepSeconds());
        if (sweepSupportBox)
            endpoint[kAnvilBody].position.z +=
                endpoint[kAnvilBody].linearVelocityAndInverseMass.z *
                static_cast<float>(runtime.timestepSeconds());
        if (mismatchSupportVelocity)
            endpoint[kAnvilBody].linearVelocityAndInverseMass.x += 0.01f;
        if (mismatchSupportOrientation) {
            endpoint[kAnvilBody].orientation.y = 1.0e-3f;
            endpoint[kAnvilBody].orientation.w =
                std::sqrt(1.0f - endpoint[kAnvilBody].orientation.y *
                    endpoint[kAnvilBody].orientation.y);
        }
        auto postEndpoint = endpoint;
        if (rejectPostCommit)
            postEndpoint[kPunchBody].flagsAndIndices[0] = MR_MOTION_DYNAMIC;
        id<MTLBuffer> endpointBuffer = makeBuffer(device, endpoint);
        id<MTLBuffer> postBuffer = rejectPostCommit
            ? makeBuffer(device, postEndpoint) : endpointBuffer;
        std::vector<MRMetalWorldStatusGPU> statusValues(world.dispatch.environmentCount);
        for (std::uint32_t env = 0u; env < world.dispatch.environmentCount; ++env)
            statusValues[env].environment = env;
        std::memcpy(statuses.contents, statusValues.data(),
                    statusValues.size() * sizeof(MRMetalWorldStatusGPU));

        NMCohesiveActivationCommandGPU activation{};
        activation.identity = {0u, 0u, kFaceId, 0u};
        id<MTLBuffer> activationBuffer = commandActivation
            ? makeBuffer(device, std::vector<NMCohesiveActivationCommandGPU>{activation})
            : nil;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        require(command != nil, "Metal command buffer unavailable");
        EncodeRequest request;
        request.commandBuffer = (__bridge void*)command;
        request.environmentStatuses = (__bridge void*)statuses;
        request.rigid.currentBodies = (__bridge void*)endpointBuffer;
        request.rigid.currentBodyCount = 2u;
        request.rigid.currentBodyStride = 2u;
        if (world.dispatch.rigidGeneralizedCapacity == 6u) {
            request.rigid.sceneBodies = (__bridge void*)endpointBuffer;
            request.rigid.sceneBodyCount = 2u;
            request.rigid.sceneStride = 2u;
        }
        if (commandActivation) {
            request.cohesiveActivationCommands = (__bridge void*)activationBuffer;
            request.cohesiveActivationCommandCount = 1u;
            request.cohesiveActivationCommandStride =
                sizeof(NMCohesiveActivationCommandGPU);
        }
        request.controlStep = controlStep;
        request.physicsSubsteps = 1u;
        request.timestepSeconds = runtime.timestepSeconds();
        request.runAdaptiveTransfer = false;
        request.phase = EncodePhase::preDynamics;
        stage = "encode_pre_dynamics";
        auto encoded = runtime.encode(request);
        require(encoded.encoded, "preDynamics encode failed: " + encoded.message);
        request.phase = EncodePhase::postCommit;
        request.cohesiveActivationCommands = nullptr;
        request.cohesiveActivationCommandCount = 0u;
        request.cohesiveActivationCommandStride = 0u;
        request.rigid.currentBodies = (__bridge void*)postBuffer;
        stage = "encode_post_commit";
        encoded = runtime.encode(request);
        require(encoded.encoded, "postCommit encode failed: " + encoded.message);
        stage = "command_completion";
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
                "native command buffer did not complete");
        ++nativeFEMTransactionsExecuted;
        stage = "snapshot_after_completion";
        result = snapshot();
        } catch (...) {
            if (diagnosticStateOutput)
                diagnosticStateOutput->finish(diagnosticIndex, nullptr, stage);
            throw;
        }
        if (diagnosticStateOutput)
            diagnosticStateOutput->finish(diagnosticIndex, &result, stage);
        if (result.statuses.size() == 1u &&
            result.statuses[0].code == NM_STATUS_SUCCESS)
            bodies = std::move(endpoint);
        return {std::move(result), {}};
    }
};

enum class PrivateSelfSweepCase {
    unbondedCrossing,
    separatedMiss,
    fullyCoveredCrossing,
    partialUncoveredCrossing,
    partialCoveredCrossing,
    initiallyTouchingSeparating,
};

NMMatterStatusGPU runPrivateSelfSweepKernelCase(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLComputePipelineState> pipeline,
        const PrivateSelfSweepCase fixtureCase) {
    const bool hasMortar =
        fixtureCase == PrivateSelfSweepCase::fullyCoveredCrossing ||
        fixtureCase == PrivateSelfSweepCase::partialUncoveredCrossing ||
        fixtureCase == PrivateSelfSweepCase::partialCoveredCrossing;
    const bool partialMortar =
        fixtureCase == PrivateSelfSweepCase::partialUncoveredCrossing ||
        fixtureCase == PrivateSelfSweepCase::partialCoveredCrossing;
    const bool separatedMiss =
        fixtureCase == PrivateSelfSweepCase::separatedMiss;
    const bool initiallyTouching =
        fixtureCase == PrivateSelfSweepCase::initiallyTouchingSeparating;
    NMMatterDispatchGPU dispatch{};
    dispatch.environmentCount = 1u;
    dispatch.objectCount = 1u;
    dispatch.femNodeCount = 8u;
    dispatch.tetrahedronCount = 2u;
    dispatch.cohesiveFaceCount = hasMortar ? 1u : 0u;
    dispatch.deformableContactCapacity = 1u;
    dispatch.numericalLimits = {1.0e-3f, 0.0f, 1.0e-8f, 1.0e20f};

    NMMixedSolverGPU solver{};
    solver.contactAcceptance.x = 0.01f;

    NMContinuumObjectGPU object{};
    object.representation = NM_REPRESENTATION_FEM;

    std::vector<NMFEMNodeStateGPU> base(8u), trial(8u);
    const std::array<std::array<float, 3>, 3> triangle = {{
        {{0.0f, 0.0f, 0.0f}},
        {{4.0e-3f, 0.0f, 0.0f}},
        {{0.0f, 4.0e-3f, 0.0f}},
    }};
    for (std::uint32_t featureIndex = 0u; featureIndex < 3u;
         ++featureIndex) {
        const auto& point = triangle[featureIndex];
        base[featureIndex + 1u].positionAndMass =
            {point[0], point[1], point[2], 1.0f};
        trial[featureIndex + 1u] = base[featureIndex + 1u];
        const float startZ = initiallyTouching ? 0.0f : 2.0e-4f;
        float endZ = 2.0e-4f;
        if (fixtureCase == PrivateSelfSweepCase::unbondedCrossing ||
            fixtureCase == PrivateSelfSweepCase::fullyCoveredCrossing ||
            fixtureCase == PrivateSelfSweepCase::partialUncoveredCrossing)
            endZ = -2.0e-4f;
        else if (fixtureCase == PrivateSelfSweepCase::partialCoveredCrossing &&
                 featureIndex == 0u)
            endZ = -2.0e-4f;
        const float lateral = separatedMiss ? 2.0e-2f : 0.0f;
        base[featureIndex + 5u].positionAndMass =
            {point[0] + lateral, point[1], startZ, 1.0f};
        trial[featureIndex + 5u].positionAndMass =
            {point[0] + lateral, point[1], endZ, 1.0f};
    }

    std::vector<NMContinuumSurfacePrimitiveGPU> primitives(8u);
    const auto setPrimitive = [&] (
            const std::uint32_t primitiveIndex,
            const std::uint32_t parentIdentity,
            const std::array<std::uint32_t, 3>& nodes) {
        NMContinuumSurfacePrimitiveGPU primitive{};
        primitive.nodesAndObject = {nodes[0], nodes[1], nodes[2], 0u};
        primitive.identity = {
            parentIdentity, NM_CONTINUUM_SURFACE_TRIANGLE, 0u,
            NM_TOPOLOGY_ACTIVE
        };
        primitives[primitiveIndex] = primitive;
    };
    setPrimitive(0u, 0u, {1u, 2u, 3u});
    setPrimitive(4u, 4u, {5u, 6u, 7u});

    NMDeformableContactCandidateGPU candidate{};
    candidate.identity = {0u, 4u, 0u, NM_TOPOLOGY_ACTIVE};
    const std::uint32_t candidateCount = 1u;

    // Full- and partial-parent mortar records exercise the exact production
    // partition. For the partial patch, barycentric x>=0.35 is covered; the
    // covered-only crossing moves the x=1 corner through the lower face while
    // the complement remains separated by more than the CCD floor.
    std::vector<NMIncidenceRangeGPU> ranges(8u);
    std::array<nm_uint4, 2> references{};
    NMCohesiveFaceGPU face{};
    std::vector<NMCohesiveFaceGPU> faces;
    if (hasMortar) {
        ranges[0u] = {0u, 1u, 0u, 0u};
        references[0u] = {4u, 0u, 0u, 0u};
        face.nodesAndFirst = {1u, 2u, 3u, 0u};
        face.adjacency = {
            1u, 0u, kFaceId,
            NM_TOPOLOGY_ACTIVE | NM_TOPOLOGY_PAIRED_COHESIVE |
                NM_TOPOLOGY_COHESIVE_MIXED_MODE |
                NM_TOPOLOGY_COHESIVE_MORTAR
        };
        face.separatedNodes = {5u, 6u, 7u, 0u};
        face.lawIdentity.x = NM_COHESIVE_LAW_MIXED_MODE_BILINEAR;
        const std::array<nm_float4, 3> patchWeights = partialMortar
            ? std::array<nm_float4, 3>{{
                {1.0f, 0.0f, 0.0f, 0.0f},
                {0.35f, 0.65f, 0.0f, 0.0f},
                {0.35f, 0.0f, 0.65f, 0.0f},
            }}
            : std::array<nm_float4, 3>{{
                {1.0f, 0.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 1.0f, 0.0f},
            }};
        for (std::size_t index = 0u; index < patchWeights.size(); ++index) {
            face.mortarSideAVertexWeights[index] = patchWeights[index];
            face.mortarSideBVertexWeights[index] = patchWeights[index];
        }
        faces.push_back(face);
    } else {
        faces.push_back(NMCohesiveFaceGPU{});
    }

    const auto buffer = [device] (const auto& value) {
        using Element = typename std::decay_t<decltype(value)>::value_type;
        return [device newBufferWithBytes:value.data()
            length:std::max<std::size_t>(1u, value.size()) * sizeof(Element)
            options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> dispatchBuffer = [device newBufferWithBytes:&dispatch
        length:sizeof(dispatch) options:MTLResourceStorageModeShared];
    id<MTLBuffer> solverBuffer = [device newBufferWithBytes:&solver
        length:sizeof(solver) options:MTLResourceStorageModeShared];
    const std::vector<NMContinuumObjectGPU> objects{object};
    id<MTLBuffer> primitiveBuffer = buffer(primitives);
    id<MTLBuffer> candidateBuffer = [device newBufferWithBytes:&candidate
        length:sizeof(candidate) options:MTLResourceStorageModeShared];
    id<MTLBuffer> countBuffer = [device newBufferWithBytes:&candidateCount
        length:sizeof(candidateCount) options:MTLResourceStorageModeShared];
    id<MTLBuffer> baseBuffer = buffer(base);
    id<MTLBuffer> trialBuffer = buffer(trial);
    id<MTLBuffer> rangeBuffer = buffer(ranges);
    id<MTLBuffer> referenceBuffer = [device newBufferWithBytes:references.data()
        length:sizeof(references) options:MTLResourceStorageModeShared];
    id<MTLBuffer> faceBuffer = buffer(faces);
    NMMatterStatusGPU initialStatus{};
    id<MTLBuffer> statusBuffer = [device newBufferWithBytes:&initialStatus
        length:sizeof(initialStatus) options:MTLResourceStorageModeShared];
    id<MTLBuffer> objectBuffer = buffer(objects);
    for (id<MTLBuffer> item : {
             dispatchBuffer, solverBuffer, primitiveBuffer, candidateBuffer,
             countBuffer, baseBuffer, trialBuffer, rangeBuffer,
             referenceBuffer, faceBuffer, statusBuffer, objectBuffer})
        require(item != nil, "private self-sweep native fixture buffer allocation failed");

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(command != nil && encoder != nil,
            "private self-sweep native encoder allocation failed");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
    [encoder setBuffer:solverBuffer offset:0u atIndex:1u];
    [encoder setBuffer:primitiveBuffer offset:0u atIndex:2u];
    [encoder setBuffer:candidateBuffer offset:0u atIndex:3u];
    [encoder setBuffer:countBuffer offset:0u atIndex:4u];
    [encoder setBuffer:baseBuffer offset:0u atIndex:5u];
    [encoder setBuffer:trialBuffer offset:0u atIndex:6u];
    [encoder setBuffer:rangeBuffer offset:0u atIndex:7u];
    [encoder setBuffer:referenceBuffer offset:0u atIndex:8u];
    [encoder setBuffer:faceBuffer offset:0u atIndex:9u];
    [encoder setBuffer:statusBuffer offset:0u atIndex:10u];
    [encoder setBuffer:objectBuffer offset:0u atIndex:11u];
    [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
            "private self-sweep native command did not complete");
    return *static_cast<const NMMatterStatusGPU*>(statusBuffer.contents);
}

void verifyArmijoAlphaPublication() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil, "Armijo alpha publication test requires a Metal device");
    const std::string metallibPath = NUMI_MATTER_METALLIB;
    require(!metallibPath.empty(),
            "Armijo alpha publication test requires the production Matter metallib");
    NSError* libraryError = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallibPath.c_str()]]
        error:&libraryError];
    require(library != nil, "could not load production Matter metallib: " +
        (libraryError != nil ? std::string([[libraryError description] UTF8String])
                             : std::string("unknown Metal library error")));
    auto makePipeline = [&](const char* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> function = [library newFunctionWithName:
            [NSString stringWithUTF8String:name]];
        require(function != nil, std::string("production kernel is missing: ") + name);
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline = [device
            newComputePipelineStateWithFunction:function error:&error];
        require(pipeline != nil, std::string("could not create production pipeline ") +
            name + ": " + (error != nil
                ? std::string([[error description] UTF8String])
                : std::string("unknown Metal pipeline error")));
        return pipeline;
    };
    const auto begin = makePipeline(
        "numi_matter_metal::nm_fem_armijo_begin_trial");
    const auto measure = makePipeline(
        "numi_matter_metal::nm_fem_armijo_measure_trial");
    const auto decide = makePipeline(
        "numi_matter_metal::nm_fem_armijo_decide");
    const auto applyFEM = makePipeline(
        "numi_matter_metal::nm_fem_apply_solution");
    const auto applyRigid = makePipeline(
        "numi_matter_metal::nm_rigid_apply_candidate_solution");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(queue != nil, "Armijo alpha publication command queue unavailable");

    NMMatterDispatchGPU dispatch{};
    dispatch.environmentCount = 1u;
    dispatch.objectCount = 1u;
    dispatch.femNodeCount = 1u;
    dispatch.rigidGeneralizedCapacity = 6u;
    dispatch.gravityAndTimestep.w = 1.0f;
    NMMicrostepGPU microstep{};
    microstep.microtickCount = 1u;
    microstep.solverIteration = 3u;
    NMMixedSolverGPU solver{};
    solver.nonlinearIterations.w = 8u;
    solver.residualTolerances.x = 1.0e-6f;
    solver.globalization.x = 1.0e-4f;
    NMContinuumObjectGPU object{};
    object.representation = NM_REPRESENTATION_FEM;
    object.flags = NM_OBJECT_ACTIVE;
    NMIncidenceRangeGPU nodeRange{};
    nodeRange.objectIndex = 0u;
    NMSchedulerStateGPU scheduler{};
    scheduler.numerical.w = 1.0f;
    NMAdaptiveStateGPU adaptive{};
    adaptive.activeRepresentation = NM_REPRESENTATION_FEM;
    NMFEMNodeStateGPU node{};
    node.positionAndMass = {0.0f, 0.0f, 0.0f, 1.0f};
    node.velocityAndInverseMass = {0.0f, 0.0f, 0.0f, 1.0f};
    node.restAndFixed = {0.0f, 0.0f, 0.0f, 0.0f};
    NMFEMFieldStateGPU field{};
    NMMixedMaterialGPU mixedMaterial{};
    NMFGMRESLayoutGPU layout{};
    NMMatterStatusGPU healthy{};
    NMFGMRESStateGPU fgmres{};

    id<MTLBuffer> dispatchBuffer = makeBuffer(
        device, std::vector<NMMatterDispatchGPU>{dispatch});
    id<MTLBuffer> metricsBuffer = makeBuffer(
        device, std::vector<nm_float4>{{1.0f, 0.0f, 1.0f, 0.0f}});
    id<MTLBuffer> lineBuffer = makeBuffer(
        device, std::vector<nm_float4>{{1.0f, 0.0f, 1.0f, 0.0f}});
    id<MTLBuffer> environmentLineBuffer = makeBuffer(
        device, std::vector<float>{-1.0f});
    id<MTLBuffer> trialStatusBuffer = makeBuffer(
        device, std::vector<NMMatterStatusGPU>{healthy});
    id<MTLBuffer> statusBuffer = makeBuffer(
        device, std::vector<NMMatterStatusGPU>{healthy});
    id<MTLBuffer> fgmresBuffer = makeBuffer(
        device, std::vector<NMFGMRESStateGPU>{fgmres});
    id<MTLBuffer> solutionBuffer = makeBuffer(
        device, std::vector<nm_float4>(8u));
    id<MTLBuffer> residualBuffer = makeBuffer(
        device, std::vector<nm_float4>(8u));
    id<MTLBuffer> acceptedBuffer = makeBuffer(
        device, std::vector<NMFEMNodeStateGPU>{node});
    id<MTLBuffer> candidateBuffer = makeBuffer(
        device, std::vector<NMFEMNodeStateGPU>{node});
    id<MTLBuffer> trialCandidateBuffer = makeBuffer(
        device, std::vector<NMFEMNodeStateGPU>{node});
    id<MTLBuffer> objectBuffer = makeBuffer(
        device, std::vector<NMContinuumObjectGPU>{object});
    id<MTLBuffer> nodeRangeBuffer = makeBuffer(
        device, std::vector<NMIncidenceRangeGPU>{nodeRange});
    id<MTLBuffer> schedulerBuffer = makeBuffer(
        device, std::vector<NMSchedulerStateGPU>{scheduler});
    id<MTLBuffer> adaptiveBuffer = makeBuffer(
        device, std::vector<NMAdaptiveStateGPU>{adaptive});
    id<MTLBuffer> fieldBuffer = makeBuffer(
        device, std::vector<NMFEMFieldStateGPU>{field});
    id<MTLBuffer> mixedMaterialBuffer = makeBuffer(
        device, std::vector<NMMixedMaterialGPU>{mixedMaterial});
    id<MTLBuffer> rigidCandidateBuffer = makeBuffer(
        device, std::vector<float>(6u));
    for (id<MTLBuffer> buffer : {
             dispatchBuffer, metricsBuffer, lineBuffer, environmentLineBuffer,
             trialStatusBuffer, statusBuffer, fgmresBuffer, solutionBuffer,
             residualBuffer, acceptedBuffer, candidateBuffer,
             trialCandidateBuffer, objectBuffer, nodeRangeBuffer,
             schedulerBuffer, adaptiveBuffer, fieldBuffer,
             mixedMaterialBuffer, rigidCandidateBuffer})
        require(buffer != nil, "Armijo alpha publication buffer allocation failed");

    auto dispatchOne = [&](id<MTLComputePipelineState> pipeline, auto bind) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            command != nil ? [command computeCommandEncoder] : nil;
        require(command != nil && encoder != nil,
                "Armijo alpha publication encoder allocation failed");
        [encoder setComputePipelineState:pipeline];
        bind(encoder);
        [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
                "Armijo alpha publication kernel command did not complete");
    };
    auto dispatchDecision = [&](const std::uint32_t trialIndex) {
        dispatchOne(begin, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
            [encoder setBytes:&microstep length:sizeof(microstep) atIndex:1u];
            [encoder setBuffer:metricsBuffer offset:0u atIndex:2u];
            [encoder setBuffer:lineBuffer offset:0u atIndex:3u];
            [encoder setBuffer:trialStatusBuffer offset:0u atIndex:4u];
            [encoder setBytes:&trialIndex length:sizeof(trialIndex) atIndex:5u];
            [encoder setBytes:&solver length:sizeof(solver) atIndex:6u];
        });
        dispatchOne(measure, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
            [encoder setBuffer:residualBuffer offset:0u atIndex:1u];
            [encoder setBuffer:metricsBuffer offset:0u atIndex:2u];
            [encoder setBuffer:trialStatusBuffer offset:0u atIndex:3u];
            [encoder setBuffer:trialCandidateBuffer offset:0u atIndex:4u];
        });
        dispatchOne(decide, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
            [encoder setBytes:&microstep length:sizeof(microstep) atIndex:1u];
            [encoder setBytes:&solver length:sizeof(solver) atIndex:2u];
            [encoder setBuffer:metricsBuffer offset:0u atIndex:3u];
            [encoder setBuffer:lineBuffer offset:0u atIndex:4u];
            [encoder setBuffer:trialStatusBuffer offset:0u atIndex:5u];
            [encoder setBuffer:statusBuffer offset:0u atIndex:6u];
            [encoder setBytes:&trialIndex length:sizeof(trialIndex) atIndex:7u];
            [encoder setBuffer:fgmresBuffer offset:0u atIndex:8u];
            [encoder setBuffer:solutionBuffer offset:0u atIndex:9u];
            [encoder setBuffer:environmentLineBuffer offset:0u atIndex:10u];
        });
    };

    // Force four rejected residual trials, then accept the fifth at alpha 1/16.
    const std::array<float, 5> trialNorms{1.10f, 1.05f, 1.02f, 1.001f, 0.98f};
    auto* solution = static_cast<nm_float4*>(solutionBuffer.contents);
    solution[0] = {4.0f, 0.0f, 0.0f, 0.0f};
    solution[2] = {4.0f, 0.0f, 0.0f, 0.0f};
    auto* residual = static_cast<nm_float4*>(residualBuffer.contents);
    auto* metrics = static_cast<nm_float4*>(metricsBuffer.contents);
    auto* line = static_cast<nm_float4*>(lineBuffer.contents);
    auto* environmentAlpha = static_cast<float*>(environmentLineBuffer.contents);
    for (std::uint32_t trialIndex = 0u; trialIndex < trialNorms.size();
         ++trialIndex) {
        residual[0] = {trialNorms[trialIndex], 0.0f, 0.0f, 0.0f};
        dispatchDecision(trialIndex);
        if (trialIndex + 1u < trialNorms.size())
            require(metrics[0].w <= 0.5f && environmentAlpha[0] == -1.0f,
                    "Armijo accepted a non-descending prefix or published an unaccepted environment alpha");
    }
    require(metrics[0].w == 1.0f && line[0].x == 0.0625f &&
            environmentAlpha[0] == 0.0625f && line[0].z == 5.0f &&
            line[0].w == float(microstep.solverIteration),
            "accepted FEM Armijo alpha or trial identity was not published consistently");

    auto applyCandidates = [&] {
        dispatchOne(applyFEM, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
            [encoder setBytes:&microstep length:sizeof(microstep) atIndex:1u];
            [encoder setBuffer:objectBuffer offset:0u atIndex:2u];
            [encoder setBuffer:nodeRangeBuffer offset:0u atIndex:3u];
            [encoder setBuffer:schedulerBuffer offset:0u atIndex:4u];
            [encoder setBuffer:adaptiveBuffer offset:0u atIndex:5u];
            [encoder setBuffer:acceptedBuffer offset:0u atIndex:6u];
            [encoder setBuffer:solutionBuffer offset:0u atIndex:7u];
            [encoder setBuffer:candidateBuffer offset:0u atIndex:8u];
            [encoder setBuffer:lineBuffer offset:0u atIndex:9u];
            [encoder setBuffer:mixedMaterialBuffer offset:0u atIndex:10u];
            [encoder setBuffer:fieldBuffer offset:0u atIndex:11u];
        });
        dispatchOne(applyRigid, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
            [encoder setBuffer:solutionBuffer offset:0u atIndex:1u];
            [encoder setBuffer:environmentLineBuffer offset:0u atIndex:2u];
            [encoder setBuffer:rigidCandidateBuffer offset:0u atIndex:3u];
            [encoder setBytes:&layout length:sizeof(layout) atIndex:30u];
        });
    };
    applyCandidates();
    const auto femVelocity = static_cast<const NMFEMNodeStateGPU*>(
        candidateBuffer.contents)[0].velocityAndInverseMass.x;
    const auto rigidVelocity = static_cast<const float*>(
        rigidCandidateBuffer.contents)[0];
    require(femVelocity == 0.25f && rigidVelocity == 0.25f,
            "FEM and free-rigid corrections did not consume the same accepted 1/16 alpha");

    // A certified no-op is deliberately different: the candidate correction is
    // zeroed and alpha one is published so both application kernels preserve it.
    *static_cast<nm_float4*>(metricsBuffer.contents) =
        {0.5e-6f, INFINITY, 1.0f, 0.0f};
    *static_cast<nm_float4*>(lineBuffer.contents) = {1.0f, 0.0f, 1.0f, 0.0f};
    *static_cast<float*>(environmentLineBuffer.contents) = -1.0f;
    *static_cast<NMMatterStatusGPU*>(trialStatusBuffer.contents) = {};
    *static_cast<NMMatterStatusGPU*>(statusBuffer.contents) = {};
    *static_cast<NMFEMNodeStateGPU*>(candidateBuffer.contents) = node;
    *static_cast<NMFEMNodeStateGPU*>(acceptedBuffer.contents) = node;
    std::fill_n(static_cast<float*>(rigidCandidateBuffer.contents), 6u, 0.0f);
    std::fill_n(static_cast<nm_float4*>(solutionBuffer.contents), 8u,
                nm_float4{});
    solution = static_cast<nm_float4*>(solutionBuffer.contents);
    solution[0] = {4.0f, 0.0f, 0.0f, 0.0f};
    solution[2] = {4.0f, 0.0f, 0.0f, 0.0f};
    residual[0] = {0.5e-6f, 0.0f, 0.0f, 0.0f};
    dispatchDecision(0u);
    require(metrics[0].w == 2.0f && line[0].x == 1.0f &&
            environmentAlpha[0] == 1.0f &&
            solution[0].x == 0.0f && solution[2].x == 0.0f,
            "certified no-op did not publish alpha one with an exactly zero correction");
    applyCandidates();
    require(static_cast<const NMFEMNodeStateGPU*>(candidateBuffer.contents)[0]
                .velocityAndInverseMass.x == 0.0f &&
            static_cast<const float*>(rigidCandidateBuffer.contents)[0] == 0.0f,
            "certified no-op changed a FEM or rigid candidate");
    std::cout << "armijo_environment_alpha_publication=pass"
              << " rejected_prefix_trials=4 selected_alpha=0.0625"
              << " applied_fem_delta=0.25 applied_rigid_delta=0.25"
              << " certified_noop_fem_alpha=1 certified_noop_rigid_alpha=1"
              << " noop_correction_zero=true direct_production_kernels=true"
              << " physical_validation=false\n";
}

void verifyPrivateSelfSweepValidator() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil, "private self-sweep test requires a Metal device");
    const std::string metallibPath = NUMI_MATTER_METALLIB;
    require(!metallibPath.empty(),
            "private self-sweep test requires the production Matter metallib");
    NSError* libraryError = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallibPath.c_str()]]
        error:&libraryError];
    require(library != nil, "could not load production Matter metallib: " +
        (libraryError != nil ? std::string([[libraryError description] UTF8String])
                             : std::string("unknown Metal library error")));
    id<MTLFunction> function = [library
        newFunctionWithName:
            @"numi_matter_metal::nm_contact_validate_armijo_deformable_sweep"];
    require(function != nil,
            "production private self-sweep validator kernel is missing");
    NSError* pipelineError = nil;
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function error:&pipelineError];
    require(pipeline != nil, "could not create production self-sweep pipeline: " +
        (pipelineError != nil ? std::string([[pipelineError description] UTF8String])
                              : std::string("unknown Metal pipeline error")));
    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(queue != nil, "private self-sweep test command queue unavailable");

    const NMMatterStatusGPU crossing = runPrivateSelfSweepKernelCase(
        device, queue, pipeline, PrivateSelfSweepCase::unbondedCrossing);
    require(crossing.code == NM_STATUS_CONTACT_FAILURE,
            "endpoint-clear triangle sweep crossing was not rejected by production validator");
    const NMMatterStatusGPU miss = runPrivateSelfSweepKernelCase(
        device, queue, pipeline, PrivateSelfSweepCase::separatedMiss);
    require(miss.code == NM_STATUS_SUCCESS,
            "separated triangle sweep miss was not admitted by production validator");
    const NMMatterStatusGPU covered = runPrivateSelfSweepKernelCase(
        device, queue, pipeline, PrivateSelfSweepCase::fullyCoveredCrossing);
    require(covered.code == NM_STATUS_SUCCESS,
            "fully covered active mortar parent pair duplicated self-contact rejection");
    const NMMatterStatusGPU partialUncovered = runPrivateSelfSweepKernelCase(
        device, queue, pipeline,
        PrivateSelfSweepCase::partialUncoveredCrossing);
    require(partialUncovered.code == NM_STATUS_CONTACT_FAILURE,
            "crossing through the uncovered complement of a partial active mortar patch was admitted");
    const NMMatterStatusGPU partialCovered = runPrivateSelfSweepKernelCase(
        device, queue, pipeline, PrivateSelfSweepCase::partialCoveredCrossing);
    require(partialCovered.code == NM_STATUS_SUCCESS,
            "crossing confined to a partial active mortar patch duplicated self-contact rejection");
    const NMMatterStatusGPU touchingSeparating = runPrivateSelfSweepKernelCase(
        device, queue, pipeline,
        PrivateSelfSweepCase::initiallyTouchingSeparating);
    require(touchingSeparating.code == NM_STATUS_CONTACT_FAILURE,
            "initially touching unbonded separating ray was not explicitly rejected as unsupported");
    std::cout << "private_self_sweep_validator=pass"
              << " endpoint_clear_crossing=rejected separated_miss=accepted"
              << " fully_covered_mortar_crossing=excluded_from_self_contact"
              << " partial_uncovered_crossing=rejected"
              << " partial_covered_crossing=excluded_from_self_contact"
              << " initial_unbonded_touch_separating=rejected_unsupported"
              << " direct_production_kernel=true runtime_transaction=false\n";
}

double maxFreeDisplacement(const RuntimeStateSnapshot& a,
                           const RuntimeStateSnapshot& b,
                           const std::uint32_t nodeCount) {
    double maximum = 0.0;
    for (std::uint32_t nodeIndex = 0u; nodeIndex < nodeCount; ++nodeIndex) {
        const auto& current = a.femNodes.at(nodeIndex);
        const auto& start = b.femNodes.at(nodeIndex);
        if (start.restAndFixed.w > 0.5f || start.positionAndMass.w <= 0.0f)
            continue;
        const double dx = double(current.positionAndMass.x) - start.positionAndMass.x;
        const double dy = double(current.positionAndMass.y) - start.positionAndMass.y;
        const double dz = double(current.positionAndMass.z) - start.positionAndMass.z;
        maximum = std::max(maximum, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    return maximum;
}

void verifySelectedTrialsAndCandidateAgreement(const std::string& trace) {
    const std::vector<TrialRecord> records = traceRecords(trace);
    require(!records.empty(), "full-contact residual Armijo emitted no per-Newton records");
    const bool acceptedTrial = std::any_of(records.begin(), records.end(),
        [](const TrialRecord& row) {
            return row.accepted && std::isfinite(row.alpha) && row.alpha > 0.0 &&
                std::abs(row.alpha - row.selected) <= 2.0e-6;
        });
    require(acceptedTrial,
        "no tested Armijo fraction was selected by the production FEM update");
    bool matchedProductionReassembly = false;
    double worstRelative = 0.0;
    for (std::size_t i = 0u; i + 1u < records.size(); ++i) {
        const auto& selected = records[i];
        const auto& nextProductionBase = records[i + 1u];
        if (!selected.accepted || selected.iteration + 1u != nextProductionBase.iteration ||
            !std::isfinite(selected.trial) || !std::isfinite(nextProductionBase.base))
            continue;
        const double relative = std::abs(selected.trial - nextProductionBase.base) /
            std::max({1.0e-8, std::abs(selected.trial), std::abs(nextProductionBase.base)});
        worstRelative = std::max(worstRelative, relative);
        matchedProductionReassembly = true;
    }
    require(matchedProductionReassembly,
        "trace lacks an accepted-trial/next-production-base candidate pair");
    require(worstRelative <= 5.0e-3,
        "private trial norm does not match the next production residual at the same candidate");
    std::cout << "selected_trial=pass accepted_alpha_matches_applied_alpha=true"
              << " private_to_next_production_base_relative_error=" << worstRelative << '\n';
}

void requireAccepted(const RuntimeStateSnapshot& snapshot) {
    require(snapshot.statuses.size() == 1u &&
            snapshot.solverCertificates.size() == 1u,
            "single-environment fixture status/certificate layout mismatch");
    if (snapshot.statuses[0].code != NM_STATUS_SUCCESS) {
        const NMMatterStatusGPU& status = snapshot.statuses[0];
        const NMSolverCertificateGPU& certificate =
            snapshot.solverCertificates[0];
        std::ostringstream context;
        context << std::setprecision(9)
                << "native full-contact transaction rejected"
                << " code=" << status.code
                << " env=" << status.environment
                << " object=" << status.objectIndex
                << " failing_index=" << status.failingIndex
                << " completed_microsteps=" << status.completedMicrosteps
                << " fgmres_iterations=" << status.fgmresIterations
                << " status_contact_count=" << status.contactCount
                << " status_event_count=" << status.eventCount
                << " status_diag=[" << status.diagnostics.x << ','
                << status.diagnostics.y << ',' << status.diagnostics.z << ','
                << status.diagnostics.w << ']'
                << " certificate_valid=" << certificate.validity.w
                << " certificate_nonlinear=[" << certificate.nonlinear.x << ','
                << certificate.nonlinear.y << ',' << certificate.nonlinear.z
                << ',' << certificate.nonlinear.w << ']'
                << " certificate_contact=[" << certificate.contact.x << ','
                << certificate.contact.y << ',' << certificate.contact.z << ','
                << certificate.contact.w << ']'
                << " certificate_validity=[" << certificate.validity.x << ','
                << certificate.validity.y << ',' << certificate.validity.z << ','
                << certificate.validity.w << ']'
                << " contact_samples=" << snapshot.contactSamples.size();

        struct NearbyContact {
            std::size_t slot = 0u;
            float separation = std::numeric_limits<float>::infinity();
            const NMContactSampleGPU* sample = nullptr;
        };
        std::vector<NearbyContact> validContacts;
        validContacts.reserve(snapshot.contactSamples.size());
        for (std::size_t slot = 0u; slot < snapshot.contactSamples.size(); ++slot) {
            const NMContactSampleGPU& sample = snapshot.contactSamples[slot];
            if ((sample.identity.w & NM_CONTACT_VALID) == 0u ||
                !std::isfinite(sample.pointAndSeparation.w)) {
                continue;
            }
            validContacts.push_back({slot, sample.pointAndSeparation.w, &sample});
        }
        std::stable_sort(validContacts.begin(), validContacts.end(),
            [](const NearbyContact& a, const NearbyContact& b) {
                return a.separation < b.separation;
            });
        context << " valid_contact_samples=" << validContacts.size();
        constexpr std::size_t kMaxNearbyContactDiagnostics = 4u;
        for (std::size_t i = 0u;
             i < std::min(kMaxNearbyContactDiagnostics, validContacts.size());
             ++i) {
            const NearbyContact& nearby = validContacts[i];
            const NMContactSampleGPU& sample = *nearby.sample;
            context << " nearest_contact_" << i
                    << "={pair_slot:" << nearby.slot
                    << ",node:" << sample.identity.x
                    << ",proxy:" << sample.identity.y
                    << ",object:" << sample.identity.z
                    << ",flags:" << sample.identity.w
                    << ",gap_m:" << nearby.separation
                    << ",normal_velocity:" << sample.normalAndVelocity.w
                    << ",normal_impulse:" << sample.impulseAndNormal.w
                    << ",barrier_k:" << sample.barrier.w << '}';
        }
        throw std::runtime_error(context.str());
    }
    require(snapshot.solverCertificates[0].validity.w > 0.5f,
            "accepted full-contact transaction lacks final production certificate");
}

bool hasMortarParentContact(const RuntimeStateSnapshot& snapshot,
                            const MeshAndWorld& source) {
    return std::any_of(snapshot.deformableContactHistories.begin(),
        snapshot.deformableContactHistories.end(),
        [&](const NMDeformableContactHistoryGPU& history) {
            if (!(history.laggedTangentAndFriction.w > 0.5f)) return false;
            return (history.source.x == source.mortarPrimitiveA &&
                    history.source.y == source.mortarPrimitiveB) ||
                   (history.source.x == source.mortarPrimitiveB &&
                    history.source.y == source.mortarPrimitiveA);
        });
}

void verifyPartialMortarSource(const MeshAndWorld& source) {
    const ObjectSource& object = source.world.objects.at(0u);
    const auto& patch = object.femCohesiveMortarPatches.at(0u);
    const auto& bary = patch.sideAVertexBarycentric;
    const double areaRatio = std::abs(
        (bary[1][0] - bary[0][0]) * (bary[2][1] - bary[0][1]) -
        (bary[1][1] - bary[0][1]) * (bary[2][0] - bary[0][0]));
    require(std::isfinite(areaRatio) && areaRatio > 1.0e-6 &&
            areaRatio < 0.99,
            "partial-mortar fixture does not cover a strict subset of its parent face");
    require(source.mortarPrimitiveA != NM_INVALID_INDEX &&
            source.mortarPrimitiveB != NM_INVALID_INDEX,
            "partial-mortar fixture lacks stable parent primitive identifiers");
    require(source.nodes.size() == 36u && object.tetrahedra.size() == 48u &&
            source.interfaceGapM > 0.0 &&
            std::isfinite(source.interfaceGapM),
            "split-pad source dimensions or interface gap are invalid");
    require(patch.firstTetrahedron < 24u &&
            patch.secondTetrahedron >= 24u &&
            patch.secondTetrahedron < object.tetrahedra.size(),
            "partial cohesive face is not bound to one lower and one upper parent tet");

    auto containsFace = [](const TetrahedronSource& tetrahedron,
                           const std::array<std::uint32_t, 3>& face) {
        return std::all_of(face.begin(), face.end(),
            [&](const std::uint32_t index) {
                return std::find(tetrahedron.nodes.begin(),
                    tetrahedron.nodes.end(), index) != tetrahedron.nodes.end();
            });
    };
    require(containsFace(object.tetrahedra[patch.firstTetrahedron],
                         patch.sideANodes) &&
            containsFace(object.tetrahedra[patch.secondTetrahedron],
                         patch.sideBNodes),
            "cohesive mortar side vertices do not match their recorded parent tetrahedra");

    constexpr double kGeometryTolerance = 1.0e-12;
    const double upperInterfaceZ = source.interfaceZ + source.interfaceGapM;
    for (std::size_t tetIndex = 0u;
         tetIndex < object.tetrahedra.size(); ++tetIndex) {
        const auto& tet = object.tetrahedra[tetIndex].nodes;
        std::array<std::array<double, 3>, 4> points{};
        double minimumZ = std::numeric_limits<double>::infinity();
        double maximumZ = -std::numeric_limits<double>::infinity();
        for (unsigned local = 0u; local < 4u; ++local) {
            require(tet[local] < source.nodes.size(),
                    "split-pad tetrahedron references an invalid FEM node");
            points[local] = source.nodes[tet[local]];
            minimumZ = std::min(minimumZ, points[local][2]);
            maximumZ = std::max(maximumZ, points[local][2]);
        }
        require(std::isfinite(signedTetrahedronVolume6(points)) &&
                signedTetrahedronVolume6(points) > 0.0,
                "split-pad source contains an inverted or degenerate tetrahedron");
        if (tetIndex < 24u)
            require(maximumZ <= source.interfaceZ + kGeometryTolerance,
                    "lower-half tetrahedron crosses the split plane");
        else
            require(minimumZ >= upperInterfaceZ - kGeometryTolerance,
                    "upper-half tetrahedron intrudes into the split gap");
    }

    const auto hasContactNode = [&](const std::uint32_t index) {
        return std::binary_search(object.femContactNodes.begin(),
            object.femContactNodes.end(), index);
    };
    for (std::size_t index = 0u;
         index < source.lowerInterfaceNodes.size(); ++index) {
        const auto& lower = source.nodes[source.lowerInterfaceNodes[index]];
        const auto& upper = source.nodes[source.upperInterfaceNodes[index]];
        require(hasContactNode(source.lowerInterfaceNodes[index]) &&
                hasContactNode(source.upperInterfaceNodes[index]),
                "split interface omits an exposed node from the self-contact surface");
        require(lower[2] <= source.interfaceZ + kGeometryTolerance &&
                upper[2] >= upperInterfaceZ - kGeometryTolerance &&
                std::abs((upper[2] - lower[2]) - source.interfaceGapM) <=
                    kGeometryTolerance &&
                lower[0] == upper[0] && lower[1] == upper[1],
                "split interface vertices do not preserve the positive, uniform gap");
    }
    for (std::size_t index = 0u;
         index < object.femContactNodes.size(); ++index) {
        require(object.femContactNodes[index] < source.nodes.size(),
                "self-contact surface contains an invalid FEM node index");
        if (index != 0u)
            require(object.femContactNodes[index - 1u] <
                        object.femContactNodes[index],
                    "self-contact surface node list is not unique and sorted");
    }
    for (unsigned local = 0u; local < 3u; ++local) {
        const auto& lower = source.nodes[patch.sideANodes[local]];
        const auto& upper = source.nodes[patch.sideBNodes[local]];
        require(std::abs((upper[2] - lower[2]) - source.interfaceGapM) <=
                    kGeometryTolerance &&
                lower[0] == upper[0] && lower[1] == upper[1],
                "partial mortar patch pairs mismatched lower/upper triangle vertices");
    }
}

void runDormantUncoveredContact(const MaterialProgram& bulk) {
    MeshAndWorld source = makeWorld(bulk, false, true, 5.0e-6);
    verifyPartialMortarSource(source);
    Fixture run(source);
    const RuntimeStateSnapshot initial = run.snapshot();
    require(initial.cohesiveFaces.size() == 1u &&
            (initial.cohesiveFaces[0].adjacency.w &
                NM_TOPOLOGY_COHESIVE_DORMANT) != 0u,
            "partial mortar control must begin dormant");
    const StepResult accepted = run.step(0u, false, false);
    requireAccepted(accepted.snapshot);
    require((accepted.snapshot.cohesiveFaces[0].adjacency.w &
                NM_TOPOLOGY_COHESIVE_DORMANT) != 0u,
            "uncommanded dormant mortar face activated");
    require(hasMortarParentContact(accepted.snapshot, source),
            "dormant partial mortar face lost ordinary contact on its parent pair");
    std::cout << "dormant_mortar_uncovered_contact=pass"
              << " interface_remained_dormant=true parent_face_contact_history=true\n";
}

AcceptedRollbackSequence runAcceptedAndRollback(
        const MaterialProgram& bulk,
        const bool aliasProductionSurfaceScratch = false,
        const bool aliasCohesiveCandidateScratch = false) {
    MeshAndWorld source = makeWorld(bulk, false, true, 5.0e-6);
    verifyPartialMortarSource(source);
    Fixture run(source, false, false, false, false,
                aliasProductionSurfaceScratch,
                aliasCohesiveCandidateScratch);
    const RuntimeStateSnapshot initial = run.snapshot();
    require(initial.cohesiveFaces.size() == 1u &&
            (initial.cohesiveFaces[0].adjacency.w &
                NM_TOPOLOGY_COHESIVE_DORMANT) != 0u,
            "fixture must begin with one dormant mixed-mode mortar face");
    require(std::count_if(source.world.objects[0].femFixedNodes.begin(),
                source.world.objects[0].femFixedNodes.end(),
                [](std::uint32_t) { return true; }) == 4,
            "fixture must have four clamps and free FEM unknowns");

    std::string captured;
    StepResult closed;
    {
        StderrCapture stderrCapture;
        // A fresh public control cursor is zero. Use distinct nonzero labels
        // so this test observes advancement, rather than comparing 0 with 0
        // after a genuinely accepted first transaction.
        closed = run.step(1u, false, true);
        captured = stderrCapture.finish();
    }
    closed.trace = captured;
    retainFixtureTrace("activation", captured);
    emitSelectedDeformableContactCapture(captured);
    require(captured.find("fem_full_contact_residual_armijo_enabled=") !=
                std::string::npos &&
            captured.find("fem_full_contact_residual_armijo_skipped=") ==
                std::string::npos,
            "full-contact Armijo did not admit the mixed/tool/self-contact source: " + captured);
    requireAccepted(closed.snapshot);
    require(closed.snapshot.cohesiveFaces.size() == 1u &&
            (closed.snapshot.cohesiveFaces[0].adjacency.w &
                NM_TOPOLOGY_COHESIVE_DORMANT) == 0u,
            "accepted explicit setting command did not activate the mortar face");
    require(hasMortarParentContact(closed.snapshot, source),
            "active partial mortar face lost ordinary contact on its uncovered parent region");
    require(closed.snapshot.controlStep > initial.controlStep,
            "accepted endpoint activation did not advance the accepted cursor");

    // The second selected step starts from the accepted, active interface. It
    // must execute a nonzero FEM correction while the mixed face, rigid tools,
    // and private deformable self-contact path are all enabled.
    setenv("NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT", "2", 1);
    std::string loadedTrace;
    StepResult loaded;
    {
        StderrCapture stderrCapture;
        loaded = run.step(2u);
        loadedTrace = stderrCapture.finish();
    }
    loaded.trace = loadedTrace;
    retainFixtureTrace("loaded", loadedTrace);
    emitSelectedDeformableContactCapture(loadedTrace);
    requireAccepted(loaded.snapshot);
    require(loaded.snapshot.cohesiveFaces.size() == 1u &&
            (loaded.snapshot.cohesiveFaces[0].adjacency.w &
                NM_TOPOLOGY_COHESIVE_DORMANT) == 0u,
            "loaded solve lost its accepted mixed mortar interface");
    require(hasMortarParentContact(loaded.snapshot, source),
            "loaded active mortar solve lost uncovered-parent contact history");
    require(maxFreeDisplacement(loaded.snapshot, closed.snapshot, run.nodeCount) > 1.0e-8,
            "accepted full-contact Armijo step had no nonzero free-node response");
    require(loadedTrace.find("fem_full_contact_residual_armijo_enabled=") !=
                std::string::npos &&
            loadedTrace.find("fem_residual_armijo_iteration=") != std::string::npos,
            "active-interface nonlinear correction did not enter the Armijo private trial path");
    verifySelectedTrialsAndCandidateAgreement(loadedTrace);

    setenv("NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT", "3", 1);
    const RuntimeStateSnapshot beforeFailure = loaded.snapshot;
    std::string rejectedTrace;
    StepResult rejected;
    {
        StderrCapture stderrCapture;
        rejected = run.step(3u, true, false);
        rejectedTrace = stderrCapture.finish();
    }
    rejected.trace = rejectedTrace;
    retainFixtureTrace("rejected", rejectedTrace);
    require(rejectedTrace.find("fem_full_contact_residual_armijo_enabled=") !=
                std::string::npos &&
            rejectedTrace.find("fem_residual_armijo_iteration=") !=
                std::string::npos,
            "rejected full-contact trial did not execute the selected native Armijo path");
    require(rejected.snapshot.statuses[0].code != NM_STATUS_SUCCESS,
            "postCommit motion-type fault unexpectedly accepted");
    require(rejected.snapshot.solverCertificates.size() == 1u &&
            rejected.snapshot.solverCertificates[0].validity.w <= 0.5f,
            "rejected full-contact transaction retained its accepted-certificate bit");
    require(rejected.snapshot.controlStep == beforeFailure.controlStep &&
            rejected.snapshot.physicsSubstep == beforeFailure.physicsSubstep,
            "rejected full-contact transaction advanced accepted cursor");
    require(rejected.snapshot.femNodes.size() == beforeFailure.femNodes.size() &&
            std::memcmp(rejected.snapshot.femNodes.data(), beforeFailure.femNodes.data(),
                beforeFailure.femNodes.size() * sizeof(NMFEMNodeStateGPU)) == 0,
            "rejected full-contact transaction changed accepted FEM state");
    require(rejected.snapshot.cohesiveFaces.size() == beforeFailure.cohesiveFaces.size() &&
            std::memcmp(rejected.snapshot.cohesiveFaces.data(), beforeFailure.cohesiveFaces.data(),
                beforeFailure.cohesiveFaces.size() * sizeof(NMCohesiveFaceGPU)) == 0,
            "rejected full-contact transaction changed accepted cohesive history");
    require(rejected.snapshot.femMaterialState.size() == beforeFailure.femMaterialState.size() &&
            std::memcmp(rejected.snapshot.femMaterialState.data(),
                beforeFailure.femMaterialState.data(),
                beforeFailure.femMaterialState.size() * sizeof(float)) == 0,
            "rejected full-contact transaction changed accepted bulk history");
    require(rejected.snapshot.deformableContactHistories.size() ==
                beforeFailure.deformableContactHistories.size() &&
            std::memcmp(rejected.snapshot.deformableContactHistories.data(),
                beforeFailure.deformableContactHistories.data(),
                beforeFailure.deformableContactHistories.size() *
                    sizeof(NMDeformableContactHistoryGPU)) == 0,
            "rejected full-contact transaction changed accepted deformable-contact history");
    std::cout << "full_contact_rejection_rollback=pass certificate_cleared=true"
              << " cursor_and_all_histories_unchanged=true\n";
    return {initial, std::move(closed), std::move(loaded), std::move(rejected)};
}

template<class T>
void requireVectorBitIdentical(const std::vector<T>& expected,
                               const std::vector<T>& actual,
                               const char* field,
                               const char* stage) {
    static_assert(std::is_trivially_copyable_v<T>);
    require(expected.size() == actual.size(),
            std::string("sparse/dense ") + stage + " " + field +
                " count differs");
    if (!expected.empty())
        require(std::memcmp(expected.data(), actual.data(),
                    expected.size() * sizeof(T)) == 0,
                std::string("sparse/dense ") + stage + " " + field +
                    " bytes differ");
}

void requireSnapshotOperatorParity(const RuntimeStateSnapshot& dense,
                                   const RuntimeStateSnapshot& sparse,
                                   const char* stage) {
    require(dense.available && sparse.available,
            std::string("sparse/dense ") + stage + " snapshot unavailable");
    require(dense.sourcePhysicsFingerprint == sparse.sourcePhysicsFingerprint,
            std::string("sparse/dense ") + stage +
                " source physics identity differs");
    require(dense.controlStep == sparse.controlStep &&
                dense.physicsSubstep == sparse.physicsSubstep &&
                dense.materialStateStride == sparse.materialStateStride,
            std::string("sparse/dense ") + stage +
                " accepted cursor or material stride differs");
    requireVectorBitIdentical(dense.femNodes, sparse.femNodes,
                              "FEM nodes", stage);
    requireVectorBitIdentical(dense.femMaterialState, sparse.femMaterialState,
                              "bulk history", stage);
    requireVectorBitIdentical(dense.cohesiveFaces, sparse.cohesiveFaces,
                              "cohesive history", stage);
    requireVectorBitIdentical(dense.contactHistories, sparse.contactHistories,
                              "rigid-contact history", stage);
    requireVectorBitIdentical(dense.deformableContactHistories,
                              sparse.deformableContactHistories,
                              "deformable-contact history", stage);
    requireVectorBitIdentical(dense.contactSamples, sparse.contactSamples,
                              "accepted contact samples", stage);
    requireVectorBitIdentical(dense.rigidStates, sparse.rigidStates,
                              "rigid states", stage);
    requireVectorBitIdentical(dense.reactions, sparse.reactions,
                              "rigid reactions", stage);
    requireVectorBitIdentical(dense.statuses, sparse.statuses,
                              "status", stage);
    requireVectorBitIdentical(dense.solverCertificates,
                              sparse.solverCertificates,
                              "solver certificate", stage);
}

void requireTraceOperatorParity(const std::string& dense,
                               const std::string& sparse,
                               const char* stage) {
    const std::vector<TrialRecord> denseTrials = traceRecords(dense);
    const std::vector<TrialRecord> sparseTrials = traceRecords(sparse);
    require(!denseTrials.empty() && !sparseTrials.empty(),
            std::string("sparse/dense ") + stage +
                " comparison requires nonempty production Armijo traces");
    require(denseTrials.size() == sparseTrials.size(),
            std::string("sparse/dense ") + stage +
                " Armijo trial count differs");
    for (std::size_t index = 0u; index < denseTrials.size(); ++index) {
        const TrialRecord& a = denseTrials[index];
        const TrialRecord& b = sparseTrials[index];
        const auto sameNumber = [](const double x, const double y) {
            return std::isfinite(x) && std::isfinite(y) &&
                std::abs(x - y) <= 2.0e-7 *
                    std::max({1.0e-12, std::abs(x), std::abs(y)});
        };
        require(a.iteration == b.iteration && a.accepted == b.accepted &&
                    sameNumber(a.base, b.base) && sameNumber(a.trial, b.trial) &&
                    sameNumber(a.alpha, b.alpha) &&
                    sameNumber(a.selected, b.selected),
                std::string("sparse/dense ") + stage +
                    " production residual, trial, or selected alpha differs");
    }
}

void requireSequenceOperatorParity(const AcceptedRollbackSequence& dense,
                                   const AcceptedRollbackSequence& candidate,
                                   const bool expectSparseCandidate) {
    requireSnapshotOperatorParity(dense.initial, candidate.initial, "initial");
    const std::array<const StepResult*, 3> denseSteps = {
        &dense.activation, &dense.loaded, &dense.rejected};
    const std::array<const StepResult*, 3> candidateSteps = {
        &candidate.activation, &candidate.loaded, &candidate.rejected};
    constexpr std::array<const char*, 3> stageNames = {
        "activation", "loaded", "rejected"};
    for (std::size_t index = 0u; index < stageNames.size(); ++index) {
        requireSnapshotOperatorParity(denseSteps[index]->snapshot,
                                      candidateSteps[index]->snapshot,
                                      stageNames[index]);
        requireTraceOperatorParity(denseSteps[index]->trace,
                                   candidateSteps[index]->trace,
                                   stageNames[index]);
        const std::string denseMarker =
            "\"private_contact_layout\":\"dense\"";
        const std::string sparseMarker =
            "\"private_contact_layout\":\"sparse\"";
        require(denseSteps[index]->trace.find(denseMarker) != std::string::npos,
                std::string("dense ") + stageNames[index] +
                    " did not execute the dense private residual path");
        if (expectSparseCandidate) {
            require(candidateSteps[index]->trace.find(sparseMarker) !=
                        std::string::npos &&
                    candidateSteps[index]->trace.find("\"sparse_parity\":true") !=
                        std::string::npos,
                    std::string("sparse ") + stageNames[index] +
                        " did not execute the parity-checked sparse path");
        } else {
            require(candidateSteps[index]->trace.find(denseMarker) !=
                        std::string::npos,
                    std::string("candidate ") + stageNames[index] +
                        " did not execute the dense private residual path");
        }
    }
}

void runSparsePrivateParity(const MaterialProgram& bulk) {
    const MeshAndWorld geometry = makeWorld(bulk, false, true, 5.0e-6);
    const CompiledWorld compiled = compile(geometry.world);
    const std::uint32_t pairCount = compiled.dispatch.contactPairCount;
    require(pairCount > 1u,
            "sparse parity fixture requires at least two canonical FEM/tool pairs");

    ScopedEnvironmentValue clearCapacity(
        "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_CAPACITY", nullptr);
    ScopedEnvironmentValue clearParity(
        "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_PARITY", nullptr);
    const std::string outputRoot = diagnosticStateOutputPath;
    std::unique_ptr<DiagnosticStateOutput> parentDiagnosticOutput =
        std::move(diagnosticStateOutput);
    const auto selectOutput = [&](const char* label) {
        if (outputRoot.empty()) return;
        diagnosticStateOutput.reset();
        const auto directory = std::filesystem::path(outputRoot) / label;
        if (std::filesystem::exists(directory))
            throw std::runtime_error(
                "sparse parity trace subdirectory must be new: " +
                directory.string());
        diagnosticStateOutputPath = directory.string();
        diagnosticStateOutput = std::make_unique<DiagnosticStateOutput>(
            diagnosticStateOutputPath);
    };
    const auto restoreOutput = [&]() {
        diagnosticStateOutput.reset();
        diagnosticStateOutput = std::move(parentDiagnosticOutput);
        diagnosticStateOutputPath = outputRoot;
    };

    AcceptedRollbackSequence dense;
    AcceptedRollbackSequence sparse;
    try {
        selectOutput("dense");
        dense = runAcceptedAndRollback(bulk);

        const std::string capacity = std::to_string(pairCount - 1u);
        ScopedEnvironmentValue sparseCapacity(
            "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_CAPACITY", capacity.c_str());
        ScopedEnvironmentValue sparseParity(
            "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_PARITY", "1");
        selectOutput("sparse");
        sparse = runAcceptedAndRollback(bulk);
        restoreOutput();
    } catch (...) {
        restoreOutput();
        throw;
    }
    requireSequenceOperatorParity(dense, sparse, true);
    std::cout << "sparse_private_dense_parity=pass"
              << " canonical_pair_count=" << pairCount
              << " sparse_capacity=" << (pairCount - 1u)
              << " candidate_set_strictly_bounded_below_dense=true"
              << " internal_sample_field_parity=true"
              << " accepted_and_rollback_histories_bit_identical=true"
              << " physical_validation=false synthetic_law=true\n";
}

#if defined(NUMI_CARDBOARD_METAL_WORLD)
struct SparseFreeBodyRun {
    std::vector<RuntimeStateSnapshot> accepted;
    std::vector<std::string> traces;
    std::vector<MRBodyStateGPU> finalSceneBodies;
    float initialBodyZ = 0.0f;
    float finalBodyZ = 0.0f;
    float finalBodyVelocityZ = 0.0f;
    bool terminalAccepted = false;
    std::uint32_t terminalStatus = NM_STATUS_INVALID_DISPATCH;
    double activationGapM = 0.0;
    RuntimeStateSnapshot initial;
    RuntimeStateSnapshot activated;
    std::string activationTrace;
    RuntimeStateSnapshot terminal;
};

void requireAcceptedStateUnchanged(const RuntimeStateSnapshot& before,
                                   const RuntimeStateSnapshot& after,
                                   const char* reason);
void requireFullSnapshotRestoreParity(const RuntimeStateSnapshot& before,
                                      const RuntimeStateSnapshot& after,
                                      const char* reason);

MeshAndWorld makeSparseFreeBodyWorld(const MaterialProgram& bulk,
                                     const double supportGapM) {
    MeshAndWorld source = makeWorld(bulk, false, true, kFreeBodyMortarGap);
    source.world.mixedSolver.femPreconditioner =
        FEMPreconditionerMode::scalarDiagonal;
    source.world.objects.at(0u).femPreconditionerRegionIds.clear();
    RigidProxySource& support = source.world.rigidProxies.at(kAnvilBody);
    require(support.shape == NM_RIGID_BOX &&
                std::isfinite(supportGapM) && supportGapM > 0.0,
            "sparse free-body source requires a BOX and positive clearance");
    // This fixture uses a centered box whose scene-body origin is its center
    // of mass. Match the native engine box inertia and the Matter proxy shape.
    support.localCenter = {0.0, 0.0, 0.0};
    support.bodyIndex = 4u;
    support.sceneBodyIndex = 4u;
    support.articulated = false;
    support.dynamic = true;
    support.prescribedEndPoseTranslation = false;
    require(source.world.rigidProxies.size() == 2u,
            "sparse free-body fixture must retain one capsule and one BOX");
    return source;
}

MRBodyStateGPU sparseFreeBodyToolSceneState(const ToolPose pose) {
    return numi::cardboard::makeShippingKinematicSceneBody(
        {pose.translation.x, pose.translation.y, pose.translation.z},
        {pose.orientation.x, pose.orientation.y,
         pose.orientation.z, pose.orientation.w});
}

RuntimeStateSnapshot activateSparseFreeBodyMortar(
        const MeshAndWorld& source,
        Runtime& runtime,
        numi::cardboard::ShippingRigidWorldRunner& runner,
        const std::uint32_t controlStep,
        const double authoredGapM,
        double& measuredAcceptedGapM,
        std::string& capturedTrace) {
    require(std::isfinite(authoredGapM) && authoredGapM > 0.0 &&
                authoredGapM <= source.world.contactSlop &&
                std::abs(authoredGapM - source.interfaceGapM) <= 1.0e-12,
            "free-body activation must use the authored positive mortar gap within contact slop");
    const RuntimeStateSnapshot before = runtime.snapshot();
    require(before.available && before.cohesiveFaces.size() == 1u &&
                (before.cohesiveFaces[0].adjacency.w &
                    NM_TOPOLOGY_COHESIVE_DORMANT) != 0u,
            "free-body setup must begin from one accepted-state dormant mixed face");
    require(before.femNodes.size() == source.nodes.size() &&
                source.world.objects.size() == 1u,
            "free-body accepted activation geometry lacks the expected single-object node map");
    double maximumAcceptedNodeGap = 0.0;
    for (std::size_t index = 0u;
         index < source.lowerInterfaceNodes.size(); ++index) {
        const NMFEMNodeStateGPU& lower = before.femNodes.at(
            source.lowerInterfaceNodes[index]);
        const NMFEMNodeStateGPU& upper = before.femNodes.at(
            source.upperInterfaceNodes[index]);
        const double dx = double(upper.positionAndMass.x) -
                          double(lower.positionAndMass.x);
        const double dy = double(upper.positionAndMass.y) -
                          double(lower.positionAndMass.y);
        const double dz = double(upper.positionAndMass.z) -
                          double(lower.positionAndMass.z);
        const double gap = std::sqrt(dx * dx + dy * dy + dz * dz);
        require(std::isfinite(gap) && std::abs(gap - authoredGapM) <= 2.0e-7,
                "accepted FEM node geometry does not match the authored mortar gap");
        maximumAcceptedNodeGap = std::max(maximumAcceptedNodeGap, gap);
    }
    const std::uint64_t diagnosticIndex = diagnosticStateOutput
        ? diagnosticStateOutput->begin(controlStep, before) : 0u;

    NMCohesiveActivationCommandGPU command{};
    command.identity = {0u, 0u, kFaceId, 0u};
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil,
            "native Metal device unavailable for free-body interface activation");
    id<MTLBuffer> commandBuffer = makeBuffer(
        device, std::vector<NMCohesiveActivationCommandGPU>{command});
    std::vector<MRABABodyWrenchGPU> zeroWrenches(5u);
    id<MTLBuffer> zeroWrenchBuffer = makeBuffer(device, zeroWrenches);
    require(commandBuffer != nil && zeroWrenchBuffer != nil &&
                commandBuffer.length >= sizeof(command) &&
                zeroWrenchBuffer.length >=
                    zeroWrenches.size() * sizeof(zeroWrenches[0]),
            "free-body activation command or zero-wrench buffer is unavailable");

    const std::string traceSelector = std::to_string(controlStep);
    numi::cardboard::ShippingRigidWorldRunner::StepResult activationStep;
    {
        ScopedEnvironmentValue trialTrace(
            "NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT", traceSelector.c_str());
        StderrCapture capture;
        activationStep = runner.step(
            std::span<const MRBodyStateGPU>(runner.sceneBodies().data(),
                                             runner.toolSceneBodyCount()),
            nullptr, 0u, (__bridge void*)commandBuffer, 1u,
            (__bridge void*)zeroWrenchBuffer, 5u, 5u,
            controlStep, 0.0f, false);
        capturedTrace = capture.finish();
    }
    if (activationStep.diagnostics.succeeded())
        ++nativeFEMTransactionsExecuted;
    const RuntimeStateSnapshot after = runtime.snapshot();
    require(after.available,
            "free-body activation completion snapshot unavailable: " +
                after.message);
    if (diagnosticStateOutput)
        diagnosticStateOutput->finish(
            diagnosticIndex, &after,
            activationStep.matterAcceptance.accepted
                ? "mortar_activation_accepted"
                : "mortar_activation_rejected");
    require(activationStep.diagnostics.succeeded() &&
                activationStep.matterAcceptance.accepted,
            "accepted-state mortar activation setup rejected: " +
                activationStep.diagnostics.message +
                " trace=" + capturedTrace);
    requireAccepted(after);
    require(after.cohesiveFaces.size() == 1u &&
                (after.cohesiveFaces[0].adjacency.w &
                    NM_TOPOLOGY_COHESIVE_DORMANT) == 0u &&
                (after.cohesiveFaces[0].adjacency.w &
                    NM_TOPOLOGY_PAIRED_COHESIVE) != 0u &&
                (after.cohesiveFaces[0].adjacency.w &
                    NM_TOPOLOGY_COHESIVE_MIXED_MODE) != 0u &&
                after.solverCertificates.size() == 1u &&
                after.solverCertificates[0].validity.w > 0.5f,
            "accepted activation did not retain a valid active mixed-face certificate");
    double maximumLatchedGap = 0.0;
    for (const nm_float4& gap : after.cohesiveFaces[0].activationLocalGap) {
        const double norm = std::sqrt(double(gap.x) * gap.x +
                                      double(gap.y) * gap.y +
                                      double(gap.z) * gap.z);
        require(std::isfinite(norm),
                "accepted activation latched a nonfinite mortar gap");
        maximumLatchedGap = std::max(maximumLatchedGap, norm);
    }
    require(std::abs(maximumLatchedGap - maximumAcceptedNodeGap) <= 2.0e-7 &&
                maximumLatchedGap <= source.world.objects[0]
                    .femCohesiveMortarPatches[0].activationGapToleranceM,
            "accepted mortar activation reference does not match its measured accepted geometry");
    measuredAcceptedGapM = maximumAcceptedNodeGap;
    require(hasMortarParentContact(after, source),
            "accepted mortar setup lost uncovered-parent self-contact history");
    // Activation commands are intentionally submitted in an ordinary
    // accepted transaction. The bounded full-contact private Armijo slice
    // rejects request-scoped activation features; callers restore this exact
    // completed snapshot into the dense/sparse measurement runtimes below.
    return after;
}

SparseFreeBodyRun runSparseFreeBodySequence(
        const MaterialProgram& bulk,
        const bool sparse,
        const std::uint32_t sparseCapacity,
        const double supportGapM,
        const double appliedForceZNewtons,
        const std::uint32_t controlStepBase) {
    require(!regionalContactPreconditioner,
            "bounded free-BOX Armijo does not admit a regional FEM preconditioner");
    const std::string capacityText = std::to_string(sparseCapacity);
    ScopedEnvironmentValue fullContactSelector(
        "NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT", "all");
    ScopedEnvironmentValue sparseCapacityValue(
        "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_CAPACITY",
        sparse ? capacityText.c_str() : nullptr);
    ScopedEnvironmentValue sparseParityValue(
        "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_PARITY",
        sparse && sparseCapacity > 1u ? "1" : nullptr);
    ScopedEnvironmentValue clearResidualRoot(
        "NM_FEM_RESIDUAL_ARMIJO_ROOT", nullptr);
    ScopedEnvironmentValue clearCapsuleRoot(
        "NM_FEM_CAPSULE_RESIDUAL_ARMIJO_ROOT", nullptr);
    ScopedEnvironmentValue clearCapsuleBoxRoot(
        "NM_FEM_CAPSULE_BOX_RESIDUAL_ARMIJO_ROOT", nullptr);
    ScopedEnvironmentValue clearCohesiveRoot(
        "NM_FEM_COHESIVE_RESIDUAL_ARMIJO_ROOT", nullptr);

    MeshAndWorld source = makeSparseFreeBodyWorld(bulk, supportGapM);
    const CompiledWorld matterWorld = compile(source.world);
    require(matterWorld.dispatch.environmentCount == 1u &&
                matterWorld.dispatch.rigidGeneralizedCapacity == 6u &&
                matterWorld.dispatch.rigidQCapacity == 0u &&
                matterWorld.dispatch.rigidProxyCount == 2u &&
                matterWorld.dispatch.contactPairCount > 1u &&
                matterWorld.contact.rigidProxies.at(0u).shapeKind ==
                    NM_RIGID_CAPSULE &&
                matterWorld.contact.rigidProxies.at(0u).flags ==
                    NM_RIGID_PRESCRIBED_TRANSLATION &&
                matterWorld.contact.rigidProxies.at(1u).shapeKind == NM_RIGID_BOX &&
                matterWorld.contact.rigidProxies.at(1u).flags == NM_RIGID_DYNAMIC &&
                matterWorld.contact.rigidProxies.at(1u).bodyIndex == 4u &&
                matterWorld.contact.rigidProxies.at(1u).sceneBodyIndex == 4u,
            "free-body fixture escaped the admitted capsule plus one scene BOX layout");
    if (sparse) {
        const auto admission = detail::admitSparsePrivateContact(
            matterWorld.dispatch.environmentCount,
            matterWorld.dispatch.contactPairCount,
            sparseCapacity, sparseCapacity > 1u, true, true);
        require(admission.admitted(),
                "sparse free-body fixture's private arena is not admitted: " +
                    std::string(detail::sparsePrivateContactAdmissionErrorText(
                        admission.error)));
    }

    const auto& supportProxy = source.world.rigidProxies.at(kAnvilBody);
    numi::cardboard::ShippingRigidBodySpec bodySpec;
    bodySpec.massKg = 1.0;
    bodySpec.halfExtentM = {supportProxy.localExtent[0],
                            supportProxy.localExtent[1],
                            supportProxy.localExtent[2]};
    bodySpec.initialCenterM = {
        0.0, 0.0, -supportProxy.localExtent[2] - supportGapM};
    bodySpec.gravityMPerS2 = {0.0, 0.0, 0.0};
    bodySpec.timestepSeconds = source.world.frameTimestep;

    std::array<MRBodyStateGPU, 4> toolBodies{
        sparseFreeBodyToolSceneState(source.tooling.punchInitialPose),
        numi::cardboard::makeShippingKinematicSceneBody({1.0, 1.0, 1.0}),
        numi::cardboard::makeShippingKinematicSceneBody({2.0, 2.0, 2.0}),
        numi::cardboard::makeShippingKinematicSceneBody({3.0, 3.0, 3.0}),
    };
    auto rigidBundle = numi::cardboard::makeShippingRigidWorld(
        toolBodies, bodySpec);
    const std::string geometryMismatch =
        numi::cardboard::shippingRigidBranchGeometryMismatchReason(
            matterWorld.contact.rigidProxies, rigidBundle, bodySpec);
    require(geometryMismatch.empty(),
            "free-body fixture scene/proxy identity mismatch: " + geometryMismatch);

    RuntimeConfiguration configuration;
    configuration.metallib = NUMI_MATTER_METALLIB;
    configuration.environmentCount = matterWorld.dispatch.environmentCount;
    configuration.captureDiagnostics = true;
    configuration.adaptiveTransfer = false;

    RuntimeStateSnapshot initialSnapshot;
    RuntimeStateSnapshot activatedSnapshot;
    std::vector<MRBodyStateGPU> activatedSceneBodies;
    double acceptedActivationGapM = 0.0;
    std::string activationTrace;
    float initialBodyZ = 0.0f;
    {
        // The activation event is a separate, ordinary accepted transaction.
        // It latches the true accepted FEM geometry without asking the
        // full-contact private Armijo transaction to consume a request-scoped
        // event that is outside that solver slice.
        ScopedEnvironmentValue clearFullContactSelector(
            "NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT", nullptr);
        ScopedEnvironmentValue clearSparseCapacity(
            "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_CAPACITY", nullptr);
        ScopedEnvironmentValue clearSparseParity(
            "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_PARITY", nullptr);
        Runtime activationRuntime;
        const RuntimeDiagnostics activationInitialized =
            activationRuntime.initialize(matterWorld, configuration);
        require(activationInitialized.encoded,
                "free-body activation runtime initialization failed: " +
                    activationInitialized.message);
        numi::cardboard::ShippingRigidWorldRunner activationRunner(
            rigidBundle, activationRuntime,
            static_cast<float>(source.world.frameTimestep), matterWorld);
        initialSnapshot = activationRuntime.snapshot();
        require(initialSnapshot.available,
                "free-body initial Matter snapshot is unavailable: " +
                    initialSnapshot.message);
        initialBodyZ = activationRunner.sceneBodies().at(4u).position.z;
        activatedSnapshot = activateSparseFreeBodyMortar(
            source, activationRuntime, activationRunner,
            controlStepBase - 1u, source.interfaceGapM,
            acceptedActivationGapM, activationTrace);
        activatedSceneBodies = activationRunner.sceneBodies();
        require(activationRunner.q().empty() && activationRunner.v().empty(),
                "free-body activation setup unexpectedly introduced articulated q/v state");
        require(activatedSceneBodies.size() == rigidBundle.initialSceneBodies.size(),
                "accepted activation scene-body checkpoint has the wrong size");
    }

    Runtime runtime;
    const RuntimeDiagnostics initialized = runtime.initialize(matterWorld, configuration);
    require(initialized.encoded,
            "sparse free-body runtime initialization failed: " + initialized.message);
    const RuntimeDiagnostics restored = runtime.restore(activatedSnapshot);
    require(restored.encoded,
            "accepted activation snapshot could not be restored into the full-contact runtime: " +
                restored.message);
    numi::cardboard::ShippingRigidWorldRunner runner(
        std::move(rigidBundle), runtime,
        static_cast<float>(source.world.frameTimestep), matterWorld);
    runner.restoreAcceptedSceneBodies(activatedSceneBodies);
    const RuntimeStateSnapshot restoredActivation = runtime.snapshot();
    requireSnapshotOperatorParity(
        activatedSnapshot, restoredActivation,
        "restored accepted mortar activation");
    requireAcceptedStateUnchanged(
        activatedSnapshot, restoredActivation,
        "restored accepted mortar activation");
    requireFullSnapshotRestoreParity(
        activatedSnapshot, restoredActivation,
        "restored accepted mortar activation");
    if (!sparse) {
        const auto rejectCorruptActivationRestore = [&](
                const char* label,
                const auto& corrupt) {
            RuntimeStateSnapshot malformed = activatedSnapshot;
            require(malformed.cohesiveFaces.size() == 1u,
                    "activation restore corruption fixture has no cohesive face");
            corrupt(malformed.cohesiveFaces[0]);
            const RuntimeDiagnostics denied = runtime.restore(malformed);
            require(!denied.encoded,
                    std::string("activated mortar restore accepted corrupt ") +
                        label);
            const RuntimeStateSnapshot unchanged = runtime.snapshot();
            requireFullSnapshotRestoreParity(
                activatedSnapshot, unchanged,
                "rejected corrupted mortar activation restore");
        };
        rejectCorruptActivationRestore("reference normal", [](auto& face) {
            face.geometry.x += 0.01f;
        });
        rejectCorruptActivationRestore("reference area", [](auto& face) {
            face.geometry.w = -face.geometry.w;
        });
        rejectCorruptActivationRestore("mortar vertex mapping", [](auto& face) {
            face.mortarSideAVertexWeights[0].x = std::nextafter(
                face.mortarSideAVertexWeights[0].x, 0.0f);
        });
        rejectCorruptActivationRestore("cohesive law identity", [](auto& face) {
            ++face.lawIdentity.y;
        });
    }
    require(runner.sceneBodies().size() == activatedSceneBodies.size() &&
                std::memcmp(runner.sceneBodies().data(),
                            activatedSceneBodies.data(),
                            activatedSceneBodies.size() * sizeof(MRBodyStateGPU)) == 0,
            "full-contact runtime did not retain the accepted activation scene-body checkpoint");

    std::vector<MRABABodyWrenchGPU> wrenches(5u);
    wrenches[4].force = {0.0f, 0.0f,
        static_cast<float>(appliedForceZNewtons), 0.0f};
    wrenches[4].torque = {0.0f, 0.0f, 0.0f, 0.0f};
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil, "native Metal device unavailable for free-body wrench");
    id<MTLBuffer> wrenchBuffer = makeBuffer(device, wrenches);
    require(wrenchBuffer != nil &&
                wrenchBuffer.length >= wrenches.size() * sizeof(wrenches[0]),
            "free-body fixture could not allocate the transaction wrench input");

    SparseFreeBodyRun result;
    result.initial = std::move(initialSnapshot);
    result.initialBodyZ = initialBodyZ;
    result.activated = std::move(activatedSnapshot);
    result.activationGapM = acceptedActivationGapM;
    result.activationTrace = std::move(activationTrace);
    const std::uint32_t stepCount = appliedForceZNewtons == 0.0 ? 1u : 3u;
    for (std::uint32_t localStep = 0u; localStep < stepCount; ++localStep) {
        const std::span<const MRBodyStateGPU> kinematicTools(
            runner.sceneBodies().data(), runner.toolSceneBodyCount());
        numi::cardboard::ShippingRigidWorldRunner::StepResult nativeStep;
        std::string trace;
        const std::uint32_t controlStep = controlStepBase + localStep;
        const std::string traceSelector = std::to_string(controlStep);
        RuntimeStateSnapshot diagnosticBefore;
        std::uint64_t diagnosticIndex = 0u;
        bool diagnosticFinished = false;
        if (diagnosticStateOutput) {
            diagnosticBefore = runtime.snapshot();
            require(diagnosticBefore.available,
                    "free-body diagnostic before snapshot is unavailable: " +
                        diagnosticBefore.message);
            diagnosticIndex = diagnosticStateOutput->begin(
                controlStep, diagnosticBefore);
        }
        const auto finishDiagnostic = [&](const RuntimeStateSnapshot* after,
                                          const char* stage) {
            if (diagnosticStateOutput && !diagnosticFinished) {
                diagnosticStateOutput->finish(diagnosticIndex, after, stage);
                diagnosticFinished = true;
            }
        };
        try {
            {
                ScopedEnvironmentValue trialTrace(
                    "NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT",
                    traceSelector.c_str());
                StderrCapture capture;
                nativeStep = runner.step(
                    kinematicTools, nullptr, 0u, nullptr, 0u,
                    (__bridge void*)wrenchBuffer, 5u, 5u,
                    controlStep, 0.0f, false);
                trace = capture.finish();
            }
            retainFixtureTrace(
                ("free-body-control-" + traceSelector).c_str(), trace);
            if (nativeStep.diagnostics.succeeded())
                ++nativeFEMTransactionsExecuted;
            require(nativeStep.diagnostics.succeeded(),
                    "native free-body transaction failed: " +
                        nativeStep.diagnostics.message + " trace=" + trace);
            if (!nativeStep.matterAcceptance.accepted) {
                const RuntimeStateSnapshot rejected = runtime.snapshot();
                finishDiagnostic(&rejected, "native_step_rejected");
                std::ostringstream reason;
                reason << "Matter rejected free-body step status="
                       << nativeStep.matterAcceptance.statusCode
                       << " reason=" << nativeStep.matterAcceptance.reason
                       << " trace=" << trace;
                throw std::runtime_error(reason.str());
            }
            const RuntimeStateSnapshot accepted = runtime.snapshot();
            require(accepted.available,
                    "free-body accepted snapshot is unavailable: " +
                        accepted.message);
            finishDiagnostic(&accepted, "snapshot_after_completion");
            result.traces.push_back(std::move(trace));
            result.accepted.push_back(accepted);
        } catch (...) {
            // A failed engine transaction can still expose its rolled-back
            // Matter state. Retain it so the first failure and rollback can
            // be audited instead of ending at an accepted-state prefix.
            const RuntimeStateSnapshot failed = runtime.snapshot();
            finishDiagnostic(failed.available ? &failed : nullptr,
                             "native_step_exception");
            throw;
        }
    }
    result.finalSceneBodies = runner.sceneBodies();
    result.finalBodyZ = result.finalSceneBodies.at(4u).position.z;
    result.finalBodyVelocityZ =
        result.finalSceneBodies.at(4u).linearVelocityAndInverseMass.z;
    result.terminal = result.accepted.back();
    result.terminalAccepted = true;
    result.terminalStatus = result.terminal.statuses.at(0u).code;
    return result;
}

void runSparseFreeBodyParity(const MaterialProgram& bulk) {
    require(!regionalContactPreconditioner,
            "sparse free-body parity requires the scalar-diagonal preconditioner");
    const MeshAndWorld source = makeSparseFreeBodyWorld(bulk, 9.0e-6);
    const CompiledWorld compiled = compile(source.world);
    const std::uint32_t pairCount = compiled.dispatch.contactPairCount;
    require(pairCount > 2u,
            "free-body sparse parity needs more than two canonical contact pairs");
    const std::string outputRoot = diagnosticStateOutputPath;
    std::unique_ptr<DiagnosticStateOutput> parentDiagnosticOutput =
        std::move(diagnosticStateOutput);
    const auto selectOutput = [&](const char* label) {
        if (outputRoot.empty()) return;
        diagnosticStateOutput.reset();
        const auto directory = std::filesystem::path(outputRoot) / label;
        if (std::filesystem::exists(directory))
            throw std::runtime_error(
                "free-body parity trace subdirectory must be new: " +
                directory.string());
        diagnosticStateOutputPath = directory.string();
        diagnosticStateOutput = std::make_unique<DiagnosticStateOutput>(
            diagnosticStateOutputPath);
    };
    const auto restoreOutput = [&]() {
        diagnosticStateOutput.reset();
        diagnosticStateOutput = std::move(parentDiagnosticOutput);
        diagnosticStateOutputPath = outputRoot;
    };
    SparseFreeBodyRun dense;
    SparseFreeBodyRun sparse;
    try {
        selectOutput("dense");
        dense = runSparseFreeBodySequence(
            bulk, false, 0u, 9.0e-6, 100.0, 1200u);
        selectOutput("sparse");
        sparse = runSparseFreeBodySequence(
            bulk, true, pairCount - 1u, 9.0e-6, 100.0, 1200u);
        restoreOutput();
    } catch (...) {
        restoreOutput();
        throw;
    }
    requireSnapshotOperatorParity(dense.initial, sparse.initial,
                                  "free-body initial");
    requireSnapshotOperatorParity(dense.activated, sparse.activated,
                                  "free-body activation setup");
    require(dense.initial.cohesiveFaces.size() == 1u &&
                sparse.initial.cohesiveFaces.size() == 1u &&
                (dense.initial.cohesiveFaces[0].adjacency.w &
                    NM_TOPOLOGY_COHESIVE_DORMANT) != 0u &&
                (sparse.initial.cohesiveFaces[0].adjacency.w &
                    NM_TOPOLOGY_COHESIVE_DORMANT) != 0u &&
                dense.activated.cohesiveFaces.size() == 1u &&
                sparse.activated.cohesiveFaces.size() == 1u &&
                (dense.activated.cohesiveFaces[0].adjacency.w &
                    NM_TOPOLOGY_COHESIVE_DORMANT) == 0u &&
                (sparse.activated.cohesiveFaces[0].adjacency.w &
                    NM_TOPOLOGY_COHESIVE_DORMANT) == 0u,
            "dense/sparse setup did not retain dormant then accepted-active mortar history");
    require(dense.accepted.size() == sparse.accepted.size() &&
                dense.accepted.size() == 3u,
            "free-body parity sequence did not retain all three accepted steps");
    for (std::size_t index = 0u; index < dense.accepted.size(); ++index) {
        requireSnapshotOperatorParity(dense.accepted[index],
                                      sparse.accepted[index],
                                      "free-body accepted step");
        requireTraceOperatorParity(dense.traces[index], sparse.traces[index],
                                   "free-body accepted step");
        require(dense.traces[index].find(
                    "\"private_contact_layout\":\"dense\"") !=
                    std::string::npos &&
                sparse.traces[index].find(
                    "\"private_contact_layout\":\"sparse\"") !=
                    std::string::npos &&
                sparse.traces[index].find("\"sparse_parity\":true") !=
                    std::string::npos,
                "free-body parity did not exercise both private layouts and internal sample parity");
    }
    const auto hasContact = [](const SparseFreeBodyRun& run) {
        return std::any_of(run.accepted.begin(), run.accepted.end(),
            [](const RuntimeStateSnapshot& snapshot) {
                return !snapshot.statuses.empty() &&
                    snapshot.statuses[0].contactCount > 0u;
            });
    };
    require(hasContact(dense) && hasContact(sparse),
            "free-body sparse parity never exercised a nonzero active contact set");
    require(dense.finalSceneBodies.size() == sparse.finalSceneBodies.size() &&
                std::memcmp(dense.finalSceneBodies.data(),
                            sparse.finalSceneBodies.data(),
                            dense.finalSceneBodies.size() *
                                sizeof(MRBodyStateGPU)) == 0,
            "dense and sparse MetalWorld accepted scene bodies differ");
    require(std::abs(dense.finalBodyZ - dense.initialBodyZ) > 1.0e-8f &&
                std::abs(sparse.finalBodyZ - sparse.initialBodyZ) > 1.0e-8f &&
                std::abs(dense.finalBodyVelocityZ) > 0.0f &&
                std::abs(sparse.finalBodyVelocityZ) > 0.0f,
            "accepted native contact/wrench transactions did not translate the free body");
    std::cout << std::setprecision(9)
              << "sparse_private_free_body_parity=pass"
              << " body_dof=6 nonarticulated_box=true"
              << " accepted_steps=" << dense.accepted.size()
              << " separate_accepted_mortar_activation_setup=true"
              << " authored_mortar_gap_m=" << kFreeBodyMortarGap
              << " measured_accepted_mortar_gap_m=" << dense.activationGapM
              << " activation_geometry=accepted_contact_range_gap_no_overlap_claim"
              << " initial_z_m=" << dense.initialBodyZ
              << " final_z_m=" << dense.finalBodyZ
              << " final_vz_m_per_s=" << dense.finalBodyVelocityZ
              << " canonical_pairs=" << pairCount
              << " sparse_capacity=" << pairCount - 1u
              << " external_wrench_native=true dense_sparse_scene_and_fem_parity=true"
              << " physical_validation=false synthetic_fixture=true\n";
}

void runSparseFreeBodyOverflow(const MaterialProgram& bulk) {
    ScopedEnvironmentValue fullContactSelector(
        "NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT", "all");
    ScopedEnvironmentValue sparseCapacityValue(
        "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_CAPACITY", "1");
    ScopedEnvironmentValue clearParity(
        "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_PARITY", nullptr);
    ScopedEnvironmentValue clearResidualRoot(
        "NM_FEM_RESIDUAL_ARMIJO_ROOT", nullptr);
    ScopedEnvironmentValue clearCapsuleRoot(
        "NM_FEM_CAPSULE_RESIDUAL_ARMIJO_ROOT", nullptr);
    ScopedEnvironmentValue clearCapsuleBoxRoot(
        "NM_FEM_CAPSULE_BOX_RESIDUAL_ARMIJO_ROOT", nullptr);
    ScopedEnvironmentValue clearCohesiveRoot(
        "NM_FEM_COHESIVE_RESIDUAL_ARMIJO_ROOT", nullptr);

    MeshAndWorld source = makeSparseFreeBodyWorld(bulk, 5.0e-6);
    const CompiledWorld matterWorld = compile(source.world);
    const auto& supportProxy = source.world.rigidProxies.at(kAnvilBody);
    numi::cardboard::ShippingRigidBodySpec bodySpec;
    bodySpec.massKg = 1.0;
    bodySpec.halfExtentM = {supportProxy.localExtent[0],
                            supportProxy.localExtent[1],
                            supportProxy.localExtent[2]};
    bodySpec.initialCenterM = {
        0.0, 0.0, -supportProxy.localExtent[2] - 5.0e-6};
    bodySpec.gravityMPerS2 = {0.0, 0.0, 0.0};
    bodySpec.timestepSeconds = source.world.frameTimestep;
    std::array<MRBodyStateGPU, 4> toolBodies{
        sparseFreeBodyToolSceneState(source.tooling.punchInitialPose),
        numi::cardboard::makeShippingKinematicSceneBody({1.0, 1.0, 1.0}),
        numi::cardboard::makeShippingKinematicSceneBody({2.0, 2.0, 2.0}),
        numi::cardboard::makeShippingKinematicSceneBody({3.0, 3.0, 3.0}),
    };
    auto rigidBundle = numi::cardboard::makeShippingRigidWorld(
        toolBodies, bodySpec);
    const std::string geometryMismatch =
        numi::cardboard::shippingRigidBranchGeometryMismatchReason(
            matterWorld.contact.rigidProxies, rigidBundle, bodySpec);
    require(geometryMismatch.empty(),
            "free-body overflow scene/proxy identity mismatch: " + geometryMismatch);
    RuntimeConfiguration configuration;
    configuration.metallib = NUMI_MATTER_METALLIB;
    configuration.environmentCount = matterWorld.dispatch.environmentCount;
    configuration.captureDiagnostics = true;
    configuration.adaptiveTransfer = false;
    RuntimeStateSnapshot activated;
    std::vector<MRBodyStateGPU> activatedSceneBodies;
    double acceptedActivationGapM = 0.0;
    std::string activationTrace;
    {
        // The overflow transaction opts into the sparse private residual
        // path, but accepted mortar activation is a separate ordinary
        // transaction. Initialize its Runtime only after clearing the sparse
        // selectors, then restore that accepted snapshot into the sparse
        // Runtime below. Runtime selectors are captured at initialization.
        ScopedEnvironmentValue clearFullContactSelector(
            "NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT", nullptr);
        ScopedEnvironmentValue clearSparseCapacity(
            "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_CAPACITY", nullptr);
        ScopedEnvironmentValue clearSparseParity(
            "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_PARITY", nullptr);
        Runtime activationRuntime;
        const RuntimeDiagnostics activationInitialized =
            activationRuntime.initialize(matterWorld, configuration);
        require(activationInitialized.encoded,
                "sparse free-body overflow activation initialization failed: " +
                    activationInitialized.message);
        numi::cardboard::ShippingRigidWorldRunner activationRunner(
            rigidBundle, activationRuntime,
            static_cast<float>(source.world.frameTimestep), matterWorld);
        const RuntimeStateSnapshot initial = activationRuntime.snapshot();
        require(initial.available,
                "sparse free-body overflow activation initial snapshot is unavailable: " +
                    initial.message);
        activated = activateSparseFreeBodyMortar(
            source, activationRuntime, activationRunner, 1299u,
            source.interfaceGapM, acceptedActivationGapM, activationTrace);
        activatedSceneBodies = activationRunner.sceneBodies();
        require(activationRunner.q().empty() && activationRunner.v().empty() &&
                    activatedSceneBodies.size() ==
                        rigidBundle.initialSceneBodies.size(),
                "sparse free-body overflow activation checkpoint has invalid rigid state");
    }
    Runtime runtime;
    const RuntimeDiagnostics initialized = runtime.initialize(matterWorld, configuration);
    require(initialized.encoded,
            "sparse free-body overflow initialization failed: " + initialized.message);
    const RuntimeDiagnostics restored = runtime.restore(activated);
    require(restored.encoded,
            "sparse free-body overflow could not restore the accepted activation snapshot: " +
                restored.message);
    numi::cardboard::ShippingRigidWorldRunner runner(
        std::move(rigidBundle), runtime,
        static_cast<float>(source.world.frameTimestep), matterWorld);
    runner.restoreAcceptedSceneBodies(activatedSceneBodies);
    const RuntimeStateSnapshot restoredActivation = runtime.snapshot();
    requireSnapshotOperatorParity(
        activated, restoredActivation,
        "restored overflow mortar activation");
    requireAcceptedStateUnchanged(
        activated, restoredActivation,
        "restored overflow mortar activation");
    requireFullSnapshotRestoreParity(
        activated, restoredActivation,
        "restored overflow mortar activation");
    require(runner.sceneBodies().size() == activatedSceneBodies.size() &&
                std::memcmp(runner.sceneBodies().data(),
                            activatedSceneBodies.data(),
                            activatedSceneBodies.size() *
                                sizeof(MRBodyStateGPU)) == 0,
            "sparse overflow runtime did not retain the accepted activation scene-body checkpoint");
    std::vector<MRABABodyWrenchGPU> wrenches(5u);
    wrenches[4].force = {0.0f, 0.0f, 100.0f, 0.0f};
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil, "native Metal device unavailable for overflow wrench");
    id<MTLBuffer> wrenchBuffer = makeBuffer(device, wrenches);
    const RuntimeStateSnapshot before = restoredActivation;
    require(before.available,
            "free-body overflow initial snapshot is unavailable");
    const std::uint64_t diagnosticIndex = diagnosticStateOutput
        ? diagnosticStateOutput->begin(1300u, activated) : 0u;
    const std::vector<MRBodyStateGPU> beforeScene = runner.sceneBodies();
    numi::cardboard::ShippingRigidWorldRunner::StepResult nativeStep;
    std::string trace;
    {
        ScopedEnvironmentValue trialTrace(
            "NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT", "1300");
        StderrCapture capture;
        nativeStep = runner.step(
            std::span<const MRBodyStateGPU>(runner.sceneBodies().data(), 4u),
            nullptr, 0u, nullptr, 0u, (__bridge void*)wrenchBuffer,
            5u, 5u, 1300u, 0.0f, false);
        trace = capture.finish();
    }
    retainFixtureTrace("free-body-overflow-control-1300", trace);
    if (nativeStep.diagnostics.succeeded())
        ++nativeFEMTransactionsExecuted;
    const RuntimeStateSnapshot after = runtime.snapshot();
    require(after.available,
            "free-body overflow rollback snapshot is unavailable");
    if (diagnosticStateOutput)
        diagnosticStateOutput->finish(
            diagnosticIndex, &after, "native_step_rejected_or_accepted");
    const bool overflowTerminal = after.statuses.size() == 1u &&
        after.statuses[0].code == NM_STATUS_CAPACITY_OVERFLOW;
    const bool overflowTrial = trace.find(
        "\"trial_status\":{\"code\":2") != std::string::npos;
    require((overflowTerminal || overflowTrial) &&
                !nativeStep.matterAcceptance.accepted,
            "one-sample sparse free-body trial did not fail closed on active candidate overflow; trace=" +
                trace + " engine_diagnostic=" + nativeStep.diagnostics.message);
    requireAcceptedStateUnchanged(activated, after,
                                  "sparse free-body candidate overflow");
    require(runner.sceneBodies().size() == beforeScene.size() &&
                std::memcmp(runner.sceneBodies().data(), beforeScene.data(),
                            beforeScene.size() * sizeof(MRBodyStateGPU)) == 0,
            "sparse free-body candidate overflow changed the accepted engine scene");
    require(after.solverCertificates.size() == 1u &&
                after.solverCertificates[0].validity.w <= 0.5f,
            "sparse free-body candidate overflow retained a valid certificate");
    std::cout << "sparse_private_free_body_overflow_rollback=pass"
              << " capacity=1 terminal_status="
              << (after.statuses.empty() ? NM_INVALID_INDEX :
                  after.statuses[0].code)
              << " candidate_overflow_trial="
              << (overflowTrial ? "true" : "false")
              << " accepted_mortar_activation_setup=true"
              << " setup_control_step=1299 authored_gap_m="
              << source.interfaceGapM
              << " measured_accepted_gap_m=" << acceptedActivationGapM
              << " gap_within_contact_slop=true overlap_claim=false"
              << " gap_within_contact_slop=true overlap_claim=false"
              << " fem_material_rigid_and_engine_scene_unchanged=true"
              << " physical_validation=false synthetic_fixture=true\n";
}
#endif

void runSelfSurfaceScratchAliasParity(const MaterialProgram& bulk) {
    const std::string outputRoot = diagnosticStateOutputPath;
    std::unique_ptr<DiagnosticStateOutput> parentDiagnosticOutput =
        std::move(diagnosticStateOutput);
    const auto selectOutput = [&](const char* label) {
        if (outputRoot.empty()) return;
        diagnosticStateOutput.reset();
        const auto directory = std::filesystem::path(outputRoot) / label;
        if (std::filesystem::exists(directory))
            throw std::runtime_error(
                "surface scratch parity subdirectory must be new: " +
                directory.string());
        diagnosticStateOutputPath = directory.string();
        diagnosticStateOutput = std::make_unique<DiagnosticStateOutput>(
            diagnosticStateOutputPath);
    };
    const auto restoreOutput = [&]() {
        diagnosticStateOutput.reset();
        diagnosticStateOutput = std::move(parentDiagnosticOutput);
        diagnosticStateOutputPath = outputRoot;
    };

    AcceptedRollbackSequence baseline;
    AcceptedRollbackSequence aliased;
    try {
        selectOutput("baseline");
        baseline = runAcceptedAndRollback(bulk, false);
        selectOutput("aliased");
        aliased = runAcceptedAndRollback(bulk, true);
        restoreOutput();
    } catch (...) {
        restoreOutput();
        throw;
    }
    requireSequenceOperatorParity(baseline, aliased, false);
    std::cout << "self_surface_scratch_alias_parity=pass"
              << " workspace=production_surface_geometry_scratch"
              << " aliasing_opt_in=true geometry_order_residual_alpha=true"
              << " accepted_and_rollback_histories_bit_identical=true"
              << " physical_validation=false synthetic_law=true\n";
}

void requireExactCohesiveAliasTraceParity(const std::string& baseline,
                                          const std::string& aliased,
                                          const char* stage) {
    const std::string privateMarker =
        "\"cohesive_scratch_layout\":\"private\"";
    const std::string aliasMarker =
        "\"cohesive_scratch_layout\":\"production_candidate_force_alias\"";
    require(baseline.find(privateMarker) != std::string::npos,
            std::string("baseline ") + stage +
                " trace did not identify private cohesive scratch");
    require(aliased.find(aliasMarker) != std::string::npos,
            std::string("aliased ") + stage +
                " trace did not identify production cohesive scratch reuse");
    std::string normalized = aliased;
    std::size_t replaced = 0u;
    for (std::size_t offset = 0u;
         (offset = normalized.find(aliasMarker, offset)) != std::string::npos;
         offset += privateMarker.size()) {
        normalized.replace(offset, aliasMarker.size(), privateMarker);
        ++replaced;
    }
    require(replaced != 0u && normalized == baseline,
            std::string("cohesive alias ") + stage +
                " full trace differs beyond the explicit scratch-layout field");
}

void runCohesiveScratchAliasParity(const MaterialProgram& bulk) {
    require(!diagnosticStateOutputPath.empty(),
            "cohesive scratch alias parity requires retained full traces and state snapshots");
    const std::string outputRoot = diagnosticStateOutputPath;
    std::unique_ptr<DiagnosticStateOutput> parentDiagnosticOutput =
        std::move(diagnosticStateOutput);
    const auto selectOutput = [&](const char* label) {
        diagnosticStateOutput.reset();
        const auto directory = std::filesystem::path(outputRoot) / label;
        if (std::filesystem::exists(directory))
            throw std::runtime_error(
                "cohesive scratch parity subdirectory must be new: " +
                directory.string());
        diagnosticStateOutputPath = directory.string();
        diagnosticStateOutput = std::make_unique<DiagnosticStateOutput>(
            diagnosticStateOutputPath);
    };
    const auto restoreOutput = [&]() {
        diagnosticStateOutput.reset();
        diagnosticStateOutput = std::move(parentDiagnosticOutput);
        diagnosticStateOutputPath = outputRoot;
    };

    AcceptedRollbackSequence baseline;
    AcceptedRollbackSequence aliased;
    try {
        selectOutput("baseline");
        baseline = runAcceptedAndRollback(bulk);
        selectOutput("cohesive-aliased");
        aliased = runAcceptedAndRollback(bulk, false, true);
        restoreOutput();
    } catch (...) {
        restoreOutput();
        throw;
    }
    requireSequenceOperatorParity(baseline, aliased, false);
    requireExactCohesiveAliasTraceParity(
        baseline.activation.trace, aliased.activation.trace, "activation");
    requireExactCohesiveAliasTraceParity(
        baseline.loaded.trace, aliased.loaded.trace, "loaded");
    requireExactCohesiveAliasTraceParity(
        baseline.rejected.trace, aliased.rejected.trace, "rejected");
    std::cout << "cohesive_scratch_alias_parity=pass"
              << " workspace=production_candidate_and_force_buffers"
              << " opt_in=true"
              << " activation_loaded_rejection_full_trace_byte_identical=true"
              << " selected_alpha_certificate_and_rollback_bit_identical=true"
              << " physical_validation=false synthetic_law=true\n";
}

void requireAcceptedStateUnchanged(const RuntimeStateSnapshot& before,
                                   const RuntimeStateSnapshot& after,
                                   const char* reason) {
    require(after.controlStep == before.controlStep &&
                after.physicsSubstep == before.physicsSubstep,
            std::string(reason) + " advanced the accepted cursor");
    requireVectorBitIdentical(before.femNodes, after.femNodes,
                              "FEM nodes", reason);
    requireVectorBitIdentical(before.femMaterialState, after.femMaterialState,
                              "bulk history", reason);
    requireVectorBitIdentical(before.cohesiveFaces, after.cohesiveFaces,
                              "cohesive history", reason);
    requireVectorBitIdentical(before.contactHistories, after.contactHistories,
                              "rigid-contact history", reason);
    requireVectorBitIdentical(before.deformableContactHistories,
                              after.deformableContactHistories,
                              "deformable-contact history", reason);
    requireVectorBitIdentical(before.rigidStates, after.rigidStates,
                              "rigid states", reason);
}

void requireFullSnapshotRestoreParity(const RuntimeStateSnapshot& before,
                                      const RuntimeStateSnapshot& after,
                                      const char* reason) {
    require(before.available && after.available,
            std::string(reason) + " snapshot is unavailable");
    require(before.sourcePhysicsFingerprint == after.sourcePhysicsFingerprint &&
                before.deviceProgramFingerprint == after.deviceProgramFingerprint,
            std::string(reason) + " source/device identity changed");
    require(before.controlStep == after.controlStep &&
                before.physicsSubstep == after.physicsSubstep &&
                before.identificationGeneration == after.identificationGeneration &&
                before.identificationCheckpoint == after.identificationCheckpoint &&
                before.identificationAdvanced == after.identificationAdvanced &&
                before.sutureProxyBindingRevision == after.sutureProxyBindingRevision &&
                before.coupledTimestepMultiplier == after.coupledTimestepMultiplier &&
                before.coupledTimestepDivisor == after.coupledTimestepDivisor &&
                before.fgmresIterationBudgetOverride ==
                    after.fgmresIterationBudgetOverride &&
                before.newtonIterationBudgetOverride ==
                    after.newtonIterationBudgetOverride &&
                before.allocationGeneration == after.allocationGeneration &&
                before.learnedWeightRevision == after.learnedWeightRevision &&
                before.materialStateStride == after.materialStateStride,
            std::string(reason) + " snapshot metadata changed");
#define REQUIRE_RESTORED_VECTOR(field, label) \
    requireVectorBitIdentical(before.field, after.field, label, reason)
    REQUIRE_RESTORED_VECTOR(sutureProxyEdges, "suture proxy bindings");
    REQUIRE_RESTORED_VECTOR(particles, "particles");
    REQUIRE_RESTORED_VECTOR(femNodes, "FEM nodes");
    REQUIRE_RESTORED_VECTOR(femFields, "FEM fields");
    REQUIRE_RESTORED_VECTOR(vascularState, "vascular state");
    REQUIRE_RESTORED_VECTOR(vascularClock, "vascular clock");
    REQUIRE_RESTORED_VECTOR(femTopologyNodes, "FEM topology nodes");
    REQUIRE_RESTORED_VECTOR(femTopologyTetrahedra, "FEM topology tetrahedra");
    REQUIRE_RESTORED_VECTOR(cohesiveFaces, "cohesive faces");
    REQUIRE_RESTORED_VECTOR(punctureChannels, "puncture channels");
    REQUIRE_RESTORED_VECTOR(topologyStates, "topology state");
    REQUIRE_RESTORED_VECTOR(statuses, "completion status");
    REQUIRE_RESTORED_VECTOR(solverCertificates, "solver certificates");
    REQUIRE_RESTORED_VECTOR(mpmActiveNodeIndices, "MPM active nodes");
    REQUIRE_RESTORED_VECTOR(mpmNodeToActive, "MPM node map");
    REQUIRE_RESTORED_VECTOR(mpmActiveNodeCounts, "MPM active counts");
    REQUIRE_RESTORED_VECTOR(rigidGeneralizedCandidate, "rigid generalized state");
    REQUIRE_RESTORED_VECTOR(learnedWeights, "learned weights");
    REQUIRE_RESTORED_VECTOR(adaptive, "adaptive state");
    REQUIRE_RESTORED_VECTOR(schedulers, "scheduler state");
    REQUIRE_RESTORED_VECTOR(reactions, "rigid reactions");
    REQUIRE_RESTORED_VECTOR(rigidStates, "rigid states");
    REQUIRE_RESTORED_VECTOR(contactSamples, "contact samples");
    REQUIRE_RESTORED_VECTOR(deformableContactFailures, "deformable contact failures");
    REQUIRE_RESTORED_VECTOR(contactHistories, "rigid contact histories");
    REQUIRE_RESTORED_VECTOR(humanSupportHistories, "human support histories");
    REQUIRE_RESTORED_VECTOR(humanSupportConsequences, "human support consequences");
    REQUIRE_RESTORED_VECTOR(deformableContactHistories,
                            "deformable contact histories");
    REQUIRE_RESTORED_VECTOR(particleMaterialState, "particle material state");
    REQUIRE_RESTORED_VECTOR(femMaterialState, "FEM material state");
    REQUIRE_RESTORED_VECTOR(identification, "identification posterior");
    REQUIRE_RESTORED_VECTOR(environmentParameters, "environment parameters");
    REQUIRE_RESTORED_VECTOR(femRegionalGalerkinLinearizationNodes,
                            "regional linearization nodes");
    REQUIRE_RESTORED_VECTOR(femRegionalGalerkinLinearizationMaterialState,
                            "regional linearization material history");
    REQUIRE_RESTORED_VECTOR(femRegionalGalerkinLinearizationSchedulerExponent,
                            "regional linearization scheduler");
    REQUIRE_RESTORED_VECTOR(femRegionalGalerkinLinearizationLocalFHigh,
                            "regional linearization deformation high words");
    REQUIRE_RESTORED_VECTOR(femRegionalGalerkinLinearizationLocalFLow,
                            "regional linearization deformation low words");
    REQUIRE_RESTORED_VECTOR(femRegionalGalerkinBasisP, "regional basis P");
    REQUIRE_RESTORED_VECTOR(femRegionalGalerkinEffectiveBasisP,
                            "regional effective basis P");
    REQUIRE_RESTORED_VECTOR(femRegionalGalerkinOperatorAP, "regional operator AP");
    REQUIRE_RESTORED_VECTOR(femRegionalGalerkinPivotedRAP, "regional RAP");
    REQUIRE_RESTORED_VECTOR(femRegionalGalerkinPivots, "regional pivots");
    REQUIRE_RESTORED_VECTOR(femRegionalGalerkinFactorValid,
                            "regional factor validity");
#undef REQUIRE_RESTORED_VECTOR
    require(before.femRegionalGalerkinDiagnosticsAvailable ==
                after.femRegionalGalerkinDiagnosticsAvailable &&
                before.femRegionalGalerkinModeCount ==
                    after.femRegionalGalerkinModeCount &&
                before.femRegionalGalerkinDiagnosticsMessage ==
                    after.femRegionalGalerkinDiagnosticsMessage,
            std::string(reason) + " regional diagnostic metadata changed");
}

void runSparsePrivateOverflow(const MaterialProgram& bulk,
                              const std::uint32_t capacity) {
    require(capacity == 1u,
            "sparse overflow regression uses its authored one-sample capacity");
    const std::string capacityText = std::to_string(capacity);
    ScopedEnvironmentValue sparseCapacity(
        "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_CAPACITY", capacityText.c_str());
    ScopedEnvironmentValue clearParity(
        "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_PARITY", nullptr);
    MeshAndWorld source = makeWorld(bulk, false, true, 5.0e-6);
    verifyPartialMortarSource(source);
    Fixture run(source);
    const RuntimeStateSnapshot before = run.snapshot();
    std::string trace;
    StepResult rejected;
    {
        StderrCapture capture;
        setenv("NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT", "1", 1);
        try {
            rejected = run.step(1u, false, true);
        } catch (...) {
            trace = capture.finish();
            retainFixtureTrace("sparse-overflow", trace);
            throw;
        }
        trace = capture.finish();
    }
    rejected.trace = trace;
    retainFixtureTrace("sparse-overflow", trace);
    const std::uint32_t observedTerminalStatus =
        rejected.snapshot.statuses.size() == 1u
            ? rejected.snapshot.statuses[0].code : NM_INVALID_INDEX;
    const bool observedTrialCapacityOverflow = trace.find(
        "\"trial_status\":{\"code\":2") != std::string::npos;
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
              << "sparse_private_overflow_observation= status="
              << observedTerminalStatus
              << " trial_capacity_overflow="
              << (observedTrialCapacityOverflow ? "true" : "false");
    if (rejected.snapshot.solverCertificates.size() == 1u) {
        const NMSolverCertificateGPU& observed =
            rejected.snapshot.solverCertificates[0];
        std::cout << " certificate_nonlinear=[" << observed.nonlinear.x << ','
                  << observed.nonlinear.y << ',' << observed.nonlinear.z << ','
                  << observed.nonlinear.w << "] certificate_contact=["
                  << observed.contact.x << ',' << observed.contact.y << ','
                  << observed.contact.z << ',' << observed.contact.w
                  << "] certificate_transport=[" << observed.transport.x << ','
                  << observed.transport.y << ',' << observed.transport.z << ','
                  << observed.transport.w << "] certificate_validity=["
                  << observed.validity.x << ',' << observed.validity.y << ','
                  << observed.validity.z << ',' << observed.validity.w << ']';
    } else {
        std::cout << " certificate=missing";
    }
    std::cout << " raw_trace_retained="
              << (!diagnosticStateOutputPath.empty() ? "true" : "false")
              << " physical_validation=false\n";
    require(trace.find("\"private_contact_layout\":\"sparse\"") !=
                std::string::npos,
            "sparse overflow case did not select compact private samples");
    require(rejected.snapshot.statuses.size() == 1u,
            "sparse overflow case returned no environment status");
    const std::uint32_t terminalStatus = rejected.snapshot.statuses[0].code;
    const bool terminalCapacityOverflow =
        terminalStatus == NM_STATUS_CAPACITY_OVERFLOW;
    const bool trialCapacityOverflow = trace.find(
        "\"trial_status\":{\"code\":2") != std::string::npos;
    if (!terminalCapacityOverflow && !trialCapacityOverflow) {
        const NMMatterStatusGPU& status = rejected.snapshot.statuses[0];
        std::ostringstream detail;
        detail << "undersized sparse private contact arena produced no capacity-overflow status"
               << " terminal_code=" << status.code
               << " object_index=" << status.objectIndex
               << " failing_index=" << status.failingIndex
               << " diagnostics=(" << status.diagnostics.x << ','
               << status.diagnostics.y << ',' << status.diagnostics.z << ','
               << status.diagnostics.w << ") trace=" << trace;
        throw std::runtime_error(detail.str());
    }
    require(rejected.snapshot.solverCertificates.size() == 1u &&
                rejected.snapshot.solverCertificates[0].validity.w <= 0.5f,
            "sparse candidate overflow retained a valid solver certificate");
    requireAcceptedStateUnchanged(before, rejected.snapshot,
                                  "sparse candidate overflow");
    std::cout << "sparse_private_overflow_rollback=pass"
              << " capacity=" << capacity
              << " terminal_status=" << terminalStatus
              << " candidate_overflow_trial="
              << (trialCapacityOverflow ? "true" : "false")
              << " accepted_state_and_histories_unchanged=true"
              << " certificate_cleared=true physical_validation=false\n";
}

void requireSparseInitializationRejected(const CompiledWorld& world,
                                         const char* capacity,
                                         const char* parity,
                                         const char* selector,
                                         const char* expectedReason) {
    ScopedEnvironmentValue capacityValue(
        "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_CAPACITY", capacity);
    ScopedEnvironmentValue parityValue(
        "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_PARITY", parity);
    ScopedEnvironmentValue selectorValue(
        "NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT", selector);
    Runtime runtime;
    RuntimeConfiguration configuration;
    configuration.metallib = NUMI_MATTER_METALLIB;
    configuration.environmentCount = world.dispatch.environmentCount;
    configuration.captureDiagnostics = false;
    const RuntimeDiagnostics initialized = runtime.initialize(world, configuration);
    require(!initialized.encoded &&
                initialized.message.find(expectedReason) != std::string::npos,
            std::string("sparse runtime admission did not reject as expected: ") +
                expectedReason + "; got: " + initialized.message);
}

void verifySparsePrivateAdmission(const MaterialProgram& bulk,
                                  const CompiledWorld& staticWorld) {
    requireSparseInitializationRejected(staticWorld, "0", nullptr, "all",
        "must be positive");
    requireSparseInitializationRejected(staticWorld, "4x", nullptr, "all",
        "only decimal digits");
    requireSparseInitializationRejected(staticWorld, nullptr, "1", "all",
        "explicit candidate capacity");
    requireSparseInitializationRejected(staticWorld, "1", nullptr, nullptr,
        "requires the full-contact Armijo selector");

    MeshAndWorld freeBodySource = makeWorld(bulk, false, true, 5.0e-6);
    // This case verifies runtime admission of the one supported sparse free
    // rigid scope. Keep it independent of the optional regional contact
    // preconditioner, whose authored scope excludes free generalized bodies.
    freeBodySource.world.mixedSolver.femPreconditioner =
        FEMPreconditionerMode::scalarDiagonal;
    // makeWorld() authors all-zero regional labels when the fixture-wide
    // selector is enabled. This free-body case deliberately uses the
    // scalar preconditioner, so remove those labels as well; otherwise source
    // compilation rejects the inconsistent selector before sparse-runtime
    // admission gets a chance to exercise the intended free-DOF guard.
    freeBodySource.world.objects.at(0u).femPreconditionerRegionIds.clear();
    freeBodySource.world.rigidProxies.at(kAnvilBody).dynamic = true;
    freeBodySource.world.rigidProxies.at(kAnvilBody).sceneBodyIndex = kAnvilBody;
    const CompiledWorld freeBodyWorld = compile(freeBodySource.world);
    require(freeBodyWorld.dispatch.rigidGeneralizedCapacity == 6u,
            "sparse free-rigid admission fixture did not author one free body");
    const std::string capacity = std::to_string(
        std::max(1u, freeBodyWorld.dispatch.contactPairCount - 1u));
    ScopedEnvironmentValue sparseCapacity(
        "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_CAPACITY", capacity.c_str());
    ScopedEnvironmentValue clearParity(
        "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_PARITY", nullptr);
    ScopedEnvironmentValue fullContactSelector(
        "NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT", "all");
    Runtime freeBodyRuntime;
    RuntimeConfiguration freeBodyConfiguration;
    freeBodyConfiguration.metallib = NUMI_MATTER_METALLIB;
    freeBodyConfiguration.environmentCount = freeBodyWorld.dispatch.environmentCount;
    freeBodyConfiguration.captureDiagnostics = false;
    const RuntimeDiagnostics freeBodyInitialized =
        freeBodyRuntime.initialize(freeBodyWorld, freeBodyConfiguration);
    require(freeBodyInitialized.encoded,
            "sparse private contact rejected the bounded free-BOX source before allocation: " +
                freeBodyInitialized.message);
    std::cout << "sparse_private_admission=pass malformed_capacity=true"
              << " missing_selector=true bounded_free_box_initialized=true\n";
}

void runSparsePrivateHighPairAdmission(const MaterialProgram& bulk) {
    require(diagnosticStateOutputPath.empty(),
            "high-pair initialization mode does not accept transaction state dumps");
    MeshAndWorld source = makeWorld(
        bulk, false, true, kActivationGap, true);
    const CompiledWorld world = compile(source.world);
    require(world.dispatch.femNodeCount <= 8192u &&
                world.dispatch.tetrahedronCount <= 32768u &&
                world.dispatch.cohesiveFaceCount == 1u &&
                world.dispatch.rigidProxyCount == 9u &&
                world.dispatch.contactPairCount >
                    detail::kDensePrivateContactPairLimit &&
                world.dispatch.deformableContactCapacity <= 131072u,
            "high-pair constructor escaped another bounded full-contact dimension");
    require(world.dispatch.environmentCount == 1u &&
                world.dispatch.rigidGeneralizedCapacity == 0u &&
                world.dispatch.rigidQCapacity == 0u,
            "high-pair fixture must have one static/kinematic FEM environment");

    ScopedEnvironmentValue fullContactSelector(
        "NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT", "all");
    const std::string capacity = "64";
    ScopedEnvironmentValue sparseCapacity(
        "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_CAPACITY", capacity.c_str());
    ScopedEnvironmentValue sparseParity(
        "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_PARITY", "1");
    Runtime runtime;
    RuntimeConfiguration configuration;
    configuration.metallib = NUMI_MATTER_METALLIB;
    configuration.environmentCount = world.dispatch.environmentCount;
    configuration.captureDiagnostics = false;
    const RuntimeDiagnostics initialized = runtime.initialize(world, configuration);
    require(initialized.encoded,
            "sparse private initialization rejected the bounded P>65536 world: " +
                initialized.message);
    const auto sizing = detail::admitSparsePrivateContact(
        world.dispatch.environmentCount,
        world.dispatch.contactPairCount,
        64u,
        true,
        false);
    require(sizing.admitted(),
            "high-pair fixture disagrees with shared sparse byte admission");
    std::cout << "sparse_private_high_pair_admission=pass"
              << " environments=" << world.dispatch.environmentCount
              << " nodes=" << world.dispatch.femNodeCount
              << " tets=" << world.dispatch.tetrahedronCount
              << " cohesive_faces=" << world.dispatch.cohesiveFaceCount
              << " static_proxies=" << world.dispatch.rigidProxyCount
              << " canonical_pairs=" << world.dispatch.contactPairCount
              << " sparse_capacity=64 parity=true sparse_arena_bytes="
              << sizing.totalBytes
              << " dense_limit=" << detail::kDensePrivateContactPairLimit
              << " native_fem_transactions_executed=0 physical_validation=false\n";
}

void verifySparsePrivateSizingAdmission() {
    using detail::SparsePrivateContactAdmissionError;
    using detail::admitSparsePrivateContact;

    static_assert(sizeof(NMContactSampleGPU) == 160u);
    require(detail::densePrivateContactPairDomainSupported(65536u) &&
                !detail::densePrivateContactPairDomainSupported(65537u),
            "legacy dense private pair-domain cap changed with sparse admission");
    const auto justAboveDenseDomain =
        admitSparsePrivateContact(1u, 65537u, 64u, true, false);
    require(justAboveDenseDomain.admitted() &&
                justAboveDenseDomain.sampleBytes == 64u * 160u &&
                justAboveDenseDomain.pairSlotBytes == 65537u * 4u &&
                justAboveDenseDomain.candidateIdBytes == 64u * 4u &&
                justAboveDenseDomain.parityReferenceBytes == 65537u * 160u &&
                justAboveDenseDomain.candidateCountBytes == 16u &&
                justAboveDenseDomain.totalBytes ==
                    64u * 160u + 65537u * 4u + 64u * 4u +
                    65537u * 160u + 16u &&
                justAboveDenseDomain.totalBytes <=
                    detail::kSparsePrivateContactScratchBudgetBytes,
            "sparse byte/index admission failed immediately above the legacy dense cap");

    const auto oversizedFlattenedDomain = admitSparsePrivateContact(
        4u, std::numeric_limits<std::uint32_t>::max(), 1u, false, false);
    require(!oversizedFlattenedDomain.admitted() &&
                oversizedFlattenedDomain.error ==
                    SparsePrivateContactAdmissionError::
                        flattened_pair_domain_exceeds_uint32,
            "sparse admission allowed a flattened Metal uint index to wrap");

    const auto overBudget = admitSparsePrivateContact(
        1u, 1'000'000u, 1u, true, false);
    require(!overBudget.admitted() &&
                overBudget.error ==
                    SparsePrivateContactAdmissionError::scratch_budget_exceeded &&
                overBudget.totalBytes >
                    detail::kSparsePrivateContactScratchBudgetBytes,
            "sparse admission allowed the private arena to exceed 36 MiB");

    const auto freeRigidUnsupported =
        admitSparsePrivateContact(1u, 65537u, 64u, false, true);
    require(!freeRigidUnsupported.admitted() &&
                freeRigidUnsupported.error ==
                    SparsePrivateContactAdmissionError::free_rigid_unknowns_unsupported,
            "sparse admission allowed free rigid generalized unknowns");
    const auto boundedFreeBox =
        admitSparsePrivateContact(1u, 65537u, 64u, false, true, true);
    require(boundedFreeBox.admitted(),
            "sparse admission rejected the explicitly bounded one-BOX six-DOF scope");

    const auto capacityExceedsPairDomain =
        admitSparsePrivateContact(1u, 8u, 9u, false, false);
    require(!capacityExceedsPairDomain.admitted() &&
                capacityExceedsPairDomain.error ==
                    SparsePrivateContactAdmissionError::
                        candidate_capacity_exceeds_pair_domain,
            "sparse admission allowed candidate capacity past the canonical pair domain");

    NMCohesiveFaceGPU dormantMixed{};
    dormantMixed.adjacency.w = NM_TOPOLOGY_ACTIVE |
        NM_TOPOLOGY_PAIRED_COHESIVE | NM_TOPOLOGY_COHESIVE_MIXED_MODE |
        NM_TOPOLOGY_COHESIVE_DORMANT;
    dormantMixed.lawIdentity.x = NM_COHESIVE_LAW_MIXED_MODE_BILINEAR;
    dormantMixed.lawIdentity.y = 0u;
    require(!numi::matter::detail::fullContactArmijoSupportsCohesiveFaces({}) &&
                numi::matter::detail::fullContactArmijoSupportsCohesiveFaces(
                    std::span<const NMCohesiveFaceGPU>(&dormantMixed, 1u)),
            "full-contact admission must allow a valid authored dormant mixed face");
    NMCohesiveFaceGPU activeMixed = dormantMixed;
    activeMixed.adjacency.w &= ~NM_TOPOLOGY_COHESIVE_DORMANT;
    require(numi::matter::detail::fullContactArmijoSupportsCohesiveFaces(
                std::span<const NMCohesiveFaceGPU>(&activeMixed, 1u)),
            "activation state must not change structural full-contact eligibility");
    NMCohesiveFaceGPU unsupportedDormant = dormantMixed;
    unsupportedDormant.adjacency.w &= ~NM_TOPOLOGY_COHESIVE_MIXED_MODE;
    unsupportedDormant.lawIdentity.x = NM_COHESIVE_LAW_TRYDING_MODE_I;
    require(!numi::matter::detail::fullContactArmijoSupportsCohesiveFaces(
                std::span<const NMCohesiveFaceGPU>(&unsupportedDormant, 1u)),
            "inactive unsupported cohesive law must remain outside full-contact scope");

    std::cout << "sparse_private_sizing=pass"
              << " P_above_65536=65537 flattened_u32_overflow=rejected"
              << " dense_pair_limit=" << detail::kDensePrivateContactPairLimit
              << " private_budget_bytes="
              << detail::kSparsePrivateContactScratchBudgetBytes
              << " free_rigid_unknowns=rejected"
              << " dormant_mixed_authored_scope=admitted"
              << " inactive_unsupported_law=rejected\n";
}

void runOverflowRollback(const MaterialProgram& bulk,
                         const bool prescribeSupportBoxTranslation = false) {
    MeshAndWorld source = makeWorld(bulk, true, true, 5.0e-6);
    if (prescribeSupportBoxTranslation)
        source.world.rigidProxies.at(1u).prescribedEndPoseTranslation = true;
    verifyPartialMortarSource(source);
    Fixture run(source, true, prescribeSupportBoxTranslation);
    const RuntimeStateSnapshot before = run.snapshot();
    require(before.rigidStates.size() == run.world.dispatch.rigidProxyCount,
            "overflow rollback fixture rigid-state count mismatch");
    if (prescribeSupportBoxTranslation)
        require(run.world.contact.rigidProxies[1].flags ==
                    NM_RIGID_PRESCRIBED_TRANSLATION,
                "prescribed BOX translation was not retained in compiled proxy flags");
    StepResult rejected;
    std::string captured;
    {
        StderrCapture stderrCapture;
        rejected = run.step(0u, false, true);
        captured = stderrCapture.finish();
    }
    require(rejected.snapshot.statuses.size() == 1u &&
            rejected.snapshot.statuses[0].code == NM_STATUS_CAPACITY_OVERFLOW,
            "deliberately undersized self-contact arena did not report capacity overflow");
    require(rejected.snapshot.solverCertificates.size() == 1u &&
            rejected.snapshot.solverCertificates[0].validity.w <= 0.5f,
            "self-contact overflow retained an accepted certificate bit");
    require(rejected.snapshot.controlStep == before.controlStep &&
            rejected.snapshot.physicsSubstep == before.physicsSubstep &&
            std::memcmp(rejected.snapshot.femNodes.data(), before.femNodes.data(),
                before.femNodes.size() * sizeof(NMFEMNodeStateGPU)) == 0 &&
            std::memcmp(rejected.snapshot.cohesiveFaces.data(), before.cohesiveFaces.data(),
                before.cohesiveFaces.size() * sizeof(NMCohesiveFaceGPU)) == 0,
            "overflow rejection failed to roll back accepted FEM/cohesive state");
    require(rejected.snapshot.rigidStates.size() == before.rigidStates.size() &&
            std::memcmp(rejected.snapshot.rigidStates.data(), before.rigidStates.data(),
                before.rigidStates.size() * sizeof(NMRigidStateGPU)) == 0,
            "overflow rejection failed to roll back projected rigid proxy states");
    require(captured.find("fem_full_contact_residual_armijo_enabled=") !=
                std::string::npos,
            "overflow fixture did not enter the bounded full-contact path");
    std::cout << "full_contact_capacity_overflow=pass capacity=1"
              << " prescribed_box_translation="
              << (prescribeSupportBoxTranslation ? "true" : "false")
              << " rollback_certificate_cleared=true\n";
}

void runPrescribedBoxSweepRejection(const MaterialProgram& bulk) {
    MeshAndWorld source = makeWorld(bulk, false, true, 5.0e-6);
    source.world.rigidProxies.at(1u).prescribedEndPoseTranslation = true;
    verifyPartialMortarSource(source);
    Fixture run(source, false, true, true);
    const RuntimeStateSnapshot before = run.snapshot();
    require(run.world.contact.rigidProxies[1].flags ==
                NM_RIGID_PRESCRIBED_TRANSLATION,
            "swept-box fixture did not compile a prescribed BOX proxy");
    const StepResult rejected = run.step(0u, false, false);
    require(rejected.snapshot.statuses.size() == 1u &&
            rejected.snapshot.statuses[0].code == NM_STATUS_CONTACT_FAILURE,
            "a prescribed BOX swept through FEM nodes without a contact rejection");
    require(rejected.snapshot.solverCertificates.size() == 1u &&
            rejected.snapshot.solverCertificates[0].validity.w <= 0.5f,
            "swept BOX rejection retained an accepted solver certificate");
    require(rejected.snapshot.controlStep == before.controlStep &&
            rejected.snapshot.physicsSubstep == before.physicsSubstep &&
            rejected.snapshot.femNodes.size() == before.femNodes.size() &&
            std::memcmp(rejected.snapshot.femNodes.data(), before.femNodes.data(),
                before.femNodes.size() * sizeof(NMFEMNodeStateGPU)) == 0 &&
            rejected.snapshot.femMaterialState.size() ==
                before.femMaterialState.size() &&
            std::memcmp(rejected.snapshot.femMaterialState.data(),
                before.femMaterialState.data(),
                before.femMaterialState.size() * sizeof(float)) == 0 &&
            rejected.snapshot.rigidStates.size() == before.rigidStates.size() &&
            std::memcmp(rejected.snapshot.rigidStates.data(), before.rigidStates.data(),
                before.rigidStates.size() * sizeof(NMRigidStateGPU)) == 0 &&
            rejected.snapshot.cohesiveFaces.size() ==
                before.cohesiveFaces.size() &&
            std::memcmp(rejected.snapshot.cohesiveFaces.data(),
                before.cohesiveFaces.data(),
                before.cohesiveFaces.size() * sizeof(NMCohesiveFaceGPU)) == 0 &&
            rejected.snapshot.deformableContactHistories.size() ==
                before.deformableContactHistories.size() &&
            std::memcmp(rejected.snapshot.deformableContactHistories.data(),
                before.deformableContactHistories.data(),
                before.deformableContactHistories.size() *
                    sizeof(NMDeformableContactHistoryGPU)) == 0,
            "swept BOX rejection changed accepted FEM, material, rigid, cohesive, or contact state");
    std::cout << "prescribed_box_swept_crossing=reject"
              << " rigid_and_fem_rollback=true contact_floor_enforced=true\n";
}

void verifyPrescribedBoxRollback(const RuntimeStateSnapshot& rejected,
                                 const RuntimeStateSnapshot& accepted,
                                 const char* mismatchName) {
    require(rejected.statuses.size() == 1u &&
            rejected.statuses[0].code != NM_STATUS_SUCCESS,
            std::string("prescribed BOX ") + mismatchName +
                " mismatch unexpectedly accepted");
    require(rejected.solverCertificates.size() == 1u &&
            rejected.solverCertificates[0].validity.w <= 0.5f,
            std::string("prescribed BOX ") + mismatchName +
                " rejection retained an accepted certificate");
    require(rejected.controlStep == accepted.controlStep &&
            rejected.physicsSubstep == accepted.physicsSubstep,
            std::string("prescribed BOX ") + mismatchName +
                " rejection advanced the accepted cursor");
    require(rejected.femNodes.size() == accepted.femNodes.size() &&
            std::memcmp(rejected.femNodes.data(), accepted.femNodes.data(),
                accepted.femNodes.size() * sizeof(NMFEMNodeStateGPU)) == 0 &&
            rejected.femMaterialState.size() == accepted.femMaterialState.size() &&
            std::memcmp(rejected.femMaterialState.data(), accepted.femMaterialState.data(),
                accepted.femMaterialState.size() * sizeof(float)) == 0 &&
            rejected.rigidStates.size() == accepted.rigidStates.size() &&
            std::memcmp(rejected.rigidStates.data(), accepted.rigidStates.data(),
                accepted.rigidStates.size() * sizeof(NMRigidStateGPU)) == 0 &&
            rejected.cohesiveFaces.size() == accepted.cohesiveFaces.size() &&
            std::memcmp(rejected.cohesiveFaces.data(), accepted.cohesiveFaces.data(),
                accepted.cohesiveFaces.size() * sizeof(NMCohesiveFaceGPU)) == 0 &&
            rejected.deformableContactHistories.size() ==
                accepted.deformableContactHistories.size() &&
            std::memcmp(rejected.deformableContactHistories.data(),
                accepted.deformableContactHistories.data(),
                accepted.deformableContactHistories.size() *
                    sizeof(NMDeformableContactHistoryGPU)) == 0,
            std::string("prescribed BOX ") + mismatchName +
                " rejection changed accepted FEM/material/rigid/cohesive/contact state");
}

void runPrescribedBoxTranslationAcceptance(const MaterialProgram& bulk) {
    // This is a prescribed-geometry acceptance/rollback instrument, not an
    // active self-contact or cohesive solve. Keep the partial mortar patch
    // authored and self-contact enabled, but start it at the existing dormant
    // activation gap. A 5 um split here is inside the fixture's 10 um contact
    // slop and adds an unrelated loaded nonlinear solve to the BOX contract.
    MeshAndWorld source = makeWorld(bulk, false, true, kActivationGap);
    source.world.rigidProxies.at(1u).prescribedEndPoseTranslation = true;
    verifyPartialMortarSource(source);
    Fixture run(source, false, true, false, true);
    const RuntimeStateSnapshot before = run.snapshot();
    require(kActivationGap > source.world.contactSlop,
            "prescribed BOX translation fixture must keep its dormant split outside contact slop");
    require(!hasMortarParentContact(before, source),
            "prescribed BOX translation fixture unexpectedly begins with active mortar-parent contact");
    require(before.femNodes.size() == source.nodes.size() &&
            before.rigidStates.size() == 2u,
            "prescribed BOX translation clearance preflight lacks the authored FEM or tool state");
    double highestFEMNodeZ = -std::numeric_limits<double>::infinity();
    for (const NMFEMNodeStateGPU& nodeState : before.femNodes)
        highestFEMNodeZ = std::max(highestFEMNodeZ,
            double(nodeState.positionAndMass.z));
    // Initial rigidStates have not necessarily been projected at the first
    // preDynamics boundary. Read the exact shared MRBodyStateGPU input that
    // was uploaded by Fixture and combine it with the authored capsule proxy
    // endpoints; do not use an uninitialized projected-state snapshot.
    require(run.bodyBuffer != nil && run.bodyBuffer.contents != nullptr &&
            run.bodyBuffer.length >= 2u * sizeof(MRBodyStateGPU),
            "prescribed BOX clearance preflight cannot read its uploaded body poses");
    const auto* const uploadedBodies =
        static_cast<const MRBodyStateGPU*>(run.bodyBuffer.contents);
    const MRBodyStateGPU& uploadedPunchBody = uploadedBodies[kPunchBody];
    const MRBodyStateGPU uploadedSupportBody = uploadedBodies[kAnvilBody];
    const RigidProxySource& punchProxy =
        source.world.rigidProxies.at(kPunchBody);
    const RigidProxySource& supportProxy =
        source.world.rigidProxies.at(kAnvilBody);
    require(punchProxy.shape == NM_RIGID_CAPSULE &&
            std::isfinite(punchProxy.radiusOrOffset) &&
            punchProxy.radiusOrOffset > 0.0,
            "prescribed BOX clearance preflight requires the authored punch capsule");
    require(supportProxy.shape == NM_RIGID_BOX,
            "prescribed BOX translation fixture requires the authored support BOX");
    const ToolQuaternion punchOrientation = normalized(ToolQuaternion{
        double(uploadedPunchBody.orientation.x),
        double(uploadedPunchBody.orientation.y),
        double(uploadedPunchBody.orientation.z),
        double(uploadedPunchBody.orientation.w),
    });
    const ToolVec3 punchPosition{
        double(uploadedPunchBody.position.x),
        double(uploadedPunchBody.position.y),
        double(uploadedPunchBody.position.z),
    };
    const auto worldEndpoint = [&](const std::array<double, 3>& local) {
        return punchPosition + rotate(punchOrientation,
            ToolVec3{local[0], local[1], local[2]});
    };
    const ToolVec3 endpointA = worldEndpoint(punchProxy.localCenter);
    const ToolVec3 endpointB = worldEndpoint(punchProxy.localExtent);
    const double lowestPunchZ = std::min(endpointA.z, endpointB.z) -
                                punchProxy.radiusOrOffset;
    const double measuredPunchClearance = lowestPunchZ - highestFEMNodeZ;
    std::cout << std::setprecision(9)
              << "prescribed_box_unrelated_punch_clearance_preflight="
              << (std::isfinite(measuredPunchClearance) &&
                  measuredPunchClearance >= kInitialClearance - 1.0e-7
                    ? "pass" : "fail")
              << " highest_node_z_m=" << highestFEMNodeZ
              << " lowest_punch_z_m=" << lowestPunchZ
              << " clearance_m=" << measuredPunchClearance
              << " declared_clearance_m=" << kInitialClearance
              << " body_pose_source=uploaded_MRBodyStateGPU"
              << " geometry_source=authored_capsule_proxy_endpoints\n";
    require(std::isfinite(measuredPunchClearance) &&
            measuredPunchClearance >= kInitialClearance - 1.0e-7,
            "BOX translation fixture's uploaded parked punch does not clear the actual highest FEM node by its declared margin");
    require(run.world.contact.rigidProxies[1].flags ==
                NM_RIGID_PRESCRIBED_TRANSLATION,
            "separated-translation fixture did not compile a prescribed BOX proxy");

    const ToolQuaternion supportStartOrientation = normalized(ToolQuaternion{
        double(uploadedSupportBody.orientation.x),
        double(uploadedSupportBody.orientation.y),
        double(uploadedSupportBody.orientation.z),
        double(uploadedSupportBody.orientation.w),
    });
    const ToolVec3 supportStartPosition{
        double(uploadedSupportBody.position.x),
        double(uploadedSupportBody.position.y),
        double(uploadedSupportBody.position.z),
    };
    const ToolVec3 supportLocalCenter{
        supportProxy.localCenter[0], supportProxy.localCenter[1],
        supportProxy.localCenter[2],
    };
    const ToolVec3 expectedStartProxyCenter = supportStartPosition +
        rotate(supportStartOrientation, supportLocalCenter);
    const double stepDuration = run.runtime.timestepSeconds();
    const ToolVec3 supportVelocity{
        double(uploadedSupportBody.linearVelocityAndInverseMass.x),
        double(uploadedSupportBody.linearVelocityAndInverseMass.y),
        double(uploadedSupportBody.linearVelocityAndInverseMass.z),
    };
    // Match the native fixture's uploaded float endpoint arithmetic exactly;
    // this is the commanded body pose, then the authored proxy transform is
    // applied to obtain its expected accepted center.
    const float endpointDt = static_cast<float>(stepDuration);
    const ToolVec3 expectedEndBodyPosition{
        double(uploadedSupportBody.position.x +
            uploadedSupportBody.linearVelocityAndInverseMass.x * endpointDt),
        double(uploadedSupportBody.position.y +
            uploadedSupportBody.linearVelocityAndInverseMass.y * endpointDt),
        double(uploadedSupportBody.position.z +
            uploadedSupportBody.linearVelocityAndInverseMass.z * endpointDt),
    };
    const ToolVec3 expectedEndProxyCenter = expectedEndBodyPosition +
        rotate(supportStartOrientation, supportLocalCenter);

    // Request control step 1 so the public cursor's controlStep field visibly
    // advances from the freshly initialized zero cursor.
    const StepResult acceptedStep = run.step(1u);
    const RuntimeStateSnapshot& accepted = acceptedStep.snapshot;
    requireAccepted(accepted);
    require(!hasMortarParentContact(accepted, source),
            "separated BOX translation unexpectedly activated mortar-parent self-contact");
    require(accepted.controlStep > before.controlStep,
            "accepted prescribed BOX translation did not advance the native cursor");
    require(accepted.rigidStates.size() == before.rigidStates.size() &&
            accepted.rigidStates.size() == 2u,
            "prescribed BOX translation changed the rigid-state layout");
    const NMRigidStateGPU& newBox = accepted.rigidStates[kAnvilBody];
    const ToolVec3 measuredEndBodyPosition{
        double(newBox.bodyCenter.x), double(newBox.bodyCenter.y),
        double(newBox.bodyCenter.z),
    };
    const ToolVec3 measuredEndProxyCenter{
        double(newBox.centerAndRadius.x), double(newBox.centerAndRadius.y),
        double(newBox.centerAndRadius.z),
    };
    constexpr double kAcceptedPoseTolerance = 2.0e-7;
    require(length(measuredEndProxyCenter - expectedEndProxyCenter) <=
                kAcceptedPoseTolerance &&
            length(measuredEndBodyPosition - expectedEndBodyPosition) <=
                kAcceptedPoseTolerance,
            "accepted prescribed BOX center did not advance by dt times its authored velocity");
    const ToolQuaternion expectedSupportOrientation = multiply(
        supportStartOrientation, normalized(ToolQuaternion{
            supportProxy.localOrientation[0],
            supportProxy.localOrientation[1],
            supportProxy.localOrientation[2],
            supportProxy.localOrientation[3],
        }));
    const bool orientationMatches =
        std::abs(double(newBox.orientation.x) - expectedSupportOrientation.x) <= 1.0e-6 &&
        std::abs(double(newBox.orientation.y) - expectedSupportOrientation.y) <= 1.0e-6 &&
        std::abs(double(newBox.orientation.z) - expectedSupportOrientation.z) <= 1.0e-6 &&
        std::abs(double(newBox.orientation.w) - expectedSupportOrientation.w) <= 1.0e-6;
    const bool extentMatches =
        std::abs(double(newBox.extent.x) - supportProxy.localExtent[0]) <= 1.0e-7 &&
        std::abs(double(newBox.extent.y) - supportProxy.localExtent[1]) <= 1.0e-7 &&
        std::abs(double(newBox.extent.z) - supportProxy.localExtent[2]) <= 1.0e-7;
    const bool velocityMatches =
        std::abs(double(newBox.linearVelocityAndInverseMass.x) - supportVelocity.x) <= 1.0e-7 &&
        std::abs(double(newBox.linearVelocityAndInverseMass.y) - supportVelocity.y) <= 1.0e-7 &&
        std::abs(double(newBox.linearVelocityAndInverseMass.z) - supportVelocity.z) <= 1.0e-7;
    const double measuredTranslation = measuredEndBodyPosition.x -
                                       supportStartPosition.x;
    const double expectedTranslation = expectedEndBodyPosition.x -
                                       supportStartPosition.x;
    require(extentMatches && orientationMatches && velocityMatches,
            "accepted prescribed BOX translation changed its fixed shape/orientation/velocity");

    // Each mismatch is attempted from the same last accepted translated pose.
    // A rejected request must leave both its rigid state and the rest of the
    // accepted native transaction untouched.
    const StepResult velocityMismatch = run.step(2u, false, false, true, false);
    verifyPrescribedBoxRollback(velocityMismatch.snapshot, accepted,
                                "endpoint-velocity");
    const StepResult orientationMismatch = run.step(2u, false, false, false, true);
    verifyPrescribedBoxRollback(orientationMismatch.snapshot, accepted,
                                "orientation");

    std::cout << "prescribed_box_separated_translation=pass"
              << " initial_body_center_m=(" << supportStartPosition.x << ','
              << supportStartPosition.y << ',' << supportStartPosition.z << ')'
              << " initial_proxy_center_m=(" << expectedStartProxyCenter.x << ','
              << expectedStartProxyCenter.y << ',' << expectedStartProxyCenter.z << ')'
              << " expected_final_body_center_m=(" << expectedEndBodyPosition.x << ','
              << expectedEndBodyPosition.y << ',' << expectedEndBodyPosition.z << ')'
              << " measured_final_body_center_m=(" << measuredEndBodyPosition.x << ','
              << measuredEndBodyPosition.y << ',' << measuredEndBodyPosition.z << ')'
              << " expected_final_proxy_center_m=(" << expectedEndProxyCenter.x << ','
              << expectedEndProxyCenter.y << ',' << expectedEndProxyCenter.z << ')'
              << " measured_final_proxy_center_m=(" << measuredEndProxyCenter.x << ','
              << measuredEndProxyCenter.y << ',' << measuredEndProxyCenter.z << ')'
              << " expected_translation_m=" << expectedTranslation
              << " measured_translation_m=" << measuredTranslation
              << " endpoint_velocity_mismatch=reject orientation_mismatch=reject"
              << " rejected_state_rollback=true kinematic_geometry_contract_only=true\n";
}

} // namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        try {
            require(argc >= 2,
                "usage: cardboard-full-contact-armijo-check BULK.nmatter [--cpu-only|--overflow|--prescribed-box-overflow|--prescribed-box-sweep-reject|--prescribed-box-translation|--private-self-sweep|--sparse-private-parity|--sparse-private-overflow|--sparse-private-admission|--sparse-private-high-pair-admission|--sparse-free-body-parity|--sparse-free-body-overflow|--self-surface-scratch-alias-parity|--cohesive-scratch-alias-parity] [--regional-contact-preconditioner] [--extended-budget] [--accurate-paper] [--newton-iterations N] [--diagnostic-state-output DIR]");
            bool cpuOnly = false;
            bool overflow = false;
            bool prescribedBoxOverflow = false;
            bool prescribedBoxSweepReject = false;
            bool prescribedBoxTranslation = false;
            bool privateSelfSweep = false;
            bool armijoAlphaPublication = false;
            bool sparsePrivateParity = false;
            bool sparsePrivateOverflow = false;
            bool sparsePrivateAdmission = false;
            bool sparsePrivateHighPairAdmission = false;
            bool sparseFreeBodyParity = false;
            bool sparseFreeBodyOverflow = false;
            bool selfSurfaceScratchAliasParity = false;
            bool cohesiveScratchAliasParity = false;
            for (int index = 2; index < argc; ++index) {
                const std::string option(argv[index]);
                if (option == "--cpu-only") cpuOnly = true;
                else if (option == "--overflow") overflow = true;
                else if (option == "--prescribed-box-overflow")
                    prescribedBoxOverflow = true;
                else if (option == "--prescribed-box-sweep-reject")
                    prescribedBoxSweepReject = true;
                else if (option == "--prescribed-box-translation")
                    prescribedBoxTranslation = true;
                else if (option == "--private-self-sweep")
                    privateSelfSweep = true;
                else if (option == "--armijo-alpha-publication")
                    armijoAlphaPublication = true;
                else if (option == "--sparse-private-parity")
                    sparsePrivateParity = true;
                else if (option == "--sparse-private-overflow")
                    sparsePrivateOverflow = true;
                else if (option == "--sparse-private-admission")
                    sparsePrivateAdmission = true;
                else if (option == "--sparse-private-high-pair-admission")
                    sparsePrivateHighPairAdmission = true;
                else if (option == "--sparse-free-body-parity")
                    sparseFreeBodyParity = true;
                else if (option == "--sparse-free-body-overflow")
                    sparseFreeBodyOverflow = true;
                else if (option == "--self-surface-scratch-alias-parity")
                    selfSurfaceScratchAliasParity = true;
                else if (option == "--cohesive-scratch-alias-parity")
                    cohesiveScratchAliasParity = true;
                else if (option == "--regional-contact-preconditioner")
                    regionalContactPreconditioner = true;
                else if (option == "--extended-budget") extendedBudget = true;
                else if (option == "--accurate-paper") accuratePaper = true;
                else if (option == "--diagnostic-state-output") {
                    require(index + 1 < argc, "missing diagnostic state output directory");
                    require(diagnosticStateOutputPath.empty(),
                            "diagnostic state output directory was specified more than once");
                    diagnosticStateOutputPath = argv[++index];
                    require(!diagnosticStateOutputPath.empty(),
                            "diagnostic state output directory must not be empty");
                }
                else if (option == "--newton-iterations") {
                    require(index + 1 < argc, "missing Newton iteration budget");
                    const std::string value(argv[++index]);
                    require(!value.empty() && std::all_of(value.begin(), value.end(),
                        [](const char c) { return c >= '0' && c <= '9'; }),
                        "Newton iteration budget must be an integer");
                    const unsigned long parsed = std::stoul(value);
                    require(parsed >= 1u && parsed <= 256u,
                        "diagnostic Newton iteration budget must be in [1,256]");
                    authoredNewtonIterations = static_cast<std::uint32_t>(parsed);
                } else throw std::runtime_error("unknown option: " + option);
            }
            require(!(cpuOnly && (overflow || prescribedBoxOverflow ||
                                  prescribedBoxSweepReject || prescribedBoxTranslation ||
                                  privateSelfSweep || armijoAlphaPublication ||
                                  sparsePrivateParity ||
                                  sparsePrivateOverflow || sparsePrivateAdmission ||
                                  sparsePrivateHighPairAdmission ||
                                  sparseFreeBodyParity || sparseFreeBodyOverflow ||
                                  selfSurfaceScratchAliasParity ||
                                  cohesiveScratchAliasParity)),
                    "CPU and overflow modes are exclusive");
            require(!armijoAlphaPublication ||
                        !(cpuOnly || overflow || prescribedBoxOverflow ||
                          prescribedBoxSweepReject || prescribedBoxTranslation ||
                          privateSelfSweep || sparsePrivateParity ||
                          sparsePrivateOverflow || sparsePrivateAdmission ||
                          sparsePrivateHighPairAdmission || sparseFreeBodyParity ||
                          sparseFreeBodyOverflow || selfSurfaceScratchAliasParity ||
                          cohesiveScratchAliasParity || regionalContactPreconditioner ||
                          extendedBudget || accuratePaper ||
                          authoredNewtonIterations != 0u ||
                          !diagnosticStateOutputPath.empty()),
                    "Armijo alpha publication mode must run alone");
            require(!privateSelfSweep || !(overflow || prescribedBoxOverflow ||
                                  prescribedBoxSweepReject || prescribedBoxTranslation),
                    "private self-sweep regression mode must run alone");
            require(!(overflow && (prescribedBoxOverflow || prescribedBoxTranslation)),
                    "choose only one overflow fixture mode");
            require(!(prescribedBoxSweepReject &&
                      (overflow || prescribedBoxOverflow || prescribedBoxTranslation)),
                    "choose only one prescribed-box regression mode");
            require(!(prescribedBoxTranslation && prescribedBoxOverflow),
                    "choose only one prescribed-box regression mode");
            const unsigned sparseModeCount =
                static_cast<unsigned>(sparsePrivateParity) +
                static_cast<unsigned>(sparsePrivateOverflow) +
                static_cast<unsigned>(sparsePrivateAdmission) +
                static_cast<unsigned>(sparsePrivateHighPairAdmission) +
                static_cast<unsigned>(sparseFreeBodyParity) +
                static_cast<unsigned>(sparseFreeBodyOverflow);
            require(sparseModeCount <= 1u,
                    "choose only one sparse private-contact regression mode");
            require(!selfSurfaceScratchAliasParity || sparseModeCount == 0u,
                    "surface scratch alias parity must run separately from sparse private modes");
            require(!cohesiveScratchAliasParity || sparseModeCount == 0u,
                    "cohesive scratch alias parity must run separately from sparse private modes");
            require(!(selfSurfaceScratchAliasParity && cohesiveScratchAliasParity),
                    "run one scratch-alias parity mode at a time");
            require(!selfSurfaceScratchAliasParity ||
                        !(overflow || prescribedBoxOverflow ||
                          prescribedBoxSweepReject || prescribedBoxTranslation ||
                          privateSelfSweep || cohesiveScratchAliasParity),
                    "surface scratch alias parity must run as its own full-contact sequence");
            require(!cohesiveScratchAliasParity ||
                        !(overflow || prescribedBoxOverflow ||
                          prescribedBoxSweepReject || prescribedBoxTranslation ||
                          privateSelfSweep),
                    "cohesive scratch alias parity must run as its own full-contact sequence");
            require(sparseModeCount == 0u ||
                        !(overflow || prescribedBoxOverflow ||
                          prescribedBoxSweepReject || prescribedBoxTranslation ||
                          privateSelfSweep),
                    "sparse private-contact modes must run separately");
            require(!(sparseFreeBodyParity && sparseFreeBodyOverflow),
                    "choose either free-body parity or its overflow control");
#if !defined(NUMI_CARDBOARD_METAL_WORLD)
            require(!sparseFreeBodyParity && !sparseFreeBodyOverflow,
                    "sparse free-body checks require the integrated MetalWorld target");
#endif
            require(diagnosticStateOutputPath.empty() || sparsePrivateParity ||
                        sparsePrivateOverflow || selfSurfaceScratchAliasParity ||
                        cohesiveScratchAliasParity || sparseFreeBodyParity ||
                        sparseFreeBodyOverflow,
                    "diagnostic state output is supported only for retained contact parity or overflow modes");
            require(!(cpuOnly && !diagnosticStateOutputPath.empty()),
                    "diagnostic state export requires a native transaction run");
            if (extendedBudget) std::cout <<
                "authored_extended_budget=fgmres256,restart16,line_search24 relative_residual_unchanged=1e-7\n";
            std::cout << "authored_newton_iterations=" <<
                (authoredNewtonIterations != 0u ? authoredNewtonIterations :
                    (extendedBudget ? 32u : 16u)) <<
                " accurate_paper=" << (accuratePaper ? "true" : "false") << '\n';
            MaterialProgram bulk = readMaterial(argv[1]);
            if (accuratePaper) {
                bulk.paperHillScalarInitializer = true;
                bulk.paperCompensatedArithmetic = true;
                bulk.paperHillAccurateProjection = true;
                bulk = scalePaperProjectionResiduals(std::move(bulk));
            }
            verifyBarrierRegularization();
            verifyArmijoRequiresResolvedDescent();
            verifyAuthoredLineSearchBudget(bulk);
            verifyPrescribedBoxCompilerAdmission(bulk);
            verifySparsePrivateSizingAdmission();
            if (armijoAlphaPublication) {
                verifyArmijoAlphaPublication();
                std::cout << "PASS Armijo environment-alpha publication kernel check"
                          << " checks=" << checks << '\n';
                return 0;
            }
            if (privateSelfSweep) {
                verifyPrivateSelfSweepValidator();
                std::cout << "PASS private self-sweep validator fixture"
                          << " checks=" << checks << '\n';
                return 0;
            }
            MeshAndWorld source = makeWorld(
                bulk, overflow || prescribedBoxOverflow);
            if (prescribedBoxOverflow || prescribedBoxSweepReject ||
                prescribedBoxTranslation)
                source.world.rigidProxies.at(1u).prescribedEndPoseTranslation = true;
            const CompiledWorld compiled = compile(source.world);
            const std::uint32_t expectedPreconditioner =
                regionalContactPreconditioner
                    ? static_cast<std::uint32_t>(
                        FEMPreconditionerMode::regionalGalerkinContact)
                    : static_cast<std::uint32_t>(
                        FEMPreconditionerMode::scalarDiagonal);
            require(compiled.mixedSolver.executionBudgets.z ==
                        expectedPreconditioner,
                    "compiled source selected a different FEM preconditioner than authored");
            std::cout << "selected_fem_preconditioner="
                      << (regionalContactPreconditioner
                            ? "regional_galerkin_contact"
                            : "scalar_diagonal")
                      << " selector=" << expectedPreconditioner << '\n';
            require(compiled.dispatch.environmentCount == 1u &&
                    compiled.dispatch.cohesiveFaceCount == 1u &&
                    compiled.dispatch.rigidProxyCount == 2u &&
                    compiled.dispatch.contactPairCount > 0u &&
                    compiled.dispatch.deformableContactCapacity > 0u,
                "full-contact source failed required bounded native dimensions");
            require((compiled.objects[0].flags & NM_OBJECT_DISABLE_SELF_CONTACT) == 0u &&
                    (compiled.objects[0].flags & NM_OBJECT_DISABLE_DEFORMABLE_CONTACT) == 0u,
                "compiled source did not enable self-contact and deformable contact");
            if (cpuOnly) {
                MeshAndWorld geometrySource = makeWorld(
                    bulk, false, true, kActivationGap);
                verifyPartialMortarSource(geometrySource);
                verifyLegacyDetachedTetGeometryRejected();
                std::cout << "split_pad_cpu_geometry=pass"
                          << " interface_triangles=8 duplicated_interface_nodes=9"
                          << " positive_gap_m=" << geometrySource.interfaceGapM
                          << " old_one_face_overlap_rejected=true\n";
                std::cout << "PASS full-contact source admission native_fem_transactions_executed=0"
                          << " physical_validation=false synthetic_law=true"
                          << " nodes=" << compiled.dispatch.femNodeCount
                          << " tets=" << compiled.dispatch.tetrahedronCount
                          << " contact_pairs=" << compiled.dispatch.contactPairCount
                          << " deformable_capacity=" << compiled.dispatch.deformableContactCapacity
                          << '\n';
                return 0;
            }
            if (!diagnosticStateOutputPath.empty())
                diagnosticStateOutput = std::make_unique<DiagnosticStateOutput>(
                    diagnosticStateOutputPath);
            if (sparsePrivateAdmission) {
                verifySparsePrivateAdmission(bulk, compiled);
                std::cout << "PASS sparse private-contact admission fixture"
                          << " native_fem_transactions_executed="
                          << nativeFEMTransactionsExecuted
                          << " physical_validation=false synthetic_law=true\n";
                return 0;
            }
            if (sparsePrivateHighPairAdmission) {
                runSparsePrivateHighPairAdmission(bulk);
                std::cout << "PASS sparse private high-pair initialization fixture"
                          << " native_fem_transactions_executed="
                          << nativeFEMTransactionsExecuted
                          << " physical_validation=false synthetic_law=true\n";
                return 0;
            }
            if (sparsePrivateParity) {
                runSparsePrivateParity(bulk);
                std::cout << "PASS sparse private-contact parity fixture"
                          << " checks=" << checks
                          << " native_fem_transactions_executed="
                          << nativeFEMTransactionsExecuted
                          << " physical_validation=false synthetic_law=true\n";
                return 0;
            }
            if (selfSurfaceScratchAliasParity) {
                runSelfSurfaceScratchAliasParity(bulk);
                std::cout << "PASS full-contact self-surface scratch alias parity"
                          << " checks=" << checks
                          << " native_fem_transactions_executed="
                          << nativeFEMTransactionsExecuted
                          << " physical_validation=false synthetic_law=true\n";
                return 0;
            }
            if (cohesiveScratchAliasParity) {
                runCohesiveScratchAliasParity(bulk);
                std::cout << "PASS full-contact cohesive scratch alias parity"
                          << " checks=" << checks
                          << " native_fem_transactions_executed="
                          << nativeFEMTransactionsExecuted
                          << " physical_validation=false synthetic_law=true\n";
                return 0;
            }
            if (sparsePrivateOverflow) {
                runSparsePrivateOverflow(bulk, 1u);
                std::cout << "PASS sparse private-contact overflow fixture"
                          << " checks=" << checks
                          << " native_fem_transactions_executed="
                          << nativeFEMTransactionsExecuted
                          << " physical_validation=false synthetic_law=true\n";
                return 0;
            }
#if defined(NUMI_CARDBOARD_METAL_WORLD)
            if (sparseFreeBodyParity) {
                runSparseFreeBodyParity(bulk);
                std::cout << "PASS sparse private free-body dense parity"
                          << " checks=" << checks
                          << " native_fem_transactions_executed="
                          << nativeFEMTransactionsExecuted
                          << " physical_validation=false synthetic_fixture=true\n";
                return 0;
            }
            if (sparseFreeBodyOverflow) {
                runSparseFreeBodyOverflow(bulk);
                std::cout << "PASS sparse private free-body overflow rollback"
                          << " checks=" << checks
                          << " native_fem_transactions_executed="
                          << nativeFEMTransactionsExecuted
                          << " physical_validation=false synthetic_fixture=true\n";
                return 0;
            }
#endif
            if (overflow || prescribedBoxOverflow)
                runOverflowRollback(bulk, prescribedBoxOverflow);
            else if (prescribedBoxSweepReject)
                runPrescribedBoxSweepRejection(bulk);
            else if (prescribedBoxTranslation)
                runPrescribedBoxTranslationAcceptance(bulk);
            else {
                runDormantUncoveredContact(bulk);
                runAcceptedAndRollback(bulk);
            }
            std::cout << "PASS full-contact residual Armijo native fixture"
                      << " checks=" << checks
                      << " native_fem_transactions_executed="
                      << nativeFEMTransactionsExecuted
                      << " physical_validation=false synthetic_law=true\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "FAIL full-contact residual Armijo fixture: "
                      << error.what() << '\n';
            return 1;
        }
    }
}

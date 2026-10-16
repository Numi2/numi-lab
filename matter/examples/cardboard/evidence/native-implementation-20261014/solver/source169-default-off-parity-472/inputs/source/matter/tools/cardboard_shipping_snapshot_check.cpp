#include "cardboard_shipping_snapshot.hpp"
#include "cardboard_shipping_driver.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>

namespace {

using namespace numi::cardboard;
using numi::matter::RuntimeStateSnapshot;

void check(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

RuntimeStateSnapshot acceptedState() {
    RuntimeStateSnapshot state;
    state.available = true;
    state.sourcePhysicsFingerprint = 0x123456789abcdef0ull;
    state.deviceProgramFingerprint = 0x0fedcba987654321ull;
    state.controlStep = 17u;
    state.statuses.resize(1u);
    state.statuses[0].code = NM_STATUS_SUCCESS;
    state.solverCertificates.resize(1u);
    state.solverCertificates[0].validity.w = 1.0f;
    state.solverCertificates[0].nonlinear = {1.0e-7f, 1.0e-7f, 2.0e-7f, 3.0e-7f};
    return state;
}

numi::matter::CompiledWorld shippingLayoutFixture() {
    numi::matter::CompiledWorld world;
    world.dispatch.environmentCount = 1u;
    world.dispatch.objectCount = 1u;
    world.dispatch.femNodeCount = 4u;
    world.dispatch.tetrahedronCount = 1u;
    world.dispatch.rigidProxyCount = 1u;
    world.dispatch.contactPairCount = 1u;
    world.dispatch.materialStateStride = 2u;
    world.dispatch.cohesiveFaceCount = 1u;
    world.dispatch.rigidGeneralizedCapacity = 6u;
    world.dispatch.deformableContactCapacity = 3u;
    world.dispatch.mpmActiveNodeCapacity = 2u;
    world.dispatch.gridNodeCount = 3u;
    world.objects.resize(1u);
    world.fem.nodes.resize(4u);
    world.fem.tetrahedra.resize(1u);
    world.fem.fields.resize(1u);
    world.fem.topologyNodes.resize(4u);
    world.fem.cohesiveFaces.resize(1u);
    world.contact.rigidProxies.resize(1u);
    world.contact.pairs.resize(1u);
    world.parameters.resize(2u);
    world.learnedWeights.resize(1u);
    world.identification.resize(1u);
    world.adaptive.resize(1u);
    world.schedulers.resize(1u);
    return world;
}

RuntimeStateSnapshot completeShippingState(
    const numi::matter::CompiledWorld& world) {
    RuntimeStateSnapshot state = acceptedState();
    state.materialStateStride = world.dispatch.materialStateStride;
    state.femNodes.resize(4u);
    state.femFields.resize(1u);
    state.femMaterialState.resize(2u);
    state.femTopologyNodes.resize(4u);
    state.femTopologyTetrahedra.resize(1u);
    state.cohesiveFaces.resize(1u);
    state.topologyStates.resize(1u);
    state.solverCertificates.resize(1u);
    state.particles.clear();
    state.particleMaterialState.clear();
    state.environmentParameters.resize(2u);
    state.vascularState.clear();
    state.vascularClock.clear();
    state.rigidStates.resize(1u);
    state.rigidGeneralizedCandidate.resize(6u);
    state.reactions.resize(1u);
    state.contactHistories.resize(1u);
    state.deformableContactHistories.resize(3u);
    state.adaptive.resize(1u);
    state.schedulers.resize(1u);
    state.learnedWeights.resize(1u);
    state.identification.resize(1u);
    state.mpmActiveNodeIndices.resize(2u);
    state.mpmNodeToActive.resize(3u);
    state.mpmActiveNodeCounts.resize(1u);
    state.contactSamples.resize(1u);
    state.deformableContactFailures.resize(1u);
    return state;
}

void writeHeaderAndIdentity(std::ostream& output,
                            const RuntimeStateSnapshot& state) {
    using namespace shipping_snapshot_detail;
    output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    writePod(output, kVersion);
    writePod(output, kEndianMarker);
    writePod(output, static_cast<std::uint32_t>(sizeof(RuntimeStateSnapshot)));
    writePod(output, std::uint8_t{1u});
    writeString(output, state.message);
    writeScalarFields(output, state);
}

void testAcceptedRoundTripAndRejectedState() {
    const auto path = std::filesystem::temp_directory_path() /
        "numi-cardboard-shipping-snapshot-check.nmcb";
    auto state = acceptedState();
    writeShippingAcceptedSnapshot(path, state);
    const auto restored = readShippingAcceptedSnapshot(path,
        state.sourcePhysicsFingerprint, state.deviceProgramFingerprint, 4096u);
    check(restored.controlStep == state.controlStep &&
          restored.statuses.size() == 1u &&
          restored.statuses[0].code == NM_STATUS_SUCCESS,
          "accepted snapshot round trip changed cursor/status");

    state.statuses[0].code = NM_STATUS_NONLINEAR_SOLVER_FAILURE;
    bool rejected = false;
    try { writeShippingAcceptedSnapshot(path, state); }
    catch (const std::runtime_error&) { rejected = true; }
    check(rejected, "checkpoint writer accepted a failed native environment");
    std::filesystem::remove(path);
}

void testTruncatedAndHugeCountsRejectBeforeAllocation() {
    const auto path = std::filesystem::temp_directory_path() /
        "numi-cardboard-shipping-snapshot-malformed.nmcb";
    const auto state = acceptedState();
    for (const std::uint64_t count : {std::uint64_t{2u},
                                      std::numeric_limits<std::uint64_t>::max()}) {
        {
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            check(output.good(), "cannot create malformed snapshot fixture");
            writeHeaderAndIdentity(output, state);
            // The first vector in the frozen format is sutureProxyEdges. No
            // payload bytes follow this count; readVector must use the actual
            // remaining file size before allocating.
            shipping_snapshot_detail::writePod(output, count);
        }
        bool rejected = false;
        try {
            (void)readShippingAcceptedSnapshot(path,
                state.sourcePhysicsFingerprint, state.deviceProgramFingerprint, 4096u);
        } catch (const std::exception&) { rejected = true; }
        check(rejected, "truncated or oversized vector count was admitted");
    }
    std::filesystem::remove(path);
}

void testFingerprintRejectedBeforeMalformedVectors() {
    const auto path = std::filesystem::temp_directory_path() /
        "numi-cardboard-shipping-snapshot-identity.nmcb";
    const auto state = acceptedState();
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        check(output.good(), "cannot create identity-gate fixture");
        writeHeaderAndIdentity(output, state);
        shipping_snapshot_detail::writePod(output,
            std::numeric_limits<std::uint64_t>::max());
    }
    bool identityRejected = false;
    try {
        (void)readShippingAcceptedSnapshot(path,
            state.sourcePhysicsFingerprint ^ 1u,
            state.deviceProgramFingerprint, 4096u);
    } catch (const std::runtime_error& error) {
        identityRejected = std::string(error.what()).find("fingerprint") != std::string::npos;
    }
    check(identityRejected,
          "checkpoint parser consumed malformed vectors before checking fingerprint identity");
    std::filesystem::remove(path);
}

void testIncompleteShippingCheckpointRejectedBeforePublication() {
    const auto path = std::filesystem::temp_directory_path() /
        "numi-cardboard-shipping-snapshot-incomplete.nmcb";
    std::filesystem::remove(path);
    const auto world = shippingLayoutFixture();
    auto state = completeShippingState(world);
    writeShippingAcceptedSnapshot(path, state, world, 1u, true);
    check(std::filesystem::exists(path),
          "complete shipping snapshot was not published");
    std::filesystem::remove(path);

    state.femMaterialState.pop_back();
    bool rejected = false;
    try { writeShippingAcceptedSnapshot(path, state, world, 1u, true); }
    catch (const std::runtime_error&) { rejected = true; }
    check(rejected && !std::filesystem::exists(path),
          "incomplete material-history checkpoint was published as accepted");
}

void testHmaInputRequiresExplicitUncertaintyAndCitation() {
    std::istringstream valid(
        "schema=numi.cardboard.shipping-hma-mixed-mode-bilinear.v1\n"
        "material_id=explicit-analogue-transfer\n"
        "source_class=bounded_published_analogue_transfer\n"
        "source_citation=doi:10.example/not-a-corrugated-calibration\n"
        "transfer_uncertainty_fraction=0.5\n"
        "normal_penalty_pa_per_m=1e9\n"
        "shear_penalty_pa_per_m=1e9\n"
        "compression_penalty_pa_per_m=1e9\n"
        "normal_strength_pa=1e6\n"
        "shear_strength_pa=1e6\n"
        "fracture_energy_mode_i_j_per_m2=100\n"
        "fracture_energy_mode_ii_j_per_m2=200\n"
        "bk_exponent=1.5\n"
        "activation_gap_tolerance_m=1e-5\n");
    const auto law = parseShippingHmaLawParameters(valid, "synthetic_parser_fixture");
    check(law.sourceClass == "bounded_published_analogue_transfer" &&
          law.transferUncertaintyFraction == 0.5,
          "HMA parser dropped source-transfer provenance");
    std::istringstream missingUncertainty(
        "schema=numi.cardboard.shipping-hma-mixed-mode-bilinear.v1\n"
        "material_id=implicit-default\n"
        "source_class=unvalidated_engineering_assumption\n"
        "source_citation=none\n"
        "normal_penalty_pa_per_m=1e9\n"
        "shear_penalty_pa_per_m=1e9\n"
        "compression_penalty_pa_per_m=1e9\n"
        "normal_strength_pa=1e6\n"
        "shear_strength_pa=1e6\n"
        "fracture_energy_mode_i_j_per_m2=100\n"
        "fracture_energy_mode_ii_j_per_m2=200\n"
        "bk_exponent=1.5\n"
        "activation_gap_tolerance_m=1e-5\n");
    bool rejected = false;
    try {
        (void)parseShippingHmaLawParameters(missingUncertainty,
                                            "missing_uncertainty_fixture");
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "HMA parser silently supplied a missing transfer uncertainty");
}

} // namespace

int main() {
    try {
        testAcceptedRoundTripAndRejectedState();
        testTruncatedAndHugeCountsRejectBeforeAllocation();
        testFingerprintRejectedBeforeMalformedVectors();
        testIncompleteShippingCheckpointRejectedBeforePublication();
        testHmaInputRequiresExplicitUncertaintyAndCitation();
        std::cout << "shipping snapshot/provenance checks passed: 5\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "shipping snapshot checks failed: " << error.what() << '\n';
        return 1;
    }
}

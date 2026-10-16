#include "../candidate/matter/tools/cardboard_coupon_resume.hpp"

#include <cassert>
#include <sstream>

using namespace numi::cardboard::coupon_resume;

static Metadata validMetadata() {
    Metadata value;
    value.checkpointSha256 = std::string(64u, 'a');
    value.checkpointBytes = 4096u;
    value.sourcePhysicsFingerprint = 17u;
    value.deviceProgramFingerprint = 19u;
    value.protocolManifestSha256 = std::string(64u, 'b');
    value.runtimeEnvironmentSha256 = std::string(64u, 'f');
    value.prefixManifestSha256 = std::string(64u, 'c');
    value.prefixManifestFile = "coupon-control-000019.prefix";
    value.sourceRunDirectory = "/tmp/accepted-source-run";
    value.sourceRunId = "test-run-identity";
    value.environmentCount = 2u;
    value.nodeCount = 160u;
    value.tetrahedronCount = 384u;
    value.materialStateStride = 15u;
    value.acceptedControlIndex = 19u;
    value.nextControlIndex = 20u;
    value.physicsSubstep = 1u;
    value.sourcePrefixControlCount = 20u;
    value.timestepFloatBits = 0x38d1b717u;
    value.scheduledElapsedSecondsBits = 0x3fb999999999999au;
    value.phase = "loading";
    value.nextPhase = "loading";
    value.nextTargetAngleBits = 0x3fb999999999999au;
    return value;
}

static void mustReject(const auto& thunk) {
    bool rejected = false;
    try { thunk(); }
    catch (const std::exception&) { rejected = true; }
    assert(rejected);
}

int main() {
    const std::string environmentA = canonicalRuntimeEnvironment({
        {"NUMI_MATTER_MODE", "coupon"}, {"NM_FEM_CGLS", "64"},
        {"PATH", "/ignored"}});
    const std::string environmentB = canonicalRuntimeEnvironment({
        {"NM_FEM_CGLS", "64"}, {"PATH", "/different"},
        {"NUMI_MATTER_MODE", "coupon"}});
    assert(environmentA == environmentB);
    assert(environmentA != canonicalRuntimeEnvironment({
        {"NUMI_MATTER_MODE", "coupon"}, {"NM_FEM_CGLS", "32"}}));
    mustReject([&] {
        (void)canonicalRuntimeEnvironment({
            {"NM_FEM_CGLS", "64"}, {"NM_FEM_CGLS", "32"}});
    });

    const Metadata original = validMetadata();
    const std::string encoded = serializeMetadata(original);
    std::istringstream validInput(encoded);
    const Metadata decoded = parseMetadata(validInput, "valid-test");
    assert(decoded.checkpointSha256 == original.checkpointSha256);
    assert(decoded.protocolManifestSha256 == original.protocolManifestSha256);
    assert(decoded.nextControlIndex == 20u);
    assert(decoded.nextTargetAngleBits == original.nextTargetAngleBits);
    assert(decoded.scheduledElapsedSecondsBits == original.scheduledElapsedSecondsBits);
    validateRuntimeFingerprints(decoded, 17u, 19u);
    mustReject([&] { validateRuntimeFingerprints(decoded, 17u, 23u); });

    std::istringstream duplicate(encoded + "next_control_index=20\n");
    mustReject([&] { (void)parseMetadata(duplicate, "duplicate-test"); });
    std::string badHash = encoded;
    const auto hashPosition = badHash.find(std::string(64u, 'a'));
    badHash.replace(hashPosition, 64u, std::string(64u, 'A'));
    std::istringstream invalidHash(badHash);
    mustReject([&] { (void)parseMetadata(invalidHash, "hash-test"); });
    std::istringstream unknown(encoded + "unexpected=true\n");
    mustReject([&] { (void)parseMetadata(unknown, "unknown-test"); });
    std::string falseAcceptance = encoded;
    const auto acceptedPosition = falseAcceptance.find("both_environments_accepted=true");
    falseAcceptance.replace(acceptedPosition,
        std::string("both_environments_accepted=true").size(),
        "both_environments_accepted=false");
    std::istringstream rejectedState(falseAcceptance);
    mustReject([&] { (void)parseMetadata(rejectedState, "acceptance-test"); });
    std::string cursorMismatch = encoded;
    const auto cursorPosition = cursorMismatch.find("next_control_index=20");
    cursorMismatch.replace(cursorPosition,
        std::string("next_control_index=20").size(), "next_control_index=21");
    std::istringstream cursor(cursorMismatch);
    mustReject([&] { (void)parseMetadata(cursor, "cursor-test"); });
    mustReject([&] {
        Metadata invalid = original;
        invalid.prefixManifestFile = "../escape.prefix";
        (void)serializeMetadata(invalid);
    });

    PrefixManifest prefix;
    prefix.sourceRunDirectory = "/tmp/accepted-source-run";
    prefix.acceptedControlIndex = 19u;
    prefix.observationsPrefixBytes = 192u;
    prefix.observationsPrefixSha256 = std::string(64u, 'd');
    prefix.files = {
        {"manifest.json", 128u, std::string(64u, 'e'), false},
        {"observations.csv", 192u, prefix.observationsPrefixSha256, true},
    };
    const std::string prefixBytes = serializePrefixManifest(prefix);
    std::istringstream prefixInput(prefixBytes);
    const PrefixManifest prefixDecoded = parsePrefixManifest(prefixInput, "prefix-test");
    assert(prefixDecoded.files.size() == 2u);
    assert(prefixDecoded.acceptedControlIndex == 19u);
    assert(prefixDecoded.files[1].observationsPrefix);
    PrefixManifest inconsistentPrefix = prefix;
    ++inconsistentPrefix.observationsPrefixBytes;
    mustReject([&] { (void)serializePrefixManifest(inconsistentPrefix); });
    std::string traversal = prefixBytes;
    const auto manifestPosition = traversal.find("manifest.json");
    traversal.replace(manifestPosition, std::string("manifest.json").size(), "../manifest.json");
    std::istringstream traversalInput(traversal);
    mustReject([&] { (void)parsePrefixManifest(traversalInput, "traversal-test"); });
    std::istringstream missingObservation(
        "schema=numi.cardboard.accepted-coupon-prefix.v1\n"
        "source_run_directory=/tmp/run\naccepted_control_index=2\n"
        "observations_prefix_bytes=0\nobservations_prefix_sha256=" +
        std::string(64u, '0') + "\n");
    mustReject([&] { (void)parsePrefixManifest(missingObservation, "missing-prefix"); });

    Request request;
    request.saveControlRequested = true;
    request.saveControlIndex = 19u;
    request.environmentCount = 2u;
    request.totalControls = 64u;
    validateRequest(request);
    request.checkpointProvided = true;
    request.metadataProvided = false;
    mustReject([&] { validateRequest(request); });
    request.checkpointProvided = true;
    request.metadataProvided = true;
    mustReject([&] { validateRequest(request); });
    request.saveControlRequested = false;
    request.shipping = true;
    mustReject([&] { validateRequest(request); });
    request.shipping = false;
    request.environmentCount = 1u;
    mustReject([&] { validateRequest(request); });
    request.environmentCount = 2u;
    request.metadataProvided = false;
    mustReject([&] { validateRequest(request); });
    request.checkpointProvided = false;
    request.saveControlRequested = true;
    request.saveControlIndex = 63u;
    request.totalControls = 64u;
    mustReject([&] { validateRequest(request); });
    request.saveControlIndex = 62u;
    request.compileOnly = true;
    mustReject([&] { validateRequest(request); });

    assert(stopAfterAcceptedCheckpoint(true, 19u, 19u));
    assert(!stopAfterAcceptedCheckpoint(true, 19u, 18u));
    assert(!stopAfterAcceptedCheckpoint(false, 19u, 19u));
}

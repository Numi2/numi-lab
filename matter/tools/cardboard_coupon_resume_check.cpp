#include "cardboard_coupon_resume.hpp"

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
    const auto outputPaths = couponRuntimePathMap({
        {"NM_FEM_COUPLED_CAPTURE_PATH", "/run-a/capture"}});
    value.runtimeOutputPathMapHex = couponHexEncode(
        canonicalCouponOutputPathMap(outputPaths));
    value.runtimeOutputPathMapSha256 = std::string(64u, 'e');
    value.sourceRunnerCaseDirectory = "/tmp/accepted-source-case";
    value.sourceRunnerEnvironmentRecordSha256 = std::string(64u, '1');
    value.sourceRunnerNonOutputSettingsSha256 = std::string(64u, '4');
    value.sourceRunnerPredictionSha256 = std::string(64u, '2');
    value.sourceRunnerContractSha256 = std::string(64u, '3');
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
    assert(kRegisteredOutputPathKeys.size() == 7u);
    assert(kRegisteredDiagnosticReplaySelectorKeys.size() == 5u);
    const std::string environmentA = canonicalRuntimeEnvironment({
        {"NUMI_MATTER_MODE", "coupon"}, {"NM_FEM_CGLS", "64"},
        {"PATH", "/ignored"}});
    const std::string environmentB = canonicalRuntimeEnvironment({
        {"NM_FEM_CGLS", "64"}, {"PATH", "/different"},
        {"NUMI_MATTER_MODE", "coupon"}});
    assert(environmentA == environmentB);
    assert(environmentA != canonicalRuntimeEnvironment({
        {"NUMI_MATTER_MODE", "coupon"}, {"NM_FEM_CGLS", "32"}}));
    const auto nonPathA = canonicalCouponRuntimeSettings({
        {"NM_FEM_CGLS", "64"}, {"NM_FEM_COUPLED_CAPTURE_PATH", "/case-a"},
        {"NUMI_MATTER_MODE", "coupon"}});
    const auto nonPathB = canonicalCouponRuntimeSettings({
        {"NM_FEM_CGLS", "64"}, {"NM_FEM_COUPLED_CAPTURE_PATH", "/case-b"},
        {"NUMI_MATTER_MODE", "coupon"}});
    assert(nonPathA == nonPathB);
    assert(nonPathA != canonicalCouponRuntimeSettings({
        {"NM_FEM_CGLS", "32"}, {"NM_FEM_COUPLED_CAPTURE_PATH", "/case-b"},
        {"NUMI_MATTER_MODE", "coupon"}}));
    assert(nonPathA != canonicalCouponRuntimeSettings({
        {"NM_FEM_CGLS", "64"}, {"NM_FEM_COUPLED_CAPTURE_PATH", "/case-b"},
        {"NUMI_MATTER_MODE", ""}}));
    assert(canonicalCouponRuntimeSettings({{"NM_FEM_CGLS", "64"}}) !=
        canonicalCouponRuntimeSettings({
            {"NM_FEM_CGLS", "64"}, {"NUMI_MATTER_MODE", ""}}));
    assert(canonicalCouponRuntimeSettings({{"NM_FEM_CGLS", "64"}}) !=
        canonicalCouponRuntimeSettings({
            {"NM_FEM_CGLS", "64"}, {"NM_UNREGISTERED_CAPTURE_PATH", "/a"}}));
    mustReject([&] {
        (void)canonicalRuntimeEnvironment({
            {"NM_FEM_CGLS", "64"}, {"NM_FEM_CGLS", "32"}});
    });
    const RuntimePathMap sourcePaths = couponRuntimePathMap({
        {"NM_FEM_COUPLED_CAPTURE_PATH", "/source/capture"}});
    const RuntimePathMap relocatedPaths = couponRuntimePathMap({
        {"NM_FEM_COUPLED_CAPTURE_PATH", "/resume/capture"}});
    assert(!validateCouponOutputPathMapTransition(
        sourcePaths, sourcePaths, false, false));
    assert(validateCouponOutputPathMapTransition(
        sourcePaths, relocatedPaths, true, true));
    mustReject([&] { (void)validateCouponOutputPathMapTransition(
        sourcePaths, relocatedPaths, true, false); });
    const RuntimePathMap emptySource = couponRuntimePathMap({
        {"NM_FEM_COUPLED_CAPTURE_PATH", ""}});
    mustReject([&] { (void)validateCouponOutputPathMapTransition(
        emptySource, relocatedPaths, true, true); });
    const RuntimePathMap noOutputPaths = couponRuntimePathMap({});
    const RuntimePathMap replayOutputPaths = couponRuntimePathMap({
        {"NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH", "/resume/regional"}});
    mustReject([&] { (void)validateCouponOutputPathMapTransition(
        noOutputPaths, replayOutputPaths, true, true); });
    assert(validateCouponOutputPathMapTransition(
        noOutputPaths, replayOutputPaths, true, true, true));
    assert(validateCouponOutputPathMapTransition(
        replayOutputPaths, noOutputPaths, true, true, true));
    mustReject([&] { (void)validateCouponOutputPathMapTransition(
        noOutputPaths, replayOutputPaths, false, true, true); });
    DiagnosticRunnerProof sourceProof{
        "/source/case", std::string(64u, '1'), std::string(64u, '4'),
        std::string(64u, '2'), std::string(64u, '3'), sourcePaths};
    validateDiagnosticRunnerProof(sourceProof, "/source/case", sourcePaths);
    mustReject([&] {
        validateDiagnosticRunnerProof(sourceProof, "/other/case", sourcePaths);
    });
    mustReject([&] {
        validateDiagnosticRunnerProof(sourceProof, "/source/case", relocatedPaths);
    });
    DiagnosticRunnerProof currentNoPathProof{
        "/resume/case", std::string(64u, '5'), std::string(64u, '6'),
        std::string(64u, '7'), std::string(64u, '8'), noOutputPaths};
    validateDiagnosticRunnerProof(currentNoPathProof, "/resume/case",
                                  noOutputPaths);
    const bool currentRunnerPresent =
        couponOutputPathMapHasAnySetValue(currentNoPathProof.outputPaths);
    const bool diagnosticReplayEnabled = true;
    const bool currentRunnerValidated =
        currentRunnerPresent || diagnosticReplayEnabled;
    assert(!currentRunnerPresent && currentRunnerValidated);
    assert(couponValidatedRunnerDigest(currentRunnerValidated,
        currentNoPathProof.environmentRecordSha256) ==
        currentNoPathProof.environmentRecordSha256);
    assert(couponValidatedRunnerDigest(currentRunnerValidated,
        currentNoPathProof.nonOutputSettingsSha256) ==
        currentNoPathProof.nonOutputSettingsSha256);
    assert(couponValidatedRunnerDigest(currentRunnerValidated,
        currentNoPathProof.predictionSha256) ==
        currentNoPathProof.predictionSha256);
    assert(couponValidatedRunnerDigest(currentRunnerValidated,
        currentNoPathProof.contractSha256) ==
        currentNoPathProof.contractSha256);
    assert(couponCurrentRunnerProofMode(currentRunnerValidated, true,
        currentRunnerPresent) == "source_bound_diagnostic_replay_contract");
    assert(couponValidatedRunnerDigest(false,
        currentNoPathProof.environmentRecordSha256) == "none");
    assert(couponCurrentRunnerProofMode(false, false, false) == "not_validated");
    assert(couponSourceRunnerProofMode(true, true, false) ==
        "checkpoint_metadata_bound");
    assert(couponSourceRunnerProofMode(true, false, true) ==
        "source_run_reconstructed_for_diagnostic_replay");
    mustReject([&] { (void)couponValidatedRunnerDigest(true, "bad-digest"); });
    const std::vector<std::pair<std::string, std::string>> replaySource{
        {"NM_FEM_BASE_CGLS_CAPTURE_NEWTON", "12"},
        {"NM_FEM_BASE_CGLS_CAPTURE_PATH", "/source/cgls"},
        {"NM_FEM_BASE_CGLS_STATE_CAPTURE", "0"},
        {"NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH", "/source/regional"},
        {"NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR", "20:0:12:1:4:0"},
        {"NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT", "20"},
        {"NM_FEM_TRUE_RESIDUAL_ROOT", "19"},
        {"NM_FEM_BASE_CGLS_ITERATIONS", "64"},
    };
    const std::vector<std::pair<std::string, std::string>> replayCurrent{
        {"NM_FEM_BASE_CGLS_CAPTURE_NEWTON", "15"},
        {"NM_FEM_BASE_CGLS_CAPTURE_PATH", "/resume/cgls"},
        {"NM_FEM_BASE_CGLS_STATE_CAPTURE", "1"},
        {"NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH", "/resume/regional"},
        {"NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR", "21:0:15:1:7:1"},
        {"NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT", "21"},
        {"NM_FEM_TRUE_RESIDUAL_ROOT", "19"},
        {"NM_FEM_BASE_CGLS_ITERATIONS", "64"},
    };
    assert(couponDiagnosticReplaySettingsMatch(replaySource, replayCurrent));
    assert(canonicalCouponRuntimeSettings(replaySource) !=
           canonicalCouponRuntimeSettings(replayCurrent));
    assert(canonicalRuntimeEnvironment(replaySource) !=
           canonicalRuntimeEnvironment(replayCurrent));
    // Ordinary resume retains the exact runtime-settings digest check; only
    // explicit diagnostic replay may ignore these registered observer keys.
    assert(canonicalCouponRuntimeSettings(replaySource) !=
           canonicalCouponRuntimeSettings(replayCurrent));
    Request ordinaryResume;
    ordinaryResume.checkpointProvided = true;
    ordinaryResume.metadataProvided = true;
    ordinaryResume.environmentCount = 2u;
    validateRequest(ordinaryResume);
    Request replayWithoutCheckpoint;
    replayWithoutCheckpoint.diagnosticReplayRequested = true;
    mustReject([&] { validateRequest(replayWithoutCheckpoint); });
    auto changedSolverSetting = replayCurrent;
    changedSolverSetting.back().second = "128";
    assert(!couponDiagnosticReplaySettingsMatch(replaySource, changedSolverSetting));
    validateDiagnosticReplaySelectors(replayCurrent, 19u, 2u);
    mustReject([&] { validateDiagnosticReplaySelectors(replayCurrent, 21u, 2u); });
    auto wrongPathOrder = sourcePaths;
    std::swap(wrongPathOrder[0], wrongPathOrder[1]);
    mustReject([&] { (void)canonicalCouponOutputPathMap(wrongPathOrder); });
    mustReject([&] { (void)parseCouponOutputPathMap(
        "numi.cardboard.registered-diagnostic-output-path-map.v1\n"
        "NM_FEM_UNKNOWN_CAPTURE_PATH\tabsent\n"); });
    const std::string sourcePathBytes = canonicalCouponOutputPathMap(sourcePaths);
    assert(parseCouponOutputPathMap(sourcePathBytes) == sourcePaths);
    assert(couponHexDecode(couponHexEncode(sourcePathBytes)) == sourcePathBytes);
    assert(couponOutputPathMapHasAnySetValue(sourcePaths));
    assert(!couponOutputPathMapHasAnySetValue(couponRuntimePathMap({})));

    const Metadata original = validMetadata();
    const std::string encoded = serializeMetadata(original);
    std::istringstream validInput(encoded);
    const Metadata decoded = parseMetadata(validInput, "valid-test");
    assert(decoded.checkpointSha256 == original.checkpointSha256);
    assert(decoded.protocolManifestSha256 == original.protocolManifestSha256);
    assert(decoded.runtimeOutputPathMapHex == original.runtimeOutputPathMapHex);
    assert(decoded.sourceRunnerCaseDirectory == original.sourceRunnerCaseDirectory);
    assert(decoded.sourceRunnerPredictionSha256 == original.sourceRunnerPredictionSha256);
    assert(decoded.nextControlIndex == 20u);
    assert(decoded.nextTargetAngleBits == original.nextTargetAngleBits);
    assert(decoded.scheduledElapsedSecondsBits == original.scheduledElapsedSecondsBits);
    validateRuntimeFingerprints(decoded, 17u, 19u);
    mustReject([&] { validateRuntimeFingerprints(decoded, 17u, 23u); });
    mustReject([&] {
        Metadata missingRunnerReceipt = original;
        missingRunnerReceipt.sourceRunnerEnvironmentRecordSha256 = "none";
        missingRunnerReceipt.sourceRunnerNonOutputSettingsSha256 = "none";
        missingRunnerReceipt.sourceRunnerPredictionSha256 = "none";
        missingRunnerReceipt.sourceRunnerContractSha256 = "none";
        missingRunnerReceipt.sourceRunnerCaseDirectory = "none";
        (void)serializeMetadata(missingRunnerReceipt);
    });
    mustReject([&] {
        Metadata emptyRegisteredPath = original;
        const RuntimePathMap emptyPath = couponRuntimePathMap({
            {"NM_FEM_COUPLED_CAPTURE_PATH", ""}});
        emptyRegisteredPath.runtimeOutputPathMapHex = couponHexEncode(
            canonicalCouponOutputPathMap(emptyPath));
        (void)serializeMetadata(emptyRegisteredPath);
    });

    std::istringstream duplicate(encoded + "next_control_index=20\n");
    mustReject([&] { (void)parseMetadata(duplicate, "duplicate-test"); });
    std::string badHash = encoded;
    const auto hashPosition = badHash.find(std::string(64u, 'a'));
    badHash.replace(hashPosition, 64u, std::string(64u, 'A'));
    std::istringstream invalidHash(badHash);
    mustReject([&] { (void)parseMetadata(invalidHash, "hash-test"); });
    std::istringstream unknown(encoded + "unexpected=true\n");
    mustReject([&] { (void)parseMetadata(unknown, "unknown-test"); });
    std::string badPathMap = encoded;
    const auto pathMapPosition = badPathMap.find(original.runtimeOutputPathMapHex);
    assert(pathMapPosition != std::string::npos);
    badPathMap.replace(pathMapPosition, original.runtimeOutputPathMapHex.size(), "xyz");
    std::istringstream invalidPathMap(badPathMap);
    mustReject([&] { (void)parseMetadata(invalidPathMap, "path-map-test"); });
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

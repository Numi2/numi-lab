#define MRNX_EXACT_RUNTIME_LIFECYCLE_EMBEDDED 1
#include "numanx_exact_runtime_v3_lifecycle_probe.mm"

#include "metalrobo/NumiHumanProductionOwnerEvidenceWriter.hpp"
#include "metalrobo/NumiHumanProductionOwnerSnapshot.hpp"

#include <cerrno>
#include <charconv>
#include <filesystem>
#include <iomanip>
#include <map>
#include <limits>
#include <set>
#include <span>
#include <sstream>
#include <string_view>

#ifndef MRNX_SOURCE_GIT_REVISION
#define MRNX_SOURCE_GIT_REVISION "unknown"
#endif

namespace exact_runtime_horizon_probe {

struct Arguments {
    std::uint64_t roots = 0u;
    std::uint64_t timestepNanoseconds = 0u;
    float uniformExcitation = -1.0f;
    std::uint64_t snapshotStep = 0u;
    std::filesystem::path snapshotDirectory;
    std::vector<std::string> argv;
};

std::uint64_t parseUnsigned(
    const std::string_view text,
    const char* field
) {
    require(!text.empty(), field);
    std::uint64_t value = 0u;
    const auto result = std::from_chars(
        text.data(), text.data() + text.size(), value);
    require(
        result.ec == std::errc{} && result.ptr == text.data() + text.size(),
        field);
    return value;
}

float parseExcitation(const std::string_view text) {
    require(!text.empty(), "uniform excitation is empty");
    std::string owned(text);
    char* end = nullptr;
    errno = 0;
    const float value = std::strtof(owned.c_str(), &end);
    require(
        errno == 0 && end == owned.c_str() + owned.size() &&
            std::isfinite(value) && value >= 0.0f && value <= 1.0f,
        "uniform excitation is not a finite value in [0,1]");
    return value;
}

Arguments parseArguments(const int argc, char** argv) {
    Arguments result;
    require(argc >= 1 && argv != nullptr, "horizon argv is unavailable");
    result.argv.reserve(static_cast<std::size_t>(argc));
    for (int index = 0; index < argc; ++index) {
        require(argv[index] != nullptr, "horizon argv contains a null entry");
        result.argv.emplace_back(argv[index]);
    }
    bool sawRoots = false;
    bool sawTimestep = false;
    bool sawExcitation = false;
    bool sawSnapshotStep = false;
    bool sawSnapshotDirectory = false;
    for (int index = 1; index < argc; index += 2) {
        require(index + 1 < argc, "horizon option has no value");
        const std::string_view option(argv[index]);
        const std::string_view value(argv[index + 1]);
        if (option == "--roots") {
            require(!sawRoots, "--roots was supplied more than once");
            result.roots = parseUnsigned(value, "--roots is not an exact uint64");
            sawRoots = true;
        } else if (option == "--timestep-ns") {
            require(
                !sawTimestep,
                "--timestep-ns was supplied more than once");
            result.timestepNanoseconds = parseUnsigned(
                value, "--timestep-ns is not an exact uint64");
            sawTimestep = true;
        } else if (option == "--uniform-excitation") {
            require(
                !sawExcitation,
                "--uniform-excitation was supplied more than once");
            result.uniformExcitation = parseExcitation(value);
            sawExcitation = true;
        } else if (option == "--snapshot-step") {
            require(
                !sawSnapshotStep,
                "--snapshot-step was supplied more than once");
            result.snapshotStep = parseUnsigned(
                value, "--snapshot-step is not an exact uint64");
            sawSnapshotStep = true;
        } else if (option == "--snapshot-dir") {
            require(
                !sawSnapshotDirectory,
                "--snapshot-dir was supplied more than once");
            result.snapshotDirectory =
                std::filesystem::path(std::string(value));
            sawSnapshotDirectory = true;
        } else {
            throw std::runtime_error(
                "unknown horizon option: " + std::string(option));
        }
    }
    require(
        sawRoots && sawTimestep && sawExcitation && sawSnapshotStep &&
            sawSnapshotDirectory,
        "required options: --roots --timestep-ns --uniform-excitation "
        "--snapshot-step --snapshot-dir");
    require(result.roots != 0u, "--roots must be nonzero");
    require(
        result.roots <=
            (std::numeric_limits<std::uint64_t>::max() -
             exact_runtime_probe::kInitialTimestampNanoseconds) /
                exact_runtime_probe::kTimestepNanoseconds,
        "--roots overflows the exact runtime clock");
    require(
        result.timestepNanoseconds ==
            exact_runtime_probe::kTimestepNanoseconds,
        "--timestep-ns does not match the admitted exact runtime clock");
    require(
        result.snapshotStep != 0u &&
            result.snapshotStep <= result.roots,
        "--snapshot-step must be in [1, roots]");
    require(
        result.snapshotDirectory.is_absolute() &&
            result.snapshotDirectory.has_filename(),
        "--snapshot-dir must be an absolute non-root path");
    std::error_code error;
    const bool exists = std::filesystem::exists(
        result.snapshotDirectory, error);
    require(!error, "--snapshot-dir could not be inspected");
    require(
        !exists,
        "--snapshot-dir already exists; evidence publication is no-replace");
    const auto parent = result.snapshotDirectory.parent_path();
    require(
        !parent.empty() && std::filesystem::is_directory(parent, error) &&
            !error,
        "--snapshot-dir parent is not an existing directory");
    return result;
}

mrnx_runtime_config_v2 makeBaseConfig(id<MTLDevice> device) {
    mrnx_runtime_config_v2 base{};
    base.abi_version = MRNX_RUNTIME_CONFIG_ABI_V2;
    base.struct_size = sizeof(base);
    base.metal_device = (__bridge void*)device;
    base.rigid_payload_path = MRNX_FULLBODY_RIGID;
    base.muscle_payload_path = MRNX_FULLBODY_MUSCLE;
    base.support_contact_payload_path = MRNX_FULLBODY_SUPPORT_CONTACT;
    base.visual_pack_path = MRNX_FULLBODY_VISUAL_PACK;
    base.vision_profile_path = MRNX_FULLBODY_VISION_PROFILE;
    base.metalrobo_metallib_path = MRNX_METALROBO_METALLIB;
    base.matter_metallib_path = MRNX_MATTER_METALLIB;
    base.matter_material_path = MRNX_MATTER_MATERIAL;
    base.timestep_microseconds = 0u;
    base.maximum_retained_bytes = 1024ull * 1024ull * 1024ull;
    base.transaction_slot_count = 2u;
    return base;
}

constexpr std::string_view kManifestFilename =
    "numanx-exact-runtime-horizon-run-manifest.v1.json";

struct FileIdentity {
    std::filesystem::path configuredPath;
    std::filesystem::path canonicalPath;
    std::uint64_t byteCount = 0u;
    std::string sha256;
};

struct InputIdentity {
    std::string environment;
    FileIdentity file;
};

struct SnapshotIdentity {
    std::uint64_t root = 0u;
    std::uint64_t publicationEpoch = 0u;
    std::filesystem::path path;
    std::string transactionFingerprint;
    std::string payloadSHA256;
    FileIdentity file;
};

struct SnapshotSet {
    SnapshotIdentity rootOne;
    SnapshotIdentity selected;
    std::vector<SnapshotIdentity> unique;
};

std::string lowerHex(const std::span<const std::uint8_t> bytes) {
    constexpr std::string_view digits = "0123456789abcdef";
    std::string result;
    result.reserve(bytes.size() * 2u);
    for (const std::uint8_t byte : bytes) {
        result.push_back(digits[byte >> 4u]);
        result.push_back(digits[byte & 0x0fu]);
    }
    return result;
}

std::string hex64(const std::uint64_t value) {
    std::array<char, 17u> result{};
    const int length = std::snprintf(
        result.data(), result.size(), "%016llx",
        static_cast<unsigned long long>(value));
    require(length == 16, "uint64 hexadecimal formatting failed");
    return std::string(result.data(), 16u);
}

bool isLowerHex(const std::string_view text, const std::size_t length) {
    return text.size() == length &&
        std::all_of(text.begin(), text.end(), [](const char value) {
            return (value >= '0' && value <= '9') ||
                (value >= 'a' && value <= 'f');
        });
}

FileIdentity identifyFile(const std::filesystem::path& path) {
    require(!path.empty(), "evidence identity path is empty");
    std::error_code error;
    require(
        std::filesystem::is_regular_file(path, error) && !error,
        "evidence identity path is not a regular file");
    const auto canonical = std::filesystem::canonical(path, error);
    require(!error, "evidence identity path could not be canonicalized");
    const auto bytes = readPayloadBytes(path.c_str());
    require(
        bytes.size() <= std::numeric_limits<CC_LONG>::max(),
        "evidence identity path exceeds the SHA-256 input domain");
    std::array<std::uint8_t, CC_SHA256_DIGEST_LENGTH> digest{};
    require(
        CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()),
            digest.data()) != nullptr,
        "evidence identity SHA-256 failed");
    return {
        path,
        canonical,
        static_cast<std::uint64_t>(bytes.size()),
        lowerHex(digest)};
}

std::vector<InputIdentity> identifyExactInputs() {
    constexpr std::array<const char*, 5u> names{
        "MRNX_EXACT_WORLD",
        "MRNX_EXACT_INITIAL_STATE",
        "MRNX_EXACT_SUPPORT_CONTACT",
        "MRNX_EXACT_JOINT_EQUALITIES",
        "MRNX_EXACT_JOINT_LIMITS"};
    std::vector<InputIdentity> result;
    result.reserve(names.size());
    for (const char* name : names) {
        const char* value = std::getenv(name);
        require(
            value != nullptr && value[0] != '\0',
            "one exact-runtime input environment path is absent");
        result.push_back({name, identifyFile(value)});
    }
    require(result.size() == 5u, "exact-runtime input identity set is not five files");
    return result;
}

void requireSameInputs(
    const std::vector<InputIdentity>& before,
    const std::vector<InputIdentity>& after
) {
    require(
        before.size() == 5u && after.size() == before.size(),
        "exact-runtime input identity set changed cardinality");
    for (std::size_t index = 0u; index < before.size(); ++index) {
        require(
            before[index].environment == after[index].environment &&
                before[index].file.configuredPath ==
                    after[index].file.configuredPath &&
                before[index].file.canonicalPath ==
                    after[index].file.canonicalPath &&
                before[index].file.byteCount == after[index].file.byteCount &&
                before[index].file.sha256 == after[index].file.sha256,
            "an exact-runtime input changed during sustained execution");
    }
}

std::string utf8String(NSString* value, const char* message) {
    require(value != nil, message);
    const char* raw = value.UTF8String;
    require(raw != nullptr, message);
    return raw;
}

std::uint64_t exactJSONUnsigned(id value, const char* message) {
    require([value isKindOfClass:[NSNumber class]], message);
    NSNumber* number = (NSNumber*)value;
    const double floating = number.doubleValue;
    const std::uint64_t result = number.unsignedLongLongValue;
    require(
        std::isfinite(floating) && floating >= 0.0 &&
            floating == static_cast<double>(result),
        message);
    return result;
}

SnapshotIdentity inspectSnapshot(const std::filesystem::path& path) {
    @autoreleasepool {
        const auto bytes = readPayloadBytes(path.c_str());
        NSData* data = [NSData dataWithBytes:bytes.data() length:bytes.size()];
        NSError* jsonError = nil;
        id document = [NSJSONSerialization JSONObjectWithData:data
            options:0 error:&jsonError];
        require(
            jsonError == nil &&
                [document isKindOfClass:[NSDictionary class]],
            "production-owner snapshot is not one JSON object");
        NSDictionary* envelope = (NSDictionary*)document;
        id evidenceSchema = envelope[@"evidence_schema"];
        require(
            [evidenceSchema isKindOfClass:[NSString class]] &&
                [(NSString*)evidenceSchema isEqualToString:
                    @"persistent-production-owner-snapshot-evidence.v2"],
            "production-owner snapshot evidence schema is not canonical");
        id payloadValue = envelope[@"payload"];
        require(
            [payloadValue isKindOfClass:[NSDictionary class]],
            "production-owner snapshot payload is absent");
        NSDictionary* payload = (NSDictionary*)payloadValue;
        id payloadSchema = payload[@"schema"];
        require(
            [payloadSchema isKindOfClass:[NSString class]] &&
                [(NSString*)payloadSchema isEqualToString:
                    @"persistent-production-owner-snapshot.v2"],
            "production-owner snapshot payload schema is not canonical");
        require(
            exactJSONUnsigned(payload[@"format_version"],
                "production-owner snapshot payload version is not an exact uint64") ==
                metalrobo::kNumiHumanProductionOwnerSnapshotVersionV2,
            "production-owner snapshot payload version is not V2");
        id disposition = payload[@"disposition"];
        require(
            [disposition isKindOfClass:[NSString class]] &&
                [(NSString*)disposition isEqualToString:@"published"],
            "production-owner horizon snapshot is not published");
        const std::uint64_t root = exactJSONUnsigned(
            payload[@"control_step"],
            "production-owner snapshot control_step is not an exact uint64");
        const std::uint64_t publicationEpoch = exactJSONUnsigned(
            payload[@"publication_epoch"],
            "production-owner snapshot publication_epoch is not an exact uint64");
        require(
            root != 0u && publicationEpoch == root,
            "production-owner snapshot root/publication epoch is noncanonical");
        id transaction = payload[@"transaction_fingerprint"];
        require(
            [transaction isKindOfClass:[NSString class]],
            "production-owner snapshot transaction fingerprint is absent");
        const std::string transactionFingerprint = utf8String(
            (NSString*)transaction,
            "production-owner snapshot transaction fingerprint is not UTF-8");
        require(
            isLowerHex(transactionFingerprint, 16u),
            "production-owner snapshot transaction fingerprint is not canonical");
        id payloadDigest = envelope[@"payload_sha256"];
        require(
            [payloadDigest isKindOfClass:[NSString class]],
            "production-owner snapshot payload SHA-256 is absent");
        const std::string payloadSHA256 = utf8String(
            (NSString*)payloadDigest,
            "production-owner snapshot payload SHA-256 is not UTF-8");
        require(
            isLowerHex(payloadSHA256, 64u),
            "production-owner snapshot payload SHA-256 is not canonical");
        const std::string expectedName =
            "persistent-production-owner-snapshot.v2.root-" +
            std::to_string(root) + "." + transactionFingerprint +
            ".published.json";
        require(
            path.filename() == expectedName,
            "production-owner snapshot filename does not match its payload identity");
        return {
            root,
            publicationEpoch,
            path,
            transactionFingerprint,
            payloadSHA256,
            identifyFile(path)};
    }
}

SnapshotSet inspectSnapshots(const Arguments& arguments) {
    std::error_code error;
    require(
        std::filesystem::is_directory(arguments.snapshotDirectory, error) &&
            !error,
        "production-owner evidence directory was not created");
    std::set<std::uint64_t> expectedRoots{1u, arguments.snapshotStep};
    std::map<std::uint64_t, SnapshotIdentity> byRoot;
    for (std::filesystem::directory_iterator iterator(
             arguments.snapshotDirectory, error), end;
         !error && iterator != end; iterator.increment(error)) {
        const auto status = iterator->symlink_status(error);
        require(!error, "production-owner evidence status failed");
        require(
            std::filesystem::is_regular_file(status),
            "production-owner evidence directory contains a non-regular file");
        auto snapshot = inspectSnapshot(iterator->path());
        require(
            expectedRoots.contains(snapshot.root),
            "production-owner evidence contains a noncanonical root");
        require(
            byRoot.emplace(snapshot.root, snapshot).second,
            "production-owner evidence contains a duplicate canonical root");
    }
    require(!error, "production-owner evidence enumeration failed");
    require(
        byRoot.size() == expectedRoots.size(),
        "production-owner evidence is not exactly the canonical root set");
    for (const std::uint64_t root : expectedRoots) {
        require(
            byRoot.contains(root),
            "production-owner evidence omits a canonical root");
    }
    SnapshotSet result;
    result.rootOne = byRoot.at(1u);
    result.selected = byRoot.at(arguments.snapshotStep);
    result.unique.reserve(byRoot.size());
    for (auto& [root, snapshot] : byRoot) {
        (void)root;
        result.unique.push_back(std::move(snapshot));
    }
    return result;
}

std::string jsonString(const std::string_view text) {
    constexpr std::string_view digits = "0123456789abcdef";
    std::string result;
    result.reserve(text.size() + 2u);
    result.push_back('"');
    for (const char raw : text) {
        const auto value = static_cast<unsigned char>(raw);
        switch (value) {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\b': result += "\\b"; break;
        case '\f': result += "\\f"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (value < 0x20u || value >= 0x7fu) {
                result += "\\u00";
                result.push_back(digits[value >> 4u]);
                result.push_back(digits[value & 0x0fu]);
            } else {
                result.push_back(static_cast<char>(value));
            }
        }
    }
    result.push_back('"');
    return result;
}

void appendFileIdentityJSON(
    std::ostringstream& output,
    const FileIdentity& identity
) {
    output << "{\"configured_path\":"
           << jsonString(identity.configuredPath.string())
           << ",\"canonical_path\":"
           << jsonString(identity.canonicalPath.string())
           << ",\"byte_count\":" << identity.byteCount
           << ",\"sha256\":" << jsonString(identity.sha256) << '}';
}

void appendSnapshotJSON(
    std::ostringstream& output,
    const SnapshotIdentity& snapshot
) {
    output << "{\"control_step\":" << snapshot.root
           << ",\"publication_epoch\":" << snapshot.publicationEpoch
           << ",\"filename\":"
           << jsonString(snapshot.path.filename().string())
           << ",\"transaction_fingerprint\":"
           << jsonString(snapshot.transactionFingerprint)
           << ",\"payload_sha256\":"
           << jsonString(snapshot.payloadSHA256)
           << ",\"file_sha256\":"
           << jsonString(snapshot.file.sha256)
           << ",\"file_byte_count\":" << snapshot.file.byteCount << '}';
}

std::string makeManifest(
    const Arguments& arguments,
    const std::vector<InputIdentity>& inputs,
    const std::string_view deviceName,
    const std::uint64_t deviceRegistryID,
    const mrnx_runtime_info_v1& runtimeInfo,
    const mrnx_runtime_world_info_v1& worldInfo,
    const mrnx_exact_clock_info_v1& clockInfo,
    const exact_runtime_probe::RootOutcome& outcome,
    const SnapshotSet& snapshots
) {
    require(inputs.size() == 5u, "run manifest requires exactly five inputs");
    require(
        snapshots.unique.size() ==
            (arguments.snapshotStep == 1u ? 1u : 2u),
        "run manifest snapshot cardinality is noncanonical");
    std::ostringstream output;
    output << "{\"schema\":\"numanx.exact-runtime-horizon-run-manifest.v1\""
           << ",\"claims\":{\"scope\":\"sustained-execution-only\","
              "\"audit_required\":true,\"full_behavior\":false}"
           << ",\"argv_encoding\":\"bytewise-u00xx\",\"argv\":[";
    for (std::size_t index = 0u; index < arguments.argv.size(); ++index) {
        if (index != 0u) output << ',';
        output << jsonString(arguments.argv[index]);
    }
    output << "]"
           << ",\"source_git_revision\":"
           << jsonString(MRNX_SOURCE_GIT_REVISION)
           << ",\"request\":{\"roots\":" << arguments.roots
           << ",\"timestep_nanoseconds\":"
           << arguments.timestepNanoseconds
           << ",\"uniform_excitation\":"
           << std::setprecision(std::numeric_limits<float>::max_digits10)
           << arguments.uniformExcitation
           << ",\"snapshot_step\":" << arguments.snapshotStep
           << ",\"snapshot_directory\":"
           << jsonString(arguments.snapshotDirectory.string()) << '}'
           << ",\"device\":{\"name\":" << jsonString(deviceName)
           << ",\"registry_id\":" << jsonString(hex64(deviceRegistryID))
           << "}"
           << ",\"runtime\":{\"abi_version\":"
           << runtimeInfo.abi_version
           << ",\"status\":" << runtimeInfo.status
           << ",\"body_count\":" << runtimeInfo.body_count
           << ",\"q_coordinate_count\":"
           << runtimeInfo.q_coordinate_count
           << ",\"dof_count\":" << runtimeInfo.dof_count
           << ",\"muscle_count\":" << runtimeInfo.muscle_count
           << ",\"transaction_slot_count\":"
           << runtimeInfo.transaction_slot_count
           << ",\"resident_continuation_count\":"
           << runtimeInfo.resident_continuation_count
           << ",\"accepted_state_proof_program_fingerprint\":"
           << jsonString(hex64(
                  runtimeInfo.accepted_state_proof_program_fingerprint))
           << ",\"model_source_fingerprint\":"
           << jsonString(hex64(runtimeInfo.model_source_fingerprint)) << '}'
           << ",\"world\":{\"authored_package\":"
           << worldInfo.authored_package
           << ",\"object_count\":" << worldInfo.object_count
           << ",\"fem_node_count\":" << worldInfo.fem_node_count
           << ",\"fem_attachment_count\":"
           << worldInfo.fem_attachment_count
           << ",\"world_fingerprint\":"
           << jsonString(hex64(worldInfo.world_fingerprint))
           << ",\"physics_fingerprint\":"
           << jsonString(hex64(worldInfo.physics_fingerprint)) << '}'
           << ",\"clock\":{\"timestep_nanoseconds\":"
           << clockInfo.timestep_nanoseconds
           << ",\"clock_quantum_nanoseconds\":"
           << clockInfo.clock_quantum_nanoseconds
           << ",\"published_timestamp_nanoseconds\":"
           << clockInfo.published_timestamp_nanoseconds
           << ",\"publication_epoch\":"
           << clockInfo.publication_epoch << '}'
           << ",\"terminal_identity\":{\"root_program_fingerprint\":"
           << jsonString(hex64(outcome.aggregate.root.program_fingerprint))
           << ",\"transaction_fingerprint\":"
           << jsonString(hex64(
                  outcome.aggregate.root.transaction_fingerprint))
           << ",\"candidate_publication_fingerprint\":"
           << jsonString(hex64(
                  outcome.publication.candidate_publication_fingerprint))
           << ",\"accepted_physics_token_fingerprint\":"
           << jsonString(hex64(
                  outcome.publication.accepted_physics_token_fingerprint))
           << ",\"joint_commit_fingerprint\":"
           << jsonString(hex64(
                  outcome.publication.joint_commit_fingerprint))
           << ",\"publication_fingerprint\":"
           << jsonString(hex64(
                  outcome.publication.publication_fingerprint)) << '}'
           << ",\"inputs\":[";
    for (std::size_t index = 0u; index < inputs.size(); ++index) {
        if (index != 0u) output << ',';
        output << "{\"environment\":"
               << jsonString(inputs[index].environment) << ",\"file\":";
        appendFileIdentityJSON(output, inputs[index].file);
        output << '}';
    }
    output << "]"
           << ",\"snapshots\":{\"unique_count\":"
           << snapshots.unique.size()
           << ",\"roots_distinct\":"
           << (arguments.snapshotStep == 1u ? "false" : "true")
           << ",\"root1\":";
    appendSnapshotJSON(output, snapshots.rootOne);
    output << ",\"selected_root\":";
    appendSnapshotJSON(output, snapshots.selected);
    output << "}}\n";
    return output.str();
}

std::filesystem::path publishManifest(
    const Arguments& arguments,
    const std::string& contents
) {
    const auto path = arguments.snapshotDirectory /
        std::string(kManifestFilename);
    std::string error;
    if (!metalrobo::publishNumiHumanProductionOwnerEvidenceNoReplace(
            path, contents, error)) {
        throw std::runtime_error(
            "exact-runtime run manifest publication failed: " + error);
    }
    return path;
}

void validateFinalEvidenceDirectory(
    const Arguments& arguments,
    const SnapshotSet& snapshots,
    const std::filesystem::path& manifestPath,
    const std::string& manifestContents
) {
    std::set<std::string> expectedNames{manifestPath.filename().string()};
    for (const auto& snapshot : snapshots.unique)
        expectedNames.insert(snapshot.path.filename().string());
    std::error_code error;
    std::size_t count = 0u;
    for (std::filesystem::directory_iterator iterator(
             arguments.snapshotDirectory, error), end;
         !error && iterator != end; iterator.increment(error)) {
        const auto status = iterator->symlink_status(error);
        require(!error, "final evidence status failed");
        require(
            std::filesystem::is_regular_file(status),
            "final evidence directory contains a non-regular file");
        require(
            expectedNames.contains(iterator->path().filename().string()),
            "final evidence directory contains an unexpected file");
        ++count;
    }
    require(!error, "final evidence enumeration failed");
    require(
        count == expectedNames.size(),
        "final evidence directory is not exactly snapshots plus manifest");
    for (const auto& snapshot : snapshots.unique) {
        const auto current = identifyFile(snapshot.path);
        require(
            current.byteCount == snapshot.file.byteCount &&
                current.sha256 == snapshot.file.sha256,
            "production-owner snapshot changed during manifest publication");
    }
    const auto manifestBytes = readPayloadBytes(manifestPath.c_str());
    require(
        manifestBytes.size() == manifestContents.size() &&
            std::memcmp(
                manifestBytes.data(), manifestContents.data(),
                manifestContents.size()) == 0,
        "published exact-runtime run manifest bytes changed");
}

int run(const Arguments& arguments) {
    @autoreleasepool {
        const auto inputIdentities = identifyExactInputs();
        require(
            ::setenv(
                "MRNX_PRODUCTION_OWNER_SNAPSHOT_PATH",
                arguments.snapshotDirectory.c_str(), 1) == 0,
            "failed to set production-owner snapshot path");
        const std::string selectedStep =
            std::to_string(arguments.snapshotStep);
        require(
            ::setenv(
                "MRNX_PRODUCTION_OWNER_SNAPSHOT_CONTROL_STEP",
                selectedStep.c_str(), 1) == 0,
            "failed to set production-owner selected control step");

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal device unavailable");
        const std::string deviceName = utf8String(
            device.name, "Metal device name is unavailable");
        mrnx_runtime_info_v1 info{};
        mrnx_runtime_info_v1 terminalInfo{};
        mrnx_runtime_world_info_v1 worldInfo{};
        mrnx_exact_clock_info_v1 clock{};
        mrnx_runtime_v1* runtime = nullptr;
        std::uint64_t activeRoot = 0u;
        exact_runtime_probe::RootOutcome outcome{};
        try {
            runtime = makeExactRuntime(makeBaseConfig(device), info);
            require(
                runtime != nullptr && info.status == MRNX_RUNTIME_READY_V1 &&
                    info.device_registry_id == device.registryID &&
                    info.accepted_state_proof_program_fingerprint != 0u,
                "exact runtime v8 construction failed");
            worldInfo.abi_version = MRNX_BRIDGE_ABI_V1;
            worldInfo.struct_size = sizeof(worldInfo);
            require(
                mrnx_bridge_v1_runtime_copy_world_info(runtime, &worldInfo) &&
                    worldInfo.authored_package == 1u &&
                    worldInfo.world_fingerprint != 0u &&
                    worldInfo.physics_fingerprint != 0u,
                "exact runtime world identity is unavailable");
            std::uint64_t brainGeneration = 0u;
            std::uint64_t physicsGeneration = 0u;
            std::uint64_t timestampNanoseconds =
                exact_runtime_probe::kInitialTimestampNanoseconds;
            for (std::uint64_t root = 1u; root <= arguments.roots; ++root) {
                activeRoot = root;
                outcome = exact_runtime_probe::executeAcceptedRoot(
                    runtime, device, root, brainGeneration,
                    physicsGeneration, timestampNanoseconds, root == 1u,
                    arguments.uniformExcitation);
                require(
                    outcome.aggregate.publication_epoch == root &&
                        outcome.aggregate.brain_generation == root &&
                        outcome.aggregate.physics_generation == root &&
                        outcome.aggregate.sensor_generation == root &&
                        outcome.aggregate.root.control_step == root,
                    "sustained exact runtime generation sequence diverged");
                brainGeneration = outcome.aggregate.brain_generation;
                physicsGeneration = outcome.aggregate.physics_generation;
                timestampNanoseconds =
                    outcome.publication.committed_timestamp_nanoseconds;
            }
            clock.abi_version = MRNX_EXACT_CLOCK_INFO_ABI_V1;
            clock.struct_size = sizeof(clock);
            const std::uint64_t expectedTimestamp =
                exact_runtime_probe::kInitialTimestampNanoseconds +
                arguments.roots * arguments.timestepNanoseconds;
            require(
                mrnx_bridge_v1_runtime_copy_exact_clock(runtime, &clock) &&
                    clock.timestep_nanoseconds ==
                        arguments.timestepNanoseconds &&
                    clock.clock_quantum_nanoseconds == 1u &&
                    clock.published_timestamp_nanoseconds ==
                        expectedTimestamp &&
                    clock.publication_epoch == arguments.roots,
                "sustained exact clock terminal state is inconsistent");
            terminalInfo.abi_version = MRNX_BRIDGE_ABI_V1;
            terminalInfo.struct_size = sizeof(terminalInfo);
            require(
                mrnx_bridge_v1_runtime_copy_info(runtime, &terminalInfo) &&
                    terminalInfo.status == MRNX_RUNTIME_READY_V1 &&
                    terminalInfo.device_registry_id == device.registryID &&
                    terminalInfo.accepted_state_proof_program_fingerprint ==
                        info.accepted_state_proof_program_fingerprint &&
                    terminalInfo.model_source_fingerprint ==
                        info.model_source_fingerprint,
                "exact runtime terminal identity changed");
            mrnx_bridge_v1_runtime_drop(runtime);
            runtime = nullptr;
        } catch (const std::exception& error) {
            if (runtime != nullptr) mrnx_bridge_v1_runtime_drop(runtime);
            throw std::runtime_error(
                "root " + std::to_string(activeRoot) + ": " + error.what());
        } catch (...) {
            if (runtime != nullptr) mrnx_bridge_v1_runtime_drop(runtime);
            throw;
        }

        const auto finalInputIdentities = identifyExactInputs();
        requireSameInputs(inputIdentities, finalInputIdentities);
        const auto snapshots = inspectSnapshots(arguments);
        const std::string manifestContents = makeManifest(
            arguments, inputIdentities, deviceName, device.registryID,
            terminalInfo, worldInfo, clock, outcome, snapshots);
        const auto manifestPath = publishManifest(
            arguments, manifestContents);
        validateFinalEvidenceDirectory(
            arguments, snapshots, manifestPath, manifestContents);
        std::printf(
            "numanx_exact_runtime_horizon_probe=sustained-execution-only "
            "roots=%llu uniform_excitation=%.9g snapshot_step=%llu "
            "snapshot_files=%zu final_timestamp_ns=%llu "
            "audit_required=true full_behavior=false\n",
            static_cast<unsigned long long>(arguments.roots),
            static_cast<double>(arguments.uniformExcitation),
            static_cast<unsigned long long>(arguments.snapshotStep),
            snapshots.unique.size(),
            static_cast<unsigned long long>(
                outcome.publication.committed_timestamp_nanoseconds));
        std::printf(
            "production_owner_root1_snapshot=%s sha256=%s\n",
            snapshots.rootOne.path.c_str(),
            snapshots.rootOne.file.sha256.c_str());
        std::printf(
            "production_owner_selected_root_snapshot=%s sha256=%s\n",
            snapshots.selected.path.c_str(),
            snapshots.selected.file.sha256.c_str());
        std::printf("exact_runtime_run_manifest=%s\n", manifestPath.c_str());
        return 0;
    }
}

class TemporaryDirectory final {
public:
    TemporaryDirectory() {
        std::array<char, 160u> pattern{};
        const int written = std::snprintf(
            pattern.data(), pattern.size(),
            "/private/tmp/numanx-horizon-self-test.%ld.XXXXXX",
            static_cast<long>(::getpid()));
        require(
            written > 0 &&
                static_cast<std::size_t>(written) < pattern.size(),
            "horizon self-test temporary path construction failed");
        char* created = ::mkdtemp(pattern.data());
        require(
            created != nullptr,
            "horizon self-test temporary directory creation failed");
        path_ = created;
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

Arguments parseTestArguments(std::vector<std::string> values) {
    std::vector<char*> pointers;
    pointers.reserve(values.size());
    for (auto& value : values) pointers.push_back(value.data());
    return parseArguments(
        static_cast<int>(pointers.size()), pointers.data());
}

std::vector<std::string> testArguments(
    const std::filesystem::path& snapshotDirectory,
    const std::string& roots = "2",
    const std::string& snapshotStep = "2"
) {
    return {
        "numanx-horizon-self-test",
        "--roots", roots,
        "--timestep-ns", "12500",
        "--uniform-excitation", "0.25",
        "--snapshot-step", snapshotStep,
        "--snapshot-dir", snapshotDirectory.string()};
}

template <typename Function>
void expectSelfTestFailure(
    Function&& function,
    const std::string_view expected
) {
    try {
        function();
    } catch (const std::exception& error) {
        require(
            std::string_view(error.what()).find(expected) !=
                std::string_view::npos,
            "horizon self-test failed with an unexpected diagnostic");
        return;
    }
    throw std::runtime_error(
        "horizon self-test unexpectedly accepted an invalid case");
}

std::string selfTestSnapshotJSON(
    const std::uint64_t root,
    const std::string_view transaction
) {
    return
        "{\"evidence_schema\":\"persistent-production-owner-snapshot-evidence.v2\","
        "\"payload_sha256\":\"" + std::string(64u, '0') +
        "\",\"payload\":{\"schema\":\"persistent-production-owner-snapshot.v2\","
        "\"format_version\":2,"
        "\"disposition\":\"published\",\"control_step\":" +
        std::to_string(root) + ",\"publication_epoch\":" +
        std::to_string(root) + ",\"transaction_fingerprint\":\"" +
        std::string(transaction) + "\"}}\n";
}

void publishSelfTestSnapshot(
    const std::filesystem::path& directory,
    const std::uint64_t root
) {
    const std::string transaction = hex64(root);
    const auto path = directory /
        ("persistent-production-owner-snapshot.v2.root-" +
         std::to_string(root) + "." + transaction + ".published.json");
    std::string error;
    require(
        metalrobo::publishNumiHumanProductionOwnerEvidenceNoReplace(
            path, selfTestSnapshotJSON(root, transaction), error),
        "horizon self-test snapshot publication failed");
}

int runSelfTests() {
    TemporaryDirectory temporary;
    const auto validDirectory = temporary.path() / "valid-evidence";
    const auto valid = parseTestArguments(testArguments(validDirectory));
    require(
        valid.roots == 2u && valid.snapshotStep == 2u &&
            valid.argv.size() == 11u,
        "horizon valid parser self-test failed");

    const std::uint64_t maximumRoots =
        (std::numeric_limits<std::uint64_t>::max() -
         exact_runtime_probe::kInitialTimestampNanoseconds) /
        exact_runtime_probe::kTimestepNanoseconds;
    expectSelfTestFailure(
        [&] {
            (void)parseTestArguments(testArguments(
                temporary.path() / "overflow-evidence",
                std::to_string(maximumRoots + 1u)));
        },
        "--roots overflows the exact runtime clock");
    expectSelfTestFailure(
        [&] {
            (void)parseTestArguments(testArguments(
                temporary.path() / "malformed-evidence", "2tail"));
        },
        "--roots is not an exact uint64");

    const auto existingDirectory = temporary.path() / "existing-evidence";
    std::error_code filesystemError;
    require(
        std::filesystem::create_directory(
            existingDirectory, filesystemError) && !filesystemError,
        "horizon self-test existing directory setup failed");
    expectSelfTestFailure(
        [&] {
            (void)parseTestArguments(testArguments(existingDirectory));
        },
        "evidence publication is no-replace");

    const auto writerTarget = temporary.path() / "writer" /
        std::string(kManifestFilename);
    const std::string first = "{\"manifest\":\"first\"}\n";
    const std::string replacement = "{\"manifest\":\"replacement\"}\n";
    std::string writerError;
    require(
        metalrobo::publishNumiHumanProductionOwnerEvidenceNoReplace(
            writerTarget, first, writerError),
        "horizon self-test first no-replace publication failed");
    require(
        !metalrobo::publishNumiHumanProductionOwnerEvidenceNoReplace(
            writerTarget, replacement, writerError) &&
            writerError == "production-owner evidence target already exists",
        "horizon self-test no-replace publication was not rejected");
    const auto retained = readPayloadBytes(writerTarget.c_str());
    require(
        retained.size() == first.size() &&
            std::memcmp(retained.data(), first.data(), first.size()) == 0,
        "horizon self-test no-replace publication changed retained bytes");

    const auto snapshotDirectory = temporary.path() / "snapshot-set";
    publishSelfTestSnapshot(snapshotDirectory, 1u);
    publishSelfTestSnapshot(snapshotDirectory, 2u);
    Arguments inspection;
    inspection.snapshotDirectory = snapshotDirectory;
    inspection.snapshotStep = 2u;
    const auto snapshots = inspectSnapshots(inspection);
    require(
        snapshots.unique.size() == 2u && snapshots.rootOne.root == 1u &&
            snapshots.selected.root == 2u,
        "horizon canonical snapshot-set self-test failed");

    const auto fractionalVersionDirectory =
        temporary.path() / "fractional-version-snapshot";
    const std::string fractionalTransaction = hex64(1u);
    const auto fractionalVersionPath = fractionalVersionDirectory /
        ("persistent-production-owner-snapshot.v2.root-1." +
         fractionalTransaction + ".published.json");
    std::string fractionalVersionJSON =
        selfTestSnapshotJSON(1u, fractionalTransaction);
    const auto versionOffset = fractionalVersionJSON.find(
        "\"format_version\":2,");
    require(versionOffset != std::string::npos,
        "horizon fractional-version self-test fixture changed");
    fractionalVersionJSON.replace(
        versionOffset, std::string_view("\"format_version\":2").size(),
        "\"format_version\":2.5");
    std::string fractionalVersionError;
    require(
        metalrobo::publishNumiHumanProductionOwnerEvidenceNoReplace(
            fractionalVersionPath, fractionalVersionJSON,
            fractionalVersionError),
        "horizon fractional-version self-test publication failed");
    expectSelfTestFailure(
        [&] { (void)inspectSnapshot(fractionalVersionPath); },
        "payload version is not an exact uint64");

    publishSelfTestSnapshot(snapshotDirectory, 3u);
    expectSelfTestFailure(
        [&] { (void)inspectSnapshots(inspection); },
        "noncanonical root");

    std::puts(
        "numanx_exact_runtime_horizon_self_test=pass "
        "parser_cases=3 no_replace=true canonical_root_set=true metal=false");
    return 0;
}

} // namespace exact_runtime_horizon_probe

int main(const int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
        try {
            return exact_runtime_horizon_probe::runSelfTests();
        } catch (const std::exception& error) {
            std::fprintf(
                stderr, "numanx_exact_runtime_horizon_self_test: %s\n",
                error.what());
            return 1;
        }
    }
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::printf(
            "usage: %s --roots N --timestep-ns 12500 "
            "--uniform-excitation X --snapshot-step N "
            "--snapshot-dir ABSOLUTE_NEW_PATH\n",
            argv[0]);
        return 0;
    }
    try {
        return exact_runtime_horizon_probe::run(
            exact_runtime_horizon_probe::parseArguments(argc, argv));
    } catch (const std::exception& error) {
        std::fprintf(
            stderr, "numanx_exact_runtime_horizon_probe: %s\n",
            error.what());
        return 1;
    }
}

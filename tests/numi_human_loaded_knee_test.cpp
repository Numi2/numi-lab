#include "metalrobo/MatterSnapshotArchive.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/NumiHumanLoadedKnee.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {

using metalrobo::NumiHumanLoadedKneeDigest;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

class ScopedTempDirectory final {
public:
    ScopedTempDirectory() {
        std::string pattern =
            (std::filesystem::temp_directory_path() /
             "numi-loaded-knee-writer-XXXXXX").string();
        char* const created = ::mkdtemp(pattern.data());
        require(created != nullptr,
                "could not create immutable-writer test directory");
        path_ = created;
    }
    ~ScopedTempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    ScopedTempDirectory(const ScopedTempDirectory&) = delete;
    ScopedTempDirectory& operator=(const ScopedTempDirectory&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

class ScopedCurrentPath final {
public:
    ScopedCurrentPath() : original_(std::filesystem::current_path()) {}
    ~ScopedCurrentPath() {
        std::error_code error;
        std::filesystem::current_path(original_, error);
    }
    ScopedCurrentPath(const ScopedCurrentPath&) = delete;
    ScopedCurrentPath& operator=(const ScopedCurrentPath&) = delete;

private:
    std::filesystem::path original_;
};

std::vector<std::byte> readBytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input),
            "immutable-writer test output could not be opened");
    const std::vector<char> chars{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    require(!input.bad(), "immutable-writer test output read failed");
    std::vector<std::byte> bytes(chars.size());
    std::transform(
        chars.begin(), chars.end(), bytes.begin(),
        [](const char value) {
            return static_cast<std::byte>(
                static_cast<unsigned char>(value));
        });
    return bytes;
}

void testImmutableWriter() {
    using metalrobo::writeNumiHumanLoadedKneeImmutableFileV1;
    ScopedTempDirectory directory;
    const std::vector<std::byte> first{
        std::byte{0x00}, std::byte{0x7f},
        std::byte{0x80}, std::byte{0xff}};
    const std::vector<std::byte> second{
        std::byte{0x10}, std::byte{0x20}, std::byte{0x30}};
    std::string error;

    {
        ScopedCurrentPath currentPath;
        std::filesystem::current_path(directory.path());
        require(writeNumiHumanLoadedKneeImmutableFileV1(
                    "basename.bin", first, error), error);
    }
    const auto basename = directory.path() / "basename.bin";
    require(readBytes(basename) == first,
            "basename immutable publication changed bytes");
    require(writeNumiHumanLoadedKneeImmutableFileV1(
                basename, first, error),
            "identical immutable publication was rejected: " + error);
    require(!writeNumiHumanLoadedKneeImmutableFileV1(
                basename, second, error),
            "conflicting immutable publication replaced its destination");
    require(readBytes(basename) == first,
            "conflicting immutable publication changed its destination");

    const auto symlinkTarget = directory.path() / "symlink-target.bin";
    const auto symlinkOutput = directory.path() / "symlink-output.bin";
    require(writeNumiHumanLoadedKneeImmutableFileV1(
                symlinkTarget, first, error), error);
    std::error_code filesystemError;
    std::filesystem::create_symlink(
        symlinkTarget.filename(), symlinkOutput, filesystemError);
    require(!filesystemError,
            "could not create immutable-writer symlink fixture");
    require(!writeNumiHumanLoadedKneeImmutableFileV1(
                symlinkOutput, first, error),
            "immutable publication followed a destination symlink");
    require(readBytes(symlinkTarget) == first,
            "rejected destination symlink changed its target");

    const auto realParent = directory.path() / "real-parent";
    const auto linkedParent = directory.path() / "linked-parent";
    std::filesystem::create_directory(realParent, filesystemError);
    require(!filesystemError,
            "could not create immutable-writer parent fixture");
    std::filesystem::create_directory_symlink(
        realParent.filename(), linkedParent, filesystemError);
    require(!filesystemError,
            "could not create immutable-writer parent-link fixture");
    require(!writeNumiHumanLoadedKneeImmutableFileV1(
                linkedParent / "through-link.bin", first, error),
            "immutable publication followed its final parent symlink");
    require(!std::filesystem::exists(realParent / "through-link.bin"),
            "rejected parent symlink published an output");

    std::vector<std::byte> raceFirst(256u * 1024u, std::byte{0x55});
    std::vector<std::byte> raceSecond(256u * 1024u, std::byte{0xaa});
    raceFirst.front() = std::byte{0x01};
    raceSecond.front() = std::byte{0x02};
    const auto raceOutput = directory.path() / "race.bin";
    std::atomic<std::uint32_t> ready{0u};
    std::atomic<bool> start{false};
    bool firstSucceeded = false;
    bool secondSucceeded = false;
    std::string firstError;
    std::string secondError;
    const auto publish = [&](const std::vector<std::byte>& bytes,
                             bool& succeeded,
                             std::string& writerError) {
        ready.fetch_add(1u, std::memory_order_release);
        while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();
        succeeded = writeNumiHumanLoadedKneeImmutableFileV1(
            raceOutput, bytes, writerError);
    };
    std::thread firstThread(
        publish, std::cref(raceFirst), std::ref(firstSucceeded),
        std::ref(firstError));
    std::thread secondThread(
        publish, std::cref(raceSecond), std::ref(secondSucceeded),
        std::ref(secondError));
    while (ready.load(std::memory_order_acquire) != 2u)
        std::this_thread::yield();
    start.store(true, std::memory_order_release);
    firstThread.join();
    secondThread.join();
    require(firstSucceeded != secondSucceeded,
            "concurrent immutable publications did not admit exactly one writer");
    const auto raceBytes = readBytes(raceOutput);
    require(raceBytes == (firstSucceeded ? raceFirst : raceSecond),
            "concurrent immutable publication did not preserve the winner's bytes");
    for (const auto& entry :
         std::filesystem::directory_iterator(directory.path())) {
        require(entry.path().filename().string().find(
                    ".numi-loaded-knee-stage-") != 0u,
                "immutable publication left a staging file behind");
    }
}

NumiHumanLoadedKneeDigest digest(const std::uint8_t seed) {
    NumiHumanLoadedKneeDigest value{};
    for (std::size_t index = 0u; index < value.size(); ++index) {
        value[index] = static_cast<std::uint8_t>(seed + index);
    }
    return value;
}

NumiHumanLoadedKneeDigest hexDigest(const std::string& text) {
    require(text.size() == 64u, "test SHA-256 text has wrong length");
    NumiHumanLoadedKneeDigest value{};
    const auto nibble = [](const char character) -> std::uint8_t {
        if (character >= '0' && character <= '9')
            return static_cast<std::uint8_t>(character - '0');
        if (character >= 'a' && character <= 'f')
            return static_cast<std::uint8_t>(10 + character - 'a');
        throw std::runtime_error("test SHA-256 text is not lowercase hex");
    };
    for (std::size_t index = 0u; index < value.size(); ++index) {
        value[index] = static_cast<std::uint8_t>(
            (nibble(text[2u * index]) << 4u) |
            nibble(text[2u * index + 1u]));
    }
    return value;
}

metalrobo::NumiHumanLoadedKneeImmutableIdentityV1 identity(
    const std::string& schema,
    const std::uint8_t seed
) {
    return {
        .schema = schema,
        .fileSHA256 = digest(seed),
        .identitySHA256 = digest(static_cast<std::uint8_t>(seed + 1u)),
    };
}

metalrobo::NumiHumanLoadedKneeAuthoringV1 authoringFixture() {
    using namespace metalrobo;
    NumiHumanLoadedKneeAuthoringV1 value;
    value.schema = "HumanPack.loaded-anatomy-knee.v1";
    value.compiler = "numilab-human.loaded-anatomy-knee.1";
    value.status = "candidate";
    value.sourceOwnershipStatus = "partial";
    value.side = "left";
    value.manifestSHA256 = digest(1u);
    value.manifestCanonicalization =
        "utf8-json-sorted-keys-compact-ensure_ascii=false-allow_nan=false";
    value.manifestHashExclusion = "top-level manifest_sha256";
    value.ownershipInput = identity("HumanPack.ownership.v1", 2u);
    value.ownershipManifestSHA256 =
        value.ownershipInput.identitySHA256;
    value.authoringProfileInput = identity(
        "numi.human.loaded-anatomy-knee-authoring.v1", 4u);
    value.authoringProfileInput.fileSHA256 = hexDigest(
        "ec311c48c7ab58dede83da228686d6574d8ebd1c47d22539dbee09967131d3f5");
    value.authoringProfileInput.identitySHA256 = hexDigest(
        "c4bbde523018016b94c5e176fd01e519cfbb2a811f84005f964d0718753b6437");
    value.tendonPayloadInput = identity(
        "numi.human.tendon-attachment-envelope-payload.v3", 6u);
    value.xReferenceInput = identity(
        "numi.human.loaded-anatomy-knee-x-ref-f32le.v1", 8u);
    value.labExportInput = identity(
        "numi.lab.loaded-knee-authoring-export.v1", 10u);
    value.sourceModelFingerprintSHA256 = digest(11u);
    value.sourceRigidPayloadSHA256 = hexDigest(
        "6328f7e84663c611c5498624d1386b00b2d5b0e162c4cc2967c7b1dc49ab0c44");
    value.equalityPayloadSHA256 = hexDigest(
        "b97f755c769d0af16e02ab5deb9d85bd0cc921649197f71d308e98130ac69b6a");
    value.sourceDefaultPoseID =
        "numi-human:unprojected-myosim-default-body-pose";
    value.sourceDefaultPoseSHA256 = digest(18u);
    value.projectedReferencePoseID =
        "numi-human:equality-projected-default-reference-body-pose";
    value.projectedReferencePoseSHA256 = digest(51u);
    value.sourceToReferenceMappingID =
        "numi-lab.open-knee-restWorld-to-equality-projected-default-body-poses.1";
    value.sourceToReferenceMappingAlgorithm =
        "moving-enthesis-tetrahedral-harmonic-map";
    value.sourceToReferenceMappingCodeSHA256 = digest(19u);
    value.kneePayload = {
        .schema = "numi.human.open-knee-oks003-payload.v3",
        .magic = "NHKNEE1",
        .abi = 3u,
        .byteCount = kNumiHumanLoadedKneePayloadBytesV1,
        .fileSHA256 = {
            0x2e, 0x38, 0x20, 0x15, 0x28, 0xde, 0x25, 0x91,
            0x1e, 0xa4, 0x96, 0x16, 0x46, 0x02, 0xea, 0x7e,
            0x82, 0x3c, 0xd9, 0xc2, 0xe3, 0xc9, 0x4e, 0xfa,
            0x55, 0x0e, 0xe5, 0x26, 0x89, 0x32, 0x4a, 0xe5,
        },
    };
    value.tendonPayload = {
        .schema = "numi.human.tendon-attachment-envelope-payload.v3",
        .magic = "NHTENDON3",
        .abi = 3u,
        .byteCount = 238'288u,
        .fileSHA256 = {
            0x72, 0xca, 0x0e, 0xc4, 0xef, 0x53, 0xf6, 0x47,
            0x78, 0x4a, 0x89, 0xf7, 0x61, 0xe7, 0x84, 0x95,
            0xd4, 0x99, 0x7c, 0x7d, 0x72, 0xcf, 0xe6, 0xa1,
            0xd9, 0xee, 0xdb, 0x24, 0xdd, 0x97, 0xfe, 0xaa,
        },
    };
    value.subjectID = "OpenKnee:oks003:left";
    value.coverageLeafSHA256s = {digest(12u), digest(13u)};
    value.datasetID = "OpenKnee:oks003";
    value.licenseFileSHA256 = digest(14u);
    value.topology.identitySHA256 = hexDigest(
        "0569112a4d39eccc84bcb81a19a870b690cb2b3d642a74708d41881923f3c349");
    value.topology.executableFEMTopologySHA256 = hexDigest(
        "56af24e63c9a4c2ecafb2c94bab98fa95bbcbc77085f425c84b9c1a95e3a9040");
    value.topology.sourceGlobalNodeIndexSHA256 = hexDigest(
        "e451aa9e773a90dfb0cde77d4fb1555c7f1393c0362c7fbb3614909045a6602b");
    value.topology.anchorOwnershipSHA256 = hexDigest(
        "04a5c374f15f8681af40ae2e07fe35258f0bbc805d4d9d8291eae1f891d23de6");
    value.topology.nodeCount = kNumiHumanLoadedKneeNodeCount;
    value.topology.tetrahedronCount = kNumiHumanLoadedKneeTetrahedronCount;
    value.topology.surfaceCount = kNumiHumanLoadedKneeSurfaceCount;
    value.topology.surfaceFaceCount = kNumiHumanLoadedKneeSurfaceFaceCount;
    value.topology.nodeSetCount = kNumiHumanLoadedKneeNodeSetCount;
    value.topology.nodeSetMembershipCount =
        kNumiHumanLoadedKneeNodeSetMembershipCount;
    value.topology.sourceSurfacePairCount =
        kNumiHumanLoadedKneeSourceSurfacePairCount;
    value.topology.loadedNodeCount = kNumiHumanLoadedKneeLoadedNodeCount;
    value.topology.loadedTetrahedronCount =
        kNumiHumanLoadedKneeLoadedTetrahedronCount;
    value.topology.loadedRegionNames = {
        "ACL", "LCL", "MCL", "PCL", "PTL", "QAT"};
    value.coordinates.sourceStateSHA256 = digest(16u);
    value.coordinates.referenceStateSHA256 = digest(17u);
    value.coordinates.sourceStateNodeCount =
        kNumiHumanLoadedKneeLoadedNodeCount;
    value.coordinates.referenceStateNodeCount =
        kNumiHumanLoadedKneeLoadedNodeCount;
    value.coordinates.sourceStateEncoding = "float32-le-xyz";
    value.coordinates.referenceStateEncoding = "float32-le-xyz";
    value.coordinates.sourceFrameID = "myosim-world-m";
    value.coordinates.referenceFrameID = "myosim-world-m";
    value.coordinates.referenceStateClass = "projected-rest-candidate";
    value.coordinates.constructionID =
        "numi-lab.open-knee-moving-enthesis-projected-rest.1";
    value.coordinates.prestrainResetStatus = "unresolved";
    value.coordinates.volumetricPrestressStatus = "not_applied";
    value.coordinates.currentStateOwner =
        "separate-lab-runtime-acceptance-receipt";
    value.coordinates.currentStateHashAlgorithm = "sha256";
    value.coordinates.currentStateHashScope =
        "six-loaded-regions-in-authoring-profile-order";
    value.coordinates.currentStateRequired = true;
    value.densitySourceValue = 1.0e-9;
    value.densitySourceUnit = "tonne_per_mm3";
    value.densityConversionFactorToKgPerM3 = 1.0e12;
    value.densityRuntimeKgPerM3 = 1000.0;
    value.densityCalibrationStatus = "source_population_prior";

    struct MaterialRow {
        double c1;
        double c3;
        double c4;
        double c5;
        double lambdaMaximum;
        double bulk;
        double stretch;
    };
    constexpr std::array<MaterialRow, 6u> materials{{
        {1.95e6, 0.0139e6, 116.22, 535.039e6, 1.046, 146.41e6, 1.016},
        {1.44e6, 0.57e6, 48.0, 467.1e6, 1.063, 793.65e6, 1.027},
        {1.44e6, 0.57e6, 48.0, 467.1e6, 1.063, 793.65e6, 1.034},
        {3.25e6, 0.1196e6, 87.178, 431.063e6, 1.035, 243.9e6, 1.0},
        {2.75e6, 0.065e6, 115.89, 777.56e6, 1.042, 206.61e6, 1.0},
        {2.75e6, 0.065e6, 115.89, 777.56e6, 1.042, 206.61e6, 1.0},
    }};
    constexpr std::array<std::array<double, 3u>, 6u> fibers{{
        {{0.023879600688815117, 0.5208771824836731,
          0.8532975912094116}},
        {{-0.1950301229953766, -0.1955101639032364,
          0.961113452911377}},
        {{-0.28888770937919617, -0.0032872306182980537,
          0.9573573470115662}},
        {{0.02508438006043434, 0.7640138864517212,
          -0.644711971282959}},
        {{-0.1883269101381302, -0.4512072503566742,
          0.872321605682373}},
        {{-0.035553932189941406, 0.22192594408988953,
          0.974415123462677}},
    }};
    for (std::size_t index = 0u; index < value.regions.size(); ++index) {
        auto& region = value.regions[index];
        region.sourceRegionName = value.topology.loadedRegionNames[index];
        region.semanticID =
            "open_knee_oks003:Geometry.feb#/febio_spec[1]/Geometry[1]/Elements[name=" +
            region.sourceRegionName + "][1]";
        region.donorBodyIndex = region.sourceRegionName == "PTL"
            ? NUMI_HUMAN_KNEE_TIBIA_BODY
            : NUMI_HUMAN_KNEE_FEMUR_BODY;
        region.topologyIdentitySHA256 = digest(
            static_cast<std::uint8_t>(30u + index));
        region.material = {
            .sourceType = "trans iso Mooney-Rivlin",
            .c1Pascals = materials[index].c1,
            .c2Pascals = 0.0,
            .c3Pascals = materials[index].c3,
            .c4 = materials[index].c4,
            .c5Pascals = materials[index].c5,
            .lambdaMaximum = materials[index].lambdaMaximum,
            .bulkModulusPascals = materials[index].bulk,
            .initialStretch = materials[index].stretch,
            .homogeneousFiberWorld = fibers[index],
            .calibrationStatus = "source_population_prior",
        };
        const std::string ownerPrefix =
            "humanpack:loaded-anatomy-knee:left/region/" +
            region.sourceRegionName;
        region.physicalVolumeOwnerID = ownerPrefix + "/physical-volume";
        region.mechanicalMassOwnerID = ownerPrefix + "/mechanical-mass";
        region.materialOwnerID = ownerPrefix + "/material";
        region.stateOwnerID =
            "humanpack:loaded-anatomy-knee:left/full-state";
        if (region.sourceRegionName == "QAT") {
            region.activeForceOwnerStatus = "candidate";
            region.activeForceOwnerID =
                "humanpack:loaded-anatomy-knee:left/QAT/active-force";
        } else {
            region.activeForceOwnerStatus = "none";
        }
    }

    constexpr std::array<const char*, 7u> pairNames{{
        "TBC-L_To_FMC", "TBC-L_To_MNS-L", "PTC_To_FMC",
        "MNS-L_To_FMC", "MNS-M_To_TBC-M", "MNS-M_To_FMC",
        "TBC-M_To_FMC"}};
    constexpr std::array<const char*, 7u> master{{
        "TBC-L_@_FMC_ContactFaces", "TBC-L_@_MNS-L_ContactFaces",
        "PTC_@_FMC_ContactFaces", "MNS-L_@_FMC_ContactFaces",
        "MNS-M_@_TBC-M_ContactFaces", "MNS-M_@_FMC_ContactFaces",
        "TBC-M_@_FMC_ContactFaces"}};
    constexpr std::array<const char*, 7u> slave{{
        "FMC_@_TBC-L_ContactFaces", "MNS-L_@_TBC-L_ContactFaces",
        "FMC_@_PTC_ContactFaces", "FMC_@_MNS-L_ContactFaces",
        "TBC-M_@_MNS-M_ContactFaces", "FMC_@_MNS-M_ContactFaces",
        "FMC_@_TBC-M_ContactFaces"}};
    for (std::size_t index = 0u; index < value.contactPairs.size(); ++index) {
        value.contactPairs[index] = {
            .semanticID =
                std::string("open_knee_oks003:Geometry.feb#/febio_spec[1]/Geometry[1]/SurfacePair[name=") +
                pairNames[index] + "][1]",
            .sourcePairName = pairNames[index],
            .masterSurface = master[index],
            .slaveSurface = slave[index],
            .ownerID =
                std::string("humanpack:loaded-anatomy-knee:left/contact/") +
                pairNames[index] + "/state",
        };
    }

    constexpr std::array<std::uint32_t, 4u> actuator{{405u, 413u, 414u, 415u}};
    constexpr std::array<const char*, 4u> route{{
        "recfem_l", "vasint_l", "vaslat_l", "vasmed_l"}};
    constexpr std::array<std::uint32_t, 4u> loadEndpoint{{810u, 826u, 828u, 830u}};
    constexpr std::array<std::uint32_t, 4u> loadRoute{{1798u, 1830u, 1836u, 1842u}};
    constexpr std::array<std::uint32_t, 4u> loadSite{{1548u, 1715u, 1717u, 1719u}};
    constexpr std::array<std::uint32_t, 4u> loadBody{{128u, 145u, 145u, 145u}};
    constexpr std::array<std::uint32_t, 4u> anchorEndpoint{{811u, 827u, 829u, 831u}};
    constexpr std::array<std::uint32_t, 4u> anchorRoute{{1803u, 1835u, 1841u, 1847u}};
    constexpr std::array<std::uint32_t, 4u> anchorSite{{1751u, 1765u, 1766u, 1767u}};
    for (std::size_t index = 0u; index < value.activeReplacements.size(); ++index) {
        const std::string semantic =
            "myosim_fullbody:composed/myofullbody/muscles/" +
            std::string(route[index]);
        value.activeReplacements[index] = {
            .semanticID = semantic,
            .sourceActuatorIndex = actuator[index],
            .sourceRouteName = route[index],
            .loadEndpointIndex = loadEndpoint[index],
            .loadRouteNodeIndex = loadRoute[index],
            .loadSourceSiteIndex = loadSite[index],
            .loadBodyIndex = loadBody[index],
            .anchorEndpointIndex = anchorEndpoint[index],
            .anchorRouteNodeIndex = anchorRoute[index],
            .anchorSourceSiteIndex = anchorSite[index],
            .anchorBodyIndex = NUMI_HUMAN_KNEE_TIBIA_BODY,
            .sourceOwnerID = semantic + "/owner/source-jt",
            .candidateOwnerID =
                "humanpack:loaded-anatomy-knee:left/QAT/active-force",
            .mode = "replacement",
            .replacementFraction = 1.0,
        };
    }
    constexpr std::array<const char*, 5u> passive{{
        "ACL", "LCL", "MCL", "PCL", "PTL"}};
    for (std::size_t index = 0u; index < value.passiveOwners.size(); ++index) {
        value.passiveOwners[index] = {
            .semanticID =
                std::string("humanpack:loaded-anatomy-knee:left/region/") +
                passive[index] + "/passive-force",
            .sourceRegionName = passive[index],
            .ownerID =
                std::string("humanpack:loaded-anatomy-knee:left/region/") +
                passive[index] + "/material",
        };
    }
    value.authoredMassPartitionSHA256 = digest(40u);
    value.authoredRawF32NodeMassSHA256 = digest(42u);
    value.fullStateSemanticID =
        "humanpack:loaded-anatomy-knee:left/full-state-authority";
    value.fullStateOwnerID =
        "numi-lab:human-matter/accepted-step-transaction";
    value.fullStateSchema = "NumiLab.HumanMatterAcceptedState.v1";
    value.fullStateIdentitySHA256 = digest(41u);
    value.requiredSnapshotComponents = {
        "articulated-q-v-root-time",
        "articular-contact-history",
        "fem-position-velocity-mass-material-status",
        "passive-ligament-state",
        "solver-adaptive-state",
        "tendon-transfer-replacement-state",
    };
    value.candidateOnly = true;
    value.donorMassSubtractionRequired = true;
    value.massMomentClosureRequired = true;
    value.boundary = std::string(kNumiHumanLoadedKneeHumanPackBoundaryV1);
    return value;
}

float sourcePressureFloat(const double pascals) {
    return 1.0e6f * static_cast<float>(pascals / 1.0e6);
}

std::filesystem::path exactFEBioMaterialPath() {
    return std::filesystem::path(__FILE__).parent_path().parent_path() /
        "matter" / "materials" /
        "open_knee_ligament_febio_exp_linear.nmatter";
}

void setMaterialParameter(
    numi::matter::MaterialProgram& material,
    const std::string& name,
    const double value
) {
    const auto found = std::find_if(
        material.parameters.begin(), material.parameters.end(),
        [&name](const numi::matter::Parameter& parameter) {
            return parameter.name == name;
        });
    require(found != material.parameters.end(),
            "test material parameter is missing: " + name);
    found->defaultValue = value;
}

numi::matter::CompiledWorld materialWorldFixture(
    const metalrobo::NumiHumanLoadedKneeAuthoringV1& authoring
) {
    const auto parsed = numi::matter::parseMatterFile(
        exactFEBioMaterialPath());
    require(parsed.succeeded(),
            "exact FEBio test material did not parse");
    numi::matter::WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 50.0e-6;
    source.gravity = {0.0, 0.0, 0.0};
    for (std::size_t region = 0u;
         region < authoring.regions.size(); ++region) {
        auto material = parsed.material;
        const auto& row = authoring.regions[region].material;
        material.name =
            "open_knee_" + authoring.regions[region].sourceRegionName +
            "_live_human_febio_exp_linear_v1";
        material.fingerprint = 0u;
        for (auto& parameter : material.parameters)
            parameter.identifiable = false;
        setMaterialParameter(material, "density",
                             authoring.densityRuntimeKgPerM3);
        setMaterialParameter(material, "c1", sourcePressureFloat(
            row.c1Pascals));
        setMaterialParameter(material, "c3", sourcePressureFloat(
            row.c3Pascals));
        setMaterialParameter(material, "c4", row.c4);
        setMaterialParameter(material, "c5", sourcePressureFloat(
            row.c5Pascals));
        setMaterialParameter(material, "lambda_max", row.lambdaMaximum);
        setMaterialParameter(
            material, "fiber_scale",
            region < metalrobo::kNumiHumanLoadedKneePassiveOwnerCount
                ? 0.0 : 1.0);
        setMaterialParameter(material, "bulk", sourcePressureFloat(
            row.bulkModulusPascals));
        setMaterialParameter(
            material, "initial_stretch",
            region < metalrobo::kNumiHumanLoadedKneePassiveOwnerCount
                ? 1.0 : row.initialStretch);
        setMaterialParameter(
            material, "fiber_x", row.homogeneousFiberWorld[0u]);
        setMaterialParameter(
            material, "fiber_y", row.homogeneousFiberWorld[1u]);
        setMaterialParameter(
            material, "fiber_z", row.homogeneousFiberWorld[2u]);
        setMaterialParameter(material, "numerical_viscosity", 25.0);
        source.materials.push_back(std::move(material));

        const double offset = 0.1 * static_cast<double>(region);
        numi::matter::ObjectSource object;
        object.name = "loaded_knee_test_" +
            authoring.regions[region].sourceRegionName;
        object.materialIndex = static_cast<std::uint32_t>(region);
        object.representation = numi::matter::Representation::fem;
        object.mixedFEM = false;
        object.deformableContact = false;
        object.deformableSelfContact = false;
        object.characteristicLength = 0.01;
        object.femNodes = {
            {offset, 0.0, 0.0},
            {offset + 0.01, 0.0, 0.0},
            {offset, 0.01, 0.0},
            {offset, 0.0, 0.01},
        };
        object.tetrahedra.push_back({{0u, 1u, 2u, 3u}});
        source.objects.push_back(std::move(object));
    }
    numi::matter::CompileOptions options;
    options.maximumRateExponent = 0u;
    options.emitSpecializedMetal = false;
    const auto compiled = numi::matter::compileWorld(source, options);
    std::string diagnostics;
    for (const auto& diagnostic : compiled.diagnostics)
        diagnostics += diagnostic.message + "; ";
    require(compiled.succeeded(),
            "exact FEBio test world failed: " + diagnostics);
    return compiled.world;
}

struct PassiveExecutionFixture {
    std::vector<metalrobo::NumiHumanLoadedKneeExecutedAnchorV1> anchors;
    std::array<NMNumiHumanPassiveLigamentGPU,
               metalrobo::kNumiHumanLoadedKneePassiveOwnerCount> rows{};
};

PassiveExecutionFixture passiveExecutionFixture(
    const metalrobo::NumiHumanLoadedKneeAuthoringV1& authoring,
    const numi::matter::CompiledWorld& world
) {
    using namespace metalrobo;
    PassiveExecutionFixture fixture;
    fixture.anchors.resize(world.fem.nodes.size());
    for (std::size_t index = 0u; index < fixture.anchors.size(); ++index) {
        fixture.anchors[index].sourceGlobalNodeIndex =
            static_cast<std::uint32_t>(index);
        fixture.anchors[index].bodyIndex = NUMI_HUMAN_KNEE_INVALID_INDEX;
    }
    constexpr std::array<std::uint32_t,
                         kNumiHumanLoadedKneePassiveOwnerCount>
        firstBodies{{
            NUMI_HUMAN_KNEE_FEMUR_BODY,
            NUMI_HUMAN_KNEE_FEMUR_BODY,
            NUMI_HUMAN_KNEE_FEMUR_BODY,
            NUMI_HUMAN_KNEE_FEMUR_BODY,
            NUMI_HUMAN_KNEE_PATELLA_BODY}};
    for (std::size_t region = 0u; region < fixture.rows.size(); ++region) {
        const auto& object = world.objects[region];
        require(object.stateCount == 4u && object.elementCount == 1u,
                "test passive object extent drifted");
        auto& first = fixture.anchors[object.stateOffset];
        first.bodyIndex = firstBodies[region];
        first.flags = 1u;
        first.localPoint = {
            static_cast<float>(0.001 * region), 0.002f, -0.003f};
        auto& second = fixture.anchors[object.stateOffset + 1u];
        second.bodyIndex = NUMI_HUMAN_KNEE_TIBIA_BODY;
        second.flags = 1u;
        second.localPoint = {
            static_cast<float>(0.001 * region + 0.01),
            -0.004f, 0.005f};

        const auto& firstReference =
            world.fem.nodes[object.stateOffset].restAndFixed;
        const auto& secondReference =
            world.fem.nodes[object.stateOffset + 1u].restAndFixed;
        const float dx = secondReference.x - firstReference.x;
        const float dy = secondReference.y - firstReference.y;
        const float dz = secondReference.z - firstReference.z;
        const float length = std::sqrt(dx * dx + dy * dy + dz * dz);
        const auto& tet =
            world.fem.tetrahedra[object.elementOffset].nodes;
        const auto point = [&world](const std::uint32_t index) {
            const auto& value = world.fem.nodes[index].restAndFixed;
            return std::array<float, 3u>{value.x, value.y, value.z};
        };
        const auto a = point(tet.x);
        const auto b = point(tet.y);
        const auto c = point(tet.z);
        const auto d = point(tet.w);
        const std::array<float, 3u> ab{{
            b[0u] - a[0u], b[1u] - a[1u], b[2u] - a[2u]}};
        const std::array<float, 3u> ac{{
            c[0u] - a[0u], c[1u] - a[1u], c[2u] - a[2u]}};
        const std::array<float, 3u> ad{{
            d[0u] - a[0u], d[1u] - a[1u], d[2u] - a[2u]}};
        const std::array<float, 3u> cross{{
            ac[1u] * ad[2u] - ac[2u] * ad[1u],
            ac[2u] * ad[0u] - ac[0u] * ad[2u],
            ac[0u] * ad[1u] - ac[1u] * ad[0u]}};
        const float triple = ab[0u] * cross[0u] +
            ab[1u] * cross[1u] + ab[2u] * cross[2u];
        const double volume =
            std::abs(static_cast<double>(triple)) / 6.0;
        const auto& material = authoring.regions[region].material;
        auto& row = fixture.rows[region];
        row.firstBodyIndex = first.bodyIndex;
        row.secondBodyIndex = second.bodyIndex;
        row.flags = NM_NUMI_HUMAN_PASSIVE_LIGAMENT_ACTIVE;
        row.firstLocalPoint = {
            first.localPoint[0u], first.localPoint[1u],
            first.localPoint[2u], 0.0f};
        row.secondLocalPoint = {
            second.localPoint[0u], second.localPoint[1u],
            second.localPoint[2u], 0.0f};
        row.material = {
            sourcePressureFloat(material.c3Pascals),
            static_cast<float>(material.c4),
            sourcePressureFloat(material.c5Pascals),
            static_cast<float>(material.lambdaMaximum)};
        row.reference = {
            length, static_cast<float>(volume / length),
            static_cast<float>(material.initialStretch), 0.0f};
    }
    return fixture;
}

numi::matter::RuntimeStateSnapshot snapshot(
    const std::uint32_t step,
    const std::uint64_t sourcePhysicsFingerprint
) {
    numi::matter::RuntimeStateSnapshot value;
    value.available = true;
    value.sourcePhysicsFingerprint = sourcePhysicsFingerprint;
    value.deviceProgramFingerprint = 0x2468u;
    value.controlStep = step - 1u;
    value.physicsSubstep = 0u;
    value.femNodes.resize(
        metalrobo::kNumiHumanLoadedKneeLoadedNodeCount);
    value.femNodes.front().positionAndMass = {
        0.001f * static_cast<float>(step), 0.0f, 0.0f, 1.0f};
    return value;
}

metalrobo::NumiHumanLoadedKneeRuntimeEvidenceV1 runtimeEvidence(
    const metalrobo::NumiHumanLoadedKneeAuthoringV1& authoring,
    const metalrobo::NumiHumanLoadedKneeMassEvidenceV1& mass,
    const std::uint32_t step
) {
    using namespace metalrobo;
    NumiHumanLoadedKneeRuntimeEvidenceV1 value;
    value.acceptedStepIndex = step;
    value.acceptedTimestampNanoseconds =
        static_cast<std::uint64_t>(step) *
        kNumiHumanLoadedKneeStepNanoseconds;
    value.timestepNanoseconds = kNumiHumanLoadedKneeStepNanoseconds;
    value.loadedFEMNodeCount = kNumiHumanLoadedKneeLoadedNodeCount;
    value.executedSourceCoordinateSHA256 =
        authoring.coordinates.sourceStateSHA256;
    value.executedReferenceCoordinateSHA256 =
        authoring.coordinates.referenceStateSHA256;
    value.executedInitialCoordinateSHA256 = digest(50u);
    value.referencePoseSHA256 = digest(51u);
    value.initialPoseSHA256 = digest(52u);
    std::string error;
    require(digestNumiHumanLoadedKneeCoordinateDerivationV1(
                authoring, value.executedInitialCoordinateSHA256,
                value.referencePoseSHA256, value.initialPoseSHA256,
                value.coordinateDerivationSHA256, error), error);
    value.executedMassClosureSHA256 = mass.closureSHA256;
    value.executedRawF32NodeMassSHA256 =
        mass.executedRawF32NodeMassSHA256;
    value.executedSourceGlobalNodeIndexSHA256 =
        mass.sourceGlobalNodeIndexSHA256;
    value.executedFEMTopologySHA256 =
        mass.executableFEMTopologySHA256;
    value.executedSourceAnchorOwnershipSHA256 =
        mass.sourceAnchorOwnershipSHA256;
    value.executedAnchorOwnershipSHA256 =
        mass.executedAnchorOwnershipSHA256;
    value.executedMaterialExecutionSHA256 =
        mass.materialExecutionSHA256;
    value.executedSourceRigidModelFingerprint =
        mass.sourceRigidModelFingerprint;
    value.executedSourcePhysicsFingerprint =
        mass.executedSourcePhysicsFingerprint;
    value.executedRigidModelFingerprint =
        mass.executedRigidModelFingerprint;
    value.contactPairNormalForceNewtons.fill(1.0);
    value.contactAggregateNormalForceNewtons =
        static_cast<double>(kNumiHumanLoadedKneeContactPairCount);
    value.prescribedClosurePairForceNewtons.fill(1.0);
    value.executedContactPairs = authoring.contactPairs;
    value.executedActiveReplacements = authoring.activeReplacements;
    value.executedPassiveOwners = authoring.passiveOwners;
    value.articulatedQVRootTimeSHA256 = digest(
        static_cast<std::uint8_t>(70u + step));
    value.muscleTendonStateSHA256 = digest(
        static_cast<std::uint8_t>(80u + step));
    value.adapterAcceptedStateSHA256 = digest(
        static_cast<std::uint8_t>(90u + step));
    value.snapshot = snapshot(step, mass.executedSourcePhysicsFingerprint);
    return value;
}

} // namespace

int main() {
    try {
        using namespace metalrobo;
        std::string error;
        testImmutableWriter();
        const auto authoring = authoringFixture();
        require(validateNumiHumanLoadedKneeAuthoringV1(
                    authoring, nullptr, error), error);
        auto overclaim = authoring;
        overclaim.coordinates.unloadedReferenceQualified = true;
        require(!validateNumiHumanLoadedKneeAuthoringV1(
                    overclaim, nullptr, error),
                "unloaded-reference overclaim was admitted");
        overclaim = authoring;
        overclaim.boundary =
            "Production and clinical qualification established.";
        require(!validateNumiHumanLoadedKneeAuthoringV1(
                    overclaim, nullptr, error),
                "mutated Human qualification boundary was admitted");
        overclaim = authoring;
        overclaim.coordinates.authoredAcceptedCurrentStatePresent = true;
        require(!validateNumiHumanLoadedKneeAuthoringV1(
                    overclaim, nullptr, error),
                "authoring-owned x_current was admitted");
        overclaim = authoring;
        overclaim.contactPairs[0u].masterSurface = "invented-short-name";
        require(!validateNumiHumanLoadedKneeAuthoringV1(
                    overclaim, nullptr, error),
                "wrong ABI3 surface identity was admitted");
        overclaim = authoring;
        overclaim.activeReplacements[0u].loadEndpointIndex = 0u;
        require(!validateNumiHumanLoadedKneeAuthoringV1(
                    overclaim, nullptr, error),
                "wrong endpoint tuple was admitted");
        overclaim = authoring;
        overclaim.regions[0u].activeForceOwnerStatus = "candidate";
        require(!validateNumiHumanLoadedKneeAuthoringV1(
                    overclaim, nullptr, error),
                "passive active-force owner sentinel drift was admitted");
        overclaim = authoring;
        std::swap(overclaim.regions[0u].material.homogeneousFiberWorld[0u],
                  overclaim.regions[0u].material.homogeneousFiberWorld[1u]);
        require(!validateNumiHumanLoadedKneeAuthoringV1(
                    overclaim, nullptr, error),
                "unit-norm but wrong anisotropic fiber direction was admitted");
        overclaim = authoring;
        overclaim.sourceOwnershipStatus = "production";
        require(!validateNumiHumanLoadedKneeAuthoringV1(
                    overclaim, nullptr, error),
                "production source-ownership laundering was admitted");
        overclaim = authoring;
        overclaim.authoringProfileInput.fileSHA256[0u] ^= 1u;
        require(!validateNumiHumanLoadedKneeAuthoringV1(
                    overclaim, nullptr, error),
                "unfrozen Human authoring profile was admitted");
        overclaim = authoring;
        overclaim.authoringProfileInput.fileSHA256 = hexDigest(
            "bc3b1bdb5cf79890207d456eff1aaaaf435aced6133e7449a7d00ff96fc0c262");
        overclaim.authoringProfileInput.identitySHA256 = hexDigest(
            "195c5ab22f735d467fd85eb6c800c21989e04883a0b470c31a2fc391c767eaad");
        require(!validateNumiHumanLoadedKneeAuthoringV1(
                    overclaim, nullptr, error),
                "pre-source-order Human authoring profile was admitted");
        overclaim = authoring;
        overclaim.topology.identitySHA256 = hexDigest(
            "5355ead21494481a0d6ed85344ce15cf9d3df86021acce9bcd010b80530d45ef");
        require(!validateNumiHumanLoadedKneeAuthoringV1(
                    overclaim, nullptr, error),
                "pre-source-order Human topology identity was admitted");

        const auto exactMatterSHA256 = hexDigest(
            "c5166dfcdbcb8402cd107d087ac5071e548f8154b41f088bee3924e40ac9871e");
        require(verifyNumiHumanLoadedKneeImmutableFileV1(
                    exactFEBioMaterialPath(), exactMatterSHA256, 2'702u,
                    error), error);
        const auto materialLink = std::filesystem::temp_directory_path() /
            ("numi-loaded-knee-material-link-" +
             std::to_string(static_cast<unsigned long>(::getpid())));
        std::error_code linkError;
        std::filesystem::remove(materialLink, linkError);
        linkError.clear();
        std::filesystem::create_symlink(
            exactFEBioMaterialPath(), materialLink, linkError);
        require(!linkError,
                "could not create immutable-file symlink rejection fixture");
        require(!verifyNumiHumanLoadedKneeImmutableFileV1(
                    materialLink, exactMatterSHA256, 2'702u, error),
                "immutable material verifier followed a symlink");
        std::filesystem::remove(materialLink, linkError);

        std::vector<std::array<float, 3u>> coordinates(
            kNumiHumanLoadedKneeLoadedNodeCount);
        coordinates.back() = {1.0f, 2.0f, 3.0f};
        NumiHumanLoadedKneeDigest firstCoordinates{};
        require(digestNumiHumanLoadedKneeCoordinatesV1(
                    coordinates, firstCoordinates, error), error);
        coordinates.back()[2u] = 4.0f;
        NumiHumanLoadedKneeDigest secondCoordinates{};
        require(digestNumiHumanLoadedKneeCoordinatesV1(
                    coordinates, secondCoordinates, error), error);
        require(firstCoordinates != secondCoordinates,
                "coordinate digest ignored a row mutation");

        auto donorAuthoring = authoring;
        donorAuthoring.coordinates.referenceStateSHA256 = secondCoordinates;
        EngineModel donorSource;
        donorSource.bodies.resize(NUMI_HUMAN_KNEE_PATELLA_BODY + 1u);
        std::vector<MRBodyStateGPU> donorBodies(donorSource.bodies.size());
        std::vector<NumiHumanTissueMassNode> wrongDonorNodes(
            kNumiHumanLoadedKneeLoadedNodeCount);
        constexpr std::array<std::uint32_t,
                             kNumiHumanLoadedKneeRegionCount>
            regionNodeCounts{{15792u, 2960u, 15693u, 3714u, 9280u,
                              14963u}};
        std::uint32_t firstRegionNode = 0u;
        for (std::size_t region = 0u; region < regionNodeCounts.size();
             ++region) {
            for (std::uint32_t local = 0u;
                 local < regionNodeCounts[region]; ++local) {
                const std::uint32_t nodeIndex = firstRegionNode + local;
                wrongDonorNodes[nodeIndex].nodeIndex = nodeIndex;
                wrongDonorNodes[nodeIndex].donorBody =
                    donorAuthoring.regions[region].donorBodyIndex;
            }
            firstRegionNode += regionNodeCounts[region];
        }
        require(firstRegionNode == wrongDonorNodes.size(),
                "test region spans did not cover the loaded nodes");
        wrongDonorNodes.front().donorBody = NUMI_HUMAN_KNEE_TIBIA_BODY;
        NumiHumanLoadedKneeMassEvidenceV1 rejectedDonorMass;
        rejectedDonorMass.closureSHA256 = digest(59u);
        const auto donorSentinel = rejectedDonorMass.closureSHA256;
        require(!prepareNumiHumanLoadedKneeMassV1(
                    donorAuthoring, donorSource, coordinates, donorBodies,
                    wrongDonorNodes, rejectedDonorMass, error),
                "wrong per-region donor assignment was admitted");
        require(rejectedDonorMass.closureSHA256 == donorSentinel,
                "wrong-donor rejection changed prior mass evidence");

        NumiHumanKneeNode sourceAnchor;
        sourceAnchor.anchorBodyIndex = NUMI_HUMAN_KNEE_FEMUR_BODY;
        sourceAnchor.anchorLocal = {0.5f, -0.25f, 1.0f};
        sourceAnchor.rigidlyAttached = true;
        constexpr std::array<double, 3u> donorCOMOffset{{
            0.125, -0.0625, 0.25}};
        NMFEMNodeStateGPU anchoredMatterNode{};
        anchoredMatterNode.restAndFixed.w = 2.0f;
        NumiHumanLoadedKneeExecutedAnchorV1 executedAnchor{
            .sourceGlobalNodeIndex = 1234u,
            .bodyIndex = NUMI_HUMAN_KNEE_FEMUR_BODY,
            .flags = 1u,
            .localPoint = {
                sourceAnchor.anchorLocal[0u] -
                    static_cast<float>(donorCOMOffset[0u]),
                sourceAnchor.anchorLocal[1u] -
                    static_cast<float>(donorCOMOffset[1u]),
                sourceAnchor.anchorLocal[2u] -
                    static_cast<float>(donorCOMOffset[2u])},
        };
        require(validateNumiHumanLoadedKneeExecutedAnchorRowV1(
                    1234u, sourceAnchor, donorCOMOffset,
                    anchoredMatterNode, executedAnchor, error), error);
        auto fixedInsteadOfAttachedNode = anchoredMatterNode;
        fixedInsteadOfAttachedNode.restAndFixed.w = 1.0f;
        require(!validateNumiHumanLoadedKneeExecutedAnchorRowV1(
                    1234u, sourceAnchor, donorCOMOffset,
                    fixedInsteadOfAttachedNode, executedAnchor, error),
                "statically fixed loaded-knee anchor was admitted as a direct Matter attachment");
        auto bitMutatedAnchor = executedAnchor;
        bitMutatedAnchor.localPoint[0u] = std::bit_cast<float>(
            std::bit_cast<std::uint32_t>(
                bitMutatedAnchor.localPoint[0u]) ^ 1u);
        require(!validateNumiHumanLoadedKneeExecutedAnchorRowV1(
                    1234u, sourceAnchor, donorCOMOffset,
                    anchoredMatterNode, bitMutatedAnchor, error),
                "one-bit executed anchor drift was admitted");
        auto coordinateMutatedAnchor = executedAnchor;
        coordinateMutatedAnchor.localPoint[1u] += 1.0e-4f;
        require(!validateNumiHumanLoadedKneeExecutedAnchorRowV1(
                    1234u, sourceAnchor, donorCOMOffset,
                    anchoredMatterNode, coordinateMutatedAnchor, error),
                "executed anchor coordinate drift was admitted");

        constexpr std::uint32_t executableNode = 1234u;
        constexpr std::uint32_t objectIndex = 2u;
        NMFEMHumanAttachmentGPU compiledAttachment{};
        compiledAttachment.identity = {
            executableNode,
            executedAnchor.bodyIndex,
            objectIndex,
            kNumiHumanLoadedKneeAttachmentStableIdentifierBase +
                executableNode + 1u};
        compiledAttachment.localPoint = {
            executedAnchor.localPoint[0u],
            executedAnchor.localPoint[1u],
            executedAnchor.localPoint[2u], 0.0f};
        require(validateNumiHumanLoadedKneeCompiledAttachmentRowV1(
                    executableNode, objectIndex, sourceAnchor,
                    executedAnchor, anchoredMatterNode,
                    compiledAttachment, error), error);
        auto wrongStableAttachment = compiledAttachment;
        ++wrongStableAttachment.identity.w;
        require(!validateNumiHumanLoadedKneeCompiledAttachmentRowV1(
                    executableNode, objectIndex, sourceAnchor,
                    executedAnchor, anchoredMatterNode,
                    wrongStableAttachment, error),
                "loaded-knee compiled attachment stable-identifier drift was admitted");
        auto wrongObjectAttachment = compiledAttachment;
        ++wrongObjectAttachment.identity.z;
        require(!validateNumiHumanLoadedKneeCompiledAttachmentRowV1(
                    executableNode, objectIndex, sourceAnchor,
                    executedAnchor, anchoredMatterNode,
                    wrongObjectAttachment, error),
                "loaded-knee compiled attachment object drift was admitted");
        auto wrongLocalAttachment = compiledAttachment;
        wrongLocalAttachment.localPoint.x = std::bit_cast<float>(
            std::bit_cast<std::uint32_t>(
                wrongLocalAttachment.localPoint.x) ^ 1u);
        require(!validateNumiHumanLoadedKneeCompiledAttachmentRowV1(
                    executableNode, objectIndex, sourceAnchor,
                    executedAnchor, anchoredMatterNode,
                    wrongLocalAttachment, error),
                "loaded-knee compiled attachment local-point drift was admitted");

        NumiHumanLoadedKneeMassEvidenceV1 mass;
        mass.partitions.resize(2u);
        mass.partitions[0u].donorBody = NUMI_HUMAN_KNEE_FEMUR_BODY;
        mass.partitions[1u].donorBody = NUMI_HUMAN_KNEE_TIBIA_BODY;
        mass.cookedNodes.resize(kNumiHumanLoadedKneeLoadedNodeCount);
        mass.closureSHA256 = digest(60u);
        mass.cookedNodeMassSHA256 = digest(61u);
        mass.authoredRawF32NodeMassSHA256 =
            authoring.authoredRawF32NodeMassSHA256;
        mass.executedRawF32NodeMassSHA256 =
            authoring.authoredRawF32NodeMassSHA256;
        mass.referenceCoordinateSHA256 =
            authoring.coordinates.referenceStateSHA256;
        mass.sourceGlobalNodeIndexSHA256 =
            authoring.topology.sourceGlobalNodeIndexSHA256;
        mass.executableFEMTopologySHA256 =
            authoring.topology.executableFEMTopologySHA256;
        mass.sourceAnchorOwnershipSHA256 =
            authoring.topology.anchorOwnershipSHA256;
        const auto materialWorld = materialWorldFixture(authoring);
        const auto passiveExecution = passiveExecutionFixture(
            authoring, materialWorld);
        require(digestNumiHumanLoadedKneeExecutedAnchorsV1(
                    passiveExecution.anchors,
                    mass.executedAnchorOwnershipSHA256, error), error);
        mass.executedSourcePhysicsFingerprint =
            materialWorld.physicsFingerprint;
        mass.rebasedModel = makeFreeSphereEngineModel();
        mass.sourceRigidModelFingerprint =
            engineModelFingerprint(mass.rebasedModel);
        require(bindNumiHumanLoadedKneeMassToRigidExecutionV1(
                    mass.rebasedModel, mass, error), error);
        require(mass.executedRigidModelFingerprint ==
                    engineModelFingerprint(mass.rebasedModel),
                "rebased rigid fingerprint was not retained");

        const auto assertMaterialRejectsAtomically = [
            &authoring, &mass, &passiveExecution, &error](
            const numi::matter::CompiledWorld& candidateWorld,
            const auto& candidateOwners,
            const auto& candidateRows,
            const std::filesystem::path& materialPath,
            const std::string& message) {
            auto rejected = mass;
            rejected.materialExecutionSHA256 = digest(121u);
            const auto sentinel = rejected.materialExecutionSHA256;
            require(!bindNumiHumanLoadedKneeMaterialExecutionV1(
                        authoring, materialPath, candidateWorld,
                        passiveExecution.anchors, candidateOwners,
                        candidateRows, rejected, error), message);
            require(rejected.materialExecutionSHA256 == sentinel,
                    message + " changed prior material evidence");
        };
        auto c5Mutation = materialWorld;
        c5Mutation.constitutive[0u].material.parameters[4u]
            .defaultValue += 1024.0;
        assertMaterialRejectsAtomically(
            c5Mutation, authoring.passiveOwners, passiveExecution.rows,
            exactFEBioMaterialPath(),
            "compiled c5 mutation was admitted");
        auto expandedIRMutation = materialWorld;
        require(!expandedIRMutation.constitutive[0u]
                     .material.expressions.nodes.empty(),
                "compiled material fixture has no expanded expression IR");
        expandedIRMutation.constitutive[0u]
            .material.expressions.nodes[0u].constant = std::nextafter(
                expandedIRMutation.constitutive[0u]
                    .material.expressions.nodes[0u].constant,
                std::numeric_limits<double>::infinity());
        assertMaterialRejectsAtomically(
            expandedIRMutation, authoring.passiveOwners,
            passiveExecution.rows, exactFEBioMaterialPath(),
            "compiled expanded-expression mutation was admitted");
        auto bytecodeMutation = materialWorld;
        require(!bytecodeMutation.constitutive[0u]
                     .stress[0u].instructions.empty(),
                "compiled material fixture has no stress bytecode");
        bytecodeMutation.constitutive[0u]
            .stress[0u].instructions[0u].reserved ^= 1u;
        assertMaterialRejectsAtomically(
            bytecodeMutation, authoring.passiveOwners,
            passiveExecution.rows, exactFEBioMaterialPath(),
            "compiled stress-bytecode mutation was admitted");
        auto lambdaMutation = materialWorld;
        lambdaMutation.constitutive[1u].material.parameters[5u]
            .defaultValue += 0.001;
        assertMaterialRejectsAtomically(
            lambdaMutation, authoring.passiveOwners,
            passiveExecution.rows, exactFEBioMaterialPath(),
            "compiled lambda_max mutation was admitted");
        auto fiberScaleMutation = materialWorld;
        fiberScaleMutation.constitutive[2u].material.parameters[6u]
            .defaultValue = 1.0;
        assertMaterialRejectsAtomically(
            fiberScaleMutation, authoring.passiveOwners,
            passiveExecution.rows, exactFEBioMaterialPath(),
            "passive continuum fiber_scale mutation was admitted");
        auto ownerMutation = authoring.passiveOwners;
        ownerMutation[3u].ownerID += "/wrong";
        assertMaterialRejectsAtomically(
            materialWorld, ownerMutation, passiveExecution.rows,
            exactFEBioMaterialPath(),
            "reduced passive owner mutation was admitted");
        auto rowMutation = passiveExecution.rows;
        rowMutation[4u].firstLocalPoint.x = std::bit_cast<float>(
            std::bit_cast<std::uint32_t>(
                rowMutation[4u].firstLocalPoint.x) ^ 1u);
        assertMaterialRejectsAtomically(
            materialWorld, authoring.passiveOwners, rowMutation,
            exactFEBioMaterialPath(),
            "reduced passive enthesis-row mutation was admitted");
        auto stretchMutation = passiveExecution.rows;
        stretchMutation[0u].reference.z = std::bit_cast<float>(
            std::bit_cast<std::uint32_t>(
                stretchMutation[0u].reference.z) ^ 1u);
        assertMaterialRejectsAtomically(
            materialWorld, authoring.passiveOwners, stretchMutation,
            exactFEBioMaterialPath(),
            "reduced passive source-stretch mutation was admitted");
        assertMaterialRejectsAtomically(
            materialWorld, authoring.passiveOwners,
            passiveExecution.rows,
            exactFEBioMaterialPath().parent_path() /
                "open_knee_ligament_transverse_isotropic_smooth.nmatter",
            "smooth approximation asset was admitted");
        require(bindNumiHumanLoadedKneeMaterialExecutionV1(
                    authoring, exactFEBioMaterialPath(), materialWorld,
                    passiveExecution.anchors, authoring.passiveOwners,
                    passiveExecution.rows, mass, error), error);
        require(mass.materialExecutionSHA256 !=
                    NumiHumanLoadedKneeDigest{},
                "material execution digest was not published");

        auto differentModel = mass.rebasedModel;
        differentModel.name += "_different";
        require(engineModelFingerprint(differentModel) !=
                    engineModelFingerprint(mass.rebasedModel),
                "different rigid fixture did not change model identity");
        NumiHumanLoadedKneeDigest sourceModelSHA256{};
        NumiHumanLoadedKneeDigest differentModelSHA256{};
        require(digestNumiHumanLoadedKneeSourceModelV1(
                    mass.rebasedModel, sourceModelSHA256, error), error);
        require(digestNumiHumanLoadedKneeSourceModelV1(
                    differentModel, differentModelSHA256, error), error);
        require(sourceModelSHA256 != differentModelSHA256,
                "full source EngineModel SHA-256 ignored a model mutation");
        auto unchangedMass = mass;
        require(!bindNumiHumanLoadedKneeMassToRigidExecutionV1(
                    differentModel, unchangedMass, error),
                "unrebased/different rigid execution was admitted");
        require(unchangedMass.executedRigidModelFingerprint ==
                    mass.executedRigidModelFingerprint,
                "rejected rigid binding mutated prior evidence");

        const auto baseSnapshot = snapshot(
            1u, mass.executedSourcePhysicsFingerprint);
        NumiHumanLoadedKneeDigest baseSnapshotDigest{};
        require(digestNumiHumanLoadedKneeMatterSnapshotV1(
                    baseSnapshot, baseSnapshotDigest, error), error);
        auto changedAuthority = baseSnapshot;
        changedAuthority.humanSupportHistories.push_back(
            {1.0f, 2.0f, 3.0f, 4.0f});
        NumiHumanLoadedKneeDigest changedAuthorityDigest{};
        require(digestNumiHumanLoadedKneeMatterSnapshotV1(
                    changedAuthority, changedAuthorityDigest, error), error);
        require(baseSnapshotDigest != changedAuthorityDigest &&
                    !sameMatterSnapshotAuthority(
                        baseSnapshot, changedAuthority),
                "full snapshot digest ignored non-FEM continuation state");
        auto diagnosticOnly = baseSnapshot;
        diagnosticOnly.deformableContactFailures.resize(1u);
        NumiHumanLoadedKneeDigest diagnosticDigest{};
        require(digestNumiHumanLoadedKneeMatterSnapshotV1(
                    diagnosticOnly, diagnosticDigest, error), error);
        require(baseSnapshotDigest == diagnosticDigest &&
                    sameMatterSnapshotAuthority(
                        baseSnapshot, diagnosticOnly),
                "diagnostic deformable-contact failure unexpectedly owned replay");

        std::array<NumiHumanLoadedKneeAcceptanceReceiptV1,
                   kNumiHumanLoadedKneeAcceptedStepCount> receipts{};
        for (std::uint32_t step = 1u;
             step <= kNumiHumanLoadedKneeAcceptedStepCount; ++step) {
            auto evidence = runtimeEvidence(authoring, mass, step);
            const auto* previous = step == 1u
                ? nullptr : &receipts[step - 2u];
            require(acceptNumiHumanLoadedKneeRuntimeStateV1(
                        authoring, mass, evidence, previous,
                        receipts[step - 1u], error), error);
            require(receipts[step - 1u].acceptedStepIndex == step &&
                        receipts[step - 1u].sourceOwnershipStatus ==
                            authoring.sourceOwnershipStatus &&
                        receipts[step - 1u]
                            .executedRigidModelFingerprint ==
                            mass.executedRigidModelFingerprint &&
                        receipts[step - 1u].materialExecutionSHA256 ==
                            mass.materialExecutionSHA256,
                    "accepted transaction omitted step, rigid, or material identity");
        }
        for (std::size_t index = 1u; index < receipts.size(); ++index) {
            require(receipts[index].previousTransactionSHA256 ==
                        receipts[index - 1u].transactionSHA256 &&
                        receipts[index].transactionSHA256 !=
                            receipts[index - 1u].transactionSHA256,
                    "eight-state transaction chain did not advance exactly");
        }

        auto rejectedOutput = receipts.back();
        const auto sentinelRoot = rejectedOutput.transactionSHA256;
        auto badEvidence = runtimeEvidence(authoring, mass, 1u);
        badEvidence.executedRigidModelFingerprint ^= 1u;
        require(!acceptNumiHumanLoadedKneeRuntimeStateV1(
                    authoring, mass, badEvidence, nullptr,
                    rejectedOutput, error),
                "runtime from a different rigid world was admitted");
        require(rejectedOutput.transactionSHA256 == sentinelRoot,
                "rigid mismatch rejection changed the prior root");
        badEvidence = runtimeEvidence(authoring, mass, 1u);
        badEvidence.executedRawF32NodeMassSHA256[0u] ^= 1u;
        require(!acceptNumiHumanLoadedKneeRuntimeStateV1(
                    authoring, mass, badEvidence, nullptr,
                    rejectedOutput, error),
                "different executable raw f32 node-mass distribution was admitted");
        require(rejectedOutput.transactionSHA256 == sentinelRoot,
                "node-mass rejection changed the prior root");
        badEvidence = runtimeEvidence(authoring, mass, 1u);
        badEvidence.executedMaterialExecutionSHA256[0u] ^= 1u;
        require(!acceptNumiHumanLoadedKneeRuntimeStateV1(
                    authoring, mass, badEvidence, nullptr,
                    rejectedOutput, error),
                "runtime with a different material execution was admitted");
        require(rejectedOutput.transactionSHA256 == sentinelRoot,
                "material mismatch rejection changed the prior root");
        badEvidence = runtimeEvidence(authoring, mass, 1u);
        badEvidence.contactPairNormalForceNewtons[3u] = -1.0;
        require(!acceptNumiHumanLoadedKneeRuntimeStateV1(
                    authoring, mass, badEvidence, nullptr,
                    rejectedOutput, error),
                "negative actual contact pair was admitted");
        require(rejectedOutput.transactionSHA256 == sentinelRoot,
                "contact rejection changed the prior root");
        badEvidence = runtimeEvidence(authoring, mass, 1u);
        badEvidence.executedActiveReplacements[2u].replacementFraction = 0.5;
        require(!acceptNumiHumanLoadedKneeRuntimeStateV1(
                    authoring, mass, badEvidence, nullptr,
                    rejectedOutput, error),
                "partial executed quadriceps replacement was admitted");
        require(rejectedOutput.transactionSHA256 == sentinelRoot,
                "replacement rejection changed the prior root");
        badEvidence = runtimeEvidence(authoring, mass, 1u);
        badEvidence.snapshot.sourcePhysicsFingerprint ^= 1u;
        require(!acceptNumiHumanLoadedKneeRuntimeStateV1(
                    authoring, mass, badEvidence, nullptr,
                    rejectedOutput, error),
                "snapshot from a different Matter world was admitted");
        require(rejectedOutput.transactionSHA256 == sentinelRoot,
                "Matter mismatch rejection changed the prior root");
        badEvidence = runtimeEvidence(authoring, mass, 1u);
        ++badEvidence.snapshot.controlStep;
        require(!acceptNumiHumanLoadedKneeRuntimeStateV1(
                    authoring, mass, badEvidence, nullptr,
                    rejectedOutput, error),
                "snapshot clock unrelated to the receipt step was admitted");
        require(rejectedOutput.transactionSHA256 == sentinelRoot,
                "snapshot-clock rejection changed the prior root");
        badEvidence = runtimeEvidence(authoring, mass, 1u);
        badEvidence.loadedFEMNodeFirst = 1u;
        badEvidence.snapshot.femNodes.insert(
            badEvidence.snapshot.femNodes.begin(), NMFEMNodeStateGPU{});
        require(!acceptNumiHumanLoadedKneeRuntimeStateV1(
                    authoring, mass, badEvidence, nullptr,
                    rejectedOutput, error),
                "offset FEM subarena was admitted as the loaded-knee arena");
        require(rejectedOutput.transactionSHA256 == sentinelRoot,
                "FEM-arena rejection changed the prior root");

        auto blockedAuthoring = authoring;
        blockedAuthoring.sourceOwnershipStatus = "blocked";
        badEvidence = runtimeEvidence(blockedAuthoring, mass, 2u);
        require(!acceptNumiHumanLoadedKneeRuntimeStateV1(
                    blockedAuthoring, mass, badEvidence, &receipts[0u],
                    rejectedOutput, error),
                "source ownership status changed within an accepted chain");
        require(rejectedOutput.transactionSHA256 == sentinelRoot,
                "ownership-status rejection changed the prior root");

        std::array<NumiHumanLoadedKneeRuntimeEvidenceV1,
                   kNumiHumanLoadedKneeAcceptedStepCount> accepted{};
        for (std::uint32_t index = 0u; index < accepted.size(); ++index) {
            accepted[index] = runtimeEvidence(
                authoring, mass, index + 1u);
        }
        auto replayed = accepted;
        require(verifyNumiHumanLoadedKneeRestoreReplayV1(
                    accepted, replayed, error), error);
        std::array<std::byte, 4u> externalBytes{
            std::byte{0x01}, std::byte{0x02},
            std::byte{0x03}, std::byte{0x04}};
        NumiHumanLoadedKneeDigest externalBase{};
        NumiHumanLoadedKneeDigest externalMutation{};
        NumiHumanLoadedKneeDigest externalOtherComponent{};
        require(digestNumiHumanLoadedKneeExternalComponentV1(
                    "articulated-q-v-root-time", externalBytes,
                    externalBase, error), error);
        externalBytes[2u] ^= std::byte{0x01};
        require(digestNumiHumanLoadedKneeExternalComponentV1(
                    "articulated-q-v-root-time", externalBytes,
                    externalMutation, error) &&
                    externalMutation != externalBase,
                "external full-state digest ignored a one-byte mutation");
        require(digestNumiHumanLoadedKneeExternalComponentV1(
                    "muscle-activation-tendon-transfer-corrections",
                    externalBytes, externalOtherComponent, error) &&
                    externalOtherComponent != externalMutation,
                "external full-state digest ignored component-domain identity");
        replayed[3u].executedMaterialExecutionSHA256[0u] ^= 1u;
        require(!verifyNumiHumanLoadedKneeRestoreReplayV1(
                    accepted, replayed, error),
                "restore/replay ignored material execution identity");
        replayed = accepted;
        replayed[2u].contactPairNormalForceNewtons[4u] =
            std::nextafter(
                replayed[2u].contactPairNormalForceNewtons[4u],
                std::numeric_limits<double>::infinity());
        require(!verifyNumiHumanLoadedKneeRestoreReplayV1(
                    accepted, replayed, error),
                "restore/replay ignored accepted pair-force evidence");
        replayed = accepted;
        replayed[5u].snapshot.environmentParameters.push_back(1.0f);
        require(!verifyNumiHumanLoadedKneeRestoreReplayV1(
                    accepted, replayed, error),
                "restore/replay ignored environment continuation state");
        auto relabeledAccepted = accepted;
        auto relabeledReplay = accepted;
        relabeledAccepted[2u].snapshot.controlStep += 17u;
        relabeledReplay[2u].snapshot.controlStep += 17u;
        require(!verifyNumiHumanLoadedKneeRestoreReplayV1(
                    relabeledAccepted, relabeledReplay, error),
                "restore/replay admitted a mutually matching relabeled clock");
        auto shiftedAccepted = accepted;
        auto shiftedReplay = accepted;
        shiftedAccepted[4u].loadedFEMNodeFirst = 1u;
        shiftedReplay[4u].loadedFEMNodeFirst = 1u;
        shiftedAccepted[4u].snapshot.femNodes.insert(
            shiftedAccepted[4u].snapshot.femNodes.begin(),
            NMFEMNodeStateGPU{});
        shiftedReplay[4u].snapshot.femNodes.insert(
            shiftedReplay[4u].snapshot.femNodes.begin(),
            NMFEMNodeStateGPU{});
        require(!verifyNumiHumanLoadedKneeRestoreReplayV1(
                    shiftedAccepted, shiftedReplay, error),
                "restore/replay admitted a mutually matching shifted FEM arena");

        std::cout
            << "loaded_knee_acceptance_test=passed"
            << " chain_steps=" << receipts.size()
            << " rigid_fingerprint="
            << mass.executedRigidModelFingerprint
            << " material_execution_bound=true"
            << " diagnostic_deformable_contact_failures_excluded=true\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "loaded_knee_acceptance_test=failed reason="
                  << exception.what() << "\n";
        return 1;
    }
}

#include <Foundation/Foundation.h>

#include "numi/matter/matter.hpp"
#include "vascular_cavity_fixture.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace numi::matter;
namespace fixture = vascular_cavity_fixture;

namespace {
void requireCondition(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string stringValue(id value, const char* label) {
    requireCondition([value isKindOfClass:[NSString class]], std::string(label) + " is not a string");
    return std::string([(NSString*)value UTF8String]);
}

std::uint32_t u32Value(id value, const char* label) {
    requireCondition([value isKindOfClass:[NSNumber class]], std::string(label) + " is not a number");
    const unsigned long long raw = [(NSNumber*)value unsignedLongLongValue];
    requireCondition(raw <= std::numeric_limits<std::uint32_t>::max(), std::string(label) + " exceeds uint32");
    return static_cast<std::uint32_t>(raw);
}

id requiredValue(NSDictionary* dictionary, NSString* key) {
    id value = [dictionary objectForKey:key];
    requireCondition(value != nil, std::string("missing manifest key: ") + [key UTF8String]);
    return value;
}

std::uint64_t hexWord(const char* text, std::size_t offset) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 16; ++i) {
        const char c = text[offset + i];
        unsigned digit = 0;
        if (c >= '0' && c <= '9') {
            digit = static_cast<unsigned>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            digit = static_cast<unsigned>(c - 'a' + 10);
        } else if (c >= 'A' && c <= 'F') {
            digit = static_cast<unsigned>(c - 'A' + 10);
        } else {
            throw std::runtime_error("SHA-256 contains a non-hex digit");
        }
        value = (value << 4u) | digit;
    }
    return value;
}

std::array<std::uint64_t, 4> shaWords(id value, const char* label) {
    const std::string text = stringValue(value, label);
    requireCondition(text.size() == 64, std::string(label) + " is not a SHA-256 hex string");
    return {hexWord(text.data(), 0), hexWord(text.data(), 16),
            hexWord(text.data(), 32), hexWord(text.data(), 48)};
}

std::string messages(const CompileResult& result) {
    std::string output;
    for (const auto& diagnostic : result.diagnostics) {
        output += diagnostic.message;
        output += "; ";
    }
    return output;
}

struct Binding {
    std::uint32_t sourceIndex = 0;
    std::uint32_t compartment = 0;
    std::string label;
    std::string semanticId;
    std::string memberId;
    std::string sourceHash;
};

struct ParsedManifest {
    std::array<std::uint64_t, 4> identity{};
    std::array<std::uint64_t, 4> moments{};
    std::array<std::uint64_t, 4> inventory{};
    std::vector<Binding> bindings;
};

ParsedManifest readManifest(const std::string& path) {
    NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
    NSError* readError = nil;
    NSData* data = [NSData dataWithContentsOfFile:nsPath options:0 error:&readError];
    requireCondition(data != nil, "cannot read Human bridge manifest");
    NSError* jsonError = nil;
    id json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
    requireCondition(json != nil && [json isKindOfClass:[NSDictionary class]],
                     jsonError == nil ? "Human bridge is not a JSON object"
                                       : std::string([[jsonError localizedDescription] UTF8String]));
    NSDictionary* root = (NSDictionary*)json;
    requireCondition(stringValue(requiredValue(root, @"schema"), "schema") ==
                         "HumanPack.organ-blood-cavity-bridge.v1",
                     "unexpected Human bridge schema");
    ParsedManifest result;
    result.identity = shaWords(requiredValue(root, @"identity_sha256"), "identity_sha256");
    NSDictionary* receipt = (NSDictionary*)requiredValue(root, @"source_moment_receipt");
    result.moments = shaWords(requiredValue(receipt, @"sha256"), "source moment receipt SHA");
    result.inventory = shaWords(requiredValue(receipt, @"inventory_sha256"), "source inventory SHA");
    NSArray* rows = (NSArray*)requiredValue(root, @"bindings");
    requireCondition([rows isKindOfClass:[NSArray class]] && rows.count == 4,
                     "Human bridge must contain exactly four chamber bindings");
    const std::array<std::uint32_t, 4> expectedIndices{15, 16, 19, 20};
    for (NSUInteger i = 0; i < rows.count; ++i) {
        requireCondition([rows[i] isKindOfClass:[NSDictionary class]], "binding is not an object");
        NSDictionary* row = (NSDictionary*)rows[i];
        Binding binding;
        binding.sourceIndex = u32Value(requiredValue(row, @"source_index"), "source_index");
        requireCondition(binding.sourceIndex == expectedIndices[i], "unexpected cardiac source index");
        binding.compartment = u32Value(requiredValue(row, @"compartment_stable_identifier"),
                                        "compartment_stable_identifier");
        binding.label = stringValue(requiredValue(row, @"source_label"), "source_label");
        binding.semanticId = stringValue(requiredValue(row, @"semantic_id"), "semantic_id");
        binding.memberId = stringValue(requiredValue(row, @"member_id"), "member_id");
        binding.sourceHash = stringValue(requiredValue(row, @"source_member_sha256"),
                                         "source_member_sha256");
        requireCondition(binding.compartment != 0 && !binding.label.empty() &&
                             !binding.semanticId.empty() && !binding.memberId.empty(),
                         "binding has an empty identity field");
        requireCondition(binding.sourceHash.size() == 64, "binding source hash is not SHA-256");
        (void)hexWord(binding.sourceHash.data(), 0);
        (void)hexWord(binding.sourceHash.data(), 16);
        (void)hexWord(binding.sourceHash.data(), 32);
        (void)hexWord(binding.sourceHash.data(), 48);
        result.bindings.push_back(std::move(binding));
    }
    return result;
}

WorldSource makeNativeAdmissionSource(const ParsedManifest& manifest,
                                      std::vector<fixture::Shell>& shells) {
    WorldSource source = fixture::world(false, .001, false);
    source.environmentCount = 2;
    source.frameTimestep = .001;
    source.gravity = {0.0, 0.0, 0.0};
    source.objects.clear();

    auto& network = source.vascular;
    network.species.clear();
    network.compartments.clear();
    network.connections.clear();
    network.tissues.clear();
    network.exchanges.clear();
    network.cavities.clear();
    network.contentIdentity = manifest.identity;
    network.sourceIdentity = manifest.moments;
    network.authoredIdentity = manifest.inventory;
    network.species.push_back({1u, "human_bridge_fixture_tracer", 1.0e-6, 1.0e-5});

    shells.clear();
    shells.reserve(manifest.bindings.size());
    for (std::size_t i = 0; i < manifest.bindings.size(); ++i) {
        const Binding& binding = manifest.bindings[i];
        fixture::Shell shell = fixture::hollowShell();
        for (auto& point : shell.object.femNodes) {
            point[0] += 0.20 * static_cast<double>(i);
        }
        shell.object.name = "human_bridge_fixture/" + binding.memberId;
        shell.object.materialIndex = 0u;
        source.objects.push_back(shell.object);
        shells.push_back(std::move(shell));

        const double cavityVolume = fixture::volume(
            shells.back().object.femNodes, shells.back().lumenFaces);
        requireCondition(cavityVolume > 0.0, "synthetic cavity volume is not positive");

        VascularCompartmentSource compartment;
        compartment.stableIdentifier = binding.compartment;
        compartment.anatomicalIdentifier = "Human/BodyParts3D/" + binding.memberId;
        compartment.initialVolume = cavityVolume;
        compartment.volumeScale = 1.0e-6;
        compartment.volumeResidualTolerance = 1.0e-5;
        compartment.initialSpeciesAmounts = {0.0};
        compartment.pressureLaw = VascularPressureLaw::deformingCavity;
        network.compartments.push_back(std::move(compartment));

        VascularTissueSource owner;
        owner.stableIdentifier = 200u + static_cast<std::uint32_t>(i);
        owner.anatomicalIdentifier = "Human/organ/" + binding.memberId;
        owner.volume = 1.0e-6;
        owner.initialSpeciesAmounts = {0.0};
        owner.objectIndex = static_cast<std::uint32_t>(i);
        owner.bloodCompartment = binding.compartment;
        // This is an explicit native admission fixture density. It is not a
        // subject-specific calibration and is intentionally not copied into
        // the Human source bridge.
        owner.bloodDensity = 1060.0;
        const double weight = 1.0 / static_cast<double>(shells.back().innerNodes.size());
        for (const auto node : shells.back().innerNodes) {
            owner.femRegion.push_back({node, weight});
        }
        network.tissues.push_back(std::move(owner));

        VascularCavitySource cavity;
        cavity.stableIdentifier = 100u + static_cast<std::uint32_t>(i);
        cavity.compartment = binding.compartment;
        cavity.objectIndex = static_cast<std::uint32_t>(i);
        cavity.initialPressure = 0.0;
        cavity.pressureScale = 100.0;
        cavity.geometryResidualTolerance = 1.0e-5;
        cavity.sourceIdentity = shaWords(
            [NSString stringWithUTF8String:binding.sourceHash.c_str()], "binding source hash");
        cavity.mechanicalIdentity = manifest.moments;
        for (std::size_t face = 0; face < shells.back().lumenFaces.size(); ++face) {
            cavity.faces.push_back({
                100000u + static_cast<std::uint32_t>(i * 100u + face),
                shells.back().lumenFaces[face],
                VascularCavityFaceRole::materialWall});
        }
        network.cavities.push_back(std::move(cavity));
    }
    return source;
}

bool sameWords(const nm_u64* actual, const std::array<std::uint64_t, 4>& expected) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            return false;
        }
    }
    return true;
}

std::string cookedName(const std::vector<std::uint8_t>& names, std::uint32_t offset) {
    requireCondition(offset < names.size(), "cooked name offset is outside the native name arena");
    const auto begin = names.begin() + offset;
    const auto end = std::find(begin, names.end(), static_cast<std::uint8_t>(0));
    requireCondition(end != names.end(), "cooked name is not NUL terminated");
    return std::string(begin, end);
}

void run(const std::string& manifestPath) {
    const ParsedManifest manifest = readManifest(manifestPath);
    std::vector<fixture::Shell> shells;
    const WorldSource source = makeNativeAdmissionSource(manifest, shells);
    const CompileResult compiled = compileWorld(source, {.maximumRateExponent = 0});
    requireCondition(compiled.succeeded(), "native Human bridge compile failed: " + messages(compiled));
    std::string layoutError;
    requireCondition(validateCompiledWorldLayout(compiled.world, &layoutError),
                     "native Human bridge layout failed: " + layoutError);

    const auto& world = compiled.world;
    const auto& vascular = world.vascular;
    requireCondition(sameWords(vascular.identity.source, manifest.moments),
                     "native network source identity did not retain the moments receipt SHA");
    requireCondition(sameWords(vascular.identity.authored, manifest.inventory),
                     "native network authored identity did not retain the source inventory SHA");
    requireCondition(vascular.compartments.size() == 4 && vascular.cavities.size() == 4 &&
                         vascular.tissues.size() == 4,
                     "native world did not retain four compartments, cavities, and tissue owners");
    requireCondition(vascular.tissueBindings.size() == 32,
                     "native world did not retain all four eight-node owner regions");

    for (std::size_t i = 0; i < manifest.bindings.size(); ++i) {
        const Binding& binding = manifest.bindings[i];
        requireCondition(vascular.compartments[i].identity.x == binding.compartment,
                         "native compartment stable ID differs from Human bridge");
        requireCondition(cookedName(vascular.names, vascular.compartments[i].identity.y) ==
                             "Human/BodyParts3D/" + binding.memberId,
                         "native compartment anatomical identity differs from Human member");
        requireCondition(vascular.cavities[i].identity.x == 100u + i &&
                             vascular.cavities[i].identity.y == i &&
                             vascular.cavities[i].identity.z == i,
                         "native cavity identity or object binding is not canonical");
        requireCondition(sameWords(vascular.cavities[i].sourceIdentity,
                                      shaWords([NSString stringWithUTF8String:binding.sourceHash.c_str()],
                                               "binding source hash")),
                         "native cavity source identity differs from Human member hash");
        requireCondition(sameWords(vascular.cavities[i].mechanicalIdentity, manifest.moments),
                         "native cavity mechanical provenance differs from moments receipt");
        requireCondition(vascular.compartmentCavity[i] == i,
                         "native compartment-to-cavity ownership is not one-to-one");
        requireCondition(vascular.tissues[i].identity.w == i + 1u &&
                             vascular.tissues[i].physical.y == 1060.0f &&
                             vascular.tissues[i].physical.z > 0.0f,
                         "native tissue owner did not admit explicit density and mass");
        requireCondition(vascular.tissues[i].region.y == 8u,
                         "native tissue owner region cardinality changed");
    }

    std::cout << "human_organ_blood_native_binding=pass\n"
              << "abi=" << NM_MATTER_ABI_VERSION
              << " chamber_bindings=" << vascular.cavities.size()
              << " tissue_owners=" << vascular.tissues.size()
              << " fem_binding_nodes=" << vascular.tissueBindings.size() << "\n"
              << "source_moments_identity=pass source_member_hashes=pass "
                 "compartment_cavity_ownership=pass blood_owner_mass_cooking=pass\n"
              << "fixture_geometry=synthetic native_density_calibration=unqualified "
                 "pressure_momentum=unqualified two_way_transfer=unqualified "
                 "standing_walking=unqualified\n";
}
} // namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        try {
            requireCondition(argc == 2, "usage: numi-matter-vascular-human-binding-check BRIDGE_JSON");
            run(argv[1]);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "human organ blood native binding: " << error.what() << "\n";
            return 1;
        }
    }
}


#include "numi/matter/matter.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace numi::matter;
namespace {
unsigned checks = 0;
void require(bool ok, const std::string& reason) {
    ++checks;
    if (!ok) throw std::runtime_error(reason);
}
std::string diagnostics(const CompileResult& result) {
    std::string message;
    for (const auto& d : result.diagnostics) message += d.message + "; ";
    return message;
}
WorldSource fixture(bool framed = true) {
    const auto material = parseMatterFile(std::filesystem::path(__FILE__).parent_path().parent_path()
        / "materials" / "damageable_silicone.nmatter");
    if (!material.succeeded()) throw std::runtime_error("fixture material parse failed");
    WorldSource source;
    source.materials.push_back(material.material);
    source.gravity = {0, 0, 0};
    ObjectSource object;
    object.name = "synthetic_two_tet_material_field";
    object.materialIndex = 0;
    object.representation = Representation::fem;
    object.deformableContact = false;
    object.mixedFEM = false;
    object.femNodes = {{{0,0,0}}, {{.01,0,0}}, {{0,.01,0}}, {{0,0,.01}}, {{0,0,-.01}}};
    object.tetrahedra = {{{0,1,2,3}}, {{0,2,1,4}}};
    if (framed) {
        object.femMaterialFrameRotations = {{0,0,0,1}, {0,std::sin(.3),0,std::cos(.3)}};
        // Synthetic provenance identifier; never represents an anatomical source.
        object.femMaterialFrameSourceIdentity = {1,2,3,4};
    }
    source.objects.push_back(object);
    return source;
}
void checkContactConnectedClockLock() {
    WorldSource source = fixture(false);
    source.frameTimestep = 1.0 / 60.0;
    source.objects.front().deformableContact = true;
    source.objects.front().characteristicLength = 0.2;

    ObjectSource second = source.objects.front();
    second.name = "synthetic_contact_clock_member_b";
    second.characteristicLength = 0.1;
    for (auto& node : second.femNodes) node[0] += 0.1;
    source.objects.push_back(second);

    ObjectSource disabled = second;
    disabled.name = "synthetic_contact_clock_disabled";
    disabled.characteristicLength = 0.01;
    disabled.deformableContact = false;
    disabled.deformableSelfContact = false;
    for (auto& node : disabled.femNodes) node[0] += 0.1;
    source.objects.push_back(disabled);

    CompileOptions options;
    options.maximumRateExponent = 8u;
    options.emitSpecializedMetal = false;
    const auto defaultCook = compileWorld(source, options);
    require(defaultCook.succeeded(),
        "default contact-clock world failed: " + diagnostics(defaultCook));
    const auto defaultAgain = compileWorld(source, options);
    require(defaultAgain.succeeded() &&
            defaultAgain.world.fingerprint == defaultCook.world.fingerprint &&
            std::memcmp(defaultAgain.world.schedulers.data(),
                defaultCook.world.schedulers.data(),
                defaultCook.world.schedulers.size() *
                    sizeof(NMSchedulerStateGPU)) == 0,
        "default contact-clock policy is not deterministic");
    require(defaultCook.world.schedulers[0].baseExponent <
                defaultCook.world.schedulers[1].baseExponent &&
            defaultCook.world.schedulers[1].baseExponent <
                defaultCook.world.schedulers[2].baseExponent,
        "clock fixture does not distinguish participant and disabled rates");

    options.synchronizeDeformableContactClocks = true;
    const auto locked = compileWorld(source, options);
    require(locked.succeeded(),
        "locked contact-clock world failed: " + diagnostics(locked));
    const auto expected = defaultCook.world.schedulers[1].baseExponent;
    require(locked.world.schedulers[0].baseExponent == expected &&
            locked.world.schedulers[1].baseExponent == expected &&
            locked.world.objects[0].solver.x == expected &&
            locked.world.objects[1].solver.x == expected,
        "contact-capable FEM objects did not share their finest authored rate");
    require(locked.world.schedulers[2].baseExponent ==
                defaultCook.world.schedulers[2].baseExponent &&
            locked.world.objects[2].solver.x ==
                defaultCook.world.objects[2].solver.x,
        "clock lock changed a contact-disabled object");
    for (std::size_t i = 0u; i < locked.world.schedulers.size(); ++i)
        require(locked.world.schedulers[i].activeExponent ==
                    locked.world.schedulers[i].baseExponent &&
                locked.world.schedulers[i].requestedExponent ==
                    locked.world.schedulers[i].baseExponent,
            "compiled scheduler did not initialize at the locked base rate");
    require(locked.world.fingerprint != defaultCook.world.fingerprint &&
            locked.world.physicsFingerprint !=
                defaultCook.world.physicsFingerprint,
        "clock profile is absent from cooked identity");
    require(locked.world.fem.nodes.size() ==
                defaultCook.world.fem.nodes.size() &&
            std::memcmp(locked.world.fem.nodes.data(),
                defaultCook.world.fem.nodes.data(),
                locked.world.fem.nodes.size() *
                    sizeof(NMFEMNodeStateGPU)) == 0,
        "clock profile altered source FEM geometry or mass");
}
void denied(const WorldSource& source, const char* role) {
    const auto result = compileWorld(source);
    require(!result.succeeded(), std::string("source accepted: ") + role);
    std::cout << "denied_source=" << role << " diagnostic=" << diagnostics(result) << '\n';
}
void deniedCooked(CompiledWorld world, const char* role) {
    world.fingerprint = compiledWorldFingerprint(world);
    std::string error;
    require(!validateCompiledWorldLayout(world, &error), std::string("cooked accepted: ") + role);
    std::cout << "denied_cooked=" << role << " diagnostic=" << error << '\n';
}
template<class T> bool sameBytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()*sizeof(T)) == 0;
}
}
int main() {
    try {
        checkContactConnectedClockLock();
        const auto source = fixture();
        const auto framed = compileWorld(source);
        require(framed.succeeded(), "framed cook failed: " + diagnostics(framed));
        const auto legacy = compileWorld(fixture(false));
        require(legacy.succeeded(), "legacy cook failed: " + diagnostics(legacy));
        const auto& world = framed.world;
        require(sameBytes(world.fem.nodes, legacy.world.fem.nodes), "frames changed nodes/masses/rest coordinates");
        require(world.fem.tetrahedra.size() == 2, "tet count changed");
        for (std::size_t i=0; i<world.fem.tetrahedra.size(); ++i) {
            auto tet = world.fem.tetrahedra[i];
            tet.materialFrameRotation = {};
            require(std::memcmp(&tet, &legacy.world.fem.tetrahedra[i], sizeof(tet)) == 0,
                "frames changed topology/rest gradient/volume/material identity");
        }
        require((world.objects[0].flags & NM_OBJECT_FEM_MATERIAL_FRAME) != 0, "frame flag absent");
        require(world.objects[0].materialFrameSourceIdentity[3] == 4, "provenance identity lost");
        require(world.fem.tetrahedra[0].materialFrameRotation.w == 1
            && world.fem.tetrahedra[1].materialFrameRotation.y == float(std::sin(.3)), "frame order changed");
        auto sourceChanged = source;
        sourceChanged.objects[0].femMaterialFrameSourceIdentity[3] ^= 1;
        auto changed = compileWorld(sourceChanged);
        require(changed.succeeded() && changed.world.fingerprint != world.fingerprint
            && changed.world.physicsFingerprint != world.physicsFingerprint, "source field identity unbound");
        sourceChanged = source;
        sourceChanged.objects[0].femMaterialFrameRotations[1] = {0,0,0,1};
        changed = compileWorld(sourceChanged);
        require(changed.succeeded() && changed.world.fingerprint != world.fingerprint
            && changed.world.physicsFingerprint != world.physicsFingerprint, "frame values unbound");
        auto invalid = source;
        invalid.objects[0].femMaterialFrameSourceIdentity = {}; denied(invalid,"missing identity");
        invalid=source; invalid.objects[0].femMaterialFrameRotations.clear(); denied(invalid,"identity without frames");
        invalid=source; invalid.objects[0].femMaterialFrameRotations.pop_back(); denied(invalid,"count mismatch");
        invalid=source; invalid.objects[0].femMaterialFrameRotations[1]={}; denied(invalid,"zero quaternion");
        invalid=source; invalid.objects[0].femMaterialFrameRotations[1]={0,0,0,2}; denied(invalid,"nonunit quaternion");
        invalid=source; invalid.objects[0].femMaterialFrameRotations[1][0]=std::numeric_limits<double>::quiet_NaN(); denied(invalid,"NaN quaternion");
        invalid=source; invalid.objects[0].femMaterialFrameRotations[1][0]=std::numeric_limits<double>::infinity(); denied(invalid,"infinite quaternion");
        invalid=source; invalid.objects[0].representation=Representation::rigid; denied(invalid,"non-FEM representation");
        invalid=source; invalid.objects[0].adaptive=true; denied(invalid,"adaptive frame object");
        invalid=source; invalid.objects[0].mutationPolicy.enabled=true; denied(invalid,"mutable frame object");
        invalid=source; invalid.objects[0].mutationCommands.push_back({}); denied(invalid,"authored mutation command");
        invalid=source; invalid.objects.push_back(fixture(false).objects[0]); invalid.objects[1].adaptive=true;
        denied(invalid,"adaptive neighboring object");
        invalid=source; invalid.objects.push_back(fixture(false).objects[0]); invalid.objects[1].mutationPolicy.enabled=true;
        denied(invalid,"mutable neighboring object");
        auto bad = world;
        bad.fem.tetrahedra[0].materialFrameRotation.w=2; deniedCooked(bad,"nonunit frame");
        bad=world; bad.fem.tetrahedra[0].materialFrameRotation.x=std::numeric_limits<float>::quiet_NaN(); deniedCooked(bad,"NaN frame");
        bad=world; bad.objects[0].flags &= ~NM_OBJECT_FEM_MATERIAL_FRAME; deniedCooked(bad,"identity without flag");
        bad=world; for (auto& word : bad.objects[0].materialFrameSourceIdentity) word=0; deniedCooked(bad,"flag without identity");
        bad=legacy.world; bad.fem.tetrahedra[0].materialFrameRotation.w=1; deniedCooked(bad,"invented legacy frame");
        bad=world; bad.objects[0].flags |= NM_OBJECT_ADAPTIVE; deniedCooked(bad,"cooked adaptive world");
        const auto path = std::filesystem::temp_directory_path() / ("numi-frame-"
            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".nmpkg");
        std::string error;
        require(writePackage(framed,path,&error), "package write: " + error);
        CompiledWorld decoded;
        require(readPackage(path,decoded,nullptr,&error), "package read: " + error);
        std::filesystem::remove(path);
        require(decoded.fingerprint==world.fingerprint && decoded.physicsFingerprint==world.physicsFingerprint,
            "package identity drift");
        require(sameBytes(decoded.fem.tetrahedra,world.fem.tetrahedra)
            && sameBytes(decoded.objects,world.objects), "package field byte drift");
        std::cout << "PASS fem_material_frame_compiler contact_clock_lock=passed checks=" << checks
            << " physical_steps=0 abi=" << NM_MATTER_ABI_VERSION << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL fem_material_frame_compiler checks=" << checks << " error=" << error.what() << '\n';
        return 1;
    }
}

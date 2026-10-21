#include "numi/matter/matter.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>

using namespace numi::matter;
namespace {
unsigned checks=0;
void require(bool condition,const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
WorldSource fixture() {
    auto parsed=parseMatterFile(std::filesystem::path(__FILE__).parent_path().parent_path()/
                                "materials/starch2007_finite_glue_elastic.nmatter");
    require(parsed.succeeded(),"fixture material parse failed");
    WorldSource source;
    source.materials={parsed.material}; source.gravity={0,0,0};
    ObjectSource object;
    object.name="immutable-surface-map"; object.representation=Representation::fem;
    object.mixedFEM=false;
    object.femNodes={{{0,0,0}},{{.01,0,0}},{{0,.01,0}},{{0,0,.01}},{{0,0,-.01}}};
    object.tetrahedra={{{0,1,2,3}},{{0,2,1,4}}};
    source.objects={object};
    return source;
}
template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size() && (a.empty() ||
        std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
void reject(const WorldSource& source) {
    require(!compileWorld(source).succeeded(),"unsupported mutable source admitted");
}
void verifyRigidContactAdmission() {
    auto source=fixture();
    RigidProxySource plane;
    plane.shape=NM_RIGID_PLANE;
    plane.materialIndex=0u;
    plane.localCenter={0.0,0.0,1.0};
    plane.radiusOrOffset=0.0;
    source.rigidProxies.push_back(plane);
    const auto defaultCook=compileWorld(source);
    require(defaultCook.succeeded(),"default rigid-contact fixture failed to compile");
    require(source.objects[0].rigidContactEnabled,
            "source-object rigid contact did not default to admitted");
    require(defaultCook.world.contact.pairs.size()==5u,
            "default source-object gate changed legacy FEM plane pair coverage");
    auto explicitDefault=source;
    explicitDefault.objects[0].rigidContactEnabled=true;
    const auto explicitDefaultCook=compileWorld(explicitDefault);
    require(explicitDefaultCook.succeeded() &&
                explicitDefaultCook.world.fingerprint==defaultCook.world.fingerprint &&
                explicitDefaultCook.world.physicsFingerprint==defaultCook.world.physicsFingerprint,
            "explicit default gate changed legacy compiled identities");

    auto filtered=source;
    filtered.objects[0].rigidContactEnabled=false;
    const auto filteredCook=compileWorld(filtered);
    require(filteredCook.succeeded() && filteredCook.world.contact.pairs.empty(),
            "disabled object still received rigid-proxy pair rows");
    require(same(filteredCook.world.fem.nodes,defaultCook.world.fem.nodes) &&
                same(filteredCook.world.fem.tetrahedra,defaultCook.world.fem.tetrahedra) &&
                same(filteredCook.world.fem.surfaceFaces,defaultCook.world.fem.surfaceFaces) &&
                same(filteredCook.world.materials,defaultCook.world.materials),
            "rigid-contact admission altered FEM continuum topology or material");
    require(filteredCook.world.physicsFingerprint!=defaultCook.world.physicsFingerprint,
            "rigid-contact admission was omitted from compiled physics identity");
    auto duplicateNodes=filtered;
    duplicateNodes.objects[0].femContactNodes={0u,0u};
    require(!compileWorld(duplicateNodes).succeeded(),
            "disabled rigid-contact object bypassed duplicate-node source validation");

    auto selected=source;
    selected.objects[0].femContactNodes={0u,4u};
    const auto selectedCook=compileWorld(selected);
    require(selectedCook.succeeded() && selectedCook.world.contact.pairs.size()==2u,
            "explicit rigid-contact node selection changed");
    for(const auto& pair:selectedCook.world.contact.pairs)
        require(pair.objectIndex==0u &&
                    (pair.continuumNode==0u || pair.continuumNode==4u),
                "explicit node filter admitted a nonselected node");

    auto mixed=fixture();
    mixed.compactImmutableFEMSurfaces=true;
    auto mpmMaterial=parseMatterFile(std::filesystem::path(__FILE__).parent_path().parent_path()/
                                     "materials/silicone.nmatter");
    require(mpmMaterial.succeeded(),"rigid-contact MPM material parse failed");
    mixed.materials.push_back(mpmMaterial.material);
    ObjectSource mpm;
    mpm.name="rigid-contact-disabled-mpm";
    mpm.materialIndex=1u;
    mpm.representation=Representation::mpm;
    mpm.twoWayCoupling=false;
    mpm.rigidContactEnabled=false;
    mpm.characteristicLength=0.02;
    mpm.mpmGridMinimum={0.0,0.0,0.0};
    mpm.mpmGridMaximum={0.06,0.06,0.06};
    mpm.particles.push_back(ParticleSource{
        {0.03,0.03,0.03},{0.0,0.0,0.0},0.001,1.0e-6,true});
    mixed.objects.push_back(mpm);
    mixed.rigidProxies.push_back(plane);
    const auto mixedCook=compileWorld(mixed);
    require(mixedCook.succeeded(),"mixed rigid-contact admission fixture failed");
    std::size_t femPairs=0u,mpmPairs=0u;
    for(const auto& pair:mixedCook.world.contact.pairs) {
        if(pair.objectIndex==0u) ++femPairs;
        if(pair.objectIndex==1u) ++mpmPairs;
    }
    require(femPairs==5u && mpmPairs==0u,
            "object-level rigid-contact gate did not independently suppress MPM pairs");
}
}

void verifyExternalCandidateAuthority() {
    auto source=fixture();
    auto baseline=compileWorld(source);
    require(baseline.succeeded(),"external authority baseline failed");
    source.externalCoupledCandidateAuthority=true;
    source.articulatedDofCapacity=6u;
    source.articulatedQCapacity=7u;
    auto external=compileWorld(source);
    require(external.succeeded(),"external source authority did not compile");
    require(external.world.dispatch.rigidGeneralizedCapacity==6u &&
            external.world.dispatch.rigidQCapacity==7u &&
            (external.world.dispatch.flags &
             NM_MATTER_EXTERNAL_COUPLED_CANDIDATE_AUTHORITY)!=0u,
            "external source authority lost its exact domains");
    require(external.world.fingerprint!=baseline.world.fingerprint &&
            external.world.physicsFingerprint!=baseline.world.physicsFingerprint,
            "external source authority aliases source-free identity");
    std::string error;
    require(validateCompiledWorldLayout(external.world,&error),
            "external source authority failed layout validation");
    auto stripped=external.world;
    stripped.dispatch.flags &= ~NM_MATTER_EXTERNAL_COUPLED_CANDIDATE_AUTHORITY;
    require(!validateCompiledWorldLayout(stripped,&error),
            "unowned generalized rows admitted after authority removal");
    source.compactImmutableFEMSurfaces=true;
    auto compact=compileWorld(source);
    require(compact.succeeded(),"external authority canonical recursion failed");
    require(compact.world.physicsFingerprint==external.world.physicsFingerprint,
            "compact external source changed allocation-independent physics");
    auto changed=source;
    changed.articulatedDofCapacity=7u;changed.articulatedQCapacity=8u;
    auto resized=compileWorld(changed);
    require(resized.succeeded() &&
            resized.world.physicsFingerprint!=compact.world.physicsFingerprint,
            "different external source shape aliases physics identity");
    changed=source;changed.articulatedQCapacity=0u;
    require(!compileWorld(changed).succeeded(),
            "external source with no q domain admitted");
}

int main() {
    try {
        verifyExternalCandidateAuthority();
        verifyRigidContactAdmission();
        auto source=fixture(); auto legacy=compileWorld(source);
        require(legacy.succeeded(),"legacy fixture compilation failed");
        source.compactImmutableFEMSurfaces=true;
        auto compact=compileWorld(source);
        if (!compact.succeeded()) for (auto& d:compact.diagnostics) std::cerr<<d.message<<'\n';
        require(compact.succeeded(),"compact fixture compilation failed");
        require(legacy.world.fem.surfaceFaces.size()==7,"legacy unique-face graph changed");
        require(compact.world.fem.surfaceFaces.size()==6,"compact exterior count incorrect");
        require(compact.world.dispatch.surfaceFaceCount==6 &&
            (compact.world.dispatch.flags & NM_MATTER_STATIC_FEM_SURFACE_MAP),"compact dispatch contract missing");
        require(same(legacy.world.fem.nodes,compact.world.fem.nodes) &&
                same(legacy.world.fem.tetrahedra,compact.world.fem.tetrahedra) &&
                same(legacy.world.materials,compact.world.materials) &&
                same(legacy.world.parameters,compact.world.parameters),"surface optimization changed mechanics inputs");
        require(compact.world.physicsFingerprint==legacy.world.physicsFingerprint &&
                compact.world.fingerprint!=legacy.world.fingerprint,"physics/allocation identities are not separated");
        std::set<unsigned> ids;
        for (auto& f:compact.world.fem.surfaceFaces) ids.insert(4*f.adjacency.x+f.sides.x);
        require(ids.size()==6 && !ids.contains(3) && !ids.contains(7),"source face IDs lost shared-face exclusion");
        auto package=std::filesystem::temp_directory_path()/
            ("numi-static-surfaces-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".nmpkg");
        std::string error; CompiledWorld decoded;
        require(writePackage(compact,package,&error),"compact package write failed");
        require(readPackage(package,decoded,nullptr,&error),"compact package read failed");
        std::filesystem::remove(package);
        require(decoded.fingerprint==compact.world.fingerprint && decoded.dispatch.surfaceFaceCount==6,
                "compact package roundtrip changed map identity");
        auto corrupt=compact.world; corrupt.fem.surfaceFaces.pop_back();
        corrupt.dispatch.surfaceFaceCount=corrupt.fem.surfaceFaces.size();
        corrupt.fingerprint=compiledWorldFingerprint(corrupt);
        require(!validateCompiledWorldLayout(corrupt,&error),"omitted exterior face admitted");
        corrupt=compact.world; corrupt.fem.surfaceFaces.push_back(corrupt.fem.surfaceFaces.front());
        corrupt.dispatch.surfaceFaceCount=corrupt.fem.surfaceFaces.size();
        corrupt.fingerprint=compiledWorldFingerprint(corrupt);
        require(!validateCompiledWorldLayout(corrupt,&error),"duplicate exterior face admitted");
        auto separated=source;
        separated.objects[0].femNodes={{{0,0,0}},{{.01,0,0}},{{0,.01,0}},{{0,0,.01}},
                                      {{0,0,0}},{{.01,0,0}},{{0,.01,0}},{{0,0,-.01}}};
        separated.objects[0].tetrahedra={{{0,1,2,3}},{{4,6,5,7}}};
        auto dual=compileWorld(separated);
        require(dual.succeeded() && dual.world.fem.surfaceFaces.size()==8,
                "duplicated interface nodes lost either exposed side");
        auto invalid=source; invalid.objects[0].femCapacity.nodes=6; reject(invalid);
        invalid=source; invalid.objects[0].femCapacity.tetrahedra=3; reject(invalid);
        invalid=source; invalid.objects[0].adaptive=true; reject(invalid);
        invalid=source; invalid.objects[0].mutationPolicy.enabled=true; reject(invalid);
        invalid=source; invalid.objects[0].mutationPolicy.cohesiveFracture=true; reject(invalid);
        invalid=source; invalid.objects[0].automaticRepresentation=true; reject(invalid);
        invalid=source; invalid.objects[0].femCapacity.mutationCommands=1; reject(invalid);
        invalid=source; invalid.objects[0].representation=Representation::mpm; reject(invalid);
        auto selected=fixture();
        selected.compactImmutableFEMSurfaces=true;
        selected.objects[0].femDisabledContactSurfaceFaces={0u};
        auto selectedCook=compileWorld(selected);
        require(selectedCook.succeeded(),"authored exterior exclusion was rejected");
        auto selectedFace=std::find_if(
            selectedCook.world.fem.surfaceFaces.begin(),
            selectedCook.world.fem.surfaceFaces.end(),
            [](const NMFEMSurfaceFaceGPU& face) {
                return face.adjacency.x==0u && face.sides.x==0u;
            });
        require(selectedFace!=selectedCook.world.fem.surfaceFaces.end() &&
                (selectedFace->sides.w & NM_FEM_SURFACE_DISABLE_DEFORMABLE_CONTACT)!=0u,
                "source face exclusion was not cooked into the existing face flag");
        auto canonical=compileWorld(fixture());
        require(canonical.succeeded(),"canonical full-face fixture compilation failed");
        for (auto& face:canonical.world.fem.surfaceFaces)
            if (face.adjacency.x==0u && face.sides.x==0u)
                face.sides.w|=NM_FEM_SURFACE_DISABLE_DEFORMABLE_CONTACT;
        canonical.world.fingerprint=compiledWorldFingerprint(canonical.world);
        require(selectedCook.world.physicsFingerprint==canonical.world.fingerprint,
                "allocation-independent physics identity omitted authored contact eligibility");
        auto selectedB=fixture();
        selectedB.compactImmutableFEMSurfaces=true;
        selectedB.objects[0].femDisabledContactSurfaceFaces={1u};
        auto selectedBCook=compileWorld(selectedB);
        require(selectedBCook.succeeded() &&
                selectedBCook.world.physicsFingerprint!=selectedCook.world.physicsFingerprint,
                "different authored face selectors aliased in the physics identity");
        const std::size_t selectedIndex=static_cast<std::size_t>(
            selectedFace-selectedCook.world.fem.surfaceFaces.begin());
        auto selectedPackage=std::filesystem::temp_directory_path()/
            ("numi-static-surfaces-selected-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".nmpkg");
        require(writePackage(selectedCook,selectedPackage,&error),"selected-face package write failed");
        require(readPackage(selectedPackage,decoded,nullptr,&error),"selected-face package read failed");
        std::filesystem::remove(selectedPackage);
        require(decoded.fingerprint==selectedCook.world.fingerprint &&
                decoded.fem.surfaceFaces.size()==6u &&
                (decoded.fem.surfaceFaces[selectedIndex].sides.w &
                    NM_FEM_SURFACE_DISABLE_DEFORMABLE_CONTACT)!=0u,
                "selected-face flag did not survive package roundtrip");
        auto invalidFace=selected; invalidFace.objects[0].femDisabledContactSurfaceFaces={3u}; reject(invalidFace);
        invalidFace=selected; invalidFace.objects[0].femDisabledContactSurfaceFaces={8u}; reject(invalidFace);
        invalidFace=selected; invalidFace.objects[0].femDisabledContactSurfaceFaces={0u,0u}; reject(invalidFace);
        invalidFace=selected; invalidFace.objects[0].femDisabledContactSurfaceFaces={2u,0u}; reject(invalidFace);
        invalidFace=selected; invalidFace.compactImmutableFEMSurfaces=false; reject(invalidFace);

        auto mixed=fixture();
        mixed.compactImmutableFEMSurfaces=true;
        auto mpmMaterial=parseMatterFile(std::filesystem::path(__FILE__).parent_path().parent_path()/
                                         "materials/silicone.nmatter");
        require(mpmMaterial.succeeded(),"MPM material parse failed");
        mixed.materials.push_back(mpmMaterial.material);
        ObjectSource mpm;
        mpm.name="explicit-mpm-probe"; mpm.materialIndex=1u;
        mpm.representation=Representation::mpm; mpm.twoWayCoupling=false;
        mpm.characteristicLength=0.02;
        mpm.mpmGridMinimum={0.0,0.0,0.0};
        mpm.mpmGridMaximum={0.06,0.06,0.06};
        mpm.particles.push_back(ParticleSource{{0.03,0.03,0.03},{0.0,0.0,0.0},0.001,1.0e-6,true});
        mixed.objects.push_back(mpm);
        auto mixedCook=compileWorld(mixed);
        if (!mixedCook.succeeded()) for (auto& d:mixedCook.diagnostics) std::cerr<<d.message<<'\n';
        require(mixedCook.succeeded(),"explicit immutable FEM+MPM compact-map compile failed");
        require(mixedCook.world.fem.surfaceFaces.size()==6u &&
                mixedCook.world.dispatch.particleCount==1u &&
                mixedCook.world.dispatch.mpmGridCount==1u &&
                (mixedCook.world.dispatch.flags & NM_MATTER_STATIC_FEM_SURFACE_MAP)!=0u,
                "mixed compact map lost FEM faces or appended MPM state");
        require(validateCompiledWorldLayout(mixedCook.world,&error),
                "mixed compact map failed cooked-layout validation");

        std::cout<<"PASS immutable FEM surface map "<<checks<<" checks; physical_fem_steps=0\n";
    } catch (const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}

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
}
int main() {
    try {
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

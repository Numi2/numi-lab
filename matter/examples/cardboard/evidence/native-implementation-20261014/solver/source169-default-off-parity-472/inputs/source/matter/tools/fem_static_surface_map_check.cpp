#include "numi/matter/matter.hpp"
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
        std::cout<<"PASS immutable FEM surface map "<<checks<<" checks; physical_fem_steps=0\n";
    } catch (const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}

#include "numi/matter/matter.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>
using namespace numi::matter;
int main(int argc,char** argv) {
    if(argc!=3) return 2;
    const auto material=parseMatterFile(
        std::filesystem::path(__FILE__).parent_path().parent_path()/
        "materials"/"damageable_silicone.nmatter");
    if(!material.succeeded()) throw std::runtime_error("fixture material parse failed");
    WorldSource source;
    source.materials.push_back(material.material);
    source.gravity={0,0,0};
    ObjectSource object;
    object.name="legacy_package_byte_fixture";
    object.materialIndex=0;
    object.representation=Representation::fem;
    object.automaticRepresentation=false;
    object.mixedFEM=false;
    object.deformableContact=false;
    object.femNodes={{{0,0,0}},{{.01,0,0}},{{0,.01,0}},{{0,0,.01}},{{0,0,-.01}}};
    object.tetrahedra={{{0,1,2,3}},{{0,2,1,4}}};
    if(std::string(argv[1])=="explicit") {
        object.femTetrahedronMassesKg={2.0e-4,4.0e-4};
        object.femTetrahedronMassSourceIdentity={0x0123456789abcdefULL,
            0x1023456789abcdefULL,0x2123456789abcdefULL,0x3123456789abcdefULL};
    }
    source.objects.push_back(object);
    const auto compiled=compileWorld(source);
    if(!compiled.succeeded()) {
        for(const auto& d:compiled.diagnostics) std::cerr<<d.message<<"; ";
        return 3;
    }
    std::string error;
    if(!writePackage(compiled,argv[2],&error)) {
        std::cerr<<error<<'\n';
        return 4;
    }
    return 0;
}

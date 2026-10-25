
#include "numi/matter/matter.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <iterator>
#include <cstdint>

using namespace numi::matter;
namespace {
int checks=0;
void require(bool ok,const std::string& message){
    ++checks;
    if(!ok)throw std::runtime_error(message);
}
std::string diagnostics(const CompileResult& result){
    std::string out;
    for(const auto& diagnostic:result.diagnostics)out+=diagnostic.message+"; ";
    return out;
}
WorldSource fixture(){
    const auto parsed=parseMatterFile(std::filesystem::path(__FILE__).parent_path().parent_path()/
        "materials"/"damageable_silicone.nmatter");
    if(!parsed.succeeded())throw std::runtime_error("silicone fixture parse failed");
    WorldSource source;
    source.materials.push_back(parsed.material);
    source.gravity={0,0,0};
    ObjectSource object;
    object.name="synthetic_explicit_element_mass";
    object.materialIndex=0;
    object.representation=Representation::fem;
    object.automaticRepresentation=false;
    object.mixedFEM=false;
    object.deformableContact=false;
    object.femNodes={{{0,0,0}},{{.01,0,0}},{{0,.01,0}},{{0,0,.01}},{{0,0,-.01}}};
    object.tetrahedra={{{0,1,2,3}},{{0,2,1,4}}};
    source.objects.push_back(object);
    return source;
}
void denied(WorldSource source,const char* label){
    const auto result=compileWorld(source);
    require(!result.succeeded(),std::string("invalid source was accepted: ")+label);
    std::cout<<"denied="<<label<<" diagnostics="<<diagnostics(result)<<'\n';
}
void checkMassAssembly(){
    auto legacy=compileWorld(fixture());
    require(legacy.succeeded(),"legacy fixture failed: "+diagnostics(legacy));
    auto explicitSource=fixture();
    auto& object=explicitSource.objects.front();
    object.femTetrahedronMassesKg={2.0e-4,4.0e-4};
    object.femTetrahedronMassSourceIdentity={0x0123456789abcdefULL,
        0x1023456789abcdefULL,0x2123456789abcdefULL,0x3123456789abcdefULL};
    object.femFixedNodes={4};
    const auto explicitCook=compileWorld(explicitSource);
    require(explicitCook.succeeded(),"explicit-mass fixture failed: "+diagnostics(explicitCook));
    const auto& cooked=explicitCook.world;
    require(cooked.fem.nodes.size()==5u&&cooked.fem.tetrahedra.size()==2u,
        "explicit mass changed node/tetrahedron identities");
    const std::array<double,5> expected{1.5e-4,1.5e-4,1.5e-4,5.0e-5,1.0e-4};
    for(std::size_t node=0;node<expected.size();++node){
        const auto& state=cooked.fem.nodes[node];
        require(std::abs(double(state.positionAndMass.w)-expected[node])<=
                std::max(1.0e-11,expected[node]*2.0e-7),
            "wrong compiled per-node mass at row "+std::to_string(node));
        const float inverse=state.velocityAndInverseMass.w;
        if(node==4u)require(inverse==0.0f,"fixed node inverse mass was not zero");
        else require(std::abs(double(inverse)-1.0/expected[node])<=
                2.0e-6/expected[node],"compiled inverse mass does not match explicit element ledger");
    }
    require(cooked.fem.nodes[4].positionAndMass.w>0.0f,
        "fixed node lost its authored mass");
    require(cooked.materials.size()==legacy.world.materials.size()&&
            std::memcmp(cooked.materials.data(),legacy.world.materials.data(),
                        cooked.materials.size()*sizeof(NMMaterialGPU))==0,
        "explicit masses changed the authored material density or material record");
    require(cooked.fem.tetrahedra.size()==legacy.world.fem.tetrahedra.size()&&
            std::memcmp(cooked.fem.tetrahedra.data(),legacy.world.fem.tetrahedra.data(),
                        cooked.fem.tetrahedra.size()*sizeof(NMTetrahedronGPU))==0,
        "explicit masses changed element material/topology identity");
    require(cooked.fingerprint!=legacy.world.fingerprint&&
            cooked.physicsFingerprint!=legacy.world.physicsFingerprint,
        "compiled mass values are absent from executable world identity");
    auto changedSource=explicitSource;
    changedSource.objects.front().femTetrahedronMassesKg[1]=4.5e-4;
    const auto changed=compileWorld(changedSource);
    require(changed.succeeded(),"changed mass fixture failed: "+diagnostics(changed));
    require(changed.world.fingerprint!=cooked.fingerprint&&
            changed.world.physicsFingerprint!=cooked.physicsFingerprint,
        "changed per-element mass did not change cooked identity");
    require(changed.world.fem.nodes[0].positionAndMass.w!=
            cooked.fem.nodes[0].positionAndMass.w,
        "changed child mass did not reach compiled FEM nodal inertia");
    auto provenanceSource=explicitSource;
    provenanceSource.objects.front().femTetrahedronMassSourceIdentity[2]^=0x100ULL;
    const auto provenanceChanged=compileWorld(provenanceSource);
    require(provenanceChanged.succeeded(),
        "changed mass-source identity fixture failed: "+diagnostics(provenanceChanged));
    require(provenanceChanged.world.fem.nodes[0].positionAndMass.w==
                cooked.fem.nodes[0].positionAndMass.w &&
            provenanceChanged.world.fingerprint!=cooked.fingerprint &&
            provenanceChanged.world.physicsFingerprint!=cooked.physicsFingerprint,
        "mass-source identity changed physics values or was omitted from compiled identity");
    require(cooked.femTetrahedronMassSourceIdentities.size()==1u &&
            cooked.femTetrahedronMassSourceIdentities[0]==
                object.femTetrahedronMassSourceIdentity,
        "mass-source identity was not retained in compiled provenance");

    const auto packagePath=std::filesystem::temp_directory_path()/
        "numi-explicit-fem-tet-mass-check.nmpkg";
    const auto legacyPackagePath=std::filesystem::temp_directory_path()/
        "numi-legacy-fem-tet-mass-check.nmpkg";
    const auto tamperedPackagePath=std::filesystem::temp_directory_path()/
        "numi-tampered-fem-tet-mass-check.nmpkg";
    std::string packageError;
    require(writePackage(explicitCook,packagePath,&packageError),
        "explicit mass package write failed: "+packageError);
    CompiledWorld decoded;
    require(readPackage(packagePath,decoded,nullptr,&packageError),
        "explicit mass package read failed: "+packageError);
    require(decoded.femTetrahedronMassSourceIdentities==
                cooked.femTetrahedronMassSourceIdentities &&
            decoded.femTetrahedronMassesKg==cooked.femTetrahedronMassesKg &&
            decoded.fingerprint==cooked.fingerprint &&
            decoded.physicsFingerprint==cooked.physicsFingerprint,
        "package round trip lost mass-source identity or fingerprint");
    {
        std::ifstream package(packagePath,std::ios::binary);
        require(bool(package),"cannot reopen explicit package for version/tamper checks");
        std::vector<char> bytes((std::istreambuf_iterator<char>(package)),{});
        require(bytes.size()>32u,"explicit package is unexpectedly short");
        std::uint32_t version=0u,sectionCount=0u;
        std::memcpy(&version,bytes.data()+16u,sizeof(version));
        std::memcpy(&sectionCount,bytes.data()+28u,sizeof(sectionCount));
        require(version==20u && sectionCount>0u,
            "explicit source identity did not use the versioned package extension");
        bytes.back()^=char{1};
        std::ofstream tampered(tamperedPackagePath,std::ios::binary|std::ios::trunc);
        tampered.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));
        tampered.close();
        CompiledWorld rejected;
        require(!readPackage(tamperedPackagePath,rejected,nullptr,&packageError),
            "tampered package mass-source metadata was accepted");
        std::filesystem::remove(tamperedPackagePath);
    }
    std::filesystem::remove(packagePath);
    require(writePackage(legacy,legacyPackagePath,&packageError),
        "legacy package write failed: "+packageError);
    CompiledWorld legacyDecoded;
    require(readPackage(legacyPackagePath,legacyDecoded,nullptr,&packageError),
        "legacy package read failed: "+packageError);
    require(legacyDecoded.femTetrahedronMassSourceIdentities.empty() &&
            legacyDecoded.fingerprint==legacy.world.fingerprint &&
            legacyDecoded.physicsFingerprint==legacy.world.physicsFingerprint,
        "legacy package acquired mass metadata or changed its fingerprint");
    {
        std::ifstream package(legacyPackagePath,std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(package)),{});
        std::uint32_t version=0u,sectionCount=0u;
        std::memcpy(&version,bytes.data()+16u,sizeof(version));
        std::memcpy(&sectionCount,bytes.data()+28u,sizeof(sectionCount));
        require(version==19u && sectionCount>1u,
            "empty-metadata package did not retain the legacy header version/count");
    }
    std::filesystem::remove(legacyPackagePath);
    {
        CompiledWorld malformed=cooked;
        malformed.femTetrahedronMassSourceIdentities.push_back({});
        CompileResult badPackage=explicitCook;
        badPackage.world=std::move(malformed);
        require(!writePackage(badPackage,packagePath,&packageError),
            "wrong-sized mass-source metadata table was packaged");
    }
    {
        CompiledWorld malformed=cooked;
        malformed.femTetrahedronMassesKg.pop_back();
        CompileResult badPackage=explicitCook;
        badPackage.world=std::move(malformed);
        require(!writePackage(badPackage,packagePath,&packageError),
            "wrong-sized per-tetrahedron mass table was packaged");
    }
    {
        CompiledWorld malformed=cooked;
        malformed.objects.front().representation=
            static_cast<nm_u32>(Representation::rigid);
        std::string layoutError;
        require(!validateCompiledWorldLayout(malformed,&layoutError) &&
                layoutError.find("must identify an object with compiled FEM elements")!=std::string::npos,
            "non-FEM object with a nonzero mass-source identity was accepted: "+layoutError);
    }

    auto invalid=explicitSource;
    invalid.objects.front().femTetrahedronMassSourceIdentity={};
    denied(std::move(invalid),"mass identity missing");
    invalid=explicitSource;
    invalid.objects.front().femTetrahedronMassesKg.clear();
    denied(std::move(invalid),"identity without masses");
    invalid=explicitSource;
    invalid.objects.front().femTetrahedronMassesKg.pop_back();
    denied(std::move(invalid),"partial mass vector");
    invalid=explicitSource;
    invalid.objects.front().femTetrahedronMassesKg[0]=0.0;
    denied(std::move(invalid),"zero mass");
    invalid=explicitSource;
    invalid.objects.front().femTetrahedronMassesKg[0]=-1.0e-4;
    denied(std::move(invalid),"negative mass");
    invalid=explicitSource;
    invalid.objects.front().femTetrahedronMassesKg[0]=
        std::numeric_limits<double>::quiet_NaN();
    denied(std::move(invalid),"NaN mass");
    invalid=explicitSource;
    invalid.objects.front().femTetrahedronMassesKg[0]=
        std::numeric_limits<double>::infinity();
    denied(std::move(invalid),"infinite mass");
    invalid=explicitSource;
    invalid.objects.front().representation=Representation::rigid;
    denied(std::move(invalid),"non-FEM representation");
    invalid=explicitSource;
    invalid.objects.front().mixedFEM=true;
    denied(std::move(invalid),"mixed FEM");
    invalid=explicitSource;
    invalid.objects.front().adaptive=true;
    denied(std::move(invalid),"adaptive explicit-mass object");
}
}
int main(){
    try{
        checkMassAssembly();
        std::cout<<"PASS explicit_fem_tetrahedron_mass_compiler checks="<<checks
                 <<" human_admission=not_tested\n";
        return 0;
    }catch(const std::exception& error){
        std::cerr<<"FAIL explicit_fem_tetrahedron_mass_compiler checks="<<checks
                 <<" error="<<error.what()<<'\n';
        return 1;
    }
}

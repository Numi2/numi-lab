#include "numi/matter/vascular.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace numi::matter;
namespace {
void require(bool condition, const std::string& reason) { if (!condition) throw std::runtime_error(reason); }
std::string messages(const CompileResult& c) { std::string r;for(const auto& d:c.diagnostics)r+=d.message+"; ";return r; }
WorldSource fixture() {
    // Dimensioned synthetic passive network for compiler tests; these are not
    // organ calibration values and this executable does not advance physics.
    WorldSource s;s.environmentCount=2;s.frameTimestep=0.01;s.gravity={0,0,0};
    auto& n=s.vascular;n.contentIdentity={0x31415926u,0x27182818u,0x12345678u,0xabcdef01u};
    n.sourceIdentity={1u,2u,3u,4u};n.authoredIdentity={5u,6u,7u,8u};
    n.species={{9u,"synthetic_tracer_beta",1e-3,1e-5},{3u,"synthetic_tracer_alpha",2e-3,2e-5}};
    n.compartments={{20u,"fixture/vascular_b",0.002,0.0021,0,1e-7,0,0.002,1e-5,{0.001,0.002}},
                    {10u,"fixture/vascular_a",0.001,0.0012,10,2e-7,20,0.001,1e-5,{0.003,0.004}}};
    n.connections={{7u,20u,10u,1e8,1e5,-1e-5,1e-5,1000,1e-5}};
    VascularTissueSource tissue;tissue.stableIdentifier=4;tissue.anatomicalIdentifier="fixture/organ_reservoir";tissue.volume=.001;tissue.initialSpeciesAmounts={.0001,.0002};n.tissues={tissue};
    n.exchanges={{8u,20u,4u,9u,1e-6,2.0},{2u,10u,4u,3u,2e-6,1.0}};
    return s;
}
void rejected(const WorldSource& source,const char* reason) {auto c=compileWorld(source);require(!c.succeeded(),std::string("invalid source accepted: ")+reason);}
void rejectedCooked(CompiledWorld world,const char* reason) {
    // Recompute the integrity hash so this check exercises semantic validation,
    // not merely the outer corruption checksum.
    world.fingerprint=compiledWorldFingerprint(world);std::string error;
    require(!validateCompiledWorldLayout(world,&error),std::string("invalid cooked world accepted: ")+reason);
}
}
int main() {
    try {
        const auto s=fixture();const auto compiled=compileWorld(s);
        require(compiled.succeeded(),"vascular-only world failed: "+messages(compiled));
        const auto& w=compiled.world;const auto& v=w.vascular;
        require(w.materials.empty()&&w.objects.empty(),"vascular-only source acquired fabricated mechanical objects");
        require(v.layout.ranges.z==9u&&v.layout.offsets.y==2u&&v.layout.offsets.z==3u&&v.layout.offsets.w==7u,"unknown layout mismatch");
        require(v.species[0].identity.x==3u&&v.compartments[0].identity.x==10u&&v.connections[0].identity.y==1u&&v.connections[0].identity.z==0u,"stable identifiers did not resolve canonically");
        require(v.unknowns[3].initialAndScaling.x==float(.004)&&v.unknowns[4].initialAndScaling.x==float(.003)&&v.unknowns[7].initialAndScaling.x==float(.0002),"species permutation changed physical initial amounts");
        require(v.unknowns[2].initialAndScaling.x==float(-1e-5)&&v.unknowns[2].initialAndScaling.z==1000,"signed flow or pressure residual scale changed");
        require(v.connectionIncidence==std::vector<std::uint32_t>({0u,0u})&&v.bloodExchangeIncidence==std::vector<std::uint32_t>({0u,1u})&&v.tissueExchangeIncidence==std::vector<std::uint32_t>({0u,1u}),"canonical incidence mismatch");
        auto permutation=s;std::reverse(permutation.vascular.species.begin(),permutation.vascular.species.end());
        for(auto& x:permutation.vascular.compartments)std::reverse(x.initialSpeciesAmounts.begin(),x.initialSpeciesAmounts.end());
        for(auto& x:permutation.vascular.tissues)std::reverse(x.initialSpeciesAmounts.begin(),x.initialSpeciesAmounts.end());
        std::reverse(permutation.vascular.compartments.begin(),permutation.vascular.compartments.end());std::reverse(permutation.vascular.exchanges.begin(),permutation.vascular.exchanges.end());
        const auto canonical=compileWorld(permutation);require(canonical.succeeded()&&canonical.world.fingerprint==w.fingerprint,"source ordering changed canonical fingerprint");
        const auto checkIdentity=[&](WorldSource changed,const char* role){const auto c=compileWorld(changed);require(c.succeeded()&&c.world.fingerprint!=w.fingerprint&&c.world.physicsFingerprint!=w.physicsFingerprint,std::string("unbound identity: ")+role);};
        auto changed=s;changed.vascular.contentIdentity[0]^=1u;checkIdentity(changed,"content SHA");
        changed=s;changed.vascular.sourceIdentity[0]^=1u;checkIdentity(changed,"source SHA");
        changed=s;changed.vascular.authoredIdentity[0]^=1u;checkIdentity(changed,"authored SHA");
        changed=s;changed.vascular.tissues[0].anatomicalIdentifier+="/different";checkIdentity(changed,"anatomical string");
        changed=s;changed.vascular.species[0].name+="_different";checkIdentity(changed,"species name");
        changed=s;changed.vascular.compartments[0].initialVolume*=1.01;checkIdentity(changed,"initial volume");
        changed=s;changed.vascular.connections[0].pressureScale*=2;checkIdentity(changed,"residual scale");
        changed=s;changed.vascular.connections[0].flowResidualTolerance*=2;checkIdentity(changed,"residual tolerance");
        changed=s;changed.vascular.compartments[1].stableIdentifier=20;rejected(changed,"duplicate compartment ID");
        changed=s;changed.vascular.connections[0].fromCompartment=999;rejected(changed,"missing endpoint");
        changed=s;changed.vascular.connections[0].toCompartment=20;rejected(changed,"self connection");
        changed=s;changed.vascular.species[0].name=changed.vascular.species[1].name;rejected(changed,"duplicate species name");
        changed=s;changed.vascular.species[0].name=std::string("bad\0name",8);rejected(changed,"embedded NUL name");
        changed=s;changed.vascular.species[0].name="\xc0\xaf";rejected(changed,"invalid UTF-8");
        changed=s;changed.vascular.compartments[0].initialSpeciesAmounts[0]=-1;rejected(changed,"negative amount");
        changed=s;changed.vascular.compartments[0].initialSpeciesAmounts.pop_back();rejected(changed,"missing species amount");
        changed=s;changed.vascular.compartments[0].compliance=0;rejected(changed,"zero compliance");
        changed=s;changed.vascular.connections[0].resistance=-1;rejected(changed,"active negative resistance");
        changed=s;changed.vascular.connections[0].inertance=-1;rejected(changed,"negative inertance");
        changed=s;changed.vascular.connections[0].flowScale=0;rejected(changed,"zero variable scale");
        changed=s;changed.vascular.connections[0].pressureScale=0;rejected(changed,"zero residual scale");
        changed=s;changed.vascular.species[0].amountResidualTolerance=0;rejected(changed,"zero tolerance");
        changed=s;changed.vascular.species[0].amountScale=std::numeric_limits<double>::denorm_min();rejected(changed,"underflowed scale");
        changed=s;changed.vascular.compartments[0].initialVolume=std::numeric_limits<double>::infinity();rejected(changed,"infinite volume");
        changed=s;changed.vascular.tissues[0].mechanicsFeedback=true;rejected(changed,"unsupported mechanics feedback");
        changed=s;changed.vascular.tissues[0].objectIndex=0;changed.vascular.tissues[0].femRegion={{0,1}};rejected(changed,"fabricated FEM binding");
        changed=s;changed.vascular.exchanges[0].partitionCoefficient=0;rejected(changed,"zero partition coefficient");
        changed=s;changed.vascular.exchanges[0].species=999;rejected(changed,"missing exchange species");
        changed=s;changed.vascular.tissues[0].volume=1e30;changed.vascular.exchanges[0].partitionCoefficient=1e30;rejected(changed,"overflowed partition volume");
        changed=s;changed.vascular.tissues[0].volume=1e-30;changed.vascular.exchanges[0].partitionCoefficient=1e-30;rejected(changed,"underflowed partition volume");
        changed=s;changed.vascular.compartments[0].initialSpeciesAmounts[0]=1e37;rejected(changed,"overflowed initial concentration");
        changed=s;changed.vascular.compartments[0].compliance=1e-40;rejected(changed,"unrepresentable inverse compliance");
        changed=s;changed.vascular.contentIdentity={};rejected(changed,"missing source content identity");
        changed=s;changed.environmentCount=std::numeric_limits<std::uint32_t>::max();rejected(changed,"environment-times-Krylov overflow");
        auto bad=w;bad.vascular.connectionIncidence[1]=1;rejectedCooked(bad,"invalid incidence");
        bad=w;bad.vascular.connectionRanges[0].reserved0=1;rejectedCooked(bad,"unknown range flag");
        bad=w;bad.vascular.unknowns[0].initialAndScaling.y=0;rejectedCooked(bad,"cooked zero scale");
        bad=w;bad.vascular.unknowns[3].initialAndScaling.w*=2;rejectedCooked(bad,"species scale mismatch");
        bad=w;bad.vascular.names.push_back('x');rejectedCooked(bad,"unowned name bytes");
        bad=w;bad.vascular.layout.offsets.z++;rejectedCooked(bad,"unknown overlap");
        bad=w;bad.vascular.exchanges[0].identity.y=999;rejectedCooked(bad,"out-of-range exchange");
        bad=w;bad.vascular.compartments[0].identity.z=1;rejectedCooked(bad,"unsupported compartment law flag");
        // A tissue association must resolve actual authored FEM topology.
        auto associated=s;
        const auto material=parseMatterFile(std::filesystem::path(__FILE__).parent_path().parent_path()/"materials"/"damageable_silicone.nmatter");
        require(material.succeeded(),"FEM association material did not parse");
        associated.materials.push_back(material.material);
        ObjectSource organ;organ.name="synthetic_region_fem";organ.materialIndex=0;organ.representation=Representation::fem;organ.deformableContact=false;
        organ.femNodes={{{0,0,0}},{{.01,0,0}},{{0,.01,0}},{{0,0,.01}}};
        organ.tetrahedra.push_back({{0u,1u,2u,3u}});associated.objects.push_back(organ);
        associated.vascular.tissues[0].objectIndex=0;associated.vascular.tissues[0].femRegion={{2u,.25},{0u,.75}};
        const auto bound=compileWorld(associated);require(bound.succeeded(),"real FEM association rejected: "+messages(bound));
        require(bound.world.vascular.tissueBindings.size()==2&&bound.world.vascular.tissueBindings[0].identity.y==0&&bound.world.vascular.tissueBindings[1].identity.y==2,"FEM region not canonically resolved");
        changed=associated;changed.vascular.tissues[0].femRegion={{0u,.5},{0u,.5}};rejected(changed,"duplicate FEM region node");
        changed=associated;changed.vascular.tissues[0].femRegion={{0u,.9}};rejected(changed,"unnormalized FEM region");
        changed=associated;changed.objects[0].mutationPolicy.enabled=true;rejected(changed,"unsupported vascular topology remap");
        bad=bound.world;bad.vascular.tissueBindings[0].identity.y=999;rejectedCooked(bad,"stale FEM association");
        const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
        const auto path=std::filesystem::temp_directory_path()/("numi-vascular-compiler-"+std::to_string(stamp)+".nmatterpack");std::string error;
        require(writePackage(compiled,path,&error),"package write failed: "+error);
        CompiledWorld decoded;require(readPackage(path,decoded,nullptr,&error),"package read failed: "+error);
        require(decoded.fingerprint==w.fingerprint&&decoded.physicsFingerprint==w.physicsFingerprint&&decoded.vascular.names==v.names&&decoded.vascular.identity.authored[3]==8u,"package changed graph state or provenance");
        std::fstream corrupt(path,std::ios::in|std::ios::out|std::ios::binary);corrupt.seekp(-1,std::ios::end);corrupt.put('x');corrupt.close();
        require(!readPackage(path,decoded,nullptr,&error),"corrupted package accepted");std::filesystem::remove(path);
        std::cout<<"vascular_compiler=pass boundary=source_cooking_package_validation_only laws=passive_compliance_resistance_inertance_transport_exchange canonical_order=true semantic_rejection=true provenance_bound=true\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"vascular_compiler=fail reason="<<e.what()<<'\n';return 1;}
}

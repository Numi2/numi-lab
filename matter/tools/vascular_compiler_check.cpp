#include "numi/matter/vascular.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
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
        // Numerical constitutive fixtures only; actual source replication uses
        // the separately pinned Human source payload and generated CellML oracle.
        auto cardiac=s;cardiac.vascular.species.clear();cardiac.vascular.tissues.clear();cardiac.vascular.exchanges.clear();
        for(auto& node:cardiac.vascular.compartments)node.initialSpeciesAmounts.clear();
        auto& chamber=cardiac.vascular.compartments[0];
        chamber.pressureLaw=VascularPressureLaw::ventricularElastance;chamber.compliance=0;
        chamber.elastanceMin=1e7;chamber.elastanceMax=3e8;chamber.periodSeconds=1;
        chamber.activationStart=.3;chamber.activationEnd=.45;chamber.sourcePi=3.14159;
        auto& storage=cardiac.vascular.compartments[1];storage.storageKind=VascularStorageKind::storageDisplacement;
        storage.referenceVolume=0;storage.initialVolume=0;
        auto& valve=cardiac.vascular.connections[0];valve.flowLaw=VascularFlowLaw::oneWayOrifice;
        valve.resistance=0;valve.inertance=0;valve.orificeCoefficient=3e-5;valve.initialFlow=0;
        const auto heart=compileWorld(cardiac);require(heart.succeeded(),"cardiac hydraulic fixture rejected: "+messages(heart));
        const auto& hv=heart.world.vascular;
        require(hv.compartments[0].identity.z==1 && hv.unknowns[0].initialAndScaling.x==0 && hv.compartments[1].periodTicks>0,"storage or periodic chamber lost during cooking");
        changed=cardiac;changed.vascular.compartments[1].initialVolume=-1e-4;
        require(compileWorld(changed).succeeded(),"signed hydraulic storage rejected");
        changed=cardiac;changed.vascular.compartments[0].pressureLaw=VascularPressureLaw::atrialElastance;
        changed.vascular.compartments[0].activationStart=.92;changed.vascular.compartments[0].activationEnd=.09;
        const auto atrium=compileWorld(changed);require(atrium.succeeded(),"source atrial wrapped waveform rejected");
        changed.vascular.compartments[0].activationEnd=.01;rejected(changed,"unsupported nonwrapping source atrial law");
        changed=cardiac;changed.vascular.compartments[0].activationEnd=.2;rejected(changed,"unordered ventricular phase");
        changed=cardiac;changed.vascular.compartments[0].elastanceMin=-1;rejected(changed,"negative elastance");
        changed=cardiac;changed.vascular.compartments[0].elastanceMax=1;rejected(changed,"peak below minimum");
        changed=cardiac;changed.frameTimestep=1e-300;rejected(changed,"base timestep underflows Float32");
        changed=cardiac;changed.frameTimestep=1e300;rejected(changed,"base timestep overflows Float32");
        changed=cardiac;changed.vascular.compartments[0].periodSeconds=1e20;rejected(changed,"period exceeds exact tick bound");
        changed=cardiac;changed.vascular.compartments[0].periodSeconds=1+std::ldexp(1.0,-50);rejected(changed,"period silently quantized");
        changed=cardiac;changed.vascular.compartments[0].sourcePi=0;rejected(changed,"missing source angular constant");
        changed=cardiac;changed.vascular.compartments[1].elastanceMin=1;rejected(changed,"unused waveform payload");
        changed=cardiac;changed.vascular.connections[0].initialFlow=-1e-6;rejected(changed,"reverse source valve initial flow");
        changed=cardiac;changed.vascular.connections[0].resistance=1;rejected(changed,"orifice mixed with unsupported resistance");
        changed=cardiac;changed.vascular.connections[0].orificeCoefficient=0;rejected(changed,"zero orifice coefficient");
        changed=s;changed.vascular.compartments[0].storageKind=VascularStorageKind::storageDisplacement;rejected(changed,"blood concentration derived from hydraulic displacement");
        changed=cardiac;changed.vascular.compartments[0].storageKind=VascularStorageKind::storageDisplacement;rejected(changed,"cardiac source lacks absolute chamber volume");
        bad=heart.world;bad.vascular.layout.clock.x++;rejectedCooked(bad,"altered clock quantum");
        bad=heart.world;bad.vascular.compartments[1].periodTicks=0;rejectedCooked(bad,"missing cooked cardiac period");
        bad=heart.world;bad.vascular.compartments[1].periodMultiplier=0;rejectedCooked(bad,"unused cardiac bytes");
        bad=heart.world;bad.vascular.connections[0].identity.w=4;rejectedCooked(bad,"unknown flow law");
        const auto cardiacIdentity=[&](WorldSource change,const char* role){auto c=compileWorld(change);require(c.succeeded()&&c.world.fingerprint!=heart.world.fingerprint&&c.world.physicsFingerprint!=heart.world.physicsFingerprint,std::string("unbound cardiac field: ")+role);};
        changed=cardiac;changed.vascular.compartments[0].periodSeconds=2;cardiacIdentity(changed,"period");
        changed=cardiac;changed.vascular.compartments[0].sourcePi=3.141592653589793;cardiacIdentity(changed,"source pi");
        changed=cardiac;changed.vascular.compartments[0].elastanceMax*=1.1;cardiacIdentity(changed,"Emax");
        changed=cardiac;changed.vascular.connections[0].orificeCoefficient*=1.1;cardiacIdentity(changed,"CV");
        auto regional=s;
        auto& vein=regional.vascular.compartments[0];vein.pressureLaw=VascularPressureLaw::atanCompliance;
        vein.maximumVolumeDisplacement=.001;vein.sourcePi=std::acos(-1.0);
        auto& linearValve=regional.vascular.connections[0];linearValve.flowLaw=VascularFlowLaw::oneWayResistance;
        linearValve.initialFlow=0;linearValve.inertance=0;
        const auto bounded=compileWorld(regional);require(bounded.succeeded(),"atan blood volume / linear valve rejected: "+messages(bounded));
        for(double volume:{.001,.003,.0009,.0031}){changed=regional;changed.vascular.compartments[0].initialVolume=volume;rejected(changed,"atan pressure endpoint or exterior accepted");}
        changed=regional;changed.vascular.compartments[0].initialVolume=.0015;require(compileWorld(changed).succeeded(),"signed atan displacement within positive absolute volume rejected");
        changed=regional;changed.vascular.compartments[0].storageKind=VascularStorageKind::storageDisplacement;rejected(changed,"atan storage must remain absolute volume");
        changed=regional;changed.vascular.compartments[0].referenceVolume=2;changed.vascular.compartments[0].maximumVolumeDisplacement=1;
        changed.vascular.compartments[0].initialVolume=2.999999761581421;changed.vascular.compartments[0].volumeScale=43561.0234375;
        rejected(changed,"normalization reconstructs initial volume onto atan endpoint");
        changed=regional;changed.vascular.compartments[0].sourcePi=3.14159;rejected(changed,"atan angular domain requires mathematical pi");
        changed=regional;changed.vascular.compartments[0].maximumVolumeDisplacement=0;rejected(changed,"zero atan displacement bound");
        changed=regional;changed.vascular.connections[0].downstreamPressureFloor=100;rejected(changed,"non-Starling unused floor");
        changed=regional;changed.vascular.connections[0].flowLaw=VascularFlowLaw::starlingResistance;changed.vascular.connections[0].downstreamPressureFloor=100;
        const auto starling=compileWorld(changed);require(starling.succeeded(),"Starling floor rejected: "+messages(starling));
        require(starling.world.physicsFingerprint!=bounded.world.physicsFingerprint,"flow-law floor absent from fingerprint");
        changed.vascular.connections[0].initialFlow=-1e-6;rejected(changed,"reverse Starling initial flow");
        auto rational=cardiac;auto& pulse=rational.vascular.compartments[0];
        pulse.pressureLaw=VascularPressureLaw::cosinePulseElastance;pulse.periodSeconds=0;
        pulse.periodNumeratorSeconds=6;pulse.periodDenominator=7;pulse.phaseDelay=.1;pulse.sourcePi=std::acos(-1.0);
        const auto timed=compileWorld(rational);require(timed.succeeded(),"exact 6/7 second delayed pulse rejected: "+messages(timed));
        const auto& timing=timed.world.vascular.compartments[1];
        require(timing.periodMultiplier==7 && timing.periodTicks==std::uint64_t(std::ldexp(6.0,-std::ilogb(float(rational.frameTimestep))+23+16)),"rational period changed in lowering");
        changed=rational;changed.vascular.compartments[0].periodDenominator=14;rejected(changed,"noncanonical rational period");
        changed=rational;changed.vascular.compartments[0].periodSeconds=1;rejected(changed,"ambiguous period authorities");
        changed=rational;changed.vascular.compartments[0].periodNumeratorSeconds=0;rejected(changed,"missing rational numerator");
        changed=rational;changed.vascular.compartments[0].periodNumeratorSeconds=std::numeric_limits<std::uint64_t>::max();changed.vascular.compartments[0].periodDenominator=1;rejected(changed,"rational numerator overflow");
        changed=rational;changed.vascular.compartments[0].periodNumeratorSeconds=1;changed.vascular.compartments[0].periodDenominator=std::numeric_limits<std::uint64_t>::max();require(compileWorld(changed).succeeded(),"full-width denominator unnecessarily rounded or rejected");
        changed=rational;changed.vascular.compartments[0].phaseDelay=.6;rejected(changed,"delayed pulse runs across unimplemented wrap");
        changed=rational;changed.vascular.compartments[0].pressureLaw=VascularPressureLaw::ventricularElastance;rejected(changed,"legacy pulse accepted unused delay");
        bad=timed.world;bad.vascular.compartments[1].periodMultiplier=14;rejectedCooked(bad,"noncanonical cooked rational time");
        bad=bounded.world;bad.vascular.compartments[1].pressureParameters.y=1;rejectedCooked(bad,"unowned atan parameter");
        const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
        const auto path=std::filesystem::temp_directory_path()/("numi-vascular-compiler-"+std::to_string(stamp)+".nmatterpack");std::string error;
        require(writePackage(compiled,path,&error),"package write failed: "+error);
        CompiledWorld decoded;require(readPackage(path,decoded,nullptr,&error),"package read failed: "+error);
        require(decoded.fingerprint==w.fingerprint&&decoded.physicsFingerprint==w.physicsFingerprint&&decoded.vascular.names==v.names&&decoded.vascular.identity.authored[3]==8u,"package changed graph state or provenance");
        std::fstream corrupt(path,std::ios::in|std::ios::out|std::ios::binary);corrupt.seekp(-1,std::ios::end);corrupt.put('x');corrupt.close();
        require(!readPackage(path,decoded,nullptr,&error),"corrupted package accepted");std::filesystem::remove(path);
        require(writePackage(heart,path,&error),"cardiac package write failed: "+error);
        require(readPackage(path,decoded,nullptr,&error),"cardiac package read failed: "+error);
        require(decoded.fingerprint==heart.world.fingerprint && decoded.vascular.compartments[1].periodTicks==hv.compartments[1].periodTicks,"cardiac package changed source time or fingerprint");
        std::filesystem::remove(path);
        require(writePackage(timed,path,&error),"rational package write failed: "+error);
        require(readPackage(path,decoded,nullptr,&error),"rational package read failed: "+error);
        require(decoded.fingerprint==timed.world.fingerprint && decoded.vascular.compartments[1].periodMultiplier==7 && decoded.vascular.compartments[1].waveform.y==float(.1),"rational package changed phase or delay");
        std::filesystem::remove(path);
        require(writePackage(bounded,path,&error),"atan package write failed: "+error);
        require(readPackage(path,decoded,nullptr,&error),"atan package read failed: "+error);
        require(decoded.fingerprint==bounded.world.fingerprint && decoded.vascular.compartments[1].pressureParameters.x==float(.001),"atan package lost displacement bound");
        std::filesystem::remove(path);
        std::cout<<"vascular_compiler=pass boundary=source_cooking_package_validation_only laws=compliance_atan_rational_elastance_orifice_diode_starling_transport_exchange canonical_order=true semantic_rejection=true provenance_bound=true\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"vascular_compiler=fail reason="<<e.what()<<'\n';return 1;}
}

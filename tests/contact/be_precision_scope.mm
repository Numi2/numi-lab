// Scope, fingerprint and persistent-state checks for disabled BE precision mode.
#define main numi_plane_contact_diagnostic_main
#include "../../apps/plane_contact_probe.mm"
#undef main
#include <cstdlib>
int main(int argc,char**argv){@autoreleasepool{try{
 require(argc==2,"usage: be_precision_scope METALLIB");
 const auto world=model(2,.000001,.0002,true,false,true,0.,1.);
 numi::matter::RuntimeConfiguration config;config.metallib=argv[1];config.environmentCount=1;
 config.captureDiagnostics=true;config.adaptiveTransfer=false;config.experimentalFEMPlaneAreaIPC=true;
 constexpr const char* name="NM_MATTER_BE_VELOCITY_COMPENSATION";
 unsetenv(name);numi::matter::Runtime baseline;auto plain=baseline.initialize(world,config);
 require(plain.encoded && baseline.valid(),"baseline rejected: "+plain.message);
 auto plainSnapshot=baseline.snapshot();require(plainSnapshot.available,"baseline snapshot unavailable");
 setenv(name,"1",1);numi::matter::Runtime candidate;auto result=candidate.initialize(world,config);
 require(result.encoded && candidate.valid(),"BE precision mode rejected: "+result.message);
 require(candidate.deviceProgramFingerprint()!=baseline.deviceProgramFingerprint(),"mode fingerprint did not change");
 auto snap=candidate.snapshot();require(snap.available && snap.femMidpointVelocityLow.size()==world.dispatch.femNodeCount,"low velocity state missing");
 require(candidate.restore(snap).encoded,"valid state rejected");
 require(!candidate.restore(plainSnapshot).encoded,"baseline state accepted under another precision fingerprint");
 auto bad=snap;bad.femMidpointVelocityLow.pop_back();require(!candidate.restore(bad).encoded,"short state accepted");
 bad=snap;bad.femMidpointVelocityLow[0].x=std::numeric_limits<float>::quiet_NaN();
 require(!candidate.restore(bad).encoded,"nonfinite low velocity accepted");
 setenv(name,"0",1);numi::matter::Runtime invalid;require(!invalid.initialize(world,config).encoded,"invalid opt-in accepted");
 setenv(name,"1",1);auto midpoint=config;midpoint.experimentalFEMPlaneMidpoint=true;
 numi::matter::Runtime mixed;require(!mixed.initialize(world,midpoint).encoded,"midpoint accepted BE precision option");
 auto noArea=config;noArea.experimentalFEMPlaneAreaIPC=false;numi::matter::Runtime outside;
 require(!outside.initialize(world,noArea).encoded,"non-area law accepted BE precision option");
 auto otherMaterial=world;otherMaterial.materials[0].constitutiveKind=NM_CONSTITUTIVE_COROTATED;
 numi::matter::Runtime unsupportedMaterial;auto rejectedMaterial=unsupportedMaterial.initialize(otherMaterial,config);
 require(!rejectedMaterial.encoded && rejectedMaterial.message==
   "backward Euler precision research requires the scoped physical fixed-plane fixture",
   "unqualified material accepted or rejected outside the BE precision scope");
 auto noCapture=config;noCapture.captureDiagnostics=false;numi::matter::Runtime uncaptured;
 require(!uncaptured.initialize(world,noCapture).encoded,"unqualified capture mode accepted");
 unsetenv(name);std::cout<<"BE precision scope, fingerprints and state validation verified\n";return 0;
 }catch(const std::exception&e){unsetenv("NM_MATTER_BE_VELOCITY_COMPENSATION");std::cerr<<e.what()<<"\n";return 1;}}}

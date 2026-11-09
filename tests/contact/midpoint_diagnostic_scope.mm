// Scope and fingerprint checks for the explicitly selected midpoint capture.
#define main numi_plane_contact_diagnostic_main
#include "../../apps/plane_contact_probe.mm"
#undef main
#include <cstdlib>

int main(int argc, char** argv) {
 @autoreleasepool { try {
    require(argc == 2, "usage: midpoint_diagnostic_scope METALLIB");
    const auto world = model(2, 0.000015625, 0.0002, true, false, true, 0., 1.);
    numi::matter::RuntimeConfiguration config;
    config.metallib=argv[1]; config.environmentCount=1; config.captureDiagnostics=true;
    config.adaptiveTransfer=false; config.experimentalFEMPlaneAreaIPC=true;
    config.experimentalFEMPlaneMidpoint=true;
    constexpr const char* name="NM_MATTER_MIDPOINT_DIAGNOSTIC_ROOT";
    unsetenv(name);
    unsetenv("NM_MATTER_MIDPOINT_PRECONTACT_POLISH");
    numi::matter::Runtime normal;
    auto a=normal.initialize(world,config);
    require(a.encoded && normal.valid(),"normal fixture initialization failed: "+a.message);
    require(!normal.femMidpointCandidateDiagnosticBuffer() &&
        !normal.femMidpointEndpointDiagnosticBuffer() &&
        !normal.femMidpointResidualDiagnosticBuffer(),"default run exposes capture buffers");
    const auto base=normal.deviceProgramFingerprint();
    setenv(name,"0",1);
    numi::matter::Runtime captured0;
    auto b=captured0.initialize(world,config);
    require(b.encoded && captured0.valid(),"capture root zero rejected: "+b.message);
    require(captured0.femMidpointCandidateDiagnosticBuffer() &&
        captured0.femMidpointEndpointDiagnosticBuffer() &&
        captured0.femMidpointResidualDiagnosticBuffer(),"capture buffers unavailable");
    setenv(name,"203",1);
    numi::matter::Runtime captured203;
    auto c=captured203.initialize(world,config);
    require(c.encoded && captured203.valid(),"capture root 203 rejected: "+c.message);
    require(base!=captured0.deviceProgramFingerprint() &&
        base!=captured203.deviceProgramFingerprint() &&
        captured0.deviceProgramFingerprint()!=captured203.deviceProgramFingerprint(),
        "diagnostic roots not distinguished by fingerprints");
    for(const char* root: {"", "-1", "1x", "4294967296"}) {
        setenv(name,root,1);
        numi::matter::Runtime denied;
        auto result=denied.initialize(world,config);
        require(!result.encoded && !denied.valid() &&
            result.message.find("midpoint diagnostic root")!=std::string::npos,
            "invalid diagnostic root accepted");
    }
    setenv(name,"1",1);
    auto noMidpoint=config; noMidpoint.experimentalFEMPlaneMidpoint=false;
    numi::matter::Runtime denied;
    auto d=denied.initialize(world,noMidpoint);
    require(!d.encoded && !denied.valid() &&
        d.message.find("midpoint diagnostic root")!=std::string::npos,
        "capture accepted outside midpoint fixture");
    unsetenv(name);
    constexpr const char* polishName="NM_MATTER_MIDPOINT_PRECONTACT_POLISH";
    setenv(polishName,"1",1);
    numi::matter::Runtime polish;
    auto e=polish.initialize(world,config);
    require(e.encoded && polish.valid() && polish.deviceProgramFingerprint()!=base,
        "opt-in polish unavailable or not fingerprinted");
    setenv(polishName,"0",1);
    numi::matter::Runtime invalidPolish;
    auto f=invalidPolish.initialize(world,config);
    require(!f.encoded && !invalidPolish.valid() &&
        f.message.find("midpoint precontact polish")!=std::string::npos,
        "invalid polish opt-in accepted");
    setenv(polishName,"1",1);
    numi::matter::Runtime outsidePolish;
    auto g=outsidePolish.initialize(world,noMidpoint);
    require(!g.encoded && !outsidePolish.valid() &&
        g.message.find("midpoint precontact polish")!=std::string::npos,
        "polish accepted outside the midpoint fixture");
    unsetenv(polishName);
    constexpr const char* lowName="NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION";
    constexpr const char* stageName="NM_MATTER_MIDPOINT_STAGE_COMPENSATION";
    unsetenv(lowName);setenv(stageName,"1",1);
    numi::matter::Runtime stageWithoutLow;
    auto h=stageWithoutLow.initialize(world,config);
    require(!h.encoded && !stageWithoutLow.valid(),"stage low accepted without endpoint low");
    setenv(lowName,"1",1);unsetenv(stageName);
    numi::matter::Runtime low;
    auto i=low.initialize(world,config);
    require(i.encoded && low.valid() && low.deviceProgramFingerprint()!=base,"low mode unavailable or not fingerprinted");
    setenv(stageName,"1",1);
    numi::matter::Runtime stage;
    auto j=stage.initialize(world,config);
    require(j.encoded && stage.valid() && stage.deviceProgramFingerprint()!=low.deviceProgramFingerprint(),
        "stage mode unavailable or not fingerprinted: "+j.message);
    auto snapshot=stage.snapshot();
    require(snapshot.available && snapshot.femMidpointVelocityLow.size()==world.dispatch.femNodeCount,
        "compensated physical state missing velocity low word");
    auto malformed=snapshot;malformed.femMidpointVelocityLow.pop_back();
    require(!stage.restore(malformed).encoded,"malformed low arena accepted");
    malformed=snapshot;malformed.femMidpointVelocityLow[0].x=std::numeric_limits<float>::quiet_NaN();
    require(!stage.restore(malformed).encoded,"nonfinite low velocity accepted");
    require(stage.restore(snapshot).encoded,"valid compensated snapshot rejected");
    setenv(name,"1",1);
    numi::matter::Runtime capturedStage;
    require(!capturedStage.initialize(world,config).encoded,"unsupported stage capture accepted");
    unsetenv(name);setenv(stageName,"0",1);
    numi::matter::Runtime invalidStage;
    require(!invalidStage.initialize(world,config).encoded,"invalid stage opt-in accepted");
    unsetenv(stageName);unsetenv(lowName);
    std::cout<<"midpoint scope, state validation and fingerprints verified\n";
    return 0;
 } catch(const std::exception& e) {
    unsetenv("NM_MATTER_MIDPOINT_DIAGNOSTIC_ROOT");
    unsetenv("NM_MATTER_MIDPOINT_PRECONTACT_POLISH");
    unsetenv("NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION");
    unsetenv("NM_MATTER_MIDPOINT_STAGE_COMPENSATION");
    std::cerr<<"midpoint_diagnostic_scope: "<<e.what()<<"\n"; return 1;
 } }
}

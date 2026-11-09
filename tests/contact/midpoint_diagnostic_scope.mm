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
    std::cout<<"midpoint capture scope and fingerprints verified\n";
    return 0;
 } catch(const std::exception& e) {
    unsetenv("NM_MATTER_MIDPOINT_DIAGNOSTIC_ROOT");
    std::cerr<<"midpoint_diagnostic_scope: "<<e.what()<<"\n"; return 1;
 } }
}

// Protect the scoped midpoint transaction until a two-owner scheme is certified.
#define main numi_plane_contact_diagnostic_main
#include "plane_contact_probe.mm"
#undef main

int main(int argc,char** argv) {
 @autoreleasepool { try {
    require(argc==2,"usage: midpoint_dynamic_guard_probe METALLIB");
    constexpr double h=0.000015625;
    const auto fixed=model(2,h,0.0002,true,false,true,0.,1.);
    // The physical residual policy currently rejects dynamic rows at compile
    // time. Use the legacy policy to reach and test the subsequent owner guard.
    const auto articulated=model(2,h,0.0002,false,false,true,0.,1.,true);
    require(articulated.contact.rigidProxies.size()==1u &&
        articulated.contact.rigidProxies[0].bodyIndex==1u &&
        (articulated.contact.rigidProxies[0].flags & NM_RIGID_ARTICULATED),
        "fixture did not compile a real articulated body-bound proxy");
    numi::matter::RuntimeConfiguration config;
    config.metallib=argv[1];config.environmentCount=1;config.captureDiagnostics=true;
    config.adaptiveTransfer=false;config.experimentalFEMPlaneAreaIPC=true;
    config.experimentalFEMPlaneMidpoint=true;
    numi::matter::Runtime accepted;
    auto fixedResult=accepted.initialize(fixed,config);
    require(fixedResult.encoded && accepted.valid(),"fixed fixture rejected: "+fixedResult.message);
    numi::matter::Runtime denied;
    auto dynamicResult=denied.initialize(articulated,config);
    require(!dynamicResult.encoded && !denied.valid() &&
        dynamicResult.message.find("fixed plane")!=std::string::npos,
        "articulated body was not rejected by the intended fixed-plane guard: "+dynamicResult.message);
    std::cout<<"{\"fixed_plane_initialized\":true,\"articulated_plane_initialized\":false,"
        "\"owner_boundary\":\"fixed plane required\"}\n";
    return 0;
 } catch(const std::exception& e) {std::cerr<<"midpoint_dynamic_guard_probe: "<<e.what()<<"\n";return 1;} }
}

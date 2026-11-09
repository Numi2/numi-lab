// CPU-only physical-residual schema scope test. No Metal runtime is started.
#define main numi_plane_contact_diagnostic_main
#include "../../apps/plane_contact_probe.mm"
#undef main

int main() {
 @autoreleasepool { try {
    constexpr double h=0.000015625;
    const auto world=model(2,h,0.0002,true,false,true,0.,1.,true);
    require(world.dispatch.rigidGeneralizedCapacity==1u &&
        world.dispatch.rigidQCapacity==1u && world.contact.rigidProxies.size()==1u,
        "physical slider fixture lost its owner");
    std::string error;
    require(numi::matter::validateCompiledWorldLayout(world,&error),
        "physical slider fixture rejected: "+error);
    auto rejected=[&](auto candidate,const char* name) {
        std::string reason;
        require(!numi::matter::validateCompiledWorldLayout(candidate,&reason),
            std::string("invalid schema admitted: ")+name);
    };
    auto candidate=world;
    candidate.dispatch.rigidGeneralizedCapacity=2u;
    rejected(candidate,"two generalized coordinates");
    candidate=world; candidate.dispatch.rigidQCapacity=2u;
    rejected(candidate,"two position coordinates");
    candidate=world; candidate.contact.rigidProxies[0].bodyIndex=2u;
    rejected(candidate,"different body identity");
    candidate=world; candidate.contact.rigidProxies[0].shapeKind=NM_RIGID_SPHERE;
    rejected(candidate,"non-plane shape");
    candidate=world; candidate.contact.rigidProxies[0].flags=0u;
    rejected(candidate,"non-articulated proxy");
    numi::matter::RuntimeConfiguration configuration;
    configuration.environmentCount=1u;
    configuration.experimentalFEMPlaneAreaIPC=true;
    configuration.experimentalFEMPlaneMidpoint=true;
    auto denied=[&](const char* expected) {
        numi::matter::Runtime runtime;
        const auto result=runtime.initialize(world,configuration);
        require(!result.encoded && !runtime.valid() &&
            result.message.find(expected)!=std::string::npos,
            std::string("runtime guard failed: ")+result.message);
    };
    denied("requires explicit slider midpoint research admission");
    configuration.experimentalFEMPlaneSliderMidpoint=true;
    denied("not yet qualified for execution");
    configuration.experimentalFEMPlaneMidpoint=false;
    denied("requires the restricted one-axis");
    configuration.experimentalFEMPlaneMidpoint=true;
    configuration.experimentalFEMPlaneAreaIPC=false;
    denied("requires the restricted one-axis");
    configuration.experimentalFEMPlaneAreaIPC=true;
    configuration.experimentalFEMPlaneAugmentedLagrangian=true;
    denied("requires the restricted one-axis");
    std::cout<<"{\"physical_slider_schema\":true,\"out_of_scope_rejected\":5,"
        "\"runtime_guards_passed\":5}\n";
    return 0;
 } catch(const std::exception& e) {std::cerr<<"slider_schema_scope: "<<e.what()<<"\n";return 1;} }
}

// Protect the scoped midpoint transaction until a two-owner scheme is certified.
#define main numi_plane_contact_diagnostic_main
#include "plane_contact_probe.mm"
#undef main

int main(int argc,char** argv) {
 @autoreleasepool { try {
    require(argc==2,"usage: midpoint_dynamic_guard_probe METALLIB");
    constexpr double h=0.000015625;
    const auto fixed=model(2,h,0.0002,true,false,true,0.,1.);
    // Exercise both the legacy fixed-plane guard and the restricted physical
    // fixture admission. Neither path permits a reacting-body transaction.
    const auto articulated=model(2,h,0.0002,false,false,true,0.,1.,true);
    require(articulated.contact.rigidProxies.size()==1u &&
        articulated.contact.rigidProxies[0].bodyIndex==1u &&
        (articulated.contact.rigidProxies[0].flags & NM_RIGID_ARTICULATED),
        "fixture did not compile a real articulated body-bound proxy");
    require(articulated.dispatch.rigidGeneralizedCapacity==1u &&
        articulated.dispatch.rigidQCapacity==1u,
        "articulated fixture must match the one-axis owner's capacities");
    const auto physicalArticulated=model(2,h,0.0002,true,false,true,0.,1.,true);
    require(physicalArticulated.dispatch.rigidGeneralizedCapacity==1u &&
        physicalArticulated.dispatch.rigidQCapacity==1u &&
        physicalArticulated.contact.rigidProxies.size()==1u,
        "physical articulated fixture did not retain its owner");
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
    numi::matter::Runtime deniedPhysical;
    auto physicalResult=deniedPhysical.initialize(physicalArticulated,config);
    require(!physicalResult.encoded && !deniedPhysical.valid() &&
        physicalResult.message.find("explicit slider midpoint research admission")!=std::string::npos,
        "physical owner was not rejected without explicit admission: "+physicalResult.message);
    config.experimentalFEMPlaneSliderMidpoint=true;
    numi::matter::Runtime deniedResearch;
    auto researchResult=deniedResearch.initialize(physicalArticulated,config);
    require(!researchResult.encoded && !deniedResearch.valid() &&
        researchResult.message.find("not yet qualified for execution")!=std::string::npos,
        "unqualified research owner was not rejected: "+researchResult.message);
    config.experimentalFEMPlaneMidpoint=false;
    numi::matter::Runtime deniedPrerequisite;
    auto prerequisiteResult=deniedPrerequisite.initialize(physicalArticulated,config);
    require(!prerequisiteResult.encoded && !deniedPrerequisite.valid() &&
        prerequisiteResult.message.find("requires the restricted one-axis")!=std::string::npos,
        "incomplete research configuration was not rejected: "+prerequisiteResult.message);
    std::cout<<"{\"fixed_plane_initialized\":true,\"articulated_plane_initialized\":false,"
        "\"physical_owner_default_rejected\":true,\"research_execution_rejected\":true,"
        "\"missing_prerequisite_rejected\":true}\n";
    return 0;
 } catch(const std::exception& e) {std::cerr<<"midpoint_dynamic_guard_probe: "<<e.what()<<"\n";return 1;} }
}

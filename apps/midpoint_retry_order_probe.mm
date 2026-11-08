// Verify that failed diagnostic ordering cannot affect a later valid retry.
#define main numi_plane_contact_diagnostic_main
#include "plane_contact_probe.mm"
#undef main

int main(int argc,char** argv){
 @autoreleasepool {try {
    require(argc==2,"usage: midpoint_retry_order_probe METALLIB");
    constexpr double h=0.0000625;
    const auto world=model(2,h,0.00009,true,false,true,0.,1.);
    numi::matter::Runtime runtime;numi::matter::RuntimeConfiguration config;
    config.metallib=argv[1];config.environmentCount=1;config.captureDiagnostics=true;
    config.adaptiveTransfer=false;config.experimentalFEMPlaneAreaIPC=true;
    config.experimentalFEMPlaneMidpoint=true;
    auto initialized=runtime.initialize(world,config);require(initialized.encoded,"initialize: "+initialized.message);
    auto initial=runtime.snapshot();require(initial.available && initial.statuses.size()==1,"initial snapshot unavailable");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLBuffer> statuses=[device newBufferWithLength:sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
    auto* outer=static_cast<MRMetalWorldStatusGPU*>(statuses.contents);
    const auto retry=[&](unsigned failingIndex,float diagnostic){
        auto state=initial;
        state.statuses[0].code=NM_STATUS_INVALID_DEFORMATION;
        state.statuses[0].objectIndex=0;
        state.statuses[0].failingIndex=failingIndex;
        state.statuses[0].diagnostics={diagnostic,1.f,0.f,0.f};
        auto restored=runtime.restore(state);require(restored.encoded,"restore failed: "+restored.message);
        *outer={};outer->code=MR_STEP_SUCCESS;
        id<MTLCommandBuffer> command=[queue commandBuffer];numi::matter::EncodeRequest request;
        request.commandBuffer=(__bridge void*)command;request.environmentStatuses=(__bridge void*)statuses;
        request.controlStep=0;request.physicsSubsteps=1;request.timestepSeconds=h;
        request.phase=numi::matter::EncodePhase::preDynamics;auto encoded=runtime.encode(request);
        require(encoded.encoded,"encode pre: "+encoded.message);
        request.phase=numi::matter::EncodePhase::postCommit;encoded=runtime.encode(request);
        require(encoded.encoded,"encode post: "+encoded.message);
        [command commit];[command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted && outer->code==MR_STEP_SUCCESS,"retry rejected");
        auto output=runtime.snapshot();require(output.available && output.statuses[0].code==NM_STATUS_SUCCESS,"retry status failed");
        return output;
    };
    const auto a=retry(0u,-.4989f),b=retry(32u,-.4990f);
    const bool identical=physicalEqual(a,b) && a.controlStep==b.controlStep &&
        a.physicsSubstep==b.physicsSubstep && bytesEqual(a.statuses,b.statuses);
    bool contact=false;for(const auto& c:a.contactHistories)contact|=c.w>0.f;
    std::cout<<"{\"diagnostic_indices\":[0,32],\"both_accepted\":true,\"identical_authoritative_state_and_status\":"
      <<(identical?"true":"false")<<",\"nonzero_contact\":"<<(contact?"true":"false")<<"}\n";
    require(identical && contact,"prior diagnostic ordering changed accepted retry");return 0;
 }catch(const std::exception& e){std::cerr<<"midpoint_retry_order_probe: "<<e.what()<<"\n";return 1;} }
}

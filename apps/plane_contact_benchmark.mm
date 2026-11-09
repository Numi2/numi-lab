// Timed native Matter transactions with no per-step diagnostic snapshots.
// The diagnostic probe owns the exact fixture constructor used in accuracy runs.
#define main numi_plane_contact_diagnostic_main
#include "plane_contact_probe.mm"
#undef main

int main(int argc, char** argv) {
 @autoreleasepool { try {
    require(argc==6 || (argc==7 && std::string(argv[6])=="--graded-z"),"usage: plane_contact_benchmark METALLIB timestep steps divisions ipc_area|ipc_area_midpoint [--graded-z]");
    const bool gradedZ=argc==7;
    const double h=std::stod(argv[2]);
    const int steps=std::stoi(argv[3]), divisions=std::stoi(argv[4]);
    const std::string arm=argv[5];
    require(h>0 && steps>0 && steps<=320000 && divisions>=1 && divisions<=8,"invalid workload");
    require(arm=="ipc_area" || arm=="ipc_area_midpoint","invalid arm");
    const auto world=model(divisions,h,2.e-4,true,false,true,0.,1.,false,gradedZ);
    numi::matter::Runtime runtime;numi::matter::RuntimeConfiguration config;
    config.metallib=argv[1];config.environmentCount=1;config.captureDiagnostics=true;
    config.adaptiveTransfer=false;config.experimentalFEMPlaneAreaIPC=true;
    config.experimentalFEMPlaneMidpoint=arm=="ipc_area_midpoint";
    auto initialized=runtime.initialize(world,config);
    require(initialized.encoded,"initialize: "+initialized.message);
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLBuffer> statuses=[device newBufferWithLength:sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
    auto* outer=static_cast<MRMetalWorldStatusGPU*>(statuses.contents);
    double gpu=0;int accepted=0,rejected=0;std::string failure;
    const auto start=std::chrono::steady_clock::now();
    for(int i=0;i<steps;++i){
      @autoreleasepool {
        *outer={};outer->code=MR_STEP_SUCCESS;
        id<MTLCommandBuffer> command=[queue commandBuffer];numi::matter::EncodeRequest request;
        request.commandBuffer=(__bridge void*)command;request.environmentStatuses=(__bridge void*)statuses;
        request.controlStep=i;request.physicsSubsteps=1;request.timestepSeconds=h;request.runAdaptiveTransfer=false;
        request.phase=numi::matter::EncodePhase::preDynamics;auto encoded=runtime.encode(request);
        if(!encoded.encoded){failure="encode pre: "+encoded.message;++rejected;break;}
        request.phase=numi::matter::EncodePhase::postCommit;encoded=runtime.encode(request);
        if(!encoded.encoded){failure="encode post: "+encoded.message;++rejected;break;}
        [command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted){failure="GPU command failed";++rejected;break;}
        gpu+=command.GPUEndTime-command.GPUStartTime;
        if(outer->code!=MR_STEP_SUCCESS){failure="outer transaction rejected";++rejected;break;}
        ++accepted;
      }
    }
    const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    // One completion-boundary snapshot after the timed interval checks the
    // final status; it is never taken within the measured rollout.
    const auto final=runtime.snapshot();require(final.available,"final snapshot unavailable");
    const unsigned status=final.statuses.empty()?NM_STATUS_INVALID_DISPATCH:final.statuses[0].code;
    std::cout<<std::setprecision(17)<<"{\"schema\":\"numi.matter.rollout-cost.v1\",\"device\":\""<<device.name.UTF8String
      <<"\",\"arm\":\""<<arm<<"\",\"h\":"<<h<<",\"divisions\":"<<divisions
      <<",\"graded_z\":"<<(gradedZ?"true":"false")<<",\"requested_steps\":"<<steps<<",\"accepted_steps\":"<<accepted<<",\"rejected_steps\":"<<rejected
      <<",\"status\":"<<status<<",\"wall_seconds\":"<<wall<<",\"gpu_seconds\":"<<gpu
      <<",\"accepted_simulated_seconds\":"<<accepted*h<<",\"accepted_seconds_per_wall_second\":"<<accepted*h/wall
      <<",\"final_physical_sha256\":\""<<physicalDigest(final)<<"\""
      <<",\"per_step_snapshots\":0,\"capture_diagnostics\":true,\"source_fingerprint\":"<<world.physicsFingerprint
      <<",\"device_fingerprint\":"<<runtime.deviceProgramFingerprint()<<"}\n";
    if(!failure.empty())std::cerr<<failure<<"\n";
    return rejected || accepted!=steps || status!=NM_STATUS_SUCCESS ? 1 : 0;
 } catch(const std::exception& e) {std::cerr<<"plane_contact_benchmark: "<<e.what()<<"\n";return 1;} }
}

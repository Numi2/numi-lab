// Capture one previously observed midpoint rejection without modifying the solve.
#define main numi_plane_contact_diagnostic_main
#include "plane_contact_probe.mm"
#undef main
#include <fstream>
#include <filesystem>

int main(int argc,char** argv) {
 @autoreleasepool {try {
    require(argc==3,"usage: midpoint_failure_capture METALLIB CAPTURE_BINARY");
    require(!std::filesystem::exists(argv[2]),"capture already exists; refusing overwrite");
    constexpr double h=0.00000390625;
    constexpr unsigned failingStep=203u;
    const auto world=model(8,h,.0002,true,false,true,0.,1.);
    numi::matter::Runtime runtime;numi::matter::RuntimeConfiguration config;
    config.metallib=argv[1];config.environmentCount=1;config.captureDiagnostics=true;
    config.adaptiveTransfer=false;config.experimentalFEMPlaneAreaIPC=true;
    config.experimentalFEMPlaneMidpoint=true;
    auto initialized=runtime.initialize(world,config);
    require(initialized.encoded,"initialize: "+initialized.message);
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLBuffer> statuses=[device newBufferWithLength:sizeof(MRMetalWorldStatusGPU)
        options:MTLResourceStorageModeShared];
    const size_t n=world.dispatch.femNodeCount;
    const size_t nodesBytes=n*sizeof(NMFEMNodeStateGPU),residualBytes=n*sizeof(nm_float4);
    id<MTLBuffer> capture=[device newBufferWithLength:2*nodesBytes+residualBytes
        options:MTLResourceStorageModeShared];
    require(statuses && capture,"diagnostic allocation failed");
    auto* outer=static_cast<MRMetalWorldStatusGPU*>(statuses.contents);
    numi::matter::RuntimeStateSnapshot before;
    for(unsigned i=0;i<=failingStep;++i) {
        if(i==failingStep) {before=runtime.snapshot();require(before.available,"accepted snapshot unavailable");}
        *outer={};outer->code=MR_STEP_SUCCESS;
        id<MTLCommandBuffer> command=[queue commandBuffer];
        numi::matter::EncodeRequest request;
        request.commandBuffer=(__bridge void*)command;request.environmentStatuses=(__bridge void*)statuses;
        request.controlStep=i;request.physicsSubsteps=1;request.timestepSeconds=h;
        request.phase=numi::matter::EncodePhase::preDynamics;
        auto encoded=runtime.encode(request);require(encoded.encoded,"encode pre: "+encoded.message);
        if(i==failingStep) {
            id<MTLBuffer> midpoint=(__bridge id<MTLBuffer>)runtime.femMidpointCandidateDiagnosticBuffer();
            id<MTLBuffer> endpoint=(__bridge id<MTLBuffer>)runtime.femMidpointEndpointDiagnosticBuffer();
            id<MTLBuffer> residual=(__bridge id<MTLBuffer>)runtime.femMechanicalResidualBuffer();
            require(midpoint && endpoint && residual,"midpoint scratch views unavailable");
            id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
            [blit copyFromBuffer:midpoint sourceOffset:0 toBuffer:capture destinationOffset:0 size:nodesBytes];
            [blit copyFromBuffer:endpoint sourceOffset:0 toBuffer:capture destinationOffset:nodesBytes size:nodesBytes];
            [blit copyFromBuffer:residual sourceOffset:0 toBuffer:capture destinationOffset:2*nodesBytes size:residualBytes];
            [blit endEncoding];
        }
        request.phase=numi::matter::EncodePhase::postCommit;
        encoded=runtime.encode(request);require(encoded.encoded,"encode post: "+encoded.message);
        [command commit];[command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted,"GPU command failed");
        if(i<failingStep)require(outer->code==MR_STEP_SUCCESS,"failure occurred before retained step");
    }
    auto after=runtime.snapshot();require(after.available && after.statuses.size()==1,"failure snapshot unavailable");
    require(after.statuses[0].code==NM_STATUS_NONLINEAR_SOLVER_FAILURE && outer->code!=MR_STEP_SUCCESS,
        "retained nonlinear failure was not reproduced");
    require(physicalEqual(before,after) && before.controlStep==after.controlStep &&
        before.physicsSubstep==after.physicsSubstep,"failed step changed accepted state");
    require(before.femNodes.size()==n,"unexpected accepted node count");
    const auto* bytes=static_cast<const unsigned char*>(capture.contents);
    std::vector<NMFEMNodeStateGPU> midpoint(n),endpoint(n);
    std::vector<nm_float4> residual(n);
    std::memcpy(midpoint.data(),bytes,nodesBytes);
    std::memcpy(endpoint.data(),bytes+nodesBytes,nodesBytes);
    std::memcpy(residual.data(),bytes+2*nodesBytes,residualBytes);
    double stage2=0,round2=0,total2=0,maxStage=0,maxRound=0,maxTotal=0;
    for(size_t i=0;i<n;++i) {
        const auto& a=before.femNodes[i];const auto& m=midpoint[i];const auto& e=endpoint[i];
        const double av[3]={a.velocityAndInverseMass.x,a.velocityAndInverseMass.y,a.velocityAndInverseMass.z};
        const double mv[3]={m.velocityAndInverseMass.x,m.velocityAndInverseMass.y,m.velocityAndInverseMass.z};
        const double ev[3]={e.velocityAndInverseMass.x,e.velocityAndInverseMass.y,e.velocityAndInverseMass.z};
        const double rv[3]={residual[i].x,residual[i].y,residual[i].z};
        for(int j=0;j<3;++j) {
            const double stage=2*rv[j];
            const double rounding=double(a.positionAndMass.w)*(2*(mv[j]-av[j])-(ev[j]-av[j]));
            const double total=stage+rounding;
            require(std::isfinite(total),"nonfinite captured candidate");
            stage2+=stage*stage;round2+=rounding*rounding;total2+=total*total;
            maxStage=std::max(maxStage,std::abs(stage));maxRound=std::max(maxRound,std::abs(rounding));
            maxTotal=std::max(maxTotal,std::abs(total));
        }
    }
    auto endpointSnapshot=after;endpointSnapshot.femNodes=endpoint;
    // The retained failure is before contact, so candidate samples cannot
    // contribute a physically accepted contact impulse to this reconstruction.
    endpointSnapshot.contactSamples.clear();
    const auto physical=fixtureResidual(world,before,endpointSnapshot,h,true);
    std::ofstream binary(argv[2],std::ios::binary|std::ios::out|std::ios::trunc);
    require(bool(binary),"cannot create capture binary");
    const std::uint32_t header[4]={0x4e4d4d31u,std::uint32_t(n),
        std::uint32_t(sizeof(NMFEMNodeStateGPU)),std::uint32_t(sizeof(nm_float4))};
    binary.write(reinterpret_cast<const char*>(header),sizeof(header));
    binary.write(reinterpret_cast<const char*>(before.femNodes.data()),nodesBytes);
    binary.write(reinterpret_cast<const char*>(midpoint.data()),nodesBytes);
    binary.write(reinterpret_cast<const char*>(endpoint.data()),nodesBytes);
    binary.write(reinterpret_cast<const char*>(residual.data()),residualBytes);
    binary.close();require(bool(binary),"failed to write capture binary");
    const auto d=after.statuses[0].diagnostics;
    std::cout<<std::setprecision(17)<<"{\"step\":"<<failingStep<<",\"nodes\":"<<n
       <<",\"status\":"<<after.statuses[0].code<<",\"outer_status\":"<<outer->code
       <<",\"restored_exact\":true,\"accepted_sha256\":\""<<physicalDigest(before)<<"\""
       <<",\"gpu_reported_norm_Ns\":"<<d.x<<",\"gpu_limit_Ns\":"<<d.y
       <<",\"gpu_max_kinematic_defect_m\":"<<d.z
       <<",\"twice_stage_norm_Ns\":"<<std::sqrt(stage2)
       <<",\"reflection_correction_norm_Ns\":"<<std::sqrt(round2)
       <<",\"reconstructed_norm_Ns\":"<<std::sqrt(total2)
       <<",\"max_stage_component_Ns\":"<<maxStage
       <<",\"max_reflection_component_Ns\":"<<maxRound
       <<",\"max_reconstructed_component_Ns\":"<<maxTotal
       <<",\"independent_physical_norm_Ns\":"<<physical.l2
       <<",\"independent_local_scaled_max\":"<<physical.local
       <<",\"independent_kinematic_defect_m\":"<<physical.kinematic<<"}\n";
    return 0;
 }catch(const std::exception& e){std::cerr<<"midpoint_failure_capture: "<<e.what()<<"\n";return 1;} }
}

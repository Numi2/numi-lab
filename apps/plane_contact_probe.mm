#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void require(bool value,const std::string& text){if(!value)throw std::runtime_error(text);}
template<class T> bool bytesEqual(const std::vector<T>& a,const std::vector<T>& b){
    return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
bool physicalEqual(const numi::matter::RuntimeStateSnapshot& a,const numi::matter::RuntimeStateSnapshot& b){
    return bytesEqual(a.femNodes,b.femNodes)&&bytesEqual(a.femMaterialState,b.femMaterialState)&&
        bytesEqual(a.contactHistories,b.contactHistories)&&bytesEqual(a.schedulers,b.schedulers)&&
        bytesEqual(a.adaptive,b.adaptive)&&bytesEqual(a.femFields,b.femFields)&&
        bytesEqual(a.topologyStates,b.topologyStates)&&bytesEqual(a.deformableContactHistories,b.deformableContactHistories);
}
numi::matter::CompiledWorld model(int divisions,double h,double startGap){
    auto parsed=numi::matter::parseMatter(R"MAT(
material test_pad {
 parameter density : kg/m^3 = 1100 in [1000,1200];
 parameter mu : Pa = 2000 in [1000,3000];
 parameter lambda : Pa = 5000 in [4000,6000];
 model neo_hookean;
 energy = neo_hookean(mu,lambda);
 valid = J()-0.05;
 supports fem;
 interface { static_friction = 0; dynamic_friction = 0; restitution = 0; adhesion = 0; }
 limits { minimum_J = 0.05; maximum_J = 20; maximum_stress = 1e9; maximum_energy_density = 1e9; }
}
)MAT");
    require(parsed.succeeded(),"material parse failed");
    numi::matter::WorldSource source;source.environmentCount=1;source.frameTimestep=h;
    source.gravity={0,0,-9.81};source.contactSlop=1.e-4;source.materials.push_back(parsed.material);
    source.mixedSolver.newtonIterations=16;
    source.mixedSolver.fgmresRestart=16;source.mixedSolver.fgmresIterations=64;
    source.mixedSolver.relativeResidual=1.e-9;
    source.mixedSolver.minimumContactSeparationRatio=1.e-3;
    numi::matter::RigidProxySource plane;plane.shape=NM_RIGID_PLANE;plane.localCenter={0,0,1};
    source.rigidProxies.push_back(plane);
    numi::matter::ObjectSource object;object.name="frictionless_fem_pad";
    object.materialIndex=0;object.mixedFEM=false;object.representation=numi::matter::Representation::fem;
    object.characteristicLength=.02/divisions;object.deformableContact=false;object.deformableSelfContact=false;
    const auto index=[&](int x,int y,int z){return std::uint32_t((z*(divisions+1)+y)*(divisions+1)+x);};
    for(int z=0;z<=divisions;++z)for(int y=0;y<=divisions;++y)for(int x=0;x<=divisions;++x)
        object.femNodes.push_back({.02*x/divisions,.02*y/divisions,startGap+.02*z/divisions});
    const int tetra[6][4]={{0,1,3,7},{0,3,2,7},{0,2,6,7},{0,6,4,7},{0,4,5,7},{0,5,1,7}};
    for(int z=0;z<divisions;++z)for(int y=0;y<divisions;++y)for(int x=0;x<divisions;++x){
        const std::uint32_t v[8]={index(x,y,z),index(x+1,y,z),index(x,y+1,z),index(x+1,y+1,z),
            index(x,y,z+1),index(x+1,y,z+1),index(x,y+1,z+1),index(x+1,y+1,z+1)};
        for(auto& t:tetra)object.tetrahedra.push_back({{v[t[0]],v[t[1]],v[t[2]],v[t[3]]}});
    }
    source.objects.push_back(object);numi::matter::CompileOptions options;options.maximumRateExponent=0;
    auto compiled=numi::matter::compileWorld(source,options);
    if(!compiled.succeeded())for(const auto& d:compiled.diagnostics)std::cerr<<d.message<<"\n";
    require(compiled.succeeded(),"world compilation failed");return std::move(compiled.world);
}
}
int main(int argc,char** argv){
 @autoreleasepool {try{
    require(argc>=5,"usage: plane_contact_probe METALLIB timestep steps divisions [arm] [initial-gap]");
    const double h=std::stod(argv[2]);const int steps=std::stoi(argv[3]),divisions=std::stoi(argv[4]);
    require(h>0&&steps>0&&steps<=10000&&divisions>0&&divisions<=8,"invalid workload");
    const std::string arm=argc>5?argv[5]:"ipc";require(arm=="ipc"||arm=="al","invalid arm");
    const double startGap=argc>6?std::stod(argv[6]):2.e-4;
    const auto world=model(divisions,h,startGap);
    numi::matter::Runtime runtime;numi::matter::RuntimeConfiguration config;
    config.metallib=argv[1];config.environmentCount=1;config.captureDiagnostics=true;config.adaptiveTransfer=false;
    config.experimentalFEMPlaneAugmentedLagrangian=arm=="al";
    std::cerr<<"instrument_world contact_pairs="<<world.dispatch.contactPairCount<<" deformable_capacity="<<world.dispatch.deformableContactCapacity
        <<" rigid_generalized="<<world.dispatch.rigidGeneralizedCapacity<<" mutation="<<world.dispatch.mutationCommandCount
        <<" object_flags="<<world.objects[0].flags<<" plane_flags="<<world.contact.rigidProxies[0].flags
        <<" floor="<<world.mixedSolver.contactAcceptance.x*world.dispatch.numericalLimits.x<<" slop="<<world.dispatch.numericalLimits.x<<"\n";
    auto initialized=runtime.initialize(world,config);require(initialized.encoded,"initialize: "+initialized.message);
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLBuffer> statuses=[device newBufferWithLength:sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
    auto* outer=static_cast<MRMetalWorldStatusGPU*>(statuses.contents);
    auto step=[&](int i,bool forceRollback=false){
        *outer={};outer->code=MR_STEP_SUCCESS;
        id<MTLCommandBuffer> command=[queue commandBuffer];numi::matter::EncodeRequest request;
        request.commandBuffer=(__bridge void*)command;request.environmentStatuses=(__bridge void*)statuses;
        request.controlStep=i;request.physicsSubsteps=1;request.timestepSeconds=h;request.runAdaptiveTransfer=false;
        request.phase=numi::matter::EncodePhase::preDynamics;auto encoded=runtime.encode(request);
        require(encoded.encoded,"encode pre: "+encoded.message);
        if(forceRollback){
            id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
            [blit fillBuffer:statuses range:NSMakeRange(0,sizeof(MRMetalWorldStatusGPU)) value:2];
            [blit endEncoding];
        }
        request.phase=numi::matter::EncodePhase::postCommit;encoded=runtime.encode(request);
        require(encoded.encoded,"encode post: "+encoded.message);[command commit];[command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted,"GPU command failed");
        return command.GPUEndTime-command.GPUStartTime;
    };
    auto initial=runtime.snapshot();require(initial.available,"initial snapshot unavailable");
    double gpu=0;int accepted=0,rejected=0;double minimumGap=1.e100;bool first=true;
    std::cout<<std::setprecision(12)<<"{\"schema\":\"numi.matter.plane-contact.v1\",\"device\":\""<<device.name.UTF8String
        <<"\",\"arm\":\""<<arm<<"\",\"h\":"<<h<<",\"divisions\":"<<divisions<<",\"initial_gap\":"<<startGap<<",\"nodes\":"<<world.dispatch.femNodeCount
        <<",\"source_fingerprint\":"<<world.physicsFingerprint<<",\"device_fingerprint\":"<<runtime.deviceProgramFingerprint()<<",\"traces\":[";
    auto previous=initial;double totalWork=0,totalKineticWork=0,maxLawError=0,maxMomentumError=0,maxAcceptedDual=0;
    const auto start=std::chrono::steady_clock::now();
    for(int i=0;i<steps;++i){
        const double timing=step(i);gpu+=timing;auto snap=runtime.snapshot();require(snap.available,"snapshot unavailable");
        const bool ok=!snap.statuses.empty()&&snap.statuses[0].code==NM_STATUS_SUCCESS&&outer->code==MR_STEP_SUCCESS;
        if(ok)++accepted;else ++rejected;
        double low=1.e100,high=-1.e100,force=0,momentum=0,com=0,mass=0;int contacts=0;
        for(const auto& node:snap.femNodes){low=std::min(low,double(node.positionAndMass.z));high=std::max(high,double(node.positionAndMass.z));mass+=node.positionAndMass.w;com+=node.positionAndMass.w*double(node.positionAndMass.z);momentum+=node.positionAndMass.w*double(node.velocityAndInverseMass.z);}
        for(const auto& sample:snap.contactSamples)if(sample.identity.w&NM_CONTACT_VALID){force+=sample.impulseAndNormal.w/h;++contacts;}
        double expectedEnergy=0;
        if(arm=="al")for(const auto& sample:snap.contactSamples) {
            if(!(sample.identity.w&NM_CONTACT_VALID))continue;
            const double normal=std::max(0.,double(sample.barrier.w)-config.experimentalFEMPlanePenalty*
                (double(sample.pointAndSeparation.w)-config.experimentalFEMPlaneClearance));
            const double expectedImpulse=normal*h,curvature=normal>0?config.experimentalFEMPlanePenalty*h*h:0.;
            maxLawError=std::max(maxLawError,std::abs(expectedImpulse-sample.impulseAndNormal.w)/std::max(1e-9,expectedImpulse));
            maxLawError=std::max(maxLawError,std::abs(curvature-sample.barrier.y)/std::max(1e-9,curvature));
            const double n[3]={sample.normalAndVelocity.x,sample.normalAndVelocity.y,sample.normalAndVelocity.z};
            const nm_float4 rows[3]={sample.barrierHessianRow0,sample.barrierHessianRow1,sample.barrierHessianRow2};
            for(int i=0;i<3;++i){const double row[3]={rows[i].x,rows[i].y,rows[i].z};
                for(int j=0;j<3;++j)maxLawError=std::max(maxLawError,std::abs(row[j]-curvature*n[i]*n[j])/std::max(1e-9,curvature));}
            expectedEnergy+=normal*normal/(2*config.experimentalFEMPlanePenalty);
        }
        if(arm=="al")maxLawError=std::max(maxLawError,
            std::abs(expectedEnergy-snap.solverCertificates[0].contact.w)/std::max(1e-12,expectedEnergy));
        if(ok){
            double previousMomentum=0;
            for(const auto& node:previous.femNodes)previousMomentum+=double(node.positionAndMass.w)*node.velocityAndInverseMass.z;
            maxMomentumError=std::max(maxMomentumError,std::abs(momentum-previousMomentum+mass*9.81*h-force*h));
            for(const auto& dual:snap.contactHistories)maxAcceptedDual=std::max(maxAcceptedDual,double(dual.w));
        }
        double contactWork=0,kineticWork=0;
        if(ok)for(const auto& sample:snap.contactSamples) {
            if(!(sample.identity.w&NM_CONTACT_VALID)||sample.identity.x<world.dispatch.gridNodeCount)continue;
            const auto i=sample.identity.x-world.dispatch.gridNodeCount;require(i<snap.femNodes.size(),"contact node outside FEM arena");
            const auto& a=previous.femNodes[i];const auto& b=snap.femNodes[i];
            const double j[3]={-sample.impulseAndNormal.x,-sample.impulseAndNormal.y,-sample.impulseAndNormal.z};
            const double dx[3]={b.positionAndMass.x-a.positionAndMass.x,b.positionAndMass.y-a.positionAndMass.y,b.positionAndMass.z-a.positionAndMass.z};
            const double v[3]={double(a.velocityAndInverseMass.x)+b.velocityAndInverseMass.x,double(a.velocityAndInverseMass.y)+b.velocityAndInverseMass.y,double(a.velocityAndInverseMass.z)+b.velocityAndInverseMass.z};
            for(int axis=0;axis<3;++axis){contactWork+=j[axis]*dx[axis]/h;kineticWork+=.5*j[axis]*v[axis];}
        }
        totalWork+=contactWork;totalKineticWork+=kineticWork;
        previous=snap;
        minimumGap=std::min(minimumGap,low);
        if(!first)std::cout<<",";first=false;
        std::cout<<"{\"step\":"<<i<<",\"accepted\":"<<(ok?"true":"false")<<",\"status\":"<<snap.statuses[0].code<<",\"gap\":"<<low<<",\"height\":"<<high-low<<",\"force\":"<<force<<",\"contact_work_J\":"<<contactWork<<",\"contact_impulse_midvelocity_work_J\":"<<kineticWork<<",\"accepted_normal_impulse\":"<<(ok?force*h:0.)<<",\"normal_impulse\":"<<force*h<<",\"mass\":"<<mass<<",\"com_z\":"<<com/mass<<",\"momentum_z\":"<<momentum<<",\"contacts\":"<<contacts<<",\"fgmres_iterations\":"<<snap.statuses[0].fgmresIterations<<",\"outer_status\":"<<outer->code
            <<",\"certificate_residual\":"<<snap.solverCertificates[0].nonlinear.x<<",\"certificate_min_J\":"<<snap.solverCertificates[0].validity.x
            <<",\"certificate_accepted\":"<<snap.solverCertificates[0].validity.w<<",\"status_values\":["<<snap.statuses[0].diagnostics.x<<","<<snap.statuses[0].diagnostics.y<<","<<snap.statuses[0].diagnostics.z<<","<<snap.statuses[0].diagnostics.w<<"]"
            <<",\"gpu_seconds\":"<<timing<<"}";
        if(!ok)break;
    }
    const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    const auto final=runtime.snapshot();
    auto restored=runtime.restore(initial);require(restored.encoded,"restore initial: "+restored.message);
    for(int i=0;i<accepted+rejected;++i)(void)step(i);
    const auto continued=runtime.snapshot();const bool replay=physicalEqual(final,continued) && final.controlStep==continued.controlStep && final.physicsSubstep==continued.physicsSubstep && bytesEqual(final.statuses,continued.statuses);
    auto mismatch=initial;mismatch.deviceProgramFingerprint^=1u;
    const bool identityRejected=!runtime.restore(mismatch).encoded;
    restored=runtime.restore(final);require(restored.encoded,"restore for rollback failed");
    const auto beforeRollback=runtime.snapshot();
    (void)step(steps,true);const auto rolled=runtime.snapshot();
    const bool rollback=physicalEqual(beforeRollback,rolled);
    restored=runtime.restore(initial);require(restored.encoded,"restore before reset failed");
    // Snapshot restore is exercised above with both nodes and persistent duals.

    std::cout<<"],\"accepted_steps\":"<<accepted<<",\"rejected_steps\":"<<rejected<<",\"wall_seconds\":"<<wall
        <<",\"contact_work_J\":"<<totalWork<<",\"contact_impulse_midvelocity_work_J\":"<<totalKineticWork<<",\"gpu_seconds\":"<<gpu<<",\"accepted_simulated_seconds\":"<<accepted*h
        <<",\"accepted_simulated_seconds_per_wall_second\":"<<accepted*h/wall<<",\"minimum_gap\":"<<minimumGap
        <<",\"law_max_relative_error\":"<<maxLawError<<",\"momentum_max_absolute_error_Ns\":"<<maxMomentumError<<",\"maximum_accepted_multiplier_impulse_Ns\":"<<maxAcceptedDual<<",\"identity_mismatch_rejected\":"<<(identityRejected?"true":"false")<<",\"forced_rollback_exact\":"<<(rollback?"true":"false")<<",\"restore_replay\":"<<(replay?"true":"false")<<"}\n";
    return rejected||!replay||!rollback||!identityRejected||maxLawError>1e-4||maxMomentumError>1e-7?1:0;
 }catch(const std::exception& e){std::cerr<<"plane_contact_probe: "<<e.what()<<"\n";return 1;}}
}

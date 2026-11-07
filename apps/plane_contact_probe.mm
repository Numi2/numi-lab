#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
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

// Independent FP64 evaluation of the fixture equations, without integration.
struct FixtureResidual { double l2=0,local=0,kinematic=0,velocity=0; };
FixtureResidual fixtureResidual(const numi::matter::CompiledWorld& world,
    const numi::matter::RuntimeStateSnapshot& old,
    const numi::matter::RuntimeStateSnapshot& cur,double timestep) {
    using V=std::array<double,3>;using M=std::array<V,3>;
    const double h=float(timestep),g=float(-9.81);
    std::vector<V> force(cur.femNodes.size());
    for(const auto& t:world.fem.tetrahedra) {
        const unsigned ids[4]={t.nodes.x,t.nodes.y,t.nodes.z,t.nodes.w};
        const auto a=t.inverseRestRow0,b=t.inverseRestRow1,c=t.inverseRestRow2;
        const M B={V{a.x,a.y,a.z},V{b.x,b.y,b.z},V{c.x,c.y,c.z}};
        V x[4];bool reference=true;
        for(int j=0;j<4;++j)reference&=cur.femNodes[ids[j]].referenceDisplacementAndMode.w==1;
        for(int j=0;j<4;++j) {
            const auto& n=cur.femNodes[ids[j]];
            if(reference) x[j]={double(n.referenceDisplacementAndMode.x)+n.referenceDisplacementLow.x,
                double(n.referenceDisplacementAndMode.y)+n.referenceDisplacementLow.y,
                double(n.referenceDisplacementAndMode.z)+n.referenceDisplacementLow.z};
            else x[j]={n.positionAndMass.x,n.positionAndMass.y,n.positionAndMass.z};
        }
        M F{};for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
            F[i][j]=reference&&i==j?1:0;
            for(int k=0;k<3;++k)F[i][j]+=(x[k+1][i]-x[0][i])*B[k][j];
        }
        const double J=F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1])
            -F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0])
            +F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]);
        require(J>0,"FP64 diagnostic found inverted tet");
        M invT{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)
            invT[i][j]=(F[(i+1)%3][(j+1)%3]*F[(i+2)%3][(j+2)%3]
                -F[(i+1)%3][(j+2)%3]*F[(i+2)%3][(j+1)%3])/J;
        M P{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)
            P[i][j]=2000.*F[i][j]+(5000.*std::log(J)-2000.)*invT[i][j];
        for(int j=0;j<3;++j)for(int i=0;i<3;++i) {
            double f=0;for(int k=0;k<3;++k)f-=a.w*P[i][k]*B[j][k];
            force[ids[j+1]][i]+=f;force[ids[0]][i]-=f;
        }
    }
    std::vector<V> impulse(cur.femNodes.size());
    for(const auto& s:cur.contactSamples)if(s.identity.w&NM_CONTACT_VALID) {
        require(s.identity.x>=world.dispatch.gridNodeCount,"diagnostic requires FEM contact");
        const auto i=s.identity.x-world.dispatch.gridNodeCount;
        require(i<impulse.size(),"diagnostic contact outside FEM nodes");
        impulse[i][0]-=s.impulseAndNormal.x;impulse[i][1]-=s.impulseAndNormal.y;
        impulse[i][2]-=s.impulseAndNormal.z;
    }
    FixtureResidual r;
    for(unsigned i=0;i<cur.femNodes.size();++i) {
        const auto& a=old.femNodes[i];const auto& b=cur.femNodes[i];
        const double mass=b.positionAndMass.w;
        const V x={b.positionAndMass.x,b.positionAndMass.y,b.positionAndMass.z};
        const V y={a.positionAndMass.x,a.positionAndMass.y,a.positionAndMass.z};
        const V v={b.velocityAndInverseMass.x,b.velocityAndInverseMass.y,b.velocityAndInverseMass.z};
        const V u={a.velocityAndInverseMass.x,a.velocityAndInverseMass.y,a.velocityAndInverseMass.z};
        double sum=0;for(int j=0;j<3;++j) {
            const double e=h*(force[i][j]+(j==2?mass*g:0))-mass*(v[j]-u[j])+impulse[i][j];
            sum+=e*e;
            r.kinematic=std::max(r.kinematic,std::abs(x[j]-y[j]-h*v[j]));
            r.velocity=std::max(r.velocity,std::abs(v[j]-u[j]-(j==2?h*g:0)));
        }
        r.l2+=sum;r.local=std::max(r.local,std::sqrt(sum)/(mass*std::abs(g)*h+1e-10));
    }
    r.l2=std::sqrt(r.l2);return r;
}
numi::matter::CompiledWorld model(int divisions,double h,double startGap,bool physical,bool freefall,bool reference){
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
    if(reference)for(auto& p:source.materials.back().parameters)p.identifiable=false;
    source.mixedSolver.newtonIterations=16;
    source.mixedSolver.fgmresRestart=16;source.mixedSolver.fgmresIterations=64;
    source.mixedSolver.relativeResidual=1.e-9;
    source.mixedSolver.minimumContactSeparationRatio=1.e-3;
    numi::matter::RigidProxySource plane;plane.shape=NM_RIGID_PLANE;plane.localCenter={0,0,1};
    if(!freefall)source.rigidProxies.push_back(plane);
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
    if(reference) {
        object.femReferenceNodes=object.femNodes;
        object.femMaterialIndices.assign(object.tetrahedra.size(),0);
        object.femReferenceDisplacementGradient=true;
        std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> hash;
        CC_SHA256(object.femNodes.data(),CC_LONG(object.femNodes.size()*sizeof(object.femNodes[0])),hash.data());
        std::memcpy(object.femReferenceSourceIdentity.data(),hash.data(),hash.size());
        CC_SHA256(object.femMaterialIndices.data(),CC_LONG(object.femMaterialIndices.size()*sizeof(std::uint32_t)),hash.data());
        std::memcpy(object.femMaterialSourceIdentity.data(),hash.data(),hash.size());
    }
    if(physical) {
        std::vector<double> nodalMass(object.femNodes.size(),0.0);
        const double portion=1100.*.02*.02*.02/(24.*divisions*divisions*divisions);
        for(const auto& tet:object.tetrahedra)for(auto id:tet.nodes)nodalMass[id]+=portion;
        double sum=0.;for(double m:nodalMass)sum+=m*m;
        source.mixedSolver.mechanicalAbsoluteImpulseTolerance=.00025*9.81*h*(*std::min_element(nodalMass.begin(),nodalMass.end()));
        source.mixedSolver.mechanicalRelativeImpulseTolerance=.00225*(*std::min_element(nodalMass.begin(),nodalMass.end()))/std::sqrt(sum);
        source.mixedSolver.mechanicalImpulseScale=9.81*h*std::sqrt(sum);
    }
    source.objects.push_back(object);numi::matter::CompileOptions options;options.maximumRateExponent=0;
    auto compiled=numi::matter::compileWorld(source,options);
    if(!compiled.succeeded())for(const auto& d:compiled.diagnostics)std::cerr<<d.message<<"\n";
    require(compiled.succeeded(),"world compilation failed");return std::move(compiled.world);
}
}
int main(int argc,char** argv){
 @autoreleasepool {try{
    require(argc>=5,"usage: plane_contact_probe METALLIB timestep steps divisions [ipc|al|ipc_area] [initial-gap] [legacy|physical] [contact|freefall] [world|reference]");
    const double h=std::stod(argv[2]);const int steps=std::stoi(argv[3]),divisions=std::stoi(argv[4]);
    require(h>0&&steps>0&&steps<=10000&&divisions>0&&divisions<=8,"invalid workload");
    const std::string arm=argc>5?argv[5]:"ipc";require(arm=="ipc"||arm=="al"||arm=="ipc_area","invalid arm");
    const double startGap=argc>6?std::stod(argv[6]):2.e-4;
    const std::string policy=argc>7?argv[7]:"legacy";
    require(policy=="physical"||policy=="legacy","invalid residual policy");
    const bool freefall=argc>8&&std::string(argv[8])=="freefall";
    const bool reference=argc>9&&std::string(argv[9])=="reference";
    const auto world=model(divisions,h,startGap,policy=="physical",freefall,reference);
    numi::matter::Runtime runtime;numi::matter::RuntimeConfiguration config;
    config.metallib=argv[1];config.environmentCount=1;config.captureDiagnostics=true;config.adaptiveTransfer=false;
    config.experimentalFEMPlaneAugmentedLagrangian=arm=="al";
    config.experimentalFEMPlaneAreaIPC=arm=="ipc_area";
    if(argc>10)config.experimentalFEMPlaneAreaIPCK=std::stof(argv[10]);
    std::cerr<<"instrument_world contact_pairs="<<world.dispatch.contactPairCount<<" deformable_capacity="<<world.dispatch.deformableContactCapacity
        <<" rigid_generalized="<<world.dispatch.rigidGeneralizedCapacity<<" mutation="<<world.dispatch.mutationCommandCount
        <<" object_flags="<<world.objects[0].flags<<" plane_flags="<<(world.contact.rigidProxies.empty()?0:world.contact.rigidProxies[0].flags)
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
        <<",\"area_ipc_k_N_per_m5\":"<<(arm=="ipc_area"?config.experimentalFEMPlaneAreaIPCK:0.f)<<",\"residual_policy\":\""<<policy<<"\",\"coordinate_mode\":\""<<(reference?"reference_displacement":"world_position")<<"\",\"freefall_control\":"<<(freefall?"true":"false")<<",\"mechanical_absolute_tolerance_Ns\":"<<world.mixedSolver.contactAcceptance.y<<",\"mechanical_relative_tolerance\":"<<world.mixedSolver.contactAcceptance.z<<",\"mechanical_impulse_scale_Ns\":"<<world.mixedSolver.contactAcceptance.w<<",\"source_fingerprint\":"<<world.physicsFingerprint<<",\"device_fingerprint\":"<<runtime.deviceProgramFingerprint()<<",\"traces\":[";
    auto previous=initial;double totalWork=0,totalKineticWork=0,maxLawError=0,maxMomentumError=0,maxAcceptedDual=0,maxLocalResidual=0,maxKinematicDefect=0,maxFreefallVelocityDefect=0;
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
        FixtureResidual independent;
        if(ok) { independent=fixtureResidual(world,previous,snap,h);maxLocalResidual=std::max(maxLocalResidual,independent.local);maxKinematicDefect=std::max(maxKinematicDefect,independent.kinematic);if(freefall)maxFreefallVelocityDefect=std::max(maxFreefallVelocityDefect,independent.velocity); }
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
            <<",\"independent_residual_l2_Ns\":"<<independent.l2<<",\"independent_max_local_scaled_residual\":"<<independent.local<<",\"position_velocity_defect_m\":"<<independent.kinematic<<",\"gpu_seconds\":"<<timing<<"}";
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
        <<",\"law_max_relative_error\":"<<maxLawError<<",\"momentum_max_absolute_error_Ns\":"<<maxMomentumError<<",\"independent_max_local_scaled_residual\":"<<maxLocalResidual<<",\"position_velocity_max_defect_m\":"<<maxKinematicDefect<<",\"freefall_velocity_max_defect_mps\":"<<maxFreefallVelocityDefect<<",\"maximum_accepted_multiplier_impulse_Ns\":"<<maxAcceptedDual<<",\"identity_mismatch_rejected\":"<<(identityRejected?"true":"false")<<",\"forced_rollback_exact\":"<<(rollback?"true":"false")<<",\"restore_replay\":"<<(replay?"true":"false")<<"}\n";
    return rejected||!replay||!rollback||!identityRejected||maxLawError>1e-4||maxMomentumError>1e-7||(policy=="physical"&&maxLocalResidual>.005)?1:0;
 }catch(const std::exception& e){std::cerr<<"plane_contact_probe: "<<e.what()<<"\n";return 1;}}
}

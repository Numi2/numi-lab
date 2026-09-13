#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/MujocoMuscleReference.hpp"
#include "metalrobo/mujoco_muscle_gpu.h"
#include "metalrobo/numi_human_tendon_gpu.h"
#include "metalrobo/numi_human_extensor_hood_gpu.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Constitutive/geometry evaluation only. This driver never advances physics.
// Independent FP64 Core source equations are the route/force oracle; an
// independently derived collinear tension/foundation equilibrium is the hood
// oracle. Neither uses the compensated arithmetic implementation.
namespace {
using namespace metalrobo;
using V = std::array<double, 3>;
unsigned checks = 0;
std::string scenario;
void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(scenario+" "+message);
}
void near(double a, double b, double tolerance, const std::string& message) {
    require(std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=tolerance,
        message+": actual="+std::to_string(a)+" expected="+std::to_string(b));
}
mr_float4 f4(double x=0, double y=0, double z=0, double w=0) {
    return {float(x),float(y),float(z),float(w)};
}
double component(mr_float4 x, unsigned i) { return i==0?x.x:i==1?x.y:x.z; }
V cross(V a,V b) { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
constexpr unsigned dofs=7, bodies=3, jacobianCount=bodies*4*3*dofs;

struct Fixture {
    EngineModel model;
    std::vector<MujocoMuscleSite> sites;
    std::vector<MujocoWrapGeometry> wraps;
    MujocoMuscleDefinition definition;
    MujocoMuscleState state{double(.43f),double(.37f)};
    std::vector<MRMujocoMuscleSiteGPU> gpuSites;
    std::vector<MRMujocoMuscleWrapGPU> gpuWraps;
    std::vector<MRMujocoMuscleRouteNodeGPU> routes;
    MRMujocoMuscleGPU muscle{};
    MRMujocoMuscleStateGPU muscleState{};
    std::vector<double> velocity=std::vector<double>(dofs,0);
    Fixture(unsigned wrapKind) {
        model=makeFreeSphereEngineModel();
        auto body=model.bodies[1];
        model.bodies.assign(bodies,body);
        model.shapes.clear();model.materials.clear();
        auto& w=model.world;
        w.bodyCount=bodies;w.jointCount=2;w.shapeCount=0;w.materialCount=0;w.nq=8;w.nv=dofs;
        model.articulations[0]={0,MR_ROOT_FLOATING,0,bodies,0,2,0,8,0,dofs,0,0};
        model.joints.resize(2);
        for(unsigned i=0;i<2;++i) {
            auto& j=model.joints[i];j.parentBody=0;j.childBody=i+1;
            j.jointType=i==0?MR_JOINT_PRISMATIC:MR_JOINT_FIXED;
            j.qOffset=i==0?7:8;j.vOffset=i==0?6:7;j.nq=i==0?1:0;j.nv=i==0?1:0;
            j.axis0=f4(0,1,0);j.parentRotation=f4(0,0,0,1);j.childRotation=j.parentRotation;
            j.parentAnchor=i==0?f4(-.2,.01,.03):f4(.2,.01,-.02);
        }
        for(unsigned i=0;i<bodies;++i) {
            auto& b=model.bodies[i];b.articulationIndex=0;
            b.parentBody=i==0?MR_INVALID_INDEX:0;b.inboundJoint=i==0?MR_INVALID_INDEX:i-1;
        }
        model.dofs.resize(dofs);
        model.dofs[6]={0,0,7,6,0,0,0,0,{},{}};
        model.defaultQ.assign(8,0);model.defaultQ[6]=1;model.defaultV.assign(dofs,0);
        std::string reason;require(model.valid(&reason),"source oracle model: "+reason);
        sites={{0,{double(-.28f),double(.08f),double(.04f)}},{1,{double(.007f),double(.003f),double(.002f)}},
               {2,{double(-.004f),double(.001f),double(.003f)}},{0,{0,double(.1f),0}}};
        MujocoWrapGeometry wrap{};wrap.bodyIndex=0;
        wrap.type=wrapKind==3?MujocoRouteNodeType::cylinder:MujocoRouteNodeType::sphere;
        wrap.rotationBody={1,0,0,0,1,0,0,0,1};wrap.radius=double(.04f);wraps.push_back(wrap);
        definition.route={{MujocoRouteNodeType::site,0,MR_INVALID_INDEX},{MujocoRouteNodeType::site,1,MR_INVALID_INDEX}};
        if(wrapKind)definition.route.push_back({wrap.type,0,3});
        definition.route.push_back({MujocoRouteNodeType::site,2,MR_INVALID_INDEX});
        definition.lengthRange={double(.3f),double(.7f)};definition.accelerationScale=1;
        definition.controlRange={0,1};
        definition.gainParameters={double(.5f),double(1.5f),20,200,double(.5f),double(1.6f),10,double(1.3f),double(1.2f),0};
        definition.biasParameters=definition.gainParameters;
        definition.dynamicParameters={double(.01f),double(.04f),0,0,0,0,0,0,0,0};
        for(auto s:sites)gpuSites.push_back({s.bodyIndex,0,0,0,f4(s.localPoint[0],s.localPoint[1],s.localPoint[2])});
        MRMujocoMuscleWrapGPU gw{};gw.bodyIndex=0;gw.type=unsigned(wrap.type);gw.rotationRow0=f4(1,0,0);gw.rotationRow1=f4(0,1,0);gw.rotationRow2=f4(0,0,1);gw.radius=f4(wrap.radius);gpuWraps.push_back(gw);
        for(auto r:definition.route)routes.push_back({unsigned(r.type),r.targetIndex,r.sideSiteIndex,0});
        muscle.route={0,unsigned(routes.size()),0,0};muscle.lengthRangeAndAcceleration=f4(definition.lengthRange[0],definition.lengthRange[1],1);muscle.controlRange=f4(0,1);
        auto writeParameters=[](mr_float4* target,const std::array<double,10>& values){
            for(unsigned b=0;b<3;++b) {
                float v[4]{};for(unsigned a=0;a<4 && b*4+a<10;++a)v[a]=float(values[b*4+a]);
                target[b]={v[0],v[1],v[2],v[3]};
            }
        };
        writeParameters(muscle.gainParameters,definition.gainParameters);writeParameters(muscle.biasParameters,definition.biasParameters);writeParameters(muscle.dynamicParameters,definition.dynamicParameters);
        muscleState.excitationAndActivation=f4(state.excitation,state.activation);
        velocity[6]=double(.07f);
    }
    MujocoMuscleResult oracle(double joint,bool suffix=false,const std::vector<MujocoMuscleSite>* overrideSites=nullptr) const {
        std::vector<double> q(8,0);q[6]=1;q[7]=joint;
        auto def=definition;if(suffix)def.route.erase(def.route.begin());
        MujocoMuscleResult result;
        auto status=evaluateMujocoMuscle(model,0,q,velocity,overrideSites?*overrideSites:sites,wraps,def,state,result);
        require(status.succeeded(),"independent FP64 route evaluation "+std::to_string(unsigned(status.status)));
        return result;
    }
};

struct Geometry {
    std::vector<MRArticulatedBodyPoseGPU> poses=std::vector<MRArticulatedBodyPoseGPU>(bodies);
    std::vector<mr_float4> low=std::vector<mr_float4>(bodies);
    std::vector<float> jacobian=std::vector<float>(jacobianCount);
    Geometry(const Fixture& f,double origin,double joint) {
        for(unsigned b=0;b<bodies;++b) {
            V relative{};
            if(b)for(unsigned a=0;a<3;++a)relative[a]=component(f.model.joints[b-1].parentAnchor,a);
            if(b==1)relative[1]+=joint;
            V world{origin+relative[0],-origin*.5+relative[1],origin*.25+relative[2]};
            poses[b].position=f4(world[0],world[1],world[2]);poses[b].orientation=f4(0,0,0,1);
            low[b]=f4(world[0]-poses[b].position.x,world[1]-poses[b].position.y,world[2]-poses[b].position.z);
            for(unsigned p=0;p<4;++p)for(unsigned d=0;d<dofs;++d) {
                V value{},arm=relative;if(p)arm[p-1]+=1;
                if(d<3)value[d]=1;
                else if(d<6){V axis{};axis[d-3]=1;value=cross(axis,arm);}
                else if(b==1)value[1]=1;
                for(unsigned a=0;a<3;++a)jacobian[((b*4+p)*3+a)*dofs+d]=float(value[a]);
            }
        }
    }
};

struct GPU {
    id<MTLDevice> device;id<MTLCommandQueue> queue;id<MTLLibrary> library;
    explicit GPU(const char* path) {
        device=MTLCreateSystemDefaultDevice();require(device!=nil,"Metal device");queue=[device newCommandQueue];
        NSError* error=nil;library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&error];
        require(library!=nil,"source library");
    }
    id<MTLBuffer> buffer(const void* bytes,std::size_t size) {
        require(size>0,"nonempty GPU fixture allocation");
        id<MTLBuffer> b=bytes?[device newBufferWithBytes:bytes length:size options:MTLResourceStorageModeShared]:[device newBufferWithLength:size options:MTLResourceStorageModeShared];
        require(b!=nil,"GPU allocation");if(!bytes)std::memset([b contents],0,size);return b;
    }
    template<class T> id<MTLBuffer> object(const T& v) {return buffer(&v,sizeof(v));}
    template<class T> id<MTLBuffer> vector(const std::vector<T>& v) {return buffer(v.data(),v.size()*sizeof(T));}
    void dispatch(const char* name,std::initializer_list<std::pair<unsigned,id<MTLBuffer>>> bindings,unsigned count) {
        NSError* error=nil;id<MTLFunction> fn=[library newFunctionWithName:[NSString stringWithUTF8String:name]];
        require(fn!=nil,std::string("production kernel ")+name);
        auto pipeline=[device newComputePipelineStateWithFunction:fn error:&error];require(pipeline!=nil,"pipeline");
        auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:pipeline];
        for(auto [slot,b]:bindings)[enc setBuffer:b offset:0 atIndex:slot];
        [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<unsigned>(count,32),1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
        require(cb.status==MTLCommandBufferStatusCompleted,std::string("kernel completed ")+name);
    }
};
struct RouteOutput {MRMujocoMuscleResultGPU result;std::array<float,dofs> force,suffix;};
RouteOutput evaluateGPU(GPU& gpu,const Fixture& f,const Geometry& g,bool paired) {
    MRMujocoMuscleReferenceDispatchGPU d{};d.abiVersion=MR_MUJOCO_MUSCLE_REFERENCE_GPU_ABI_VERSION;
    d.muscleCount=1;d.siteCount=unsigned(f.sites.size());d.wrapCount=1;d.routeNodeCount=unsigned(f.routes.size());d.environmentCount=1;d.bodyPoseStride=bodies;d.dofCount=dofs;d.pointJacobianStride=jacobianCount;d.bodyJacobianPointStride=4;d.timestepSecondsAndReserved=f4(.000025);
    std::vector<float> velocity(f.velocity.begin(),f.velocity.end());
    auto pose=gpu.vector(g.poses),low=gpu.vector(g.low),jac=gpu.vector(g.jacobian),v=gpu.vector(velocity),db=gpu.object(d),muscle=gpu.object(f.muscle),state=gpu.object(f.muscleState),sites=gpu.vector(f.gpuSites),wraps=gpu.vector(f.gpuWraps),routes=gpu.vector(f.routes),result=gpu.buffer(nullptr,sizeof(MRMujocoMuscleResultGPU)),force=gpu.buffer(nullptr,dofs*sizeof(float));
    gpu.dispatch(paired?"mr_mujoco_muscle_reference_compensated":"mr_mujoco_muscle_reference",{{7,v},{8,pose},{10,low},{11,jac},{23,force},{24,db},{25,muscle},{26,state},{27,sites},{28,wraps},{29,routes},{30,result}},1);
    MRMujocoMuscleRouteCutDispatchGPU cd{MR_MUJOCO_MUSCLE_ROUTE_CUT_GPU_ABI_VERSION,1,0,0};MRMujocoMuscleRouteCutGPU cut{0,1,0,0};
    auto suffix=gpu.buffer(nullptr,dofs*sizeof(float));
    gpu.dispatch(paired?"mr_mujoco_muscle_route_suffix_jacobian_compensated":"mr_mujoco_muscle_route_suffix_jacobian",{{0,pose},{1,jac},{2,db},{3,muscle},{4,sites},{5,wraps},{6,routes},{7,gpu.object(cd)},{8,gpu.object(cut)},{9,suffix},{10,low}},1);
    RouteOutput out{};std::memcpy(&out.result,[result contents],sizeof(out.result));std::memcpy(out.force.data(),[force contents],dofs*sizeof(float));std::memcpy(out.suffix.data(),[suffix contents],dofs*sizeof(float));return out;
}

void terminalCheck(GPU& gpu,const Fixture& f,const Geometry& g,const RouteOutput& route) {
    MRNumiHumanTendonTransferDispatchGPU d{};d.abiVersion=MR_NUMI_HUMAN_TENDON_TRANSFER_GPU_ABI_VERSION;d.endpointCount=2;d.envelopeCount=1;d.muscleCount=1;d.environmentCount=1;d.dofCount=dofs;d.bodyPoseStride=bodies;d.pointJacobianStride=jacobianCount;d.bodyJacobianPointStride=4;
    std::vector<MRNumiHumanTendonBindingGPU> bindings(2);
    bindings[0].bodyIndex=0;bindings[0].mode=MR_NUMI_HUMAN_TENDON_TRANSFER_SOURCE_POINT;bindings[0].envelopeIndex=MR_INVALID_INDEX;bindings[0].sourceLocalPoint=f.gpuSites[0].localPoint;
    bindings[1].endpointOrdinal=1;bindings[1].bodyIndex=2;bindings[1].mode=MR_NUMI_HUMAN_TENDON_TRANSFER_DISTRIBUTED_ENVELOPE;bindings[1].boneStableId=1;bindings[1].sourceLocalPoint=f.gpuSites[2].localPoint;
    MRNumiHumanTendonEnvelopeGPU envelope{};envelope.bodyIndex=2;envelope.boneStableId=1;envelope.nodeCount=4;envelope.metrics=f4(0,.004,1,1);
    const int signs[4][3]={{1,1,1},{1,-1,-1},{-1,1,-1},{-1,-1,1}};
    for(unsigned n=0;n<4;++n){auto s=bindings[1].sourceLocalPoint;envelope.localNodes[n]=f4(s.x+signs[n][0]*.002,s.y+signs[n][1]*.002,s.z+signs[n][2]*.002);for(unsigned a=0;a<3;++a)envelope.forceMapRows[n*3+a]=f4(a==0?.25:0,a==1?.25:0,a==2?.25:0);}
    auto results=gpu.buffer(nullptr,2*sizeof(MRNumiHumanTendonTransferResultGPU)),correction=gpu.buffer(nullptr,2*dofs*sizeof(float));
    gpu.dispatch("mr_numi_human_tendon_transfer_compensated",{{0,gpu.object(d)},{1,gpu.vector(bindings)},{2,gpu.object(envelope)},{3,gpu.object(route.result)},{4,gpu.vector(g.poses)},{5,gpu.vector(g.jacobian)},{6,results},{7,correction},{8,gpu.vector(g.low)}},2);
    auto r=static_cast<const MRNumiHumanTendonTransferResultGPU*>([results contents]);auto c=static_cast<const float*>([correction contents]);
    for(unsigned e=0;e<2;++e){require(r[e].status==0,"terminal source/distributed status");for(unsigned j=0;j<dofs;++j)near(c[e*dofs+j],0,2e-5,"conservative terminal generalized correction");}
    for(unsigned a=0;a<3;++a){double sum=0;for(unsigned n=0;n<4;++n)sum+=component(r[1].nodalWorldForces[n],a);near(sum,component(r[1].terminalWorldForce,a),2e-6,"distributed force conservation");}
    near(r[1].residualsAndForce.x,0,2e-6,"terminal force residual");near(r[1].residualsAndForce.y,0,2e-7,"terminal moment residual");
}

void cpuChecks() {
    for(unsigned kind:{0u,2u,3u}) {
        Fixture f(kind);const double q=double(.003f);auto oracle=f.oracle(q),suffix=f.oracle(q,true);
        std::vector<double> configuration(8,0);configuration[6]=1;configuration[7]=q;
        std::vector<ArticulatedBodyKinematics> bodyReference(bodies);
        require(computeArticulatedBodyKinematics(f.model,0,configuration,f.velocity,bodyReference).succeeded(),"fixture independent body FK");
        Geometry geometry(f,0,q);
        for(unsigned b=0;b<bodies;++b)for(unsigned a=0;a<3;++a) {
            const double manual=component(geometry.poses[b].position,a)+component(geometry.low[b],a);
            near(manual,bodyReference[b].centerOfMassPosition[a],1e-14,"manufactured/FP64 body position kind="+std::to_string(kind)+" body="+std::to_string(b)+" axis="+std::to_string(a));
            if(kind==0)std::cout<<"fixture_body body="<<b<<" axis="<<a<<" oracle="<<bodyReference[b].centerOfMassPosition[a]<<" high="<<component(geometry.poses[b].position,a)<<" low="<<component(geometry.low[b],a)<<'\n';
        }
        std::cout<<"source_fixture kind="<<kind<<" length="<<oracle.path.length<<" suffix_length="<<suffix.path.length<<'\n';
        require(oracle.path.appliedWrapCount==(kind?1u:0u),"fixture executes intended wrap branch");
        constexpr double h=1e-6;
        near((f.oracle(q+h).path.length-f.oracle(q-h).path.length)/(2*h),oracle.path.lengthJacobian[6],2e-8,"independent source route FD");
        near((f.oracle(q+h,true).path.length-f.oracle(q-h,true).path.length)/(2*h),suffix.path.lengthJacobian[6],2e-8,"independent source suffix FD");
        require(std::abs(oracle.path.lengthJacobian[6]-suffix.path.lengthJacobian[6])>.1,"suffix removes a nontrivial prefix");
        require(std::abs(oracle.actuatorForce)>.1,"nonzero physical source force control");
        for(unsigned d=0;d<6;++d)near(oracle.path.lengthJacobian[d],0,2e-14,"rigid-motion route invariance FP64");
    }
}

void routeChecks(GPU& gpu) {
    double maximumLengthError=0,maximumJacobianError=0,maximumForceError=0,maximumOriginError=0;bool legacyDetected=false;
    for(unsigned kind:{0u,2u,3u}) {
        Fixture f(kind);const double q=double(.003f);auto oracle=f.oracle(q),suffixOracle=f.oracle(q,true);
        const auto baseline=evaluateGPU(gpu,f,Geometry(f,0,q),true);
        for(double origin:{0.,64.,4096.}) {
            scenario="kind="+std::to_string(kind)+" origin="+std::to_string(origin);
            Geometry geometry(f,origin,q);auto out=evaluateGPU(gpu,f,geometry,true);
            std::cout<<"route_scenario kind="<<kind<<" origin="<<origin<<" native_length="<<out.result.pathForceAndActivationDerivative.x<<" oracle_length="<<oracle.path.length<<" status="<<out.result.status<<'\n';
            if(origin==0) {
                const auto old=evaluateGPU(gpu,f,geometry,false);
                std::cout<<"route_discriminator kind="<<kind<<" origin="<<origin<<" paired_status="<<out.result.status<<" legacy_status="<<old.result.status
                    <<" paired_length="<<out.result.pathForceAndActivationDerivative.x<<" legacy_length="<<old.result.pathForceAndActivationDerivative.x<<" fp64_length="<<oracle.path.length
                    <<" paired_Jv="<<out.result.pathForceAndActivationDerivative.y<<" legacy_Jv="<<old.result.pathForceAndActivationDerivative.y<<" fp64_Jv="<<oracle.path.velocity
                    <<" paired_force="<<out.result.pathForceAndActivationDerivative.z<<" legacy_force="<<old.result.pathForceAndActivationDerivative.z<<" fp64_force="<<oracle.actuatorForce
                    <<" paired_J6="<<out.force[6]/out.result.pathForceAndActivationDerivative.z<<" legacy_J6="<<old.force[6]/old.result.pathForceAndActivationDerivative.z<<" fp64_J6="<<oracle.path.lengthJacobian[6]
                    <<" paired_suffixJ6="<<out.suffix[6]<<" legacy_suffixJ6="<<old.suffix[6]<<" fp64_suffixJ6="<<suffixOracle.path.lengthJacobian[6]<<std::endl;
            }
            require(out.result.status==0,"paired full route status");require(out.result.appliedWrapCount==oracle.path.appliedWrapCount,"GPU/FP64 wrap branch");
            const double L=out.result.pathForceAndActivationDerivative.x,F=out.result.pathForceAndActivationDerivative.z;
            near(L,oracle.path.length,2e-7,"route length FP64");near(F,oracle.actuatorForce,2e-4,"source force FP64");near(out.result.pathForceAndActivationDerivative.y,oracle.path.velocity,2e-6,"source route Jv FP64");
            maximumLengthError=std::max(maximumLengthError,std::abs(L-oracle.path.length));maximumForceError=std::max(maximumForceError,std::abs(F-oracle.actuatorForce));
            near(L,baseline.result.pathForceAndActivationDerivative.x,3e-8,"world-origin route length invariance");near(F,baseline.result.pathForceAndActivationDerivative.z,2e-5,"world-origin source force invariance");maximumOriginError=std::max(maximumOriginError,std::abs(L-baseline.result.pathForceAndActivationDerivative.x));
            for(unsigned d=0;d<dofs;++d) {
                const double J=out.force[d]/F;near(J,oracle.path.lengthJacobian[d],1e-5,"source Jacobian FP64");near(out.suffix[d],suffixOracle.path.lengthJacobian[d],1e-5,"source suffix Jacobian FP64");near(J,baseline.force[d]/baseline.result.pathForceAndActivationDerivative.z,2e-6,"world-origin route Jacobian invariance");near(out.suffix[d],baseline.suffix[d],2e-6,"world-origin suffix invariance");maximumJacobianError=std::max(maximumJacobianError,std::abs(J-oracle.path.lengthJacobian[d]));
            }
            // Endpoint derivatives use independent perturbations of the FP64
            // source attachment coordinates, not the GPU tangent machinery.
            constexpr double h=1e-6;
            for(unsigned endpoint=0;endpoint<2;++endpoint)for(unsigned a=0;a<3;++a) {
                const unsigned site=endpoint?2:0;auto plus=f.sites,minus=f.sites;plus[site].localPoint[a]+=h;minus[site].localPoint[a]-=h;
                const double gradient=(f.oracle(q,false,&plus).path.length-f.oracle(q,false,&minus).path.length)/(2*h);
                near(component(out.result.endpointLengthGradients[endpoint],a),gradient,1e-5,"endpoint force gradient FP64");
            }
            terminalCheck(gpu,f,geometry,out);
            if(origin==4096) {
                auto old=evaluateGPU(gpu,f,geometry,false);
                legacyDetected|=old.result.status!=0 || std::abs(old.result.pathForceAndActivationDerivative.x-oracle.path.length)>2e-7;
            }
        }
        // This finite difference is deliberately away from wrap switches.
        constexpr double h=2e-4;auto plus=evaluateGPU(gpu,f,Geometry(f,4096,q+h),true),minus=evaluateGPU(gpu,f,Geometry(f,4096,q-h),true);
        near((plus.result.pathForceAndActivationDerivative.x-minus.result.pathForceAndActivationDerivative.x)/(2*h),oracle.path.lengthJacobian[6],4e-4,"production route length/J finite difference");
    }
    require(legacyDetected,"large-origin high-only negative control detects lost geometry");
    std::cout<<"route_max_length_error_m="<<maximumLengthError<<" route_max_J_error="<<maximumJacobianError<<" route_max_force_error_N="<<maximumForceError<<" origin_length_error_m="<<maximumOriginError<<'\n';
}

struct HoodFixture {
    MRNumiHumanExtensorHoodDispatchGPU dispatch{};
    std::vector<MRNumiHumanExtensorHoodRayGPU> rays;
    std::vector<MRNumiHumanExtensorHoodNodeGPU> nodes;
    std::vector<MRNumiHumanExtensorHoodElementGPU> elements;
    std::vector<MRNumiHumanExtensorHoodInputGPU> inputs;
    std::vector<MRMujocoMuscleSiteGPU> sites;
    std::vector<MRMujocoMuscleRouteNodeGPU> routes;
    MRMujocoMuscleGPU muscle{};
    MRMujocoMuscleResultGPU muscleResult{};
    std::array<double,4> x{0,double(-.01f),double(-.02f),double(.04f)};
    double solution=0,energy=0;
    const V axis{1,.5,-.25};
    const double axisLength=std::sqrt(1.3125);
    std::array<double,4> force{};
    HoodFixture() {
        auto& d=dispatch;d.abiVersion=MR_NUMI_HUMAN_EXTENSOR_HOOD_GPU_ABI_VERSION;
        d.environmentCount=1;d.rayCount=8;d.nodeCount=32;d.elementCount=24;d.inputCount=8;
        d.muscleCount=1;d.siteCount=2;d.routeNodeCount=2;d.dofCount=dofs;d.bodyPoseStride=bodies;
        d.pointJacobianStride=jacobianCount;d.generalizedForceStride=dofs;
        d.maximumIterations=32;d.maximumLineSearchSteps=16;d.wrapCount=1;
        d.solver=f4(1e-6,1e-9,1e-8,1e-4);d.foundation=f4(100);
        sites={{0,0,0,0,f4(double(.08f),double(.08f)*.5,double(.08f)*-.25)},
               {1,0,0,0,f4(x[3],x[3]*.5,x[3]*-.25)}};
        routes={{MR_MUJOCO_MUSCLE_ROUTE_SITE,0,MR_INVALID_INDEX,0},{MR_MUJOCO_MUSCLE_ROUTE_SITE,1,MR_INVALID_INDEX,0}};
        muscle.route={0,2,0,0};muscleResult.status=MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS;
        muscleResult.activeForceAndReserved=f4(-.03);
        for(unsigned r=0;r<8;++r) {
            rays.push_back({{r*4,4,r/4,2+r%4},{r*3,3,r,1}});
            for(unsigned n=0;n<4;++n)nodes.push_back({n<3?0u:1u,n<3?MR_NUMI_HUMAN_EXTENSOR_HOOD_NODE_FIXED:0u,n,n==3?1u:MR_INVALID_INDEX,f4(x[n],x[n]*.5,x[n]*-.25)});
            for(unsigned n=0;n<3;++n)elements.push_back({r*4+n,r*4+3,0,0,f4(.98*(x[3]-x[n])*axisLength,1e5,1e-6)});
            MRNumiHumanExtensorHoodInputGPU in{};in.nodeIndex=r*4+3;in.proximalBodyIndex=0;
            in.routeNodeOrdinal=0;in.targetRouteNodeOrdinal=1;in.proximalLocalPoint=sites[0].localPoint;inputs.push_back(in);
        }
        // All bars stay tensile and collinear; external load direction is
        // exactly along axis throughout. Thus the scalar equilibrium is affine and
        // this closed form is independent of both CPU and Metal Newton code.
        double stiffness=d.foundation.x,numerator=std::abs(muscleResult.activeForceAndReserved.x)/axisLength+double(d.foundation.x)*x[3];
        for(unsigned n=0;n<3;++n){const auto& m=elements[n].material;const double k=double(m.y)*double(m.z)/double(m.x);stiffness+=k;numerator+=k*(x[n]+double(m.x)/axisLength);}
        solution=numerator/stiffness;
        for(unsigned n=0;n<3;++n){const auto& m=elements[n].material;const double k=double(m.y)*double(m.z)/double(m.x),extension=axisLength*(solution-x[n])-double(m.x);force[n]=k*extension;energy+=.5*k*extension*extension;}
        force[3]=double(d.foundation.x)*(solution-x[3])*axisLength;
        require(solution>x[3] && solution<sites[0].localPoint.x,"hood analytic solution in source branch");
        for(unsigned n=0;n<3;++n)require(force[n]>0,"hood source bars remain tensile");
        near(force[0]+force[1]+force[2]+force[3],std::abs(muscleResult.activeForceAndReserved.x),1e-14,"independent hood force equilibrium");
    }
};

void hoodChecks(GPU& gpu) {
    HoodFixture h;Fixture f(0);double maxPositionError=0,maxForceError=0,maxOriginError=0;
    std::vector<mr_float4> baselineLocal;
    std::vector<MRNumiHumanExtensorHoodNodeResultGPU> baselineResult;
    for(double origin:{0.,64.,4096.}) {
        Geometry geometry(f,origin,0);
        // Independent manufactured root + zero-anchor prismatic child. The
        // free hood node belongs to that child, so dof6 receives a nonzero
        // replacement correction that can be checked in closed form.
        geometry.poses[1]=geometry.poses[0];geometry.low[1]=geometry.low[0];
        for(unsigned p=0;p<4;++p)for(unsigned a=0;a<3;++a)for(unsigned j=0;j<dofs;++j)
            geometry.jacobian[((4+p)*3+a)*dofs+j]=j==6?float(a==1):geometry.jacobian[(p*3+a)*dofs+j];
        auto poses=gpu.vector(geometry.poses),low=gpu.vector(geometry.low),jac=gpu.vector(geometry.jacobian),d=gpu.object(h.dispatch),rays=gpu.vector(h.rays),nodes=gpu.vector(h.nodes),elements=gpu.vector(h.elements),inputs=gpu.vector(h.inputs),muscles=gpu.object(h.muscle),sites=gpu.vector(h.sites),routes=gpu.vector(h.routes),muscleResult=gpu.object(h.muscleResult),wraps=gpu.vector(f.gpuWraps);
        auto nodeResult=gpu.buffer(nullptr,32*sizeof(MRNumiHumanExtensorHoodNodeResultGPU));
        auto rayResult=gpu.buffer(nullptr,8*sizeof(MRNumiHumanExtensorHoodRayResultGPU));
        auto relative=gpu.buffer(nullptr,32*sizeof(mr_float4));
        gpu.dispatch("mr_numi_human_solve_extensor_hood_compensated",{{0,d},{1,rays},{2,nodes},{3,elements},{4,inputs},{5,muscles},{6,sites},{7,routes},{8,muscleResult},{9,poses},{10,nodeResult},{11,rayResult},{12,wraps},{13,low},{14,relative}},8);
        const auto* rr=static_cast<const MRNumiHumanExtensorHoodRayResultGPU*>([rayResult contents]);
        const auto* nr=static_cast<const MRNumiHumanExtensorHoodNodeResultGPU*>([nodeResult contents]);
        const auto* local=static_cast<const mr_float4*>([relative contents]);
        for(unsigned r=0;r<8;++r) {
            require(rr[r].status==0,"paired hood solve status: "+std::to_string(rr[r].status));
            near(rr[r].energyAndCounts.x,h.energy,1e-8,"hood strain energy analytic FP64");
            near(rr[r].energyAndCounts.y,3,0,"hood active bars");
            near(rr[r].forceClosureAndMaximumResidual.x,0,1.1e-6,"hood force closure");
            for(unsigned n=0;n<4;++n) {
                unsigned i=r*4+n;const double expected=n==3?h.solution:h.x[n];
                for(unsigned a=0;a<3;++a){
                    near(component(local[i],a),expected*h.axis[a],2e-8,"private hood position analytic FP64");
                    near(component(nr[i].bodyForce,a),h.force[n]*h.axis[a]/h.axisLength,3e-6,"hood nodal force analytic FP64");
                    maxPositionError=std::max(maxPositionError,std::abs(component(local[i],a)-expected*h.axis[a]));
                    maxForceError=std::max(maxForceError,std::abs(component(nr[i].bodyForce,a)-h.force[n]*h.axis[a]/h.axisLength));
                }
                near(local[i].w,0,0,"hood private padding");
                near(nr[i].position.x,float(origin+double(local[i].x)),0,"legacy diagnostic projection only");
                if(origin!=0){near(local[i].x,baselineLocal[i].x,2e-8,"hood origin invariant local solve");near(nr[i].bodyForce.x,baselineResult[i].bodyForce.x,3e-6,"hood origin invariant body load");maxOriginError=std::max(maxOriginError,std::abs(double(local[i].x)-baselineLocal[i].x));}
            }
        }
        // The suffix connects the proximal root site to the prismatic child.
        // Its derivative is exactly -unit(axis).y, derived geometrically.
        std::vector<float> suffix(8*dofs,0);for(unsigned r=0;r<8;++r)suffix[r*dofs+6]=float(-h.axis[1]/h.axisLength);
        auto correction=gpu.buffer(nullptr,dofs*sizeof(float));
        gpu.dispatch("mr_numi_human_assemble_extensor_hood_correction_compensated",{{0,d},{1,rays},{2,nodes},{3,inputs},{4,muscles},{5,sites},{6,routes},{7,muscleResult},{8,poses},{9,jac},{10,nodeResult},{11,rayResult},{12,correction},{13,gpu.vector(suffix)},{14,low},{15,relative}},dofs);
        const auto* c=static_cast<const float*>([correction contents]);
        const double replacement=8*(h.force[3]-std::abs(h.muscleResult.activeForceAndReserved.x))*h.axis[1]/h.axisLength;
        require(std::abs(replacement)>1e-3,"hood replacement is nontrivial");
        for(unsigned j=0;j<dofs;++j)near(c[j],j==6?replacement:0,9e-6,"hood source correction analytic resultant");
        if(origin==0){baselineLocal.assign(local,local+32);baselineResult.assign(nr,nr+32);}
    }
    std::cout<<"hood_max_position_error_m="<<maxPositionError<<" hood_max_force_error_N="<<maxForceError<<" hood_origin_error_m="<<maxOriginError<<'\n';
}
} // namespace

int main(int argc,char** argv) {
    @autoreleasepool {try {
        std::cout<<std::setprecision(17);cpuChecks();HoodFixture hoodOracle;
        if(argc==2 && std::string(argv[1])=="--cpu") {std::cout<<"source_route_precision_cpu PASS checks="<<checks<<'\n';return 0;}
        require(argc==2,"usage: source-route-precision-check --cpu|metallib");GPU gpu(argv[1]);routeChecks(gpu);hoodChecks(gpu);
        std::cout<<"source_route_precision_metal PASS checks="<<checks<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<"source_route_precision FAIL "<<e.what()<<'\n';return 1;}}
}

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/shared.h"
#include "metalrobo/engine_types.h"

#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#define NUMI_MATTER_METALLIB ""
#endif

namespace {
using V3 = std::array<double, 3>;
using Q4 = std::array<double, 4>; // x,y,z,w
using M3 = std::array<double, 9>; // row-major

void need(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

V3 add(V3 a, V3 b) { for (int i=0;i<3;++i) a[i]+=b[i]; return a; }
V3 sub(V3 a, V3 b) { for (int i=0;i<3;++i) a[i]-=b[i]; return a; }
V3 mul(V3 a, double s) { for (double& x:a) x*=s; return a; }
double dot(V3 a,V3 b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
double norm(V3 a){return std::sqrt(dot(a,a));}
V3 cross(V3 a,V3 b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
V3 unit(V3 a){return mul(a,1.0/norm(a));}
M3 transpose(M3 a){return {a[0],a[3],a[6],a[1],a[4],a[7],a[2],a[5],a[8]};}
M3 multiply(M3 a,M3 b){M3 c{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)c[3*i+j]+=a[3*i+k]*b[3*k+j];return c;}
V3 multiply(M3 a,V3 b){return {a[0]*b[0]+a[1]*b[1]+a[2]*b[2],a[3]*b[0]+a[4]*b[1]+a[5]*b[2],a[6]*b[0]+a[7]*b[1]+a[8]*b[2]};}
M3 inverse(M3 a){
    const double c00=a[4]*a[8]-a[5]*a[7],c01=a[5]*a[6]-a[3]*a[8],c02=a[3]*a[7]-a[4]*a[6];
    const double det=a[0]*c00+a[1]*c01+a[2]*c02;
    need(std::isfinite(det)&&std::abs(det)>1e-20,"oracle inertia matrix is singular");
    const double scale=1.0/det;
    return {c00*scale,(a[2]*a[7]-a[1]*a[8])*scale,(a[1]*a[5]-a[2]*a[4])*scale,
        c01*scale,(a[0]*a[8]-a[2]*a[6])*scale,(a[2]*a[3]-a[0]*a[5])*scale,
        c02*scale,(a[1]*a[6]-a[0]*a[7])*scale,(a[0]*a[4]-a[1]*a[3])*scale};
}
M3 rotation(Q4 q){
    const double n=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    for(double& x:q)x/=n; const auto [x,y,z,w]=q;
    return {1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
            2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
            2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)};
}
Q4 qmul(Q4 a,Q4 b){return {a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]};}
Q4 integrate(Q4 source,V3 omega,double dt){
    const V3 phi=mul(omega,dt); const double angle=norm(phi),half=.5*angle;
    const double scale=angle>1e-6?std::sin(half)/angle:.5-angle*angle/48.0;
    Q4 delta{phi[0]*scale,phi[1]*scale,phi[2]*scale,std::cos(half)};
    Q4 out=qmul(delta,source); double n=0;for(double x:out)n+=x*x;n=std::sqrt(n);for(double& x:out)x/=n;return out;
}
nm_float4 f4(V3 v,double w=0){return {float(v[0]),float(v[1]),float(v[2]),float(w)};}
nm_float4 q4(Q4 q){return {float(q[0]),float(q[1]),float(q[2]),float(q[3])};}
V3 v3(nm_float4 v){return {v.x,v.y,v.z};}
Q4 q4(nm_float4 q){return {q.x,q.y,q.z,q.w};}
mr_float4 mf4(V3 v,double w=0){return {float(v[0]),float(v[1]),float(v[2]),float(w)};}
mr_float4 mq4(Q4 q){return {float(q[0]),float(q[1]),float(q[2]),float(q[3])};}
V3 v3(mr_float4 v){return {v.x,v.y,v.z};}
Q4 q4(mr_float4 q){return {q.x,q.y,q.z,q.w};}

template<class T> id<MTLBuffer> makeBuffer(id<MTLDevice> device,const std::vector<T>& values){
    const T zero{}; const void* data=values.empty()?static_cast<const void*>(&zero):values.data();
    return [device newBufferWithBytes:data length:std::max<std::size_t>(1,values.size())*sizeof(T) options:MTLResourceStorageModeShared];
}
template<class T> id<MTLBuffer> makeBuffer(id<MTLDevice> device,const T& value){
    return [device newBufferWithBytes:&value length:sizeof(T) options:MTLResourceStorageModeShared];
}

struct Device {
    id<MTLDevice> gpu=nil; id<MTLCommandQueue> queue=nil; id<MTLLibrary> library=nil;
    id<MTLComputePipelineState> validator=nil,evaluate=nil,apply=nil,projectFreeRigid=nil,applyFreeRigid=nil;
    id<MTLArgumentEncoder> primalContactEncoder=nil;
    Device(){
        gpu=MTLCreateSystemDefaultDevice(); need(gpu!=nil,"Metal device unavailable");
        queue=[gpu newCommandQueue]; need(queue!=nil,"Metal command queue unavailable");
        const std::string path=NUMI_MATTER_METALLIB; need(!path.empty(),"NUMI_MATTER_METALLIB is empty");
        NSError* error=nil; library=[gpu newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]] error:&error];
        need(library!=nil,"cannot load production metallib");
        validator=makePipeline(@"numi_matter_metal::nm_contact_validate_fem_rigid_armijo_path");
        evaluate=makePipeline(@"numi_matter_metal::nm_contact_evaluate");
        apply=makePipeline(@"numi_matter_metal::nm_fgmres_apply_primal_rigid_contacts");
        projectFreeRigid=makePipeline(@"numi_matter_metal::nm_project_primal_free_rigid_candidate");
        applyFreeRigid=makePipeline(@"numi_matter_metal::nm_fgmres_apply_free_rigid");
        id<MTLFunction> layout=[library newFunctionWithName:@"numi_matter_metal::nm_primal_contact_argument_layout"];
        primalContactEncoder=layout==nil?nil:[layout newArgumentEncoderWithBufferIndex:0];
        need(primalContactEncoder!=nil,"cannot construct production primal-contact encoder");
    }
    id<MTLComputePipelineState> makePipeline(NSString* name){
        NSError* error=nil; id<MTLFunction> f=[library newFunctionWithName:name];
        need(f!=nil,"production kernel missing: "+std::string([name UTF8String]));
        auto p=[gpu newComputePipelineStateWithFunction:f error:&error]; need(p!=nil,"cannot create production pipeline");return p;
    }
};

NMRigidStateGPU projectedRigid(const NMRigidProxyGPU& proxy,const MRBodyStateGPU& body,const M3& inverseInertiaWorld){
    NMRigidStateGPU r{}; const Q4 qb=q4(body.orientation),qp=q4(proxy.localOrientation),q=qmul(qb,qp);
    const V3 center=add(v3(body.position),multiply(rotation(qb),v3(proxy.localCenterAndRadius)));
    r.centerAndRadius=f4(center,proxy.localCenterAndRadius.w);r.extent=proxy.localExtent;r.orientation=q4(q);
    r.linearVelocityAndInverseMass=f4(add(v3(body.linearVelocityAndInverseMass),cross(v3(body.angularVelocity),sub(center,v3(body.position)))),body.linearVelocityAndInverseMass.w);
    r.angularVelocity=f4(v3(body.angularVelocity));r.bodyCenter=f4(v3(body.position));
    // Matter stores the body's world inverse inertia directly on the proxy
    // state; local proxy orientation does not rotate the body inertia tensor.
    const M3& inv=inverseInertiaWorld;
    r.inverseInertiaRow0={float(inv[0]),float(inv[1]),float(inv[2]),0};
    r.inverseInertiaRow1={float(inv[3]),float(inv[4]),float(inv[5]),0};
    r.inverseInertiaRow2={float(inv[6]),float(inv[7]),float(inv[8]),0};return r;
}

struct SweepCase { std::string name; V3 localCenter; V3 half; V3 acceptedPoint,basePoint,trialPoint; V3 baseOmega,trialOmega; bool expectedAlphaClear,expectedPhysicalClear; };

struct SweepOracle {
    double minimumGap = std::numeric_limits<double>::infinity();
    double fractionAtMinimum = 0.0;
    double midpointGap = std::numeric_limits<double>::quiet_NaN();
    double floor = 0.0;
};

double quantized(double value) { return double(float(value)); }
V3 quantized(V3 value) { for (double& x : value) x = quantized(x); return value; }

double boxGapFP64(const SweepCase& c, double fraction, bool physicalSweep) {
    constexpr double dt = double(float(0.1f));
    const Q4 identity{0.0,0.0,0.0,1.0};
    const V3 baseOmega = quantized(c.baseOmega);
    const V3 trialOmega = quantized(c.trialOmega);
    V3 angular{};
    double poseDt = dt;
    if (physicalSweep) {
        angular = trialOmega;
        poseDt = fraction * dt;
    } else {
        for (int i=0;i<3;++i)
            angular[i] = (1.0-fraction)*baseOmega[i] + fraction*trialOmega[i];
    }
    Q4 bodyQ = integrate(identity, angular, poseDt);
    // The kernel returns the exact projected endpoint records at f=1 and the
    // base record at alpha f=0. Use those same FP32 endpoint quaternions in the
    // oracle; interior samples use the documented source integration map.
    if (physicalSweep && fraction == 1.0) {
        bodyQ = integrate(identity, c.trialOmega, dt);
        for (double& x : bodyQ) x = quantized(x);
    } else if (!physicalSweep && fraction == 0.0) {
        bodyQ = integrate(identity, c.baseOmega, dt);
        for (double& x : bodyQ) x = quantized(x);
    } else if (!physicalSweep && fraction == 1.0) {
        bodyQ = integrate(identity, c.trialOmega, dt);
        for (double& x : bodyQ) x = quantized(x);
    }
    const M3 bodyR = rotation(bodyQ);
    const V3 bodyPosition{0.0,0.0,0.0}; // test source has zero COM velocity
    const V3 localCenter = quantized(c.localCenter);
    const V3 proxyCenter = add(bodyPosition, multiply(bodyR, localCenter));
    const V3 first = quantized(physicalSweep ? c.acceptedPoint : c.basePoint);
    const V3 last = quantized(c.trialPoint);
    const V3 point = add(first, mul(sub(last, first), fraction));
    const V3 local = multiply(transpose(bodyR), sub(point, proxyCenter));
    const V3 half = quantized(c.half);
    V3 outside{};
    for (int i=0;i<3;++i) outside[i] = std::max(std::abs(local[i])-half[i],0.0);
    const double outsideDistance = norm(outside);
    if (outsideDistance > 0.0) return outsideDistance;
    double penetration = std::numeric_limits<double>::infinity();
    for (int i=0;i<3;++i) penetration = std::min(penetration, half[i]-std::abs(local[i]));
    return -penetration;
}

SweepOracle sampleSweepFP64(const SweepCase& c, bool physicalSweep) {
    SweepOracle result;
    const V3 points{quantized(c.acceptedPoint[0]),quantized(c.basePoint[0]),quantized(c.trialPoint[0])};
    const V3 half=quantized(c.half);
    double coordinateScale=std::max(double(float(0.01f)),
        std::max({std::abs(points[0]),std::abs(points[1]),std::abs(points[2]),
                  std::abs(quantized(c.acceptedPoint[1])),std::abs(quantized(c.acceptedPoint[2])),
                  std::abs(quantized(c.basePoint[1])),std::abs(quantized(c.basePoint[2])),
                  std::abs(quantized(c.trialPoint[1])),std::abs(quantized(c.trialPoint[2])),
                  std::abs(half[0]),std::abs(half[1]),std::abs(half[2])}));
    result.floor=std::max(double(float(0.001f))*double(float(0.01f)),
        16.0*double(std::numeric_limits<float>::epsilon())*std::max(coordinateScale,1.0e-12));
    constexpr unsigned samples=65536u;
    for(unsigned i=0;i<=samples;++i){
        const double f=double(i)/samples;
        const double gap=boxGapFP64(c,f,physicalSweep);
        if(gap<result.minimumGap){result.minimumGap=gap;result.fractionAtMinimum=f;}
        if(i==samples/2)result.midpointGap=gap;
    }
    return result;
}

NMMatterStatusGPU runValidator(Device& d,const SweepCase& c){
    constexpr float dt=.1f; const Q4 q0{0,0,0,1};
    NMMatterDispatchGPU dispatch{};dispatch.environmentCount=1;dispatch.objectCount=1;dispatch.femNodeCount=1;dispatch.rigidProxyCount=1;dispatch.contactPairCount=1;dispatch.rigidGeneralizedCapacity=6;dispatch.maximumRateExponent=0;dispatch.numericalLimits={.01f,0,1e-8f,1e20f};
    NMBridgeDispatchGPU bridge{};bridge.environmentCount=1;bridge.rigidProxyCount=1;bridge.currentBodyCount=1;bridge.currentBodyStride=1;bridge.time={1/dt,dt,0,0};
    NMMixedSolverGPU solver{};solver.contactAcceptance.x=.001f;
    NMContinuumObjectGPU object{};object.representation=NM_REPRESENTATION_FEM;object.flags=NM_OBJECT_ACTIVE;
    NMRigidProxyGPU proxy{};proxy.shapeKind=NM_RIGID_BOX;proxy.bodyIndex=0;proxy.sceneBodyIndex=0;proxy.flags=NM_RIGID_DYNAMIC;proxy.generalizedFreeBodyIndex=0;proxy.localCenterAndRadius=f4(c.localCenter);proxy.localExtent=f4(c.half);proxy.localOrientation={0,0,0,1};
    NMContactPairGPU pair{};pair.continuumNode=0;pair.rigidProxy=0;pair.objectIndex=0;
    MRBodyStateGPU source{};source.position={0,0,0,0};source.orientation=mq4(q0);source.linearVelocityAndInverseMass={0,0,0,1};source.angularVelocity={0,0,0,0};source.flagsAndIndices[0]=MR_MOTION_DYNAMIC;
    const Q4 qb=integrate(q0,c.baseOmega,dt);
    const Q4 qt=integrate(q0,c.trialOmega,dt);
    MRBodyStateGPU baseBody=source; baseBody.orientation=mq4(qb);baseBody.angularVelocity=mf4(c.baseOmega);
    MRBodyStateGPU trialBody=source;trialBody.orientation=mq4(qt);trialBody.angularVelocity=mf4(c.trialOmega);
    const M3 invI{1,0,0,0,1,0,0,0,1};
    const auto baseRigid=projectedRigid(proxy,baseBody,invI),trialRigid=projectedRigid(proxy,trialBody,invI);
    auto acceptedNodes=std::vector<NMFEMNodeStateGPU>(1),baseNodes=acceptedNodes,trialNodes=acceptedNodes;
    acceptedNodes[0].positionAndMass=f4(c.acceptedPoint,1);baseNodes[0].positionAndMass=f4(c.basePoint,1);trialNodes[0].positionAndMass=f4(c.trialPoint,1);
    std::vector<NMMatterStatusGPU> statuses(1);std::vector<NMRigidStateGPU> bases{baseRigid},trials{trialRigid};
    const auto dispatchB=makeBuffer(d.gpu,dispatch),bridgeB=makeBuffer(d.gpu,bridge),solverB=makeBuffer(d.gpu,solver);
    const auto objectB=makeBuffer(d.gpu,std::vector<NMContinuumObjectGPU>{object});
    const auto baseB=makeBuffer(d.gpu,baseNodes),trialB=makeBuffer(d.gpu,trialNodes),acceptedB=makeBuffer(d.gpu,acceptedNodes);
    const auto proxyB=makeBuffer(d.gpu,std::vector<NMRigidProxyGPU>{proxy});
    const auto baseRigidB=makeBuffer(d.gpu,bases),trialRigidB=makeBuffer(d.gpu,trials);
    const auto pairB=makeBuffer(d.gpu,std::vector<NMContactPairGPU>{pair});
    const auto bodyB=makeBuffer(d.gpu,std::vector<MRBodyStateGPU>{source});const auto statusB=makeBuffer(d.gpu,statuses);
    id<MTLCommandBuffer> cmd=[d.queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
    need(cmd&&enc,"validator encoder failed");[enc setComputePipelineState:d.validator];
    [enc setBuffer:dispatchB offset:0 atIndex:0];[enc setBuffer:objectB offset:0 atIndex:1];[enc setBuffer:baseB offset:0 atIndex:2];[enc setBuffer:trialB offset:0 atIndex:3];[enc setBuffer:proxyB offset:0 atIndex:4];[enc setBuffer:baseRigidB offset:0 atIndex:5];[enc setBuffer:trialRigidB offset:0 atIndex:6];[enc setBuffer:pairB offset:0 atIndex:7];[enc setBuffer:solverB offset:0 atIndex:8];[enc setBuffer:statusB offset:0 atIndex:9];[enc setBuffer:acceptedB offset:0 atIndex:10];[enc setBuffer:bridgeB offset:0 atIndex:11];[enc setBuffer:bodyB offset:0 atIndex:12];
    [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[enc endEncoding];[cmd commit];[cmd waitUntilCompleted];
    need(cmd.status==MTLCommandBufferStatusCompleted,"validator command failed");return *static_cast<const NMMatterStatusGPU*>(statusB.contents);
}

struct ContactCase { std::string name; V3 boxLocalPoint; };
struct ContactEval { NMContactSampleGPU sample{}; NMMatterStatusGPU status{}; NMRigidStateGPU rigid{}; };

class ProductionContact {
    Device& d_; float dt_=0.01f,thickness_=0.012f;
    NMMatterDispatchGPU dispatch_{};NMMicrostepGPU micro_{};NMBridgeDispatchGPU bridge_{};
    NMContinuumObjectGPU object_{};NMMaterialGPU material_{};NMSchedulerStateGPU scheduler_{};NMAdaptiveStateGPU adaptive_{};
    NMRigidProxyGPU proxy_{};NMContactPairGPU pair_{};
    M3 sourceInvI_{};V3 localCenter_{.004,-.003,.002};Q4 localOrientation_{0.08,-0.03,0.05,0.994};V3 half_{.05,.04,.03};
    id<MTLBuffer> objectB_,materialB_,proxyB_,pairB_,schedulerB_,adaptiveB_,nodeIncidenceB_,nodeRangeB_,activePairB_,slotB_,activeCountB_,deformableB_,defIncidenceB_,defRangesB_,mpmToActiveB_,mpmCountsB_,defCountsB_,pointJ_; id<MTLBuffer> primalArgB_;
    void makePrimalArgs(id<MTLBuffer> sampleB,id<MTLBuffer> rigidB){
        const std::vector<NMIncidenceRangeGPU> ranges(1);const std::vector<uint> zero{0},invalid{NM_INVALID_INDEX};const std::vector<NMDeformableContactGPU> deformable(1);
        auto pairsB=pairB_;nodeIncidenceB_=makeBuffer(d_.gpu,std::vector<uint>(1));nodeRangeB_=makeBuffer(d_.gpu,ranges);defIncidenceB_=makeBuffer(d_.gpu,std::vector<uint>(1));defRangesB_=makeBuffer(d_.gpu,ranges);
        auto proxyB=proxyB_;deformableB_=makeBuffer(d_.gpu,deformable);auto activePairB=activePairB_;auto slotB=slotB_;auto activeCountB=activeCountB_;mpmToActiveB_=makeBuffer(d_.gpu,invalid);mpmCountsB_=makeBuffer(d_.gpu,zero);defCountsB_=makeBuffer(d_.gpu,zero);
        primalArgB_=[d_.gpu newBufferWithLength:d_.primalContactEncoder.encodedLength options:MTLResourceStorageModeShared];need(primalArgB_!=nil,"primal argument storage allocation failed");
        [d_.primalContactEncoder setArgumentBuffer:primalArgB_ offset:0];
        id<MTLBuffer> args[15]={pairsB,sampleB,nodeIncidenceB_,nodeRangeB_,activePairB,slotB,activeCountB,deformableB_,defIncidenceB_,defRangesB_,mpmToActiveB_,mpmCountsB_,proxyB,rigidB,defCountsB_};
        for(NSUInteger i=0;i<15;++i)[d_.primalContactEncoder setBuffer:args[i] offset:0 atIndex:i];
    }
    NMRigidStateGPU project(const MRBodyStateGPU& source,const std::array<double,6>& delta){
        const auto dispatchB=makeBuffer(d_.gpu,dispatch_),bridgeB=makeBuffer(d_.gpu,bridge_);
        const uint articulatedNv=0;
        const auto proxyB=makeBuffer(d_.gpu,std::vector<NMRigidProxyGPU>{proxy_});
        const auto bodyB=makeBuffer(d_.gpu,std::vector<MRBodyStateGPU>{source});
        std::vector<float> generalized(6);for(size_t i=0;i<6;++i)generalized[i]=float(delta[i]);
        const auto candidateB=makeBuffer(d_.gpu,generalized);
        std::vector<NMRigidStateGPU> states(1);const auto stateB=makeBuffer(d_.gpu,states);
        std::vector<NMMatterStatusGPU> statuses(1);const auto statusB=makeBuffer(d_.gpu,statuses);
        id<MTLCommandBuffer> cmd=[d_.queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
        need(cmd&&enc,"free-rigid projection encoder failed");[enc setComputePipelineState:d_.projectFreeRigid];
        [enc setBuffer:dispatchB offset:0 atIndex:0];[enc setBuffer:bridgeB offset:0 atIndex:1];[enc setBytes:&articulatedNv length:sizeof(articulatedNv) atIndex:2];
        [enc setBuffer:proxyB offset:0 atIndex:3];[enc setBuffer:bodyB offset:0 atIndex:4];[enc setBuffer:candidateB offset:0 atIndex:5];[enc setBuffer:stateB offset:0 atIndex:6];[enc setBuffer:statusB offset:0 atIndex:7];
        [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[enc endEncoding];[cmd commit];[cmd waitUntilCompleted];
        need(cmd.status==MTLCommandBufferStatusCompleted,"free-rigid projection command failed");
        const auto status=*static_cast<const NMMatterStatusGPU*>(statusB.contents);
        need(status.code==NM_STATUS_SUCCESS,"free-rigid projection status="+std::to_string(status.code));
        return *static_cast<const NMRigidStateGPU*>(stateB.contents);
    }
public:
    explicit ProductionContact(Device& d):d_(d){
        dispatch_.environmentCount=1;dispatch_.objectCount=1;dispatch_.materialCount=1;dispatch_.femNodeCount=1;dispatch_.rigidProxyCount=1;dispatch_.contactPairCount=1;dispatch_.rigidGeneralizedCapacity=6;dispatch_.reservedMixed1=1;dispatch_.gravityAndTimestep.w=dt_;dispatch_.numericalLimits={thickness_,0,1e-8f,1e20f};
        bridge_.environmentCount=1;bridge_.rigidProxyCount=1;bridge_.currentBodyCount=1;bridge_.currentBodyStride=1;bridge_.time={1/dt_,dt_,0,0};
        micro_.microtickCount=1;micro_.time={dt_,1/dt_,0,dt_};
        object_.representation=NM_REPRESENTATION_FEM;object_.materialIndex=0;object_.flags=NM_OBJECT_ACTIVE;object_.solver.y=0;
        material_.interfaceResponse={0,0,0,0};scheduler_.numerical.w=1;adaptive_.activeRepresentation=NM_REPRESENTATION_FEM;
        proxy_.shapeKind=NM_RIGID_BOX;proxy_.bodyIndex=0;proxy_.sceneBodyIndex=0;proxy_.flags=NM_RIGID_DYNAMIC;proxy_.generalizedFreeBodyIndex=0;proxy_.adaptiveObjectIndex=NM_INVALID_INDEX;
        proxy_.localCenterAndRadius=f4(localCenter_);proxy_.localExtent=f4(half_);proxy_.localOrientation=q4(localOrientation_);
        pair_.continuumNode=0;pair_.rigidProxy=0;pair_.objectIndex=0;pair_.materialInterface=0;
        sourceInvI_={2.0,0,0,0,1.3,0,0,0,.7};
        objectB_=makeBuffer(d_.gpu,std::vector<NMContinuumObjectGPU>{object_});materialB_=makeBuffer(d_.gpu,std::vector<NMMaterialGPU>{material_});proxyB_=makeBuffer(d_.gpu,std::vector<NMRigidProxyGPU>{proxy_});pairB_=makeBuffer(d_.gpu,std::vector<NMContactPairGPU>{pair_});schedulerB_=makeBuffer(d_.gpu,std::vector<NMSchedulerStateGPU>{scheduler_});adaptiveB_=makeBuffer(d_.gpu,std::vector<NMAdaptiveStateGPU>{adaptive_});
        activePairB_=makeBuffer(d_.gpu,std::vector<uint>{0});slotB_=makeBuffer(d_.gpu,std::vector<uint>{0});activeCountB_=makeBuffer(d_.gpu,std::vector<uint>{1});pointJ_=makeBuffer(d_.gpu,std::vector<float>{0});
    }
    struct State { MRBodyStateGPU body{};NMRigidStateGPU rigid{};NMFEMNodeStateGPU node{};V3 boxLocalPoint{}; };
    State baseline(const ContactCase& c){
        State s;
        const Q4 authoredQ{0.03,-0.04,0.02,0.9985};
        const double qn=std::sqrt(dot(V3{authoredQ[0],authoredQ[1],authoredQ[2]},V3{authoredQ[0],authoredQ[1],authoredQ[2]})+authoredQ[3]*authoredQ[3]);
        Q4 qsource=authoredQ;for(double& v:qsource)v/=qn;
        const V3 omega{.7,-.4,.5};
        const V3 center{.001,-.002,.0005};
        const V3 vlocal=c.boxLocalPoint;
        const V3 ext=v3(proxy_.localExtent);
        // A zero-unknown native projection establishes the exact FP32 source
        // pose consumed by both the contact kernel and the independent oracle.
        const Q4 qcandidate=integrate(q4(mq4(qsource)),v3(mf4(omega)),dt_);
        const Q4 proxyQ=qmul(qcandidate,q4(proxy_.localOrientation));
        const M3 Rproxy=rotation(q4(mq4(proxyQ)));
        const V3 outsideDirection{(std::abs(vlocal[0])>ext[0])?std::copysign(1.0,vlocal[0]):0.0,(std::abs(vlocal[1])>ext[1])?std::copysign(1.0,vlocal[1]):0.0,(std::abs(vlocal[2])>ext[2])?std::copysign(1.0,vlocal[2]):0.0};
        const V3 normalApprox=unit(multiply(Rproxy,outsideDirection));
        const M3 Rbody=rotation(q4(mq4(qcandidate)));
        const V3 proxyCenterOffset=multiply(Rbody,v3(proxy_.localCenterAndRadius));
        const V3 pointOffset=add(proxyCenterOffset,multiply(Rproxy,vlocal));
        const V3 bodyLinearVelocity=add(mul(normalApprox,.5),mul(cross(v3(mf4(omega)),pointOffset),-1.0));
        const V3 rigidPointVelocity=add(bodyLinearVelocity,cross(v3(mf4(omega)),pointOffset));
        const V3 relativePointVelocity=mul(rigidPointVelocity,-1.0); // node velocity is zero below
        need(dot(relativePointVelocity,normalApprox)<-0.49,
             "fixture must approach the BOX so production predictor admits a sample");
        s.body.position=mf4(center);s.body.orientation=mq4(qsource);s.body.linearVelocityAndInverseMass=mf4(bodyLinearVelocity,.4);s.body.angularVelocity=mf4(omega);s.body.flagsAndIndices[0]=MR_MOTION_DYNAMIC;
        // Accepted body inverse inertia is authored in world coordinates at qsource.
        const M3 Rsource=rotation(q4(s.body.orientation)),Iworld0=multiply(multiply(Rsource,sourceInvI_),transpose(Rsource));
        s.body.inverseInertiaWorldRow0={float(Iworld0[0]),float(Iworld0[1]),float(Iworld0[2]),0};s.body.inverseInertiaWorldRow1={float(Iworld0[3]),float(Iworld0[4]),float(Iworld0[5]),0};s.body.inverseInertiaWorldRow2={float(Iworld0[6]),float(Iworld0[7]),float(Iworld0[8]),0};
        s.rigid=project(s.body,{});
        const V3 nodePos=add(v3(s.rigid.centerAndRadius),multiply(rotation(q4(s.rigid.orientation)),vlocal));
        s.node.positionAndMass=f4(nodePos,1);s.node.velocityAndInverseMass={0,0,0,1};s.node.deltaVelocity={0,0,0,0};s.boxLocalPoint=vlocal;
        return s;
    }
    ContactEval evaluate(const State& s,V3 velocityDelta={},V3 angularDelta={}){
        std::array<double,6> delta{};for(int i=0;i<3;++i){delta[i]=velocityDelta[i];delta[i+3]=angularDelta[i];}
        const NMRigidStateGPU rigid=project(s.body,delta);
        std::vector<NMFEMNodeStateGPU> node{s.node};std::vector<NMRigidStateGPU> rigs{rigid};std::vector<NMMatterStatusGPU> statuses(1);std::vector<nm_float4> history(1);std::vector<NMGridNodeStateGPU> grid(1);std::vector<uint> generation(1),candidateIds{0},candidateCounts{1};std::vector<NMPunctureChannelGPU> punctures(1);
        const uint cap=1,compact=0;const auto dispatchB=makeBuffer(d_.gpu,dispatch_),microB=makeBuffer(d_.gpu,micro_),bridgeB=makeBuffer(d_.gpu,bridge_);
        auto nodeB=makeBuffer(d_.gpu,node),rigidB=makeBuffer(d_.gpu,rigs),statusB=makeBuffer(d_.gpu,statuses),sampleB=makeBuffer(d_.gpu,std::vector<NMContactSampleGPU>(1));
        const auto gridB=makeBuffer(d_.gpu,grid),genB=makeBuffer(d_.gpu,generation),historyB=makeBuffer(d_.gpu,history),bodyB=makeBuffer(d_.gpu,std::vector<MRBodyStateGPU>{s.body}),punctureB=makeBuffer(d_.gpu,punctures),candidateIdsB=makeBuffer(d_.gpu,candidateIds),candidateCountsB=makeBuffer(d_.gpu,candidateCounts),capB=makeBuffer(d_.gpu,cap),compactB=makeBuffer(d_.gpu,compact);
        id<MTLCommandBuffer> cmd=[d_.queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];need(cmd&&enc,"contact evaluator encoder failed");[enc setComputePipelineState:d_.evaluate];
        id<MTLBuffer> all[]={dispatchB,microB,objectB_,materialB_,gridB,nodeB,proxyB_,rigidB,pairB_,schedulerB_,adaptiveB_,sampleB,genB,statusB,historyB,bridgeB,bodyB,punctureB,candidateIdsB,candidateCountsB,capB,compactB};for(NSUInteger i=0;i<22;++i)[enc setBuffer:all[i] offset:0 atIndex:i];
        [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[enc endEncoding];[cmd commit];[cmd waitUntilCompleted];need(cmd.status==MTLCommandBufferStatusCompleted,"contact evaluator command failed");
        return {*static_cast<const NMContactSampleGPU*>(sampleB.contents),*static_cast<const NMMatterStatusGPU*>(statusB.contents),rigid};
    }
    std::array<double,6> action(const NMContactSampleGPU& sample,const State& state,const std::array<double,6>& direction){
        const auto dispatchB=makeBuffer(d_.gpu,dispatch_);const uint articulated=0;const auto sampleB=makeBuffer(d_.gpu,std::vector<NMContactSampleGPU>{sample}),rigidB=makeBuffer(d_.gpu,std::vector<NMRigidStateGPU>{state.rigid});
        std::vector<nm_float4> dvec(8),work(8);for(int i=0;i<6;++i)dvec[2+i]={float(direction[i]),0,0,0};const auto directionB=makeBuffer(d_.gpu,dvec),workB=makeBuffer(d_.gpu,work),statesB=makeBuffer(d_.gpu,std::vector<NMFGMRESStateGPU>(1));
        makePrimalArgs(sampleB,rigidB);
        id<MTLCommandBuffer> cmd=[d_.queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];need(cmd&&enc,"operator encoder failed");[enc setComputePipelineState:d_.apply];[enc setBuffer:dispatchB offset:0 atIndex:0];[enc setBytes:&articulated length:sizeof(articulated) atIndex:1];[enc setBuffer:directionB offset:0 atIndex:2];[enc setBuffer:workB offset:0 atIndex:3];[enc setBuffer:primalArgB_ offset:0 atIndex:4];[enc setBuffer:statesB offset:0 atIndex:5];[enc setBuffer:pointJ_ offset:0 atIndex:6];
        for(id<MTLBuffer> resource in @[pairB_,sampleB,nodeIncidenceB_,nodeRangeB_,activePairB_,slotB_,activeCountB_,deformableB_,defIncidenceB_,defRangesB_,mpmToActiveB_,mpmCountsB_,proxyB_,rigidB,defCountsB_])[enc useResource:resource usage:MTLResourceUsageRead];
        [enc dispatchThreads:MTLSizeMake(6,1,1) threadsPerThreadgroup:MTLSizeMake(6,1,1)];[enc endEncoding];[cmd commit];[cmd waitUntilCompleted];need(cmd.status==MTLCommandBufferStatusCompleted,"FGMRES contact action command failed");const auto* out=static_cast<const nm_float4*>(workB.contents);std::array<double,6> result{};for(int i=0;i<6;++i)result[i]=out[2+i].x;return result;
    }
    V3 fixedPoint(const State& s)const{return v3(s.node.positionAndMass);}
    double timestep()const{return dt_;}double thickness()const{return thickness_;}
    const NMRigidProxyGPU& proxy()const{return proxy_;}M3 bodyInvI()const{return sourceInvI_;}
    const NMMatterDispatchGPU& dispatch()const{return dispatch_;}
    const NMRigidProxyGPU& proxySource()const{return proxy_;}
    double nodeInverseMass(const State& state)const{return state.node.velocityAndInverseMass.w;}
    double characteristicLength()const{return norm(v3(proxy_.localCenterAndRadius))+norm(v3(proxy_.localExtent));}
    NMRigidStateGPU candidateRigid(const State& state,const std::array<double,6>& candidate){return project(state.body,candidate);}
    std::array<double,6> freeRigidAction(const NMContactSampleGPU& sample,const NMRigidStateGPU& rigid,const std::array<double,6>& direction,const std::array<double,6>& candidate){
        const uint articulatedNv=0;const auto dispatchB=makeBuffer(d_.gpu,dispatch_);
        std::vector<nm_float4> dvec(8),work(8);for(int i=0;i<6;++i)dvec[2+i]={float(direction[i]),0,0,0};
        const auto directionB=makeBuffer(d_.gpu,dvec),workB=makeBuffer(d_.gpu,work),statesB=makeBuffer(d_.gpu,std::vector<NMFGMRESStateGPU>(1));
        std::vector<float> candidateFloat(6);for(int i=0;i<6;++i)candidateFloat[i]=float(candidate[i]);const auto candidateB=makeBuffer(d_.gpu,candidateFloat);
        const auto rigidB=makeBuffer(d_.gpu,std::vector<NMRigidStateGPU>{rigid}),sampleB=makeBuffer(d_.gpu,std::vector<NMContactSampleGPU>{sample});makePrimalArgs(sampleB,rigidB);
        id<MTLCommandBuffer> cmd=[d_.queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];need(cmd&&enc,"free-rigid inertia operator encoder failed");[enc setComputePipelineState:d_.applyFreeRigid];
        [enc setBuffer:dispatchB offset:0 atIndex:0];[enc setBytes:&articulatedNv length:sizeof(articulatedNv) atIndex:1];[enc setBuffer:directionB offset:0 atIndex:2];[enc setBuffer:workB offset:0 atIndex:3];[enc setBuffer:primalArgB_ offset:0 atIndex:4];[enc setBuffer:statesB offset:0 atIndex:5];[enc setBuffer:candidateB offset:0 atIndex:6];
        for(id<MTLBuffer> resource in @[proxyB_,rigidB])[enc useResource:resource usage:MTLResourceUsageRead];
        [enc dispatchThreads:MTLSizeMake(6,1,1) threadsPerThreadgroup:MTLSizeMake(6,1,1)];[enc endEncoding];[cmd commit];[cmd waitUntilCompleted];need(cmd.status==MTLCommandBufferStatusCompleted,"free-rigid inertia action command failed");
        const auto* out=static_cast<const nm_float4*>(workB.contents);std::array<double,6> result{};for(int i=0;i<6;++i)result[i]=out[2+i].x;return result;
    }
};
void printContactSetup(const char* stage, const ContactCase& c,
                       const ProductionContact::State& state,
                       const ContactEval& eval) {
    std::cout << std::setprecision(10)
              << "contact_setup stage=" << stage << " case=" << c.name
              << " status=" << eval.status.code
              << " object=" << eval.status.objectIndex
              << " failing_index=" << eval.status.failingIndex
              << " status_diag=" << eval.status.diagnostics.x << ","
              << eval.status.diagnostics.y << "," << eval.status.diagnostics.z
              << "," << eval.status.diagnostics.w
              << " sample_identity=" << eval.sample.identity.x << ","
              << eval.sample.identity.y << "," << eval.sample.identity.z
              << "," << eval.sample.identity.w
              << " sample_point_gap=" << eval.sample.pointAndSeparation.x << ","
              << eval.sample.pointAndSeparation.y << ","
              << eval.sample.pointAndSeparation.z << ","
              << eval.sample.pointAndSeparation.w
              << " sample_normal_velocity=" << eval.sample.normalAndVelocity.x
              << "," << eval.sample.normalAndVelocity.y << ","
              << eval.sample.normalAndVelocity.z << ","
              << eval.sample.normalAndVelocity.w
              << " sample_barrier=" << eval.sample.barrier.x << ","
              << eval.sample.barrier.y << "," << eval.sample.barrier.z
              << "," << eval.sample.barrier.w
              << " sample_hessian_markers=" << eval.sample.barrierHessianRow0.w
              << "," << eval.sample.barrierHessianRow1.w
              << " node=" << state.node.positionAndMass.x << ","
              << state.node.positionAndMass.y << ","
              << state.node.positionAndMass.z << ","
              << state.node.positionAndMass.w
              << " node_inverse_mass=" << state.node.velocityAndInverseMass.w
              << " rigid_center=" << eval.rigid.centerAndRadius.x << ","
              << eval.rigid.centerAndRadius.y << ","
              << eval.rigid.centerAndRadius.z
              << " rigid_extent=" << eval.rigid.extent.x << ","
              << eval.rigid.extent.y << "," << eval.rigid.extent.z
              << " rigid_orientation=" << eval.rigid.orientation.x << ","
              << eval.rigid.orientation.y << "," << eval.rigid.orientation.z
              << "," << eval.rigid.orientation.w
              << " rigid_inverse_mass="
              << eval.rigid.linearVelocityAndInverseMass.w
              << " authored_box_local_point=" << c.boxLocalPoint[0] << ","
              << c.boxLocalPoint[1] << "," << c.boxLocalPoint[2] << "\n";
}

struct OracleSample {
    V3 witness{},normal{},impulse{},torque{};
    double separation=0,normalVelocity=0,lambda=0,inverseResponse=0,stiffness=0;
    double forceN=0,firstSquared=0,secondSquared=0,radialCurvatureRaw=0,radialCurvatureProjected=0;
    // 0=inertial stiffness branch, 1=impact stiffness branch, -1=at the
    // nonsmooth max transition or outside the active barrier.
    int stiffnessBranch=-1;
    uint32_t feature=0;
};
OracleSample fp64Sample(const ProductionContact& scene,const ProductionContact::State& s,const NMRigidStateGPU& rigid){
    const double dt=scene.timestep(),th=scene.thickness();const V3 point=scene.fixedPoint(s);
    const V3 center=v3(rigid.centerAndRadius),bodyCenter=v3(rigid.bodyCenter);const M3 R=rotation(q4(rigid.orientation)),Rt=transpose(R);
    const V3 local=multiply(Rt,sub(point,center));const V3 half=v3(rigid.extent);V3 closest{};uint32_t feature=0;
    for(int i=0;i<3;++i){closest[i]=std::clamp(local[i],-half[i],half[i]);if(std::abs(local[i])>half[i])feature|=1u<<i;}
    const V3 outside=sub(local,closest);const double separation=norm(outside);need(separation>1e-10,"oracle requires a separated BOX point");
    const V3 normal=multiply(R,mul(outside,1.0/separation));const V3 witness=add(center,multiply(R,closest));
    const V3 lever=sub(witness,bodyCenter);const V3 omega=v3(rigid.angularVelocity);
    const V3 rigidVelocity=add(v3(rigid.linearVelocityAndInverseMass),cross(omega,sub(witness,center)));
    const double normalVelocity=dot(mul(rigidVelocity,-1.0),normal);
    const M3 invWorld{rigid.inverseInertiaRow0.x,rigid.inverseInertiaRow0.y,rigid.inverseInertiaRow0.z,rigid.inverseInertiaRow1.x,rigid.inverseInertiaRow1.y,rigid.inverseInertiaRow1.z,rigid.inverseInertiaRow2.x,rigid.inverseInertiaRow2.y,rigid.inverseInertiaRow2.z};
    const V3 angularArm=cross(lever,normal);const double inverseResponse=scene.nodeInverseMass(s)+rigid.linearVelocityAndInverseMass.w+dot(angularArm,multiply(invWorld,angularArm));
    const double distance=std::max(separation,1e-4*th),s2=distance*distance,t2=th*th,offset=s2-t2,logarithm=std::log(s2/t2);
    const double remaining=std::max(th-distance,1e-3*th),closing=std::max(-normalVelocity,0.0);
    const double inertial=10.0/std::max(inverseResponse*dt*dt*t2,1e-20),impact=closing/std::max(inverseResponse*dt*remaining*t2,1e-20);
    const double stiffness=std::max(inertial,impact);
    const double branchGap=std::abs(inertial-impact)/
        std::max(std::max(inertial,impact),1e-30);
    const int stiffnessBranch=branchGap>1e-6
        ? (impact>inertial?1:0) : -1;
    const double first=-stiffness*(2*offset*logarithm+offset*offset/s2);
    const double second=-stiffness*(2*logarithm+4*offset/s2-offset*offset/(s2*s2));
    const double radialRaw=4*s2*second+2*first;
    const double radialProjected=separation<1e-4*th?0.0:std::max(radialRaw,0.0);
    const double force=-2*distance*first;const double lambda=std::max(force,0.0)*dt;
    const V3 impulse=mul(normal,-lambda),torque=cross(lever,impulse);
    return {witness,normal,impulse,torque,separation,normalVelocity,lambda,inverseResponse,stiffness,
        force,first,second,radialRaw,radialProjected,stiffnessBranch,feature};
}

std::array<double,6> contactResidual(const OracleSample& sample){return {sample.impulse[0],sample.impulse[1],sample.impulse[2],sample.torque[0],sample.torque[1],sample.torque[2]};}
double vectorNorm(const std::array<double,6>& a){double n=0;for(double x:a)n+=x*x;return std::sqrt(n);}
double maxAbs(const std::array<double,6>& a){double m=0;for(double x:a)m=std::max(m,std::abs(x));return m;}
double maxDifference(const std::array<double,6>& a,const std::array<double,6>& b){double m=0;for(size_t i=0;i<a.size();++i)m=std::max(m,std::abs(a[i]-b[i]));return m;}
std::array<double,6> sampleResidual(const NMContactSampleGPU& sample){return {sample.impulseAndNormal.x,sample.impulseAndNormal.y,sample.impulseAndNormal.z,sample.angularImpulseAndTangent.x,sample.angularImpulseAndTangent.y,sample.angularImpulseAndTangent.z};}
std::array<double,6> energyScaled(std::array<double,6> residual,double characteristicLength){for(size_t i=3;i<6;++i)residual[i]/=characteristicLength;return residual;}

std::array<double,3> fp64InertialResidual(const NMRigidStateGPU& rigid,const std::array<double,6>& candidate){
    const M3 invI{rigid.inverseInertiaRow0.x,rigid.inverseInertiaRow0.y,rigid.inverseInertiaRow0.z,rigid.inverseInertiaRow1.x,rigid.inverseInertiaRow1.y,rigid.inverseInertiaRow1.z,rigid.inverseInertiaRow2.x,rigid.inverseInertiaRow2.y,rigid.inverseInertiaRow2.z};
    // Production assembles R = external impulse - M*candidate. FGMRES solves
    // A = -dR, so the reference residual must carry the same negative sign.
    const V3 omega{float(candidate[3]),float(candidate[4]),float(candidate[5])};const V3 value=mul(multiply(inverse(invI),omega),-1.0);return {value[0],value[1],value[2]};
}


std::uint32_t fp32Bits(float value){return std::bit_cast<std::uint32_t>(value);}
std::string fp32WordList(const std::vector<float>& values){
    std::ostringstream out;out<<std::hex<<std::setfill('0');
    for(size_t i=0;i<values.size();++i){if(i)out<<',';out<<std::setw(8)<<fp32Bits(values[i]);}
    return out.str();
}
M3 sourceWorldInverseInertia(const MRBodyStateGPU& source){
    return {source.inverseInertiaWorldRow0.x,source.inverseInertiaWorldRow0.y,source.inverseInertiaWorldRow0.z,
            source.inverseInertiaWorldRow1.x,source.inverseInertiaWorldRow1.y,source.inverseInertiaWorldRow1.z,
            source.inverseInertiaWorldRow2.x,source.inverseInertiaWorldRow2.y,source.inverseInertiaWorldRow2.z};
}
V3 sourceOmega(const MRBodyStateGPU& source){return {source.angularVelocity.x,source.angularVelocity.y,source.angularVelocity.z};}
V3 leftJacobianApply(const V3& phi,const V3& direction){
    const double theta2=dot(phi,phi);double a,b;
    if(theta2<1e-12){
        a=.5-theta2/24.0+theta2*theta2/720.0;
        b=1.0/6.0-theta2/120.0+theta2*theta2/5040.0;
    }else{
        const double theta=std::sqrt(theta2);
        a=(1.0-std::cos(theta))/theta2;
        b=(theta-std::sin(theta))/(theta2*theta);
    }
    const V3 first=cross(phi,direction);
    return add(add(direction,mul(first,a)),mul(cross(phi,first),b));
}
M3 fp64InertiaAtCandidate(const MRBodyStateGPU& source,const V3& candidate,double dt){
    const V3 total=add(sourceOmega(source),candidate);
    const Q4 deltaQ=integrate(Q4{0,0,0,1},total,dt);
    const M3 deltaR=rotation(deltaQ),sourceInv=sourceWorldInverseInertia(source);
    const M3 candidateInv=multiply(multiply(deltaR,sourceInv),transpose(deltaR));
    return inverse(candidateInv);
}
V3 fp64FullInertiaResidual(const MRBodyStateGPU& source,const V3& candidate,double dt){
    return mul(multiply(fp64InertiaAtCandidate(source,candidate,dt),candidate),-1.0);
}
V3 fp64FullInertiaAction(const MRBodyStateGPU& source,const V3& candidate,const V3& direction,double dt){
    const M3 inertia=fp64InertiaAtCandidate(source,candidate,dt);
    const V3 phi=mul(add(sourceOmega(source),candidate),dt);
    const V3 dtheta=mul(leftJacobianApply(phi,direction),dt);
    return add(multiply(inertia,direction),sub(cross(dtheta,multiply(inertia,candidate)),multiply(inertia,cross(dtheta,candidate))));
}
double norm3(V3 value){return norm(value);}
double maxAbs3(V3 value){return std::max({std::abs(value[0]),std::abs(value[1]),std::abs(value[2])});}
std::string inertiaInputBinding(const MRBodyStateGPU& source,const V3& candidate,double dt){
    std::vector<float> words={source.orientation.x,source.orientation.y,source.orientation.z,source.orientation.w,
        source.angularVelocity.x,source.angularVelocity.y,source.angularVelocity.z,float(dt),
        source.inverseInertiaWorldRow0.x,source.inverseInertiaWorldRow0.y,source.inverseInertiaWorldRow0.z,
        source.inverseInertiaWorldRow1.x,source.inverseInertiaWorldRow1.y,source.inverseInertiaWorldRow1.z,
        source.inverseInertiaWorldRow2.x,source.inverseInertiaWorldRow2.y,source.inverseInertiaWorldRow2.z,
        float(candidate[0]),float(candidate[1]),float(candidate[2])};
    return fp32WordList(words);
}

// Continuous FP64 candidate projection for the BOX contact residual oracle.
// Every accepted/source quantity is read from the production FP32 structs
// and promoted exactly; only the directional candidate perturbation is kept
// in double. This deliberately avoids the native FP32 projection used by the
// legacy candidate-FD diagnostic below.
struct FP64BoxCandidate {
    V3 bodyPosition{}, linearVelocity{}, angularVelocity{}, center{}, half{};
    Q4 bodyOrientation{}, proxyOrientation{};
    M3 inverseInertiaWorld{};
    double bodyInverseMass=0.0;
};

Q4 normalizeQ(Q4 q){
    double n2=0.0;for(double x:q)n2+=x*x;
    if(!(n2>1.0e-30)||!std::isfinite(n2))return {NAN,NAN,NAN,NAN};
    const double inverseNorm=1.0/std::sqrt(n2);for(double& x:q)x*=inverseNorm;return q;
}
Q4 conjugateQ(Q4 q){return {-q[0],-q[1],-q[2],q[3]};}
V3 rotateQ(Q4 q,V3 value){return multiply(rotation(q),value);}

FP64BoxCandidate fp64ProjectBoxCandidate(
    const ProductionContact& scene,
    const ProductionContact::State& state,
    const std::array<double,6>& candidate
){
    const MRBodyStateGPU& source=state.body;
    const NMRigidProxyGPU& proxy=scene.proxySource();
    FP64BoxCandidate out;
    const V3 sourcePosition=v3(source.position);
    const Q4 sourceOrientation=normalizeQ(q4(source.orientation));
    const V3 sourceLinear=v3(source.linearVelocityAndInverseMass);
    const V3 sourceAngular=v3(source.angularVelocity);
    for(int i=0;i<3;++i){out.linearVelocity[i]=sourceLinear[i]+candidate[i];out.angularVelocity[i]=sourceAngular[i]+candidate[i+3];}
    out.bodyPosition=add(sourcePosition,mul(out.linearVelocity,scene.timestep()));
    out.bodyOrientation=integrate(sourceOrientation,out.angularVelocity,scene.timestep());
    const Q4 deltaOrientation=normalizeQ(qmul(out.bodyOrientation,conjugateQ(sourceOrientation)));
    const M3 deltaRotation=rotation(deltaOrientation);
    const M3 sourceInverse=sourceWorldInverseInertia(source);
    out.inverseInertiaWorld=multiply(multiply(deltaRotation,sourceInverse),transpose(deltaRotation));
    out.proxyOrientation=qmul(out.bodyOrientation,normalizeQ(q4(proxy.localOrientation)));
    out.center=add(out.bodyPosition,rotateQ(out.bodyOrientation,v3(proxy.localCenterAndRadius)));
    out.half=v3(proxy.localExtent);
    out.bodyInverseMass=source.linearVelocityAndInverseMass.w;
    return out;
}

struct FP64ContactProbe {
    OracleSample sample{};
    bool supportedSmoothBranch=false;
    bool featureMargin=false;
    bool barrierMargin=false;
    bool strictInertialBranch=false;
};

FP64ContactProbe fp64ContinuousBoxContact(
    const ProductionContact& scene,
    const ProductionContact::State& state,
    const std::array<double,6>& candidate
){
    const FP64BoxCandidate rigid=fp64ProjectBoxCandidate(scene,state,candidate);
    const V3 point=v3(state.node.positionAndMass);
    const V3 nodeVelocity=v3(state.node.velocityAndInverseMass);
    const M3 proxyRotation=rotation(rigid.proxyOrientation),proxyRotationT=transpose(proxyRotation);
    const V3 local=multiply(proxyRotationT,sub(point,rigid.center));
    V3 closest{},signedOutside{},outside{};uint32_t feature=0u;bool featureMargin=true;
    const double featureScale=std::max({maxAbs3(point),maxAbs3(rigid.bodyPosition),maxAbs3(rigid.half),scene.thickness(),1.0e-12});
    const double featureTolerance=64.0*double(std::numeric_limits<float>::epsilon())*featureScale;
    for(int i=0;i<3;++i){
        signedOutside[i]=std::abs(local[i])-rigid.half[i];
        closest[i]=std::clamp(local[i],-rigid.half[i],rigid.half[i]);
        // Preserve the signed local offset for the outward normal. The
        // unsigned distance vector has the same norm but mirrors negative
        // face/edge normals and corrupts the independent residual oracle.
        outside[i]=local[i]-closest[i];
        if(signedOutside[i]>0.0)feature|=1u<<i;
        if(std::abs(signedOutside[i])<=featureTolerance)featureMargin=false;
    }
    const double separation=norm(outside);
    if(!(separation>1.0e-12)||!std::isfinite(separation))return {};
    const V3 normal=rotateQ(rigid.proxyOrientation,mul(outside,1.0/separation));
    const V3 witness=add(rigid.center,rotateQ(rigid.proxyOrientation,closest));
    const V3 lever=sub(witness,rigid.bodyPosition);
    const V3 proxyVelocity=add(rigid.linearVelocity,cross(rigid.angularVelocity,sub(rigid.center,rigid.bodyPosition)));
    const V3 rigidPointVelocity=add(proxyVelocity,cross(rigid.angularVelocity,sub(witness,rigid.center)));
    const double normalVelocity=dot(sub(nodeVelocity,rigidPointVelocity),normal);
    const V3 angularArm=cross(lever,normal);
    const double inverseResponse=scene.nodeInverseMass(state)+rigid.bodyInverseMass+
        dot(angularArm,multiply(rigid.inverseInertiaWorld,angularArm));
    const double dt=scene.timestep(),th=scene.thickness();
    const double distance=std::max(separation,1.0e-4*th),s2=distance*distance,t2=th*th,offset=s2-t2,logarithm=std::log(s2/t2);
    const double remaining=std::max(th-distance,1.0e-3*th),closing=std::max(-normalVelocity,0.0);
    const double inertialDenominator=inverseResponse*dt*dt*t2;
    const double impactDenominator=inverseResponse*dt*remaining*t2;
    const double inertial=10.0/std::max(inertialDenominator,1.0e-20);
    const double impact=closing/std::max(impactDenominator,1.0e-20);
    const double stiffness=std::max(inertial,impact);
    const double first=-stiffness*(2.0*offset*logarithm+offset*offset/s2);
    const double second=-stiffness*(2.0*logarithm+4.0*offset/s2-offset*offset/(s2*s2));
    const double radialRaw=4.0*s2*second+2.0*first;
    const double radialProjected=separation<1.0e-4*th?0.0:std::max(radialRaw,0.0);
    const double force=-2.0*distance*first,lambda=std::max(force,0.0)*dt;
    const bool strictInertial=std::isfinite(inertial)&&std::isfinite(impact)&&inertial>0.0&&
        impact<inertial*(1.0-32.0*double(std::numeric_limits<float>::epsilon()));
    const bool barrierMargin=separation>1.0e-4*th&&separation<th&&
        th-distance>1.0e-3*th&&inertialDenominator>1.0e-20&&impactDenominator>1.0e-20&&
        force>0.0&&lambda>0.0&&inverseResponse>1.0e-12;
    const bool localDistanceConsistent=std::abs(norm(outside)-separation)<=64.0*double(std::numeric_limits<float>::epsilon())*
        std::max({norm(outside),separation,th});
    OracleSample sample{};
    sample.witness=witness;sample.normal=normal;sample.impulse=mul(normal,-lambda);sample.torque=cross(lever,sample.impulse);
    sample.separation=separation;sample.normalVelocity=normalVelocity;sample.lambda=lambda;sample.inverseResponse=inverseResponse;sample.stiffness=stiffness;
    sample.forceN=force;sample.firstSquared=first;sample.secondSquared=second;sample.radialCurvatureRaw=radialRaw;sample.radialCurvatureProjected=radialProjected;
    sample.stiffnessBranch=strictInertial?0:(impact>inertial?1:-1);sample.feature=feature;
    const bool supported=featureMargin&&barrierMargin&&strictInertial&&localDistanceConsistent&&
        std::isfinite(norm(sample.impulse))&&std::isfinite(norm(sample.torque));
    return {sample,supported,featureMargin,barrierMargin,strictInertial};
}

std::string fullContactInputBinding(const ProductionContact& scene,const ProductionContact::State& state,
    const std::array<double,6>& candidate,const std::array<double,6>& direction){
    std::vector<float> words;
    auto append=[&](nm_float4 v){words.insert(words.end(),{v.x,v.y,v.z,v.w});};
    auto appendMR=[&](mr_float4 v){words.insert(words.end(),{v.x,v.y,v.z,v.w});};
    appendMR(state.body.position);appendMR(state.body.orientation);appendMR(state.body.linearVelocityAndInverseMass);appendMR(state.body.angularVelocity);
    appendMR(state.body.inverseInertiaWorldRow0);appendMR(state.body.inverseInertiaWorldRow1);appendMR(state.body.inverseInertiaWorldRow2);
    const auto& proxy=scene.proxySource();append(proxy.localCenterAndRadius);append(proxy.localExtent);append(proxy.localOrientation);
    append(state.node.positionAndMass);append(state.node.velocityAndInverseMass);
    words.push_back(float(scene.timestep()));words.push_back(float(scene.thickness()));
    for(double x:candidate)words.push_back(float(x));for(double x:direction)words.push_back(float(x));
    return fp32WordList(words);
}

std::string values6(const std::array<double,6>& values){
    std::ostringstream out;out<<std::setprecision(17);
    for(size_t i=0;i<values.size();++i){if(i)out<<',';out<<values[i];}
    return out.str();
}
struct ContinuousFD {
    std::array<double,6> action{},residualPlus{},residualMinus{},residualPlus2{},residualMinus2{};
    bool resolvedBranch=false;std::string mode="none";uint32_t plusFeature=0,minusFeature=0;
    int plusStiffness=-2,minusStiffness=-2;
    double plusGap=0.0,minusGap=0.0,plusResponse=0.0,minusResponse=0.0,plusStiff=0.0,minusStiff=0.0;
};
bool sameContinuousBranch(const FP64ContactProbe& center,const FP64ContactProbe& other){
    return center.supportedSmoothBranch&&other.supportedSmoothBranch&&center.sample.feature==other.sample.feature&&
        center.sample.stiffnessBranch==other.sample.stiffnessBranch;
}
ContinuousFD fp64ContinuousDirectionalAction(const ProductionContact& scene,const ProductionContact::State& state,
    const std::array<double,6>& centerCandidate,const std::array<double,6>& direction,double epsilon,
    const FP64ContactProbe& center){
    auto candidateAt=[&](double scale){auto value=centerCandidate;for(size_t i=0;i<6;++i)value[i]+=scale*direction[i];return value;};
    const auto plus=fp64ContinuousBoxContact(scene,state,candidateAt(epsilon));
    const auto minus=fp64ContinuousBoxContact(scene,state,candidateAt(-epsilon));
    ContinuousFD out;out.plusFeature=plus.sample.feature;out.minusFeature=minus.sample.feature;out.plusStiffness=plus.sample.stiffnessBranch;out.minusStiffness=minus.sample.stiffnessBranch;
    out.plusGap=plus.sample.separation;out.minusGap=minus.sample.separation;out.plusResponse=plus.sample.inverseResponse;out.minusResponse=minus.sample.inverseResponse;
    out.plusStiff=plus.sample.stiffness;out.minusStiff=minus.sample.stiffness;
    out.residualPlus=energyScaled(contactResidual(plus.sample),scene.characteristicLength());
    out.residualMinus=energyScaled(contactResidual(minus.sample),scene.characteristicLength());
    const auto r0=energyScaled(contactResidual(center.sample),scene.characteristicLength());
    if(sameContinuousBranch(center,plus)&&sameContinuousBranch(center,minus)){
        for(size_t i=0;i<6;++i)out.action[i]=-(out.residualPlus[i]-out.residualMinus[i])/(2.0*epsilon);
        out.resolvedBranch=true;out.mode="central";return out;
    }
    // At a true one-sided feature boundary, keep the candidate's center
    // feature and use a second-order one-sided stencil only when both points
    // remain on that same fully smooth feature/stiffness branch.
    const auto plus2=fp64ContinuousBoxContact(scene,state,candidateAt(2.0*epsilon));
    const auto minus2=fp64ContinuousBoxContact(scene,state,candidateAt(-2.0*epsilon));
    out.residualPlus2=energyScaled(contactResidual(plus2.sample),scene.characteristicLength());
    out.residualMinus2=energyScaled(contactResidual(minus2.sample),scene.characteristicLength());
    if(sameContinuousBranch(center,plus)&&sameContinuousBranch(center,plus2)){
        for(size_t i=0;i<6;++i)out.action[i]=-(-3.0*r0[i]+4.0*out.residualPlus[i]-out.residualPlus2[i])/(2.0*epsilon);
        out.resolvedBranch=true;out.mode="forward_second_order";return out;
    }
    if(sameContinuousBranch(center,minus)&&sameContinuousBranch(center,minus2)){
        for(size_t i=0;i<6;++i)out.action[i]=-(3.0*r0[i]-4.0*out.residualMinus[i]+out.residualMinus2[i])/(2.0*epsilon);
        out.resolvedBranch=true;out.mode="backward_second_order";return out;
    }
    return out;
}

bool testValidator(Device& d){
    const double omega=3.141592653589793/.1;
    const SweepCase crossing{"physical_rotating_endpoints_clear_midpath_hit",{0,0,0},{.1,.005,.005},{0,.07,0},{0,.07,0},{0,.07,0},{0,0,omega},{0,0,omega},true,false};
    const SweepCase miss{"rotating_separated_miss",{0,0,0},{.1,.005,.005},{0,.2,0},{0,.2,0},{0,.2,0},{0,0,omega},{0,0,omega},true,true};
    const SweepCase offset{"offset_proxy_swept_center_bound",{.12,0,0},{.005,.004,.003},{0,.12,0},{0,.12,0},{0,.12,0},{0,0,omega},{0,0,omega},true,false};
    const SweepCase alphaOnly{"alpha_path_hit_physical_path_clear",{0,0,0},{.1,.005,.005},{0,.07,0},{0,.07,0},{0,.07,0},{0,0,omega},{0,0,0},false,true};
    const SweepCase physicalOnly{"physical_path_hit_alpha_path_clear",{0,0,0},{.1,.005,.005},{0,-.2,0},{0,.07,0},{0,.07,0},{0,0,omega},{0,0,omega},true,false};
    bool allExpected=true;
    for(const auto& c:{crossing,miss,offset,alphaOnly,physicalOnly}){
        NMMatterStatusGPU status{};
        try { status=runValidator(d,c); }
        catch(const std::exception& e) {
            allExpected=false;
            std::cout<<"validator_case="<<c.name<<" status=UNAVAILABLE error="<<e.what()<<"\n";
            continue;
        }
        const auto alpha=sampleSweepFP64(c,false);
        const auto physical=sampleSweepFP64(c,true);
        const bool alphaOracleClear=alpha.minimumGap>alpha.floor;
        const bool physicalOracleClear=physical.minimumGap>physical.floor;
        const bool oracleMatchesAuthored=alphaOracleClear==c.expectedAlphaClear &&
            physicalOracleClear==c.expectedPhysicalClear;
        const bool productionAccepted=status.code==NM_STATUS_SUCCESS;
        const bool productionRejected=status.code==NM_STATUS_CONTACT_FAILURE;
        const bool actualAlphaCertified=productionAccepted ||
            (productionRejected && status.diagnostics.w>0.5f);
        const bool expectedReject=!c.expectedAlphaClear || !c.expectedPhysicalClear;
        const bool productionStatusMatches=expectedReject ? productionRejected : productionAccepted;
        const bool alphaCertificateMatches=actualAlphaCertified==alphaOracleClear;
        const bool pathPass=oracleMatchesAuthored && productionStatusMatches && alphaCertificateMatches;
        allExpected=allExpected&&pathPass;
        const char* productionPhysicalState=productionAccepted
            ? "certified_clear"
            : (productionRejected && actualAlphaCertified
                ? "physical_sweep_rejected_or_uncertified"
                : "not_isolated_alpha_path_also_failed");
        // Emit raw status and both gap observations before any gate can fail.
        // The production physical diagnostic is a minimum sampled gap; it is
        // not a CCD-clear certificate. The independent oracle is also sampled.
        std::cout<<std::setprecision(12)
            <<"validator_case="<<c.name
            <<" status="<<status.code
            <<" object="<<status.objectIndex
            <<" failing_index="<<status.failingIndex
            <<" diag_alpha_min_gap="<<status.diagnostics.x
            <<" diag_physical_min_sample_gap="<<status.diagnostics.y
            <<" diag_floor="<<status.diagnostics.z
            <<" diag_alpha_certified="<<(status.diagnostics.w>0.5f?1:0)
            <<" production_physical_state="<<productionPhysicalState
            <<" fp64_alpha_min_sample_gap="<<alpha.minimumGap
            <<" fp64_alpha_min_fraction="<<alpha.fractionAtMinimum
            <<" fp64_alpha_mid_gap="<<alpha.midpointGap
            <<" fp64_physical_min_sample_gap="<<physical.minimumGap
            <<" fp64_physical_min_fraction="<<physical.fractionAtMinimum
            <<" fp64_physical_mid_gap="<<physical.midpointGap
            <<" fp64_floor="<<physical.floor
            <<" fp64_alpha_sampled_clear="<<(alphaOracleClear?1:0)
            <<" fp64_physical_sampled_clear="<<(physicalOracleClear?1:0)
            <<" authored_expect_alpha_clear="<<(c.expectedAlphaClear?1:0)
            <<" authored_expect_physical_clear="<<(c.expectedPhysicalClear?1:0)
            <<" path_case="<<(pathPass?"PASS":"FAIL")<<"\n";
    }
    std::cout<<"production_rotating_path_fixture_suite="<<(allExpected?"PASS":"FAIL")
        <<" interpretation=sampled_FP64_fixture_oracle_plus_production_certificate; positive_min_sample_gap_does_not_prove_swept_clearance\n";
    return allExpected;
}

struct ContactProbe {std::array<double,6> residual{};bool stable=false;int stiffnessBranch=-1;uint32_t feature=0;};
ContactProbe projectContactResidual(ProductionContact& scene,const ProductionContact::State& state,const std::array<double,6>& delta){
    const auto projected=scene.candidateRigid(state,delta);const auto oracle=fp64Sample(scene,state,projected);
    // Approaching contacts are valid. The active branch is the max of the
    // inertial and impact stiffnesses; reject only a branch switch or a
    // sample outside the same positive, smooth barrier state.
    const bool stable=oracle.separation>0 && oracle.separation<scene.thickness() &&
        oracle.lambda>0 && oracle.stiffnessBranch>=0;
    return {contactResidual(oracle),stable,oracle.stiffnessBranch,oracle.feature};
}

enum class Outcome { pass, fail, inconclusive };
const char* outcomeName(Outcome outcome){
    switch(outcome){case Outcome::pass:return "PASS";case Outcome::fail:return "FAIL";case Outcome::inconclusive:return "INCONCLUSIVE";}
    return "FAIL";
}

Outcome testActions(Device& d){
    ProductionContact scene(d);
    const std::vector<ContactCase> cases={
        {"face_pos",{.052,.011,-.007}}, {"face_neg",{-.052,-.009,.006}},
        {"edge_mixed",{-.052,.042,.006}}, {"edge_mixed_2",{.052,-.042,-.004}},
        {"corner_mixed",{-.052,.042,-.032}}, {"corner_pos",{.052,.042,.032}},
    };
    std::mt19937 rng(0x8f12a6u);std::uniform_real_distribution<double> dist(-1,1);
    constexpr double targetTolerance=1e-3;
    constexpr std::array<double,5> epsilons{4e-3,2e-3,1e-3,5e-4,2.5e-4};
    constexpr std::array<double,5> continuousEpsilons{1e-3,5e-4,2.5e-4,1.25e-4,6.25e-5};
    bool allSampleChecks=true,allResolved=true,anyResolvedActionFailure=false;double worstAbs=0,worstRel=0,worstDirectSampleRel=0,worstRepeatFloor=0;unsigned directionsChecked=0;
    bool allContinuousResolved=true,anyContinuousActionFailure=false;double worstContinuousRel=0,worstContinuousAbs=0,worstContinuousCenterRel=0;unsigned continuousDirections=0;
    for(const auto& c:cases){
        const auto state=scene.baseline(c);const auto sampleEval=scene.evaluate(state);
        if (sampleEval.status.code != NM_STATUS_SUCCESS ||
            (sampleEval.sample.identity.w & NM_CONTACT_VALID) == 0u ||
            !(sampleEval.sample.barrierHessianRow0.w > 0.0f &&
              sampleEval.sample.barrierHessianRow1.w > 0.0f)) {
            printContactSetup("production_sample_rejected", c, state, sampleEval);
            allSampleChecks = false;
            std::cout << "action_case=" << c.name
                      << " outcome=SKIPPED_AFTER_RETAINED_SAMPLE_FAILURE\n";
            continue;
        }
        const auto oracle=fp64Sample(scene,state,sampleEval.rigid);const auto oracleR=energyScaled(contactResidual(oracle),scene.characteristicLength()),nativeR=energyScaled(sampleResidual(sampleEval.sample),scene.characteristicLength());
        double residualError=0;for(size_t i=0;i<6;++i)residualError+=std::pow(nativeR[i]-oracleR[i],2);residualError=std::sqrt(residualError);
        const double residualScale=std::max(vectorNorm(oracleR),1e-20);
        double geometryAbs=std::abs(sampleEval.sample.pointAndSeparation.w-oracle.separation);
        const std::array<double,3> nativeWitness{sampleEval.sample.pointAndSeparation.x,sampleEval.sample.pointAndSeparation.y,sampleEval.sample.pointAndSeparation.z};
        const std::array<double,3> nativeNormal{sampleEval.sample.normalAndVelocity.x,sampleEval.sample.normalAndVelocity.y,sampleEval.sample.normalAndVelocity.z};
        for(int i=0;i<3;++i){geometryAbs=std::max(geometryAbs,std::abs(nativeWitness[i]-oracle.witness[i]));geometryAbs=std::max(geometryAbs,std::abs(nativeNormal[i]-oracle.normal[i]));}
        const double velocityAbs=std::abs(double(sampleEval.sample.normalAndVelocity.w)-oracle.normalVelocity);
        const double responseAbs=std::abs(double(sampleEval.sample.barrierHessianRow1.w)-oracle.inverseResponse);
        const double stiffnessRel=std::abs(double(sampleEval.sample.barrier.w)-oracle.stiffness)/std::max(oracle.stiffness,1e-20);
        const double nativeForceN=double(sampleEval.sample.barrier.x)/scene.timestep();
        const double nativeRadialCurvature=double(sampleEval.sample.barrier.y)/(scene.timestep()*scene.timestep());
        const double radialAbs=std::abs(nativeRadialCurvature-oracle.radialCurvatureProjected);
        const double radialRel=radialAbs/std::max(std::abs(oracle.radialCurvatureProjected),1e-20);
        const double impulseRel=residualError/residualScale;worstDirectSampleRel=std::max(worstDirectSampleRel,impulseRel);
        const bool samplePass=geometryAbs<=2e-6 && velocityAbs<=2e-5 && responseAbs<=2e-4*oracle.inverseResponse && stiffnessRel<=2e-4 &&
            residualError<=2e-4*residualScale && std::abs(double(sampleEval.sample.barrier.x)-oracle.lambda)<=2e-4*std::max(oracle.lambda,1e-20) && radialRel<=2e-4;
        allSampleChecks=allSampleChecks&&samplePass;
        std::cout<<"direct_sample case="<<c.name<<" fp32_oracle_gap_m="<<oracle.separation<<" gap_abs_m="<<std::abs(double(sampleEval.sample.pointAndSeparation.w)-oracle.separation)
            <<" geometry_max_abs="<<geometryAbs<<" normal_velocity_abs="<<velocityAbs<<" inverse_response_rel="<<responseAbs/oracle.inverseResponse<<" barrier_stiffness_rel="<<stiffnessRel
            <<" energy_scaled_residual_l2_rel="<<impulseRel<<" characteristic_length_m="<<scene.characteristicLength()
            <<" barrier_stiffness="<<oracle.stiffness<<" force_oracle_N="<<oracle.forceN<<" force_native_N="<<nativeForceN
            <<" impulse_oracle_Ns="<<oracle.lambda<<" impulse_native_Ns="<<sampleEval.sample.barrier.x
            <<" first_squared_over_k_m2="<<oracle.firstSquared/oracle.stiffness
            <<" second_squared_over_k_1="<<oracle.secondSquared/oracle.stiffness
            <<" radial_curvature_raw_N_per_m="<<oracle.radialCurvatureRaw
            <<" radial_curvature_raw_over_k_m2="<<oracle.radialCurvatureRaw/oracle.stiffness
            <<" radial_curvature_projected_N_per_m="<<oracle.radialCurvatureProjected
            <<" radial_curvature_native_N_per_m="<<nativeRadialCurvature<<" radial_curvature_rel="<<radialRel
            <<" radial_curvature_clamped="<<(oracle.radialCurvatureRaw<0.0?1:0)
            <<" sample_gate="<<(samplePass?"PASS":"FAIL")<<"\n";

        // Re-evaluate exactly the same FP32 source and projected point to
        // measure, rather than assume, the absolute repeat floor.
        double repeatFloor=0.0;
        for(int repeat=0;repeat<3;++repeat){
            const auto repeated=scene.evaluate(state);need(repeated.status.code==NM_STATUS_SUCCESS,c.name+" repeated native sample status failure");
            const auto repeatedOracle=fp64Sample(scene,state,repeated.rigid);
            repeatFloor=std::max(repeatFloor,maxDifference(oracleR,energyScaled(contactResidual(repeatedOracle),scene.characteristicLength())));
            repeatFloor=std::max(repeatFloor,maxDifference(nativeR,energyScaled(sampleResidual(repeated.sample),scene.characteristicLength())));
        }
        worstRepeatFloor=std::max(worstRepeatFloor,repeatFloor);
        const std::array<double,6> continuousCandidate{};
        const auto continuousCenter=fp64ContinuousBoxContact(scene,state,continuousCandidate);
        const auto continuousCenterResidual=energyScaled(contactResidual(continuousCenter.sample),scene.characteristicLength());
        const double continuousCenterError=vectorNorm({nativeR[0]-continuousCenterResidual[0],nativeR[1]-continuousCenterResidual[1],nativeR[2]-continuousCenterResidual[2],nativeR[3]-continuousCenterResidual[3],nativeR[4]-continuousCenterResidual[4],nativeR[5]-continuousCenterResidual[5]});
        const double continuousCenterRel=continuousCenterError/std::max(vectorNorm(continuousCenterResidual),1e-30);
        worstContinuousCenterRel=std::max(worstContinuousCenterRel,continuousCenterRel);
        std::cout<<"contact_fp64_continuous_center case="<<c.name
            <<" exact_f32_source_promoted=true candidate_velocity_delta=0"
            <<" supported="<<(continuousCenter.supportedSmoothBranch?1:0)
            <<" feature="<<continuousCenter.sample.feature
            <<" stiffness_branch="<<continuousCenter.sample.stiffnessBranch
            <<" feature_margin="<<(continuousCenter.featureMargin?1:0)
            <<" barrier_margin="<<(continuousCenter.barrierMargin?1:0)
            <<" strict_inertial="<<(continuousCenter.strictInertialBranch?1:0)
            <<" separation_m="<<continuousCenter.sample.separation
            <<" normal_velocity_m_per_s="<<continuousCenter.sample.normalVelocity
            <<" lambda_Ns="<<continuousCenter.sample.lambda
            <<" residual_energy_scaled_l2="<<vectorNorm(continuousCenterResidual)
            <<" R_fp64_energy_scaled="<<values6(continuousCenterResidual)
            <<" R_native_energy_scaled="<<values6(nativeR)
            <<" native_center_residual_rel="<<continuousCenterRel
            <<" body_motion="<<state.body.flagsAndIndices[0]<<" proxy_shape="<<scene.proxySource().shapeKind
            <<" proxy_flags="<<scene.proxySource().flags<<" friction_static_dynamic=0,0 accepted_tangent_history=zero"
            <<" input_word_schema=source_body7xfloat4_proxy3xfloat4_node2xfloat4_dt_thickness_candidate6_direction6_fp32_hex\n";
        std::vector<std::array<double,6>> directions;
        for(int k=0;k<6;++k){std::array<double,6> e{};e[k]=1;directions.push_back(e);}
        for(unsigned k=0;k<8;++k){std::array<double,6> v{};double n=0;for(double& x:v){x=dist(rng);n+=x*x;}n=std::sqrt(n);for(double& x:v)x/=n;directions.push_back(v);}
        unsigned directionIndex=0;
        for(const auto& authoredDirection:directions){
            std::array<double,6> direction{};for(size_t i=0;i<6;++i)direction[i]=float(authoredDirection[i]);
            const auto action=energyScaled(scene.action(sampleEval.sample,state,direction),scene.characteristicLength());
            double nativeRepeat=0;for(int repeat=0;repeat<2;++repeat)nativeRepeat=std::max(nativeRepeat,maxDifference(action,energyScaled(scene.action(sampleEval.sample,state,direction),scene.characteristicLength())));
            repeatFloor=std::max(repeatFloor,nativeRepeat);worstRepeatFloor=std::max(worstRepeatFloor,repeatFloor);
            std::cout<<"contact_fp64_continuous_input case="<<c.name<<" direction="<<directionIndex
                <<" exact_f32_inputs_hex="<<fullContactInputBinding(scene,state,continuousCandidate,direction)
                <<" convention=candidate_velocity_delta;position_increment=dt*delta_v;SO3_left_world_rotation;A=-dR_contact\n";
            std::array<ContinuousFD,continuousEpsilons.size()> continuousFD{};
            for(size_t ei=0;ei<continuousEpsilons.size();++ei){
                continuousFD[ei]=fp64ContinuousDirectionalAction(scene,state,continuousCandidate,direction,continuousEpsilons[ei],continuousCenter);
                const auto& fd=continuousFD[ei];
                std::cout<<"contact_fp64_continuous_fd case="<<c.name<<" direction="<<directionIndex
                    <<" epsilon_velocity="<<continuousEpsilons[ei]<<" mode="<<fd.mode
                    <<" center_feature="<<continuousCenter.sample.feature<<" plus_feature="<<fd.plusFeature<<" minus_feature="<<fd.minusFeature
                    <<" center_stiffness_branch="<<continuousCenter.sample.stiffnessBranch<<" plus_stiffness_branch="<<fd.plusStiffness<<" minus_stiffness_branch="<<fd.minusStiffness
                    <<" plus_gap_m="<<fd.plusGap<<" minus_gap_m="<<fd.minusGap
                    <<" plus_response="<<fd.plusResponse<<" minus_response="<<fd.minusResponse
                    <<" plus_stiffness="<<fd.plusStiff<<" minus_stiffness="<<fd.minusStiff
                    <<" branch_resolved="<<(fd.resolvedBranch?1:0)<<" Rplus_energy_scaled="<<values6(fd.residualPlus)
                    <<" Rminus_energy_scaled="<<values6(fd.residualMinus)
                    <<" Rplus2_energy_scaled="<<values6(fd.residualPlus2)<<" Rminus2_energy_scaled="<<values6(fd.residualMinus2)
                    <<" A_minus_dR_energy_scaled="<<values6(fd.action)<<" A_minus_dR_l2="<<vectorNorm(fd.action)<<"\n";
            }
            const auto& continuousRef=continuousFD.back().action;
            const double continuousNorm=vectorNorm(continuousRef);
            auto fdDifference=[&](size_t a,size_t b){std::array<double,6> d{};for(size_t i=0;i<6;++i)d[i]=continuousFD[a].action[i]-continuousFD[b].action[i];return vectorNorm(d);};
            const double cD23=fdDifference(2,3),cD34=fdDifference(3,4);
            const bool sameFinalStencil=continuousFD[2].resolvedBranch&&continuousFD[3].resolvedBranch&&continuousFD[4].resolvedBranch&&
                continuousFD[2].mode==continuousFD[3].mode&&continuousFD[3].mode==continuousFD[4].mode;
            const double cError=vectorNorm({action[0]-continuousRef[0],action[1]-continuousRef[1],action[2]-continuousRef[2],action[3]-continuousRef[3],action[4]-continuousRef[4],action[5]-continuousRef[5]});
            const double cRelative=cError/std::max(continuousNorm,1e-30),cComponent=maxDifference(action,continuousRef);
            const bool cResolved=continuousCenter.supportedSmoothBranch&&sameFinalStencil&&continuousNorm>1e-20&&
                cD34<=1.2*cD23&&cD34<=3.0e-6*continuousNorm;
            const bool cPass=cResolved&&cRelative<=targetTolerance&&
                cComponent<=nativeRepeat+targetTolerance*continuousNorm&&continuousCenterRel<=targetTolerance;
            allContinuousResolved=allContinuousResolved&&cResolved;
            anyContinuousActionFailure=anyContinuousActionFailure||(cResolved&&!cPass);
            worstContinuousRel=std::max(worstContinuousRel,cRelative);worstContinuousAbs=std::max(worstContinuousAbs,cComponent);++continuousDirections;
            std::cout<<"contact_fp64_continuous_summary case="<<c.name<<" direction="<<directionIndex
                <<" epsilons="<<continuousEpsilons[0]<<","<<continuousEpsilons[1]<<","<<continuousEpsilons[2]<<","<<continuousEpsilons[3]<<","<<continuousEpsilons[4]
                <<" final_mode="<<continuousFD[4].mode<<" final_refinement_abs="<<cD34<<" prior_refinement_abs="<<cD23
                <<" final_refinement_estimate_rel="<<cD34/(3.0*std::max(continuousNorm,1e-30))<<" refinement_threshold=1e-6"
                <<" resolved="<<(cResolved?1:0)<<" native_action_energy_scaled="<<values6(action)
                <<" fp64_A_minus_dR_energy_scaled="<<values6(continuousRef)
                <<" native_A_minus_dR_l2_rel="<<cRelative<<" component_max="<<cComponent
                <<" center_residual_rel="<<continuousCenterRel<<" action_threshold="<<targetTolerance
                <<" outcome="<<(cPass?"PASS":(cResolved?"FAIL":"INCONCLUSIVE"))<<"\n";
            std::array<std::array<double,6>,epsilons.size()> finiteDifferences{};bool stableBranch=oracle.stiffnessBranch>=0;
            for(size_t ei=0;ei<epsilons.size();++ei){
                const double epsilon=epsilons[ei];std::array<double,6> plusDelta{},minusDelta{};
                for(size_t i=0;i<6;++i){plusDelta[i]=float(epsilon*direction[i]);minusDelta[i]=float(-epsilon*direction[i]);}
                const auto plus=projectContactResidual(scene,state,plusDelta),minus=projectContactResidual(scene,state,minusDelta);
                stableBranch=stableBranch&&plus.stable&&minus.stable&&
                    oracle.feature==plus.feature&&oracle.feature==minus.feature&&
                    oracle.stiffnessBranch==plus.stiffnessBranch&&
                    oracle.stiffnessBranch==minus.stiffnessBranch;
                const auto plusResidual=energyScaled(plus.residual,scene.characteristicLength()),minusResidual=energyScaled(minus.residual,scene.characteristicLength());
                std::array<double,6> realizedDirection{};double actualHalfStep=0;
                for(size_t i=0;i<6;++i){realizedDirection[i]=0.5*(plusDelta[i]-minusDelta[i]);actualHalfStep+=realizedDirection[i]*direction[i];}
                double orthogonal=0;for(size_t i=0;i<6;++i)orthogonal+=std::pow(realizedDirection[i]-actualHalfStep*direction[i],2);orthogonal=std::sqrt(orthogonal);
                stableBranch=stableBranch&&actualHalfStep>0&&orthogonal<=1e-5*actualHalfStep;
                // The production Krylov operator is A=-dR/dx. Contact
                // accumulation adds the rigid impulse to R, while the
                // operator kernel returns its negative directional
                // derivative. Compare against -FD(R), not FD(R).
                for(size_t i=0;i<6;++i)finiteDifferences[ei][i]=-(plusResidual[i]-minusResidual[i])/(2.0*actualHalfStep);
                const double fdNorm=vectorNorm(finiteDifferences[ei]);const double errorNorm=vectorNorm({action[0]-finiteDifferences[ei][0],action[1]-finiteDifferences[ei][1],action[2]-finiteDifferences[ei][2],action[3]-finiteDifferences[ei][3],action[4]-finiteDifferences[ei][4],action[5]-finiteDifferences[ei][5]});
                std::cout<<"contact_fd case="<<c.name<<" direction="<<directionIndex<<" epsilon="<<epsilon<<" feature="<<oracle.feature
                    <<" actual_half_step="<<actualHalfStep<<" realized_orthogonal="<<orthogonal<<" branch_stable="<<(stableBranch?1:0)<<" stiffness_branch="<<oracle.stiffnessBranch<<" A_minus_dR_fd_l2="<<fdNorm<<" action_l2="<<vectorNorm(action)<<" error_l2_rel="<<errorNorm/std::max(fdNorm,1e-30)
                    <<" error_component_max="<<maxDifference(action,finiteDifferences[ei])<<"\n";
            }
            const double d01=vectorNorm({finiteDifferences[0][0]-finiteDifferences[1][0],finiteDifferences[0][1]-finiteDifferences[1][1],finiteDifferences[0][2]-finiteDifferences[1][2],finiteDifferences[0][3]-finiteDifferences[1][3],finiteDifferences[0][4]-finiteDifferences[1][4],finiteDifferences[0][5]-finiteDifferences[1][5]});
            const double d12=vectorNorm({finiteDifferences[1][0]-finiteDifferences[2][0],finiteDifferences[1][1]-finiteDifferences[2][1],finiteDifferences[1][2]-finiteDifferences[2][2],finiteDifferences[1][3]-finiteDifferences[2][3],finiteDifferences[1][4]-finiteDifferences[2][4],finiteDifferences[1][5]-finiteDifferences[2][5]});
            const double d23=vectorNorm({finiteDifferences[2][0]-finiteDifferences[3][0],finiteDifferences[2][1]-finiteDifferences[3][1],finiteDifferences[2][2]-finiteDifferences[3][2],finiteDifferences[2][3]-finiteDifferences[3][3],finiteDifferences[2][4]-finiteDifferences[3][4],finiteDifferences[2][5]-finiteDifferences[3][5]});
            const double d34=vectorNorm({finiteDifferences[3][0]-finiteDifferences[4][0],finiteDifferences[3][1]-finiteDifferences[4][1],finiteDifferences[3][2]-finiteDifferences[4][2],finiteDifferences[3][3]-finiteDifferences[4][3],finiteDifferences[3][4]-finiteDifferences[4][4],finiteDifferences[3][5]-finiteDifferences[4][5]});
            const auto& fd=finiteDifferences.back();const double fdNorm=vectorNorm(fd);const std::array<double,6> error{action[0]-fd[0],action[1]-fd[1],action[2]-fd[2],action[3]-fd[3],action[4]-fd[4],action[5]-fd[5]};
            const double errorNorm=vectorNorm(error),relative=errorNorm/std::max(fdNorm,1e-30),componentMax=maxAbs(error);
            const double componentLimit=repeatFloor+targetTolerance*fdNorm;
            const bool resolved=stableBranch&&fdNorm>10.0*repeatFloor&&d34<=d23&&d23<=d12&&d34<=targetTolerance*fdNorm;
            const bool actionPass=resolved&&relative<=targetTolerance&&componentMax<=componentLimit;
            allResolved=allResolved&&resolved;anyResolvedActionFailure=anyResolvedActionFailure||(resolved&&!actionPass);
            worstAbs=std::max(worstAbs,componentMax);worstRel=std::max(worstRel,relative);++directionsChecked;
            std::cout<<"contact_fd_summary case="<<c.name<<" direction="<<directionIndex<<" delta_A_minus_dR_l2="<<d01<<","<<d12<<","<<d23<<","<<d34
                <<" repeat_floor_abs="<<repeatFloor<<" resolved="<<(resolved?1:0)<<" final_l2_rel="<<relative<<" final_component_max="<<componentMax
                <<" component_limit="<<componentLimit<<" action_gate="<<(actionPass?"PASS":(resolved?"FAIL":"INCONCLUSIVE"))<<"\n";
            ++directionIndex;
        }
        std::cout<<"action_case="<<c.name<<" directions="<<directions.size()<<" contact_separation="<<sampleEval.sample.pointAndSeparation.w<<" stiffness_branch="<<oracle.stiffnessBranch<<"\n";
    }
    std::cout<<"production_contact_operator_fp64_fd directions="<<directionsChecked<<" worst_component_abs="<<std::setprecision(9)<<worstAbs<<" worst_vector_relative="<<worstRel
        <<" direct_sample_worst_relative="<<worstDirectSampleRel<<" measured_repeat_floor_abs="<<worstRepeatFloor<<" threshold="<<targetTolerance
        <<" direct_sample="<<(allSampleChecks?"PASS":"FAIL")<<" resolved_fd="<<(allResolved?"PASS":"INCONCLUSIVE")
        <<" action="<<(anyResolvedActionFailure?"FAIL":(allResolved?"PASS":"INCONCLUSIVE"))
        <<" norm=torque_components_divided_by_proxy_lever_length source=nm_project_primal_free_rigid_candidate+nm_contact_evaluate+nm_fgmres_apply_primal_rigid_contacts\n";
    std::cout<<"production_contact_operator_full_fp64_continuous directions="<<continuousDirections
        <<" worst_native_action_l2_relative="<<worstContinuousRel<<" worst_component_abs="<<worstContinuousAbs
        <<" worst_center_residual_l2_relative="<<worstContinuousCenterRel<<" threshold="<<targetTolerance
        <<" resolved="<<(allContinuousResolved?"YES":"NO")
        <<" action="<<(anyContinuousActionFailure?"FAIL":(allContinuousResolved?"PASS":"INCONCLUSIVE"))
        <<" independent_of_legacy_fp32_projection_fd=true output=contact_only_rigid_generalized_rows\n";
    if(!allSampleChecks||anyResolvedActionFailure||anyContinuousActionFailure)return Outcome::fail;
    if(!allResolved)return Outcome::inconclusive;
    if(!allContinuousResolved)return Outcome::inconclusive;
    return Outcome::pass;
}

Outcome testInertiaAction(Device& d){
    ProductionContact scene(d);const ContactCase c{"inertia_rotated_candidate",{-.052,.042,-.032}};
    const auto state=scene.baseline(c);const auto sample=scene.evaluate(state);
    if (sample.status.code != NM_STATUS_SUCCESS ||
        (sample.sample.identity.w & NM_CONTACT_VALID) == 0u) {
        printContactSetup("inertia_sample_rejected", c, state, sample);
        std::cout << "inertia_action=SKIPPED_AFTER_RETAINED_SAMPLE_FAILURE\n";
        return Outcome::fail;
    }
    const std::array<double,6> candidate{.0,.0,.0,.13,-.09,.17};const auto baseRigid=scene.candidateRigid(state,candidate);
    const V3 candidateOmega{float(candidate[3]),float(candidate[4]),float(candidate[5])};
    std::cout<<"inertia_chain_binding source_q0_omega0_dt_Iinv0_candidate_f32_words="
        <<inertiaInputBinding(state.body,candidateOmega,scene.timestep())
        <<" source_material=actual_native_fixture_MRBodyStateGPU_fields\n";
    std::mt19937 rng(0x734bc1u);std::uniform_real_distribution<double> dist(-1,1);std::vector<std::array<double,6>> directions;
    for(int axis=0;axis<3;++axis){std::array<double,6> direction{};direction[axis+3]=1;directions.push_back(direction);}
    for(int trial=0;trial<5;++trial){std::array<double,6> direction{};double n=0;for(int i=3;i<6;++i){direction[i]=dist(rng);n+=direction[i]*direction[i];}n=std::sqrt(n);for(int i=3;i<6;++i)direction[i]/=n;directions.push_back(direction);}
    constexpr std::array<double,5> epsilons{4e-2,2e-2,1e-2,5e-3,2.5e-3};constexpr double tolerance=1e-3;
    constexpr std::array<double,4> fp64Epsilons{1e-4,1e-5,1e-6,1e-7};constexpr double fp64Tolerance=1e-6;
    bool allResolved=true,anyResolvedFailure=false,fp64SmoothAll=true,nativeChainAll=true;double worstRel=0,worstAbs=0,repeatFloor=0,worstFp64Rel=0,worstFp64Component=0,worstNativeChainRel=0,worstNativeChainComponent=0;
    for(size_t di=0;di<directions.size();++di){
        std::array<double,6> direction=directions[di];for(double& v:direction)v=float(v);
        const V3 directionOmega{float(direction[3]),float(direction[4]),float(direction[5])};
        const auto action=scene.freeRigidAction(sample.sample,baseRigid,direction,candidate);
        const auto repeated=scene.freeRigidAction(sample.sample,baseRigid,direction,candidate);
        double thisRepeatFloor=0;for(int i=3;i<6;++i){thisRepeatFloor=std::max(thisRepeatFloor,std::abs(action[i]-repeated[i]));repeatFloor=std::max(repeatFloor,thisRepeatFloor);}

        const V3 fp64Analytic=fp64FullInertiaAction(state.body,candidateOmega,directionOmega,scene.timestep());
        std::array<V3,fp64Epsilons.size()> fp64FD{};
        double previousFp64Difference=std::numeric_limits<double>::infinity();bool fp64Resolved=true;
        for(size_t ei=0;ei<fp64Epsilons.size();++ei){
            const double epsilon=fp64Epsilons[ei];
            const V3 plusCandidate=add(candidateOmega,mul(directionOmega,epsilon));
            const V3 minusCandidate=sub(candidateOmega,mul(directionOmega,epsilon));
            const V3 plus=fp64FullInertiaResidual(state.body,plusCandidate,scene.timestep());
            const V3 minus=fp64FullInertiaResidual(state.body,minusCandidate,scene.timestep());
            fp64FD[ei]=mul(sub(plus,minus),-0.5/epsilon);
            const V3 error=sub(fp64Analytic,fp64FD[ei]);
            const double derivativeRel=norm3(error)/std::max(norm3(fp64Analytic),1e-30);
            const double derivativeComponent=maxAbs3(error)/std::max(norm3(fp64Analytic),1e-30);
            std::cout<<"inertia_chain_fp64_fd direction="<<di<<" epsilon="<<epsilon
                <<" analytic_l2="<<norm3(fp64Analytic)<<" rel="<<derivativeRel
                <<" component_over_analytic_l2="<<derivativeComponent<<"\n";
            if(ei==2){worstFp64Rel=std::max(worstFp64Rel,derivativeRel);worstFp64Component=std::max(worstFp64Component,derivativeComponent);}
            if(ei==3){
                fp64Resolved=derivativeRel<=fp64Tolerance&&derivativeComponent<=fp64Tolerance;
                const double refinement=norm3(sub(fp64FD[3],fp64FD[2]))/std::max(norm3(fp64Analytic),1e-30);
                fp64Resolved=fp64Resolved&&refinement<=fp64Tolerance;
            }
            if(ei>0){
                const double diff=norm3(sub(fp64FD[ei],fp64FD[ei-1]))/std::max(norm3(fp64Analytic),1e-30);
                // FP64 refinement should descend until the final roundoff floor;
                // the last difference is checked against the declared tolerance.
                if(ei<3 && diff>previousFp64Difference*1.2) fp64Resolved=false;
                previousFp64Difference=diff;
            }
        }
        const V3 nativeAngular{action[3],action[4],action[5]};
        const V3 nativeChainError=sub(nativeAngular,fp64Analytic);
        const double nativeChainRel=norm3(nativeChainError)/std::max(norm3(fp64Analytic),1e-30);
        const double nativeChainComponent=maxAbs3(nativeChainError)/std::max(norm3(fp64Analytic),1e-30);
        const bool nativeChainPass=nativeChainRel<=tolerance&&nativeChainComponent<=tolerance+thisRepeatFloor/std::max(norm3(fp64Analytic),1e-30);
        fp64SmoothAll=fp64SmoothAll&&fp64Resolved;nativeChainAll=nativeChainAll&&nativeChainPass;
        worstNativeChainRel=std::max(worstNativeChainRel,nativeChainRel);worstNativeChainComponent=std::max(worstNativeChainComponent,nativeChainComponent);
        std::cout<<"inertia_chain_native_action direction="<<di<<" fp64_smooth_derivative="<<(fp64Resolved?"PASS":"FAIL")
            <<" native_A_minus_dR_l2="<<norm3(nativeAngular)<<" fp64_A_minus_dR_l2="<<norm3(fp64Analytic)
            <<" native_fp64_rel="<<nativeChainRel<<" component_over_fp64_l2="<<nativeChainComponent
            <<" repeat_floor_abs="<<thisRepeatFloor<<" threshold="<<tolerance
            <<" outcome="<<(nativeChainPass?"PASS":"FAIL")<<"\n";

        std::array<std::array<double,3>,epsilons.size()> fd{};
        for(size_t ei=0;ei<epsilons.size();++ei){
            auto plusCandidate=candidate,minusCandidate=candidate;
            for(int i=0;i<6;++i){plusCandidate[i]=float(candidate[i]+epsilons[ei]*direction[i]);minusCandidate[i]=float(candidate[i]-epsilons[ei]*direction[i]);}
            const auto plusRigid=scene.candidateRigid(state,plusCandidate),minusRigid=scene.candidateRigid(state,minusCandidate);
            const auto plus=fp64InertialResidual(plusRigid,plusCandidate),minus=fp64InertialResidual(minusRigid,minusCandidate);
            // Retained native-FP32 candidate finite differences are a separate
            // quantization-sensitive diagnostic, not replaced by the FP64 gate.
            for(int i=0;i<3;++i)fd[ei][i]=-(plus[i]-minus[i])/(2*epsilons[ei]);
            std::array<double,3> error{};for(int i=0;i<3;++i)error[i]=action[i+3]-fd[ei][i];
            double fdNorm=0;for(double v:fd[ei])fdNorm+=v*v;fdNorm=std::sqrt(fdNorm);double errorNorm=0;for(double v:error)errorNorm+=v*v;errorNorm=std::sqrt(errorNorm);
            std::cout<<"inertia_fp32_candidate_fd direction="<<di<<" epsilon="<<epsilons[ei]<<" A_minus_dR_fd_l2="<<fdNorm<<" action_l2="<<std::sqrt(action[3]*action[3]+action[4]*action[4]+action[5]*action[5])<<" error_l2_rel="<<errorNorm/std::max(fdNorm,1e-30)<<"\n";
        }
        auto difference=[&](size_t a,size_t b){double n=0;for(int i=0;i<3;++i)n+=std::pow(fd[a][i]-fd[b][i],2);return std::sqrt(n);};
        const double d12=difference(1,2),d23=difference(2,3),d34=difference(3,4);double fdNorm=0,errorNorm=0,component=0;
        for(int i=0;i<3;++i){fdNorm+=fd[4][i]*fd[4][i];errorNorm+=std::pow(action[i+3]-fd[4][i],2);component=std::max(component,std::abs(action[i+3]-fd[4][i]));}
        fdNorm=std::sqrt(fdNorm);errorNorm=std::sqrt(errorNorm);const double rel=errorNorm/std::max(fdNorm,1e-30);
        const bool resolved=fdNorm>10*repeatFloor&&d34<=d23&&d23<=d12&&d34<=tolerance*fdNorm;
        const bool pass=resolved&&rel<=tolerance&&component<=repeatFloor+tolerance*fdNorm;
        allResolved=allResolved&&resolved;anyResolvedFailure=anyResolvedFailure||(resolved&&!pass);worstRel=std::max(worstRel,rel);worstAbs=std::max(worstAbs,component);
        std::cout<<"inertia_fp32_fd_summary direction="<<di<<" refinements="<<d12<<","<<d23<<","<<d34<<" repeat_floor_abs="<<thisRepeatFloor<<" resolved="<<(resolved?1:0)<<" final_rel="<<rel<<" component_max="<<component<<" outcome="<<(pass?"PASS":(resolved?"FAIL":"INCONCLUSIVE"))<<"\n";
    }
    const bool chainFailure=!fp64SmoothAll||!nativeChainAll;
    std::cout<<"production_free_rigid_inertia_fp64_chain directions="<<directions.size()<<" smooth_derivative="<<(fp64SmoothAll?"PASS":"FAIL")
        <<" worst_fp64_fd_rel_at_1e-6="<<worstFp64Rel<<" worst_fp64_component_over_l2="<<worstFp64Component
        <<" native_action="<<(nativeChainAll?"PASS":"FAIL")<<" worst_native_rel="<<worstNativeChainRel
        <<" worst_native_component_over_fp64_l2="<<worstNativeChainComponent<<" threshold="<<tolerance
        <<" convention=A_minus_dR equation=R_minus_Iworld_times_deltaOmega actual_fp32_input_words=true\n";
    std::cout<<"production_free_rigid_inertia_fp32_candidate_fd directions="<<directions.size()<<" worst_rel="<<worstRel<<" worst_component_abs="<<worstAbs<<" repeat_floor_abs="<<repeatFloor<<" threshold="<<tolerance<<" resolved="<<(allResolved?"YES":"NO")<<" result="<<(anyResolvedFailure?"FAIL":(allResolved?"PASS":"INCONCLUSIVE"))<<" source=nm_fgmres_apply_free_rigid\n";
    if(anyResolvedFailure||chainFailure)return Outcome::fail;
    if(!allResolved)return Outcome::inconclusive;
    return Outcome::pass;
}

} // namespace

int main(){@autoreleasepool{
    try {Device device;std::cout<<"device="<<[[device.gpu name] UTF8String]<<" native_steps=0 kernel_scope=production_contact_and_fgmres\n";const bool validatorOk=testValidator(device);const Outcome contact=testActions(device),inertia=testInertiaAction(device);if(!validatorOk||contact==Outcome::fail||inertia==Outcome::fail){std::cout<<"result="<<(!validatorOk?"FAIL_VALIDATOR":(contact==Outcome::fail&&inertia==Outcome::fail?"FAIL_CONTACT_AND_INERTIA":(contact==Outcome::fail?"FAIL_CONTACT":"FAIL_INERTIA")))<<"\n";return 2;}if(contact==Outcome::inconclusive||inertia==Outcome::inconclusive){std::cout<<"result=INCONCLUSIVE contact="<<outcomeName(contact)<<" inertia="<<outcomeName(inertia)<<"\n";return 3;}std::cout<<"result=PASS\n";return 0;}
    catch(const std::exception& e){std::cerr<<"result=FAIL message="<<e.what()<<"\n";return 2;}
}}


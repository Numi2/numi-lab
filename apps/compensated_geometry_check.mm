#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include "metalrobo/compensated_geometry_gpu.h"
#include "metalrobo/opensim_spatial_transform_gpu.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
using V=std::array<long double,3>;
V add(V a,V b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
V sub(V a,V b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
V scale(V a,long double s){return {a[0]*s,a[1]*s,a[2]*s};}
V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
long double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
V vec(mr_float4 a){return {a.x,a.y,a.z};}
V exact(MRCompensatedPositionGPU a){return add(vec(a.high),vec(a.low));}
V rotate(mr_float4 q,V v){auto t=scale(cross(vec(q),v),2);return add(v,add(scale(t,q.w),cross(vec(q),t)));}
long double error(V a,V b){auto d=sub(a,b);return std::max({std::abs(d[0]),std::abs(d[1]),std::abs(d[2])});}
struct Input {mr_float4 rotation,point,radii,normal; mr_u32 kind; mr_u32 padding[3];};
std::vector<Input> inputs(){
    std::vector<Input> all;
    for(float half: {0.0f,1e-9f,-1e-9f,1e-8f,-1e-8f,1e-7f,-1e-7f,1e-6f,-1e-6f,.03f,-.13f})
        for(unsigned kind=0;kind<3;++kind) {
            const float qw=std::sqrt(1.0f-half*half);
            all.push_back({{0,half,0,qw},{.07654321f,-.01782345f,-.146789f,0},
                {.055f,.031f,.087f,0},{0,0,1,kind==1?.024f:0},kind,{0,0,0}});
        }
    return all;
}
V support(const Input& v){
    V result=rotate(v.rotation,vec(v.point));
    if(v.kind!=2)return sub(result,scale(vec(v.normal),v.normal.w));
    const mr_float4 conjugate{-v.rotation.x,-v.rotation.y,-v.rotation.z,v.rotation.w};
    V direction=rotate(conjugate,vec(v.normal));
    V scaled{v.radii.x*direction[0],v.radii.y*direction[1],v.radii.z*direction[2]};
    long double norm=std::sqrt(dot(scaled,scaled));
    V local{v.radii.x*scaled[0]/norm,v.radii.y*scaled[1]/norm,v.radii.z*scaled[2]/norm};
    return sub(result,rotate(v.rotation,local));
}
MRCompensatedPositionGPU chain(const Input& v){
    auto p=mrCompensatedVector({0,0,0,0});
    for(unsigned i=0;i<24;++i){
        const mr_float4 a{.03125f+.0001220703125f*float(i),-.012f,-.083f,0};
        const mr_float4 b{-.007f,.006f,.018f,0};
        p=mrCompensatedVectorAdd(p,mrCompensatedVectorAdd(mrCompensatedQuaternionRotate(v.rotation,a),
            mrCompensatedVectorNegate(mrCompensatedQuaternionRotate(v.rotation,b))));
    }
    return p;
}
V chainOracle(const Input& v){V p{};for(unsigned i=0;i<24;++i){
    mr_float4 a{.03125f+.0001220703125f*float(i),-.012f,-.083f,0},b{-.007f,.006f,.018f,0};
    p=add(p,sub(rotate(v.rotation,vec(a)),rotate(v.rotation,vec(b))));}return p;}
id<MTLBuffer> buffer(id<MTLDevice> d,const void* data,std::size_t size){
    id<MTLBuffer>b=data?[d newBufferWithBytes:data length:std::max<std::size_t>(16,size) options:MTLResourceStorageModeShared]:
        [d newBufferWithLength:std::max<std::size_t>(16,size) options:MTLResourceStorageModeShared];
    require(b!=nil,"buffer allocation failed");return b;
}
id<MTLComputePipelineState> pipeline(id<MTLDevice>d,id<MTLLibrary>l,const char*name){NSError*e=nil;
    auto f=[l newFunctionWithName:[NSString stringWithUTF8String:name]];require(f!=nil,"missing production geometry kernel");
    auto p=[d newComputePipelineStateWithFunction:f error:&e];require(p!=nil,"geometry pipeline failed");return p;}
std::size_t scratch(unsigned bodies,unsigned nv){auto a=[](std::size_t n){return(n+15)&~std::size_t(15);};std::size_t n=0;
    for(auto s:{32u*bodies,16u*bodies,32u*bodies,16u*bodies,4u*bodies,4u*bodies,bodies,4u*bodies*((nv+31u)/32u)})n=a(n)+s;return a(n);}
struct FKOutput{std::vector<MRArticulatedBodyPoseGPU> poses;std::vector<mr_float4> low;MRArticulatedPointWorldGPU point;mr_float4 pointLow;std::array<float,18> jacobian;};
FKOutput fk(id<MTLDevice>d,id<MTLCommandQueue>queue,id<MTLComputePipelineState>p,float half,float origin){
    constexpr unsigned N=25;MRWorldGPU world{};world.abiVersion=MR_ENGINE_ABI_VERSION;world.bodyCount=N;world.articulationCount=1;world.jointCount=N-1;world.nq=7;world.nv=6;
    MRArticulationGPU a{};a.rootType=MR_ROOT_FLOATING;a.bodyCount=N;a.jointCount=N-1;a.nq=7;a.nv=6;
    std::array<MRDofPropertiesGPU,6>dofs{};for(unsigned i=0;i<6;++i){dofs[i].jointIndex=MR_INVALID_INDEX;dofs[i].qIndex=i<3?i:MR_INVALID_INDEX;dofs[i].vIndex=i;dofs[i].localDof=i;dofs[i].flags=MR_DOF_FLAG_ROOT;}
    std::vector<MRJointDescriptorGPU>joints(N-1);std::vector<MRBodyPropertiesGPU>bodies(N);
    for(unsigned i=0;i<N;++i){auto&b=bodies[i];b.parentBody=i?i-1:MR_INVALID_INDEX;b.inboundJoint=i?i-1:MR_INVALID_INDEX;b.motionType=MR_MOTION_DYNAMIC;b.massAndInverseMass={1,1,0,0};b.inertiaRow0=b.inverseInertiaRow0={1,0,0,0};b.inertiaRow1=b.inverseInertiaRow1={0,1,0,0};b.inertiaRow2=b.inverseInertiaRow2={0,0,1,0};
        if(i){auto&j=joints[i-1];j.parentBody=i-1;j.childBody=i;j.jointType=MR_JOINT_FIXED;j.qOffset=7;j.vOffset=6;j.parentRotation=j.childRotation={0,0,0,1};j.parentAnchor={.03125f+.0001220703125f*float(i-1),-.012f,-.083f,0};j.childAnchor={-.007f,.006f,.018f,0};}}
    std::array<float,7>q{origin,-origin,1.94496f,0,half,0,1};
    MRCompensatedRootTranslationGPU root=mrCompensatedTranslationFromProjection({q[0],q[1],q[2],0});root.displacement={1e-8f,-2e-8f,3e-8f,0};
    auto projection=mrCompensatedTranslationProjection(root);q[0]=projection.x;q[1]=projection.y;q[2]=projection.z;
    MRArticulatedPointImpulseGPU point{};point.bodyIndex=N-1;point.flags=MR_ARTICULATED_POINT_ELLIPSOID_SUPPORT;point.localPoint={.07654321f,-.01782345f,-.146789f,0};point.supportPlaneNormalAndRadius={0,0,1,0};point.supportRadii={.055f,.031f,.087f,0};point.supportOrientation={0,0,0,1};
    MRArticulatedOperatorDispatchGPU dispatch{};dispatch.environmentCount=1;dispatch.pointCount=1;dispatch.flags=MR_ARTICULATED_OPERATOR_KINEMATICS_JACOBIANS_ONLY|MR_ARTICULATED_OPERATOR_COMPENSATED_TRANSLATION;dispatch.qStride=7;dispatch.pointStride=1;dispatch.bodyPoseStride=N;dispatch.pointWorldStride=1;dispatch.pointJacobianStride=18;dispatch.generalizedStride=6;
    std::array<float,24>zero{};MROpenSimSpatialTransformGPU program{};
    id<MTLBuffer>b[20]={buffer(d,&world,sizeof(world)),buffer(d,&a,sizeof(a)),buffer(d,joints.data(),joints.size()*sizeof(joints[0])),buffer(d,dofs.data(),sizeof(dofs)),buffer(d,bodies.data(),bodies.size()*sizeof(bodies[0])),buffer(d,&dispatch,sizeof(dispatch)),buffer(d,q.data(),sizeof(q)),buffer(d,&point,sizeof(point)),buffer(d,nullptr,N*sizeof(MRArticulatedBodyPoseGPU)),buffer(d,nullptr,sizeof(MRArticulatedPointWorldGPU)),buffer(d,zero.data(),sizeof(zero)),buffer(d,nullptr,18*sizeof(float)),buffer(d,nullptr,24*sizeof(float)),buffer(d,nullptr,24*sizeof(float)),buffer(d,nullptr,sizeof(MRArticulatedOperatorStatusGPU)),buffer(d,&program,sizeof(program)),buffer(d,zero.data(),sizeof(zero)),buffer(d,&root,sizeof(root)),buffer(d,nullptr,N*sizeof(mr_float4)),buffer(d,nullptr,sizeof(mr_float4))};
    auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:p];for(unsigned i=0;i<20;++i)[enc setBuffer:b[i] offset:0 atIndex:i];[enc setThreadgroupMemoryLength:scratch(N,6) atIndex:0];[enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,"production FK command failed");
    const auto*status=static_cast<const MRArticulatedOperatorStatusGPU*>(b[14].contents);if(status->code)std::fprintf(stderr,"FK status %u index %u\n",status->code,status->failingIndex);require(status->code==0,"production FK rejected manufactured chain");
    FKOutput out;out.poses.assign(static_cast<const MRArticulatedBodyPoseGPU*>(b[8].contents),static_cast<const MRArticulatedBodyPoseGPU*>(b[8].contents)+N);out.low.assign(static_cast<const mr_float4*>(b[18].contents),static_cast<const mr_float4*>(b[18].contents)+N);std::memcpy(&out.point,b[9].contents,sizeof(out.point));std::memcpy(&out.pointLow,b[19].contents,sizeof(out.pointLow));std::memcpy(out.jacobian.data(),b[11].contents,sizeof(out.jacobian));return out;
}
}
int main(int argc,char**argv){try{@autoreleasepool{
    bool cpu=argc==2&&std::strcmp(argv[1],"--cpu")==0;require(cpu||argc==2,"usage: geometry-check --cpu|METALLIB");
    auto cases=inputs();std::vector<MRCompensatedPositionGPU>expected;long double maxArithmetic=0,maxGeometry=0;unsigned checks=0;
    for(float a:{1e-15f,1e-7f,.031f,.5f,1.0f,37.0f,1e15f})for(float b:{1e-9f,.2f,1.0f,7.0f,1e9f}){
        MRCompensatedScalar x{a,a*1e-8f},y{b,-b*1e-8f};auto div=mrCompensatedDivide(x,y);long double exactDiv=((long double)x.high+x.low)/((long double)y.high+y.low);auto sq=mrCompensatedSqrt(x);long double exactSqrt=std::sqrt((long double)x.high+x.low);maxArithmetic=std::max({maxArithmetic,std::abs(((long double)div.high+div.low)/exactDiv-1),std::abs(((long double)sq.high+sq.low)/exactSqrt-1)});checks+=2;}
    for(const auto&v:cases){auto rotated=mrCompensatedQuaternionRotate(v.rotation,v.point);auto offset=mrCompensatedSupportOffset(v.rotation,v.point,{0,0,0,1},v.radii,v.normal,v.kind);auto end=chain(v);maxGeometry=std::max({maxGeometry,error(exact(rotated),rotate(v.rotation,vec(v.point))),error(exact(offset),support(v)),error(exact(end),chainOracle(v))});expected.push_back(rotated);expected.push_back(offset);expected.push_back(end);checks+=3;}
    require(maxArithmetic<5e-13L,"paired division/sqrt exceed independent long-double bound");require(maxGeometry<2e-12L,"paired geometry exceeds independent long-double bound");
    if(cpu){std::printf("{\"passed\":true,\"mode\":\"cpu\",\"checks\":%u,\"max_arithmetic_relative_error\":%.17g,\"max_geometry_error_m\":%.17g}\n",checks,double(maxArithmetic),double(maxGeometry));return 0;}
    auto d=MTLCreateSystemDefaultDevice();require(d!=nil,"no Metal device");NSError*e=nil;auto l=[d newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&e];require(l!=nil,"no library");auto queue=[d newCommandQueue];auto p=pipeline(d,l,"mr_compensated_geometry_check");auto in=buffer(d,cases.data(),cases.size()*sizeof(cases[0]));auto out=buffer(d,nullptr,expected.size()*sizeof(expected[0]));unsigned n=unsigned(cases.size());auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:p];[enc setBuffer:in offset:0 atIndex:0];[enc setBuffer:out offset:0 atIndex:1];[enc setBytes:&n length:sizeof(n) atIndex:2];[enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,"geometry helper command failed");
    // Long-double oracle is the accuracy authority; host equality is a separate arithmetic-owner check.
    require(std::memcmp(out.contents,expected.data(),expected.size()*sizeof(expected[0]))==0,"CPU/Metal derived geometry differs");
    auto fkp=pipeline(d,l,"mr_articulated_operator_compensated");const float half=1e-7f,h=1e-8f;auto center=fk(d,queue,fkp,half,0);auto left=fk(d,queue,fkp,half-h,0);auto right=fk(d,queue,fkp,half+h,0);auto translated=fk(d,queue,fkp,half,4096);
    Input v=cases.front();v.rotation={0,half,0,1};v.kind=2;v.normal={0,0,1,0};V translation{1e-8f,-2e-8f,(long double)1.94496f+3e-8f};V expectedPoint=add(translation,add(chainOracle(v),support(v)));V measured=add(vec(center.point.position),vec(center.pointLow));long double fkError=error(measured,expectedPoint);require(fkError<2e-11L,"production long-chain geometry exceeds FP64 oracle");
    V shifted=add(vec(translated.point.position),vec(translated.pointLow));shifted=sub(shifted,{4096,-4096,0});require(error(shifted,measured)<2e-10L,"production geometry depends on large world origin");
    // Half-quaternion y changes are half the angular increment at this tiny angle.
    V delta=sub(add(vec(right.point.position),vec(right.pointLow)),add(vec(left.point.position),vec(left.pointLow)));long double fd=delta[2]/(2*((long double)(half+h)-(half-h)));long double j=center.jacobian[12+4];long double jError=std::abs(fd-j);require(jError<2e-4L,"production support gap/J secant disagrees at tiny rotation");
    std::printf("{\"passed\":true,\"mode\":\"metal\",\"cpu_checks\":%u,\"max_geometry_error_m\":%.17g,\"production_fk_error_m\":%.17g,\"production_secant_error\":%.17g}\n",checks,double(maxGeometry),double(fkError),double(jError));return 0;
}}catch(const std::exception&e){std::fprintf(stderr,"geometry check failed: %s\n",e.what());return 1;}}

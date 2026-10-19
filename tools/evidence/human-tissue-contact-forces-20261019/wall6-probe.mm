#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "NumiHumanVesselWallCoupling.hpp"
#include "numi/matter/matter.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <limits>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
namespace aw=numi::human::aortic_wall; namespace vw=numi::human::vessel_wall_coupling; namespace nm=numi::matter;
using V=aw::Vec; using F=aw::Face;
static void ck(bool x,const std::string&s){if(!x)throw std::runtime_error(s);}
static V add(V a,const V&b){for(int i=0;i<3;i++)a[i]+=b[i];return a;} static V sub(V a,const V&b){for(int i=0;i<3;i++)a[i]-=b[i];return a;}
static V mul(V a,double s){for(auto&x:a)x*=s;return a;} static double dot(const V&a,const V&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static V cross(const V&a,const V&b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
static double norm(const V&a){return std::sqrt(dot(a,a));} static V unit(V a){double n=norm(a);ck(n>1e-12,"degenerate vector");return mul(a,1/n);}
static double vol6(const V&a,const V&b,const V&c,const V&d){return dot(sub(b,a),cross(sub(c,a),sub(d,a)));}
struct K{std::array<uint32_t,3>v;bool operator<(const K&o)const{return v<o.v;}};struct FRec{int n=0;F f{};};
static std::pair<F,V> target(const aw::Built&w,const aw::SectionProof&p){
 std::set<uint32_t>q(w.object.femContactNodes.begin(),w.object.femContactNodes.end());std::map<K,FRec>m;
 for(auto&t:w.object.tetrahedra){auto x=t.nodes;std::array<F,4>fs{{{x[1],x[2],x[3]},{x[0],x[3],x[2]},{x[0],x[1],x[3]},{x[0],x[2],x[1]}}};for(auto f:fs){K k{f};std::sort(k.v.begin(),k.v.end());m[k].n++;m[k].f=f;}}
 V c{};for(auto x:p.orderedExternalContour)c=add(c,mul(x,1.0/p.orderedExternalContour.size()));V axis=unit(p.planeNormal);double mid=.5*w.segmentLengthM,best=-1e9;F bf{};V bn{};
 for(auto&[k,r]:m)if(r.n==1&&q.contains(k.v[0])&&q.contains(k.v[1])&&q.contains(k.v[2])){V a=w.object.femNodes[r.f[0]],b=w.object.femNodes[r.f[1]],d=w.object.femNodes[r.f[2]],n=unit(cross(sub(b,a),sub(d,a))),fc=mul(add(add(a,b),d),1./3);V rad=sub(fc,c);double ax=dot(rad,axis);rad=sub(rad,mul(axis,ax));if(dot(n,rad)<0)n=mul(n,-1);double s=dot(n,unit(rad))-.2*std::abs(ax-mid)/w.segmentLengthM;if(s>best){best=s;bf=r.f;bn=n;}}
 ck(best>0,"no outer lateral wall target face");return {bf,bn};
}
static bool same(const auto&a,const auto&b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0]))==0);}
static void failLate(id<MTLDevice>d,id<MTLCommandBuffer>c,id<MTLBuffer>dst){MRMetalWorldStatusGPU s{};s.code=MR_STEP_DID_NOT_CONVERGE;id<MTLBuffer>src=[d newBufferWithBytes:&s length:sizeof(s) options:MTLResourceStorageModeShared];id<MTLBlitCommandEncoder>b=[c blitCommandEncoder];[b copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:sizeof(s)];[b endEncoding];[c addCompletedHandler:^(id<MTLCommandBuffer>x){(void)x;(void)src;}];}
static void transact(nm::Runtime&r,id<MTLDevice>d,id<MTLCommandQueue>q,id<MTLBuffer>ws,id<MTLBuffer>loads,id<MTLBuffer>rb,uint32_t count,uint32_t step,bool reject){
 auto*s=(MRMetalWorldStatusGPU*)ws.contents;*s={};s->code=MR_STEP_SUCCESS;id<MTLCommandBuffer>c=[q commandBuffer];nm::EncodeRequest a{};a.commandBuffer=(__bridge void*)c;a.environmentStatuses=(__bridge void*)ws;a.femExternalForces=(__bridge void*)loads;a.femExternalForceCount=count;a.controlStep=step;a.physicsSubstep=0;a.physicsSubsteps=1;a.timestepSeconds=r.timestepSeconds();a.phase=nm::EncodePhase::preDynamics;auto e=r.encode(a);ck(e.encoded,"pre: "+e.message);
 id<MTLBuffer>f=(__bridge id<MTLBuffer>)r.femConstraintContactForceBuffer();id<MTLBlitCommandEncoder>b=[c blitCommandEncoder];[b copyFromBuffer:f sourceOffset:0 toBuffer:rb destinationOffset:0 size:rb.length];[b endEncoding];if(reject)failLate(d,c,ws);a.phase=nm::EncodePhase::postCommit;e=r.encode(a);ck(e.encoded,"post: "+e.message);[c commit];[c waitUntilCompleted];ck(c.status==MTLCommandBufferStatusCompleted,"Metal command failed");
}

static void emitSnapshot(const char*stage,uint32_t step,const auto&snap){
 std::cerr<<std::setprecision(std::numeric_limits<float>::max_digits10)
          <<"{\"snapshot\":\""<<stage<<"\",\"step\":"<<step
          <<",\"status_count\":"<<snap.statuses.size();
 if(!snap.statuses.empty()){
  const auto&st=snap.statuses[0];
  std::cerr<<",\"status\":{\"code\":"<<st.code<<",\"object\":"<<st.objectIndex<<",\"failing_index\":"<<st.failingIndex<<",\"diagnostics\":["<<st.diagnostics.x<<","<<st.diagnostics.y<<","<<st.diagnostics.z<<","<<st.diagnostics.w<<"]}";
 }
 std::cerr<<",\"deformable_contact_failures\":[";bool first=true;
 for(size_t fi=0;fi<snap.deformableContactFailures.size();fi++){
  const auto&z=snap.deformableContactFailures[fi];if(z.attempt.x==0u)continue;
  if(!first)std::cerr<<",";first=false;
  std::cerr<<"{\"index\":"<<fi<<",\"attempt\":["<<z.attempt.x<<","<<z.attempt.y<<","<<z.attempt.z<<","<<z.attempt.w<<"],\"pair\":["<<z.primitivePair.x<<","<<z.primitivePair.y<<","<<z.primitivePair.z<<","<<z.primitivePair.w<<"],\"primitive_nodes_objects\":[";
  for(int p=0;p<2;p++){if(p)std::cerr<<",";const auto&q=z.primitives[p];std::cerr<<"["<<q.nodesAndObject.x<<","<<q.nodesAndObject.y<<","<<q.nodesAndObject.z<<","<<q.nodesAndObject.w<<"]";}
  std::cerr<<"],\"primitive_identity\":[";
  for(int p=0;p<2;p++){if(p)std::cerr<<",";const auto&q=z.primitives[p];std::cerr<<"["<<q.identity.x<<","<<q.identity.y<<","<<q.identity.z<<","<<q.identity.w<<"]";}
  std::cerr<<"],\"diagnostics\":["<<z.diagnostics.x<<","<<z.diagnostics.y<<","<<z.diagnostics.z<<","<<z.diagnostics.w<<"],\"parameters\":["<<z.parameters.x<<","<<z.parameters.y<<","<<z.parameters.z<<","<<z.parameters.w<<"],\"start\":[";
  for(int j=0;j<6;j++){if(j)std::cerr<<",";std::cerr<<"["<<z.start[j].x<<","<<z.start[j].y<<","<<z.start[j].z<<"]";}
  std::cerr<<"],\"finish\":[";
  for(int j=0;j<6;j++){if(j)std::cerr<<",";std::cerr<<"["<<z.finish[j].x<<","<<z.finish[j].y<<","<<z.finish[j].z<<"]";}
  std::cerr<<"]}";
 }
 std::cerr<<"]}\n";
}
static void run(bool approach,const std::filesystem::path&lib,bool pristineContinuation=false){
 constexpr double P=13300,dt=.00025; constexpr uint32_t horizon=5;
 aw::Config cfg;cfg.outerLateralContact=true;MRBodyStateGPU pose{};pose.orientation={0,0,0,1};pose.position={0,0,0,1};auto proof=vw::pinnedAbdominalAortaSectionProof();
 double re=std::sqrt(proof.externalAreaM2/M_PI),s=(re-cfg.wallThicknessM)/re,volume=proof.externalAreaM2*s*s*.08;auto w=aw::build(proof,pose,{0,0,0},0,pose,0,volume,P,cfg);w.objectIndex=0;
 for(auto&a:w.object.femHumanAttachments)w.object.femFixedNodes.push_back(a.node);w.object.femHumanAttachments.clear();
 auto pressure=aw::makeAnnularMechanicalWall(w,proof,P);for(size_t i=0;i<pressure.object.femNodes.size();i++)ck(norm(sub(pressure.object.femNodes[i],w.object.femNodes[i]))<1e-12,"pressure remap changed wall node order");
 std::vector<std::array<float,4>>wf(w.object.femNodes.size(),{0,0,0,0});std::copy(pressure.externalForces.begin(),pressure.externalForces.end(),wf.begin());auto [face,n]=target(w,proof);
 for(auto i:face)w.object.femFixedNodes.push_back(i);std::sort(w.object.femFixedNodes.begin(),w.object.femFixedNodes.end());w.object.femFixedNodes.erase(std::unique(w.object.femFixedNodes.begin(),w.object.femFixedNodes.end()),w.object.femFixedNodes.end());
 nm::WorldSource src;src.environmentCount=1;src.frameTimestep=dt;src.gravity={0,0,0};src.compactImmutableFEMSurfaces=true;src.materials.push_back(aw::makeMaterial(cfg));w.object.materialIndex=0;w.object.deformableContact=true;w.object.deformableSelfContact=false;w.object.twoWayCoupling=false;src.objects.push_back(w.object);
 nm::ObjectSource probe;probe.name="fixture/incoming-tet-not-anatomical";probe.representation=nm::Representation::fem;probe.materialIndex=0;probe.mixedFEM=false;probe.twoWayCoupling=false;probe.deformableContact=true;probe.deformableSelfContact=false;probe.characteristicLength=.003;
 double gap=approach?.0005:.05,speed=approach?.5:0;V a=w.object.femNodes[face[0]],b=w.object.femNodes[face[1]],c=w.object.femNodes[face[2]],center=mul(add(add(a,b),c),1./3);
 probe.femNodes={add(a,mul(n,gap)),add(b,mul(n,gap)),add(c,mul(n,gap)),add(center,mul(n,gap+.002))};std::array<uint32_t,4>t{0,1,2,3};if(vol6(probe.femNodes[0],probe.femNodes[1],probe.femNodes[2],probe.femNodes[3])<0)std::swap(t[1],t[2]);probe.tetrahedra.push_back({t});probe.femContactNodes={0,1,2,3};probe.femInitialVelocity=mul(n,-speed);src.objects.push_back(probe);
 nm::CompileOptions opt;opt.maximumRateExponent=0;auto cw=nm::compileWorld(src,opt);std::string ds;for(auto&x:cw.diagnostics)ds+=x.message+"; ";ck(cw.succeeded(),"compile: "+ds);auto audit=vw::auditOuterLateralContactSurfaceFaces(cw.world,w,w.object.femContactNodes);ck(audit.retainedWallFaceCount==384&&audit.excludedWallFaceCount>0,"wall face selection invalid");
 uint32_t wo=cw.world.objects[0].stateOffset,po=cw.world.objects[1].stateOffset,N=cw.world.dispatch.femNodeCount;std::vector<std::array<float,4>>loads(N,{0,0,0,0});for(size_t i=0;i<wf.size();i++)loads[wo+i]=wf[i];
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>q=[d newCommandQueue];id<MTLBuffer>lb=[d newBufferWithBytes:loads.data() length:loads.size()*sizeof(loads[0]) options:MTLResourceStorageModeShared],rb=[d newBufferWithLength:N*sizeof(mr_float4) options:MTLResourceStorageModeShared],ws=[d newBufferWithLength:sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];ck(d&&q&&lb&&rb&&ws,"Metal buffer init failed");
 nm::Runtime rt;nm::RuntimeConfiguration rc;rc.metallib=lib;rc.environmentCount=1;rc.captureEvents=true;rc.captureDiagnostics=true;rc.automaticIdentification=false;rc.adaptiveTransfer=false;auto init=rt.initialize(cw.world,rc);ck(init.encoded&&rt.valid(),"initialize: "+init.message);ck(rt.setNewtonIterationBudget(12)&&rt.setFGMRESIterationBudget(64),"could not set accepted control solver budgets");auto before=rt.snapshot();ck(before.available&&before.femNodes.size()==N,"initial snapshot");
 std::vector<F> cavityFaces;for(auto&f:w.cavity.faces)cavityFaces.push_back(f.nodes);std::vector<V>x0;for(auto&x:before.femNodes){auto p=x.positionAndMass;x0.push_back({p.x,p.y,p.z});}double cv0=aw::cavityVolume(x0,cavityFaces);
 V contactImpulse{};double maxContactForce=0;uint32_t activeMax=0;double maxbar=0;auto accepted=before;
 for(uint32_t step=0;step<horizon;step++){
  transact(rt,d,q,ws,lb,rb,N,step,false);auto frame=rt.snapshot();
  emitSnapshot("accepted",step,frame);
  if(frame.statuses.size()!=1||frame.statuses[0].code!=NM_STATUS_SUCCESS){if(!frame.statuses.empty()){const auto&st=frame.statuses[0];std::cerr<<std::setprecision(std::numeric_limits<float>::max_digits10)<<"FRAMEFAIL step="<<step<<" code="<<st.code<<" object="<<st.objectIndex<<" failing="<<st.failingIndex<<" diagnostics="<<st.diagnostics.x<<","<<st.diagnostics.y<<","<<st.diagnostics.z<<","<<st.diagnostics.w<<"\n";}}ck(frame.statuses.size()==1&&frame.statuses[0].code==NM_STATUS_SUCCESS,"accepted frame status failed at step "+std::to_string(step));
  auto*f=(mr_float4*)rb.contents;V thisFrame{};for(auto i:w.object.femFixedNodes){auto z=f[wo+i];thisFrame=add(thisFrame,V{z.x,z.y,z.z});}contactImpulse=add(contactImpulse,mul(thisFrame,dt));maxContactForce=std::max(maxContactForce,norm(thisFrame));uint32_t active=0;for(auto&h:frame.deformableContactHistories)if(h.normalAndBarrier.w>0){active++;maxbar=std::max(maxbar,double(h.normalAndBarrier.w));}activeMax=std::max(activeMax,active);std::cerr<<std::setprecision(std::numeric_limits<float>::max_digits10)<<"FRAME case="<<(approach?"approach":"control")<<" step="<<step<<" status="<<frame.statuses[0].code<<" history_count="<<frame.deformableContactHistories.size()<<" active_barrier_rows="<<active<<" fixed_contact_force_n="<<norm(thisFrame)<<"\n";accepted=std::move(frame);
 }
 V p0{},p1{};double mass=0;for(uint32_t i=0;i<4;i++){auto a0=before.femNodes[po+i],a1=accepted.femNodes[po+i];double m=a0.positionAndMass.w;mass+=m;p0=add(p0,mul(V{a0.velocityAndInverseMass.x,a0.velocityAndInverseMass.y,a0.velocityAndInverseMass.z},m));p1=add(p1,mul(V{a1.velocityAndInverseMass.x,a1.velocityAndInverseMass.y,a1.velocityAndInverseMass.z},m));}V dp=sub(p1,p0);
 double move=0;std::vector<V>x1;for(size_t i=0;i<accepted.femNodes.size();i++){auto x=accepted.femNodes[i].positionAndMass,y=before.femNodes[i].positionAndMass;x1.push_back({x.x,x.y,x.z});if(i>=wo&&i<wo+w.object.femNodes.size())move=std::max(move,norm(V{x.x-y.x,x.y-y.y,x.z-y.z}));}double cv1=aw::cavityVolume(x1,cavityFaces);
 auto checkpoint=accepted;
 if(pristineContinuation){transact(rt,d,q,ws,lb,rb,N,horizon,false);auto pristine=rt.snapshot();emitSnapshot("pristine_next_step",horizon,pristine);bool ok=pristine.statuses.size()==1&&pristine.statuses[0].code==NM_STATUS_SUCCESS;std::cout<<"{\"case\":\"pristine_next_step\",\"status_count\":"<<pristine.statuses.size()<<",\"status_code\":"<<(pristine.statuses.empty()?-1:pristine.statuses[0].code)<<",\"accepted\":"<<(ok?"true":"false")<<"}\n";return;}
 transact(rt,d,q,ws,lb,rb,N,horizon,true);auto rejected=rt.snapshot();emitSnapshot("injected_reject",horizon,rejected);bool rollback=same(rejected.femNodes,checkpoint.femNodes)&&same(rejected.femFields,checkpoint.femFields)&&same(rejected.femMaterialState,checkpoint.femMaterialState)&&same(rejected.deformableContactHistories,checkpoint.deformableContactHistories);transact(rt,d,q,ws,lb,rb,N,horizon,false);auto retry=rt.snapshot();emitSnapshot("retry",horizon,retry);bool retryOK=retry.statuses.size()==1&&retry.statuses[0].code==NM_STATUS_SUCCESS;
 std::cout<<"{\"case\":\""<<(approach?"approach":"no_touch_control")<<"\",\"positive_gap_m\":"<<gap<<",\"dt_s\":"<<dt<<",\"accepted_horizon_steps\":"<<horizon<<",\"approach_speed_m_s\":"<<speed<<",\"nodes\":"<<N<<",\"wall_nodes\":"<<w.object.femNodes.size()<<",\"wall_tets\":"<<w.object.tetrahedra.size()<<",\"probe_mass_kg\":"<<mass<<",\"active_barrier_rows_max\":"<<activeMax<<",\"max_barrier_impulse_Ns\":"<<maxbar<<",\"wall_fixed_node_contact_impulse_Ns\":"<<norm(contactImpulse)<<",\"wall_fixed_node_peak_contact_force_N\":"<<maxContactForce<<",\"probe_delta_momentum_Ns\":"<<norm(dp)<<",\"wall_max_displacement_m\":"<<move<<",\"cavity_volume_initial_m3\":"<<cv0<<",\"cavity_volume_final_m3\":"<<cv1<<",\"rollback_exact\":"<<(rollback?"true":"false")<<",\"retry_accepted\":"<<(retryOK?"true":"false")<<",\"outer_faces_retained\":"<<audit.retainedWallFaceCount<<",\"wall_faces_excluded\":"<<audit.excludedWallFaceCount<<",\"momentum_balance_claim\":false}\n";
 ck(rollback&&retryOK,"rejection/retry rollback failed");if(approach){ck(norm(contactImpulse)>1e-10&&norm(dp)>1e-10&&activeMax>0,"no measurable accepted contact response");}else ck(norm(contactImpulse)<1e-10&&activeMax==0,"control unexpectedly contacted");
}
int main(int argc,char**argv){@autoreleasepool{try{ck(argc==2,"usage: probe <MatterSource.metal>");std::filesystem::path p=argv[1];run(false,p);run(true,p,true);run(true,p,false);std::cout<<"{\"result\":\"PASS\",\"scope\":\"stable-ID-9 registered contour reduced wall, prescribed pressure, synthetic local FEM contact; not CVSim hydraulic or anatomical-neighbor qualification\"}\n";}catch(const std::exception&e){std::cerr<<"{\"result\":\"FAIL\",\"error\":\""<<e.what()<<"\"}\n";return 1;}}return 0;}
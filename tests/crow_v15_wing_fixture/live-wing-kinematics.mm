#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "v15_fixture.hpp"
#include "metalrobo/CrowArticulatedWing.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/ArticulatedDynamics.hpp"
#include <iomanip>

struct Observer { __strong id<MTLBuffer> destination; std::size_t expectedBytes; unsigned encoded=0; };
static bool encode(void* opaque, const metalrobo::MetalWorldDevicePhysicsPass& pass) {
 auto& o=*static_cast<Observer*>(opaque);
 if(pass.phase!=metalrobo::MetalWorldDevicePhysicsPhase::preDynamics)return true;
 id<MTLCommandBuffer> cb=(__bridge id<MTLCommandBuffer>)pass.commandBuffer;
 id<MTLBuffer> src=(__bridge id<MTLBuffer>)pass.currentBodies;
 if(!cb||!src||src.length<o.expectedBytes||o.destination.length<o.expectedBytes||pass.bodyStateStride!=41||pass.environmentCount!=3)return false;
 id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];if(!blit)return false;
 [blit copyFromBuffer:src sourceOffset:0 toBuffer:o.destination destinationOffset:0 size:o.expectedBytes];[blit endEncoding];++o.encoded;return true;
}
static void abortObserver(void*,void*) {}
static void four(const mr_float4& v){std::cout<<'['<<v.x<<','<<v.y<<','<<v.z<<','<<v.w<<']';}
template<class T,std::size_t N>static void array(const std::array<T,N>&v){std::cout<<'[';for(size_t i=0;i<N;++i){if(i)std::cout<<',';std::cout<<v[i];}std::cout<<']';}
int main(int argc,char**argv){@autoreleasepool{try{
 if(argc!=2)throw std::runtime_error("expected metallib");
 auto p=metalrobo::makeCrowArticulatedWingMechanicsFixture(crowHipRollModel());auto&m=p.mechanics;
 m.world.gravityAndTimestep.x=m.world.gravityAndTimestep.y=m.world.gravityAndTimestep.z=0;
 metalrobo::CompiledWorld world;auto compiled=metalrobo::compileMetalWorld(m,0,world);if(!compiled.succeeded())throw std::runtime_error(compiled.message);
 auto&a=m.articulations[0];std::vector<float> qs,vs,efforts(3*a.nv,0);
 for(int lane=0;lane<3;++lane){qs.insert(qs.end(),m.defaultQ.begin(),m.defaultQ.end());vs.insert(vs.end(),m.defaultV.begin(),m.defaultV.end());}
 auto coordinate=[&](const char* name){auto it=std::find(m.dofNames.begin(),m.dofNames.end(),name);if(it==m.dofNames.end())throw std::runtime_error("missing dof");return m.dofs.at(it-m.dofNames.begin());};
 auto le=coordinate("crow_left_elbow_flex"),lw=coordinate("crow_left_wrist_flex"),re=coordinate("crow_right_elbow_flex"),rw=coordinate("crow_right_wrist_flex");
 vs[a.nv+le.vIndex]=1;qs[2*a.nq+le.qIndex]=.9f;qs[2*a.nq+lw.qIndex]=.4f;qs[2*a.nq+re.qIndex]=.6f;qs[2*a.nq+rw.qIndex]=.8f;
 qs[2*a.nq+5]=.149438132f;qs[2*a.nq+6]=.988771078f;
 vs[2*a.nv+0]=.3f;vs[2*a.nv+1]=-.2f;vs[2*a.nv+2]=.1f;vs[2*a.nv+3]=.3f;vs[2*a.nv+4]=-.4f;vs[2*a.nv+5]=.2f;
 vs[2*a.nv+le.vIndex]=.7f;vs[2*a.nv+lw.vIndex]=-.35f;vs[2*a.nv+re.vIndex]=-.55f;vs[2*a.nv+rw.vIndex]=.45f;
 id<MTLDevice> device=MTLCreateSystemDefaultDevice();if(!device)throw std::runtime_error("no Metal device");
 Observer obs{};obs.expectedBytes=3*m.bodies.size()*sizeof(MRBodyStateGPU);obs.destination=[device newBufferWithLength:obs.expectedBytes options:MTLResourceStorageModeShared];if(!obs.destination)throw std::runtime_error("no capture buffer");memset(obs.destination.contents,0,obs.expectedBytes);
 metalrobo::MetalWorldBatch b;b.environmentCount=3;b.controlStepCount=1;b.initialQ=qs;b.initialV=vs;b.efforts=efforts;
 metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=.0001f;cfg.physicsSubsteps=1;cfg.solverMode=metalrobo::MetalWorldSolverMode::freeMotionABA;cfg.applyBodyDamping=false;
 cfg.devicePhysicsProgram.context=&obs;cfg.devicePhysicsProgram.encode=encode;cfg.devicePhysicsProgram.abort=abortObserver;cfg.devicePhysicsProgram.fingerprint=0x9c480f4e77b20d13ULL;
 metalrobo::MetalWorldConfig settings;settings.metallibPath=argv[1];settings.preferParallelABA=false;
 metalrobo::MetalWorldContext context(settings);metalrobo::MetalWorldResult out;auto diag=context.run(world,b,cfg,out);
 std::cout<<std::setprecision(12)<<"{\"compiled_fingerprint\":"<<compiled.fingerprint<<",\"gpu_ok\":"<<(diag.succeeded()?"true":"false")<<",\"dispatched\":"<<(diag.dispatched?"true":"false")<<",\"published\":"<<(diag.published?"true":"false")<<",\"steps\":"<<diag.successfulStepCount<<",\"failed\":"<<diag.failedStepCount<<",\"observer_encodes\":"<<obs.encoded<<",\"message\":"<<std::quoted(diag.message)<<",\"lanes\":[";
 bool ok=diag.succeeded()&&diag.dispatched&&diag.published&&diag.successfulStepCount==3&&diag.failedStepCount==0&&obs.encoded==1;
 const auto* capture=static_cast<const MRBodyStateGPU*>(obs.destination.contents);
 for(int lane=0;lane<3;++lane){if(lane)std::cout<<',';std::vector<double> q(qs.begin()+lane*a.nq,qs.begin()+(lane+1)*a.nq),v(vs.begin()+lane*a.nv,vs.begin()+(lane+1)*a.nv);std::vector<metalrobo::ArticulatedBodyKinematics> ref(m.bodies.size());auto d=metalrobo::computeArticulatedBodyKinematics(m,0,q,v,ref);ok=ok&&d.succeeded();
  std::cout<<"{\"lane\":"<<lane<<",\"cpu_ok\":"<<(d.succeeded()?"true":"false")<<",\"q\":[";for(size_t i=0;i<q.size();++i){if(i)std::cout<<',';std::cout<<q[i];}std::cout<<"],\"v\":[";for(size_t i=0;i<v.size();++i){if(i)std::cout<<',';std::cout<<v[i];}std::cout<<"],\"bodies\":[";
  for(size_t i=0;i<ref.size();++i){if(i)std::cout<<',';auto&r=ref[i];auto&g=capture[lane*ref.size()+i];double gp[3]={g.position.x,g.position.y,g.position.z},gv[3]={g.linearVelocityAndInverseMass.x,g.linearVelocityAndInverseMass.y,g.linearVelocityAndInverseMass.z},gw[3]={g.angularVelocity.x,g.angularVelocity.y,g.angularVelocity.z},gq[4]={g.orientation.x,g.orientation.y,g.orientation.z,g.orientation.w};double dot=0;for(int j=0;j<4;++j)dot+=gq[j]*r.orientation[j];for(int j=0;j<3;++j)ok=ok&&std::isfinite(gp[j])&&std::isfinite(gv[j])&&std::isfinite(gw[j])&&std::abs(gp[j]-r.centerOfMassPosition[j])<=1e-5&&std::abs(gv[j]-r.linearVelocity[j])<=1e-5&&std::abs(gw[j]-r.angularVelocity[j])<=1e-5;for(int j=0;j<4;++j)ok=ok&&std::isfinite(gq[j])&&std::abs(gq[j]-(dot<0?-1:1)*r.orientation[j])<=1e-5;
   std::cout<<"{\"name\":"<<std::quoted(m.bodyNames[i])<<",\"gpu_position\":";four(g.position);std::cout<<",\"gpu_orientation\":";four(g.orientation);std::cout<<",\"gpu_linear\":";four(g.linearVelocityAndInverseMass);std::cout<<",\"gpu_angular\":";four(g.angularVelocity);std::cout<<",\"cpu_position\":";array(r.centerOfMassPosition);std::cout<<",\"cpu_orientation\":";array(r.orientation);std::cout<<",\"cpu_linear\":";array(r.linearVelocity);std::cout<<",\"cpu_angular\":";array(r.angularVelocity);std::cout<<'}';
  }std::cout<<"]}";
 }
 auto speed=[](const MRBodyStateGPU&s){return std::sqrt(s.linearVelocityAndInverseMass.x*s.linearVelocityAndInverseMass.x+s.linearVelocityAndInverseMass.y*s.linearVelocityAndInverseMass.y+s.linearVelocityAndInverseMass.z*s.linearVelocityAndInverseMass.z);};
 ok=ok&&speed(capture[41+37])>0.01f&&speed(capture[41+40])<1e-6f&&speed(capture[82+37])>0.01f&&speed(capture[82+40])>0.01f;
 std::cout<<"],\"criterion_pass\":"<<(ok?"true":"false")<<"}\n";return ok?0:2;
}catch(const std::exception&e){std::cerr<<"exception: "<<e.what()<<'\n';return 1;}}}

#include "v15_fixture.hpp"
#include "metalrobo/CrowArticulatedWing.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/ArticulatedDynamics.hpp"
#include <iomanip>
template<class T>static void arr(std::span<const T> v){std::cout<<'[';for(size_t i=0;i<v.size();++i){if(i)std::cout<<',';std::cout<<v[i];}std::cout<<']';}
static void momentum(const metalrobo::ArticulatedInvariants& x){std::cout<<"{\"linear\":";arr<double>(x.linearMomentum);std::cout<<",\"angular\":";arr<double>(x.angularMomentum);std::cout<<",\"kinetic\":"<<x.kineticEnergy<<'}';}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("expected metallib");
 auto p=metalrobo::makeCrowArticulatedWingMechanicsFixture(crowHipRollModel());auto&m=p.mechanics;m.world.gravityAndTimestep.x=m.world.gravityAndTimestep.y=m.world.gravityAndTimestep.z=0;
 metalrobo::CompiledWorld world;auto c=metalrobo::compileMetalWorld(m,0,world);if(!c.succeeded())throw std::runtime_error(c.message);const auto&a=m.articulations[0];
 auto it=std::find(m.dofNames.begin(),m.dofNames.end(),"crow_left_elbow_flex");if(it==m.dofNames.end())throw std::runtime_error("missing elbow");unsigned index=m.dofs.at(it-m.dofNames.begin()).vIndex;
 std::vector<double> q(m.defaultQ.begin(),m.defaultQ.end()),v(m.defaultV.begin(),m.defaultV.end()),force(a.nv,0),cpu(a.nv);force[index]=0.001f;
 metalrobo::ArticulatedDynamicsConfig ref;ref.gravity={0,0,0};ref.applyBodyDamping=false;
 auto rd=metalrobo::computeArticulatedForwardDynamics(m,0,q,v,force,{},cpu,ref);
 metalrobo::ArticulatedInvariants before,after;auto pre=metalrobo::computeArticulatedInvariants(m,0,q,v,before,ref);
 std::vector<float> efforts(a.nv,0);efforts[index]=0.001f;metalrobo::MetalWorldBatch b;b.environmentCount=1;b.controlStepCount=1;b.initialQ=m.defaultQ;b.initialV=m.defaultV;b.efforts=efforts;
 metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=.0001f;cfg.physicsSubsteps=1;cfg.solverMode=metalrobo::MetalWorldSolverMode::freeMotionABA;cfg.applyBodyDamping=false;
 metalrobo::MetalWorldConfig setting;setting.metallibPath=argv[1];setting.preferParallelABA=false;metalrobo::MetalWorldContext context(setting);metalrobo::MetalWorldResult out;auto d=context.run(world,b,cfg,out);
 std::vector<double> fq(out.finalQ.begin(),out.finalQ.end()),fv(out.finalV.begin(),out.finalV.end());auto post=metalrobo::computeArticulatedInvariants(m,0,fq,fv,after,ref);
 std::cout<<std::setprecision(12)<<"{\"fingerprint\":"<<c.fingerprint<<",\"cpu_ok\":"<<(rd.succeeded()?"true":"false")<<",\"cpu_status\":"<<static_cast<unsigned>(rd.status)<<",\"pre_invariants_ok\":"<<(pre.succeeded()?"true":"false")<<",\"post_invariants_ok\":"<<(post.succeeded()?"true":"false")<<",\"gpu_ok\":"<<(d.succeeded()?"true":"false")<<",\"gpu_steps\":"<<d.successfulStepCount<<",\"gpu_failures\":"<<d.failedStepCount<<",\"gpu_message\":"<<std::quoted(d.message)<<",\"elbow_v\":"<<index<<",\"cpu_acceleration\":";arr<double>(cpu);std::cout<<",\"gpu_acceleration\":";arr<float>(out.accelerations);std::cout<<",\"before\":";momentum(before);std::cout<<",\"after\":";momentum(after);std::cout<<"}\n";
 bool ok=rd.succeeded()&&d.succeeded()&&pre.succeeded()&&post.succeeded()&&d.successfulStepCount==1&&d.failedStepCount==0&&out.accelerations.size()==cpu.size();
 if(ok){
  double maxRef=0,maxError=0;for(size_t i=0;i<cpu.size();++i){if(!std::isfinite(cpu[i])||!std::isfinite(out.accelerations[i]))ok=false;maxRef=std::max(maxRef,std::max(std::abs(cpu[i]),std::abs(double(out.accelerations[i]))));maxError=std::max(maxError,std::abs(cpu[i]-out.accelerations[i]));}
  ok=ok&&maxError<=.005*maxRef+.01;
  for(size_t i=0;i<3;++i)ok=ok&&std::abs(after.linearMomentum[i]-before.linearMomentum[i])<1e-7&&std::abs(after.angularMomentum[i]-before.angularMomentum[i])<1e-7;
 }
 return ok?0:2;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

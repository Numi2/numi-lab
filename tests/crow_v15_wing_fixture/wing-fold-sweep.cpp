#include "v15_fixture.hpp"
#include "metalrobo/CrowArticulatedWing.hpp"
#include "metalrobo/MetalWorld.hpp"
#include <iomanip>
int main(int argc,char**argv){
 try{
  if(argc!=2)throw std::runtime_error("expected metallib");
  auto p=metalrobo::makeCrowArticulatedWingMechanicsFixture(crowHipRollModel());auto&m=p.mechanics;
  m.world.gravityAndTimestep.x=m.world.gravityAndTimestep.y=m.world.gravityAndTimestep.z=0;
  metalrobo::CompiledWorld world;auto c=metalrobo::compileMetalWorld(m,0,world);if(!c.succeeded())throw std::runtime_error(c.message);
  std::array<unsigned,4> qi{};std::array<std::string,4> names{"crow_left_elbow_flex","crow_left_wrist_flex","crow_right_elbow_flex","crow_right_wrist_flex"};
  for(unsigned i=0;i<4;++i){auto it=std::find(m.dofNames.begin(),m.dofNames.end(),names[i]);if(it==m.dofNames.end())throw std::runtime_error("missing dof");qi[i]=m.dofs.at(it-m.dofNames.begin()).qIndex;}
  metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=.000125f;cfg.physicsSubsteps=1;cfg.solverMode=metalrobo::MetalWorldSolverMode::temporalCone;cfg.captureContactEvidence=true;
  metalrobo::MetalWorldConfig setting;setting.metallibPath=argv[1];setting.preferParallelABA=false;metalrobo::MetalWorldContext context(setting);
  std::vector<float> efforts(m.articulations[0].nv,0);std::cout<<std::setprecision(9)<<"{\"fingerprint\":"<<c.fingerprint<<",\"poses\":[";
  for(unsigned e=0;e<5;++e)for(unsigned wr=0;wr<5;++wr){
   auto q=m.defaultQ;q[qi[0]]=q[qi[2]]=float(e)*.6f;q[qi[1]]=q[qi[3]]=float(wr)*.525f;
   metalrobo::MetalWorldBatch b;b.environmentCount=1;b.controlStepCount=1;b.initialQ=q;b.initialV=m.defaultV;b.efforts=efforts;
   metalrobo::MetalWorldResult out;auto d=context.run(world,b,cfg,out);
   if(e||wr)std::cout<<',';
   std::cout<<"{\"elbow\":"<<q[qi[0]]<<",\"wrist\":"<<q[qi[1]]<<",\"succeeded\":"<<(d.succeeded()?"true":"false")<<",\"steps\":"<<d.successfulStepCount<<",\"failed_steps\":"<<d.failedStepCount<<",\"code\":"<<d.firstGPUStatusCode<<",\"message\":"<<std::quoted(d.message)<<",\"max_abs_v\":";
   float vmax=0;for(float x:out.finalV)vmax=std::max(vmax,std::abs(x));std::cout<<vmax<<",\"contacts\":[";
   auto& evidence=out.contactEvidence;size_t n=out.contactStatuses.empty()?0:out.contactStatuses[0].activeContacts;
   for(size_t i=0;i<n&&i<evidence.contacts.size();++i){auto&x=evidence.contacts[i];if(i)std::cout<<',';std::cout<<"{\"a\":"<<std::quoted(x.bodyA<m.bodyNames.size()?m.bodyNames[x.bodyA]:"UNKNOWN")<<",\"b\":"<<std::quoted(x.bodyB<m.bodyNames.size()?m.bodyNames[x.bodyB]:"UNKNOWN")<<",\"separation\":"<<x.pointAndSeparation.w<<",\"impulse\":"<<x.impulses.x<<'}';}
   std::cout<<"]}";
  }
  std::cout<<"]}\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

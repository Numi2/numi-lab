#include "v15_fixture.hpp"
#include "metalrobo/CrowArticulatedWing.hpp"
#include "metalrobo/MetalWorld.hpp"
#include <iomanip>
static void floats(const std::vector<float>& v) {std::cout<<'[';for(size_t i=0;i<v.size();++i){if(i)std::cout<<',';std::cout<<v[i];}std::cout<<']';}
int main(int argc,char**argv){
 try{
  if(argc!=2)throw std::runtime_error("expected metallib");
  std::cout<<std::setprecision(9)<<"{\"models\":[";
  for(int fixture=0;fixture<2;++fixture){
   auto p=crowHipRollModel();if(fixture)p=metalrobo::makeCrowArticulatedWingMechanicsFixture(std::move(p));auto& m=p.mechanics;
   m.world.gravityAndTimestep.x=m.world.gravityAndTimestep.y=m.world.gravityAndTimestep.z=0;
   metalrobo::CompiledWorld world;auto c=metalrobo::compileMetalWorld(m,0,world);
   if(fixture)std::cout<<',';
   std::cout<<"{\"id\":"<<std::quoted(p.id)<<",\"compiled\":"<<(c.succeeded()?"true":"false")<<",\"fingerprint\":"<<c.fingerprint<<",\"message\":"<<std::quoted(c.message);
   if(!c.succeeded()){std::cout<<'}';continue;}
   std::vector<float> efforts(m.articulations[0].nv,0);
   metalrobo::MetalWorldBatch b;b.environmentCount=1;b.controlStepCount=1;b.initialQ=m.defaultQ;b.initialV=m.defaultV;b.efforts=efforts;
   metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=.000125f;cfg.physicsSubsteps=1;cfg.solverMode=metalrobo::MetalWorldSolverMode::temporalCone;cfg.captureContactEvidence=true;
   metalrobo::MetalWorldConfig setting;setting.metallibPath=argv[1];setting.preferParallelABA=false;
   metalrobo::MetalWorldContext context(setting);metalrobo::MetalWorldResult out;auto d=context.run(world,b,cfg,out);
   std::cout<<",\"success\":"<<(d.succeeded()?"true":"false")<<",\"steps\":"<<d.successfulStepCount<<",\"failed_steps\":"<<d.failedStepCount<<",\"status_code\":"<<d.firstGPUStatusCode<<",\"run_message\":"<<std::quoted(d.message)<<",\"manifolds\":[";
   auto &e=out.contactEvidence;size_t count=e.manifoldCounts.empty()?0:e.manifoldCounts[0];
   for(size_t i=0;i<count && i<e.manifoldHeaders.size();++i){auto &h=e.manifoldHeaders[i];if(i)std::cout<<',';auto a=h.pairAndCount[1],bb=h.pairAndCount[2];std::cout<<"{\"collider_a\":"<<a<<",\"collider_b\":"<<bb<<",\"name_a\":"<<std::quoted(a<m.shapeNames.size()?m.shapeNames[a]:"UNKNOWN")<<",\"name_b\":"<<std::quoted(bb<m.shapeNames.size()?m.shapeNames[bb]:"UNKNOWN")<<",\"points\":"<<h.pairAndCount[3]<<'}';}
   std::cout<<"],\"contacts\":[";
   size_t n=out.contactStatuses.empty()?0:out.contactStatuses[0].activeContacts;
   for(size_t i=0;i<n && i<e.contacts.size();++i){auto &x=e.contacts[i];if(i)std::cout<<',';std::cout<<"{\"body_a\":"<<x.bodyA<<",\"body_b\":"<<x.bodyB<<",\"separation\":"<<x.pointAndSeparation.w<<",\"normal_impulse\":"<<x.impulses.x<<'}';}
   std::cout<<"],\"q\":";floats(out.finalQ);std::cout<<",\"v\":";floats(out.finalV);std::cout<<'}';
  }
  std::cout<<"]}\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

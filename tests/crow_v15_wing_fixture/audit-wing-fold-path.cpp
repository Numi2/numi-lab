#include "v15_fixture.hpp"
#include "metalrobo/CrowArticulatedWing.hpp"
#include "metalrobo/MetalWorld.hpp"
#include <fstream>
#include <iomanip>
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("expected metallib and q state table");
 auto p=metalrobo::makeCrowArticulatedWingMechanicsFixture(crowHipRollModel());auto&m=p.mechanics;m.world.gravityAndTimestep.x=m.world.gravityAndTimestep.y=m.world.gravityAndTimestep.z=0;
 metalrobo::CompiledWorld world;auto c=metalrobo::compileMetalWorld(m,0,world);if(!c.succeeded())throw std::runtime_error(c.message);
 metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=.000125f;cfg.physicsSubsteps=1;cfg.solverMode=metalrobo::MetalWorldSolverMode::temporalCone;cfg.captureContactEvidence=true;
 metalrobo::MetalWorldConfig setting;setting.metallibPath=argv[1];setting.preferParallelABA=false;metalrobo::MetalWorldContext context(setting);
 std::ifstream in(argv[2]);if(!in)throw std::runtime_error("cannot read q states");std::vector<float> efforts(m.articulations[0].nv,0),q(m.articulations[0].nq);std::cout<<std::setprecision(9)<<"{\"fingerprint\":"<<c.fingerprint<<",\"samples\":[";
 for(unsigned s=0;s<480;++s){for(auto&x:q)if(!(in>>x))throw std::runtime_error("missing q coordinate");
  metalrobo::MetalWorldBatch b;b.environmentCount=1;b.controlStepCount=1;b.initialQ=q;b.initialV=m.defaultV;b.efforts=efforts;metalrobo::MetalWorldResult out;auto d=context.run(world,b,cfg,out);
  if(s)std::cout<<',';std::cout<<"{\"sample\":"<<s+1<<",\"success\":"<<(d.succeeded()?"true":"false")<<",\"steps\":"<<d.successfulStepCount<<",\"failed\":"<<d.failedStepCount<<",\"code\":"<<d.firstGPUStatusCode<<",\"message\":"<<std::quoted(d.message)<<",\"contacts\":[";
  auto& ev=out.contactEvidence;size_t n=out.contactStatuses.empty()?0:out.contactStatuses[0].activeContacts;
  for(size_t i=0;i<n&&i<ev.contacts.size();++i){auto&x=ev.contacts[i];if(i)std::cout<<',';std::cout<<"{\"a\":"<<std::quoted(x.bodyA<m.bodyNames.size()?m.bodyNames[x.bodyA]:"UNKNOWN")<<",\"b\":"<<std::quoted(x.bodyB<m.bodyNames.size()?m.bodyNames[x.bodyB]:"UNKNOWN")<<",\"separation\":"<<x.pointAndSeparation.w<<",\"impulse\":"<<x.impulses.x<<'}';}std::cout<<"]}";
 }float extra;if(in>>extra)throw std::runtime_error("unexpected extra q data");std::cout<<"]}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

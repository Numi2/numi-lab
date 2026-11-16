#include "v15_fixture.hpp"
#include "metalrobo/CrowArticulatedWing.hpp"
#include "metalrobo/MetalWorld.hpp"
#include <iomanip>
static void floats(std::span<const float> v){std::cout<<'[';for(size_t i=0;i<v.size();++i){if(i)std::cout<<',';std::cout<<v[i];}std::cout<<']';}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("expected metallib");
 auto p=metalrobo::makeCrowArticulatedWingMechanicsFixture(crowHipRollModel());auto&m=p.mechanics;
 m.world.gravityAndTimestep.x=m.world.gravityAndTimestep.y=m.world.gravityAndTimestep.z=0;
 metalrobo::CompiledWorld world;auto c=metalrobo::compileMetalWorld(m,0,world);if(!c.succeeded())throw std::runtime_error(c.message);
 const auto&a=m.articulations[0];constexpr unsigned steps=480;
 std::array<std::string,4> names{"crow_left_elbow_flex","crow_left_wrist_flex","crow_right_elbow_flex","crow_right_wrist_flex"};std::array<unsigned,4> qidx{},vidx{};
 for(unsigned i=0;i<4;++i){auto it=std::find(m.dofNames.begin(),m.dofNames.end(),names[i]);if(it==m.dofNames.end())throw std::runtime_error("missing joint");auto&d=m.dofs.at(it-m.dofNames.begin());qidx[i]=d.qIndex;vidx[i]=d.vIndex;}
 std::vector<float> commands(steps*a.nv,0),targets(steps*4,0);
 for(unsigned s=0;s<steps;++s){
  for(auto&d:m.dofs)if((d.flags&MR_DOF_FLAG_DRIVE)&&d.qIndex<m.defaultQ.size()&&d.vIndex<a.nv)commands[s*a.nv+d.vIndex]=m.defaultQ[d.qIndex];
  float x=std::min(1.0f,(s+1.0f)/360.0f),smooth=x*x*(3-2*x);
  for(unsigned i=0;i<4;++i){float target=(i%2==0?1.8f:1.575f)*smooth;targets[s*4+i]=target;commands[s*a.nv+vidx[i]]=target;}
 }
 metalrobo::MetalWorldBatch b;b.environmentCount=1;b.controlStepCount=steps;b.initialQ=m.defaultQ;b.initialV=m.defaultV;b.efforts=commands;
 metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=1.0f/240;cfg.physicsSubsteps=4;cfg.solverMode=metalrobo::MetalWorldSolverMode::temporalCone;cfg.actuationMode=metalrobo::MetalWorldActuationMode::implicitPositionDrive;cfg.captureContactEvidence=true;
 metalrobo::MetalWorldConfig setting;setting.metallibPath=argv[1];setting.preferParallelABA=false;metalrobo::MetalWorldContext context(setting);metalrobo::MetalWorldResult out;auto d=context.run(world,b,cfg,out);
 std::cout<<std::setprecision(9)<<"{\"fingerprint\":"<<c.fingerprint<<",\"success\":"<<(d.succeeded()?"true":"false")<<",\"dispatched\":"<<(d.dispatched?"true":"false")<<",\"published\":"<<(d.published?"true":"false")<<",\"steps\":"<<d.successfulStepCount<<",\"failed_steps\":"<<d.failedStepCount<<",\"first_failure_step\":"<<d.firstFailingControlStep<<",\"code\":"<<d.firstGPUStatusCode<<",\"message\":"<<std::quoted(d.message)<<",\"nq\":"<<a.nq<<",\"nv\":"<<a.nv<<",\"targets\":";floats(targets);
 std::cout<<",\"q\":";floats(out.finalQ);std::cout<<",\"v\":";floats(out.finalV);std::cout<<",\"trajectory_qv\":";floats(out.observations);
 std::cout<<",\"contact_statuses\":[";for(size_t i=0;i<out.contactStatuses.size();++i){auto&s=out.contactStatuses[i];if(i)std::cout<<',';std::cout<<"{\"code\":"<<s.code<<",\"step\":"<<s.controlStep<<",\"substep\":"<<s.physicsSubstep<<",\"pairs\":"<<s.activePairs<<",\"contacts\":"<<s.activeContacts<<",\"constraints\":"<<s.requiredConstraints<<",\"first_failed_pair\":"<<s.firstFailingPair<<'}';}
 std::cout<<"],\"final_contacts\":[";size_t n=out.contactStatuses.empty()?0:out.contactStatuses.back().activeContacts;for(size_t i=0;i<n&&i<out.contactEvidence.contacts.size();++i){auto&x=out.contactEvidence.contacts[i];if(i)std::cout<<',';std::cout<<"{\"a\":"<<std::quoted(x.bodyA<m.bodyNames.size()?m.bodyNames[x.bodyA]:"UNKNOWN")<<",\"b\":"<<std::quoted(x.bodyB<m.bodyNames.size()?m.bodyNames[x.bodyB]:"UNKNOWN")<<",\"separation\":"<<x.pointAndSeparation.w<<",\"impulse\":"<<x.impulses.x<<'}';}std::cout<<"]}\n";
 return d.succeeded()&&d.failedStepCount==0&&d.successfulStepCount==steps?0:2;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

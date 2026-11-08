// A real MetalWorld owner rejection after immutable uploads are encoded.
#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/RunProgram.hpp"
#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Probe { bool reject=false; unsigned calls=0; };
static bool observe(void* context,const metalrobo::MetalWorldDeviceObservationPass&) {
    auto& p=*static_cast<Probe*>(context); ++p.calls;
    if(p.reject){p.reject=false;return false;}return true;
}
static void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char** argv){try{
 require(argc==5,"usage: probe METALLIB OUTPUT control|retry private|shared");
 const std::string storage=argv[4];require(storage=="private"||storage=="shared","storage mode");
 const bool retry=std::string(argv[3])=="retry";require(retry||std::string(argv[3])=="control","arm");
 metalrobo::RunManifest m;m.id="upload_abort_franka";auto robot=metalrobo::builtinRobotPack("franka_panda");require(robot.has_value(),"robot");m.robot=*robot;
 m.scene=metalrobo::makeFrankaPickPlaceScenePack();m.sensors.id="default";
 m.task=metalrobo::makeFrankaPickPlaceTaskPack(m.sensors.observation,m.reality.reset);
 m.reality.id="nominal";m.profile.id="probe";m.profile.environmentCount=2;m.profile.controlSteps=1;
 m.profile.physicsSubsteps=4;m.profile.controlTimestepSeconds=.02f;m.profile.capacities=m.task.capacities;m.teacher.id="none";
 metalrobo::CompiledRun run;auto compiled=metalrobo::compileRun(m,run);require(compiled.succeeded()&&run.valid(),"compile run");
 const auto nq=run.world().nq(),nv=run.world().nv();require(run.model().defaultQ.size()>=nq&&run.model().defaultV.size()>=nv,"model dimensions");
 std::vector<float> q,v;std::vector<MRBodyStateGPU> scene;
 for(unsigned e=0;e<2;++e){q.insert(q.end(),run.model().defaultQ.begin(),run.model().defaultQ.begin()+nq);v.insert(v.end(),run.model().defaultV.begin(),run.model().defaultV.begin()+nv);scene.insert(scene.end(),run.defaultSceneBodies().begin(),run.defaultSceneBodies().end());}
 const auto n=run.defaultSceneBodies().size();require(n==4,"scene dimensions");
 for(unsigned e=0;e<2;++e){scene[e*n].position={3.f,0.f,3.f,0.f};scene[e*n+3].position={4.f,0.f,3.f,0.f};}
 std::array<std::uint32_t,2> masks{0,0};std::vector<float> actions(2*run.task().layout().actionCount,0.f);
 metalrobo::MetalWorldBatch batch{.environmentCount=2,.controlStepCount=1,.initialQ=q,.initialV=v,.actions=actions,.policyRevision=19,.resetMasks=masks,.initialSceneBodies=scene};
 metalrobo::MetalWorldStepConfig config{};config.timestepSeconds=.02f;config.physicsSubsteps=4;config.solverMode=metalrobo::MetalWorldSolverMode::temporalCone;config.actuationMode=metalrobo::MetalWorldActuationMode::implicitPositionDrive;config.taskProgram=run.task();config.taskSeed=20261108;config.velocityIterations=4;config.finalVelocityIterations=2;config.publishFinalState=true;
 Probe probe{};config.deviceObservationProgram={&probe,observe};metalrobo::MetalWorldConfig ctxConfig{};ctxConfig.metallibPath=argv[1];ctxConfig.preferPrivateHeaps=(storage=="private");metalrobo::MetalWorldContext ctx(ctxConfig);
 auto execute=[&](){metalrobo::MetalWorldSubmission ticket;metalrobo::MetalWorldResult result;auto d=ctx.submit(run.world(),batch,config,ticket);if(d.succeeded())d=ticket.wait(result);return std::pair{d,std::move(result)};};
 if(retry){probe.reject=true;auto [d,r]=execute();std::cout<<"rejected_status="<<(unsigned)d.status<<" published="<<d.published<<" calls="<<probe.calls<<"\n";require(!d.succeeded()&&!d.published&&probe.calls==1,"rejection missing");}
 auto [d,r]=execute();std::cout<<"status="<<(unsigned)d.status<<" published="<<d.published<<" calls="<<probe.calls<<" message="<<d.message<<"\n";
 for(const auto& x:r.statuses)std::cout<<"world "<<x.environment<<" "<<x.code<<"\n";
 for(const auto& x:r.contactStatuses)std::cout<<"contact "<<x.environment<<" "<<x.code<<"\n";
 require(d.published&&r.statuses.size()==2&&r.contactStatuses.size()==2,"validated result");for(const auto& x:r.statuses)require(x.code==0,"world");for(const auto& x:r.contactStatuses)require(x.code==0,"contact");
 require(!r.finalQ.empty()&&!r.finalV.empty()&&!r.finalSceneBodies.empty()&&!r.actorObservations.empty(),"readback");
 std::ofstream out(argv[2],std::ios::binary);require(bool(out),"output");auto save=[&](const auto& v){std::uint64_t count=v.size();out.write((const char*)&count,8);if(count)out.write((const char*)v.data(),count*sizeof(v[0]));};
 save(r.finalQ);save(r.finalV);save(r.finalSceneBodies);save(r.actorObservations);out.flush();require(bool(out),"write");return 0;
}catch(const std::exception& e){std::cerr<<"error="<<e.what()<<"\n";return 2;}}

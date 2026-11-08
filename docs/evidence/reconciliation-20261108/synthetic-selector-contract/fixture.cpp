#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/InteractionPack.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include "metalrobo/task_program_types.h"
#include <cstdio>
#include <vector>

int main() {
 using namespace metalrobo;
 LocomotionWorld world;
 world.model=makeFreeSphereEngineModel();
 auto body=world.model.bodies[1]; world.model.bodies.assign(2,body);
 world.model.shapes.clear(); world.model.materials.clear(); world.model.shapeNames.clear();
 auto& mw=world.model.world; mw.bodyCount=2; mw.jointCount=1; mw.shapeCount=0; mw.materialCount=0; mw.nq=8; mw.nv=7;
 world.model.articulations[0]={0,MR_ROOT_FLOATING,0,2,0,1,0,8,0,7,0,0};
 world.model.joints.resize(1); auto& j=world.model.joints[0];
 j.parentBody=0; j.childBody=1; j.jointType=MR_JOINT_PRISMATIC; j.qOffset=7; j.vOffset=6; j.nq=1; j.nv=1;
 j.axis0={0,1,0,0}; j.parentRotation={0,0,0,1}; j.childRotation=j.parentRotation;
 for(unsigned i=0;i<2;++i) { auto& b=world.model.bodies[i]; b.articulationIndex=0; b.parentBody=i?0:MR_INVALID_INDEX; b.inboundJoint=i?0:MR_INVALID_INDEX; }
 world.model.dofs.resize(7); for(unsigned i=0;i<6;++i) world.model.dofs[i].jointIndex=MR_INVALID_INDEX; world.model.dofs[6]={0,0,7,6,0,MR_DOF_FLAG_ACTUATED,0,0,{},{}};
 world.model.defaultQ.assign(8,0); world.model.defaultQ[6]=1; world.model.defaultV.assign(7,0);
 world.model.bodyNames={"base","link"}; world.model.jointNames={"panda_joint1"}; world.model.dofNames={"x","y","z","rx","ry","rz","slide"};
 TaskPack task; task.id="synthetic-franka-selector";
 task.actions.push_back({.actuator="joint1"});
 task.rewards.push_back({.operation=TaskRewardOperator::constant,.weight=0});
 task.interactionStudentAuthority=0;
 task.interactionResetPhaseFraction=0.8f;
 std::vector<RobotActuatorSpec> actuators{{.id="joint1",.kind=RobotActuatorKind::jointPosition,.target="panda_joint1",.scale=0.1f}};
 TaskObservationProgram obs;
 obs.actorFrame.push_back({.source=TaskObservationSource::previousAction,.target="joint1"});
 obs.critic.push_back({.source=TaskObservationSource::previousAction,.target="joint1"});
 TaskResetProgram reset;
 CompiledLocomotionWorld compiled;
 auto worldStatus=compileLocomotionWorld(world,0,task,actuators,obs,reset,compiled);
 if (!worldStatus.succeeded()) { std::printf("world failure: %s / %s\n",worldStatus.world.message.c_str(),worldStatus.task.message.c_str()); return 1; }
 InteractionPack interactions{.id="synthetic",.sourceRepository="fixture",.sourceRevision="1",.license="test",.jointNames={"panda_joint1"}};
 InteractionClip clip{.id="clip",.desiredOutcome="test",.framesPerSecond=10,.frameCount=2};
 clip.rootTargets={0,0,0,0,0,0,1, 0,0,0,0,0,0,1}; clip.jointTargets={0,0.01f};
 interactions.clips.push_back(clip);
 std::printf("pack_valid=%d\n",validInteractionPack(interactions));
 CompiledTaskProgram guide, autonomous;
 auto g=compileTaskProgram(task,actuators,obs,reset,interactions,"clip",compiled.world,guide);
 if (!g.succeeded()) { std::printf("guide failure %s: %s\n",g.element.c_str(),g.message.c_str()); return 2; }
 task.interactionControlReference=false;
 auto a=compileTaskProgram(task,actuators,obs,reset,interactions,"clip",compiled.world,autonomous);
 if (!a.succeeded()) { std::printf("reset-only failure %s: %s\n",a.element.c_str(),a.message.c_str()); return 3; }
 std::printf("guide_schedule=%u reset_schedule=%u\n",guide.header().schedule.w,autonomous.header().schedule.w);
 std::printf("guide_task=%016llx reset_task=%016llx\n",(unsigned long long)guide.fingerprint(),(unsigned long long)autonomous.fingerprint());
 std::printf("world_equal=%d obs_equal=%d action_equal=%d\n",guide.worldFingerprint()==autonomous.worldFingerprint(),guide.observationFingerprint()==autonomous.observationFingerprint(),guide.actionFingerprint()==autonomous.actionFingerprint());
 const auto& layout=guide.layout(); PolicyPack policy; policy.id="synthetic"; policy.revision=1;
 policy.layers={{.inputCount=layout.actorObservationSize,.outputCount=layout.actionCount,.activation=PolicyActivation::identity,.weights=std::vector<float>(layout.actorObservationSize*layout.actionCount,0),.bias=std::vector<float>(layout.actionCount,0)}};
 bindPolicyPack(policy,guide); CompiledPolicyProgram result;
 auto strict=compilePolicyProgram(policy,autonomous,result);
 auto variant=compilePolicyProgramForTaskVariant(policy,guide,autonomous,result);
 auto wrong=compilePolicyProgramForTaskVariant(policy,autonomous,guide,result);
 std::printf("strict=%u variant=%u wrong_source=%u\n",unsigned(strict.status),unsigned(variant.status),unsigned(wrong.status));
 TaskPack stale=task; stale.interactionControlReference=true; stale.interactionStudentAuthority=0.1f; stale.interactionResetPhaseFraction=0;
 CompiledTaskProgram staleSource;
 auto st=compileTaskProgram(stale,actuators,obs,reset,interactions,"clip",compiled.world,staleSource);
 if (!st.succeeded()) { std::printf("stale failure: %s\n",st.message.c_str()); return 5; }
 auto staleAdmission=compilePolicyProgramForTaskVariant(policy,staleSource,autonomous,result);
 auto changedActuators=actuators; changedActuators[0].scale=0.2f;
 CompiledTaskProgram changed;
 auto ch=compileTaskProgram(task,changedActuators,obs,reset,interactions,"clip",compiled.world,changed);
 if (!ch.succeeded()) { std::printf("changed failure: %s\n",ch.message.c_str()); return 6; }
 auto changedAdmission=compilePolicyProgramForTaskVariant(policy,guide,changed,result);
 std::printf("stale_task=%016llx stale_admission=%u changed_action=%d changed_admission=%u\n",(unsigned long long)staleSource.fingerprint(),unsigned(staleAdmission.status),changed.actionFingerprint()!=guide.actionFingerprint(),unsigned(changedAdmission.status));

 bool good=(guide.header().schedule.w&MR_TASK_PROGRAM_INTERACTION_REFERENCE) && !(autonomous.header().schedule.w&MR_TASK_PROGRAM_INTERACTION_REFERENCE) && (guide.header().schedule.w&MR_TASK_PROGRAM_INTERACTION_RESET) && (autonomous.header().schedule.w&MR_TASK_PROGRAM_INTERACTION_RESET) && guide.fingerprint()!=autonomous.fingerprint() && guide.worldFingerprint()==autonomous.worldFingerprint() && guide.observationFingerprint()==autonomous.observationFingerprint() && guide.actionFingerprint()==autonomous.actionFingerprint() && !strict.succeeded() && variant.succeeded() && !wrong.succeeded() && !staleAdmission.succeeded() && changed.actionFingerprint()!=guide.actionFingerprint() && !changedAdmission.succeeded();
 return good?0:4;
}

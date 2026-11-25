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
 world.model=makeFrankaPickPlaceEngineModel();
 world.sceneBodies=makeFrankaPickPlaceSceneState();
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
 bool good=(guide.header().schedule.w&MR_TASK_PROGRAM_INTERACTION_REFERENCE) && !(autonomous.header().schedule.w&MR_TASK_PROGRAM_INTERACTION_REFERENCE) && (guide.header().schedule.w&MR_TASK_PROGRAM_INTERACTION_RESET) && (autonomous.header().schedule.w&MR_TASK_PROGRAM_INTERACTION_RESET) && guide.fingerprint()!=autonomous.fingerprint() && guide.worldFingerprint()==autonomous.worldFingerprint() && guide.observationFingerprint()==autonomous.observationFingerprint() && guide.actionFingerprint()==autonomous.actionFingerprint() && !strict.succeeded() && variant.succeeded() && !wrong.succeeded();
 return good?0:4;
}

#include "metalrobo/CrowManipulation.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include <algorithm>
#include <stdexcept>

namespace metalrobo {
TaskPack makeAmericanCrowObjectPushTaskPack(TaskObservationProgram& observations,
    TaskResetProgram& reset,const RetinalSensorProgram* retinal) {
    auto task=makeAmericanCrowEmbodiedTaskPack(observations,reset);
    task.id="american_crow_object_push_v14";task.maximumEpisodeSteps=100000;
    task.difficultyBandCount=1;task.commands={};task.commands.standingProbability=1;
    task.pushes={};reset={};
    task.rewards={
        {.operation=TaskRewardOperator::objectPosition,.target="crow_manipulandum",
         .weight=1,.parameters={.19f,.25f,.257f,.01f}},
        {.operation=TaskRewardOperator::mechanicalPower,.weight=-.0002f},
        {.operation=TaskRewardOperator::actionRateSquared,.weight=-.001f}};
    task.outcomes={
        {.id="object_goal_score",.unit="score",.source=TaskOutcomeSource::rewardContribution,
         .direction=TaskOutcomeDirection::higherIsBetter,.rewardOperation=TaskRewardOperator::objectPosition},
        {.id="mechanical_power_cost",.unit="weighted-cost",.source=TaskOutcomeSource::rewardContribution,
         .direction=TaskOutcomeDirection::higherIsBetter,.rewardOperation=TaskRewardOperator::mechanicalPower}};
    // Privileged pose is critic-only. The actor sees proprioception and real
    // camera RGB; it has no object position, contact ID or evaluation label.
    for(unsigned i=0;i<7;++i)observations.critic.push_back({
        .source=TaskObservationSource::objectTrack,.target="crow_manipulandum",.component=i});
    if(retinal) {
        if(!retinal->resourceCues.empty() || !retinal->width || !retinal->height ||
            retinal->width>4096 || retinal->height>4096)
            throw std::invalid_argument("Object push requires bounded retinal pixels and no resource cue labels");
        observations.retinalFeatureCount=retinal->width*retinal->height*6u+2u;
        for(unsigned i=0;i<observations.retinalFeatureCount;++i)
            observations.actorCurrent.push_back({.source=TaskObservationSource::retinalRGB,.component=i});
    }
    return task;
}
RunManifest makeAmericanCrowObjectPushFixtureManifest(bool crowObjectContact) {
    auto m=makeAmericanCrowForagingFixtureManifest(false);
    m.id="american_crow_object_push_v14_run";
    std::erase_if(m.scene.objects,[](const auto& x){return x.id=="head_camera_calibration_boxes";});
    m.scene.id="american_crow_object_push_support_v1";
    m.sensors={};m.sensors.id="crow_object_push_rgb_proprioception";
    m.task=makeAmericanCrowObjectPushTaskPack(m.sensors.observation,m.reality.reset);
    const LocomotionStaticBox table{"object_support",{.48f,0,.239f,1},
        {0,0,0,1},{.32f,.50f,.01f,0}};
    auto support=makeLocomotionStaticBoxComponent(m.robot.mechanics,std::span(&table,1));
    for(auto& s:support.mechanics.shapes){s.collisionGroup=4;s.collisionMask=~0u;}
    m.scene.objects.push_back({.id="object_support",.semanticClass="manipulation_support",
        .role=MR_WORLD_ASSET_FIXTURE,.render=MR_WORLD_RENDER_MESH_PBR,
        .dynamics=MR_WORLD_DYNAMICS_STATIC,.mechanics=std::move(support.mechanics),
        .defaultBodyStates=std::move(support.defaultBodyStates)});
    const LocomotionDynamicSphere sphere{.position={.225f,.030f,.257f,1},.radius=.008f,.mass=.002f};
    auto object=makeLocomotionDynamicSphereComponent(m.robot.mechanics,std::span(&sphere,1));
    object.mechanics.bodyNames[0]="crow_manipulandum";
    object.mechanics.shapeNames[0]="crow_manipulandum/collision";
    for(auto& s:object.mechanics.shapes){s.collisionGroup=2;s.collisionMask=crowObjectContact?~0u:4u;
        s.contactRestAndBoundingRadius.x=.0005f;}
    m.scene.objects.push_back({.id="crow_manipulandum",.semanticClass="movable_object",
        .role=MR_WORLD_ASSET_CLUTTER,.render=MR_WORLD_RENDER_MESH_PBR,
        .dynamics=MR_WORLD_DYNAMICS_RIGID,.mechanics=std::move(object.mechanics),
        .defaultBodyStates=std::move(object.defaultBodyStates)});
    applyAmericanCrowForagingFixtureLimits(m);
    m.reality.id="crow_object_push_nominal_reality";m.teacher.id="no_teacher";
    // Native refinement measured hybrid resting penetration2.217mm at4,
    // .562mm at8, .148mm at16 and .0444mm at32 substeps. Keep the
    // micro-object quality profile explicit; control and retina stay50Hz/15Hz.
    m.profile.physicsSubsteps=32;
    m.profile.environmentCount=8;m.profile.controlSteps=160;
    return m;
}
VisualSensorProgram authorAmericanCrowObjectPushRetinalProgram(
    const CompiledRun& run,const CrowForagingVisual& visual) {
    if(!run.valid() || !visual.resourceCues.empty() || visual.head>=run.model().bodies.size() ||
        run.model().bodyNames[visual.head]!="crow_head")
        throw std::invalid_argument("Object push requires its real crow camera/world and no resource cues");
    VisualSensorProgram p;p.cameraParentBody="crow_head";
    p.cameraPosition={.018f,.026f,.012f,0};p.cameraOrientation=crowObjectPushCameraOrientation;
    p.width=crowForagingCameraWidth;p.height=crowForagingCameraHeight;
    p.verticalFieldOfViewDegrees=45;p.nominalRateHz=15;p.minimumVisiblePixels=1;
    p.retinal=RetinalSensorProgram{};p.retinal->rightCameraOrientation=p.cameraOrientation;
    for(const auto& pack:visual.manifest.renderScene.visualPacks) {
        if(pack.assetIndex>=visual.family.worldTemplate.assets.size())
            throw std::invalid_argument("Object visual asset index invalid");
        p.assets.push_back({pack.packPath.string(),visual.family.worldTemplate.assets[pack.assetIndex].id,
            pack.contentHash,pack.semanticId,pack.instanceId});
    }
    p.fingerprint=visualSensorProgramFingerprint(p);return p;
}
} // namespace metalrobo

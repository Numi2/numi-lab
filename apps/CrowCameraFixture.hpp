#pragma once
#include "metalrobo/CollisionVisualProxy.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include "metalrobo/RunProgram.hpp"
#include "metalrobo/VisualPlatform.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <functional>
namespace crowCameraFixture {
using namespace metalrobo;
constexpr unsigned N=2,W=160,H=120,T=128;
constexpr double nearDepth=.01,farDepth=5;
inline void require(bool v,const std::string& reason) {if (!v) throw std::runtime_error(reason);}
template<class R> void check(const R& r,const char* reason) {require(r.succeeded(),std::string(reason)+": "+r.message);}
RunManifest makeManifest(bool blocked) {
    RunManifest m;
    m.id="crow_head_camera_geometry_admission";m.robot=makeAmericanCrowActiveHeadRobotPack();
    m.scene=makeBirdFlowAmericanCrowFlightScenePack();
    const std::array<LocomotionStaticBox,2> boxes{{
        {"camera_target",{1.20f,0,.275f,1},{0,0,0,1},{.065f,.065f,.065f,0}},
        {"camera_occluder",{.70f,blocked ? 0.0f : .60f,.275f,1},{0,0,0,1},{.025f,.14f,.16f,0}}}};
    auto fixture=makeLocomotionStaticBoxComponent(m.robot.mechanics,boxes);
    m.scene.objects.push_back({.id="head_camera_calibration_boxes",.semanticClass="calibration_fixture",
        .role=MR_WORLD_ASSET_FIXTURE,.render=MR_WORLD_RENDER_MESH_PBR,.collision=MR_WORLD_COLLISION_PRIMITIVES,
        .dynamics=MR_WORLD_DYNAMICS_STATIC,.mechanics=std::move(fixture.mechanics),
        .defaultBodyStates=std::move(fixture.defaultBodyStates)});
    m.sensors.id="camera_admission_proprioception";
    m.task=makeAmericanCrowFeedingStationTaskPack(m.sensors.observation,m.reality.reset);
    m.reality.id="canonical_camera_reality";m.teacher.id="no_teacher";m.profile.id="head_camera_probe";
    m.profile.environmentCount=N;m.profile.controlSteps=T;m.profile.physicsSubsteps=4;
    m.profile.controlTimestepSeconds=.02f;m.profile.minimumDifficultyBand=0;m.profile.maximumDifficultyBand=0;
    return m;
}

struct Visual {
    WorldFamily family;
    std::vector<SensorSpec> sensors;
    VisualSceneManifestV3 manifest;
    unsigned head=MR_INVALID_INDEX,target=MR_INVALID_INDEX;
};
Visual makeVisual(const CompiledRun& run,const std::filesystem::path& output,
    const std::function<void(const WorldAsset&,VisualAssetPackV2&)>& authorVisual={}) {
    const auto& model=run.model();Visual result;
    const auto find=[&](const std::string& name) {
        const auto at=std::find(model.bodyNames.begin(),model.bodyNames.end(),name);
        require(at!=model.bodyNames.end(),"Missing camera body: "+name);
        return static_cast<unsigned>(at-model.bodyNames.begin());
    };
    result.head=find("crow_head");result.target=find("camera_target");
    EpisodeTwin episode;episode.id="crow_head_proxy_camera_v1";
    WorldAsset robot;robot.id="robot";robot.semanticClass="estimated_crow_collision_proxy";
    robot.role=MR_WORLD_ASSET_ROBOT;robot.render=MR_WORLD_RENDER_MESH_PBR;
    robot.collision=MR_WORLD_COLLISION_PRIMITIVES;robot.dynamics=MR_WORLD_DYNAMICS_ARTICULATED;
    robot.articulationIndex=0;
    for (unsigned body=0;body<model.bodies.size();++body) {
        if (model.bodies[body].articulationIndex!=MR_INVALID_INDEX) robot.bodyIndices.push_back(body);
    }
    episode.assets.push_back(robot);
    const auto sceneBodies=run.world().sceneBodyIndices();
    for (unsigned local=0;local<sceneBodies.size();++local) {
        const auto body=sceneBodies[local];WorldAsset asset;
        asset.id=model.bodyNames[body];asset.semanticClass="calibration_fixture";asset.role=MR_WORLD_ASSET_FIXTURE;
        asset.render=MR_WORLD_RENDER_MESH_PBR;asset.collision=MR_WORLD_COLLISION_PRIMITIVES;
        asset.dynamics=MR_WORLD_DYNAMICS_STATIC;asset.bodyIndices={body};
        asset.initialPose.position=run.defaultSceneBodies()[local].position;
        asset.initialPose.orientation=run.defaultSceneBodies()[local].orientation;
        episode.assets.push_back(asset);
    }
    for (auto& asset : episode.assets) {
        for (unsigned shape=0;shape<model.shapes.size();++shape)
            if (std::find(asset.bodyIndices.begin(),asset.bodyIndices.end(),model.shapes[shape].bodyIndex)!=asset.bodyIndices.end()) {
                asset.shapeIndices.push_back(shape);
                asset.materialIndices.push_back(model.shapes[shape].materialIndex);
            }
        std::sort(asset.materialIndices.begin(),asset.materialIndices.end());
        asset.materialIndices.erase(std::unique(asset.materialIndices.begin(),asset.materialIndices.end()),asset.materialIndices.end());
    }
    for (unsigned camera=0;camera<2;++camera) {
        SensorSpec sensor;sensor.id=camera==0 ? "estimated_left_head_camera" : "estimated_right_head_camera";
        sensor.parentAssetId="robot";sensor.parentKind=MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK;
        sensor.parentBodyIndex=result.head;sensor.kind=MR_WORLD_SENSOR_RGBD;
        // Parallel mechanical stereo fixture outside the head collider. These
        // engineering optics/origins are NOT measured American-crow eyes.
        sensor.localPose.position={.018f,camera==0 ? .026f : -.026f,.012f,0};
        sensor.localPose.orientation={-.5f,.5f,-.5f,.5f};
        sensor.width=W;sensor.height=H;
        const float focal=.5f*H/std::tan(22.5f*3.14159265358979323846f/180.0f);
        sensor.intrinsics={focal,focal,.5f*W,.5f*H};sensor.minimumDepthMeters=nearDepth;
        sensor.maximumDepthMeters=farDepth;sensor.depthQuantumMeters=.0001f;sensor.nominalRateHz=15;
        sensor.exposureSeconds=1.0f/240;episode.sensors.push_back(sensor);
    }
    episode.task={"head_camera_geometry_admission","robot",{},{},{},.02,2.56};
    WorldTemplate worldTemplate;check(compileEpisodeTwin(episode,model,worldTemplate),"compile camera twin");
    WorldProgram program;program.id="fixed_camera_calibration_world";
    check(compileWorldFamily(worldTemplate,program,result.family),"compile camera family");
    std::vector<VisualAssetReferenceV3> references;
    for (unsigned index=0;index<episode.assets.size();++index) {
        const auto& asset=episode.assets[index];
        const mr_float4 color=index==0 ? mr_float4{.035f,.035f,.04f,1} : asset.id=="camera_target"
            ? mr_float4{.85f,.08f,.025f,1} : asset.id=="camera_occluder"
            ? mr_float4{.03f,.12f,.8f,1} : mr_float4{.25f,.29f,.25f,1};
        auto pack=makeCollisionProxyVisualAssetPack(model,asset.bodyIndices,run.world().fingerprint(),
                                                        asset.id+"_collision_proxy",color,16.0f);
        if (authorVisual) {authorVisual(asset,pack);pack.contentHash=computeVisualAssetPackContentHash(pack);}
        const auto path=output/(asset.id+".mrvpack");std::string reason;
        require(writeVisualAssetPack(pack,path,&reason),"Write collision proxy: "+reason);
        references.push_back({path,pack.contentHash,worldTemplate.assetIndex(asset.id),index+1,index+1});
    }
    std::string reason;
    require(compileVisualSceneManifestV3(worldTemplate,references,makeNeutralStudioEnvironmentV2(),
        makeIndoorAreaLightRigV1(),result.manifest,&reason),"Compile physical camera scene: "+reason);
    require(writeVisualSceneManifestV3(result.manifest,output/"visual-scene.json",&reason),reason);
    result.sensors=episode.sensors;return result;
}

}

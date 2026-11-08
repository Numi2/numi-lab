#include "metalrobo/CrowForaging.hpp"
#include "metalrobo/CollisionVisualProxy.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace metalrobo {
namespace {
void require(bool value,const std::string& reason) {if(!value)throw std::invalid_argument(reason);}
template<class R> void check(const R& r,const char* reason) {require(r.succeeded(),std::string(reason)+": "+r.message);}
void face(VisualAssetPackV2& pack,mr_float4 center,mr_float4 u,mr_float4 v,mr_float4 normal) {
    const auto first=unsigned(pack.vertices.size());const float length=std::sqrt(u.x*u.x+u.y*u.y+u.z*u.z);normal.w=1;
    const mr_float4 tangent{u.x/length,u.y/length,u.z/length,0};
    for (auto c:std::array<std::array<float,2>,4>{{{-1,-1},{1,-1},{1,1},{-1,1}}})
        pack.vertices.push_back({{center.x+u.x*c[0]+v.x*c[1],center.y+u.y*c[0]+v.y*c[1],center.z+u.z*c[0]+v.z*c[1],1},normal,tangent,{0,0,0,0},{1,1,1,1}});
    for (unsigned i:{0u,1u,2u,0u,2u,3u})pack.indices.push_back(first+i);
}
void beacon(VisualAssetPackV2& pack,unsigned body,unsigned side) {
    const unsigned instance=pack.instances.size(),primitive=pack.primitives.size(),first=pack.indices.size(),material=pack.materials.size();
    const float h=.035f;
    face(pack,{h,0,0,0},{0,h,0,0},{0,0,h,0},{1,0,0,0});face(pack,{-h,0,0,0},{0,-h,0,0},{0,0,h,0},{-1,0,0,0});
    face(pack,{0,h,0,0},{-h,0,0,0},{0,0,h,0},{0,1,0,0});face(pack,{0,-h,0,0},{h,0,0,0},{0,0,h,0},{0,-1,0,0});
    face(pack,{0,0,h,0},{h,0,0,0},{0,h,0,0},{0,0,1,0});face(pack,{0,0,-h,0},{h,0,0,0},{0,-h,0,0},{0,0,-1,0});
    auto mat=pack.materials.front();mat.baseColorAndOpacity={.03f,.9f,.05f,1};mat.emissionAndStrength={.02f,.6f,.02f,1};pack.materials.push_back(mat);
    MRVisualPrimitiveGPUV2 p{};p.geometry={first,36,material,instance};p.identity={1,1,body,1000u+side};p.boundsMinimum={-h,-h,-h,0};p.boundsMaximum={h,h,h,0};pack.primitives.push_back(p);
    MRVisualInstanceGPUV2 i{};i.translationAndScale={0,side ? -.16f : .16f,0,1};i.orientation={0,0,0,1};
    i.binding={0,body,MR_VISUAL_BINDING_RIGID_BODY,MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR|MR_VISUAL_INSTANCE_RECEIVES_SHADOW};i.identity=p.identity;i.geometry={primitive,1,0,0};pack.instances.push_back(i);
    pack.symbolicBindings.push_back({side ? "right_food_port_indicator" : "left_food_port_indicator","camera_target",instance,body,MR_VISUAL_BINDING_RIGID_BODY});
    pack.preprocessingProvenance+="; authored noncolliding food-port indicator, not a collision proxy for solid food";
}
}
RunManifest makeAmericanCrowForagingFixtureManifest(bool blocked) {
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
    m.profile.environmentCount=2;m.profile.controlSteps=128;m.profile.physicsSubsteps=4;
    m.profile.controlTimestepSeconds=.02f;m.profile.minimumDifficultyBand=0;m.profile.maximumDifficultyBand=0;
    return m;
}

CrowForagingVisual authorAmericanCrowForagingVisual(const CompiledRun& run,const std::filesystem::path& output,
    bool foodPortIndicators,mr_float4 cameraOrientation) {
    require(run.valid() && std::filesystem::is_directory(output),"Visual authoring requires a compiled run and existing output directory");
    const auto& model=run.model();CrowForagingVisual result;
    const auto find=[&](const std::string& name) {
        const auto at=std::find(model.bodyNames.begin(),model.bodyNames.end(),name);
        require(at!=model.bodyNames.end(),"Missing camera body: "+name);
        return static_cast<unsigned>(at-model.bodyNames.begin());
    };
    result.head=find("crow_head");
    const auto target=std::find(model.bodyNames.begin(),model.bodyNames.end(),"camera_target");
    if(target!=model.bodyNames.end())result.target=unsigned(target-model.bodyNames.begin());
    require(!foodPortIndicators || result.target!=MR_INVALID_INDEX,"Food indicators require their target fixture");
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
        asset.dynamics=model.bodies[body].motionType==MR_MOTION_DYNAMIC
            ? MR_WORLD_DYNAMICS_RIGID : MR_WORLD_DYNAMICS_STATIC;
        if(asset.dynamics==MR_WORLD_DYNAMICS_RIGID){asset.role=MR_WORLD_ASSET_CLUTTER;asset.semanticClass="movable_object";}
        asset.bodyIndices={body};
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
    // Validate the complete output set before writing its first pack.
    for(const auto& asset:episode.assets)
        require(!std::filesystem::exists(output/(asset.id+".mrvpack")),"Visual pack output exists");
    require(!std::filesystem::exists(output/"visual-scene.json"),"Visual scene output exists");
    for (unsigned camera=0;camera<2;++camera) {
        SensorSpec sensor;sensor.id=camera==0 ? "estimated_left_head_camera" : "estimated_right_head_camera";
        sensor.parentAssetId="robot";sensor.parentKind=MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK;
        sensor.parentBodyIndex=result.head;sensor.kind=MR_WORLD_SENSOR_RGBD;
        // Parallel mechanical stereo fixture outside the head collider. These
        // engineering optics/origins are NOT measured American-crow eyes.
        sensor.localPose.position={.018f,camera==0 ? .026f : -.026f,.012f,0};
        sensor.localPose.orientation=cameraOrientation;
        sensor.width=crowForagingCameraWidth;sensor.height=crowForagingCameraHeight;
        const float focal=.5f*crowForagingCameraHeight/std::tan(22.5f*3.14159265358979323846f/180.0f);
        sensor.intrinsics={focal,focal,.5f*crowForagingCameraWidth,.5f*crowForagingCameraHeight};sensor.minimumDepthMeters=.01;
        sensor.maximumDepthMeters=5;sensor.depthQuantumMeters=.0001f;sensor.nominalRateHz=15;
        sensor.exposureSeconds=1.0f/240;episode.sensors.push_back(sensor);
    }
    episode.task={"head_camera_geometry_admission","robot",{},{},{},.02,2.56};
    WorldTemplate worldTemplate;check(compileEpisodeTwin(episode,model,worldTemplate),"compile camera twin");
    WorldProgram program;program.id="fixed_camera_calibration_world";
    check(compileWorldFamily(worldTemplate,program,result.family),"compile camera family");
    std::vector<VisualAssetReferenceV3> references;
    unsigned meshOffset=0;
    for (unsigned index=0;index<episode.assets.size();++index) {
        const auto& asset=episode.assets[index];
        const mr_float4 color=index==0 ? mr_float4{.035f,.035f,.04f,1} : asset.id=="camera_target"
            ? mr_float4{.85f,.08f,.025f,1} : asset.id=="camera_occluder"
            ? mr_float4{.03f,.12f,.8f,1} : asset.id=="crow_manipulandum"
            ? mr_float4{.85f,.35f,.03f,1} : mr_float4{.25f,.29f,.25f,1};
        auto pack=makeCollisionProxyVisualAssetPack(model,asset.bodyIndices,run.world().fingerprint(),
                                                        asset.id+"_collision_proxy",color,16.0f);
        if (foodPortIndicators && asset.id=="camera_target") {
            for(unsigned side=0;side<2;++side) {
                result.resourceCues.push_back({{meshOffset+unsigned(pack.instances.size()),side,crowForagingCueEndTick,0}});
                beacon(pack,asset.bodyIndices.front(),side);
            }
        }
        if(foodPortIndicators)pack.contentHash=computeVisualAssetPackContentHash(pack);
        meshOffset+=pack.instances.size();
        const auto path=output/(asset.id+".mrvpack");std::string reason;
        require(!std::filesystem::exists(path),"Visual pack output exists");
        require(writeVisualAssetPack(pack,path,&reason),"Write collision proxy: "+reason);
        references.push_back({path,pack.contentHash,worldTemplate.assetIndex(asset.id),index+1,index+1});
    }
    std::string reason;
    require(compileVisualSceneManifestV3(worldTemplate,references,makeNeutralStudioEnvironmentV2(),
        makeIndoorAreaLightRigV1(),result.manifest,&reason),"Compile physical camera scene: "+reason);
    require(!std::filesystem::exists(output/"visual-scene.json"),"Visual scene output exists");
    require(writeVisualSceneManifestV3(result.manifest,output/"visual-scene.json",&reason),reason);
    require(!foodPortIndicators || result.resourceCues.size()==2,"Missing physical food-port indicator fixture");
    result.sensors=episode.sensors;return result;
}


FiniteResourceConfig authorAmericanCrowCuedForagingResources(
    const CompiledRun& physicalRun,const std::array<mr_float4,2>& calibratedPorts,
    unsigned environments,std::uint64_t allocationSeed) {
    require(physicalRun.valid(),"Cued foraging requires a compiled physical run");
    require(environments>0,"Cued foraging requires nonzero environments");
    for(const auto p:calibratedPorts)require(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z)&&std::isfinite(p.w)&&p.w>0,
        "Cued foraging ports require finite centers and positive radii");
    const auto a=calibratedPorts[0],b=calibratedPorts[1];
    const double dx=double(a.x)-b.x,dy=double(a.y)-b.y,dz=double(a.z)-b.z;
    const double radii=double(a.w)+b.w;
    require(dx*dx+dy*dy+dz*dz>radii*radii,"Cued foraging intake volumes overlap");
    auto resources=makeAmericanCrowFeedingResourceConfig(physicalRun,environments);
    resources.program.binding.z=2;resources.patches.front().centerRadius=calibratedPorts[0];
    auto second=resources.patches.front();second.centerRadius=calibratedPorts[1];resources.patches.push_back(second);
    resources.allocation={1,1,crowForagingIntakeStartTick,1};resources.allocationSeed=allocationSeed;
    return resources;
}
void applyAmericanCrowMemoryChoiceCurriculum(TaskPack& task,unsigned decisionStep,unsigned headHold,unsigned beakHold) {
    if(decisionStep<48u || decisionStep>crowForagingIntakeStartTick ||
       task.actions.size()!=18u || task.actions[15].actuator!="head.yaw" || task.actions[17].actuator!="beak.open")
        throw std::invalid_argument("Memory-choice requires exact crow actions and decision tick48..192");
    if((headHold==0)!=(beakHold==0) || headHold>512u || beakHold>512u)
        throw std::invalid_argument("Memory-choice holds require both periods in1..512 or both zero");
    task.actionAvailableAfterSteps=decisionStep;task.allowedActionMask=(1u<<15u)|(1u<<17u);
    task.actionHoldSteps.clear();
    std::erase_if(task.outcomes,[](const auto& x){return x.id=="policy_action_selection_mask";});
    if(headHold) {
        task.actionHoldSteps.assign(18u,0u);task.actionHoldSteps[15]=headHold;task.actionHoldSteps[17]=beakHold;
        task.outcomes.push_back({.id="policy_action_selection_mask",.unit="action-mask",
            .source=TaskOutcomeSource::policyActionSelectionMask,.direction=TaskOutcomeDirection::neutral});
    }
}

void applyAmericanCrowForagingFixtureLimits(RunManifest& m) {
    // This exact profile is admitted with the fixture. Inheriting unrelated
    // TaskPack capacities changes compiled world identity and source hashes.
    m.profile.capacities={};
    m.profile.capacities.rawContacts=256;m.profile.capacities.constraintBlocks=160;m.profile.capacities.constraintRows=480;
    m.profile.capacities.endpointRuntimeRecords=320;m.profile.capacities.qualityRows=480;m.profile.capacities.islandConstraintReferences=160;
    m.task.capacities=m.profile.capacities;m.task.maximumEpisodeSteps=100000;
}
CrowCuedForagingProgram authorAmericanCrowCuedForagingProgram(
    const CompiledRun& physicalRun,const CrowForagingVisual& visual,
    const std::array<mr_float4,2>& calibratedPorts,unsigned environments,std::uint64_t allocationSeed) {
    CrowCuedForagingProgram authored;
    authored.resources=authorAmericanCrowCuedForagingResources(physicalRun,calibratedPorts,environments,allocationSeed);
    require(visual.resourceCues.size()==2 && visual.resourceCues[0].binding.y==0 && visual.resourceCues[1].binding.y==1 &&
        visual.resourceCues[0].binding.z==crowForagingCueEndTick && visual.resourceCues[1].binding.z==crowForagingCueEndTick,
        "Cued foraging requires two authored port indicator bindings");
    require(visual.resourceCues[0].binding.x!=visual.resourceCues[1].binding.x &&
        visual.resourceCues[0].binding.w==0 && visual.resourceCues[1].binding.w==0,
        "Cued foraging indicator identities are invalid");
    std::string reason;
    require(visual.manifest.valid(&reason),"Invalid cued-foraging visual manifest: "+reason);
    require(visual.head<physicalRun.model().bodies.size() && physicalRun.model().bodyNames[visual.head]=="crow_head" &&
        visual.manifest.worldFingerprint==visual.family.worldTemplate.fingerprint,
        "Cued foraging visual/world provenance mismatch");
    auto& p=authored.sensor;
    p.cameraParentBody="crow_head";p.cameraPosition={.018f,.026f,.012f,0};p.cameraOrientation={-.5f,.5f,-.5f,.5f};
    p.width=crowForagingCameraWidth;p.height=crowForagingCameraHeight;p.verticalFieldOfViewDegrees=45;
    p.nominalRateHz=15;p.minimumVisiblePixels=1;p.retinal=RetinalSensorProgram{};
    p.retinal->resourceCues=visual.resourceCues;
    for(const auto& pack:visual.manifest.renderScene.visualPacks) {
        require(pack.assetIndex<visual.family.worldTemplate.assets.size(),"Cued foraging visual asset index is invalid");
        p.assets.push_back({pack.packPath.string(),visual.family.worldTemplate.assets[pack.assetIndex].id,pack.contentHash,pack.semanticId,pack.instanceId});
    }
    p.fingerprint=visualSensorProgramFingerprint(p);
    return authored;
}
TaskPack makeAmericanCrowCuedForagingTaskPack(
    TaskObservationProgram& observations,TaskResetProgram& reset,const CrowCuedForagingProgram& authored) {
    require(authored.sensor.retinal.has_value() && authored.sensor.retinal->resourceCues.size()==2,
        "Cued foraging task requires its authored retinal cue program");
    auto task=makeAmericanCrowRetinalFeedingTaskPack(observations,reset,*authored.sensor.retinal);
    const auto& r=authored.resources;
    task.deviceStateProgramFingerprint=finiteResourceProgramFingerprint(r.program,r.patches,r.publishTaskState,r.allocation);
    return task;
}
} // namespace metalrobo

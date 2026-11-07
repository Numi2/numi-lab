#pragma once
#import <Foundation/Foundation.h>
#include "metalrobo/c_api.h"
#include "metalrobo/CollisionVisualProxy.hpp"
#include "metalrobo/RunProgram.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
using namespace metalrobo;
namespace crow_retinal_fixture {
constexpr unsigned N=2,T=16,F=1251;
inline void require(bool value,const std::string& reason) {if (!value) throw std::runtime_error(reason);}
template<class R> void check(const R& result,const char* reason) {require(result.succeeded(),std::string(reason)+": "+result.message);}
inline void api(int result,const char* reason) {require(result==0,std::string(reason)+": "+mr_last_error());}
template<class V> void save(const std::filesystem::path& path,const std::vector<V>& values) {
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(V));require(bool(f),"Evidence write failed");
}
using CrowRunOwner=std::unique_ptr<MRTaskRolloutHandle,decltype(&mr_task_rollout_destroy)>;
struct Fixture {
    RunManifest manifest;CompiledRun run;VisualSensorProgram visual;
    MRRunManifestC c{};MRTaskVisualObservationConfigC camera{};MRRetinalSensorConfigC retina{};
    std::array<std::string,2> paths;std::array<MRTaskVisualPackC,2> packs{};
    Fixture(const std::filesystem::path& out,std::uint64_t seed=2651108801,bool reuse=false) {
        manifest.id="american_crow_retinal_feeding_v13_run";manifest.robot=makeAmericanCrowActiveHeadRobotPack();
        manifest.scene=makeBirdFlowAmericanCrowFlightScenePack();manifest.sensors.id="american_crow_retinal_v13_rgb_proprioception_nutrition";
        manifest.task=makeAmericanCrowFeedingStationTaskPack(manifest.sensors.observation,manifest.reality.reset);
        manifest.profile.id="native_task_rollout";manifest.profile.environmentCount=N;manifest.profile.controlSteps=1;
        manifest.profile.physicsSubsteps=4;manifest.profile.velocityIterations=4;manifest.profile.finalVelocityIterations=2;
        manifest.profile.controlTimestepSeconds=.02f;manifest.profile.seed=seed;manifest.profile.capacities=manifest.task.capacities;
        manifest.profile.minimumDifficultyBand=manifest.profile.maximumDifficultyBand=0;
        manifest.reality.id="birdflow_american_crow_estimated_hybrid_nominal_reality";manifest.teacher.id="no_teacher";
        CompiledRun base;check(compileRun(manifest,base),"compile asset source world");
        std::vector<unsigned> robot;
        for (unsigned body=0;body<base.model().bodies.size();++body) if (base.model().bodies[body].articulationIndex!=MR_INVALID_INDEX) robot.push_back(body);
        const auto ground=base.world().sceneBodyIndices();require(ground.size()==1,"Fixture is not the canonical supported station");
        const std::array<std::vector<unsigned>,2> selections{robot,std::vector<unsigned>{ground.front()}};
        const std::array<std::string,2> ids{"robot",base.model().bodyNames[ground.front()]};
        for (unsigned i=0;i<2;++i) {
            const auto pack=makeCollisionProxyVisualAssetPack(base.model(),selections[i],base.world().fingerprint(),ids[i]+"_explicit_development_proxy",
                i ? mr_float4{.25f,.29f,.25f,1} : mr_float4{.035f,.035f,.04f,1},16);
            paths[i]=(out/(ids[i]+".mrvpack")).string();std::string reason;
            if (reuse) {VisualAssetPackV2 retained;require(readVisualAssetPack(paths[i],retained,&reason) && retained.contentHash==pack.contentHash,"Retained fixture asset mismatch");}
            else require(writeVisualAssetPack(pack,paths[i],&reason),reason);
            visual.assets.push_back({paths[i],ids[i],pack.contentHash,i+1,i+1});
        }
        visual.cameraParentBody="crow_head";visual.cameraPosition={.018f,.026f,.012f,0};visual.cameraOrientation={-.5f,.5f,-.5f,.5f};
        visual.width=160;visual.height=120;visual.minimumVisiblePixels=1;visual.verticalFieldOfViewDegrees=45;
        visual.nominalRateHz=15;visual.retinal=RetinalSensorProgram{};visual.fingerprint=visualSensorProgramFingerprint(visual);
        manifest.sensors.observation={};manifest.reality.reset={};
        manifest.task=makeAmericanCrowRetinalFeedingTaskPack(manifest.sensors.observation,manifest.reality.reset,*visual.retinal);
        manifest.sensors.deviceVisual=visual;manifest.profile.capacities=manifest.task.capacities;
        check(compileRun(manifest,run),"compile expected retinal task");
        for (unsigned i=0;i<2;++i) packs[i]={paths[i].c_str(),visual.assets[i].assetId.c_str(),i+1,i+1};
        camera.packs=packs.data();camera.pack_count=2;camera.renderer_profile="sensor_fast";camera.camera_parent_body="crow_head";
        camera.camera_position[0]=.018f;camera.camera_position[1]=.026f;camera.camera_position[2]=.012f;
        const float orientation[4]={-.5f,.5f,-.5f,.5f};std::copy_n(orientation,4,camera.camera_orientation);
        camera.width=160;camera.height=120;camera.minimum_visible_pixels=1;camera.vertical_field_of_view_degrees=45;camera.nominal_rate_hz=15;
        retina.struct_size=sizeof(retina);retina.abi_version=1;retina.right_camera_position[0]=.018f;retina.right_camera_position[1]=-.026f;retina.right_camera_position[2]=.012f;
        std::copy_n(orientation,4,retina.right_camera_orientation);retina.width=16;retina.height=12;
        retina.sample_numerator=3;retina.sample_denominator=10;retina.latency_ticks=2;retina.minimum_depth_meters=.01f;retina.maximum_depth_meters=5;
        c.source=MR_RUN_SOURCE_BIRDFLOW_AMERICAN_CROW_JOURNEY;c.visual_sensor_program=&camera;
        c.profile.environment_count=N;c.profile.physics_substeps=4;c.profile.velocity_iterations=4;c.profile.final_velocity_iterations=2;
        c.profile.control_timestep_seconds=.02f;c.profile.seed=seed;c.profile.override_difficulty_band_range=1;
        c.profile.minimum_difficulty_band=c.profile.maximum_difficulty_band=0;c.profile.birdflow_journey_variant=MR_BIRDFLOW_JOURNEY_V13_RETINAL_FEEDING;
        if (!reuse) {
        std::ofstream artifact(out/"retinal-observation.json");
        artifact<<R"({"format":"numi.retinal-observation.v1","id":"explicit-supported-station-proxy-development-only","packs":[)";
        for (unsigned i=0;i<2;++i) artifact<<(i ? "," : "")<<"{\"path\":"<<std::quoted(std::filesystem::path(paths[i]).filename().string())
            <<",\"asset_id\":"<<std::quoted(visual.assets[i].assetId)<<",\"semantic_id\":"<<i+1<<",\"instance_id\":"<<i+1<<"}";
        artifact<<R"(],"camera":{"parent_body":"crow_head","position":[0.018,0.026,0.012],"orientation":[-0.5,0.5,-0.5,0.5],"width":160,"height":120,"minimum_visible_pixels":1,"vertical_field_of_view_degrees":45,"nominal_rate_hz":15,"maximum_retained_bytes":0},"retinal":{"right_position":[0.018,-0.026,0.012],"right_orientation":[-0.5,0.5,-0.5,0.5],"width":16,"height":12,"sample_numerator":3,"sample_denominator":10,"latency_ticks":2,"minimum_depth_meters":0.01,"maximum_depth_meters":5}})";
        }
    }
    CrowRunOwner create() {
        CrowRunOwner h(mr_create_task_rollout_retinal(&c,&retina),mr_task_rollout_destroy);require(bool(h),std::string("Create retinal C handle: ")+mr_last_error());
        const auto layout=mr_task_rollout_layout(h.get());
        require(layout.actor_observation_count==F && layout.critic_observation_count==97 && layout.action_count==18 &&
            layout.world_fingerprint==run.world().fingerprint() && layout.observation_fingerprint==run.task().observationFingerprint() &&
            layout.task_fingerprint==run.task().fingerprint(),"C entry differs from exact native task contract");
        return h;
    }
    std::filesystem::path policy(const std::filesystem::path& out,const std::string& id,float jaw=0) {
        PolicyPack p;p.id=id;p.observationMean.assign(F,0);p.observationInverseStandardDeviation.assign(F,1);
        PolicyDenseLayer a{.inputCount=F,.outputCount=18,.activation=PolicyActivation::identity,.weights=std::vector<float>(F*18,0),.bias=std::vector<float>(18,0)};
        a.bias[17]=jaw;p.layers.push_back(a);p.criticObservationMean.assign(97,0);p.criticObservationInverseStandardDeviation.assign(97,1);
        p.criticLayers={{.inputCount=97,.outputCount=1,.activation=PolicyActivation::identity,.weights=std::vector<float>(97,0),.bias={.25f}}};
        bindPolicyPack(p,run.task());const auto path=out/(id+".policypack");check(writePolicyPack(p,path),"save explicit control policy");return path;
    }
};
}

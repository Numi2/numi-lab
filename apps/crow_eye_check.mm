#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <ImageIO/ImageIO.h>
#include "metalrobo/CollisionVisualProxy.hpp"
#include "metalrobo/FiniteResource.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include "metalrobo/RunProgram.hpp"
#include "metalrobo/VisualPlatform.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace metalrobo;
constexpr unsigned N=2,W=160,H=120,T=128;
constexpr double nearDepth=.01,farDepth=5,depthTolerance=.0002;
void require(bool yes,const std::string& message) {if (!yes) throw std::runtime_error(message);}
template<class R> void check(const R& result,const char* label) {require(result.succeeded(),std::string(label)+": "+result.message);}
struct V {double x,y,z;};
struct Q {double x,y,z,w;};
V vector(mr_float4 a) {return {a.x,a.y,a.z};}
Q quaternion(mr_float4 a) {return {a.x,a.y,a.z,a.w};}
V operator+(V a,V b) {return {a.x+b.x,a.y+b.y,a.z+b.z};}
V operator-(V a,V b) {return {a.x-b.x,a.y-b.y,a.z-b.z};}
V operator*(V a,double b) {return {a.x*b,a.y*b,a.z*b};}
double dot(V a,V b) {return a.x*b.x+a.y*b.y+a.z*b.z;}
V cross(V a,V b) {return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
Q product(Q a,Q b) {return {a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
    a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,
    a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z};}
V rotate(Q q,V v) {const V u{q.x,q.y,q.z};return v*(q.w*q.w-dot(u,u))+u*(2*dot(u,v))+cross(u,v)*(2*q.w);}
V inverse(Q q,V v) {return rotate({-q.x,-q.y,-q.z,q.w},v);}
double maximum(V v) {return std::max({std::abs(v.x),std::abs(v.y),std::abs(v.z)});}
struct Pose {V p;Q q;};
Pose pose(const MRBodyStateGPU& state) {return {vector(state.position),quaternion(state.orientation)};}
Pose compose(Pose a,Pose b) {return {a.p+rotate(a.q,b.p),product(a.q,b.q)};}
Pose localPose(const WorldPose& p) {return {vector(p.position),quaternion(p.orientation)};}

CompiledRun makeRun(bool blocked) {
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
    CompiledRun run;check(compileRun(m,run),"compile camera run");return run;
}

std::vector<MRBodyStateGPU> acceptedBodies(const CompiledRun& run,int control,unsigned chunk) {
    MetalWorldContext context;MetalWorldResidentState resident;
    FiniteResourceContext food(makeAmericanCrowFeedingResourceConfig(run,N));
    MetalWorldStepConfig step;
    step.timestepSeconds=.02f;step.physicsSubsteps=4;step.velocityIterations=4;step.finalVelocityIterations=2;
    step.solverMode=MetalWorldSolverMode::temporalCone;step.actuationMode=MetalWorldActuationMode::implicitPositionDrive;
    step.taskProgram=run.task();step.taskSeed=2651108501u;step.minimumDifficultyBand=0;step.maximumDifficultyBand=0;
    step.publishStateTrajectory=false;step.publishFinalState=true;step.devicePhysicsProgram=food.deviceProgram();
    if (run.flappingWingProgram()) step.flappingWingProgram=*run.flappingWingProgram();
    std::vector<float> q,v;std::vector<MRBodyStateGPU> scene;
    for (unsigned env=0;env<N;++env) {
        q.insert(q.end(),run.model().defaultQ.begin(),run.model().defaultQ.end());
        v.insert(v.end(),run.model().defaultV.begin(),run.model().defaultV.end());
        scene.insert(scene.end(),run.defaultSceneBodies().begin(),run.defaultSceneBodies().end());
    }
    MetalWorldResult final;
    for (unsigned start=0;start<T;start+=chunk) {
        const unsigned count=std::min(chunk,T-start);
        std::vector<float> actions(count*N*18,0);std::vector<unsigned> resets(count*N,0);
        for (unsigned t=0;t<count;++t) for (unsigned env=0;env<N;++env) {
            if (control==1) actions[(t*N+env)*18+15]=env==0 ? 1.0f : -1.0f;
            if (control==2) actions[(t*N+env)*18+16]=-1.0f;
        }
        MetalWorldBatch batch{.environmentCount=N,.controlStepCount=count,.actions=actions,.resetMasks=resets};
        MetalWorldSubmission ticket;
        if (start==0) {
            batch.initialQ=q;batch.initialV=v;batch.initialSceneBodies=scene;
            check(context.initializeResidentState(run.world(),batch,step,resident,ticket),"initialize camera physics");
        } else check(context.submitResident(run.world(),batch,step,resident,ticket),"submit camera physics");
        const auto status=ticket.wait(final);check(status,"accept camera physics");
        require(status.failedStepCount==0,"Camera control rejected a physics step");
        for (const auto& transition : final.transitions) require(transition.termination.x==0,"Camera control terminated");
    }
    for (const auto& state : food.snapshot()) require(state.accounting.y==0,"Camera probe unexpectedly consumed food");
    std::vector<MRBodyStateGPU> result;std::string reason;
    require(composeVisualBodyStates(run.model(),N,final.finalQ,final.finalV,final.finalSceneBodies,result,&reason),reason);
    return result;
}

struct Visual {
    WorldFamily family;
    std::vector<SensorSpec> sensors;
    VisualSceneManifestV3 manifest;
    unsigned head=MR_INVALID_INDEX,target=MR_INVALID_INDEX;
};
Visual makeVisual(const CompiledRun& run,const std::filesystem::path& output) {
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
        const auto pack=makeCollisionProxyVisualAssetPack(model,asset.bodyIndices,run.world().fingerprint(),
                                                        asset.id+"_collision_proxy",color,16.0f);
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

struct Hit {double depth=std::numeric_limits<double>::infinity();unsigned body=MR_INVALID_INDEX;};
Hit referenceRay(const EngineModel& model,std::span<const MRBodyStateGPU> states,Pose camera,
                 const SensorSpec& sensor,double x,double y) {
    const V direction=rotate(camera.q,{(x-sensor.intrinsics.z)/sensor.intrinsics.x,
                                     (y-sensor.intrinsics.w)/sensor.intrinsics.y,1});
    Hit result;
    for (const auto& shape : model.shapes) {
        const Pose shapePose=compose(pose(states[shape.bodyIndex]),{vector(shape.localPosition),quaternion(shape.localRotation)});
        const V o=inverse(shapePose.q,camera.p-shapePose.p),d=inverse(shapePose.q,direction);
        double distance=std::numeric_limits<double>::infinity();
        if (shape.shapeType==MR_SHAPE_PLANE) {
            if (std::abs(d.y)>1e-12) {
                distance=-o.y/d.y;
                if (distance>=nearDepth && distance<=farDepth) {
                    const V hit=o+d*distance;
                    require(std::abs(hit.x)<16 && std::abs(hit.z)<16,"Visual plane clip excludes an in-range physical ray");
                }
            }
        } else {
            require(shape.shapeType==MR_SHAPE_BOX,"Reference ray encountered unsupported physical geometry");
            const std::array<double,3> origin{o.x,o.y,o.z},ray{d.x,d.y,d.z};
            const auto h=shape.dimensions;const std::array<double,3> half{h.x,h.y,h.z};
            double low=-std::numeric_limits<double>::infinity(),high=std::numeric_limits<double>::infinity();
            for (unsigned axis=0;axis<3;++axis) {
                if (std::abs(ray[axis])<1e-12) {if (std::abs(origin[axis])>half[axis]) {high=-1;break;}}
                else {
                    double a=(-half[axis]-origin[axis])/ray[axis],b=(half[axis]-origin[axis])/ray[axis];
                    if (a>b) std::swap(a,b);low=std::max(low,a);high=std::min(high,b);
                }
            }
            if (high>=low) distance=low>=nearDepth ? low : high;
        }
        if (distance>=nearDepth && distance<=farDepth && distance<result.depth) result={distance,shape.bodyIndex};
    }
    return result;
}

std::vector<MRHybridCameraStateGPU> readCameras(MetalHybridRenderer& renderer) {
    id<MTLBuffer> source=(__bridge id<MTLBuffer>)renderer.nativeBuffer(MetalHybridRendererBuffer::cameraStates);
    require(source!=nil && source.length>=N*sizeof(MRHybridCameraStateGPU),"Missing native camera pose buffer");
    id<MTLBuffer> staging=[source.device newBufferWithLength:N*sizeof(MRHybridCameraStateGPU) options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue=[source.device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLBlitCommandEncoder> encoder=[command blitCommandEncoder];
    require(staging!=nil && queue!=nil && command!=nil && encoder!=nil,"Camera publication allocation failed");
    [encoder copyFromBuffer:source sourceOffset:0 toBuffer:staging destinationOffset:0 size:staging.length];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];
    require(command.status==MTLCommandBufferStatusCompleted,"Camera publication blit failed");
    std::vector<MRHybridCameraStateGPU> result(N);std::memcpy(result.data(),staging.contents,staging.length);return result;
}

void saveImage(const std::filesystem::path& path,const HybridObservationBatch& image,unsigned env) {
    std::vector<unsigned char> pixels(W*H*4);
    for (unsigned pixel=0;pixel<W*H;++pixel) {
        const auto color=image.rgb[env*W*H+pixel];const std::array<float,3> rgb{color.x,color.y,color.z};
        for (unsigned channel=0;channel<3;++channel) {
            require(std::isfinite(rgb[channel]),"Native camera published a non-finite RGB sample");
            const float x=std::clamp(rgb[channel],0.0f,1.0f);
            const float s=x<=.0031308f ? 12.92f*x : 1.055f*std::pow(x,1.0f/2.4f)-.055f;
            pixels[pixel*4+channel]=static_cast<unsigned char>(std::lround(s*255));
        }
        pixels[pixel*4+3]=255;
    }
    CGColorSpaceRef colors=CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGDataProviderRef provider=CGDataProviderCreateWithData(nullptr,pixels.data(),pixels.size(),nullptr);
    const auto bitmapInfo=static_cast<CGBitmapInfo>(static_cast<std::uint32_t>(kCGImageAlphaLast)|static_cast<std::uint32_t>(kCGBitmapByteOrderDefault));
    CGImageRef bitmap=CGImageCreate(W,H,8,32,W*4,colors,bitmapInfo,provider,nullptr,false,kCGRenderingIntentDefault);
    NSURL* url=[NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    CGImageDestinationRef destination=CGImageDestinationCreateWithURL((__bridge CFURLRef)url,CFSTR("public.png"),1,nullptr);
    require(bitmap!=nullptr && destination!=nullptr,"PNG publication allocation failed");
    CGImageDestinationAddImage(destination,bitmap,nullptr);const bool saved=CGImageDestinationFinalize(destination);
    CFRelease(destination);CGImageRelease(bitmap);CGDataProviderRelease(provider);CGColorSpaceRelease(colors);
    require(saved,"PNG publication failed");
}

struct Metrics {unsigned compared=0,emptyCompared=0,identityErrors=0;double maximumDepthError=0,maximumCameraError=0;std::array<unsigned,N> targetPixels{};};
template<class R> void saveRaw(const std::filesystem::path& path,const std::vector<R>& values) {
    std::ofstream output(path,std::ios::binary);
    output.exceptions(std::ios::badbit|std::ios::failbit);
    output.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(R));
}
Metrics admit(const CompiledRun& run,const Visual& visual,unsigned cameraIndex,
              std::span<const MRBodyStateGPU> bodies,const HybridObservationBatch& image,
              std::span<const MRHybridCameraStateGPU> cameras) {
    Metrics result;const auto& sensor=visual.sensors[cameraIndex];const auto bodyCount=run.model().bodies.size();
    require(image.environmentCount==N && image.width==W && image.height==H && image.rgb.size()==N*W*H &&
            image.depth.size()==N*W*H && image.identities.size()==N*W*H && image.validity.size()==N*W*H &&
            cameras.size()==N && bodies.size()==N*bodyCount,"Incomplete physical camera evidence");
    for (unsigned env=0;env<N;++env) {
        const auto states=bodies.subspan(env*bodyCount,bodyCount);
        const Pose camera=compose(pose(states[visual.head]),localPose(sensor.localPose));
        const auto& gpu=cameras[env];
        result.maximumCameraError=std::max(result.maximumCameraError,maximum(camera.p-vector(gpu.currentPositionAndValidity)));
        const Q actual=quaternion(gpu.currentOrientation);
        const double sign=camera.q.x*actual.x+camera.q.y*actual.y+camera.q.z*actual.z+camera.q.w*actual.w<0 ? -1 : 1;
        result.maximumCameraError=std::max(result.maximumCameraError,std::max({std::abs(camera.q.x-sign*actual.x),
            std::abs(camera.q.y-sign*actual.y),std::abs(camera.q.z-sign*actual.z),std::abs(camera.q.w-sign*actual.w)}));
        require(gpu.currentPositionAndValidity.w==1,"Native camera binding is invalid");
        for (unsigned y=1;y<H-1;++y) for (unsigned x=1;x<W-1;++x) {
            const auto index=env*W*H+y*W+x;
            if (image.identities[index].z==visual.target && (image.validity[index]&MR_VISUAL_VALIDITY_DEPTH)) ++result.targetPixels[env];
            const auto hit=referenceRay(run.model(),states,camera,sensor,x+.5,y+.5);
            bool interior=true;
            for (const auto offset : std::array<std::array<int,2>,4>{{{1,0},{-1,0},{0,1},{0,-1}}})
                if (referenceRay(run.model(),states,camera,sensor,x+.5+offset[0],y+.5+offset[1]).body!=hit.body) interior=false;
            if (!interior) continue;
            if (hit.body==MR_INVALID_INDEX) {
                ++result.emptyCompared;
                if (image.validity[index]&MR_VISUAL_VALIDITY_DEPTH) ++result.identityErrors;
                continue;
            }
            ++result.compared;
            if (image.identities[index].z!=hit.body || !(image.validity[index]&MR_VISUAL_VALIDITY_DEPTH) || !std::isfinite(image.depth[index])) ++result.identityErrors;
            else result.maximumDepthError=std::max(result.maximumDepthError,std::abs(image.depth[index]-hit.depth));
        }
    }
    return result;
}
}

int main(int argc,char** argv) {@autoreleasepool {
    bool ownsOutput=false;
    try {
        require(argc==2,"Usage: metalrobo_crow_eye_check NEW_OUTPUT_DIRECTORY");
        const std::filesystem::path output=argv[1];require(std::filesystem::create_directory(output),"Eye evidence directory already exists");ownsOutput=true;
        std::ofstream(output/"plan.json")<<R"({"schema":"numi.crow-head-camera-admission.v4","purpose":"instrument calibration, not learned vision or measured crow anatomy","parent":"01 and 02 rejected vertex and instance authoring; 03 passed registered geometry; this prospective instrument extension additionally checks empty rays, malformed authoring rejection, and documents binary layouts; existing thresholds unchanged","seed":2651108501,"environments":2,"native_steps":128,"dt_seconds":0.02,"cameras":2,"width":160,"height":120,"camera_model":"estimated parallel head-mounted pinhole fixture, not biological eye calibration","controls":["neutral","occluded","opposite-physical-yaw","physical-pitch","chunk-1-vs-8","exact-render-repeat"],"predictions":{"neutral_target_pixels_per_lane_min":20,"occluded_target_pixels":0,"yaw_target_pixels":0,"physical_ray_identity_errors":0,"stable_empty_ray_false_depth":0,"malformed_authoring_cases_rejected":7,"maximum_interior_depth_error_m":0.0002,"maximum_camera_pose_component_error":0.000002},"reference":"independent FP64 ray/box and ray/+Y-plane intersections against accepted body poses; one-pixel silhouettes excluded","plane_proxy_half_extent_m":16,"depth_range_m":[0.01,5],"stopping":"four bounded native cases and one partition repeat, no retries or threshold changes","boundary":"publication-time inspection readback; device-resident actor image inputs and sensor cadence not yet admitted"})";
        {
            const auto run=makeRun(false);const auto& source=run.model();
            const std::array<unsigned,1> body{{0}};
            const auto cook=[&](const EngineModel& model,std::span<const unsigned> selected,std::uint64_t fingerprint,float extent) {
                return makeCollisionProxyVisualAssetPack(model,selected,fingerprint,"negative_control",{.2f,.2f,.2f,1},extent);
            };
            const auto rejects=[&](auto action) {
                bool rejected=false;try {action();} catch (const std::invalid_argument&) {rejected=true;}
                require(rejected,"Malformed visual authoring was accepted");
            };
            rejects([&]{cook(source,body,0,16);});
            rejects([&]{cook(source,body,run.world().fingerprint(),0);});
            const std::array<unsigned,2> duplicate{{0,0}};
            rejects([&]{cook(source,duplicate,run.world().fingerprint(),16);});
            const std::array<unsigned,1> outside{{static_cast<unsigned>(source.bodies.size())}};
            rejects([&]{cook(source,outside,run.world().fingerprint(),16);});
            const auto shape=std::find_if(source.shapes.begin(),source.shapes.end(),[](const auto& s){return s.bodyIndex==0;});
            require(shape!=source.shapes.end(),"No source shape for malformed-authoring controls");
            const auto index=shape-source.shapes.begin();
            auto bad=source;bad.shapes[index].flags|=MR_SHAPE_FLAG_SIMULATION_DISABLED;
            rejects([&]{cook(bad,body,run.world().fingerprint(),16);});
            bad=source;bad.shapes[index].shapeType=MR_SHAPE_CAPSULE;
            rejects([&]{cook(bad,body,run.world().fingerprint(),16);});
            bad=source;bad.shapes[index].dimensions.x=-1;
            rejects([&]{cook(bad,body,run.world().fingerprint(),16);});
            std::ofstream(output/"authoring-controls.json")<<R"({"status":"passed","rejected_cases":7,"cases":["missing-world-identity","zero-plane-extent","duplicate-body","outside-body","disabled-shape","unsupported-capsule","negative-box-extent"]})";
        }
        std::ofstream report(output/"measurements.jsonl");
        for (const auto& control : std::array<std::string,4>{"neutral","occluded","yaw","pitch"}) {
            const auto directory=output/control;std::filesystem::create_directory(directory);
            const auto run=makeRun(control=="occluded");
            const auto bodies=acceptedBodies(run,control=="yaw" ? 1 : control=="pitch" ? 2 : 0,8);
            saveRaw(directory/"accepted-body-states.bin",bodies);
            if (control=="yaw") {
                const auto partition=acceptedBodies(run,1,1);
                require(bodies.size()==partition.size() && std::memcmp(bodies.data(),partition.data(),bodies.size()*sizeof(bodies[0]))==0,
                        "Accepted visual body state depends on submission partition");
            }
            const auto visual=makeVisual(run,directory);
            MetalWorldFamilyContext worlds;check(worlds.compile(visual.family,N),"compile camera world device");
            check(worlds.sample(N,2651108501u),"sample camera worlds");
            MetalHybridRendererConfig config;config.width=W;config.height=H;config.maximumRetainedBytes=512ull*1024*1024;
            MetalHybridRenderer renderer(config);
            auto scene=visual.manifest.renderScene;
            check(renderer.compile(std::move(scene),VisualRendererProfileV1::sensorFast(),N),"compile physical proxy renderer");
            for (unsigned camera=0;camera<2;++camera) {
                HybridLiveStateBatch live;live.environmentCount=N;live.bodyCount=run.model().bodies.size();
                live.currentBodies=bodies;live.previousBodies=bodies;live.frameIndex=T;live.sensorSequence=camera;
                live.captureTimestampSeconds=T*.02;live.source=MR_VISUAL_SOURCE_SIMULATION;
                check(renderer.renderLive(worlds,live,camera),"render accepted head camera");
                HybridObservationBatch image;check(renderer.readback(image),"publish head image");
                const auto cameras=readCameras(renderer);
                const auto prefix=directory/("camera-"+std::to_string(camera));
                saveRaw(prefix.string()+".rgba.f32",image.rgb);saveRaw(prefix.string()+".depth.f32",image.depth);
                saveRaw(prefix.string()+".identities.u32",image.identities);saveRaw(prefix.string()+".validity.u32",image.validity);
                saveRaw(prefix.string()+".poses.bin",cameras);
                std::ofstream(prefix.string()+".layout.json")
                    <<"{\"byte_order\":\"little-endian\",\"order\":\"environment,y,x,channel\",\"environments\":"<<N
                    <<",\"width\":"<<W<<",\"height\":"<<H
                    <<",\"rgba_f32_channels\":4,\"depth_f32_unit\":\"optical-z-meters\",\"identities_u32_channels\":4,\"depth_validity_mask\":"
                    <<MR_VISUAL_VALIDITY_DEPTH<<",\"camera_pose_record_bytes\":"<<sizeof(MRHybridCameraStateGPU)
                    <<",\"body_state_record_bytes\":"<<sizeof(MRBodyStateGPU)<<",\"bodies_per_environment\":"<<run.model().bodies.size()
                    <<",\"camera_pose_fields\":[\"currentPositionAndValidity:f32x4\",\"currentOrientation:f32x4\",\"previousPositionAndValidity:f32x4\",\"previousOrientation:f32x4\"]}\n";
                const auto metrics=admit(run,visual,camera,bodies,image,cameras);
                report<<"{\"control\":"<<std::quoted(control)<<",\"camera\":"<<camera<<",\"run_fingerprint\":"<<run.fingerprint()
                    <<",\"compared_pixels\":"<<metrics.compared<<",\"empty_compared_pixels\":"<<metrics.emptyCompared<<",\"identity_errors\":"<<metrics.identityErrors
                    <<",\"maximum_depth_error_m\":"<<metrics.maximumDepthError<<",\"maximum_camera_pose_error\":"<<metrics.maximumCameraError
                    <<",\"target_pixels\":["<<metrics.targetPixels[0]<<','<<metrics.targetPixels[1]<<"]}\n";
                report.flush();
                for (unsigned env=0;env<N;++env)
                    saveImage(directory/("camera-"+std::to_string(camera)+"-lane-"+std::to_string(env)+".png"),image,env);
                require(metrics.compared>500,"Insufficient stable interior pixels for geometry admission");
                require(metrics.identityErrors==0,"Rendered depth/identity disagrees with independent physical-shape rays");
                require(metrics.maximumDepthError<=depthTolerance,"Rendered metric depth exceeds the registered physical-ray tolerance");
                require(metrics.maximumCameraError<=2e-6,"Native camera pose does not follow its accepted head transform");
                check(renderer.renderLive(worlds,live,camera),"repeat head camera");
                HybridObservationBatch repeated;check(renderer.readback(repeated),"publish repeated head image");
                require(image.rgb.size()==repeated.rgb.size() && std::memcmp(image.rgb.data(),repeated.rgb.data(),image.rgb.size()*sizeof(image.rgb[0]))==0 &&
                    image.depth==repeated.depth && image.validity==repeated.validity &&
                    std::memcmp(image.identities.data(),repeated.identities.data(),image.identities.size()*sizeof(image.identities[0]))==0,
                    "Head camera render is not exactly repeatable");
                for (unsigned env=0;env<N;++env) {
                    if (control=="neutral") require(metrics.targetPixels[env]>=20,"Neutral head camera cannot see its calibration target");
                    if (control=="occluded" || control=="yaw") require(metrics.targetPixels[env]==0,"Target remains visible through physical occlusion or an averted head");
                }
            }
        }
        const std::string result=R"({"schema":"numi.crow-head-camera-evidence.v1","status":"passed","native_cases":5,"cameras":2,"geometry_and_camera_pose_admitted":true,"exact_render_repeat":true,"actor_vision_integrated":false,"biological_eye_calibration":false,"ecological_environment_qualified":false})";
        std::ofstream(output/"summary.json")<<result<<'\n';std::cout<<result<<'\n';return 0;
    } catch (const std::exception& error) {
        if (ownsOutput) std::ofstream(std::filesystem::path(argv[1])/"failure.json")<<"{\"error\":"<<std::quoted(error.what())<<"}\n";
        std::cerr<<error.what()<<'\n';return 1;
    }
}}

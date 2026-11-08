#import <Foundation/Foundation.h>
#include "metalrobo/CrowManipulation.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/detail/Checkpoint.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace metalrobo;
namespace {
constexpr unsigned lanes=4,ticks=160;
void require(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
template<class R>void check(const R&r,const char*s){require(r.succeeded(),s);}
template<class T>void save(const std::filesystem::path&p,const std::vector<T>&v){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));require(bool(f),"Evidence write failed");}
unsigned body(const EngineModel&m,const std::string&n){const auto i=std::find(m.bodyNames.begin(),m.bodyNames.end(),n);require(i!=m.bodyNames.end(),"Missing body");return i-m.bodyNames.begin();}
double distance(const std::array<double,3>&a,const std::array<double,3>&b){double d=0;for(unsigned i=0;i<3;++i)d+=std::pow(a[i]-b[i],2);return std::sqrt(d);}
std::array<double,3> rotate(const std::array<double,4>&q,const std::array<double,3>&p){
    const double tx=2*(q[1]*p[2]-q[2]*p[1]),ty=2*(q[2]*p[0]-q[0]*p[2]),tz=2*(q[0]*p[1]-q[1]*p[0]);
    return {p[0]+q[3]*tx+q[1]*tz-q[2]*ty,p[1]+q[3]*ty+q[2]*tx-q[0]*tz,p[2]+q[3]*tz+q[0]*ty-q[1]*tx};
}
std::array<double,3> relative(const ArticulatedBodyKinematics&b,mr_float4 p){
    const auto&q=b.orientation;return rotate({-q[0],-q[1],-q[2],q[3]},
        {p.x-b.centerOfMassPosition[0],p.y-b.centerOfMassPosition[1],p.z-b.centerOfMassPosition[2]});
}
double gap(const ArticulatedBodyKinematics&b,const MRShapeGPU&s,mr_float4 p){
    auto x=relative(b,p); // Authored box local rotations are identity, checked below.
    const double h[]={s.dimensions.x,s.dimensions.y,s.dimensions.z},c[]={s.localPosition.x,s.localPosition.y,s.localPosition.z};
    double squared=0;double inside=-1e9;
    for(unsigned i=0;i<3;++i){double e=std::abs(x[i]-c[i])-h[i];squared+=std::pow(std::max(e,0.),2);inside=std::max(inside,e);}
    return std::sqrt(squared)+std::min(inside,0.)-.004;
}
void calibrate(){
    ArticulatedBodyKinematics b{};b.orientation={0,0,0,1};b.centerOfMassPosition={0,0,0};
    MRShapeGPU s{};s.dimensions={.025f,.009f,.004f,0};s.localPosition={0,0,0,0};
    require(std::abs(gap(b,s,{0,0,.009f,1})-.001)<2e-9,"Separated gap reference");
    require(std::abs(gap(b,s,{0,0,.007f,1})+.001)<2e-9,"Overlapping gap reference");
    require(std::abs(gap(b,s,{0,0,0,1})+.008)<2e-9,"Inside-box gap reference");
    b.centerOfMassPosition={.1,.2,.3};b.orientation={0,std::sqrt(.5),0,std::sqrt(.5)};
    auto p=relative(b,{.109f,.2f,.3f,1});
    require(std::abs(p[0])<2e-8&&std::abs(p[1])<2e-8&&std::abs(p[2]-.009)<2e-8,"Rotated body reference");
    std::cout<<"{\"calibration\":\"passed\",\"fp64_gap_known_values\":3,\"rotated_frame_known_values\":1}\n";
}
CompiledRun fixture(unsigned arm,unsigned substeps){
    auto m=makeAmericanCrowObjectPushFixtureManifest();m.id="crow_beak_hold_contact_surrogate_v1";
    m.scene.id=m.id;std::erase_if(m.scene.objects,[](const auto&o){return o.id=="object_support"||o.id=="crow_manipulandum";});
    m.robot.mechanics.defaultQ.back()=.23f;
    for(auto&s:m.robot.mechanics.shapes)if(m.robot.mechanics.bodyNames[s.bodyIndex]=="crow_head"||m.robot.mechanics.bodyNames[s.bodyIndex]=="crow_lower_beak"){
        s.contactRestAndBoundingRadius.x=.0005f;s.contactRestAndBoundingRadius.y=0;
    }
    LocomotionDynamicSphere sphere{.position={.215f,0,.26685f,1},.radius=.004f,.mass=.00025f};
    auto object=makeLocomotionDynamicSphereComponent(m.robot.mechanics,std::span(&sphere,1));
    object.mechanics.bodyNames[0]="crow_manipulandum";object.mechanics.shapeNames[0]="crow_manipulandum/collision";
    for(auto&s:object.mechanics.shapes){s.collisionGroup=2;s.collisionMask=arm==2?4u:~0u;s.contactRestAndBoundingRadius.x=.0005f;s.contactRestAndBoundingRadius.y=0;}
    if(arm==3)for(auto&mat:object.mechanics.materials)mat.friction={0,0,0,0};
    m.scene.objects.push_back({.id="crow_manipulandum",.semanticClass="rigid_contact_surrogate_not_ingested_food",
        .role=MR_WORLD_ASSET_CLUTTER,.render=MR_WORLD_RENDER_MESH_PBR,.dynamics=MR_WORLD_DYNAMICS_RIGID,
        .mechanics=std::move(object.mechanics),.defaultBodyStates=std::move(object.defaultBodyStates)});
    m.profile.physicsSubsteps=substeps;m.profile.environmentCount=lanes;m.profile.controlSteps=ticks;
    CompiledRun run;check(compileRun(m,run),"Compile contact fixture");return run;
}
void execute(const std::filesystem::path&dir,unsigned arm,unsigned chunk,unsigned substeps){@autoreleasepool{
    require(std::filesystem::create_directory(dir),"Output exists");const auto run=fixture(arm,substeps);const auto&model=run.model();
    const unsigned object=body(model,"crow_manipulandum"),head=body(model,"crow_head"),beak=body(model,"crow_lower_beak");
    unsigned upper=MR_INVALID_INDEX,lower=MR_INVALID_INDEX,objectShape=MR_INVALID_INDEX;
    for(unsigned i=0;i<model.shapes.size();++i){if(model.shapeNames[i]=="crow_head/upper_beak_collision")upper=i;if(model.shapes[i].bodyIndex==beak)lower=i;if(model.shapes[i].bodyIndex==object)objectShape=i;}
    require(upper!=MR_INVALID_INDEX&&lower!=MR_INVALID_INDEX&&objectShape!=MR_INVALID_INDEX,"Missing physical shapes");
    for(unsigned i:{upper,lower}){const auto&q=model.shapes[i].localRotation;require(q.x==0&&q.y==0&&q.z==0&&q.w==1,"Gap oracle requires identity local box rotations");}
    const auto ids=run.world().sceneBodyIndices();const auto it=std::find(ids.begin(),ids.end(),object);require(it!=ids.end(),"Missing object scene body");const unsigned local=it-ids.begin(),scenes=ids.size();
    std::vector<float> q,v;std::vector<MRBodyStateGPU> initial;
    for(unsigned lane=0;lane<lanes;++lane){q.insert(q.end(),model.defaultQ.begin(),model.defaultQ.end());v.insert(v.end(),model.defaultV.begin(),model.defaultV.end());initial.insert(initial.end(),run.defaultSceneBodies().begin(),run.defaultSceneBodies().end());}
    std::vector<double> dq(model.defaultQ.begin(),model.defaultQ.end()),dv(model.defaultV.begin(),model.defaultV.end());std::vector<ArticulatedBodyKinematics> k(model.bodies.size());
    check(computeArticulatedBodyKinematics(model,0,dq,dv,k),"Initial FP64 kinematics");
    const auto start=initial[local].position;const double initialUpper=gap(k[head],model.shapes[upper],start),initialLower=gap(k[beak],model.shapes[lower],start);
    require(initialUpper>2e-5&&initialLower>2e-5,"Object starts overlapping a jaw");
    MetalWorldStepConfig step;step.timestepSeconds=.02f;step.physicsSubsteps=substeps;step.velocityIterations=4;step.finalVelocityIterations=2;
    step.solverMode=MetalWorldSolverMode::temporalCone;step.actuationMode=MetalWorldActuationMode::implicitPositionDrive;
    step.taskProgram=run.task();step.taskSeed=2651511101ull;step.minimumDifficultyBand=step.maximumDifficultyBand=0;
    step.ccdMode=MetalWorldCCDMode::hybrid;step.publishFinalState=true;step.captureContactEvidence=true;
    if(run.flappingWingProgram())step.flappingWingProgram=*run.flappingWingProgram();
    MetalWorldConfig config;config.enableResidentCheckpoints=true;config.maximumInFlightSubmissions=1;
    MetalWorldContext world(config);MetalWorldResidentState resident;MetalWorldResult result;
    std::ofstream trace(dir/"trace.tsv");trace<<"tick\tlane\tx\ty\tz\tpitch\tjaw\tupper_Jn\tlower_Jn\tupper_Jt\tlower_Jt\tupper_gap_m\tlower_gap_m\trel_x\trel_y\trel_z\n"<<std::setprecision(12);
    std::array<std::array<double,3>,lanes> anchor{};std::array<double,lanes> minLift,maxSlip;minLift.fill(1e9);maxSlip.fill(0);
    std::array<unsigned,lanes> opposed{};std::vector<MRBodyStateGPU> trajectory;
    for(unsigned begin=0;begin<ticks;begin+=chunk){const unsigned count=std::min(chunk,ticks-begin);std::vector<float>actions(count*lanes*18,0);std::vector<unsigned>resets(count*lanes,0);
        for(unsigned t=0;t<count;++t)for(unsigned lane=0;lane<lanes;++lane){auto*p=actions.data()+(t*lanes+lane)*18;p[16]=begin+t>=32?-.4f:0;p[17]=(arm!=1&&begin+t<96)?-.35f:0;}
        MetalWorldBatch b{.environmentCount=lanes,.controlStepCount=count,.actions=actions,.resetMasks=resets};MetalWorldSubmission ticket;
        if(begin==0){b.initialQ=q;b.initialV=v;b.initialSceneBodies=initial;check(world.initializeResidentState(run.world(),b,step,resident,ticket),"Initialize contact trial");}
        else check(world.submitResident(run.world(),b,step,resident,ticket),"Submit contact trial");
        auto status=ticket.wait(result);check(status,"Complete contact trial");require(!status.failedStepCount,"Rejected contact trial step");
        for(const auto&t:result.transitions)require(!t.termination.x,"Physical termination");
        const auto stride=result.contactEvidence.contacts.size()/lanes;
        for(unsigned lane=0;lane<lanes;++lane){
            const auto pos=result.finalSceneBodies[lane*scenes+local].position;
            for(unsigned i=0;i<dq.size();++i)dq[i]=result.finalQ[lane*dq.size()+i];
            for(unsigned i=0;i<dv.size();++i)dv[i]=result.finalV[lane*dv.size()+i];
            check(computeArticulatedBodyKinematics(model,0,dq,dv,k),"Accepted-state FP64 kinematics");
            double normal[2]={},tangent[2]={};
            for(unsigned j=0;j<result.contactStatuses[lane].activeContacts;++j){const auto&c=result.contactEvidence.contacts[lane*stride+j];
                unsigned a=c.pairKey>>32,b=unsigned(c.pairKey);const auto other=a==objectShape?b:b==objectShape?a:MR_INVALID_INDEX;
                if(other==upper||other==lower){const unsigned s=other==upper?0:1;normal[s]+=std::abs(c.impulses.x);tangent[s]+=std::hypot(c.impulses.y,c.impulses.z);}}
            const unsigned tick=begin+count-1;const auto rel=relative(k[head],pos);
            if(tick==31)anchor[lane]=rel;
            if(tick>=64&&tick<96){minLift[lane]=std::min(minLift[lane],double(pos.z-start.z));maxSlip[lane]=std::max(maxSlip[lane],distance(rel,anchor[lane]));opposed[lane]+=normal[0]>1e-9&&normal[1]>1e-9;}
            trace<<tick<<'\t'<<lane<<'\t'<<pos.x<<'\t'<<pos.y<<'\t'<<pos.z<<'\t'<<dq[dq.size()-2]<<'\t'<<dq.back()<<'\t'<<normal[0]<<'\t'<<normal[1]<<'\t'<<tangent[0]<<'\t'<<tangent[1]<<'\t'<<gap(k[head],model.shapes[upper],pos)<<'\t'<<gap(k[beak],model.shapes[lower],pos)<<'\t'<<rel[0]<<'\t'<<rel[1]<<'\t'<<rel[2]<<'\n';
        }
        if(chunk==1)trajectory.insert(trajectory.end(),result.finalSceneBodies.begin(),result.finalSceneBodies.end());
    }
    save(dir/"final.world",world.checkpointResident(resident,step));save(dir/"final-q.f32",result.finalQ);save(dir/"final-v.f32",result.finalV);save(dir/"final-scene.bin",result.finalSceneBodies);
    if(!trajectory.empty())save(dir/"scene.bin",trajectory);
    std::ofstream summary(dir/"summary.json");summary<<std::setprecision(12)<<"{\"arm\":"<<arm<<",\"chunk\":"<<chunk<<",\"substeps\":"<<substeps<<",\"accepted_transitions\":"<<ticks*lanes<<",\"physics_rejections\":0,\"initial_upper_gap_m\":"<<initialUpper<<",\"initial_lower_gap_m\":"<<initialLower<<",\"lanes\":[";
    for(unsigned lane=0;lane<lanes;++lane){if(lane)summary<<',';summary<<"{\"minimum_lift_64_95_m\":"<<minLift[lane]<<",\"maximum_head_relative_slip_m\":"<<maxSlip[lane]<<",\"opposed_contact_samples\":"<<opposed[lane]<<",\"final_z\":"<<result.finalSceneBodies[lane*scenes+local].position.z<<'}';}summary<<"]}\n";
}}
}
int main(int argc,char**argv){@autoreleasepool{try{
    if(argc==2&&std::string(argv[1])=="--calibrate"){calibrate();return 0;}
    require(argc==5,"Usage: probe NEW_OUTPUT ARM CHUNK SUBSTEPS");const auto out=std::filesystem::absolute(argv[1]);const unsigned arm=std::stoul(argv[2]),chunk=std::stoul(argv[3]),substeps=std::stoul(argv[4]);
    require(arm<=3&&(chunk==1||chunk==8)&&(substeps==32||substeps==64),"Invalid trial selector");execute(out,arm,chunk,substeps);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}}

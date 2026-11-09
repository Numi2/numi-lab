// Native free rigid slider baseline. This is not the coupled Matter transaction.
#include "metalrobo/EngineModel.hpp"
#include "metalrobo/MetalWorld.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool x,const std::string& s){if(!x)throw std::runtime_error(s);}
static MRBodyPropertiesGPU body(std::uint32_t parent,std::uint32_t joint,float mass){
 MRBodyPropertiesGPU b{};b.articulationIndex=0;b.parentBody=parent;b.inboundJoint=joint;
 b.motionType=MR_MOTION_DYNAMIC;b.massAndInverseMass={mass,1.f/mass,0,0};
 b.inertiaRow0={.001f,0,0,0};b.inertiaRow1={0,.001f,0,0};b.inertiaRow2={0,0,.001f,0};
 b.inverseInertiaRow0={1000,0,0,0};b.inverseInertiaRow1={0,1000,0,0};b.inverseInertiaRow2={0,0,1000,0};
 b.dampingAndSpeedLimits={0,0,1e6f,1e6f};return b;
}
int main(int argc,char**argv){try{
 require(argc==2,"usage: midpoint_slider_rigid_baseline METALLIB");
 constexpr float h=.001f, v0=.02f, g=-9.81f;
 metalrobo::EngineModel model;model.name="midpoint_slider_free_baseline";
 model.world.abiVersion=MR_ENGINE_ABI_VERSION;model.world.bodyCount=2;model.world.articulationCount=1;
 model.world.jointCount=1;model.world.nq=1;model.world.nv=1;model.world.shapeCount=0;model.world.materialCount=0;
 model.world.pairCapacity=1;model.world.contactCapacity=1;model.world.constraintCapacity=1;model.world.islandCapacity=1;
 model.world.solverType=MR_SOLVER_REFERENCE_FP64;model.world.frictionConeType=MR_FRICTION_CONE_ELLIPTIC;
 model.world.gravityAndTimestep={0,0,g,h};model.world.solverScales={1e-8f,1e-9f,2,1e-4f};
 MRArticulationGPU a{};a.rootBody=0;a.rootType=MR_ROOT_FIXED;a.firstBody=0;a.bodyCount=2;a.firstJoint=0;
 a.jointCount=1;a.qOffset=0;a.nq=1;a.vOffset=0;a.nv=1;model.articulations.push_back(a);
 model.bodies.push_back(body(MR_INVALID_INDEX,MR_INVALID_INDEX,1));model.bodies.push_back(body(0,0,.01f));
 MRJointDescriptorGPU j{};j.parentBody=0;j.childBody=1;j.jointType=MR_JOINT_PRISMATIC;
 j.qOffset=0;j.nq=1;j.vOffset=0;j.nv=1;j.axis0={0,0,1,0};j.parentRotation={0,0,0,1};j.childRotation={0,0,0,1};model.joints.push_back(j);
 MRDofPropertiesGPU d{};d.articulationIndex=0;d.jointIndex=0;d.qIndex=0;d.vIndex=0;model.dofs.push_back(d);
 model.defaultQ={0};model.defaultV={v0};std::string reason;require(model.valid(&reason),"invalid slider: "+reason);
 metalrobo::CompiledWorld world;auto cooked=metalrobo::compileMetalWorld(model,0,world);require(cooked.succeeded(),"compile: "+cooked.message);
 std::vector<float> effort(1,0);metalrobo::MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,.initialQ=model.defaultQ,.initialV=model.defaultV,.efforts=effort};
 metalrobo::MetalWorldStepConfig config{};config.timestepSeconds=h;config.physicsSubsteps=1;
 config.solverMode=metalrobo::MetalWorldSolverMode::freeMotionABA;config.matrixFreeArticulatedContact=false;config.streamedArticulatedContactResponses=false;config.captureContactEvidence=false;
 metalrobo::MetalWorldConfig ownerConfig{};ownerConfig.metallibPath=argv[1];metalrobo::MetalWorldContext context(ownerConfig);metalrobo::MetalWorldResult result;auto run=context.run(world,batch,config,result);
 require(run.succeeded(),"run: "+run.message);require(result.finalQ.size()==1 && result.finalV.size()==1,"missing rigid output");
 const double q=result.finalQ[0],v=result.finalV[0],expectedV=double(v0)+double(h)*double(g);
 const double be=double(h)*expectedV,mid=double(h)*(double(v0)+expectedV)/2;
 require(std::abs(v-expectedV)<1e-6 && std::abs(q-be)<1e-8,"rigid baseline changed or failed");
 std::cout<<std::setprecision(17)<<"{\"schema\":\"numi.rigid-slider.free-baseline.v1\",\"h\":"<<h<<",\"mass_kg\":0.01,\"initial_v\":"<<v0<<",\"gravity\":"<<g<<",\"actual_q\":"<<q<<",\"actual_v\":"<<v<<",\"endpoint_velocity_q\":"<<be<<",\"midpoint_q\":"<<mid<<",\"midpoint_position_defect_m\":"<<q-mid<<"}\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}

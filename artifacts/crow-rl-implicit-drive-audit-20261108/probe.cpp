#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalWorld.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace metalrobo;
namespace {
void require(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
MRBodyPropertiesGPU body(unsigned parent,unsigned joint,float mass,float inertia){
    MRBodyPropertiesGPU b{};b.articulationIndex=0;b.parentBody=parent;b.inboundJoint=joint;b.motionType=MR_MOTION_DYNAMIC;
    b.massAndInverseMass={mass,1/mass,0,0};b.inertiaRow0={inertia,0,0,0};b.inertiaRow1={0,inertia,0,0};b.inertiaRow2={0,0,inertia,0};
    b.inverseInertiaRow0={1/inertia,0,0,0};b.inverseInertiaRow1={0,1/inertia,0,0};b.inverseInertiaRow2={0,0,1/inertia,0};b.dampingAndSpeedLimits={0,0,100000,100000};return b;
}
EngineModel model(){
    auto m=makeFreeSphereEngineModel();m.name="analytic_crow_scale_torque_limited_hinge";
    m.articulations.clear();m.joints.clear();m.dofs.clear();m.bodies.clear();m.shapes.clear();m.defaultQ={0};m.defaultV={0};
    MRArticulationGPU a{};a.rootBody=0;a.rootType=MR_ROOT_FIXED;a.bodyCount=2;a.jointCount=1;a.nq=1;a.nv=1;m.articulations.push_back(a);
    m.bodies={body(MR_INVALID_INDEX,MR_INVALID_INDEX,1,.01f),body(0,0,.004f,.000004f)};
    MRJointDescriptorGPU j{};j.parentBody=0;j.childBody=1;j.jointType=MR_JOINT_REVOLUTE;j.nq=j.nv=1;j.axis0={0,0,1,0};j.parentRotation=j.childRotation={0,0,0,1};m.joints.push_back(j);
    MRDofPropertiesGPU d{};d.articulationIndex=d.jointIndex=d.qIndex=d.vIndex=0;d.flags=MR_DOF_FLAG_ACTUATED|MR_DOF_FLAG_DRIVE|MR_DOF_FLAG_EFFORT_LIMIT;
    d.limits={0,0,0,.01f};d.drive={.30f,.015f,.000002f,0};m.dofs.push_back(d);
    m.world.bodyCount=2;m.world.articulationCount=1;m.world.jointCount=1;m.world.shapeCount=0;m.world.nq=m.world.nv=1;m.world.gravityAndTimestep={0,0,0,.001f};
    std::string reason;require(m.valid(&reason),"Analytic model invalid: "+reason);return m;
}
double implicitVelocity(double mass,double k,double d,double torque,double h,double q,double v,double target){
    const double u=(mass*v+h*k*(target-q))/(mass+h*d+h*h*k);
    return std::clamp(u,v-h*torque/mass,v+h*torque/mass);
}
double bisectVelocity(double mass,double k,double d,double torque,double h,double q,double v,double target){
    double lo=v-h*torque/mass,hi=v+h*torque/mass;
    for(unsigned i=0;i<100;++i){double x=(lo+hi)*.5;double f=mass*(x-v)/h-std::clamp(k*(target-q-h*x)-d*x,-torque,torque);if(f>0)hi=x;else lo=x;}return (lo+hi)*.5;
}
void calibrate(){
    const auto m=model();const double mass=double(m.bodies[1].inertiaRow2.z)+m.dofs[0].drive.z;
    std::array<double,1>q{0},v{0},tau{.001},acc{};ArticulatedDynamicsConfig c;c.gravity={0,0,0};c.applyBodyDamping=false;
    const auto d=computeArticulatedForwardDynamics(m,0,q,v,tau,{},acc,c);require(d.succeeded(),"FP64 hinge solve rejected");require(std::abs(acc[0]-tau[0]/mass)<1e-8,"FP64 hinge mass calibration failed");
    double error=0;for(unsigned i=0;i<256;++i){double h=.02/(16+(i%4)*32),p=.3*std::sin(i*.71),s=.4*std::cos(i*.32),t=.4*std::sin(i*.13);error=std::max(error,std::abs(implicitVelocity(mass,.3,.015,.01,h,p,s,t)-bisectVelocity(mass,.3,.015,.01,h,p,s,t)));}
    require(error<1e-12,"Independent scalar saturation solve disagrees");std::cout<<std::setprecision(17)<<"{\"model_valid\":true,\"effective_inertia\":"<<mass<<",\"fp64_mass_acceleration_error\":"<<std::abs(acc[0]-tau[0]/mass)<<",\"scalar_closed_form_vs_bisection_256_max_velocity_error\":"<<error<<"}\n";
}
void execute(const char*shader,const char*output){
    const auto m=model();CompiledWorld world;const auto compiled=compileMetalWorld(m,0,world);require(compiled.succeeded(),"Compile analytic hinge: "+compiled.message);
    require(world.nq()==1&&world.nv()==1&&world.sceneBodyCount()==0&&world.eligiblePairCount()==0,"Analytic world contains unexpected dynamics/contact state");
    const std::vector<float>q{0,.1f,0,0,0,0,0,.1f},v{0,0,.002f,0,0,-.3f,.3f,.02f},target{0,.10005f,0,.4f,-.4f,.4f,-.4f,.1f};
    const MetalWorldBatch batch{.environmentCount=q.size(),.controlStepCount=1,.initialQ=q,.initialV=v,.efforts=target};
    MetalWorldConfig cfg;cfg.metallibPath=shader;MetalWorldContext context(cfg);
    std::ofstream out(output);require(bool(out),"Cannot create native hinge observations");out<<std::setprecision(17)<<"path\tsubstep_equivalent\tlane\th\tq\tv\ttarget\tmass\tk\td\tlimit\tfinal_q\tfinal_v\taba_flags\taccepted\n";
    for(unsigned path=0;path<3;++path)for(unsigned cadence:{16u,32u,64u,128u}){
        MetalWorldStepConfig step;step.timestepSeconds=.02f/cadence;step.physicsSubsteps=1;step.applyBodyDamping=false;
        step.solverMode=path==0?MetalWorldSolverMode::freeMotionABA:MetalWorldSolverMode::temporalCone;
        step.ccdMode=path==2?MetalWorldCCDMode::hybrid:MetalWorldCCDMode::disabled;step.actuationMode=MetalWorldActuationMode::implicitPositionDrive;
        MetalWorldResult result;const auto r=context.run(world,batch,step,result);
        require(r.succeeded()&&!r.failedStepCount&&r.successfulStepCount==q.size(),"Native hinge step rejected: "+r.message);
        require(result.finalQ.size()==q.size()&&result.finalV.size()==q.size(),"Incomplete native hinge state");
        for(unsigned lane=0;lane<q.size();++lane){
            const auto&d=m.dofs[0];out<<path<<'\t'<<cadence<<'\t'<<lane<<'\t'<<step.timestepSeconds<<'\t'<<q[lane]<<'\t'<<v[lane]<<'\t'<<target[lane]<<'\t'<<double(m.bodies[1].inertiaRow2.z)+d.drive.z<<'\t'<<d.drive.x<<'\t'<<d.drive.y<<'\t'<<d.limits.w<<'\t'<<result.finalQ[lane]<<'\t'<<result.finalV[lane]<<'\t'<<result.layout.abaDispatch.flags<<'\t'<<r.successfulStepCount<<'\n';
        }
    }
    require(bool(out),"Native hinge evidence write failed");std::cout<<"96 native one-microstep hinge observations completed\n";
}
}
int main(int argc,char**argv){try{if(argc==2&&std::string(argv[1])=="--calibrate"){calibrate();return 0;}require(argc==3,"Usage: probe --calibrate OR probe METALLIB NEW_TSV");execute(argv[1],argv[2]);return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

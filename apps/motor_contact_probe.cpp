#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalWorld.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace metalrobo;
namespace {
void need(bool b,const std::string& s){if(!b)throw std::runtime_error(s);}
EngineModel model(unsigned variant,bool contact){
 auto m=makeFreeSphereEngineModel();m.name="bounded_motor_contact_reaction";
 const float mass=variant==0?.004f:variant==1?.011f:variant==2?.007f:.019f;
 const float arm=variant%2==0?.001f:.002f;
 auto root=m.bodies[1];root.articulationIndex=0;root.parentBody=root.inboundJoint=MR_INVALID_INDEX;
 auto child=root;child.parentBody=1;child.inboundJoint=0;child.massAndInverseMass={mass,1/mass,0,0};
 child.inertiaRow0={.000001f,0,0,0};child.inertiaRow1={0,.000001f,0,0};child.inertiaRow2={0,0,.000001f,0};
 child.inverseInertiaRow0={1000000,0,0,0};child.inverseInertiaRow1={0,1000000,0,0};child.inverseInertiaRow2={0,0,1000000,0};
 m.bodies[1]=root;m.bodies.push_back(child);
 MRArticulationGPU a{};a.rootBody=a.firstBody=1;a.rootType=MR_ROOT_FIXED;a.bodyCount=2;a.jointCount=1;a.nq=a.nv=1;m.articulations={a};
 MRJointDescriptorGPU j{};j.parentBody=1;j.childBody=2;j.jointType=MR_JOINT_PRISMATIC;j.nq=j.nv=1;j.axis0={0,1,0,0};j.parentAnchor={0,.01f,0,0};j.parentRotation=j.childRotation={0,0,0,1};m.joints={j};
 MRDofPropertiesGPU d{};d.flags=MR_DOF_FLAG_ACTUATED|MR_DOF_FLAG_DRIVE|MR_DOF_FLAG_EFFORT_LIMIT;d.limits={0,0,0,.01f};d.drive={300,8,arm,0};m.dofs={d};
 m.shapes[1].bodyIndex=2;m.shapes[1].dimensions={.01f,0,0,0};m.shapes[1].contactRestAndBoundingRadius={0,0,.01f,0};m.shapes[1].collisionMask=contact?~0u:0u;
 m.materials[0].friction={0,0,0,0};m.materials[0].response={0,0,0,0};m.materials[0].geometry={0,0,0,0};
 m.defaultQ={0};m.defaultV={0};m.world.bodyCount=3;m.world.jointCount=1;m.world.nq=m.world.nv=1;m.world.gravityAndTimestep={0,0,0,.001f};
 std::string reason;need(m.valid(&reason),"Invalid slider: "+reason);return m;
}
double mass(const EngineModel& m){
 std::array<double,1>q{0},M{};ArticulatedDynamicsConfig c;c.gravity={0,0,0};c.applyBodyDamping=false;
 need(computeArticulatedMassMatrix(m,0,q,M,c).succeeded(),"Independent mass failed");return M[0];
}
void calibrate(){double error=0;unsigned conditions=0;
 for(unsigned variant=0;variant<4;++variant)for(bool contact:{false,true}){
  const auto m=model(variant,contact);const double M=mass(m),expected=double(m.bodies[2].massAndInverseMass.x)+m.dofs[0].drive.z;error=std::max(error,std::abs(M-expected));
  CompiledWorld w;const auto c=compileMetalWorld(m,0,w);need(c.succeeded()&&w.nq()==1&&w.nv()==1&&w.sceneBodyCount()==1&&w.eligiblePairCount()==unsigned(contact),"Slider host compilation failed: "+c.message);++conditions;
 }
 need(error<=1e-14,"Slider reference mass mismatch");
 std::cout<<std::setprecision(17)<<"{\"conditions\":"<<conditions<<",\"maximum_mass_error\":"<<error<<"}\n";
}
void execute(const char* shader,const char* output,bool followup){
 MetalWorldConfig cfg;cfg.metallibPath=shader;MetalWorldContext context(cfg);
 std::ofstream out(output);need(bool(out),"Cannot create native output");
 out<<std::setprecision(17)<<"variant\tcontact\tbounded\tlane\th\tmass\tk\td\tcap\ttarget\tcommand\tq\tv\tsmooth_acceleration\tapplied_force\tfinal_law_force\timpulse_y\tnormal_impulse\tmomentum_residual\tq_residual\tcontacts\tcontact_target\tcontact_regularization\tseparation\taccepted\n";
 const std::array<float,4> originalTargets{-.00002f,-.00005f,-.004f,.00002f};
 const std::array<float,4> followupTargets{-.00003f,-.00008f,-.003f,.00003f};
 const auto& targets=followup?followupTargets:originalTargets;
 const std::array<float,3> originalH{.00025f,.000625f,.001f};
 const std::array<float,3> followupH{.000375f,.00075f,.000875f};
 const auto& times=followup?followupH:originalH;
 unsigned accepted=0;
 // Native effort controls first, then free bounded controls, then contact test.
 for(unsigned phase=0;phase<3;++phase)for(unsigned v=0;v<2;++v)for(float h:times)for(unsigned enabled=0;enabled<(phase==0?2u:1u);++enabled){
  const bool bounded=phase!=0,contact=phase==0?enabled!=0:phase==2;const unsigned variant=v+(followup?2:0);const auto m=model(variant,contact);const double M=mass(m);
  CompiledWorld world;const auto c=compileMetalWorld(m,0,world);need(c.succeeded(),c.message);
  std::array<float,4>q{},vel{},commands{};std::array<MRBodyStateGPU,4>scene{};
  for(unsigned lane=0;lane<4;++lane){commands[lane]=bounded?targets[lane]:std::clamp(300*targets[lane],-.01f,.01f);scene[lane].position.w=scene[lane].orientation.w=1;scene[lane].flagsAndIndices[0]=MR_MOTION_STATIC;scene[lane].flagsAndIndices[1]=MR_INVALID_INDEX;scene[lane].flagsAndIndices[2]=0;}
  const MetalWorldBatch batch{.environmentCount=4,.controlStepCount=1,.initialQ=q,.initialV=vel,.efforts=commands,.initialSceneBodies=scene};
  MetalWorldStepConfig s;s.timestepSeconds=h;s.physicsSubsteps=1;s.applyBodyDamping=false;s.ccdMode=MetalWorldCCDMode::disabled;s.solverMode=MetalWorldSolverMode::temporalCone;s.actuationMode=bounded?MetalWorldActuationMode::boundedPositionDrive:MetalWorldActuationMode::effort;s.warmStart=false;s.captureContactEvidence=true;s.matrixFreeArticulatedContact=false;s.streamedArticulatedContactResponses=false;s.velocityIterations=s.finalVelocityIterations=8;
  MetalWorldResult r;const auto d=context.run(world,batch,s,r);
  need(d.succeeded()&&d.successfulStepCount==4&&d.failedStepCount==0,"Native slider invocation failed: "+d.message+" GPU="+std::to_string(d.firstGPUStatusCode));
  need(r.finalQ.size()==4&&r.finalV.size()==4&&r.accelerations.size()==4&&r.contactStatuses.size()==4,"Missing native slider measurements");accepted+=4;
  for(unsigned lane=0;lane<4;++lane){const auto& status=r.contactStatuses[lane];const unsigned stride=r.contactEvidence.contacts.size()/4;double p=0,normal=0,target=0,reg=0,sep=0;
   for(unsigned i=0;i<status.requiredConstraints;++i){need(i<stride,"Contact stride exceeded");const auto& row=r.contactEvidence.contacts[lane*stride+i];const double sign=row.bodyB==2?1:row.bodyA==2?-1:0;need(sign!=0,"Contact not incident on slider");p+=sign*row.normal.y*row.impulses.x;normal+=row.impulses.x;sep=row.pointAndSeparation.w;const auto& e=r.contactEvidence.evaluatedRows[(lane*stride+i)*3];target=e.targetVelocity;reg=e.regularization;}
   const double applied=M*r.accelerations[lane];const double law=std::clamp(300*(double(targets[lane])-r.finalQ[lane])-8*r.finalV[lane],-.01,.01);
   const double balance=M*r.finalV[lane]-h*applied-p,qr=r.finalQ[lane]-double(h)*r.finalV[lane];
   out<<variant<<'\t'<<contact<<'\t'<<bounded<<'\t'<<lane<<'\t'<<h<<'\t'<<M<<'\t'<<300<<'\t'<<8<<'\t'<<m.dofs[0].limits.w<<'\t'<<targets[lane]<<'\t'<<commands[lane]<<'\t'<<r.finalQ[lane]<<'\t'<<r.finalV[lane]<<'\t'<<r.accelerations[lane]<<'\t'<<applied<<'\t'<<law<<'\t'<<p<<'\t'<<normal<<'\t'<<balance<<'\t'<<qr<<'\t'<<status.requiredConstraints<<'\t'<<target<<'\t'<<reg<<'\t'<<sep<<'\t'<<1<<'\n';
   need(std::isfinite(applied)&&std::isfinite(law)&&std::abs(balance)<=2e-9&&std::abs(qr)<=2e-9&&normal>=-1e-12,"Native force/momentum instrument gate failed");
   if(!bounded||!contact){const double expected=bounded?std::clamp(300*double(targets[lane])/(1+(double(h)*8+double(h)*h*300)/M),-.01,.01):commands[lane];need(std::abs(applied-expected)<=1e-7,"Native force positive/null control failed");}
   if(!contact)need(status.requiredConstraints==0&&p==0,"Collision-off control produced contact");
  }
  out.flush();need(bool(out),"Native evidence write failed");std::cout<<"phase="<<phase<<" variant="<<variant<<" h="<<h<<" contact="<<contact<<" accepted=4 device="<<d.deviceName<<" gpu_ms="<<d.gpuElapsedMilliseconds<<'\n';
 }
 need(accepted==96,"Native slider sample budget mismatch");std::cout<<"96 accepted native slider conditions completed\n";
}
}
int main(int argc,char** argv){try{if(argc==2&&std::string(argv[1])=="--calibrate"){calibrate();return 0;}need(argc==3||argc==4,"Usage: motor-contact-probe --calibrate OR METALLIB NEW_TSV [--followup]");need(argc!=4||std::string(argv[3])=="--followup","Unknown selector");execute(argv[1],argv[2],argc==4);return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

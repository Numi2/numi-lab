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
 const std::array<float,6> masses{.004f,.011f,.007f,.019f,.009f,.017f};
 need(variant<masses.size(),"Unknown slider variant");const float mass=masses[variant];
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
void addMotorIR(EngineModel& m,float h,float target){
 const float denominator=8+h*300;
 const std::array<ConstraintIREndpoint,1> endpoints{makeConstraintIRGeneralizedEndpoint(0,0,0,0,1)};
 ConstraintIRRow row{};row.targetVelocity=300*target/denominator;row.compliance=h/denominator;row.impulseLower=-h*.01f;row.impulseUpper=h*.01f;
 const std::array<ConstraintIRRow,1> rows{row};const std::array<float,1> warm{0};
 ConstraintIRSourceBlock source{};source.key.words[0]=0x4d4f5452;source.endpoints=endpoints;source.rows=rows;source.warmImpulses=warm;
 const auto ir=compileConstraintIR(std::span(&source,1));need(ir.succeeded(),"Motor IR compilation failed: "+ir.diagnostics.message);m.constraintProgram=ir.ir;
}
void calibrateIR(bool fresh=false){
 double maxForceError=0;unsigned cases=0;
 const std::array<unsigned,2> variants=fresh?std::array<unsigned,2>{4,5}:std::array<unsigned,2>{2,3};
 const std::array<float,3> times=fresh?std::array<float,3>{.000425f,.000675f,.000925f}:std::array<float,3>{.000375f,.00075f,.000875f};
 const std::array<float,4> targets=fresh?std::array<float,4>{-.00004f,-.00007f,-.0025f,.00004f}:std::array<float,4>{-.00003f,-.00008f,-.003f,.00003f};
 for(unsigned variant:variants)for(float h:times)for(float target:targets){
  auto m=model(variant,false);const double M=mass(m);addMotorIR(m,h,target);CompiledWorld w;const auto c=compileMetalWorld(m,0,w);need(c.succeeded(),"Motor IR host compilation failed: "+c.message);
  const auto& row=m.constraintProgram.rows[0];const double reg=double(row.compliance)/(double(h)*h);
  // Independently derive the physical free motor impulse and the row's KKT law.
  const double expected=std::clamp(double(h)*300*target/(M+double(h)*8+double(h)*h*300),-double(h)*.01/M,double(h)*.01/M);
  const double impulse=std::clamp((double(row.targetVelocity)-expected)/reg,double(row.impulseLower),double(row.impulseUpper));
  maxForceError=std::max(maxForceError,std::abs(impulse/double(h)-M*expected/double(h)));++cases;
 }
 need(maxForceError<1e-7,"Motor IR analytic reference gate failed");std::cout<<std::setprecision(17)<<"{\"conditions\":"<<cases<<",\"maximum_force_mapping_error_N\":"<<maxForceError<<"}\n";
}
void executeCoupled(const char* shader,const char* output,unsigned selection=0){
 MetalWorldConfig cfg;cfg.metallibPath=shader;MetalWorldContext context(cfg);std::ofstream out(output);need(bool(out),"Cannot create coupled output");
 out<<std::setprecision(17)<<"variant\tcontact\tarm\tlane\th\tmass\ttarget\tq\tv\tapplied_force\tfinal_law_force\tcontact_impulse_y\tmotor_impulse\tmomentum_residual\tq_residual\tcontacts\tquality_optimality\tquality_feasibility\n";unsigned accepted=0;
 const bool fresh=selection==3;
 const std::array<unsigned,2> variants=fresh?std::array<unsigned,2>{4,5}:std::array<unsigned,2>{2,3};
 const std::array<float,3> times=fresh?std::array<float,3>{.000425f,.000675f,.000925f}:std::array<float,3>{.000375f,.00075f,.000875f};
 for(unsigned variant:variants)for(float h:times)for(bool contact:{false,true})for(unsigned lane=0;lane<4;++lane)for(unsigned arm=0;arm<3;++arm){
  const bool exposed=variant==2 && h==.000375f && !contact && lane==0;
  if((selection==1 && exposed)||(selection==2 && !(exposed && arm==1))||(selection==4 && !(variant==2 && h==.000875f && contact && lane==1 && arm==1)))continue;
  const std::array<float,4> targets=fresh?std::array<float,4>{-.00004f,-.00007f,-.0025f,.00004f}:std::array<float,4>{-.00003f,-.00008f,-.003f,.00003f};const float target=targets[lane];auto m=model(variant,contact);m.materials[0].friction={.5f,.5f,0,0};const double M=mass(m);
  if(arm==2)addMotorIR(m,h,target);
  CompiledWorld world;const auto c=compileMetalWorld(m,0,world);need(c.succeeded(),c.message);
  std::array<float,1> q{},v{},effort{arm==0?target:arm==1?float(std::clamp(300*double(target)/(1+(double(h)*8+double(h)*h*300)/M),-.01,.01)):0};
  MRBodyStateGPU scene{};scene.position.w=scene.orientation.w=1;scene.flagsAndIndices[0]=MR_MOTION_STATIC;scene.flagsAndIndices[1]=MR_INVALID_INDEX;scene.flagsAndIndices[2]=0;
  const MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,.initialQ=q,.initialV=v,.efforts=effort,.initialSceneBodies=std::span(&scene,1)};
  MetalWorldStepConfig s;s.timestepSeconds=h;s.physicsSubsteps=1;s.applyBodyDamping=false;s.ccdMode=MetalWorldCCDMode::disabled;s.solverMode=arm==0?MetalWorldSolverMode::temporalCone:MetalWorldSolverMode::qualityNewton;s.actuationMode=arm==0?MetalWorldActuationMode::boundedPositionDrive:MetalWorldActuationMode::effort;s.warmStart=false;s.captureContactEvidence=true;s.matrixFreeArticulatedContact=false;s.streamedArticulatedContactResponses=false;s.velocityIterations=s.finalVelocityIterations=8;s.quality.optimalityTolerance=selection>=3?1e-10f:1e-7f;s.quality.feasibilityTolerance=1e-7f;
  std::cout<<"begin variant="<<variant<<" h="<<h<<" contact="<<contact<<" lane="<<lane<<" arm="<<arm<<'\n'<<std::flush;
  MetalWorldResult r;const auto d=context.run(world,batch,s,r);
  if(!d.succeeded()){std::cerr<<"host="<<unsigned(d.status)<<" gpu="<<d.firstGPUStatusCode<<" message="<<d.message;if(!r.contactStatuses.empty()){const auto& z=r.contactStatuses[0];std::cerr<<" contact_code="<<z.code<<" constraint="<<z.firstFailingConstraint<<" residuals="<<z.residuals.x<<','<<z.residuals.y<<','<<z.residuals.z<<','<<z.residuals.w;}std::cerr<<'\n';}
  need(d.succeeded()&&d.successfulStepCount==1&&!d.failedStepCount,"Native coupled invocation failed");
  need(r.finalQ.size()==1&&r.finalV.size()==1&&r.accelerations.size()==1&&r.contactStatuses.size()==1,"Missing coupled measurement");++accepted;
  const auto& status=r.contactStatuses[0];double p=0,motor=0;
  for(unsigned i=0;i<status.requiredConstraints;++i){const auto& row=r.contactEvidence.contacts[i];if((row.flags&MR_CONSTRAINT_FLAG_GENERALIZED)!=0){motor+=row.impulses.x;continue;}const double sign=row.bodyB==2?1:row.bodyA==2?-1:0;need(sign!=0,"Invalid contact incidence");p+=sign*row.normal.y*row.impulses.x;}
  const double force=arm==2?motor/h:M*r.accelerations[0],law=std::clamp(300*(double(target)-r.finalQ[0])-8*r.finalV[0],-.01,.01),balance=M*r.finalV[0]-h*force-p,qr=r.finalQ[0]-double(h)*r.finalV[0];
  out<<variant<<'\t'<<contact<<'\t'<<arm<<'\t'<<lane<<'\t'<<h<<'\t'<<M<<'\t'<<target<<'\t'<<r.finalQ[0]<<'\t'<<r.finalV[0]<<'\t'<<force<<'\t'<<law<<'\t'<<p<<'\t'<<motor<<'\t'<<balance<<'\t'<<qr<<'\t'<<status.requiredConstraints<<'\t'<<status.residuals.y<<'\t'<<status.residuals.z<<'\n'<<std::flush;
  need(std::isfinite(force)&&std::isfinite(law)&&std::abs(balance)<=2e-9&&std::abs(qr)<=2e-9,"Coupled force/momentum validity gate failed");
 }
 const unsigned expected=selection==2||selection==4?1:selection==1?141:144;need(accepted==expected,"Coupled sample budget mismatch");std::cout<<expected<<" accepted native coupled comparison conditions completed\n";
}
}
int main(int argc,char** argv){try{if(argc==2&&std::string(argv[1])=="--calibrate-precision"){calibrateIR(true);return 0;}if(argc==4&&std::string(argv[3])=="--coupled-precision"){executeCoupled(argv[1],argv[2],3);return 0;}if(argc==4&&std::string(argv[3])=="--momentum-replay"){executeCoupled(argv[1],argv[2],4);return 0;}if(argc==4&&std::string(argv[3])=="--coupled-unused"){executeCoupled(argv[1],argv[2],1);return 0;}if(argc==4&&std::string(argv[3])=="--empty-quality-replay"){executeCoupled(argv[1],argv[2],2);return 0;}if(argc==2&&std::string(argv[1])=="--calibrate-ir"){calibrateIR();return 0;}if(argc==4&&std::string(argv[3])=="--coupled"){executeCoupled(argv[1],argv[2]);return 0;}if(argc==2&&std::string(argv[1])=="--calibrate"){calibrate();return 0;}need(argc==3||argc==4,"Usage: motor-contact-probe --calibrate OR METALLIB NEW_TSV [--followup]");need(argc!=4||std::string(argv[3])=="--followup","Unknown selector");execute(argv[1],argv[2],argc==4);return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

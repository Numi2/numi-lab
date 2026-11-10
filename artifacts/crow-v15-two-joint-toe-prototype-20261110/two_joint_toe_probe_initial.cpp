#include "metalrobo/RunProgram.hpp"
#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalArticulatedABA.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace metalrobo;
static void need(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
static void close(double a, double b, const char* why) { need(std::abs(a-b)<=2e-7*(1+std::abs(a)),why); }
static void diagonal(MRBodyPropertiesGPU& b,float x,float y,float z) {
  need(x>0 && y>0 && z>0,"nonpositive residual inertia");
  b.inertiaRow0={x,0,0,0}; b.inertiaRow1={0,y,0,0}; b.inertiaRow2={0,0,z,0};
  b.inverseInertiaRow0={1/x,0,0,0}; b.inverseInertiaRow1={0,1/y,0,0}; b.inverseInertiaRow2={0,0,1/z,0};
}
static RobotPack toeModel() {
  auto p=makeAmericanCrowActiveHeadRobotPack(); auto& m=p.mechanics;
  p.id="american_crow_toe_prototype_v0"; m.name=p.id; p.sourceRevision+="+exploratory-toes-v0";
  constexpr float mass=0.0001f, hx=0.018f, hy=0.0025f, hz=0.0092f;
  constexpr float ix=mass*(hy*hy+hz*hz)/3, iy=mass*(hx*hx+hz*hz)/3, iz=mass*(hx*hx+hy*hy)/3;
  for (int side=0;side<2;side++) {
    const std::string footName=side==0?"crow_left_foot":"crow_right_foot";
    auto it=std::find(m.bodyNames.begin(),m.bodyNames.end(),footName); need(it!=m.bodyNames.end(),"foot missing"); const unsigned foot=static_cast<unsigned>(it-m.bodyNames.begin()); auto& parent=m.bodies.at(foot);
    float dx[4]={.020666667f,.020666667f,.020666667f,-.062f},dy[4]={-.012f,0,.012f,0};
    float sx=0,sy=0,sz=0;
    for (int t=0;t<4;t++) { sx+=ix+mass*dy[t]*dy[t]; sy+=iy+mass*dx[t]*dx[t]; sz+=iz+mass*(dx[t]*dx[t]+dy[t]*dy[t]); }
    const float pm=parent.massAndInverseMass.x-4*mass; need(pm>0,"negative foot mass");
    parent.massAndInverseMass={pm,1/pm,0,0}; diagonal(parent,parent.inertiaRow0.x-sx,parent.inertiaRow1.y-sy,parent.inertiaRow2.z-sz);
    for (unsigned s=0;s<m.shapeNames.size();s++) if (m.shapeNames[s]==(side==0?"crow_left_foot/sole":"crow_right_foot/sole")) {
      auto& shape=m.shapes[s]; shape.dimensions.x=.010f;
      shape.contactRestAndBoundingRadius.z=std::sqrt(.010f*.010f+shape.dimensions.y*shape.dimensions.y+shape.dimensions.z*shape.dimensions.z);
    }
    for (int t=0;t<4;t++) {
      const unsigned bi=m.bodies.size(), ji=m.joints.size(), qi=m.defaultQ.size(), vi=m.defaultV.size();
      const float sign=t==3?-1.f:1.f;
      std::string name=std::string("crow_")+(side==0?"left_":"right_")+(t==3?"rear_toe":("front_toe_"+std::to_string(t+1)));
      MRBodyPropertiesGPU b{}; b.articulationIndex=0; b.parentBody=foot; b.inboundJoint=ji; b.motionType=MR_MOTION_DYNAMIC;
      b.massAndInverseMass={mass,1/mass,0,0}; diagonal(b,ix,iy,iz); b.dampingAndSpeedLimits={.015f,.015f,30,35}; m.bodies.push_back(b); m.bodyNames.push_back(name);
      MRJointDescriptorGPU j{}; j.parentBody=foot; j.childBody=bi; j.jointType=MR_JOINT_REVOLUTE; j.qOffset=qi; j.nq=1; j.vOffset=vi; j.nv=1;
      j.axis0={0,sign,0,0}; j.parentAnchor={dx[t]-sign*hx,dy[t],0,0}; j.childAnchor={-sign*hx,0,0,0}; j.parentRotation=j.childRotation={0,0,0,1};
      m.joints.push_back(j); m.jointNames.push_back(name+"_flex");
      MRDofPropertiesGPU d{}; d.articulationIndex=0; d.jointIndex=ji; d.qIndex=qi; d.vIndex=vi;
      d.flags=MR_DOF_FLAG_ACTUATED|MR_DOF_FLAG_POSITION_LIMIT|MR_DOF_FLAG_VELOCITY_LIMIT|MR_DOF_FLAG_EFFORT_LIMIT|MR_DOF_FLAG_DRIVE;
      d.limits={-.3f,1.2f,12.f,.02f}; d.drive={.1f,.003f,.00005f,0}; m.dofs.push_back(d); m.dofNames.push_back(name+"_flex"); m.defaultQ.push_back(0); m.defaultV.push_back(0);
      if (!m.actuatorProfiles.empty()) m.actuatorProfiles.emplace_back();
      MRShapeGPU shape=m.shapes.front(); shape.bodyIndex=bi; shape.localPosition={0,0,0,1}; shape.localRotation={0,0,0,1}; shape.dimensions={hx,hy,hz,0};
      shape.slotGeneration=m.shapes.size()+1; shape.contactRestAndBoundingRadius.z=std::sqrt(hx*hx+hy*hy+hz*hz); m.shapes.push_back(shape); m.shapeNames.push_back(name+"/collision");
    }
  }
  auto& a=m.articulations.at(0); a.bodyCount=m.world.bodyCount=m.bodies.size(); a.jointCount=m.world.jointCount=m.joints.size(); a.nq=m.world.nq=m.defaultQ.size(); a.nv=m.world.nv=m.defaultV.size(); m.world.shapeCount=m.shapes.size();
  for(auto& role:p.roles) { if(role.id=="whole_body")role.members=m.bodyNames; if(role.id=="all_joints")role.members=m.jointNames; if(role.id=="all_dofs")role.members=m.dofNames; }
  std::string reason; need(m.valid(&reason),"invalid model: "+reason); return p;
}
static RobotPack twoJointModel() {
 auto p=toeModel(); auto& m=p.mechanics; p.id="american_crow_v15_two_joint_toe_probe";m.name=p.id;p.sourceRevision+="+two-joint-toe-probe";
 constexpr float mass=0.00005f,hx=0.009f,hy=0.0025f,hz=0.0092f;
 constexpr float ix=mass*(hy*hy+hz*hz)/3,iy=mass*(hx*hx+hz*hz)/3,iz=mass*(hx*hx+hy*hy)/3;
 for(int side=0;side<2;side++) for(int t=0;t<4;t++) {
  const float sign=t==3?-1.f:1.f;
  const std::string name=std::string("crow_")+(side==0?"left_":"right_")+(t==3?"rear_toe":("front_toe_"+std::to_string(t+1)));
  auto bIt=std::find(m.bodyNames.begin(),m.bodyNames.end(),name);need(bIt!=m.bodyNames.end(),"proximal toe missing");unsigned parent=static_cast<unsigned>(bIt-m.bodyNames.begin());
  auto shIt=std::find(m.shapeNames.begin(),m.shapeNames.end(),name+"/collision");need(shIt!=m.shapeNames.end(),"toe shape missing");unsigned shapeIndex=static_cast<unsigned>(shIt-m.shapeNames.begin());
  MRBodyPropertiesGPU proximal=m.bodies[parent];proximal.massAndInverseMass={mass,1/mass,0,0};proximal.centerOfMass={-sign*hx,0,0,0};diagonal(proximal,ix,iy,iz);m.bodies[parent]=proximal;
  MRShapeGPU original=m.shapes[shapeIndex];m.shapes[shapeIndex].localPosition={-sign*hx,0,0,1};m.shapes[shapeIndex].dimensions={hx,hy,hz,0};m.shapes[shapeIndex].contactRestAndBoundingRadius.z=std::sqrt(hx*hx+hy*hy+hz*hz);
  const unsigned bi=m.bodies.size(),ji=m.joints.size(),qi=m.defaultQ.size(),vi=m.defaultV.size();MRBodyPropertiesGPU distal=proximal;distal.parentBody=parent;distal.inboundJoint=ji;distal.centerOfMass={0,0,0,0};m.bodies.push_back(distal);m.bodyNames.push_back(name+"_distal");
  MRJointDescriptorGPU j{};j.parentBody=parent;j.childBody=bi;j.jointType=MR_JOINT_REVOLUTE;j.qOffset=qi;j.nq=1;j.vOffset=vi;j.nv=1;j.axis0={0,sign,0,0};j.parentAnchor={0,0,0,0};j.childAnchor={-sign*hx,0,0,0};j.parentRotation=j.childRotation={0,0,0,1};m.joints.push_back(j);m.jointNames.push_back(name+"_distal_flex");
  MRDofPropertiesGPU d{};d.articulationIndex=0;d.jointIndex=ji;d.qIndex=qi;d.vIndex=vi;d.flags=MR_DOF_FLAG_ACTUATED|MR_DOF_FLAG_POSITION_LIMIT|MR_DOF_FLAG_VELOCITY_LIMIT|MR_DOF_FLAG_EFFORT_LIMIT|MR_DOF_FLAG_DRIVE;d.limits={-.3f,1.2f,12.f,.01f};d.drive={.05f,.0015f,.000025f,0};m.dofs.push_back(d);m.dofNames.push_back(name+"_distal_flex");m.defaultQ.push_back(0);m.defaultV.push_back(0);if(!m.actuatorProfiles.empty())m.actuatorProfiles.emplace_back();
  original.bodyIndex=bi;original.localPosition={0,0,0,1};original.dimensions={hx,hy,hz,0};original.slotGeneration=m.shapes.size()+1;original.contactRestAndBoundingRadius.z=std::sqrt(hx*hx+hy*hy+hz*hz);m.shapes.push_back(original);m.shapeNames.push_back(name+"_distal/collision");
 }
 auto& a=m.articulations.at(0);a.bodyCount=m.world.bodyCount=m.bodies.size();a.jointCount=m.world.jointCount=m.joints.size();a.nq=m.world.nq=m.defaultQ.size();a.nv=m.world.nv=m.defaultV.size();m.world.shapeCount=m.shapes.size();
 for(auto& role:p.roles){if(role.id=="whole_body")role.members=m.bodyNames;if(role.id=="all_joints")role.members=m.jointNames;if(role.id=="all_dofs")role.members=m.dofNames;}
 std::string reason;need(m.valid(&reason),"invalid model: "+reason);return p;
}
int main(){try{
 auto original=toeModel(),candidate=twoJointModel();auto& a=original.mechanics;auto& b=candidate.mechanics;need(b.bodies.size()==a.bodies.size()+8 && b.defaultV.size()==a.defaultV.size()+8,"topology mismatch");
 for(int k=0;k<36;k++){
  std::vector<double> aq(a.defaultQ.begin(),a.defaultQ.end()),av(a.defaultV.size()),bq(b.defaultQ.begin(),b.defaultQ.end()),bv(b.defaultV.size());const double yaw=.17*k;aq[0]=.013*k;aq[1]=-.01*k;aq[2]+=.02*k;aq[5]=std::sin(yaw/2);aq[6]=std::cos(yaw/2);
  for(unsigned i=0;i<av.size();i++)av[i]=.3*std::sin(.23*k+.77*i);for(unsigned i=aq.size()-8;i<aq.size();i++)aq[i]=.04*(k%5);std::copy(aq.begin(),aq.end(),bq.begin());std::copy(av.begin(),av.end(),bv.begin());
  ArticulatedInvariants ai,bi;need(computeArticulatedInvariants(a,0,aq,av,ai).succeeded(),"old invariant failure");need(computeArticulatedInvariants(b,0,bq,bv,bi).succeeded(),"new invariant failure");
  close(ai.kineticEnergy,bi.kineticEnergy,"kinetic energy changed");close(ai.potentialEnergy,bi.potentialEnergy,"potential energy changed");for(int i=0;i<3;i++){close(ai.linearMomentum[i],bi.linearMomentum[i],"linear momentum changed");close(ai.angularMomentum[i],bi.angularMomentum[i],"angular momentum changed");}
 }
 const unsigned n=12,nq=b.defaultQ.size(),nv=b.defaultV.size();std::vector<float> q,v,e;for(unsigned k=0;k<n;k++){q.insert(q.end(),b.defaultQ.begin(),b.defaultQ.end());v.insert(v.end(),nv,0);e.insert(e.end(),nv,0);for(unsigned t=0;t<8;t++){q[k*nq+nq-8+t]=.025f*(k%4);e[k*nv+nv-8+t]=(t%2==0?1.f:-1.f)*.00001f;}}
 MetalArticulatedABAInput in{.articulationIndex=0u,.environmentCount=n,.q=q,.v=v,.effort=e,.applyBodyDamping=false};MetalArticulatedABAContext ctx;MetalArticulatedABAResult out,again;auto status=ctx.run(b,in,out);need(status.succeeded()&&status.published&&status.failedEnvironmentCount==0,"GPU failed: "+status.message);auto repeat=ctx.run(b,in,again);need(repeat.succeeded()&&again.acceleration==out.acceleration&&again.nextQ==out.nextQ&&again.nextV==out.nextV,"GPU replay mismatch");
 double error=0;for(unsigned k=0;k<n;k++){std::vector<double> cq(q.begin()+k*nq,q.begin()+(k+1)*nq),cv(v.begin()+k*nv,v.begin()+(k+1)*nv),ce(e.begin()+k*nv,e.begin()+(k+1)*nv),acc(nv);ArticulatedDynamicsConfig config;config.timestep=b.world.gravityAndTimestep.w;config.applyBodyDamping=false;auto s=computeArticulatedForwardDynamics(b,0,cq,cv,ce,{},acc,config);need(s.succeeded(),"oracle dynamics failure");for(unsigned j=0;j<nv;j++)error=std::max(error,std::abs(out.acceleration[k*nv+j]-acc[j])/(1+std::abs(acc[j])));for(unsigned j=nv-8;j<nv;j++)need(std::abs(out.nextV[k*nv+j])>1e-7&&std::isfinite(out.nextV[k*nv+j]),"distal toe did not actuate");}
 need(error<5e-5,"GPU and oracle disagree");std::cout<<std::setprecision(17)<<"{\"status\":\"passed\",\"bodies\":"<<b.bodies.size()<<",\"dofs\":"<<nv<<",\"added_distal_hinges\":8,\"neutral_invariant_cases\":36,\"gpu_lanes\":12,\"acceleration_scaled_error\":"<<error<<",\"device\":"<<std::quoted(status.deviceName)<<",\"boundary\":\"estimated free-space mechanics only; no contact, grip, flight or policy qualification\"}"<<std::endl;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}

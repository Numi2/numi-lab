#pragma once
#include <metalrobo/RunProgram.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include "wing_model.hpp"
namespace { using namespace metalrobo;
static void crowToeNeed(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
static void close(double a, double b, const char* why) { crowToeNeed(std::abs(a-b)<=2e-7*(1+std::abs(a)),why); }
static void crowToeDiagonal(MRBodyPropertiesGPU& b,float x,float y,float z) {
  crowToeNeed(x>0 && y>0 && z>0,"nonpositive residual inertia");
  b.inertiaRow0={x,0,0,0}; b.inertiaRow1={0,y,0,0}; b.inertiaRow2={0,0,z,0};
  b.inverseInertiaRow0={1/x,0,0,0}; b.inverseInertiaRow1={0,1/y,0,0}; b.inverseInertiaRow2={0,0,1/z,0};
}
static RobotPack crowToeModel() {
  auto p=makeAmericanCrowActiveHeadRobotPack(); auto& m=p.mechanics;
  p.id="american_crow_toe_two_segment_prototype_v0"; m.name=p.id; p.sourceRevision+="+exploratory-toes-two-segment-v0";
  constexpr float mass=0.00005f, hx=0.009f, hy=0.0025f, hz=0.0092f;
  constexpr float ix=mass*(hy*hy+hz*hz)/3, iy=mass*(hx*hx+hz*hz)/3, iz=mass*(hx*hx+hy*hy)/3;
  for (int side=0;side<2;side++) {
    const std::string footName=side==0?"crow_left_foot":"crow_right_foot";
    auto it=std::find(m.bodyNames.begin(),m.bodyNames.end(),footName); crowToeNeed(it!=m.bodyNames.end(),"foot missing"); const unsigned foot=static_cast<unsigned>(it-m.bodyNames.begin()); auto& parent=m.bodies.at(foot);
    const float dx[4]={.020666667f,.020666667f,.020666667f,-.062f},dy[4]={-.012f,0,.012f,0};
    float sx=0,sy=0,sz=0;
    for(int t=0;t<4;t++) for(int seg=0;seg<2;seg++) { const float sign=t==3?-1.f:1.f; const float x=dx[t]+sign*(seg==0?-hx:hx); sx+=ix+mass*dy[t]*dy[t]; sy+=iy+mass*x*x; sz+=iz+mass*(x*x+dy[t]*dy[t]); }
    const float pm=parent.massAndInverseMass.x-8*mass;crowToeNeed(pm>0,"negative foot mass");parent.massAndInverseMass={pm,1/pm,0,0};crowToeDiagonal(parent,parent.inertiaRow0.x-sx,parent.inertiaRow1.y-sy,parent.inertiaRow2.z-sz);
    for(unsigned si=0;si<m.shapeNames.size();si++)if(m.shapeNames[si]==(side==0?"crow_left_foot/sole":"crow_right_foot/sole"))m.shapes[si].dimensions.x=.010f;
    for(int t=0;t<4;t++) {
      const float sign=t==3?-1.f:1.f;
      const std::string digit=std::string("crow_")+(side==0?"left_":"right_")+(t==3?"rear_toe":("front_toe_"+std::to_string(t+1)));
      unsigned previous=foot;
      for(int seg=0;seg<2;seg++) {
        const unsigned bi=m.bodies.size(),ji=m.joints.size(),qi=m.defaultQ.size(),vi=m.defaultV.size();const std::string name=digit+(seg==0?"_proximal":"_distal");
        MRBodyPropertiesGPU b{};b.articulationIndex=0;b.parentBody=previous;b.inboundJoint=ji;b.motionType=MR_MOTION_DYNAMIC;b.massAndInverseMass={mass,1/mass,0,0};crowToeDiagonal(b,ix,iy,iz);b.dampingAndSpeedLimits={0,0,30,35};m.bodies.push_back(b);m.bodyNames.push_back(name);
        MRJointDescriptorGPU j{};j.parentBody=previous;j.childBody=bi;j.jointType=MR_JOINT_REVOLUTE;j.qOffset=qi;j.nq=1;j.vOffset=vi;j.nv=1;j.axis0={0,sign,0,0};if(seg==0)j.parentAnchor={dx[t]-sign*2*hx,dy[t],0,0};else j.parentAnchor={sign*hx,0,0,0};j.childAnchor={-sign*hx,0,0,0};j.parentRotation=j.childRotation={0,0,0,1};m.joints.push_back(j);m.jointNames.push_back(name+"_flex");
        MRDofPropertiesGPU d{};d.articulationIndex=0;d.jointIndex=ji;d.qIndex=qi;d.vIndex=vi;d.flags=MR_DOF_FLAG_ACTUATED|MR_DOF_FLAG_POSITION_LIMIT|MR_DOF_FLAG_VELOCITY_LIMIT|MR_DOF_FLAG_EFFORT_LIMIT|MR_DOF_FLAG_DRIVE;d.limits={-.3f,1.2f,12.f,.02f};d.drive={.1f,.003f,.00005f,0};m.dofs.push_back(d);m.dofNames.push_back(name+"_flex");m.defaultQ.push_back(0);m.defaultV.push_back(0);if(!m.actuatorProfiles.empty())m.actuatorProfiles.emplace_back();
        MRShapeGPU shape=m.shapes.front();shape.bodyIndex=bi;shape.localPosition={0,0,0,1};shape.localRotation={0,0,0,1};shape.dimensions={hx,hy,hz,0};shape.slotGeneration=m.shapes.size()+1;shape.contactRestAndBoundingRadius.z=std::sqrt(hx*hx+hy*hy+hz*hz);m.shapes.push_back(shape);m.shapeNames.push_back(name+"/collision");previous=bi;
      }
    }
  }
  auto& a=m.articulations.at(0);a.bodyCount=m.world.bodyCount=m.bodies.size();a.jointCount=m.world.jointCount=m.joints.size();a.nq=m.world.nq=m.defaultQ.size();a.nv=m.world.nv=m.defaultV.size();m.world.shapeCount=m.shapes.size();
  for(auto& role:p.roles){if(role.id=="whole_body")role.members=m.bodyNames;if(role.id=="all_joints")role.members=m.jointNames;if(role.id=="all_dofs")role.members=m.dofNames;}
  std::string reason;crowToeNeed(m.valid(&reason),"invalid model: "+reason);return p;
}
static RobotPack crowHipRollModel(){
 auto p=crowToeModel();crowResearchWingInertia(p);auto &m=p.mechanics;p.id="crow_v15_experimental_hip_roll";p.sourceRevision+="+hip_roll_experiment";
 auto oldBodies=m.bodies;auto oldNames=m.bodyNames;auto oldJoints=m.joints;
 auto map=[](unsigned i){return i+(i>=8u)+(i>=11u);};
 m.bodies.clear();m.bodyNames.clear();
 const unsigned newJointBase=m.joints.size(),newQ=m.defaultQ.size(),newV=m.defaultV.size();
 for(unsigned i=0;i<oldBodies.size();++i){
  if(i==8u||i==11u){unsigned side=i==8u?0u:1u;auto proxy=oldBodies[side==0?1:2];proxy.parentBody=0;proxy.inboundJoint=newJointBase+side;proxy.centerOfMass={0,0,0,0};m.bodies.push_back(proxy);m.bodyNames.push_back(side==0?"crow_left_hip_roll_link":"crow_right_hip_roll_link");}
  auto b=oldBodies[i];if(b.parentBody!=MR_INVALID_INDEX)b.parentBody=map(b.parentBody);if(i==8u||i==11u)b.parentBody=map(i)-1u;m.bodies.push_back(b);m.bodyNames.push_back(oldNames[i]);
 }
 for(unsigned i=0;i<m.joints.size();++i){auto &j=m.joints[i];j.parentBody=map(j.parentBody);j.childBody=map(j.childBody);if(i==7u||i==10u){j.parentBody=j.childBody-1u;j.parentAnchor={0,0,0,0};}}
 for(auto& s:m.shapes)s.bodyIndex=map(s.bodyIndex);
 // Preserve precisely the torso/thigh shape exclusions that came from direct
 // adjacency before the added hip-roll proxy links changed that topology.
 for(unsigned i=0;i<m.shapes.size();++i)for(unsigned j=i+1;j<m.shapes.size();++j){const auto a=m.shapes[i].bodyIndex,b=m.shapes[j].bodyIndex;if((a==0u && (b==map(8u)||b==map(11u))) || (b==0u && (a==map(8u)||a==map(11u))))m.collisionExclusions.push_back({i,j});}
 std::sort(m.collisionExclusions.begin(),m.collisionExclusions.end(),[](const auto&a,const auto&b){return a.colliderA<b.colliderA || (a.colliderA==b.colliderA && a.colliderB<b.colliderB);});m.collisionExclusions.erase(std::unique(m.collisionExclusions.begin(),m.collisionExclusions.end(),[](const auto&a,const auto&b){return a.colliderA==b.colliderA && a.colliderB==b.colliderB;}),m.collisionExclusions.end());
 for(unsigned side=0;side<2;++side){unsigned oldJ=side==0?7u:10u;auto j=oldJoints[oldJ];j.childBody=map(j.childBody)-1u;j.parentBody=0;j.qOffset=newQ+side;j.vOffset=newV+side;j.axis0={1,0,0,0};j.childAnchor={0,0,0,0};m.joints.push_back(j);m.jointNames.push_back(side==0?"crow_left_hip_roll":"crow_right_hip_roll");auto d=m.dofs.at(oldJoints[oldJ].vOffset);d.jointIndex=newJointBase+side;d.qIndex=newQ+side;d.vIndex=newV+side;d.limits={-.35f,.35f,8.f,.22f};d.drive={8.f,.15f,.00005f,0};m.dofs.push_back(d);m.dofNames.push_back(side==0?"crow_left_hip_roll":"crow_right_hip_roll");m.defaultQ.push_back(0);m.defaultV.push_back(0);if(!m.actuatorProfiles.empty())m.actuatorProfiles.emplace_back();}
 auto&a=m.articulations.at(0);a.bodyCount=m.world.bodyCount=m.bodies.size();a.jointCount=m.world.jointCount=m.joints.size();a.nq=m.world.nq=m.defaultQ.size();a.nv=m.world.nv=m.defaultV.size();
 for(auto& r:p.roles){if(r.id=="whole_body")r.members=m.bodyNames;if(r.id=="all_joints")r.members=m.jointNames;if(r.id=="all_dofs")r.members=m.dofNames;}
 for(unsigned i : {8u,9u}){auto &d=m.dofs.at(i);crowToeNeed(d.vIndex==i && d.qIndex==i+1 && d.drive.x==800.f && d.drive.y==8.f,"unexpected wing drive metadata");std::fprintf(stderr,"CROW_MODEL_DRIVE %u %u %u %.9g %.9g %.9g %.9g\n",i,d.qIndex,d.vIndex,d.drive.x,d.drive.y,d.drive.z,d.limits.w);d.drive.x=10.f;d.drive.y=0.65f;std::fprintf(stderr,"CROW_MODEL_DRIVE_NEW %u %.9g %.9g %.9g %.9g\n",i,d.drive.x,d.drive.y,d.drive.z,d.limits.w);}
 for(unsigned i=6u;i<m.dofs.size();++i){const auto& d=m.dofs[i];std::fprintf(stderr,"CROW_MODEL_DOF_LIMIT %u %u %u %.9g %s\n",i,d.vIndex,d.flags,d.limits.z,m.dofNames[i].c_str());}
 for(unsigned i=10u;i<=11u;++i){const auto& d=m.dofs[i];std::fprintf(stderr,"CROW_PRONATION_MODEL %u %u %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",i,d.flags,d.drive.x,d.drive.y,d.drive.z,d.drive.w,d.limits.x,d.limits.y,d.limits.z,d.limits.w);}
 crowToeNeed(m.joints.at(4).axis0.x==0.f && m.joints.at(4).axis0.y==1.f && m.joints.at(4).axis0.z==0.f && m.joints.at(5).axis0.x==0.f && m.joints.at(5).axis0.y==-1.f && m.joints.at(5).axis0.z==0.f,"unexpected pronation axes");
 m.joints.at(5).axis0.y=1.f; p.sourceRevision+="+right_pronation_mirror_axis";
 for(unsigned i=0u;i<6u;++i){const auto& j=m.joints[i];std::fprintf(stderr,"CROW_WING_FRAME %u %u %u %u %u %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",i,j.parentBody,j.childBody,j.qOffset,j.vOffset,j.axis0.x,j.axis0.y,j.axis0.z,j.parentAnchor.x,j.parentAnchor.y,j.parentAnchor.z,j.childAnchor.x,j.childAnchor.y,j.childAnchor.z,j.parentRotation.x,j.parentRotation.y,j.parentRotation.z,j.parentRotation.w,j.childRotation.x,j.childRotation.y,j.childRotation.z,j.childRotation.w);}
 std::string why;crowToeNeed(m.valid(&why),"model invalid: "+why);return p;
}
}


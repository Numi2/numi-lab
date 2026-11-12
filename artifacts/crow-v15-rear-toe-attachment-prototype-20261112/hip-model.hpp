#define main toe_probe_main
#include "../crow-v15-two-segment-toe-prototype-20261111/probe.cpp"
#undef main
#include "../crow-v15-landing-com-diagnostic-20261111/wing_model.hpp"
#include <iomanip>
static RobotPack hipRollModel(){
 auto p=toeModel();crowResearchWingInertia(p);auto &m=p.mechanics;p.id="crow_v15_experimental_hip_roll";p.sourceRevision+="+hip_roll_experiment";
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
 std::string why;need(m.valid(&why),"model invalid: "+why);return p;
}

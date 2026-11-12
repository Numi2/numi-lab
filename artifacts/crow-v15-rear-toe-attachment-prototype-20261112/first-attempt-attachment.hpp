#include "hip-model.hpp"
static RobotPack attachedModel(){
 auto p=hipRollModel();auto&m=p.mechanics;
 for(const char* side:{"left","right"}){
  const std::string prefix=std::string("crow_")+side;
  const auto findName=[](const auto&v,const std::string&n){auto it=std::find(v.begin(),v.end(),n);need(it!=v.end(),"missing "+n);return static_cast<unsigned>(it-v.begin());};
  auto& sole=m.shapes.at(findName(m.shapeNames,prefix+"_foot/sole"));auto& j=m.joints.at(findName(m.jointNames,prefix+"_rear_toe_proximal_flex"));auto& distal=m.joints.at(findName(m.jointNames,prefix+"_rear_toe_distal_flex"));auto& foot=m.bodies.at(j.parentBody);const auto& proxBody=m.bodies.at(j.childBody);const auto& distalBody=m.bodies.at(distal.childBody);
  need(sole.bodyIndex==j.parentBody && distal.parentBody==j.childBody,"toe topology mismatch");need(sole.localRotation.x==0&&sole.localRotation.y==0&&sole.localRotation.z==0&&sole.localRotation.w==1,"sole rotation");need(foot.centerOfMass.x==0&&foot.centerOfMass.y==0&&foot.centerOfMass.z==0,"unexpected initial foot COM");
  const double goal=sole.localPosition.x-sole.dimensions.x;const double dx=goal-j.parentAnchor.x;const double x0=j.parentAnchor.x-j.childAnchor.x;const double x1=x0+distal.parentAnchor.x-distal.childAnchor.x;const double m0=proxBody.massAndInverseMass.x,m1=distalBody.massAndInverseMass.x,mp=foot.massAndInverseMass.x;const double comShift=-(m0+m1)*dx/mp;
  const double deltaI=m0*((x0+dx)*(x0+dx)-x0*x0)+m1*((x1+dx)*(x1+dx)-x1*x1)+mp*comShift*comShift;
  j.parentAnchor.x=static_cast<float>(goal);foot.centerOfMass.x=static_cast<float>(comShift);
  diagonal(foot,foot.inertiaRow0.x,static_cast<float>(foot.inertiaRow1.y-deltaI),static_cast<float>(foot.inertiaRow2.z-deltaI));
 }
 p.id="crow_v15_rear_attachment_experiment";p.sourceRevision+="+rear-collision-attachment-prototype";
 std::string why;need(m.valid(&why),"model invalid: "+why);return p;
}

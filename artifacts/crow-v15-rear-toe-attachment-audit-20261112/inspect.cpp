#define main prior_toe_probe_main
#include "../crow-v15-two-segment-toe-prototype-20261111/probe.cpp"
#undef main
#include <iomanip>
int main(){try{
 const auto p=toeModel();const auto&m=p.mechanics;std::cout<<std::setprecision(10);
 for(const char* side:{"left","right"}){
  const std::string foot=std::string("crow_")+side+"_foot/sole",toe=std::string("crow_")+side+"_rear_toe_proximal_flex",shape=std::string("crow_")+side+"_rear_toe_proximal/collision";
  auto si=std::find(m.shapeNames.begin(),m.shapeNames.end(),foot),ti=std::find(m.shapeNames.begin(),m.shapeNames.end(),shape),ji=std::find(m.jointNames.begin(),m.jointNames.end(),toe);
  need(si!=m.shapeNames.end() && ti!=m.shapeNames.end() && ji!=m.jointNames.end(),"shape or joint missing");
  const auto&s=m.shapes.at(si-m.shapeNames.begin());const auto&t=m.shapes.at(ti-m.shapeNames.begin());const auto&j=m.joints.at(ji-m.jointNames.begin());
  const double soleRear=s.localPosition.x-s.dimensions.x,anchor=j.parentAnchor.x,toeBase=j.parentAnchor.x;const double gap=soleRear-toeBase;
  std::cout<<side<<" sole_center_x "<<s.localPosition.x<<" sole_half_x "<<s.dimensions.x<<" sole_rotation "<<s.localRotation.x<<" "<<s.localRotation.y<<" "<<s.localRotation.z<<" "<<s.localRotation.w<<" rear_joint_parent_anchor_x "<<anchor<<" rear_toe_half_x "<<t.dimensions.x<<" neutral_gap_m "<<gap<<" foot_com_x "<<m.bodies[j.parentBody].centerOfMass.x<<"\n";
  need(std::abs(gap-0.034)<1e-6,"unexpected rear clearance gap");
 }
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}

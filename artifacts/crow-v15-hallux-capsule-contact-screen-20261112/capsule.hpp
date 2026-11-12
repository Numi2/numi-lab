#pragma once
#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
static RobotPack capsuleModel(){
 auto p=hipRollModel();auto&m=p.mechanics;
 for(const char* side:{"left","right"})for(const char* part:{"proximal","distal"}){
  std::string n=std::string("crow_")+side+"_rear_toe_"+part+"/collision";
  auto i=std::find(m.shapeNames.begin(),m.shapeNames.end(),n);need(i!=m.shapeNames.end(),"missing rear toe shape");auto&s=m.shapes.at(i-m.shapeNames.begin());
  need(s.shapeType==MR_SHAPE_BOX && std::abs(s.dimensions.x-.009f)<1e-7f && std::abs(s.dimensions.y-.0025f)<1e-7f && std::abs(s.dimensions.z-.0092f)<1e-7f,"unexpected original rear shape");
  s.shapeType=MR_SHAPE_CAPSULE;s.localPosition.z-=.0067f;s.localRotation={0.f,0.f,.7071067811865475f,.7071067811865475f};s.dimensions={.0025f,.0065f,0.f,0.f};s.contactRestAndBoundingRadius.z=.009f;
 }
 p.id="crow_v15_rear_capsule_experiment";p.sourceRevision+="+rear-capsule-prototype";std::string why;need(m.valid(&why),"invalid model "+why);return p;
}

#include "../crow-v15-rear-toe-attachment-prototype-20261112/attachment.hpp"
static RobotPack trimModel(){
 auto p=attachedModel();auto&m=p.mechanics;
 for(const char* side:{"left","right"}){
  const std::string name=std::string("crow_")+side+"_shank/collision";auto it=std::find(m.shapeNames.begin(),m.shapeNames.end(),name);need(it!=m.shapeNames.end(),"missing "+name);auto& s=m.shapes[it-m.shapeNames.begin()];need(s.dimensions.z>.003f,"short shank proxy");
  s.localPosition.z+=.003f;s.dimensions.z-=.003f;s.contactRestAndBoundingRadius.z=std::sqrt(s.dimensions.x*s.dimensions.x+s.dimensions.y*s.dimensions.y+s.dimensions.z*s.dimensions.z);
 }
 std::string why;need(m.valid(&why),"invalid model: "+why);return p;
}

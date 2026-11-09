#include "metalrobo/RunProgram.hpp"
#include <cstdio>
int main() {
 auto p=metalrobo::builtinRobotPack("american_crow_active_head_estimated_v1");
 if (!p) return 1;
 for (size_t i=0;i<p->mechanics.shapes.size();++i) {
  const auto& s=p->mechanics.shapes[i];
  std::printf("%s body=%s type=%u flags=%u group=%u mask=%u dims=%.4f,%.4f,%.4f\n",p->mechanics.shapeNames[i].c_str(),p->mechanics.bodyNames[s.bodyIndex].c_str(),s.shapeType,s.flags,s.collisionGroup,s.collisionMask,s.dimensions.x,s.dimensions.y,s.dimensions.z);
 }
}

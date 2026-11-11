#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
int main(){auto p=hipRollModel();for(unsigned i=0;i<p.mechanics.shapes.size();++i){auto s=p.mechanics.shapes[i];if(p.mechanics.shapeNames[i].find("toe")!=std::string::npos)std::cout<<p.mechanics.shapeNames[i]<<" body="<<s.bodyIndex<<" type="<<s.shapeType<<" d="<<s.dimensions.x<<","<<s.dimensions.y<<","<<s.dimensions.z<<" local="<<s.localPosition.x<<","<<s.localPosition.y<<","<<s.localPosition.z<<" flags="<<s.flags<<" mask="<<s.collisionMask<<"\n";}return 0;}

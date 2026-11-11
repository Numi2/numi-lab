#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iomanip>
int main(){try{auto robot=hipRollModel();const auto& m=robot.mechanics;std::cout<<std::setprecision(9);for(unsigned i=0;i<m.joints.size();++i){auto j=m.joints[i];std::cout<<i<<" "<<m.jointNames[i]<<" parent="<<j.parentBody<<" child="<<j.childBody<<" q="<<j.qOffset<<" axis="<<j.axis0.x<<","<<j.axis0.y<<","<<j.axis0.z<<" p="<<j.parentAnchor.x<<","<<j.parentAnchor.y<<","<<j.parentAnchor.z<<" c="<<j.childAnchor.x<<","<<j.childAnchor.y<<","<<j.childAnchor.z<<"\n";}for(unsigned i=0;i<m.bodyNames.size();++i)std::cout<<"body "<<i<<" "<<m.bodyNames[i]<<"\n";}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}

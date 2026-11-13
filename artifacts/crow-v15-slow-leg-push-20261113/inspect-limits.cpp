#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iomanip>
int main(){try{auto robot=hipRollModel();const auto&m=robot.mechanics;std::cout<<std::setprecision(12);for(unsigned j: {13u,14u,15u,16u,17u,18u}){const auto&d=m.dofs.at(j);std::cout<<j<<" "<<m.dofNames.at(j)<<" q="<<d.qIndex<<" flags="<<d.flags<<" lower="<<d.limits.x<<" upper="<<d.limits.y<<" speed="<<d.limits.z<<" effort="<<d.limits.w<<" kp="<<d.drive.x<<" kd="<<d.drive.y<<"\n";}return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}

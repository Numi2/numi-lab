#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iomanip>
int main(){try {const auto p=hipRollModel();const auto&m=p.mechanics;std::cout<<std::setprecision(10);for(unsigned i=0;i<m.shapeNames.size();i++){const auto&n=m.shapeNames[i];if(n.find("rear_toe")!=std::string::npos||n.find("foot/sole")!=std::string::npos){auto&s=m.shapes[i];std::cout<<n<<" half "<<s.dimensions.x<<" "<<s.dimensions.y<<" "<<s.dimensions.z<<" contact_offset "<<s.contactRestAndBoundingRadius.x<<" rest_offset "<<s.contactRestAndBoundingRadius.y<<" radius "<<s.contactRestAndBoundingRadius.z<<"\n";}}}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}

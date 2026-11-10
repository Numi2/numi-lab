#define main toe_free_admission_main
#include "../crow-v15-toe-grip-redamping-20261110/toe_model.cpp"
#undef main
#include <sstream>
#include <iomanip>
int main(){try{auto p=toeModel();auto&m=p.mechanics;std::string line;std::cout<<std::setprecision(12);while(std::getline(std::cin,line)){std::istringstream in(line);int t;in>>t;std::vector<double>q(m.defaultQ.size()),v(m.defaultV.size());for(auto&x:q)in>>x;for(auto&x:v)in>>x;need(bool(in),"bad state");std::vector<ArticulatedBodyKinematics>b(m.bodies.size());auto s=computeArticulatedBodyKinematics(m,0,q,v,b);need(s.succeeded(),"kinematics failure");for(unsigned i=0;i<b.size();i++){if(m.bodyNames[i].find("_foot")==std::string::npos&&m.bodyNames[i].find("_toe")==std::string::npos)continue;std::cout<<t<<" "<<m.bodyNames[i];for(auto x:b[i].centerOfMassPosition)std::cout<<" "<<x;for(auto x:b[i].linearVelocity)std::cout<<" "<<x;for(auto x:b[i].angularVelocity)std::cout<<" "<<x;std::cout<<"\n";}}}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}

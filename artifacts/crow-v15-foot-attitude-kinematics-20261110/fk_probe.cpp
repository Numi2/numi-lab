#define main toe_free_admission_main
#include "../crow-v15-toe-grip-redamping-20261110/toe_model.cpp"
#undef main
#include <sstream>
#include <iomanip>
int main(){try{auto p=toeModel();auto&m=p.mechanics;auto it=std::find(m.bodyNames.begin(),m.bodyNames.end(),"crow_left_foot");need(it!=m.bodyNames.end(),"foot missing");auto fi=it-m.bodyNames.begin();std::string line;std::cout<<std::setprecision(12);while(std::getline(std::cin,line)){std::istringstream in(line);int t;in>>t;std::vector<double>q;double z;while(in>>z)q.push_back(z);need(q.size()==m.defaultQ.size(),"bad q");std::vector<double>v(m.defaultV.size());std::vector<ArticulatedBodyKinematics>b(m.bodies.size());auto s=computeArticulatedBodyKinematics(m,0,q,v,b);need(s.succeeded(),"kinematics failure");std::cout<<t;for(auto a:b[fi].centerOfMassPosition)std::cout<<" "<<a;for(auto a:b[fi].orientation)std::cout<<" "<<a;std::cout<<"\n";}}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}

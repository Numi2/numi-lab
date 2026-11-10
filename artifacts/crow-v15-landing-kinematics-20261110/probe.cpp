#define main toe_free_admission_main
#include "../crow-v15-toe-grip-redamping-20261110/toe_model.cpp"
#undef main
#include <sstream>
#include <iomanip>
int main(){try{auto p=toeModel();auto&m=p.mechanics;std::string line;std::cout<<std::setprecision(10);
 while(std::getline(std::cin,line)){std::istringstream in(line);std::string key;in>>key;std::vector<double>q;double z;while(in>>z)q.push_back(z);need(q.size()==m.defaultQ.size(),"bad q");std::vector<double>v(m.defaultV.size());
 for(int sample=-1;sample<6;sample++){
  auto qq=q;if(sample>=0)qq[14+sample]+=.01;std::vector<ArticulatedBodyKinematics>b(m.bodies.size());auto s=computeArticulatedBodyKinematics(m,0,qq,v,b);need(s.succeeded(),"kinematics");
  std::cout<<key<<" "<<sample;
  for(unsigned i=0;i<b.size();i++)if(m.bodyNames[i].find("_foot")!=std::string::npos||m.bodyNames[i].find("_toe")!=std::string::npos)
   std::cout<<" "<<m.bodyNames[i]<<":"<<b[i].centerOfMassPosition[0]<<","<<b[i].centerOfMassPosition[1]<<","<<b[i].centerOfMassPosition[2];
  std::cout<<"\n";
 }
 }
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}

#define main toe_free_admission_main
#include "../crow-v15-toe-grip-redamping-20261110/toe_model.cpp"
#undef main
#include <sstream>
#include <iomanip>
int main(){try{auto p=toeModel();auto&m=p.mechanics;std::string line;std::cout<<std::setprecision(12);while(std::getline(std::cin,line)){std::istringstream in(line);int t;in>>t;std::vector<double>q;double z;while(in>>z)q.push_back(z);need(q.size()==m.defaultQ.size(),"bad q");std::vector<double>v(m.defaultV.size());std::vector<ArticulatedBodyKinematics>b(m.bodies.size());auto s=computeArticulatedBodyKinematics(m,0,q,v,b);need(s.succeeded(),"kinematics failure");double sum=0,x=0,y=0,zpos=0;for(unsigned i=0;i<b.size();i++){auto mass=m.bodies[i].massAndInverseMass.x;sum+=mass;x+=mass*b[i].centerOfMassPosition[0];y+=mass*b[i].centerOfMassPosition[1];zpos+=mass*b[i].centerOfMassPosition[2];}std::cout<<t<<" "<<sum<<" "<<x/sum<<" "<<y/sum<<" "<<zpos/sum<<"\n";}}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}

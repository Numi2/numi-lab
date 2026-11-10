#define main toe_free_admission_main
#include "../crow-v15-toe-grip-redamping-20261110/toe_model.cpp"
#undef main
#include "../crow-clearance-observer-20261109/box_distance.hpp"
#include <fstream>
#include <sstream>
#include <limits>
struct Out { double x,z,low; };
int main(){try{
 auto p=toeModel(); auto& m=p.mechanics;
 for(unsigned i=0;i<m.dofNames.size();i++) std::cout<<"dof "<<m.dofNames[i]<<" q "<<m.dofs[i].qIndex<<"\n";
 auto eval=[&](const std::vector<double>& q,const std::string& side){
  std::vector<double> v(m.defaultV.size()); std::vector<ArticulatedBodyKinematics> kin(m.bodies.size());
  auto s=computeArticulatedBodyKinematics(m,0,q,v,kin);need(s.succeeded(),"kinematics failed");
  std::string foot="crow_"+side+"_foot"; auto f=std::find(m.bodyNames.begin(),m.bodyNames.end(),foot);need(f!=m.bodyNames.end(),"foot missing"); auto fi=f-m.bodyNames.begin();
  double low=std::numeric_limits<double>::infinity();
  for(unsigned i=0;i<m.shapes.size();i++){auto& sh=m.shapes[i];auto bn=m.bodyNames.at(sh.bodyIndex);if(bn!=foot && bn.find("crow_"+side+"_front_toe_")!=0 && bn!="crow_"+side+"_rear_toe")continue;
   auto& k=kin.at(sh.bodyIndex);auto& c=k.centerOfMassPosition;auto& a=k.orientation;auto l=sh.localPosition;auto o=sh.localRotation;auto d=sh.dimensions;
   auto b=crowclear::box({c[0],c[1],c[2]},{a[0],a[1],a[2],a[3]},{l.x,l.y,l.z},{o.x,o.y,o.z,o.w},{d.x,d.y,d.z});
   double z=b.c.z;for(int j=0;j<3;j++)z-=std::abs(b.axis[j].z)*b.h[j];low=std::min(low,z);
  }
  return Out{kin.at(fi).centerOfMassPosition[0],kin.at(fi).centerOfMassPosition[2],low};
 };
 std::ifstream in("artifacts/crow-v15-gait-kinematic-jacobian-20261110/states.txt");std::string line;std::cout<<std::setprecision(12);
 while(std::getline(in,line)){std::istringstream ss(line);int step;ss>>step;std::vector<double> q;double x;while(ss>>x)q.push_back(x);need(q.size()==m.defaultQ.size(),"wrong input size");
  for(auto side:{std::string("left"),std::string("right")}){auto b=eval(q,side);std::cout<<"base "<<step<<" "<<side<<" "<<b.x<<" "<<b.z<<" "<<b.low<<"\n";
   for(std::string joint:{"hip","knee","ankle"}){std::string prefix="crow_"+side+"_"+joint;auto it=std::find_if(m.dofNames.begin(),m.dofNames.end(),[&](auto& n){return n.find(prefix)==0;});need(it!=m.dofNames.end(),"joint absent "+prefix);int idx=m.dofs[it-m.dofNames.begin()].qIndex;
    for(double h:{0.0001,0.0002}){auto a=q,c=q;a[idx]+=h;c[idx]-=h;auto ap=eval(a,side),am=eval(c,side);std::cout<<"deriv "<<step<<" "<<side<<" "<<*it<<" "<<h<<" "<<(ap.x-am.x)/(2*h)<<" "<<(ap.z-am.z)/(2*h)<<" "<<(ap.low-am.low)/(2*h)<<"\n";}
   }
  }
 }
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}

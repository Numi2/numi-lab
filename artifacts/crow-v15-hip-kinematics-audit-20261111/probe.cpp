#define main toe_free_admission_main
#include "toe_model.cpp"
#undef main
#include "wing_model.hpp"
int main(){try{auto p=toeModel();crowResearchWingInertia(p);auto& m=p.mechanics;auto q=m.defaultQ,v=m.defaultV;unsigned foot=0;while(foot<m.bodyNames.size()&&m.bodyNames[foot]!="crow_left_foot")foot++;need(foot<m.bodyNames.size(),"missing foot");for(double d:{-.001,0.,.001}){std::vector<double> qd(q.begin(),q.end()),vd(v.begin(),v.end());qd[14]+=d;qd[17]+=d;std::vector<ArticulatedBodyKinematics> k(m.bodies.size());auto s=computeArticulatedBodyKinematics(m,0,qd,vd,k);need(s.succeeded(),"kinematics");double mass=0,x=0;for(unsigned i=0;i<k.size();i++){double mi=m.bodies[i].massAndInverseMass.x;mass+=mi;x+=mi*k[i].centerOfMassPosition[0];}std::cout<<std::setprecision(12)<<d<<" "<<k[foot].centerOfMassPosition[0]<<" "<<x/mass<<"\n";}return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}

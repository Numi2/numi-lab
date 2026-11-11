#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <array>
#include <algorithm>
int main(){
 auto p=hipRollModel();auto &m=p.mechanics;
 for (const auto& d:m.dofs) if(d.qIndex>=14 && d.qIndex<=19) {
  std::cout << "joint " << d.qIndex << " limits " << d.limits.x << " " << d.limits.y << " default " << m.defaultQ[d.qIndex] << "\n";
 }
 std::vector<double> q(m.defaultQ.begin(),m.defaultQ.end()),v(m.defaultV.size(),0.0);
 std::vector<ArticulatedBodyKinematics> b(m.bodies.size());
 const auto foot=std::find(m.bodyNames.begin(),m.bodyNames.end(),"crow_left_foot")-m.bodyNames.begin();
 const auto body=std::find(m.bodyNames.begin(),m.bodyNames.end(),"crow_body")-m.bodyNames.begin();
 std::array<std::array<double,2>,3> bounds{};
 for(const auto& d:m.dofs) if(d.qIndex>=14 && d.qIndex<=16) bounds[d.qIndex-14]={d.limits.x,d.limits.y};
 struct Candidate{double height,x,hip,knee,ankle;};std::vector<Candidate> best;
 for(int i=0;i<=24;i++)for(int j=0;j<=24;j++)for(int k=0;k<=12;k++){
  q[14]=q[17]=bounds[0][0]+(bounds[0][1]-bounds[0][0])*i/24;
  q[15]=q[18]=bounds[1][0]+(bounds[1][1]-bounds[1][0])*j/24;
  q[16]=q[19]=bounds[2][0]+(bounds[2][1]-bounds[2][0])*k/12;
  if(!computeArticulatedBodyKinematics(m,0,q,v,b).succeeded())return 2;
  const auto& a=b[foot].centerOfMassPosition;const auto& c=b[body].centerOfMassPosition;
  best.push_back({a[2]-c[2],a[0]-c[0],q[14],q[15],q[16]});
 }
 std::sort(best.begin(),best.end(),[](const auto&a,const auto&b){return a.height>b.height;});
 for(int i=0;i<30;i++){auto a=best[i];std::cout<<"candidate "<<a.height<<" "<<a.x<<" "<<a.hip<<" "<<a.knee<<" "<<a.ankle<<"\n";}
}

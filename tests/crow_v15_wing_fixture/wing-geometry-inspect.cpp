#include "v15_fixture.hpp"
#include "metalrobo/CrowArticulatedWing.hpp"
#include "metalrobo/ArticulatedDynamics.hpp"
#include <iostream>
#include <iomanip>
int main(){
 for(int i=0;i<2;++i){auto p=crowHipRollModel();if(i)p=metalrobo::makeCrowArticulatedWingMechanicsFixture(std::move(p));auto&m=p.mechanics;
 std::vector<double> q(m.defaultQ.begin(),m.defaultQ.end()),v(m.defaultV.begin(),m.defaultV.end());std::vector<metalrobo::ArticulatedBodyKinematics> out(m.bodies.size());auto d=metalrobo::computeArticulatedBodyKinematics(m,0,q,v,out);std::cout<<"MODEL "<<p.id<<" kinematics "<<d.succeeded()<<'\n';
 for(size_t j=0;j<m.shapeNames.size();++j){auto&n=m.shapeNames[j];if(n.find("body/collision_body")==std::string::npos&&n.find("wing/collision")==std::string::npos&&n.find("forearm/collision")==std::string::npos&&n.find("hand/collision")==std::string::npos)continue;auto&s=m.shapes[j];auto&k=out[s.bodyIndex];auto&b=m.bodies[s.bodyIndex];std::cout<<std::setprecision(9)<<"SHAPE "<<n<<" body="<<s.bodyIndex<<" com_world="<<k.centerOfMassPosition[0]<<","<<k.centerOfMassPosition[1]<<","<<k.centerOfMassPosition[2]<<" quat="<<k.orientation[0]<<","<<k.orientation[1]<<","<<k.orientation[2]<<","<<k.orientation[3]<<" local="<<s.localPosition.x<<","<<s.localPosition.y<<","<<s.localPosition.z<<" half="<<s.dimensions.x<<","<<s.dimensions.y<<","<<s.dimensions.z<<" body_com="<<b.centerOfMass.x<<","<<b.centerOfMass.y<<","<<b.centerOfMass.z<<'\n';}
 }
}

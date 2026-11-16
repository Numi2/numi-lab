#include "v15_fixture.hpp"
#include "metalrobo/CrowArticulatedWing.hpp"
#include "metalrobo/ArticulatedDynamics.hpp"
#include <iomanip>
#include <algorithm>
#include <vector>
#include <string>
using namespace metalrobo;
template<class T> static void array(const T& a){std::cout<<'[';bool first=true;for(const auto& x:a){if(!first)std::cout<<',';first=false;std::cout<<x;}std::cout<<']';}
int main(){try{
 auto p=makeCrowArticulatedWingMechanicsFixture(crowHipRollModel());auto& m=p.mechanics;
 auto it=std::find(m.bodyNames.begin(),m.bodyNames.end(),"crow_left_hand");crowToeNeed(it!=m.bodyNames.end(),"hand missing");unsigned body=unsigned(it-m.bodyNames.begin());const auto& a=m.articulations.at(0);size_t n=a.nv;
 std::vector<double> q(m.defaultQ.begin(),m.defaultQ.end()),v(m.defaultV.begin(),m.defaultV.end()),J(3*n),M(n*n),X(3*n);
 std::vector<ArticulatedBodyKinematics> bodies(a.bodyCount);ArticulatedPointQuery point{body,{.008,.012,.003}};ArticulatedPointKinematics pk;
 auto k=computeArticulatedBodyKinematics(m,0,q,v,bodies);auto j=computeArticulatedPointJacobians(m,0,q,v,std::span(&point,1),std::span(&pk,1),J);auto mass=computeArticulatedMassMatrix(m,0,q,M);auto response=computeArticulatedInverseMassResponses(m,0,q,J,X);
 crowToeNeed(k.succeeded()&&j.succeeded()&&mass.succeeded()&&response.succeeded(),"native reference query failed");auto bt=std::find_if(bodies.begin(),bodies.end(),[&](const auto& b){return b.bodyIndex==body;});crowToeNeed(bt!=bodies.end(),"missing hand kinematics");
 double maxResidual=0, W[3][3]{};
 for(size_t axis=0;axis<3;++axis)for(size_t i=0;i<n;++i){double value=0;for(size_t k=0;k<n;++k)value+=M[i*n+k]*X[axis*n+k];maxResidual=std::max(maxResidual,std::abs(value-J[axis*n+i]));}
 for(size_t axis=0;axis<3;++axis)for(size_t other=0;other<3;++other)for(size_t i=0;i<n;++i)W[axis][other]+=J[axis*n+i]*X[other*n+i];
 double symmetry=0;for(size_t i=0;i<3;++i)for(size_t k=0;k<3;++k)symmetry=std::max(symmetry,std::abs(W[i][k]-W[k][i]));
 const double minor2=W[0][0]*W[1][1]-W[0][1]*W[1][0];const double det=W[0][0]*(W[1][1]*W[2][2]-W[1][2]*W[2][1])-W[0][1]*(W[1][0]*W[2][2]-W[1][2]*W[2][0])+W[0][2]*(W[1][0]*W[2][1]-W[1][1]*W[2][0]);
 auto elbow=std::find(m.dofNames.begin(),m.dofNames.end(),"crow_left_elbow_flex");crowToeNeed(elbow!=m.dofNames.end(),"elbow missing");auto ei=m.dofs.at(elbow-m.dofNames.begin()).vIndex;double elbowResponse=0;for(size_t axis=0;axis<3;++axis)elbowResponse+=std::abs(X[axis*n+ei]);
 crowToeNeed(std::isfinite(maxResidual)&&maxResidual<1e-8&&symmetry<1e-8&&W[0][0]>0&&minor2>0&&det>0&&elbowResponse>1e-12,"articulated response reference failed");
 std::cout<<std::setprecision(17)<<"{\"nv\":"<<n<<",\"body\":"<<body<<",\"hand_position\":";array(bt->centerOfMassPosition);std::cout<<",\"hand_orientation_xyzw\":";array(bt->orientation);std::cout<<",\"point\":";array(pk.position);std::cout<<",\"J\":";array(J);std::cout<<",\"M\":";array(M);std::cout<<",\"X\":";array(X);std::cout<<",\"dofs\":[";for(size_t i=0;i<m.dofNames.size();++i){if(i)std::cout<<',';std::cout<<"{\"name\":"<<std::quoted(m.dofNames[i])<<",\"v\":"<<m.dofs[i].vIndex<<'}';}std::cout<<"]}\n";
 return 0;
}catch(const std::exception& e){std::cerr<<"error "<<e.what()<<'\n';return 1;}}

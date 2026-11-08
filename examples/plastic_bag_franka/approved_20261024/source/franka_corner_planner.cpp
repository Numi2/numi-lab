#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/Franka.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using V3 = std::array<double, 3>;
using Q = std::array<double, 4>;
constexpr std::size_t kToolBody = 8u;
constexpr double kPi = 3.1415926535897932384626433832795;
constexpr double kOrientationScale = 0.15;
constexpr double kPlannerDt = 1.0/240.0;
bool initialIK=true;

V3 add(V3 a, V3 b) { return {a[0]+b[0],a[1]+b[1],a[2]+b[2]}; }
V3 sub(V3 a, V3 b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
V3 scale(V3 a, double s) { return {a[0]*s,a[1]*s,a[2]*s}; }
double norm(V3 a) { return std::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]); }
Q mul(Q a, Q b) {
  return {a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],
          a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],
          a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],
          a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]};
}
Q inv(Q q) { return {-q[0],-q[1],-q[2],q[3]}; }
Q normalized(Q q) {
  double n=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
  if (!(n>1.0e-14) || !std::isfinite(n)) throw std::runtime_error("invalid Franka orientation");
  for(double& x:q)x/=n;
  return q;
}
Q axisAngle(V3 axis,double angle) {
  double n=norm(axis);
  if (!(n>0.0)) throw std::runtime_error("invalid rotation axis");
  double s=std::sin(0.5*angle)/n;
  return {axis[0]*s,axis[1]*s,axis[2]*s,std::cos(0.5*angle)};
}
V3 rotationError(Q desired,Q current) {
  Q e=normalized(mul(desired,inv(current)));
  if(e[3]<0.0)for(double& x:e)x=-x;
  V3 v{e[0],e[1],e[2]};
  double s=norm(v);
  if(s<1.0e-12)return scale(v,2.0);
  double angle=2.0*std::atan2(s,std::clamp(e[3],-1.0,1.0));
  return scale(v,angle/s);
}
struct Pose { V3 p{}; Q q{0,0,0,1}; };
Pose toolPose(const metalrobo::EngineModel& model,const std::vector<double>& q) {
  std::vector<double> zero(model.world.nv,0.0);
  std::vector<metalrobo::ArticulatedBodyKinematics> bodies(model.world.bodyCount);
  auto d=metalrobo::computeArticulatedBodyKinematics(model,0u,q,zero,bodies);
  if(!d.succeeded())throw std::runtime_error("Franka FK failed, status="+std::to_string(static_cast<unsigned>(d.status)));
  const auto& com=model.bodies[kToolBody].centerOfMass;
  std::vector<metalrobo::ArticulatedPointQuery> queries{{static_cast<std::uint32_t>(kToolBody),{-com.x,-com.y,-com.z}}};
  std::vector<metalrobo::ArticulatedPointKinematics> points(1u);
  std::vector<double> jac(3u*model.world.nv);
  auto pd=metalrobo::computeArticulatedPointJacobians(model,0u,q,zero,queries,points,jac);
  if(!pd.succeeded())throw std::runtime_error("Franka end-effector point query failed");
  return {points[0].position,bodies[kToolBody].orientation};
}
bool solve6(std::array<std::array<double,6>,6> a,std::array<double,6> b,std::array<double,6>& x) {
  for(std::size_t c=0;c<6;++c) {
    std::size_t pivot=c;
    for(std::size_t r=c+1;r<6;++r)if(std::abs(a[r][c])>std::abs(a[pivot][c]))pivot=r;
    if(std::abs(a[pivot][c])<1e-18)return false;
    std::swap(a[pivot],a[c]);std::swap(b[pivot],b[c]);
    double d=a[c][c];for(std::size_t j=c;j<6;++j)a[c][j]/=d;b[c]/=d;
    for(std::size_t r=0;r<6;++r)if(r!=c){double f=a[r][c];for(std::size_t j=c;j<6;++j)a[r][j]-=f*a[c][j];b[r]-=f*b[c];}
  }
  x=b;return true;
}
std::pair<Pose,std::pair<double,double>> solvePose(const metalrobo::EngineModel& model,
                                                  const std::vector<double>& q,
                                                  const Pose& target,
                                                  std::vector<double>& answer) {
  answer=q;
  const std::vector<double> seed=q;
  const std::size_t nv=model.world.nv;
  for(std::size_t iteration=0;iteration<1200;++iteration) {
    std::vector<double> zero(nv,0.0);
    std::vector<metalrobo::ArticulatedBodyKinematics> bodies(model.world.bodyCount);
    auto d=metalrobo::computeArticulatedBodyKinematics(model,0u,answer,zero,bodies);
    if(!d.succeeded())throw std::runtime_error("Franka FK failed during IK");
    const auto& b=bodies[kToolBody];
    Pose current=toolPose(model,answer);
    V3 ep=sub(target.p,current.p), er=rotationError(target.q,current.q);
    double pe=norm(ep), re=norm(er);
    if(pe<1.0e-7 && re<1.0e-6)return {current,{pe,re}};
    std::array<std::array<double,9>,6> j{};
    for(std::size_t dof=0;dof<7u;++dof) {
      std::vector<double> unit(nv,0.0);
      unit[dof]=1.0;
      std::vector<metalrobo::ArticulatedBodyKinematics> moving(model.world.bodyCount);
      auto jd=metalrobo::computeArticulatedBodyKinematics(model,0u,answer,unit,moving);
      if(!jd.succeeded())throw std::runtime_error("Franka Jacobian query failed");
      const auto& com=model.bodies[kToolBody].centerOfMass;
      std::vector<metalrobo::ArticulatedPointQuery> queries{{static_cast<std::uint32_t>(kToolBody),{-com.x,-com.y,-com.z}}};
      std::vector<metalrobo::ArticulatedPointKinematics> points(1u);
      std::vector<double> pointJac(3u*nv);
      auto pd=metalrobo::computeArticulatedPointJacobians(model,0u,answer,unit,queries,points,pointJac);
      if(!pd.succeeded())throw std::runtime_error("Franka point Jacobian query failed");
      const auto& m=moving[kToolBody];
      for(std::size_t axis=0;axis<3;++axis){j[axis][dof]=points[0].linearVelocity[axis];j[axis+3][dof]=kOrientationScale*m.angularVelocity[axis];}
    }
    std::array<double,6> e{ep[0],ep[1],ep[2],kOrientationScale*er[0],kOrientationScale*er[1],kOrientationScale*er[2]};
    std::array<std::array<double,6>,6> a{};
    for(std::size_t r=0;r<6;++r)for(std::size_t c=0;c<6;++c){
      for(std::size_t dof=0;dof<7u;++dof)a[r][c]+=j[r][dof]*j[c][dof];
      if(r==c)a[r][c]+=2.5e-5;
    }
    std::array<double,6> y{};
    if(!solve6(a,e,y))throw std::runtime_error("Franka damped IK system is singular");
    for(std::size_t dof=0;dof<7u;++dof){
      double dq=0;for(std::size_t r=0;r<6;++r)dq+=j[r][dof]*y[r];
      dq=std::clamp(0.75*dq,-0.08,0.08);
      const auto& limits=model.dofs[dof].limits;
      const double speedLimit=static_cast<double>(limits.z)*0.95;
      const double low=std::max(static_cast<double>(limits.x)+1.0e-4,seed[dof]-(initialIK?10.0:speedLimit*kPlannerDt));
      const double high=std::min(static_cast<double>(limits.y)-1.0e-4,seed[dof]+(initialIK?10.0:speedLimit*kPlannerDt));
      answer[dof]=std::clamp(answer[dof]+dq,low,high);
    }
  }
  Pose actual=toolPose(model,answer);
  double pe=norm(sub(target.p,actual.p)), re=norm(rotationError(target.q,actual.q));
  if(pe>=2.0e-6 || re>=2.0e-5)throw std::runtime_error("Franka IK did not converge: position_m="+std::to_string(pe)+" orientation_rad="+std::to_string(re));
  return {actual,{pe,re}};
}
double smooth(double x){x=std::clamp(x,0.0,1.0);return x*x*(3.0-2.0*x);}
}


int main(int argc,char** argv) try {
 if(argc!=4&&argc!=5)throw std::runtime_error("usage: franka_corner_planner GRIP_CSV ROBOT_JSON REQUEST_TXT");
 auto model=metalrobo::makeFrankaPandaHandEngineModel();std::string reason;
 if(!model.valid(&reason))throw std::runtime_error(reason);
 std::vector<double> q(model.defaultQ.begin(),model.defaultQ.end()),zero(model.world.nv,0);
 const auto rootCOM=model.bodies[0].centerOfMass;
 const V3 base{.45-rootCOM.x,.108+rootCOM.z,.38+rootCOM.y};
 V3 corner{},hover{};std::ifstream request(argv[3]);for(double& x:corner)request>>x;for(double& x:q)request>>x;for(double& x:hover)request>>x;
 if(!request)throw std::runtime_error("Invalid planner request");
 std::vector<double> graspSeed=q;for(double& x:graspSeed)request>>x;
 if(!request)throw std::runtime_error("Missing grasp IK seed");
 initialIK=argc==5;
 const Q baseQ{0,.7071067811865475244,.7071067811865475244,0};
 // Local hand axes in scene coordinates: x=-X, y=(0,.5,-sqrt(1-.5^2)), z=x cross y.
 const double ny=.5,nz=-std::sqrt(1.-ny*ny);
 const V3 axis{0,ny,nz},forward{0,nz,-ny};
 // Matrix columns (-X, axis, forward): quaternion from orthonormal axes.
 const Q sceneHand{0,std::sqrt((1.+ny)/2.),-std::sqrt((1.-ny)/2.),0};
 const Q desiredQ=normalized(mul(inv(baseQ),sceneHand));
 auto sceneToBase=[&](V3 p){V3 d=sub(p,base);return V3{-d[0],d[2],d[1]};};
 auto scenePoint=[&](V3 p){return add(base,{-p[0],p[2],p[1]});};
 auto blend=[](double x){x=std::clamp(x,0.,1.);return x*x*x*(10.+x*(-15.+6.*x));};
 // Plan the clear-space approach in joint coordinates. This avoids driving
 // the hand through a singular Cartesian segment; every output pose is FK.
 const bool allowStart=initialIK;
 auto parkPose=Pose{sceneToBase(sub(hover,scale(forward,.1029))),desiredQ};
 std::vector<double> parkQ=q;initialIK=allowStart;solvePose(model,q,parkPose,parkQ);
 V3 graspTarget=add(corner,{0,.006,0});graspTarget[1]=std::max(graspTarget[1],.129);
 Pose graspPose{sceneToBase(sub(graspTarget,scale(forward,.1029))),desiredQ};
 std::vector<double> graspQ=graspSeed;initialIK=true;solvePose(model,graspSeed,graspPose,graspQ);initialIK=false;q=parkQ;
 std::ofstream csv(argv[1]),js(argv[2]);csv<<std::setprecision(17)<<"time_s,cx,cy,cz,nx,ny,nz,half_gap_m,tx,ty,tz\n";js<<std::setprecision(17)<<"{\"frames\":[";
 double maxError=0,maxSpeedRatio=0,minRobotY=1e9;std::vector<double> previous=q;
 for(unsigned frame=240;frame<=1920;++frame){double t=frame*kPlannerDt;
  const double approach=blend((t-1.0)/1.6),lift=blend((t-3.2)/3.0),closure=blend((t-2.6)/.6),release=blend((t-6.2)/.2);
  V3 target=add(corner,{0,.006,0});target[1]=std::max(target[1],.129);
  V3 center=add(add(scale(hover,1.-approach),scale(target,approach)),{.006*blend((t-2.6)/.25),.32*lift,0});
  const double halfGap=(.008*(1.-closure)+.0000175*closure)*(1.-release)+.020*release;
  Pose desired{sceneToBase(sub(center,scale(forward,.1029))),desiredQ};
  if(t<=2.6){for(unsigned j=0;j<7;++j)q[j]=parkQ[j]+approach*(graspQ[j]-parkQ[j]);}
  else {auto solved=solvePose(model,q,desired,q);maxError=std::max(maxError,solved.second.first);}
  initialIK=false;
  q[7]=q[8]=halfGap;
  if(frame>240)for(unsigned j=0;j<9;++j)maxSpeedRatio=std::max(maxSpeedRatio,std::abs(q[j]-previous[j])/(kPlannerDt*model.dofs[j].limits.z));
  auto rot=[](Q a,V3 v){Q p=mul(mul(a,{v[0],v[1],v[2],0}),inv(a));return V3{p[0],p[1],p[2]};};
  std::vector<metalrobo::ArticulatedBodyKinematics> bodies(model.world.bodyCount);
  auto status=metalrobo::computeArticulatedBodyKinematics(model,0u,q,zero,bodies);if(!status.succeeded())throw std::runtime_error("FK failed");
  // Each inner face is the true finger-local y=0 plane at z=44.5 mm.
  // Derive both faces from the articulated finger bodies, not a detached target.
  auto innerFace=[&](unsigned j){const auto& b=bodies[j];const auto cm=model.bodies[j].centerOfMass;
   return scenePoint(add(sub(b.centerOfMassPosition,rot(b.orientation,{cm.x,cm.y,cm.z})),rot(b.orientation,{0,0,.0445})));};
  const auto left=innerFace(9),right=innerFace(10);
  const V3 actualCenter=scale(add(left,right),.5),difference=sub(left,right);
  const double actualHalfGap=.5*norm(difference);
  if(!(actualHalfGap>0))throw std::runtime_error("Invalid finger FK separation");
  const V3 actualAxis=scale(difference,1./(2.*actualHalfGap));
  csv<<t;for(double v:actualCenter)csv<<','<<v;for(double v:actualAxis)csv<<','<<v;csv<<','<<actualHalfGap;const auto actualTangent=rot(normalized(mul(baseQ,bodies[9].orientation)),{1,0,0});for(double v:actualTangent)csv<<','<<v;csv<<'\n';
  if(frame>240)js<<',';js<<"{\"time_s\":"<<t<<",\"q\":[";for(unsigned j=0;j<9;++j){if(j)js<<',';js<<q[j];}js<<"],\"links\":[";
  for(unsigned j=0;j<bodies.size();++j){if(j)js<<',';const auto& b=bodies[j];const auto cm=model.bodies[j].centerOfMass;
   const Q oq=normalized(mul(baseQ,b.orientation));const V3 p=scenePoint(sub(b.centerOfMassPosition,rot(b.orientation,{cm.x,cm.y,cm.z})));
   minRobotY=std::min(minRobotY,p[1]);js<<'['<<p[0]<<','<<p[1]<<','<<p[2]<<','<<oq[0]<<','<<oq[1]<<','<<oq[2]<<','<<oq[3]<<']';
  }js<<"],\"grip_center\":["<<actualCenter[0]<<','<<actualCenter[1]<<','<<actualCenter[2]<<"],\"half_gap_m\":"<<actualHalfGap<<'}';previous=q;
 }
 if(maxSpeedRatio>.95001)throw std::runtime_error("Planned joint speed exceeds 95% limit");
 js<<"]}";std::cout<<"maximum_ik_position_error_m="<<maxError<<" maximum_joint_speed_ratio="<<maxSpeedRatio<<" minimum_link_origin_y_m="<<minRobotY<<'\n';return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}

#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include "metalrobo/MetalWorld.hpp"
struct Grip {float rear[2]{},front[2]{},lastRear[2]{},lastFoot[2]{},rootx{},rootvx{},lastUp[2]{},lastFrontX[2]{},lastRearX[2]{};unsigned stable[2]{},supported{},clearCount{};float lastClear{};bool rearLatch[2]{},frontLatch[2]{},release{},finished{},transfer{};float relX{},relY{},relVx{},relVy{},rollTarget{};unsigned ready{},openStreak{},pulseTicks{};float lastPitch{},pitchRate{},pitchRef{},pitchCorrection{};bool pitchValid{};float toeAbs{},toeSpeed{},lift{},stancePrep{},placement{},bodyTilt{};bool liftReady{};};
static float shapePerchGap(const ArticulatedBodyKinematics& k, float hx, float hy, float hz) {
 const auto& q=k.orientation;float x=q[0],y=q[1],z=q[2],w=q[3];
 float R[3][3]={{1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)},{2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)},{2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)}};
 float P[3][3]={{R[0][0],R[1][0],R[2][0]},{R[0][1],R[1][1],R[2][1]},{R[0][2],R[1][2],R[2][2]}},Q[3][3]={{1,0,0},{0,1,0},{0,0,1}},d[3]={float(-k.centerOfMassPosition[0]),float(-k.centerOfMassPosition[1]),float(-.055-k.centerOfMassPosition[2])},a[3]={hx,hy,hz},b[3]={.025f,.72f,.055f};
 float best=-1.e30f;auto test=[&](float ux,float uy,float uz){float n=std::sqrt(ux*ux+uy*uy+uz*uz);if(n<1.e-8f)return;float u[3]={ux/n,uy/n,uz/n};float dist=std::abs(d[0]*u[0]+d[1]*u[1]+d[2]*u[2]);for(int j=0;j<3;j++)dist-=a[j]*std::abs(P[j][0]*u[0]+P[j][1]*u[1]+P[j][2]*u[2])+b[j]*std::abs(u[j]);best=std::max(best,dist);};
 for(int i=0;i<3;i++){test(P[i][0],P[i][1],P[i][2]);test(Q[i][0],Q[i][1],Q[i][2]);for(int j=0;j<3;j++)test(P[i][1]*Q[j][2]-P[i][2]*Q[j][1],P[i][2]*Q[j][0]-P[i][0]*Q[j][2],P[i][0]*Q[j][1]-P[i][1]*Q[j][0]);}return best;
}
int main(){try {
 auto robot=hipRollModel();need(robot.mechanics.joints.at(5).axis0.y==-1.f,"pronation axis");robot.mechanics.joints.at(5).axis0.y=1.f;const auto&m=robot.mechanics;
 std::cout<<std::setprecision(17);int lane,step;
 while(std::cin>>lane>>step){std::vector<double> q(m.defaultQ.size()),v(m.defaultV.size());for(auto&x:q)need(bool(std::cin>>x),"q missing");for(auto&x:v)need(bool(std::cin>>x),"v missing");
  for(int j: {0,1,2})for(double dx:{-.02,0.,.02}){auto p=q;for(int digit=0;digit<3;digit++){if(j!=1)p[23+2*digit]+=dx;if(j!=0)p[24+2*digit]+=dx;}for(int k=23;k<=28;k++)need(p[k]>=-.300001 && p[k]<=1.200001,"hypothetical toe configuration exceeds joint limit");std::vector<ArticulatedBodyKinematics> k(m.bodies.size());need(computeArticulatedBodyKinematics(m,0,p,v,k).succeeded(),"FK failed");
   float foot=shapePerchGap(k[11],.01f,.01725f,.0092f),front=1.f,rear=1.f;for(int b=19;b<25;b++)front=std::min(front,shapePerchGap(k[b],.009f,.0025f,.0092f));for(int b=25;b<27;b++)rear=std::min(rear,shapePerchGap(k[b],.009f,.0025f,.0092f));
   std::cout<<lane<<" "<<step<<" "<<j<<" "<<dx<<" "<<foot<<" "<<front<<" "<<rear<<"\n";
  }
 }return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}

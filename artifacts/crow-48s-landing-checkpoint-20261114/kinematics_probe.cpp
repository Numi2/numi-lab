#define CROW_HIP_ROLL_LIBRARY
#include "/Users/n/Developer/numi-crow-rl-20261107/artifacts/crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <fstream>
#include <iomanip>
int main(int argc,char** argv){try{
 need(argc==2,"expected path to validation input");
 auto robot=hipRollModel(); auto& m=robot.mechanics;
 need(m.defaultQ.size()==41 && m.defaultV.size()==40 && m.bodyNames.size()==35,"unexpected model dimensions");
 need(m.joints.at(5).axis0.y==-1.f,"unexpected right pronation"); m.joints.at(5).axis0.y=1.f;
 std::ifstream in(argv[1]);need(bool(in),"cannot read input");
 std::vector<double> q(41),v(40);for(auto& x:q)in>>x;for(auto& x:v)in>>x;
 std::vector<ArticulatedBodyKinematics> states(35);
 auto result=computeArticulatedBodyKinematics(m,0,q,v,states);need(result.succeeded(),"native kinematics failed");
 double maxPos=0,maxAng=0;std::string worstPos,worstAng;std::cout<<std::setprecision(12);
 for(unsigned i=0;i<35;i++){
  std::string name; double p[3],rot[4];in>>name;for(double& x:p)in>>x;for(double& x:rot)in>>x;
  need(bool(in) && name==m.bodyNames[i],"missing or misordered body at "+std::to_string(i));
  double d2=0,dot=0,n1=0,n2=0;for(int k=0;k<3;k++){double d=states[i].centerOfMassPosition[k]-p[k];d2+=d*d;}
  for(int k=0;k<4;k++){dot+=states[i].orientation[k]*rot[k];n1+=states[i].orientation[k]*states[i].orientation[k];n2+=rot[k]*rot[k];}
  double pos=std::sqrt(d2),ang=2*std::acos(std::clamp(std::abs(dot)/std::sqrt(n1*n2),0.0,1.0));
  need(std::isfinite(pos)&&std::isfinite(ang),"nonfinite pose error");
  if(pos>maxPos){maxPos=pos;worstPos=name;}if(ang>maxAng){maxAng=ang;worstAng=name;}
  std::cout<<"body "<<name<<" position_error_m "<<pos<<" angle_error_rad "<<ang<<"\n";
 }
 std::cout<<"max_position_error_m "<<maxPos<<" body "<<worstPos<<"\nmax_angle_error_rad "<<maxAng<<" body "<<worstAng<<"\n";
 need(maxPos<.002 && maxAng<.005,"reconstructed pose disagrees with recorded state");
 std::cout<<"pose_reconstruction PASS; not a native resident or controller checkpoint\n";return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}}

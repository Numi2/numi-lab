// Analytic homogeneous uniaxial equilibrium, independent of the RK time solver.
// The x/y coordinates and top z are constrained; gravity is zero.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main surface_diagnostic_entry
#include "fem_plane_surface_diagnostic.cpp"
#undef main
#pragma clang diagnostic pop
int main(int argc,char**argv){
 try{
  if(argc!=3)throw std::runtime_error("usage: static_affine n nodal|gaussian");
  const int n=std::atoi(argv[1]);const std::string rule=argv[2];
  if(n<1||n>32||(rule!="nodal"&&rule!="gaussian"))throw std::runtime_error("invalid arguments");
  Model m(n,rule=="gaussian");
  const double top=.02005,depth=.02;
  auto stress=[&](double g){
   double z=(top-g)/depth;
   return m.mu*z+(m.lambda*std::log(z)-m.mu)/z;
  };
  auto balance=[&](double g){return stress(g)+m.pressure(g);};
  double lo=1e-7,hi=m.d;
  if(!(balance(lo)>0&&balance(hi)<0))throw std::runtime_error("analytic root not bracketed");
  for(int i=0;i<120;++i){double mid=(lo+hi)/2;if(balance(mid)>0)lo=mid;else hi=mid;}
  const double gap=(lo+hi)/2,stretch=(top-gap)/depth,pressure=m.pressure(gap),expectedForce=pressure*.0004;
  State s=m.initial;
  for(auto& x:s.x)x.z=gap+(x.z-.0002)*stretch;
  Change r=m.rhs(s);auto contact=m.contactForce(s);
  double maxInterior=0,sumTop=0,sumContact=0,momentX=0,momentY=0;
  for(size_t i=0;i<s.x.size();++i){
   const double fz=m.mass[i]*(r.v[i].z+9.81);
   bool topNode=std::abs(m.initial.x[i].z-.0202)<1e-12;
   if(topNode)sumTop+=fz;else maxInterior=std::max(maxInterior,std::abs(fz));
   sumContact+=contact[i];momentX+=contact[i]*s.x[i].x;momentY+=contact[i]*s.x[i].y;
  }
  const double forceError=std::abs(sumTop-expectedForce)/expectedForce;
  const double contactError=std::abs(sumContact-expectedForce)/expectedForce;
  const double localError=maxInterior/(expectedForce/(n*n));
  const double momentError=std::max(std::abs(momentX-.01*expectedForce),std::abs(momentY-.01*expectedForce))/(.01*expectedForce);
  std::cout<<std::setprecision(17)<<"{\"n\":"<<n<<",\"quadrature\":\""<<rule
   <<"\",\"gap_m\":"<<gap<<",\"stretch\":"<<stretch<<",\"pressure_Pa\":"<<pressure
   <<",\"expected_force_N\":"<<expectedForce<<",\"top_reaction_N\":"<<sumTop
   <<",\"force_error\":"<<forceError<<",\"contact_force_error\":"<<contactError
   <<",\"scaled_local_residual\":"<<localError<<",\"moment_error\":"<<momentError<<"}\n";
  if(std::max({forceError,contactError,localError,momentError})>1e-7)
   throw std::runtime_error("static analytical force/residual gate failed");
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}

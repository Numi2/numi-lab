// Independent dense numerical-Jacobian backward-Euler solve of the same CPU
// semidiscrete FEM model. No Numi solver, shader or tangent code is used.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main explicit_reference_entry
#include "fem_plane_fp64_reference.cpp"
#undef main
#pragma clang diagnostic pop
#include <limits>

static double coord(const V& v,int axis){return axis==0?v.x:axis==1?v.y:v.z;}
static double& coord(V& v,int axis){return axis==0?v.x:axis==1?v.y:v.z;}
static std::vector<double> residual(const Model& m,const State& start,
                                     const State& candidate,double h){
 Change rhs=m.rhs(candidate);std::vector<double> r(3*start.x.size());
 for(size_t i=0;i<start.x.size();++i)for(int j=0;j<3;++j)
  r[3*i+j]=coord(candidate.x[i],j)-coord(start.x[i],j)
      -h*coord(start.v[i],j)-h*h*coord(rhs.v[i],j);
 return r;
}
static double norm(const std::vector<double>& r){
 double v=0;for(double q:r)v=std::max(v,std::abs(q));return v;
}
static std::vector<double> solve(std::vector<double> a,std::vector<double>b){
 size_t n=b.size();
 for(size_t k=0;k<n;++k){
  size_t pivot=k;
  for(size_t i=k+1;i<n;++i)if(std::abs(a[i*n+k])>std::abs(a[pivot*n+k]))pivot=i;
  if(std::abs(a[pivot*n+k])<1e-16)throw std::runtime_error("singular jacobian");
  for(size_t j=k;j<n;++j)std::swap(a[k*n+j],a[pivot*n+j]);
  std::swap(b[k],b[pivot]);
  for(size_t i=k+1;i<n;++i){
   double q=a[i*n+k]/a[k*n+k];
   for(size_t j=k+1;j<n;++j)a[i*n+j]-=q*a[k*n+j];
   b[i]-=q*b[k];
  }
 }
 for(size_t i=n;i-->0;){
  for(size_t j=i+1;j<n;++j)b[i]-=a[i*n+j]*b[j];
  b[i]/=a[i*n+i];
 }
 return b;
}
static State advance(const Model& m,const State& old,double h,int& iterations,double& worst){
 State s=old;
 for(size_t i=0;i<s.x.size();++i)s.x[i]+=old.v[i]*h;
 const size_t n=3*s.x.size();constexpr double eps=1e-8;
 for(iterations=0;iterations<30;++iterations){
  auto r=residual(m,old,s,h);double magnitude=norm(r);
  if(magnitude<std::min(1e-13,1e-4*h*h)){worst=std::max(worst,magnitude);break;}
  std::vector<double> jac(n*n);
  for(size_t j=0;j<n;++j){
   State plus=s,minus=s;coord(plus.x[j/3],j%3)+=eps;coord(minus.x[j/3],j%3)-=eps;
   auto rp=residual(m,old,plus,h),rm=residual(m,old,minus,h);
   for(size_t i=0;i<n;++i)jac[i*n+j]=(rp[i]-rm[i])/(2*eps);
  }
  for(double& x:r)x=-x;
  auto correction=solve(std::move(jac),std::move(r));
  bool accepted=false;
  for(double alpha=1;alpha>=1./1024;alpha/=2){
   State trial=s;
   for(size_t j=0;j<n;++j)coord(trial.x[j/3],j%3)+=alpha*correction[j];
   try{
    if(norm(residual(m,old,trial,h))<magnitude){s=std::move(trial);accepted=true;break;}
   }catch(const std::exception&){}
  }
  if(!accepted)throw std::runtime_error("CPU backward Euler line search stalled");
 }
 if(iterations==30)throw std::runtime_error("CPU backward Euler iteration limit");
 for(size_t i=0;i<s.x.size();++i)s.v[i]=(s.x[i]-old.x[i])*(1/h);
 return s;
}
int main(int argc,char**argv){
 try{
  if(argc!=3&&argc!=4)throw std::runtime_error("usage: backward_euler timestep duration [capture_nodes]");
  const bool capture=argc==4&&std::string(argv[3])=="capture_nodes";
  double h=std::atof(argv[1]),duration=std::atof(argv[2]);
  if(h<=0||duration!=.01||std::abs(std::round(duration/h)*h-duration)>1e-12)
   throw std::runtime_error("invalid arguments");
  Model model(2);State s=model.initial;std::array<double,10> bins{};
  double impulse=0,worst=0,minGap=1;long steps=std::lround(duration/h);int maxIterations=0;
  std::cout<<std::setprecision(17)<<"{\"schema\":\"numi.independent-fem-be.v1\",\"h\":"<<h<<",\"duration\":"<<duration<<",\"traces\":[";
  for(long i=0;i<steps;++i){
   int iterations=0;s=advance(model,s,h,iterations,worst);maxIterations=std::max(maxIterations,iterations);
   double force=model.rhs(s).impulse;impulse+=force*h;bins[std::min(9,int((i+.5)*h/.001))]+=force*h/.001;
   double mass=0,pz=0,com=0,kinetic=0,gap=1;
   for(size_t j=0;j<model.mass.size();++j){
    mass+=model.mass[j];pz+=model.mass[j]*s.v[j].z;com+=model.mass[j]*s.x[j].z;
    kinetic+=model.mass[j]*(s.v[j].x*s.v[j].x+s.v[j].y*s.v[j].y+s.v[j].z*s.v[j].z)/2;
    gap=std::min(gap,s.x[j].z);
   }
   minGap=std::min(minGap,gap);
   std::cout<<(i?",":"")<<"{\"step\":"<<i<<",\"gap\":"<<gap<<",\"com_z\":"<<com/mass
    <<",\"momentum_z\":"<<pz<<",\"force\":"<<force<<",\"kinetic_J\":"<<kinetic
    <<",\"potential_J\":"<<model.energy(s)<<",\"iterations\":"<<iterations;
   if(capture){
    std::cout<<",\"node_state\":[";
    for(size_t j=0;j<s.x.size();++j)
     std::cout<<(j?",":"")<<"["<<s.x[j].x<<","<<s.x[j].y<<","<<s.x[j].z<<","<<s.v[j].x<<","<<s.v[j].y<<","<<s.v[j].z<<"]";
    std::cout<<"]";
   }
   std::cout<<"}";
  }
  std::cout<<"],\"mass_kg\":0.0088,\"minimum_gap\":"<<minGap<<",\"impulse_Ns\":"<<impulse
   <<",\"max_nonlinear_residual_m\":"<<worst<<",\"max_newton_iterations\":"<<maxIterations<<",\"force_bins_N\":[";
  for(int i=0;i<10;++i)std::cout<<(i?",":"")<<bins[i];
  std::cout<<"]}\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}

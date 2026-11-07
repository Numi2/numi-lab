// Contact-free longitudinal wave in the independent FP64 FEM model.
// See CONTACT_WAVE_DIAGNOSIS_20261107.md for the prediction and limits.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main surface_diagnostic_main
#include "fem_plane_surface_diagnostic.cpp"
#undef main
#pragma clang diagnostic pop
#include <cstdio>
Change wave_rhs(const Model& m,const State& s) {
  for(size_t i=0;i<m.area.size();++i)
    if(m.area[i] && !(s.x[i].z>m.d))
      throw std::runtime_error("plane activation encountered");
  Change r=m.rhs(s);
  if(r.impulse!=0)throw std::runtime_error("nonzero plane force");
  for(size_t i=0;i<m.mass.size();++i) {
    r.x[i].x=r.x[i].y=0;
    r.v[i].x=r.v[i].y=0;
    r.v[i].z+=9.81; // cancel body force; lateral accelerations are reactions
  }
  return r;
}
int main(int argc,char**argv) {
 try {
  if(argc!=6)throw std::runtime_error("usage: n mode h duration amplitude");
  const int n=std::atoi(argv[1]),mode=std::atoi(argv[2]);
  const double h=std::atof(argv[3]),duration=std::atof(argv[4]),A=std::atof(argv[5]);
  const double interval=.0001,L=.02,pi=std::acos(-1.),k=mode*pi/L;
  if(n<2||n>32||(mode!=2&&mode!=8)||mode>n||h<=0||duration!=.01||A<=0||
     std::abs(std::round(interval/h)*h-interval)>1e-13)
    throw std::runtime_error("invalid parameters");
  Model m(n);State s=m.initial;
  const double cp=std::sqrt((m.lambda+2*m.mu)/1100),omega=k*cp;
  double norm=0;
  for(size_t i=0;i<s.x.size();++i){
   const double shape=std::cos(k*(m.initial.x[i].z-.0002));
   s.x[i].z+=A*shape;norm+=m.mass[i]*shape*shape;
  }
  const long steps=std::lround(duration/h),trace=std::lround(interval/h);
  std::cout<<std::setprecision(17)<<"time,projection,analytical_projection,normalized_l2,min_gap,max_strain,max_stress_nonlinearity\n";
  auto emit=[&](double t) {
   double e=0,p=0,minGap=1,maxStrain=0,maxNonlinearity=0;
   for(size_t i=0;i<s.x.size();++i) {
    const double shape=std::cos(k*(m.initial.x[i].z-.0002));
    const double u=s.x[i].z-m.initial.x[i].z;
    const double d=u-A*shape*std::cos(omega*t);
    e+=m.mass[i]*d*d;p+=m.mass[i]*u*shape;
    if(m.area[i])minGap=std::min(minGap,s.x[i].z);
   }
   for(const Tet& tet:m.tets) {
    M ds=edges(s.x[tet.id[0]],s.x[tet.id[1]],s.x[tet.id[2]],s.x[tet.id[3]]);
    double fzz=0;for(int q=0;q<3;++q)fzz+=ds.a[2][q]*tet.inv.a[q][2];
    double strain=fzz-1;
    if(!(fzz>0))throw std::runtime_error("inverted tetrahedron");
    const double stress=m.mu*fzz+(m.lambda*std::log(fzz)-m.mu)/fzz;
    const double nonlinear=std::abs(stress-(m.lambda+2*m.mu)*strain)
                         /((m.lambda+2*m.mu)*A*k);
    maxStrain=std::max(maxStrain,std::abs(strain));
    maxNonlinearity=std::max(maxNonlinearity,nonlinear);
   }
   std::cout<<t<<","<<p/(A*norm)<<","<<std::cos(omega*t)<<","
            <<std::sqrt(e/norm)/A<<","<<minGap<<","<<maxStrain<<","<<maxNonlinearity<<"\n";
  };
  emit(0);
  for(long i=0;i<steps;++i) {
   Change a=wave_rhs(m,s),b=wave_rhs(m,add(s,a,h/2)),c=wave_rhs(m,add(s,b,h/2)),d=wave_rhs(m,add(s,c,h));
   for(size_t j=0;j<s.x.size();++j) {
    s.x[j]+=(a.x[j]+b.x[j]*2+c.x[j]*2+d.x[j])*(h/6);
    s.v[j]+=(a.v[j]+b.v[j]*2+c.v[j]*2+d.v[j])*(h/6);
   }
   if((i+1)%trace==0)emit((i+1)*h);
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}

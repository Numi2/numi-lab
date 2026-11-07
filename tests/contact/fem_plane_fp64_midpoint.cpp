// CPU-only implicit midpoint diagnostic. Shares the independently authored
// FP64 semidiscrete FEM model and numerical Jacobian solver, not Numi runtime.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define NM_FEM_BE_NO_MAIN
#include "fem_plane_fp64_backward_euler.cpp"
#undef NM_FEM_BE_NO_MAIN
#pragma clang diagnostic pop

static std::vector<double> midResidual(const Model& m,const State& old,const State& end,double h){
 State mid=old;
 for(size_t i=0;i<mid.x.size();++i)mid.x[i]=(old.x[i]+end.x[i])*.5;
 Change a=m.rhs(mid);std::vector<double> r(3*old.x.size());
 for(size_t i=0;i<old.x.size();++i)for(int j=0;j<3;++j)
  r[3*i+j]=coord(end.x[i],j)-coord(old.x[i],j)
   -h*coord(old.v[i],j)-.5*h*h*coord(a.v[i],j);
 return r;
}
static void clearance(const Model& m,const State& s){
 for(size_t i=0;i<s.x.size();++i)
  if(m.area[i]>0 && !(s.x[i].z>1e-7))
   throw std::runtime_error("midpoint candidate violates contact clearance");
}
static State midAdvance(const Model& m,const State& old,double h,int& iterations,double& worst){
 State s=old;
 for(size_t i=0;i<s.x.size();++i)s.x[i]+=old.v[i]*h;
 const size_t n=3*s.x.size();constexpr double eps=1e-8;
 for(iterations=0;iterations<30;++iterations){
  clearance(m,s);auto r=midResidual(m,old,s,h);double magnitude=norm(r);
  if(magnitude<std::min(1e-13,1e-4*h*h)){worst=std::max(worst,magnitude);break;}
  std::vector<double> jac(n*n);
  for(size_t j=0;j<n;++j){
   State plus=s,minus=s;coord(plus.x[j/3],j%3)+=eps;coord(minus.x[j/3],j%3)-=eps;
   auto rp=midResidual(m,old,plus,h),rm=midResidual(m,old,minus,h);
   for(size_t i=0;i<n;++i)jac[i*n+j]=(rp[i]-rm[i])/(2*eps);
  }
  for(double& x:r)x=-x;
  auto correction=solve(std::move(jac),std::move(r));
  bool accepted=false;
  for(double alpha=1;alpha>=1./1024;alpha/=2){
   State trial=s;for(size_t j=0;j<n;++j)coord(trial.x[j/3],j%3)+=alpha*correction[j];
   try{
    clearance(m,trial);
    if(norm(midResidual(m,old,trial,h))<magnitude){s=std::move(trial);accepted=true;break;}
   }catch(const std::exception&){}
  }
  if(!accepted)throw std::runtime_error("CPU midpoint line search stalled");
 }
 if(iterations==30)throw std::runtime_error("CPU midpoint iteration limit");
 for(size_t i=0;i<s.x.size();++i)s.v[i]=(s.x[i]-old.x[i])*(2/h)-old.v[i];
 return s;
}
static double contactEnergy(const Model& m,const State& s){
 double u=0;
 for(size_t i=0;i<m.mass.size();++i){
  double g=s.x[i].z;if(m.area[i]>0&&g<m.d){
   double q=g*g-m.d*m.d;u-=m.area[i]*m.k*q*q*std::log(g*g/(m.d*m.d));
  }
 }
 return u;
}
static double kinetic(const Model& m,const State& s){
 double v=0;for(size_t i=0;i<m.mass.size();++i)
  v+=m.mass[i]*(s.v[i].x*s.v[i].x+s.v[i].y*s.v[i].y+s.v[i].z*s.v[i].z)/2;
 return v;
}
static double midFunctional(const Model& m,const State& old,const State& end,double h){
 State mid=old;double value=0;
 for(size_t i=0;i<mid.x.size();++i){
  mid.x[i]=(old.x[i]+end.x[i])*.5;
  V q=end.x[i]-old.x[i]-old.v[i]*h;
  value+=m.mass[i]*(q.x*q.x+q.y*q.y+q.z*q.z)/2;
 }
 return value+h*h*m.energy(mid);
}
static void discreteCheck(){
 Model m(2);State old=m.initial,end=old;double h=.000125;
 for(size_t i=0;i<end.x.size();++i){
  old.x[i].z-=.000105;
  end.x[i]=old.x[i];end.x[i].z-=2e-6*(1+(i%3)*.2);
  end.x[i].x+=1e-6*std::sin(double(i));old.v[i].z=-.03;
 }
 auto r=midResidual(m,old,end,h);size_t n=r.size();
 std::vector<double> a(n),b(n);
 for(size_t i=0;i<n;++i){a[i]=std::sin(i+1.);b[i]=std::cos(2.*i+1.);}
 auto shifted=[&](const std::vector<double>&d,double scale){
  State t=end;for(size_t j=0;j<n;++j)coord(t.x[j/3],j%3)+=scale*d[j];return t;
 };
 const double eps=1e-8;
 double direct=0;for(size_t i=0;i<n;++i)direct+=m.mass[i/3]*r[i]*a[i];
 double finite=(midFunctional(m,old,shifted(a,eps),h)-midFunctional(m,old,shifted(a,-eps),h))/(2*eps);
 auto ra=midResidual(m,old,shifted(a,eps),h),rb=midResidual(m,old,shifted(a,-eps),h);
 auto rc=midResidual(m,old,shifted(b,eps),h),rd=midResidual(m,old,shifted(b,-eps),h);
 double aHb=0,bHa=0;
 for(size_t i=0;i<n;++i){
  aHb+=a[i]*m.mass[i/3]*(rc[i]-rd[i])/(2*eps);
  bHa+=b[i]*m.mass[i/3]*(ra[i]-rb[i])/(2*eps);
 }
 const double de=2e-8;
 auto mix=[&](double sa,double sb){
  State t=end;for(size_t j=0;j<n;++j)coord(t.x[j/3],j%3)+=de*(sa*a[j]+sb*b[j]);
  return midFunctional(m,old,t,h);
 };
 const double energyMixed=(mix(1,1)-mix(1,-1)-mix(-1,1)+mix(-1,-1))/(4*de*de);
 double tangenterr=std::abs(energyMixed-aHb)/std::max({std::abs(energyMixed),std::abs(aHb),1e-30});
 double graderr=std::abs(direct-finite)/std::max(std::abs(direct),1e-30);
 double symerr=std::abs(aHb-bHa)/std::max({std::abs(aHb),std::abs(bHa),1e-30});
 std::cout<<std::setprecision(17)<<"{\"midpoint_gradient_relative_error\":"<<graderr
  <<",\"mass_weighted_tangent_symmetry_relative_error\":"<<symerr
  <<",\"tangent_energy_mixed_derivative_relative_error\":"<<tangenterr<<"}\n";
 if(graderr>1e-5||symerr>1e-5||tangenterr>1e-5)throw std::runtime_error("midpoint discrete functional check failed");
}
int main(int argc,char**argv){
 try{
  if(argc==2&&std::string(argv[1])=="--discrete-check"){discreteCheck();return 0;}
  if(argc!=3)throw std::runtime_error("usage: midpoint timestep duration");
  double h=std::atof(argv[1]),duration=std::atof(argv[2]);
  if(h<=0||duration!=.01||std::abs(std::round(duration/h)*h-duration)>1e-12)
   throw std::runtime_error("invalid arguments");
  Model m(2);State s=m.initial;
  // Exercise a rejected candidate without modifying the authoritative state.
  State old=s;bool rejected=false;
  try{State invalid=s;invalid.x[0].z=0;clearance(m,invalid);}
  catch(const std::exception&){rejected=true;}
  bool restored=true;
  for(size_t i=0;i<s.x.size();++i)restored&=s.x[i].x==old.x[i].x&&s.x[i].y==old.x[i].y&&s.x[i].z==old.x[i].z
   &&s.v[i].x==old.v[i].x&&s.v[i].y==old.v[i].y&&s.v[i].z==old.v[i].z;
  std::array<double,10> bins{};double impulse=0,worst=0,minGap=1,work=0,maxStepWorkDefect=0;
  double initialEnergy=m.energy(s)+kinetic(m,s),initialContact=contactEnergy(m,s),energyDefect=0;int maxIterations=0;
  long steps=std::lround(duration/h);
  for(long i=0;i<steps;++i){
   old=s;int it=0;s=midAdvance(m,old,h,it,worst);maxIterations=std::max(maxIterations,it);
   State mid=old;
   for(size_t j=0;j<mid.x.size();++j)mid.x[j]=(old.x[j]+s.x[j])*.5;
   double force=m.rhs(mid).impulse;impulse+=h*force;bins[std::min(9,int((i+.5)*h/.001))]+=force*h/.001;
   double stepWork=0;
   for(size_t j=0;j<m.mass.size();++j){
    minGap=std::min(minGap,s.x[j].z);
    if(m.area[j])stepWork+=m.area[j]*m.pressure(mid.x[j].z)*(s.x[j].z-old.x[j].z);
   }
   work+=stepWork;
   maxStepWorkDefect=std::max(maxStepWorkDefect,std::abs(stepWork+contactEnergy(m,s)-contactEnergy(m,old)));
   energyDefect=std::max(energyDefect,std::abs(m.energy(s)+kinetic(m,s)-initialEnergy));
  }
  std::cout<<std::setprecision(17)<<"{\"schema\":\"numi.independent-fem-midpoint.v1\",\"h\":"<<h
   <<",\"duration\":"<<duration<<",\"steps\":"<<steps<<",\"minimum_gap_m\":"<<minGap
   <<",\"impulse_Ns\":"<<impulse<<",\"contact_work_J\":"<<work
   <<",\"initial_energy_J\":"<<initialEnergy<<",\"max_total_energy_defect_J\":"<<energyDefect
   <<",\"initial_contact_potential_J\":"<<initialContact<<",\"final_contact_potential_J\":"<<contactEnergy(m,s)
   <<",\"contact_work_plus_potential_change_J\":"<<work+contactEnergy(m,s)-initialContact
   <<",\"max_step_contact_work_plus_potential_change_J\":"<<maxStepWorkDefect
   <<",\"max_nonlinear_residual_m\":"<<worst<<",\"max_newton_iterations\":"<<maxIterations
   <<",\"forced_rejection_left_state_unchanged\":"<<(rejected&&restored?"true":"false")<<",\"force_bins_N\":[";
  for(int i=0;i<10;++i)std::cout<<(i?",":"")<<bins[i];
  std::cout<<"]}\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}

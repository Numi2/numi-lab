// Independent FP64 two-owner midpoint equations for diagnosing the native
// Matter/MetalWorld implementation. This is not a native transaction.
#define NM_FEM_MID_NO_MAIN
#include "fem_plane_fp64_midpoint.cpp"
#undef NM_FEM_MID_NO_MAIN

struct CoupledState { State fem; double z=0, vz=.02; };
constexpr double kSliderMass=.010;
static Change coupledForce(const Model& m,const CoupledState& s) {
 State relative=s.fem;
 for(auto& x:relative.x)x.z-=s.z;
 return m.rhs(relative);
}
static double coupledPotential(const Model& m,const CoupledState& s) {
 State relative=s.fem;double mass=0;
 for(size_t i=0;i<relative.x.size();++i){relative.x[i].z-=s.z;mass+=m.mass[i];}
 return m.energy(relative)+(mass+kSliderMass)*9.81*s.z;
}
static double coupledContactEnergy(const Model& m,const CoupledState& s) {
 State relative=s.fem;for(auto& x:relative.x)x.z-=s.z;return contactEnergy(m,relative);
}
static CoupledState stage(const CoupledState& a,const CoupledState& b) {
 CoupledState s=a;s.z=(a.z+b.z)*.5;
 for(size_t i=0;i<s.fem.x.size();++i)s.fem.x[i]=(a.fem.x[i]+b.fem.x[i])*.5;
 return s;
}
static void coupledClearance(const Model& m,const CoupledState& s) {
 for(size_t i=0;i<m.area.size();++i)
  if(m.area[i]>0 && !(s.fem.x[i].z-s.z>1e-7))
   throw std::runtime_error("coupled midpoint candidate violates clearance");
}
static std::vector<double> coupledResidual(const Model& m,const CoupledState& old,const CoupledState& end,double h) {
 const auto mid=stage(old,end);const auto a=coupledForce(m,mid);
 std::vector<double> r(3*old.fem.x.size()+1);
 for(size_t i=0;i<old.fem.x.size();++i)for(int k=0;k<3;++k)
  r[3*i+k]=coord(end.fem.x[i],k)-coord(old.fem.x[i],k)-h*coord(old.fem.v[i],k)-.5*h*h*coord(a.v[i],k);
 r.back()=end.z-old.z-h*old.vz-.5*h*h*(-9.81-a.impulse/kSliderMass);
 return r;
}
static CoupledState advanceCoupled(const Model& m,const CoupledState& old,double h,int& iterations,double& worst) {
 CoupledState s=old;s.z+=h*old.vz;
 for(size_t i=0;i<s.fem.x.size();++i)s.fem.x[i]+=old.fem.v[i]*h;
 const size_t n=3*s.fem.x.size()+1;constexpr double eps=1e-8;
 auto shift=[n](CoupledState& state,size_t j,double d){if(j+1==n)state.z+=d;else coord(state.fem.x[j/3],j%3)+=d;};
 for(iterations=0;iterations<30;++iterations){
  coupledClearance(m,s);coupledClearance(m,stage(old,s));
  auto r=coupledResidual(m,old,s,h);double magnitude=norm(r);
  if(magnitude<std::min(1e-13,1e-4*h*h)){worst=std::max(worst,magnitude);break;}
  std::vector<double> jac(n*n);
  for(size_t j=0;j<n;++j){
   CoupledState p=s,q=s;shift(p,j,eps);shift(q,j,-eps);
   auto rp=coupledResidual(m,old,p,h),rq=coupledResidual(m,old,q,h);
   for(size_t i=0;i<n;++i)jac[i*n+j]=(rp[i]-rq[i])/(2*eps);
  }
  for(double& x:r)x=-x;auto delta=solve(std::move(jac),std::move(r));bool accepted=false;
  for(double alpha=1;alpha>=1./1024;alpha/=2){
   CoupledState trial=s;for(size_t j=0;j<n;++j)shift(trial,j,alpha*delta[j]);
   try{coupledClearance(m,trial);coupledClearance(m,stage(old,trial));
    if(norm(coupledResidual(m,old,trial,h))<magnitude){s=std::move(trial);accepted=true;break;}
   }catch(const std::exception&){}
  }
  if(!accepted)throw std::runtime_error("coupled midpoint line search stalled");
 }
 if(iterations==30)throw std::runtime_error("coupled midpoint iteration limit");
 for(size_t i=0;i<s.fem.x.size();++i)s.fem.v[i]=(s.fem.x[i]-old.fem.x[i])*(2/h)-old.fem.v[i];
 s.vz=2*(s.z-old.z)/h-old.vz;return s;
}
static double functional(const Model& m,const CoupledState& old,const CoupledState& end,double h) {
 double value=0;
 for(size_t i=0;i<m.mass.size();++i){V q=end.fem.x[i]-old.fem.x[i]-old.fem.v[i]*h;
  value+=.5*m.mass[i]*(q.x*q.x+q.y*q.y+q.z*q.z);}
 double q=end.z-old.z-h*old.vz;value+=.5*kSliderMass*q*q;
 return value+h*h*coupledPotential(m,stage(old,end));
}
int main(int argc,char** argv){try{
 if(argc!=3 && argc!=4)throw std::runtime_error("usage: slider_midpoint timestep duration [initial_gap]");
 double h=std::atof(argv[1]),duration=std::atof(argv[2]);
 if(!(h>0&&duration>0)||std::abs(std::round(duration/h)*h-duration)>1e-12)throw std::runtime_error("invalid timestep");
 const double initialGap=argc==4?std::atof(argv[3]):.0002;
 if(!(initialGap>1e-7) || !std::isfinite(initialGap))throw std::runtime_error("invalid initial gap");
 Model m(2,initialGap);CoupledState s;s.fem=m.initial;const double initialContactEnergy=coupledContactEnergy(m,s);double impulse=0,workF=0,workR=0,minGap=1,worst=0,maxLocal=0,maxRigid=0,maxGlobal=0,peak=0;
 double gradientError=0,crossError=0,torqueX=0,torqueY=0;bool checkedContact=false;int maxIt=0;std::array<double,10> bins{};
 std::vector<std::array<double,2>> trace{{0.,0.}};
 for(long step=0;step<std::lround(duration/h);++step){
  auto old=s;int it;s=advanceCoupled(m,old,h,it,worst);maxIt=std::max(maxIt,it);
  auto mid=stage(old,s);auto force=coupledForce(m,mid);double forceSum=force.impulse;
  double totalMass=kSliderMass,dp=kSliderMass*(s.vz-old.vz);
  for(size_t i=0;i<m.mass.size();++i){
   const V actual=(s.fem.v[i]-old.fem.v[i])*m.mass[i];const V expected=force.v[i]*(h*m.mass[i]);
   maxLocal=std::max(maxLocal,std::hypot(actual.x-expected.x,actual.y-expected.y,actual.z-expected.z));
   totalMass+=m.mass[i];dp+=actual.z;minGap=std::min(minGap,s.fem.x[i].z-s.z);
   if(m.area[i]>0){const double P=m.area[i]*m.pressure(mid.fem.x[i].z-mid.z);
    workF+=P*(s.fem.x[i].z-old.fem.x[i].z);workR-=P*(s.z-old.z);
    torqueX+=h*mid.fem.x[i].y*P;torqueY-=h*mid.fem.x[i].x*P;}
  }
  maxRigid=std::max(maxRigid,std::abs(kSliderMass*(s.vz-old.vz)-h*(-kSliderMass*9.81-forceSum)));
  maxGlobal=std::max(maxGlobal,std::abs(dp+h*totalMass*9.81));
  impulse+=h*forceSum;peak=std::max(peak,forceSum);trace.push_back({(step+1)*h,impulse});
  bins[std::min(9,int((step+.5)*h/.001))]+=h*forceSum/.001;
  if(step==0 || (forceSum>0.01 && !checkedContact)){
   const auto r=coupledResidual(m,old,s,h);const size_t n=r.size();const double e=1e-8;
   std::vector<double> dir(n);for(size_t i=0;i<n;++i)dir[i]=std::sin(i+1.);
   CoupledState p=s,q=s;
   for(size_t j=0;j<n;++j){if(j+1==n){p.z+=e*dir[j];q.z-=e*dir[j];}
    else{coord(p.fem.x[j/3],j%3)+=e*dir[j];coord(q.fem.x[j/3],j%3)-=e*dir[j];}}
   double measured=(functional(m,old,p,h)-functional(m,old,q,h))/(2*e),direct=kSliderMass*r.back()*dir.back();
   for(size_t j=0;j+1<n;++j)direct+=m.mass[j/3]*r[j]*dir[j];
   gradientError=std::max(gradientError,std::abs(measured-direct));
   // Contact cross blocks must agree after physical mass weighting.
   if(forceSum>0.01){checkedContact=true;size_t node=0;while(node<m.area.size() && m.area[node]==0)++node;
    CoupledState a=s,b=s,c=s,d=s;a.z+=e;b.z-=e;c.fem.x[node].z+=e;d.fem.x[node].z-=e;
    auto ra=coupledResidual(m,old,a,h),rb=coupledResidual(m,old,b,h),rc=coupledResidual(m,old,c,h),rd=coupledResidual(m,old,d,h);
    double x=m.mass[node]*(ra[3*node+2]-rb[3*node+2])/(2*e),y=kSliderMass*(rc.back()-rd.back())/(2*e);
    crossError=std::abs(x-y)/std::max({std::abs(x),std::abs(y),1e-30});
   }
  }
 }
 std::cout<<std::setprecision(17)<<"{\"schema\":\"numi.independent-fem-slider-midpoint.v1\",\"h\":"<<h<<",\"duration\":"<<duration<<",\"initial_gap_m\":"<<initialGap
 <<",\"min_gap_m\":"<<minGap<<",\"impulse_Ns\":"<<impulse<<",\"sampled_peak_force_N\":"<<peak
 <<",\"initial_contact_potential_J\":"<<initialContactEnergy<<",\"final_contact_potential_J\":"<<coupledContactEnergy(m,s)
 <<",\"contact_work_plus_potential_change_J\":"<<workF+workR+coupledContactEnergy(m,s)-initialContactEnergy
 <<",\"fem_work_J\":"<<workF<<",\"slider_work_J\":"<<workR<<",\"pair_contact_work_J\":"<<workF+workR
 <<",\"fem_torque_impulse_x_Nms\":"<<torqueX<<",\"fem_torque_impulse_y_Nms\":"<<torqueY
 <<",\"slider_z_m\":"<<s.z<<",\"slider_v_mps\":"<<s.vz<<",\"max_fem_momentum_residual_Ns\":"<<maxLocal
 <<",\"max_slider_momentum_residual_Ns\":"<<maxRigid<<",\"max_total_momentum_residual_Ns\":"<<maxGlobal
 <<",\"max_displacement_residual_m\":"<<worst<<",\"gradient_absolute_error\":"<<gradientError<<",\"contact_cross_symmetry_relative_error\":"<<crossError
 <<",\"contact_cross_checked\":"<<(checkedContact?"true":"false")<<",\"max_newton_iterations\":"<<maxIt<<",\"force_bins_N\":[";
 for(size_t i=0;i<bins.size();++i)std::cout<<(i?",":"")<<bins[i];std::cout<<"],\"cumulative_impulse_trace\":[";
 for(size_t i=0;i<trace.size();++i)std::cout<<(i?",":"")<<"["<<trace[i][0]<<","<<trace[i][1]<<"]";std::cout<<"]}\n";
 if(maxLocal>1e-10 || maxRigid>1e-10 || maxGlobal>1e-10 || gradientError>1e-13 || !checkedContact || crossError>1e-4 || !(minGap>1e-7))
  throw std::runtime_error("independent coupled midpoint check failed");
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}

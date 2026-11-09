// Independent FP64 RK4 reference for a dynamically reacting vertical slider.
// This does not run or validate a Matter/MetalWorld native transaction.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main fixed_plane_reference_entry
#include "fem_plane_fp64_reference.cpp"
#undef main
#pragma clang diagnostic pop

struct SliderState { State fem; double z=0, vz=.02, impulse=0, tx=0, ty=0, work=0, femWork=0, sliderWork=0; };
struct SliderChange { Change fem; double z=0, vz=0, impulse=0, tx=0, ty=0, work=0, femWork=0, sliderWork=0; };
constexpr double sliderMass=.010;
SliderChange sliderRhs(const Model& m,const SliderState& s) {
 State relative=s.fem;
 for(auto& x:relative.x)x.z-=s.z;
 SliderChange k; k.fem=m.rhs(relative); k.z=s.vz;
 k.vz=-9.81-k.fem.impulse/sliderMass; k.impulse=k.fem.impulse;
 for(size_t i=0;i<m.area.size();++i)if(m.area[i]){
  const double force=m.area[i]*m.pressure(relative.x[i].z);
  k.tx+=s.fem.x[i].y*force; k.ty-=s.fem.x[i].x*force;
  k.work+=force*(s.fem.v[i].z-s.vz);
  k.femWork+=force*s.fem.v[i].z; k.sliderWork-=force*s.vz;
 }
 return k;
}
SliderState advanceState(const SliderState& s,const SliderChange& k,double h) {
 SliderState r=s; r.fem=add(s.fem,k.fem,h);r.z+=h*k.z;r.vz+=h*k.vz;
 r.impulse+=h*k.impulse;r.tx+=h*k.tx;r.ty+=h*k.ty;r.work+=h*k.work;r.femWork+=h*k.femWork;r.sliderWork+=h*k.sliderWork;return r;
}
double energy(const Model& m,const SliderState& s) {
 State relative=s.fem;double fm=0,kinetic=0;
 for(size_t i=0;i<relative.x.size();++i){
  relative.x[i].z-=s.z;fm+=m.mass[i];
  const V v=s.fem.v[i];kinetic+=.5*m.mass[i]*(v.x*v.x+v.y*v.y+v.z*v.z);
 }
 return m.energy(relative)+fm*9.81*s.z+kinetic+.5*sliderMass*s.vz*s.vz+sliderMass*9.81*s.z;
}
int main(int argc,char**argv) {try{
 if(argc!=2)throw std::runtime_error("usage: slider_reference timestep");
 const double h=std::atof(argv[1]),duration=.01;
 if(!(h>0)||std::abs(std::round(duration/h)*h-duration)>1e-12||
    std::abs(std::round(.000002/h)*h-.000002)>1e-12)throw std::runtime_error("invalid timestep");
 Model m(2);SliderState s;s.fem=m.initial;const double e0=energy(m,s);
 double totalMass=sliderMass,area=0;for(double x:m.mass)totalMass+=x;for(double x:m.area)area+=x;
 double peak=0,minGap=1,first=-1;std::array<double,10> bins{};
 std::vector<std::array<double,5>> trace{{0,0,0,0,.02}};
 std::vector<std::array<double,4>> workTrace{{0,0,0,0}};
 const long steps=std::lround(duration/h),stride=std::lround(.000002/h);
 for(long n=0;n<steps;++n){
  auto a=sliderRhs(m,s),b=sliderRhs(m,advanceState(s,a,h/2)),
       c=sliderRhs(m,advanceState(s,b,h/2)),d=sliderRhs(m,advanceState(s,c,h));
  const auto avg=[h](double w,double x,double y,double z){return h*(w+2*x+2*y+z)/6;};
  for(size_t i=0;i<s.fem.x.size();++i){
   s.fem.x[i]+=(a.fem.x[i]+b.fem.x[i]*2+c.fem.x[i]*2+d.fem.x[i])*(h/6);
   s.fem.v[i]+=(a.fem.v[i]+b.fem.v[i]*2+c.fem.v[i]*2+d.fem.v[i])*(h/6);
  }
  s.z+=avg(a.z,b.z,c.z,d.z);s.vz+=avg(a.vz,b.vz,c.vz,d.vz);
  double di=avg(a.impulse,b.impulse,c.impulse,d.impulse);s.impulse+=di;
  s.tx+=avg(a.tx,b.tx,c.tx,d.tx);s.ty+=avg(a.ty,b.ty,c.ty,d.ty);
  s.work+=avg(a.work,b.work,c.work,d.work);
  s.femWork+=avg(a.femWork,b.femWork,c.femWork,d.femWork);
  s.sliderWork+=avg(a.sliderWork,b.sliderWork,c.sliderWork,d.sliderWork);bins[std::min(9,int((n+.5)*h/.001))]+=di/.001;
  for(const auto& x:s.fem.x)minGap=std::min(minGap,x.z-s.z);
  const double f=sliderRhs(m,s).impulse;peak=std::max(peak,f);
  if(first<0 && f>0)first=(n+1)*h;
  if((n+1)%stride==0) {trace.push_back({(n+1)*h,f,s.impulse,s.z,s.vz});workTrace.push_back({(n+1)*h,s.femWork,s.sliderWork,s.work});}
 }
 double p=sliderMass*s.vz;for(size_t i=0;i<m.mass.size();++i)p+=m.mass[i]*s.fem.v[i].z;
 const double momentumError=p-(sliderMass*.02-totalMass*9.81*duration);
 std::cout<<std::setprecision(17)<<"{\"schema\":\"numi.independent-fem-slider-rk4.v1\",\"h\":"<<h
  <<",\"mesh_divisions\":2,\"slider_mass_kg\":"<<sliderMass<<",\"area_m2\":"<<area
  <<",\"min_gap_m\":"<<minGap<<",\"onset_s\":"<<first<<",\"peak_force_N\":"<<peak
  <<",\"impulse_Ns\":"<<s.impulse<<",\"torque_impulse_x_Nms\":"<<s.tx
  <<",\"torque_impulse_y_Nms\":"<<s.ty<<",\"relative_contact_work_J\":"<<s.work
  <<",\"fem_contact_work_J\":"<<s.femWork<<",\"slider_contact_work_J\":"<<s.sliderWork
  <<",\"slider_final_z_m\":"<<s.z<<",\"slider_final_v_mps\":"<<s.vz
  <<",\"momentum_balance_error_Ns\":"<<momentumError
  <<",\"initial_energy_J\":"<<e0<<",\"final_energy_J\":"<<energy(m,s)<<",\"force_bins_N\":[";
 for(size_t i=0;i<bins.size();++i)std::cout<<(i?",":"")<<bins[i];
 std::cout<<"],\"trace\":[";for(size_t i=0;i<trace.size();++i){std::cout<<(i?",[":"[");
  for(int j=0;j<5;++j)std::cout<<(j?",":"")<<trace[i][j];std::cout<<"]";}
 std::cout<<"],\"work_trace\":[";for(size_t i=0;i<workTrace.size();++i){std::cout<<(i?",[":"[");
  for(int j=0;j<4;++j)std::cout<<(j?",":"")<<workTrace[i][j];std::cout<<"]";}
 std::cout<<"]}\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}

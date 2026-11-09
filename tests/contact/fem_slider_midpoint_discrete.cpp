// Independent FP64 check for the normal-contact portion of a coupled slider
// midpoint equation. This does not run or qualify a native Matter step.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main fixed_plane_reference_entry
#include "fem_plane_fp64_reference.cpp"
#undef main
#pragma clang diagnostic pop
#include <algorithm>
#include <iomanip>

int main() { try {
 Model m(2);constexpr double tau=0.000015625/2,eps=0.0001;
 const size_t n=m.mass.size();std::vector<double> x(n),u(n),d(n);
 double ws=.023, ds=.31;
 for(size_t i=0;i<n;++i){x[i]=m.initial.x[i].z-.000105;u[i]=.01*std::sin(double(i));d[i]=std::cos(double(i)*.73);}
 const auto contact=[&](double a){
  std::vector<double> r(n+1,0.);double energy=0.;
  for(size_t i=0;i<n;++i)if(m.area[i]) {
   const double g=x[i]+tau*(u[i]+a*d[i])-tau*(ws+a*ds);
   if(!(g>0 && g<m.d))throw std::runtime_error("contact fixture out of band");
   const double p=m.area[i]*m.pressure(g);r[i]=-tau*p;r[n]+=tau*p;
   const double q=g*g-m.d*m.d;energy-=m.area[i]*m.k*q*q*std::log(g*g/(m.d*m.d));
  }
  return std::pair{r,energy};
 };
 const auto [rp,ep]=contact(eps);const auto [rm,em]=contact(-eps);const auto [r,e]=contact(0);
 double maxDiff=0,maxScale=0,workGradient=ds*r[n],sum=r[n];
 for(size_t i=0;i<n;++i){
  workGradient+=d[i]*r[i];sum+=r[i];if(!m.area[i])continue;
  const double g=x[i]+tau*(u[i]-ws),q=g*g-m.d*m.d,L=std::log(g*g/(m.d*m.d));
  const double derivative=2*m.k*(2*q*L-q*q/(g*g)+4*g*g*L+8*q);
  const double expected=-tau*tau*m.area[i]*derivative*(d[i]-ds);
  const double actual=(rp[i]-rm[i])/(2*eps);
  maxDiff=std::max(maxDiff,std::abs(expected-actual));maxScale=std::max(maxScale,std::abs(expected));
 }
 double expectedSlider=0;for(size_t i=0;i<n;++i)if(m.area[i]){
  const double g=x[i]+tau*(u[i]-ws),q=g*g-m.d*m.d,L=std::log(g*g/(m.d*m.d));
  const double derivative=2*m.k*(2*q*L-q*q/(g*g)+4*g*g*L+8*q);
  expectedSlider+=tau*tau*m.area[i]*derivative*(d[i]-ds);
 }
 const double actualSlider=(rp[n]-rm[n])/(2*eps);
 maxDiff=std::max(maxDiff,std::abs(expectedSlider-actualSlider));maxScale=std::max(maxScale,std::abs(expectedSlider));
 const double relTangent=maxDiff/std::max(maxScale,1e-30);
 const double fdGradient=(ep-em)/(2*eps);
 const double relEnergy=std::abs(fdGradient-workGradient)/std::max({std::abs(fdGradient),std::abs(workGradient),1e-30});
 const double relBalance=std::abs(sum)/std::max(std::abs(r[n]),1e-30);
 std::cout<<std::setprecision(17)<<"{\"contact_tangent_relative_error\":"<<relTangent
          <<",\"energy_gradient_relative_error\":"<<relEnergy<<",\"contact_balance_relative_error\":"<<relBalance<<"}\n";
 if(relTangent>1e-6||relEnergy>1e-6||relBalance>1e-12)throw std::runtime_error("coupled contact equation check failed");
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;} }

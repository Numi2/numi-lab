// Offline evaluation of a captured backward-Euler state; never publishes it.
#define main numi_plane_contact_diagnostic_main
#include "plane_contact_probe.mm"
#undef main
#include <fstream>
template<class T> static void readExact(const char* path,std::vector<T>& values){
 std::ifstream input(path,std::ios::binary);require(bool(input),"capture unavailable");
 input.read(reinterpret_cast<char*>(values.data()),values.size()*sizeof(T));
 require(bool(input)&&input.peek()==std::ifstream::traits_type::eof(),"capture length mismatch");
}
int main(int argc,char**argv){try{
 require(argc==5,"usage: ACCEPTED WORKING RHS CONTACTS");
 constexpr double h=.000001;auto world=model(2,h,.0002,true,false,true,0.,1.);
 const size_t n=world.dispatch.femNodeCount;
 numi::matter::RuntimeStateSnapshot old,current;old.femNodes.resize(n);current.femNodes.resize(n);
 current.contactSamples.resize(world.dispatch.contactPairCount);
 std::vector<nm_float4> rhs(2*n);readExact(argv[1],old.femNodes);readExact(argv[2],current.femNodes);
 readExact(argv[3],rhs);readExact(argv[4],current.contactSamples);
 unsigned contacts=0;for(const auto& s:current.contactSamples)contacts+=(s.identity.w&NM_CONTACT_VALID)!=0;
 double gap=1;for(const auto& x:current.femNodes)gap=std::min(gap,double(x.restAndFixed.z)+x.referenceDisplacementAndMode.z+x.referenceDisplacementLow.z);
 require(contacts==0 && gap>world.dispatch.numericalLimits.x,"capture is not precontact");
 std::vector<std::array<double,3>> components;
 const auto independent=fixtureResidual(world,old,current,h,false,nullptr,false,&components);
 double gpu2=0,minus2=0,plus2=0;
 for(size_t i=0;i<n;++i){const double v[3]={rhs[i].x,rhs[i].y,rhs[i].z};for(int j=0;j<3;++j){
  const double e=components[i][j];gpu2+=v[j]*v[j];minus2+=(v[j]-e)*(v[j]-e);plus2+=(v[j]+e)*(v[j]+e);
 }}
 const auto& policy=world.mixedSolver.contactAcceptance;
 std::cout<<std::setprecision(17)<<"{\"label\":\"offline_fp64_backward_euler_residual\",\"nodes\":"<<n
  <<",\"gap_m\":"<<gap<<",\"active_contacts\":"<<contacts
  <<",\"fp64_residual_l2_Ns\":"<<independent.l2<<",\"fp64_max_local_scaled_without_floor\":"<<independent.localWithoutFloor
  <<",\"fp64_kinematic_defect_m\":"<<independent.kinematic<<",\"gpu_rhs_first_n_l2\":"<<std::sqrt(gpu2)
  <<",\"gpu_rhs_minus_fp64_l2\":"<<std::sqrt(minus2)<<",\"gpu_rhs_plus_fp64_l2\":"<<std::sqrt(plus2)
  <<",\"physical_absolute_tolerance_Ns\":"<<policy.y<<",\"physical_relative_tolerance\":"<<policy.z
  <<",\"physical_impulse_scale_Ns\":"<<policy.w<<"}\n";
 }catch(const std::exception& e){std::cerr<<"be_fp64_capture_analysis: "<<e.what()<<"\n";return 1;}}

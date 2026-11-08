// CPU reference tests for the narrow midpoint transaction contract. This
// deliberately does not claim to exercise Matter's Metal transaction path.
#define NM_FEM_MID_NO_MAIN
#include "fem_plane_fp64_midpoint.cpp"
#undef NM_FEM_MID_NO_MAIN

struct Transaction { State accepted;std::vector<double> impulses;double work=0; };
static bool equal(const Transaction& a,const Transaction& b){
 if(a.work!=b.work||a.impulses!=b.impulses||a.accepted.x.size()!=b.accepted.x.size())return false;
 for(size_t i=0;i<a.accepted.x.size();++i)for(int j=0;j<3;++j)
  if(coord(a.accepted.x[i],j)!=coord(b.accepted.x[i],j)||
     coord(a.accepted.v[i],j)!=coord(b.accepted.v[i],j))return false;
 return true;
}
static bool reflectAndCommit(const Model& model,Transaction& owner,const State& midpoint,
 double h,double floor) {
 Transaction trial=owner;State end=midpoint;
 for(size_t i=0;i<end.x.size();++i){
  end.x[i]=midpoint.x[i]*2-owner.accepted.x[i];
  end.v[i]=midpoint.v[i]*2-owner.accepted.v[i];
  // A fixed infinite plane and linear node trajectories admit an exact
  // clearance certificate from the two segment endpoints.
  if(!std::isfinite(end.x[i].z)||!(end.x[i].z>floor)||!(owner.accepted.x[i].z>floor))
   return false;
 }
 // The same endpoint must also satisfy material geometry; no trial data has
 // been published if this check fails.
 try{(void)model.energy(end);}catch(const std::exception&){return false;}
 trial.impulses.assign(end.x.size(),0);
 for(size_t i=0;i<end.x.size();++i)if(model.area[i]){
  double force=model.area[i]*model.pressure(midpoint.x[i].z);
  const double halfImpulse=.5*h*force,fullImpulse=2*halfImpulse;
  trial.impulses[i]=fullImpulse;
  trial.work+=(fullImpulse/h)*(end.x[i].z-owner.accepted.x[i].z);
 }
 trial.accepted=end;owner=std::move(trial);return true;
}
int main(){
 try{
  Model m(2);Transaction t{m.initial,std::vector<double>(m.mass.size(),0),0};
  for(auto& p:t.accepted.x)p.z-=.000105;
  for(auto& v:t.accepted.v)v.z=-.02;
  const State original=t.accepted;const double h=.000125;double worst=0;int iter=0;
  const State mid=advance(m,original,h/2,iter,worst);
  auto rh=residual(m,original,mid,h/2);
  State expected=mid;for(size_t i=0;i<expected.x.size();++i){
   expected.x[i]=mid.x[i]*2-original.x[i];expected.v[i]=mid.v[i]*2-original.v[i];}
  auto rf=midResidual(m,original,expected,h);double residualDifference=0;
  for(size_t i=0;i<rf.size();++i)residualDifference=std::max(residualDifference,std::abs(rf[i]-2*rh[i]));
  if(residualDifference>2e-17)throw std::runtime_error("physical-step residual conversion differs");
  if(!reflectAndCommit(m,t,mid,h,1e-7))throw std::runtime_error("valid midpoint rejected");
  double impulse=0,expectedImpulse=0,expectedWork=0;
  for(size_t i=0;i<t.impulses.size();++i){
   impulse+=t.impulses[i];
   if(m.area[i]){double f=m.area[i]*m.pressure(mid.x[i].z);expectedImpulse+=h*f;
    expectedWork+=f*(t.accepted.x[i].z-original.x[i].z);}
  }
  if(std::abs(impulse-expectedImpulse)>1e-18||std::abs(t.work-expectedWork)>1e-18)
   throw std::runtime_error("physical impulse/work conversion differs");
  Transaction before=t;State unsafe=before.accepted;
  unsafe.x[0].z=(before.accepted.x[0].z+0.5e-7)*.5;
  // Midpoint itself remains above clearance, but reflection does not.
  if(!(unsafe.x[0].z>1e-7))throw std::runtime_error("invalid regression setup");
  if(reflectAndCommit(m,t,unsafe,h,1e-7)||!equal(before,t))
   throw std::runtime_error("unclear endpoint did not roll back");
  State inverted=before.accepted;
  for(size_t i=0;i<inverted.x.size();++i)inverted.x[i].x*=.45;
  if(reflectAndCommit(m,t,inverted,h,1e-7)||!equal(before,t))
   throw std::runtime_error("invalid endpoint material state did not roll back");
  std::cout<<"{\"halfstep_residual_to_fullstep_error_m\":"<<residualDifference
   <<",\"normal_impulse_Ns\":"<<impulse<<",\"contact_work_J\":"<<t.work
   <<",\"clear_midpoint_unsafe_endpoint_rejected\":true,\"inverted_endpoint_rejected\":true,"
     "\"authoritative_state_and_impulse_history_restored\":true}\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}

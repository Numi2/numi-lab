// Comparison probe for external independent assembly; no runtime dependency.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main surface_diagnostic_main
#include "fem_plane_surface_diagnostic.cpp"
#undef main
#pragma clang diagnostic pop
int main(){
 Model m(2);State s=m.initial,plus=s,minus=s;std::vector<V> direction(s.x.size());
 for(size_t i=0;i<s.x.size();++i){
  if(!(std::cin>>s.x[i].x>>s.x[i].y>>s.x[i].z>>direction[i].x>>direction[i].y>>direction[i].z))return 2;
 }
 constexpr double h=1e-8;
 for(size_t i=0;i<s.x.size();++i){plus.x[i]=s.x[i]+direction[i]*h;minus.x[i]=s.x[i]-direction[i]*h;}
 auto f=m.rhs(s),a=m.rhs(plus),b=m.rhs(minus);
 std::cout<<std::setprecision(17)<<m.energy(s)<<"\n";
 for(size_t i=0;i<s.x.size();++i){
  V v=f.v[i]*(-m.mass[i]),tv=(a.v[i]-b.v[i])*(-m.mass[i]/(2*h));
  std::cout<<v.x<<" "<<v.y<<" "<<v.z<<" "<<tv.x<<" "<<tv.y<<" "<<tv.z<<"\n";
 }
}

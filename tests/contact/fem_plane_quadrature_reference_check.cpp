// Independent CPU reference checks; this does not qualify Metal or a rollout.
#define main numi_old_reference_main
#include "fem_plane_fp64_reference.cpp"
#undef main
#include <iostream>

int main() {
 for(int n:{2,4,8}) {
  Model mesh(n,.0002,true,true);
  double mass=0,area=0;for(double x:mesh.mass)mass+=x;for(double x:mesh.area)area+=x;
  if(std::abs(mass-.0088)>1e-13 || std::abs(area-.0004)>1e-14) return 1;
  if(mesh.triangles.size()!=static_cast<size_t>(2*n*n)) return 2;
  const auto id=[n](int z){return z*(n+1)*(n+1);};
  const double low=mesh.initial.x[id(1)].z-mesh.initial.x[id(0)].z;
  const double high=mesh.initial.x[id(n)].z-mesh.initial.x[id(n-1)].z;
  if(!(low>0 && low<high)) return 3;
  for(const auto& t:mesh.tets)if(!(t.volume>0) || !std::isfinite(t.volume))return 4;
 }
 Model m(2,.0002,true,true),zero(2,.0002,true,true);zero.k=0;
 State s=m.initial;
 for(size_t i=0;i<s.x.size();++i){s.x[i].z-=.000105;if(m.area[i])s.x[i].z+=1e-6*(double(i%3)-1);}
 auto reaction=[&](const State& x){auto a=m.rhs(x),b=zero.rhs(x);std::vector<double> f(m.mass.size());
  for(size_t i=0;i<f.size();++i)f[i]=m.mass[i]*(a.v[i].z-b.v[i].z);return f;};
 auto f=reaction(s);double sum=0;for(double x:f)sum+=x;
 if(std::abs(sum-m.rhs(s).impulse)>1e-14)return 5;
 std::vector<size_t> ids;for(size_t i=0;i<m.area.size();++i)if(m.area[i]>0)ids.push_back(i);
 double maxCross=0,maxAsymmetry=0,maxDerivative=0;std::vector<std::vector<double>> J(ids.size(),std::vector<double>(ids.size()));
 constexpr double eps=1e-10;
 for(size_t j=0;j<ids.size();++j){State p=s,n=s;p.x[ids[j]].z+=eps;n.x[ids[j]].z-=eps;
  auto fp=reaction(p),fn=reaction(n);
  for(size_t i=0;i<ids.size();++i){J[i][j]=(fp[ids[i]]-fn[ids[i]])/(2*eps);maxDerivative=std::max(maxDerivative,std::abs(J[i][j]));
   if(i!=j)maxCross=std::max(maxCross,std::abs(J[i][j]));}}
 for(size_t i=0;i<ids.size();++i)for(size_t j=0;j<ids.size();++j)maxAsymmetry=std::max(maxAsymmetry,std::abs(J[i][j]-J[j][i]));
 if(!(maxCross>1) || maxAsymmetry>maxDerivative*1e-6)return 6;
 std::cout<<"graded mass/area/volume and quadrature force conservation pass; cross derivative "<<maxCross<<" symmetry error "<<maxAsymmetry<<"\n";
}

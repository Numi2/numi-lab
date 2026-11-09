// Independent FP64 explicit RK4 continuum-dynamics diagnostic for the
// structured neo-Hookean block and fixed reference-area plane pressure law.
// Does not use Numi runtime, shaders, assembled residuals or solver.
#include <algorithm>
#include <atomic>
#include <dispatch/dispatch.h>
#include <cstdint>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef NUMI_FEM_REFERENCE_MAX_DIVISIONS
#define NUMI_FEM_REFERENCE_MAX_DIVISIONS 32
#endif
struct V { double x=0,y=0,z=0; V operator+(V b)const{return {x+b.x,y+b.y,z+b.z};}
 V operator-(V b)const{return {x-b.x,y-b.y,z-b.z};} V operator*(double a)const{return {x*a,y*a,z*a};}
 V& operator+=(V b){x+=b.x;y+=b.y;z+=b.z;return *this;} };
struct M { double a[3][3]{}; };
M inverse(const M& b) {
 const auto& m=b.a;
 const double d=m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
   -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
   +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
 if (std::abs(d)<1e-30) throw std::runtime_error("singular matrix");
 M r;
 for(int i=0;i<3;++i)for(int j=0;j<3;++j)
   r.a[j][i]=(m[(i+1)%3][(j+1)%3]*m[(i+2)%3][(j+2)%3]
             -m[(i+1)%3][(j+2)%3]*m[(i+2)%3][(j+1)%3])/d;
 return r;
}
double determinant(const M& m) {
 return m.a[0][0]*(m.a[1][1]*m.a[2][2]-m.a[1][2]*m.a[2][1])
 -m.a[0][1]*(m.a[1][0]*m.a[2][2]-m.a[1][2]*m.a[2][0])
 +m.a[0][2]*(m.a[1][0]*m.a[2][1]-m.a[1][1]*m.a[2][0]);
}
M edges(V p0,V p1,V p2,V p3) {
 const V c[3]={p1-p0,p2-p0,p3-p0};M m;
 for(int j=0;j<3;++j){m.a[0][j]=c[j].x;m.a[1][j]=c[j].y;m.a[2][j]=c[j].z;}
 return m;
}
struct Tet { std::array<int,4> id; M inv; double volume; };
struct State { std::vector<V> x,v; double impulse=0; };
struct Change { std::vector<V> x,v; double impulse=0; };
struct Model {
 std::vector<Tet> tets;std::vector<double> mass,area;
 std::vector<std::vector<std::pair<std::uint32_t,std::uint8_t>>> incidence;
 double d=1e-4,k=1e16,mu=2000,lambda=5000;
 State initial;
 explicit Model(int n){
  int count=(n+1)*(n+1)*(n+1);initial.x.resize(count);initial.v.resize(count);
  mass.assign(count,0);area.assign(count,0);
  auto id=[n](int x,int y,int z){return (z*(n+1)+y)*(n+1)+x;};
  for(int z=0;z<=n;++z)for(int y=0;y<=n;++y)for(int x=0;x<=n;++x)
   initial.x[id(x,y,z)]={.02*x/n,.02*y/n,.0002+.02*z/n};
  constexpr int mesh[6][4]={{0,1,3,7},{0,3,2,7},{0,2,6,7},{0,6,4,7},{0,4,5,7},{0,5,1,7}};
  for(int z=0;z<n;++z)for(int y=0;y<n;++y)for(int x=0;x<n;++x){
   int v[8]={id(x,y,z),id(x+1,y,z),id(x,y+1,z),id(x+1,y+1,z),
             id(x,y,z+1),id(x+1,y,z+1),id(x,y+1,z+1),id(x+1,y+1,z+1)};
   for(auto& t:mesh){
    Tet e;for(int i=0;i<4;++i)e.id[i]=v[t[i]];
    M dm=edges(initial.x[e.id[0]],initial.x[e.id[1]],initial.x[e.id[2]],initial.x[e.id[3]]);
    e.volume=std::abs(determinant(dm))/6;e.inv=inverse(dm);tets.push_back(e);
    for(int i:e.id)mass[i]+=1100*e.volume/4;
   }
   if(z==0){
    const double share=.02*.02/(n*n*6.);
    for(int i:{0,1,3,0,3,2})area[v[i]]+=share;
   }
  }
  incidence.resize(count);
  for(std::uint32_t k=0;k<tets.size();++k)for(std::uint8_t slot=0;slot<4;++slot)
   incidence[tets[k].id[slot]].emplace_back(k,slot);
 }
 double pressure(double g)const {
  if(g>=d)return 0;
  if(g<=0)throw std::runtime_error("nonpositive gap");
  double q=g*g-d*d;
  return 2*g*k*(2*q*std::log(g*g/(d*d))+q*q/(g*g));
 }
 double energy(const State& s)const {
  double out=0;
  for(const Tet& e:tets){
   M ds=edges(s.x[e.id[0]],s.x[e.id[1]],s.x[e.id[2]],s.x[e.id[3]]),F;
   for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int q=0;q<3;++q)F.a[i][j]+=ds.a[i][q]*e.inv.a[q][j];
   double J=determinant(F),tr=0;
   if(!(J>0))throw std::runtime_error("inverted tetrahedron");
   for(auto& row:F.a)for(double x:row)tr+=x*x;
   double l=std::log(J);out+=e.volume*(mu*(tr-3)/2-mu*l+lambda*l*l/2);
  }
  for(size_t i=0;i<mass.size();++i){
   out+=mass[i]*9.81*s.x[i].z;
   double g=s.x[i].z;
   if(area[i] && g<d){
    if(g<=0)throw std::runtime_error("nonpositive gap");
    double q=g*g-d*d;out-=area[i]*k*q*q*std::log(g*g/(d*d));
   }
  }
  return out;
 }
 Change rhs(const State& s)const {
  Change r;r.x=s.v;r.v.assign(mass.size(),{0,0,0});std::vector<V> f(mass.size());
  std::vector<std::array<V,3>> contributions(tets.size());
  struct ElementContext {const Model* model; const State* state;
   std::vector<std::array<V,3>>* result; std::atomic_bool invalid{false};} elements{this,&s,&contributions};
  dispatch_apply_f(4,dispatch_get_global_queue(QOS_CLASS_UTILITY,0),&elements,
   +[](void* opaque,size_t worker){
    auto& ctx=*static_cast<ElementContext*>(opaque);const auto& model=*ctx.model;const auto& state=*ctx.state;
    const size_t begin=model.tets.size()*worker/4,end=model.tets.size()*(worker+1)/4;
    for(size_t eIndex=begin;eIndex<end;++eIndex){
     const Tet& e=model.tets[eIndex];
     M ds=edges(state.x[e.id[0]],state.x[e.id[1]],state.x[e.id[2]],state.x[e.id[3]]),F;
     for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int q=0;q<3;++q)F.a[i][j]+=ds.a[i][q]*e.inv.a[q][j];
     double J=determinant(F);if(!(J>0)){ctx.invalid.store(true,std::memory_order_relaxed);continue;}
     M invF=inverse(F),P;
     for(int i=0;i<3;++i)for(int j=0;j<3;++j)
      P.a[i][j]=model.mu*F.a[i][j]+(model.lambda*std::log(J)-model.mu)*invF.a[j][i];
     for(int j=0;j<3;++j){
      V v;
      for(int q=0;q<3;++q){v.x-=e.volume*P.a[0][q]*e.inv.a[j][q];
       v.y-=e.volume*P.a[1][q]*e.inv.a[j][q];v.z-=e.volume*P.a[2][q]*e.inv.a[j][q];}
      (*ctx.result)[eIndex][j]=v;
     }
    }
   });
  if(elements.invalid.load(std::memory_order_relaxed))throw std::runtime_error("inverted tetrahedron");
  struct NodeContext {const Model* model;const std::vector<std::array<V,3>>* contributions;std::vector<V>* result;}
   nodes{this,&contributions,&f};
  dispatch_apply_f(4,dispatch_get_global_queue(QOS_CLASS_UTILITY,0),&nodes,
   +[](void* opaque,size_t worker){
    auto& ctx=*static_cast<NodeContext*>(opaque);const size_t n=ctx.model->mass.size();
    const size_t begin=n*worker/4,end=n*(worker+1)/4;
    for(size_t i=begin;i<end;++i){
     V value;
     for(const auto& [tet,slot]:ctx.model->incidence[i]){
      const auto& v=(*ctx.contributions)[tet];
      if(slot==0){value+=v[0]*(-1);value+=v[1]*(-1);value+=v[2]*(-1);}
      else value+=v[slot-1];
     }
     (*ctx.result)[i]=value;
    }
   });
  for(size_t i=0;i<mass.size();++i){
   if(area[i]){double fz=area[i]*pressure(s.x[i].z);f[i].z+=fz;r.impulse+=fz;}
   r.v[i]=f[i]*(1./mass[i]);r.v[i].z-=9.81;
  }
  return r;
 }
};
State add(const State& s,const Change& k,double h) {
 State r=s;
 for(size_t i=0;i<r.x.size();++i){r.x[i]+=k.x[i]*h;r.v[i]+=k.v[i]*h;}
 r.impulse+=h*k.impulse;return r;
}
int main(int argc,char**argv){
 try{
  if(argc==2 && std::string(argv[1])=="--selftest"){
   Model m(2);State s=m.initial;
   for(auto& p:s.x){p.z=.000095+(p.z-.0002)*.98;p.x+=.02*(p.z-.000095);}
   Change f=m.rhs(s);double maxerr=0,maxforce=0;const double eps=1e-9;
   for(size_t i=0;i<s.x.size();++i)for(int j=0;j<3;++j){
    State a=s,b=s;double *u=j==0?&a.x[i].x:j==1?&a.x[i].y:&a.x[i].z;
    double *v=j==0?&b.x[i].x:j==1?&b.x[i].y:&b.x[i].z;
    *u+=eps;*v-=eps;double measured=-(m.energy(a)-m.energy(b))/(2*eps);
    double actual=m.mass[i]*(j==0?f.v[i].x:j==1?f.v[i].y:f.v[i].z);
    maxerr=std::max(maxerr,std::abs(measured-actual));maxforce=std::max(maxforce,std::abs(actual));
   }
   std::cout<<"energy-gradient absolute "<<maxerr<<" relative "<<maxerr/maxforce<<" maxforce "<<maxforce<<"\n";
   if(maxerr/std::max(maxforce,1e-30)>1e-6)throw std::runtime_error("force/energy mismatch");
   return 0;
  }
  if(argc!=4)throw std::runtime_error("usage: reference divisions h duration");
  int n=std::atoi(argv[1]);double h=std::atof(argv[2]),duration=std::atof(argv[3]);
  if(n<1||n>NUMI_FEM_REFERENCE_MAX_DIVISIONS||h<=0||duration!=.01||std::abs(std::round(duration/h)*h-duration)>1e-12
     ||std::abs(std::round(.001/h)*h-.001)>1e-12)throw std::runtime_error("invalid arguments");
  Model m(n);State s=m.initial;std::array<double,10> bins{};double minGap=1,peak=0,first=-1,lastForce=0;
  const long requestedStride=std::lround(.000002/h);
  const long traceStride=requestedStride>=1 && std::abs(requestedStride*h-.000002)<=1e-12 ? requestedStride : 0;
  std::vector<std::array<double,3>> trace{{0.,m.rhs(s).impulse,0.}};
  std::vector<std::array<double,2>> episodes;double episodeStart=-1;
  long steps=std::lround(duration/h);
  for(long i=0;i<steps;++i){
   Change a=m.rhs(s),b=m.rhs(add(s,a,h/2)),c=m.rhs(add(s,b,h/2)),d=m.rhs(add(s,c,h));
   for(size_t j=0;j<s.x.size();++j){
    s.x[j]+=(a.x[j]+b.x[j]*2+c.x[j]*2+d.x[j])*(h/6);
    s.v[j]+=(a.v[j]+b.v[j]*2+c.v[j]*2+d.v[j])*(h/6);
    minGap=std::min(minGap,s.x[j].z);
   }
   double di=h*(a.impulse+2*b.impulse+2*c.impulse+d.impulse)/6;
   s.impulse+=di;bins[std::min(9,int((i+.5)*h/.001))]+=di/.001;
   lastForce=m.rhs(s).impulse;peak=std::max(peak,lastForce);
   if(traceStride>0 && (i+1)%traceStride==0)trace.push_back({(i+1)*h,lastForce,s.impulse});
   if(first<0 && lastForce>0)first=(i+1)*h;
   if(episodeStart<0 && lastForce>0)episodeStart=(i+1)*h;
   if(episodeStart>=0 && lastForce==0){episodes.push_back({episodeStart,(i+1)*h});episodeStart=-1;}
  }
  if(episodeStart>=0)episodes.push_back({episodeStart,duration});
  double totalMass=0,totalArea=0,pz=0,com=0;
  for(size_t i=0;i<m.mass.size();++i){totalMass+=m.mass[i];totalArea+=m.area[i];pz+=m.mass[i]*s.v[i].z;com+=m.mass[i]*s.x[i].z;}
  std::cout<<std::setprecision(17)<<"{\"schema\":\"numi.independent-fem-rk4.v1\",\"mesh_divisions\":"<<n<<",\"h\":"<<h
   <<",\"duration\":"<<duration<<",\"nodes\":"<<m.mass.size()<<",\"tetrahedra\":"<<m.tets.size()
   <<",\"mass_kg\":"<<totalMass<<",\"area_m2\":"<<totalArea<<",\"min_gap_m\":"<<minGap
   <<",\"onset_time_s\":"<<first<<",\"peak_force_N\":"<<peak<<",\"impulse_Ns\":"<<s.impulse
   <<",\"final_momentum_Ns\":"<<pz<<",\"final_center_z_m\":"<<com/totalMass<<",\"force_bins_N\":[";
  for(int i=0;i<10;++i)std::cout<<(i?",":"")<<bins[i];
  std::cout<<"],\"contact_area_weights_m2\":[";
  for(size_t i=0;i<m.area.size();++i)std::cout<<(i?",":"")<<m.area[i];
  std::cout<<"],\"trace\":[";
  for(size_t i=0;i<trace.size();++i)std::cout<<(i?",":"")<<"["<<trace[i][0]<<","<<trace[i][1]<<","<<trace[i][2]<<"]";
  std::cout<<"],\"contact_episodes_s\":[";
  for(size_t i=0;i<episodes.size();++i)std::cout<<(i?",":"")<<"["<<episodes[i][0]<<","<<episodes[i][1]<<"]";
  std::cout<<"]}\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}

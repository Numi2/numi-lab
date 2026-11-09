// Bounded-memory independent reference for the same structured FEM fixture.
// Cell coloring prevents concurrent writes to shared vertices. This is research
// evidence only; equivalence must be established before using larger meshes.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main numi_reference_original_entry
#include "fem_plane_fp64_reference.cpp"
#undef main
#pragma clang diagnostic pop
#include <atomic>
#include <dispatch/dispatch.h>
#include <cstdint>
#include <limits>
#include <utility>
#include <sys/resource.h>

struct CompactModel {
 int n;
 bool surfaceQuadrature=false,gradedZ=false;
 std::vector<Triangle> triangles;
 std::vector<double> mass,area;
 State initial;
#ifdef NUMI_COMPACT_TEMPLATE_INVERSE
 std::array<M,6> templateInverse{};
#endif
 double d=1e-4,k=1e16,mu=2000,lambda=5000;
 static constexpr int mesh[6][4]={{0,1,3,7},{0,3,2,7},{0,2,6,7},{0,6,4,7},{0,4,5,7},{0,5,1,7}};
 int id(int x,int y,int z) const {return (z*(n+1)+y)*(n+1)+x;}
 void vertices(int x,int y,int z,int (&v)[8]) const {
  v[0]=id(x,y,z);v[1]=id(x+1,y,z);v[2]=id(x,y+1,z);v[3]=id(x+1,y+1,z);
  v[4]=id(x,y,z+1);v[5]=id(x+1,y,z+1);v[6]=id(x,y+1,z+1);v[7]=id(x+1,y+1,z+1);
 }
 explicit CompactModel(int divisions,bool quadratureEnabled=false,bool graded=false):
  n(divisions),surfaceQuadrature(quadratureEnabled),gradedZ(graded) {
  if(n<1||n>256)throw std::runtime_error("invalid mesh capacity");
  const int count=(n+1)*(n+1)*(n+1);
  initial.x.resize(count);initial.v.resize(count);mass.assign(count,0);area.assign(count,0);
#ifdef NUMI_COMPACT_TEMPLATE_INVERSE
  const V unit[8]={{0,0,0},{1,0,0},{0,1,0},{1,1,0},
                   {0,0,1},{1,0,1},{0,1,1},{1,1,1}};
  for(size_t i=0;i<6;++i){const auto& t=mesh[i];
   const M dm=edges(unit[t[0]],unit[t[1]],unit[t[2]],unit[t[3]]);
   if(std::abs(determinant(dm))!=1.)throw std::runtime_error("non-unit template");
   templateInverse[i]=inverse(dm);
  }
#endif
  for(int z=0;z<=n;++z)for(int y=0;y<=n;++y)for(int x=0;x<=n;++x)
   {const double u=double(z)/n;const double depth=gradedZ?(u+u*u)*.5:u;
    initial.x[id(x,y,z)]={.02*x/n,.02*y/n,.0002+.02*depth};}
  for(int z=0;z<n;++z)for(int y=0;y<n;++y)for(int x=0;x<n;++x) {
   int v[8];vertices(x,y,z,v);
   for(const auto& t:mesh) {
    const M dm=edges(initial.x[v[t[0]]],initial.x[v[t[1]]],initial.x[v[t[2]]],initial.x[v[t[3]]]);
    const double volume=std::abs(determinant(dm))/6;
    for(int i:t)mass[v[i]]+=1100*volume/4;
   }
   if(z==0){const double share=.02*.02/(n*n*6.);
    for(int i:{0,1,3,0,3,2})area[v[i]]+=share;
    if(surfaceQuadrature){const double triangleArea=.02*.02/(2.*n*n);
     triangles.push_back({{v[0],v[1],v[3]},triangleArea});
     triangles.push_back({{v[0],v[3],v[2]},triangleArea});}}
  }
 }
 double pressure(double g)const {
  if(g>=d)return 0;if(g<=0)throw std::runtime_error("nonpositive gap");
  const double q=g*g-d*d;
  return 2*g*k*(2*q*std::log(g*g/(d*d))+q*q/(g*g));
 }
 Change rhs(const State& s)const {
  Change r;r.x=s.v;r.v.assign(mass.size(),{0,0,0});std::vector<V> f(mass.size());
  struct Context {const CompactModel* m;const State* s;std::vector<V>* f;
   int cx,cy,cz;std::atomic_bool invalid{false};};
  for(int color=0;color<8;++color) {
   Context ctx{this,&s,&f,color&1,(color>>1)&1,(color>>2)&1};
   dispatch_apply_f(4,dispatch_get_global_queue(QOS_CLASS_UTILITY,0),&ctx,
    +[](void* ptr,size_t worker) {
     auto& c=*static_cast<Context*>(ptr);const auto& m=*c.m;
     const int nx=(m.n+1-c.cx)/2,ny=(m.n+1-c.cy)/2,nz=(m.n+1-c.cz)/2;
     const size_t total=size_t(nx)*ny*nz,begin=total*worker/4,end=total*(worker+1)/4;
     for(size_t idx=begin;idx<end;++idx){
      const int x=2*int(idx%nx)+c.cx,y=2*int((idx/nx)%ny)+c.cy,z=2*int(idx/(size_t(nx)*ny))+c.cz;
      int v[8];m.vertices(x,y,z,v);
      for(size_t ti=0;ti<6;++ti) {
       const auto& t=mesh[ti];
       const int a=v[t[0]],b=v[t[1]],d=v[t[2]],e=v[t[3]];
#ifdef NUMI_COMPACT_TEMPLATE_INVERSE
       // The structured reference cell is diag(dx,dy,dz) times this
       // tetrahedron template. Axis lengths come from the original rounded
       // node coordinates, rather than an assumed perfectly uniform grid.
       const double dx=m.initial.x[v[1]].x-m.initial.x[v[0]].x;
       const double dy=m.initial.x[v[2]].y-m.initial.x[v[0]].y;
       const double dz=m.initial.x[v[4]].z-m.initial.x[v[0]].z;
       const double volume=std::abs(dx*dy*dz)/6;
       M inv=m.templateInverse[ti];
       for(int row=0;row<3;++row){inv.a[row][0]/=dx;inv.a[row][1]/=dy;inv.a[row][2]/=dz;}
#else
       const M dm=edges(m.initial.x[a],m.initial.x[b],m.initial.x[d],m.initial.x[e]);
       const double volume=std::abs(determinant(dm))/6;
       const M inv=inverse(dm);
#endif
       if(!(volume>0)){c.invalid.store(true,std::memory_order_relaxed);continue;}
       const M ds=edges(c.s->x[a],c.s->x[b],c.s->x[d],c.s->x[e]);M F;
       for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int q=0;q<3;++q)F.a[i][j]+=ds.a[i][q]*inv.a[q][j];
       const double J=determinant(F);if(!(J>1e-30)||!std::isfinite(J)){c.invalid.store(true,std::memory_order_relaxed);continue;}
       const M invF=inverse(F);M P;
       for(int i=0;i<3;++i)for(int j=0;j<3;++j)P.a[i][j]=m.mu*F.a[i][j]+(m.lambda*std::log(J)-m.mu)*invF.a[j][i];
       const int nodes[3]={b,d,e};
       for(int j=0;j<3;++j){V w;
        for(int q=0;q<3;++q){w.x-=volume*P.a[0][q]*inv.a[j][q];
         w.y-=volume*P.a[1][q]*inv.a[j][q];w.z-=volume*P.a[2][q]*inv.a[j][q];}
        (*c.f)[nodes[j]]+=w;(*c.f)[a]+=w*(-1);
       }
      }
     }
    });
   if(ctx.invalid.load(std::memory_order_relaxed))throw std::runtime_error("invalid element");
  }
  if(surfaceQuadrature)for(const auto& triangle:triangles){
   requireTriangleClearance(triangle,s);
   for(const auto& q:quadrature){
   double gap=0;for(int j=0;j<3;++j)gap+=q.n[j]*s.x[triangle.id[j]].z;
   const double force=triangle.area*q.w*pressure(gap);r.impulse+=force;
   for(int j=0;j<3;++j)f[triangle.id[j]].z+=q.n[j]*force;
   }
  }
  for(size_t i=0;i<mass.size();++i){
   if(!surfaceQuadrature && area[i]){const double force=area[i]*pressure(s.x[i].z);f[i].z+=force;r.impulse+=force;}
   r.v[i]=f[i]*(1./mass[i]);r.v[i].z-=9.81;
  }
  return r;
 }
};

int main(int argc,char** argv){try{
 if(argc==3 && (std::string(argv[1])=="--capacity-check" ||
                  std::string(argv[1])=="--rk4-capacity-check")){
  const bool rk4=std::string(argv[1])=="--rk4-capacity-check";
  const int n=std::stoi(argv[2]);CompactModel m(n);
  double totalMass=0,totalArea=0,minMass=std::numeric_limits<double>::infinity();
  for(size_t i=0;i<m.mass.size();++i){totalMass+=m.mass[i];totalArea+=m.area[i];
   minMass=std::min(minMass,m.mass[i]);}
  if(!(minMass>0))throw std::runtime_error("invalid reference mass");
  double finalForce=0,velocityError=0,positionError=0;
  if(rk4){
   constexpr double h=2.5e-7;State s=m.initial;
   Change a=m.rhs(s),b=m.rhs(add(s,a,h/2)),c=m.rhs(add(s,b,h/2)),d=m.rhs(add(s,c,h));
   for(size_t i=0;i<s.x.size();++i){s.x[i]+=(a.x[i]+b.x[i]*2+c.x[i]*2+d.x[i])*(h/6);
    s.v[i]+=(a.v[i]+b.v[i]*2+c.v[i]*2+d.v[i])*(h/6);}
   finalForce=m.rhs(s).impulse;
   for(size_t i=0;i<s.x.size();++i){
    velocityError=std::max(velocityError,std::abs(s.v[i].z+9.81*h));
    positionError=std::max(positionError,std::abs(s.x[i].z-(m.initial.x[i].z-.5*9.81*h*h)));}
  }else finalForce=m.rhs(m.initial).impulse;
  if(!std::isfinite(finalForce)||!std::isfinite(velocityError)||!std::isfinite(positionError))
   throw std::runtime_error("invalid reference capacity state");
  struct rusage usage{};if(getrusage(RUSAGE_SELF,&usage)!=0)throw std::runtime_error("peak RSS unavailable");
  std::cout<<std::setprecision(17)<<"{\"max_rss_bytes_self\":"<<usage.ru_maxrss<<",\"mesh_divisions\":"<<n<<",\"nodes\":"<<m.mass.size()
   <<",\"tetrahedra\":"<<size_t(6)*n*n*n<<",\"area_m2\":"<<totalArea
   <<",\"mass_kg\":"<<totalMass<<",\"minimum_nodal_mass_kg\":"<<minMass
   <<",\"force_N\":"<<finalForce<<",\"rk4_step\":"<<(rk4?"true":"false")
   <<",\"max_velocity_error_mps\":"<<velocityError<<",\"max_position_error_m\":"<<positionError<<"}\n";
  return 0;
 }
 if(argc==2 && std::string(argv[1])=="--clearance-selftest"){
  CompactModel m(2,true,true);State s=m.initial;
  for(size_t i=0;i<m.area.size();++i)if(m.area[i]>0)s.x[i].z=10e-6;
  const auto& triangle=m.triangles.front();s.x[triangle.id[0]].z=-1e-6;
  for(const auto& q:quadrature){double gap=0;for(int j=0;j<3;++j)gap+=q.n[j]*s.x[triangle.id[j]].z;
   if(!(gap>0))throw std::runtime_error("counterexample samples must all be positive");}
  try{(void)m.rhs(s);}catch(const std::runtime_error& e){
   if(std::string(e.what())=="nonpositive or nonfinite triangle vertex gap"){
    std::cout<<"compact triangle vertex penetration rejected despite positive quadrature gaps\n";return 0;}
   throw;
  }
  throw std::runtime_error("compact triangle vertex penetration accepted");
 }
 if(argc==2 && (std::string(argv[1])=="--parity" || std::string(argv[1])=="--parity-quadrature")){
  const bool q=std::string(argv[1])=="--parity-quadrature";
  double maxRelative=0,maxMass=0,maxArea=0,maxImpulse=0;
  for(int n:{2,4,8}){Model reference(n,.0002,q,q);CompactModel compact(n,q,q);State s=reference.initial;
   for(size_t i=0;i<s.x.size();++i){s.x[i].z-=.000105;
    s.x[i].x+=1e-5*std::sin(double(i)*.13);s.x[i].z+=1e-6*std::cos(double(i)*.37);
    s.v[i]={.01*std::sin(double(i)),.02*std::cos(double(i)),0};}
   const auto a=reference.rhs(s),b=compact.rhs(s);double difference=0,scale=0;
   for(size_t i=0;i<s.x.size();++i){
    maxMass=std::max(maxMass,std::abs(reference.mass[i]-compact.mass[i]));
    maxArea=std::max(maxArea,std::abs(reference.area[i]-compact.area[i]));
    for(const auto pair:{std::pair{a.v[i].x,b.v[i].x},std::pair{a.v[i].y,b.v[i].y},std::pair{a.v[i].z,b.v[i].z}}){
     difference=std::max(difference,std::abs(pair.first-pair.second));scale=std::max(scale,std::abs(pair.first));}}
   maxRelative=std::max(maxRelative,difference/std::max(scale,1e-30));
   maxImpulse=std::max(maxImpulse,std::abs(a.impulse-b.impulse));
  }
  std::cout<<std::setprecision(17)<<"{\"acceleration_relative_max\":"<<maxRelative
   <<",\"mass_absolute_max\":"<<maxMass<<",\"area_absolute_max\":"<<maxArea
   <<",\"impulse_absolute_max\":"<<maxImpulse<<"}\n";
  if(maxRelative>1e-10||maxMass!=0||maxArea!=0||maxImpulse!=0)throw std::runtime_error("reference parity failed");
  return 0;
 }
 if(argc<4)throw std::runtime_error("usage: compact divisions timestep duration [--surface-quadrature] [--graded-z]");
 const int n=std::stoi(argv[1]);const double h=std::stod(argv[2]),duration=std::stod(argv[3]);
 bool q=false,graded=false;for(int i=4;i<argc;++i){const std::string flag=argv[i];
  if(flag=="--surface-quadrature")q=true;else if(flag=="--graded-z")graded=true;
  else throw std::runtime_error("unknown option: "+flag);}
 if(!(h>0)||duration!=.01||std::abs(std::round(duration/h)*h-duration)>1e-12||
    std::abs(std::round(.000002/h)*h-.000002)>1e-12)throw std::runtime_error("invalid time grid");
 CompactModel m(n,q,graded);State s=m.initial;std::array<double,10> bins{};double minGap=1,peak=0,first=-1,lastForce=0;
 const long traceStride=std::lround(.000002/h),steps=std::lround(duration/h);
 std::vector<std::array<double,3>> trace{{0.,m.rhs(s).impulse,0.}};
 for(long i=0;i<steps;++i){
  Change a=m.rhs(s),b=m.rhs(add(s,a,h/2)),c=m.rhs(add(s,b,h/2)),d=m.rhs(add(s,c,h));
  for(size_t j=0;j<s.x.size();++j){s.x[j]+=(a.x[j]+b.x[j]*2+c.x[j]*2+d.x[j])*(h/6);
   s.v[j]+=(a.v[j]+b.v[j]*2+c.v[j]*2+d.v[j])*(h/6);minGap=std::min(minGap,s.x[j].z);}
  const double di=h*(a.impulse+2*b.impulse+2*c.impulse+d.impulse)/6;
  s.impulse+=di;bins[std::min(9,int((i+.5)*h/.001))]+=di/.001;
  lastForce=m.rhs(s).impulse;peak=std::max(peak,lastForce);
  if((i+1)%traceStride==0)trace.push_back({(i+1)*h,lastForce,s.impulse});
  if(first<0&&lastForce>0)first=(i+1)*h;
 }
 double totalMass=0,totalArea=0;for(size_t i=0;i<m.mass.size();++i){totalMass+=m.mass[i];totalArea+=m.area[i];}
 std::cout<<std::setprecision(17)<<"{\"schema\":\"numi.independent-fem-rk4-compact.v1\",\"mesh_divisions\":"<<n<<",\"h\":"<<h
  <<",\"duration\":"<<duration<<",\"nodes\":"<<m.mass.size()<<",\"tetrahedra\":"<<size_t(6)*n*n*n
  <<",\"mass_kg\":"<<totalMass<<",\"area_m2\":"<<totalArea<<",\"min_gap_m\":"<<minGap
  <<",\"onset_time_s\":"<<first<<",\"peak_force_N\":"<<peak<<",\"impulse_Ns\":"<<s.impulse
  <<",\"force_bins_N\":[";
 for(int i=0;i<10;++i)std::cout<<(i?",":"")<<bins[i];
 std::cout<<"],\"contact_area_weights_m2\":[";
 for(size_t i=0;i<m.area.size();++i)std::cout<<(i?",":"")<<m.area[i];
 std::cout<<"],\"trace\":[";
 for(size_t i=0;i<trace.size();++i)std::cout<<(i?",":"")<<"["<<trace[i][0]<<","<<trace[i][1]<<","<<trace[i][2]<<"]";
 std::cout<<"]";if(q||graded)std::cout<<",\"surface_quadrature\":"<<(q?"true":"false")<<",\"graded_z\":"<<(graded?"true":"false");std::cout<<"}\n";
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}

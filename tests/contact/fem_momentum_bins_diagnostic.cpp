// Independently observe each bin's contact impulse through global momentum.
// Same independent FP64 fixed-law state evolution as the retained RK4 study.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main surface_diagnostic_main
#include "fem_plane_surface_diagnostic.cpp"
#undef main
#pragma clang diagnostic pop
int main(int argc,char**argv){
 try{
  if(argc!=2)throw std::runtime_error("usage: n");
  int n=std::atoi(argv[1]);if(n!=16&&n!=32)throw std::runtime_error("invalid n");
  const double h=1e-6,bin=.001;const int count=1000;
  Model m(n);State s=m.initial;double mass=0;
  for(double w:m.mass)mass+=w;
  auto momentum=[&](){double x=0;for(size_t j=0;j<m.mass.size();++j)x+=m.mass[j]*s.v[j].z;return x;};
  std::cout<<std::setprecision(17)<<"bin,time,p_start,p_end,force_from_momentum,force_from_pressure,min_gap\n";
  for(int i=0;i<10;++i){
   const double start=momentum();double impulse=0,minGap=1;
   for(int j=0;j<count;++j){
    Change a=m.rhs(s),b=m.rhs(add(s,a,h/2)),c=m.rhs(add(s,b,h/2)),d=m.rhs(add(s,c,h));
    for(size_t k=0;k<s.x.size();++k){
     s.x[k]+=(a.x[k]+b.x[k]*2+c.x[k]*2+d.x[k])*(h/6);
     s.v[k]+=(a.v[k]+b.v[k]*2+c.v[k]*2+d.v[k])*(h/6);
     if(m.area[k])minGap=std::min(minGap,s.x[k].z);
    }
    impulse+=h*(a.impulse+2*b.impulse+2*c.impulse+d.impulse)/6;
   }
   double end=momentum();
   std::cout<<i<<","<<(i+1)*bin<<","<<start<<","<<end<<","
      <<(end-start+mass*9.81*bin)/bin<<","<<impulse/bin<<","<<minGap<<"\n";
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}

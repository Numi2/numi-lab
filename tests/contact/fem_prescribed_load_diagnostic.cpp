// Contact-free, free-sided independent FEM under a common prescribed load.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main surface_diagnostic_main
#include "fem_plane_surface_diagnostic.cpp"
#undef main
#pragma clang diagnostic pop
#include <fstream>
#include <sstream>
int main(int argc,char**argv){
 try{
  if(argc!=4)throw std::runtime_error("usage: n h input.csv");
  int n=std::atoi(argv[1]);double h=std::atof(argv[2]);
  if((n!=16&&n!=32)||(h!=1e-6&&h!=2e-6))throw std::runtime_error("invalid arguments");
  std::ifstream input(argv[3]);std::string line;
  if(!std::getline(input,line)||line!="time_s,total_force_N")throw std::runtime_error("invalid input");
  std::vector<double> t,f;
  while(std::getline(input,line)){std::stringstream ss(line);std::string x,y;
   if(!std::getline(ss,x,',')||!std::getline(ss,y,','))throw std::runtime_error("invalid row");
   t.push_back(std::stod(x));f.push_back(std::stod(y));
   if(t.size()>1 && !(t.back()>t[t.size()-2]))throw std::runtime_error("nonmonotonic input");
  }
  if(t.size()!=101||t.front()!=0||std::abs(t.back()-.01)>1e-12)throw std::runtime_error("invalid input extent");
  auto prescribed=[&](double now){
   auto it=std::upper_bound(t.begin(),t.end(),now);if(it==t.begin())return f.front();
   if(it==t.end())return f.back();size_t i=it-t.begin()-1;
   double q=(now-t[i])/(t[i+1]-t[i]);return f[i]*(1-q)+f[i+1]*q;
  };
  Model m(n);m.d=-1e9;State s=m.initial;double totalArea=0,mass=0;
  for(double a:m.area)totalArea+=a;for(double x:m.mass)mass+=x;
  auto rhs=[&](const State& x,double now){
   Change r=m.rhs(x);if(r.impulse!=0)throw std::runtime_error("unexpected contact");
   double force=prescribed(now);
   for(size_t i=0;i<m.mass.size();++i)r.v[i].z+=force*m.area[i]/(totalArea*m.mass[i]);
   r.impulse=force;return r;
  };
  auto emit=[&](double now){
   double mean=0,vel=0,pz=0,com=0;
   for(size_t i=0;i<m.mass.size();++i){
    mean+=m.area[i]*(s.x[i].z-m.initial.x[i].z)/totalArea;
    vel+=m.area[i]*s.v[i].z/totalArea;
    pz+=m.mass[i]*s.v[i].z;com+=m.mass[i]*s.x[i].z/mass;
   }
   double spread=0;
   for(size_t i=0;i<m.mass.size();++i){
    double du=s.x[i].z-m.initial.x[i].z-mean;spread+=m.area[i]*du*du/totalArea;
   }
   std::cout<<now<<","<<prescribed(now)<<","<<mean<<","<<std::sqrt(spread)<<","
            <<vel<<","<<pz<<","<<com<<","<<s.impulse<<"\n";
  };
  std::cout<<std::setprecision(17)<<"time_s,force_N,surface_mean_displacement_m,surface_displacement_rms_m,surface_mean_velocity_mps,momentum_Ns,center_z_m,prescribed_impulse_Ns\n";
  emit(0);long steps=std::lround(.01/h),trace=std::lround(.0001/h);
  for(long i=0;i<steps;++i){
   double now=i*h;Change a=rhs(s,now),b=rhs(add(s,a,h/2),now+h/2),
    c=rhs(add(s,b,h/2),now+h/2),d=rhs(add(s,c,h),now+h);
   for(size_t j=0;j<s.x.size();++j){
    s.x[j]+=(a.x[j]+b.x[j]*2+c.x[j]*2+d.x[j])*(h/6);
    s.v[j]+=(a.v[j]+b.v[j]*2+c.v[j]*2+d.v[j])*(h/6);
   }
   s.impulse+=h*(a.impulse+2*b.impulse+2*c.impulse+d.impulse)/6;
   if((i+1)%trace==0)emit((i+1)*h);
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}

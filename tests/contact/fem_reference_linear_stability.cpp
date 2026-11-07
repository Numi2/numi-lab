// Gershgorin upper bound of the mass-scaled linear elastic/contact tangent
// at the stress-free reference shape and a chosen uniform active-contact gap.
// It is not a bound over the subsequent nonlinear trajectory.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main reference_entry
#include "fem_plane_fp64_reference.cpp"
#undef main
#pragma clang diagnostic pop
int main(int argc,char**argv){
 if(argc!=2)return 2;
 const int n=std::atoi(argv[1]);
 if(n<1||n>32)return 2;
 // Build the same geometry without modifying the unqualified runtime path.
 Model model(n);std::vector<std::array<double,3>> row(model.mass.size());
 for(const Tet& t:model.tets){
  V g[4]={{0,0,0},{t.inv.a[0][0],t.inv.a[0][1],t.inv.a[0][2]},
   {t.inv.a[1][0],t.inv.a[1][1],t.inv.a[1][2]},
   {t.inv.a[2][0],t.inv.a[2][1],t.inv.a[2][2]}};
  g[0]=(g[1]+g[2]+g[3])*(-1);
  auto c=[](V x,int a){return a==0?x.x:a==1?x.y:x.z;};
  auto l1=[](V x){return std::abs(x.x)+std::abs(x.y)+std::abs(x.z);};
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int a=0;a<3;++a){
   double dot=g[i].x*g[j].x+g[i].y*g[j].y+g[i].z*g[j].z;
   row[t.id[i]][a]+=t.volume*(model.lambda*std::abs(c(g[i],a))*l1(g[j])
      +model.mu*std::abs(dot)+model.mu*l1(g[i])*std::abs(c(g[j],a)));
  }
 }
 // Exact derivative of pressure p(g) for the chosen reference equilibrium gap.
 const double gap=9.697070413317377e-5,q=gap*gap-model.d*model.d;
 const double term=2*q*std::log(gap*gap/(model.d*model.d))+q*q/(gap*gap);
 const double derivative=4*gap*std::log(gap*gap/(model.d*model.d))
      +8*q/gap-2*q*q/(gap*gap*gap);
 const double stiffness=-2*model.k*(term+gap*derivative);
 double maxrow=0;
 for(size_t i=0;i<row.size();++i){
  row[i][2]+=model.area[i]*stiffness;
  for(int a=0;a<3;++a)maxrow=std::max(maxrow,row[i][a]/model.mass[i]);
 }
 const double omega=std::sqrt(maxrow);
 const double hmax=2*std::sqrt(2.)/omega;
 const size_t nnode=model.mass.size(),ntet=model.tets.size();
 std::cout<<std::setprecision(17)<<"{\"n\":"<<n<<",\"nodes\":"<<nnode
  <<",\"tetrahedra\":"<<ntet<<",\"row_sum_eigenvalue_bound_s_minus_2\":"<<maxrow
  <<",\"omega_upper_at_reference_rad_per_s\":"<<omega<<",\"rk4_hmax_at_reference_s\":"<<hmax
  <<",\"h_2us_times_omega_upper\":"<<2e-6*omega
  <<",\"tet_storage_bytes\":"<<ntet*sizeof(Tet)<<",\"single_node_vector_bytes\":"<<nnode*sizeof(V)
  <<",\"limits\":[\"bound at stress-free reference shape and chosen uniform contact gap only\","
  <<"\"not a bound over nonlinear trajectory or a temporal-accuracy certificate\"]}\n";
}

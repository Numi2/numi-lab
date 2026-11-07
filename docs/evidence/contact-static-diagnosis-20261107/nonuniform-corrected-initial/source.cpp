// Manufactured nonuniform static plane-contact transfer control.
// Analytic affine deformation, independent high-order external traction.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main surface_entry
#include "fem_plane_surface_diagnostic.cpp"
#undef main
#pragma clang diagnostic pop
struct Integration {std::vector<double> external;double normal=0,mx=0,my=0;};
static Integration externalLoad(const Model&m,const State&s,double pzz,int order){
 static constexpr double x4[]={-.8611363115940525752,-.3399810435848562648,
  .3399810435848562648,.8611363115940525752};
 static constexpr double w4[]={.3478548451374538574,.6521451548625461426,
  .6521451548625461426,.3478548451374538574};
 static constexpr double x8[]={-.9602898564975362317,-.7966664774136267396,
  -.5255324099163289858,-.1834346424956498049,.1834346424956498049,
  .5255324099163289858,.7966664774136267396,.9602898564975362317};
 static constexpr double w8[]={.1012285362903762592,.2223810344533744705,
  .3137066458778872873,.3626837833783619829,.3626837833783619829,
  .3137066458778872873,.2223810344533744705,.1012285362903762592};
 const double* xs=order==4?x4:x8;const double* ws=order==4?w4:w8;
 Integration out;out.external.assign(m.mass.size(),0);
 // The manufactured shear produces P_zx = mu * variation / length.
 // Apply the exact signed reference traction on x=0 and x=L faces;
 // ignoring it leaves a nonzero edge residual even with exact contact.
 constexpr int faces[4][3]={{1,2,3},{0,2,3},{0,1,3},{0,1,2}};
 const double sideStress=m.mu*6e-6/.02;
 for(const Tet& tet:m.tets)for(const auto& face:faces){
  int ids[3]={tet.id[face[0]],tet.id[face[1]],tet.id[face[2]]};
  const V a=m.initial.x[ids[0]],b=m.initial.x[ids[1]],c=m.initial.x[ids[2]];
  bool left=std::abs(a.x)<1e-12&&std::abs(b.x)<1e-12&&std::abs(c.x)<1e-12;
  bool right=std::abs(a.x-.02)<1e-12&&std::abs(b.x-.02)<1e-12&&std::abs(c.x-.02)<1e-12;
  if(left||right){
   const double area=std::abs((b.y-a.y)*(c.z-a.z)-(b.z-a.z)*(c.y-a.y))/2;
   const double nodal=(right?sideStress:-sideStress)*area/3;
   for(int j:ids)out.external[j]+=nodal;
  }
 }
 for(const auto& tri:m.surface)for(int i=0;i<order;++i)for(int j=0;j<order;++j){
  const double u=(xs[i]+1)/2,v=(xs[j]+1)/2;
  const double b[3]={1-u,u*(1-v),u*v};
  double x=0,y=0,z=0;
  for(int q=0;q<3;++q){x+=b[q]*s.x[tri[q]].x;y+=b[q]*s.x[tri[q]].y;z+=b[q]*s.x[tri[q]].z;}
  const double gap2=z*z,offset=gap2-1e-8;
  // Independent evaluation of -d/dg [-k(g²-d²)² log(g²/d²)].
  const double pressure=z>=1e-4?0:2*z*1e16*(2*offset*std::log(gap2/1e-8)+offset*offset/gap2);
  const double weight=2*m.triangleArea*u*ws[i]*ws[j]/4;
  const double traction=-pzz-pressure;
  for(int q=0;q<3;++q)out.external[tri[q]]+=weight*b[q]*traction;
  out.normal+=weight*pressure;out.mx+=weight*pressure*x;out.my+=weight*pressure*y;
 }
 return out;
}
int main(int argc,char**argv){
 try{
  if(argc!=3)throw std::runtime_error("usage: static_nonuniform n nodal|gaussian");
  int n=std::atoi(argv[1]);std::string rule=argv[2];
  if(n<1||n>32||(rule!="nodal"&&rule!="gaussian"))throw std::runtime_error("invalid arguments");
  Model m(n,rule=="gaussian");State s=m.initial;
  const double g0=9.6e-5,variation=6e-6,top=.02005,stretch=(top-g0)/.02;
  const double pzz=m.mu*stretch+(m.lambda*std::log(stretch)-m.mu)/stretch;
  for(auto& p:s.x)p.z=g0+variation*(p.x/.02-.5)+(p.z-.0002)*stretch;
  Integration q4=externalLoad(m,s,pzz,4),q8=externalLoad(m,s,pzz,8);
  Change r=m.rhs(s);auto cf=m.contactForce(s);
  double maxResidual=0,totalContact=0,mx=0,my=0,externalDifference=0;
  for(size_t i=0;i<s.x.size();++i){
   double fz=m.mass[i]*(r.v[i].z+9.81)+q8.external[i];
   if(std::abs(m.initial.x[i].z-.0202)>1e-12)maxResidual=std::max(maxResidual,std::abs(fz));
   totalContact+=cf[i];mx+=cf[i]*s.x[i].x;my+=cf[i]*s.x[i].y;
   externalDifference=std::max(externalDifference,std::abs(q4.external[i]-q8.external[i]));
  }
  const double scale=q8.normal/(n*n);
  const double local=maxResidual/scale,externalError=externalDifference/scale;
  const double forceError=std::abs(totalContact-q8.normal)/q8.normal;
  const double momentError=std::max(std::abs(mx-q8.mx),std::abs(my-q8.my))/(q8.normal*.02);
  std::cout<<std::setprecision(17)<<"{\"n\":"<<n<<",\"quadrature\":\""<<rule
   <<"\",\"gap_min_m\":"<<g0-variation/2<<",\"gap_max_m\":"<<g0+variation/2
   <<",\"stretch\":"<<stretch<<",\"reference_contact_force_N\":"<<q8.normal
   <<",\"max_local_residual_relative\":"<<local<<",\"integrated_force_relative\":"<<forceError
   <<",\"contact_moment_relative\":"<<momentError<<",\"external_4_vs_8_relative\":"<<externalError<<"}\n";
  if(externalError>1e-7)throw std::runtime_error("manufactured external integration not converged");
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}

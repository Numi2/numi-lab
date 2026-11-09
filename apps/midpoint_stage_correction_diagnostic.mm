// Offline same-equations feasibility probe. No authoritative state is modified.
#define main numi_plane_contact_diagnostic_main
#include "plane_contact_probe.mm"
#undef main
#include <fstream>

static float at(const nm_float4& v,int j){return j==0?v.x:j==1?v.y:v.z;}
static void put(nm_float4& v,int j,float x){if(j==0)v.x=x;else if(j==1)v.y=x;else v.z=x;}
template<class T> static void readExact(std::ifstream& s,std::vector<T>& v){
 s.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(T));require(bool(s),"short input");
}
int main(int argc,char**argv){try{
 require(argc==4,"usage: CAPTURE WORKING_NODES CORRECTION");
 constexpr double requestedH=.00000390625;const double h=float(requestedH);
 const auto world=model(8,requestedH,.0002,true,false,true,0.,1.);const size_t n=world.dispatch.femNodeCount;
 std::ifstream input(argv[1],std::ios::binary);require(bool(input),"capture unavailable");
 uint32_t header[4];input.read(reinterpret_cast<char*>(header),sizeof(header));
 require(header[0]==0x4e4d4d32u && header[1]==n && header[2]==sizeof(NMFEMNodeStateGPU) && header[3]==sizeof(nm_float4),"capture layout mismatch");
 numi::matter::RuntimeStateSnapshot old;old.femNodes.resize(n);old.femMidpointVelocityLow.resize(n);
 std::vector<NMFEMNodeStateGPU> stage(n),endpoint(n);std::vector<nm_float4> residual(n),endpointLow(n);
 readExact(input,old.femNodes);readExact(input,stage);readExact(input,endpoint);readExact(input,residual);readExact(input,old.femMidpointVelocityLow);readExact(input,endpointLow);
 require(input.peek()==std::ifstream::traits_type::eof(),"extra capture bytes");
 std::ifstream a(argv[2],std::ios::binary),b(argv[3],std::ios::binary);require(bool(a)&&bool(b),"trace input unavailable");
 std::vector<NMFEMNodeStateGPU> working(n);std::vector<nm_float4> correction(2*n);readExact(a,working);readExact(b,correction);
 require(a.peek()==std::ifstream::traits_type::eof()&&b.peek()==std::ifstream::traits_type::eof(),"extra trace bytes");
 std::cout<<std::setprecision(17)<<"{\"kind\":\"offline_unrounded_stage_correction_not_a_native_step\",\"results\":[";
 const double factors[4]={0.,.5,1.,2.};std::vector<std::array<double,3>> residuals[4];
 for(int trial=0;trial<4;++trial){
  auto next=old;next.contactSamples.clear();double minGap=1;
  for(size_t i=0;i<n;++i){auto& q=next.femNodes[i];const auto& p=old.femNodes[i];
   for(int j=0;j<3;++j){
    const double v0=physicalVelocity(old,i,j),vm=double(at(working[i].velocityAndInverseMass,j))+factors[trial]*double(at(correction[i],j));
    const double v1=2*vm-v0;const float vh=float(v1),vl=float(v1-double(vh));
    put(q.velocityAndInverseMass,j,vh);put(next.femMidpointVelocityLow[i],j,vl);
    const double x0=double(at(p.referenceDisplacementAndMode,j))+at(p.referenceDisplacementLow,j);
    const double u=x0+h*vm;const float uh=float(u),ul=float(u-double(uh));
    put(q.referenceDisplacementAndMode,j,uh);put(q.referenceDisplacementLow,j,ul);
    put(q.positionAndMass,j,float(double(at(p.restAndFixed,j))+uh+ul));
   }
   minGap=std::min(minGap,double(q.restAndFixed.z)+q.referenceDisplacementAndMode.z+q.referenceDisplacementLow.z);
  }
  require(minGap>0,"offline candidate lacks clearance");
  const auto r=fixtureResidual(world,old,next,requestedH,true,nullptr,false,&residuals[trial]);
  std::cout<<(trial?",":"")<<"{\"name\":\""<<factors[trial]<<"\",\"residual_norm_Ns\":"<<r.l2<<",\"local_scaled_max\":"<<r.localWithoutFloor<<",\"kinematic_defect_m\":"<<r.kinematic<<",\"minimum_gap_m\":"<<minGap<<"}";
 }
 double nonlinearHalf=0,nonlinearFull=0;
 for(size_t i=0;i<n;++i)for(int j=0;j<3;++j){
  const double a=residuals[2][i][j]-2*residuals[1][i][j]+residuals[0][i][j];
  const double b=residuals[3][i][j]-2*residuals[2][i][j]+residuals[0][i][j];
  nonlinearHalf+=a*a;nonlinearFull+=b*b;
 }
 std::cout<<"],\"finite_difference_curvature_half_Ns\":"<<std::sqrt(nonlinearHalf)
   <<",\"finite_difference_curvature_full_Ns\":"<<std::sqrt(nonlinearFull)<<"}\n";return 0;
 }catch(const std::exception&e){std::cerr<<"midpoint_stage_correction_diagnostic: "<<e.what()<<"\n";return 1;}}

// Independent CPU exploration of the captured full-step equations.
#define main numi_plane_contact_diagnostic_main
#include "plane_contact_probe.mm"
#undef main
#include <fstream>
#include <set>

static float component(const nm_float4& v, int j) {return j==0?v.x:j==1?v.y:v.z;}
static void component(nm_float4& v, int j, float x) {if(j==0)v.x=x;else if(j==1)v.y=x;else v.z=x;}
int main(int argc,char** argv) {try {
    require(argc==2 || (argc==3 && std::string(argv[2])=="--stage"),"usage: midpoint_rounding_iteration CAPTURE_BINARY [--stage]");
    const bool stageOnly=argc==3;
    constexpr double requestedH=.00000390625;
    const double h=float(requestedH);
    const auto world=model(8,requestedH,.0002,true,false,true,0.,1.);
    const double cutoff=double(float(1.3721466183332781e-13));
    const double clearance=double(world.mixedSolver.contactAcceptance.x)*world.dispatch.numericalLimits.x;
    const size_t n=world.dispatch.femNodeCount;
    std::ifstream in(argv[1],std::ios::binary);require(bool(in),"capture unavailable");
    std::uint32_t header[4]{};in.read(reinterpret_cast<char*>(header),sizeof(header));
    require(header[0]==0x4e4d4d31u && header[1]==n && header[2]==sizeof(NMFEMNodeStateGPU) && header[3]==sizeof(nm_float4),"invalid capture layout");
    numi::matter::RuntimeStateSnapshot before,after;
    before.femNodes.resize(n);after.femNodes.resize(n);
    std::vector<NMFEMNodeStateGPU> stage(n);std::vector<nm_float4> stageResidual(n);
    const size_t nodeBytes=n*sizeof(NMFEMNodeStateGPU);
    in.read(reinterpret_cast<char*>(before.femNodes.data()),nodeBytes);
    in.read(reinterpret_cast<char*>(stage.data()),nodeBytes);
    in.read(reinterpret_cast<char*>(after.femNodes.data()),nodeBytes);
    in.read(reinterpret_cast<char*>(stageResidual.data()),n*sizeof(nm_float4));
    require(bool(in) && in.peek()==std::ifstream::traits_type::eof(),"invalid capture bytes");
    std::set<std::string> seen; std::vector<std::array<float,3>> proposal;
    std::cout<<std::setprecision(17)<<"{\"scope\":\"independent_cpu_not_an_accepted_step\",\"stage_reflection_constrained\":"<<(stageOnly?"true":"false")<<",\"cutoff_Ns\":"<<cutoff<<",\"clearance_m\":"<<clearance<<",\"iterations\":[";
    for(int iter=0;iter<=12;++iter) {
        const auto result=fixtureResidual(world,before,after,requestedH,true,&proposal,stageOnly);
        double minGap=1.;unsigned unreachable=0;
        for(size_t i=0;i<n;++i){
            const auto& a=before.femNodes[i];const auto& b=after.femNodes[i];
            minGap=std::min(minGap,double(b.restAndFixed.z)+b.referenceDisplacementAndMode.z+b.referenceDisplacementLow.z);
            for(int j=0;j<3;++j){
                const float old=component(a.velocityAndInverseMass,j), end=component(b.velocityAndInverseMass,j);
                const float center=float((double(old)+double(end))*.5);
                bool reachable=false;
                for(float m: {std::nextafter(center,-INFINITY),center,std::nextafter(center,INFINITY)})
                    reachable |= float(2.f*m-old)==end;
                unreachable+=!reachable;
            }
        }
        const std::string digest=physicalDigest(after);const bool repeated=!seen.insert(digest).second;
        std::cout<<(iter?",":"")<<"{\"iteration\":"<<iter<<",\"norm_Ns\":"<<result.l2
          <<",\"kinematic_defect_m\":"<<result.kinematic<<",\"minimum_gap_m\":"<<minGap
          <<",\"not_reproduced_by_nearest_three_stage_floats\":"<<unreachable
          <<",\"endpoint_sha256\":\""<<digest<<"\",\"repeated\":"<<(repeated?"true":"false")<<"}";
        if(repeated || iter==12)break;
        for(size_t i=0;i<n;++i){
            const auto& a=before.femNodes[i];auto& b=after.femNodes[i];
            for(int j=0;j<3;++j){
                const double oldVelocity=component(a.velocityAndInverseMass,j);
                const float newVelocity=proposal[i][j];
                const double oldDisplacement=double(component(a.referenceDisplacementAndMode,j))+component(a.referenceDisplacementLow,j);
                const double newDisplacement=oldDisplacement+h*.5*(oldVelocity+newVelocity);
                const float hi=float(newDisplacement),lo=float(newDisplacement-double(hi));
                component(b.referenceDisplacementAndMode,j,hi);component(b.referenceDisplacementLow,j,lo);
                component(b.positionAndMass,j,float(double(component(a.restAndFixed,j))+hi+lo));
                component(b.velocityAndInverseMass,j,newVelocity);
                component(b.deltaVelocity,j,float(double(newVelocity)-oldVelocity));
            }
        }
    }
    std::cout<<"]}\n"; return 0;
 }catch(const std::exception& e){std::cerr<<"midpoint_rounding_iteration: "<<e.what()<<"\n";return 1;}}

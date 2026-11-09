// Offline FP64 analysis of a retained midpoint certificate capture.
#define main numi_plane_contact_diagnostic_main
#include "plane_contact_probe.mm"
#undef main
#include <fstream>

int main(int argc,char** argv) {
 try {
    require(argc==2,"usage: midpoint_capture_analysis CAPTURE_BINARY");
    constexpr double h=.00000390625;
    const auto world=model(8,h,.0002,true,false,true,0.,1.);
    std::ifstream in(argv[1],std::ios::binary);
    require(bool(in),"capture unavailable");
    std::uint32_t header[4]{};in.read(reinterpret_cast<char*>(header),sizeof(header));
    const size_t n=world.dispatch.femNodeCount;
    require(header[0]==0x4e4d4d31u && header[1]==n &&
        header[2]==sizeof(NMFEMNodeStateGPU) && header[3]==sizeof(nm_float4),
        "capture layout does not match fixture");
    numi::matter::RuntimeStateSnapshot before,after;
    before.femNodes.resize(n);after.femNodes.resize(n);
    std::vector<NMFEMNodeStateGPU> middle(n);
    std::vector<nm_float4> residual(n);
    const size_t nodeBytes=n*sizeof(NMFEMNodeStateGPU),residualBytes=n*sizeof(nm_float4);
    in.read(reinterpret_cast<char*>(before.femNodes.data()),nodeBytes);
    in.read(reinterpret_cast<char*>(middle.data()),nodeBytes);
    in.read(reinterpret_cast<char*>(after.femNodes.data()),nodeBytes);
    in.read(reinterpret_cast<char*>(residual.data()),residualBytes);
    require(bool(in) && in.peek()==std::ifstream::traits_type::eof(),"truncated or trailing capture bytes");
    const auto physical=fixtureResidual(world,before,after,h,true);
    double stage2=0.,round2=0.,sum2=0.,sumNet[3]={},stageNet[3]={},roundNet[3]={};
    double maxVdiff=0.,minMass=1e30,maxMass=0.;
    for(size_t i=0;i<n;++i) {
        const auto& a=before.femNodes[i];const auto& m=middle[i];const auto& e=after.femNodes[i];
        const double av[3]={a.velocityAndInverseMass.x,a.velocityAndInverseMass.y,a.velocityAndInverseMass.z};
        const double mv[3]={m.velocityAndInverseMass.x,m.velocityAndInverseMass.y,m.velocityAndInverseMass.z};
        const double ev[3]={e.velocityAndInverseMass.x,e.velocityAndInverseMass.y,e.velocityAndInverseMass.z};
        const double rv[3]={residual[i].x,residual[i].y,residual[i].z};
        minMass=std::min(minMass,double(a.positionAndMass.w));maxMass=std::max(maxMass,double(a.positionAndMass.w));
        for(int j=0;j<3;++j){
            const double stage=2*rv[j];
            const double rounding=double(a.positionAndMass.w)*(2*(mv[j]-av[j])-(ev[j]-av[j]));
            const double total=stage+rounding;
            stage2+=stage*stage;round2+=rounding*rounding;sum2+=total*total;
            stageNet[j]+=stage;roundNet[j]+=rounding;sumNet[j]+=total;
            maxVdiff=std::max(maxVdiff,std::abs(mv[j]-av[j]));
        }
    }
    require(maxVdiff>0,"captured midpoint is identical to accepted state");
    std::cout<<std::setprecision(17)<<"{\"nodes\":"<<n<<",\"capture_midpoint_changed\":true"
        <<",\"minimum_mass_kg\":"<<minMass<<",\"maximum_mass_kg\":"<<maxMass
        <<",\"max_midpoint_velocity_change_mps\":"<<maxVdiff
        <<",\"twice_stage_norm_Ns\":"<<std::sqrt(stage2)
        <<",\"reflection_rounding_norm_Ns\":"<<std::sqrt(round2)
        <<",\"reconstructed_norm_Ns\":"<<std::sqrt(sum2)
        <<",\"reconstructed_net_Ns\":["<<sumNet[0]<<","<<sumNet[1]<<","<<sumNet[2]<<"]"
        <<",\"stage_net_Ns\":["<<stageNet[0]<<","<<stageNet[1]<<","<<stageNet[2]<<"]"
        <<",\"rounding_net_Ns\":["<<roundNet[0]<<","<<roundNet[1]<<","<<roundNet[2]<<"]"
        <<",\"independent_norm_Ns\":"<<physical.l2
        <<",\"independent_max_node_Ns\":"<<physical.maxNode
        <<",\"independent_local_max_with_1e_10_floor\":"<<physical.local
        <<",\"independent_local_max_without_floor\":"<<physical.localWithoutFloor
        <<",\"independent_net_Ns\":["<<physical.net[0]<<","<<physical.net[1]<<","<<physical.net[2]<<"]"
        <<",\"independent_kinematic_defect_m\":"<<physical.kinematic<<"}\n";
    return 0;
 }catch(const std::exception& e){std::cerr<<"midpoint_capture_analysis: "<<e.what()<<"\n";return 1;}
}

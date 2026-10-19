#include "NumiHumanWholeBodyMassReference.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace metalrobo;
using namespace metalrobo::human;
using namespace metalrobo::human::whole_body_mass_reference_detail;

namespace {
void check(bool value, const std::string& why) {
    if (!value) throw std::runtime_error(why);
}
bool near(double a, double b, double absTol = 2e-7) {
    return std::abs(a - b) <= absTol;
}
EngineModel twoBodies() {
    EngineModel m = makeFreeSphereEngineModel();
    auto planar = m.bodies[1];
    planar.inertiaRow0={.1f,0,0,0}; planar.inertiaRow1={0,.1f,0,0}; planar.inertiaRow2={0,0,.2f,0};
    planar.inverseInertiaRow0={10,0,0,0}; planar.inverseInertiaRow1={0,10,0,0}; planar.inverseInertiaRow2={0,0,5,0};
    m.bodies[1]=planar;
    auto secondBody=planar;
    secondBody.articulationIndex=1u;
    secondBody.massAndInverseMass={2,.5f,0,0};
    secondBody.inertiaRow0={.4f,0,0,0}; secondBody.inertiaRow1={0,.4f,0,0}; secondBody.inertiaRow2={0,0,.4f,0};
    secondBody.inverseInertiaRow0={2.5f,0,0,0}; secondBody.inverseInertiaRow1={0,2.5f,0,0}; secondBody.inverseInertiaRow2={0,0,2.5f,0};
    m.bodies.push_back(secondBody);
    MRArticulationGPU a{}; a.rootBody=2; a.rootType=MR_ROOT_FLOATING; a.firstBody=2; a.bodyCount=1;
    a.firstJoint=0; a.jointCount=0; a.qOffset=7; a.nq=7; a.vOffset=6; a.nv=6;
    m.articulations.push_back(a);
    for(std::uint32_t i=0;i<6;++i){auto d=m.dofs[i];d.articulationIndex=1;d.qIndex=i<3?7+i:MR_INVALID_INDEX;d.vIndex=6+i;d.localDof=i;m.dofs.push_back(d);}
    m.defaultQ.insert(m.defaultQ.end(),{2,0,0,0,0,0,1}); m.defaultV.insert(m.defaultV.end(),6,0);
    m.world.bodyCount=static_cast<std::uint32_t>(m.bodies.size());
    m.world.articulationCount=static_cast<std::uint32_t>(m.articulations.size());
    m.world.nq=static_cast<std::uint32_t>(m.defaultQ.size());
    m.world.nv=static_cast<std::uint32_t>(m.defaultV.size());
    std::string why;check(m.valid(&why),"two-body fixture invalid: "+why);return m;
}
std::vector<NumiHumanTissueMassNode> nodes(){
    return {{0,1,.1,{.005,-.002,.004}},
            {1,2,.7,{.010,0,0}}, {2,2,.6,{-.006,.003,0}},
            {3,2,.5,{0,-.004,.005}}, {4,2,.4,{0,0,-.007}}};
}
}

int main(){
try{
    const auto source=twoBodies();const auto input=nodes();
    const auto result=prepareWholeBodyMassReference(source,input);
    check(result.succeeded(),"reference failed: "+result.error);
    check(result.nodes.size()==input.size()&&result.offsets.size()==source.bodies.size(),"wrong result sizes");
    check(near(result.originalMassKg,3,1e-12)&&near(result.tissueMassKg,2.3,1e-12),"wrong total mass report");
    check(near(result.residualScale,1.0-2.3/3.0,1e-12),"wrong global residual scale");
    check(input[1].massKg+input[2].massKg+input[3].massKg+input[4].massKg>source.bodies[2].massAndInverseMass.x,
          "fixture must give one donor more tissue mass than its original rigid mass");
    check(near(result.model.bodies[1].massAndInverseMass.x,result.residualScale+0.1,2e-7)&&
          near(result.model.bodies[2].massAndInverseMass.x,result.residualScale*2+2.2,2e-7),
          "composed reference donor masses are wrong");
    check(result.maximumRelativeInertiaCorrection>5e-6&&result.maximumRelativeInertiaCorrection<1e-5,
          "planar float32 eigen-floor correction missing or too large");

    double refMass=0;
    for(const auto& b:result.model.bodies)if(b.motionType==MR_MOTION_DYNAMIC)refMass+=b.massAndInverseMass.x;
    check(near(refMass,result.originalMassKg,2e-7),"reference rigid mass plus cooked tissue mass not conserved");

    for(std::size_t i=0;i<input.size();++i)for(std::size_t k=0;k<3;++k)
        check(near(result.nodes[i].localPosition[k],input[i].localPosition[k]-result.offsets[input[i].donorBody][k],1e-12),
              "node coordinates not shifted into reference COM frame");

    for(const std::uint32_t donor:{1u,2u}){
        M origS{};double correction=0;std::string why;
        check(floorSource(second(inertia(source.bodies[donor])),origS,correction,why),"fixture source moment invalid: "+why);
        double tm=0;V first{};M Q{};
        for(const auto& n:input)if(n.donorBody==donor){
            tm+=n.massKg;for(size_t r=0;r<3;++r){first[r]+=n.massKg*n.localPosition[r];
                for(size_t c=0;c<3;++c)Q[3*r+c]+=n.massKg*n.localPosition[r]*n.localPosition[c];}
        }
        const auto& rb=result.model.bodies[donor];const double Mref=rb.massAndInverseMass.x;const auto c=result.offsets[donor];
        const M Sref=second(inertia(rb));const M reconstructed=add(Sref,mul(outer(c,c),Mref));
        const M expected=add(mul(origS,result.residualScale),Q);
        for(size_t k=0;k<3;++k)check(near(c[k]*Mref,first[k],2e-7),"composed reference first moment does not close");
        for(size_t k=0;k<9;++k)check(near(reconstructed[k],expected[k],2e-7),"composed reference second moment does not close");
        check(near(Mref,result.residualScale*source.bodies[donor].massAndInverseMass.x+tm,2e-7),
              "per-donor composed mass does not close");
    }

    const auto stock=compileNumiHumanTissueMassPartition(result.model.bodies,result.nodes);
    check(stock.succeeded(),"stock compiler rejected reference: "+stock.error);
    EngineModel finalModel;std::string error;
    check(rebaseNumiHumanTissueMassPartition(result.model,stock.partitions,finalModel,error),
          "stock rebase rejected reference: "+error);
    check(finalModel.valid(&error),"stock-rebased model invalid: "+error);
    double finalMass=0;for(const auto& b:finalModel.bodies)if(b.motionType==MR_MOTION_DYNAMIC)finalMass+=b.massAndInverseMass.x;
    check(near(finalMass+result.tissueMassKg,result.originalMassKg,2e-7),"stock partition total mass not conserved");
    for(const auto&p:stock.partitions){
        for(size_t k=0;k<3;++k)check(std::abs(result.offsets[p.donorBody][k]+p.remainingCOMOffsetM[k])<2e-7,
                                     "reference and stock COM offsets do not cancel");
        check(p.packedMomentRelativeError<16*std::numeric_limits<float>::epsilon(),"stock compiler moment closure exceeds Float32 bound");
    }

    auto grosslyInvalid=source;
    grosslyInvalid.bodies[2].inertiaRow0={1,0,0,0};grosslyInvalid.bodies[2].inertiaRow1={0,.001f,0,0};grosslyInvalid.bodies[2].inertiaRow2={0,0,.001f,0};
    grosslyInvalid.bodies[2].inverseInertiaRow0={1,0,0,0};grosslyInvalid.bodies[2].inverseInertiaRow1={0,1000,0,0};grosslyInvalid.bodies[2].inverseInertiaRow2={0,0,1000,0};
    check(grosslyInvalid.valid(&error),"gross-negative test fixture must pass ordinary inertia-only EngineModel validation: "+error);
    const auto rejected=prepareWholeBodyMassReference(grosslyInvalid,input);
    check(!rejected.succeeded()&&rejected.error.find("materially negative")!=std::string::npos,
          "grossly negative source central moment was not rejected");

    std::cout<<"whole_body_mass_reference=PASS mass_and_first_second_moments=closed planar_floor=reported stock_compile_rebase=PASS net_offsets=zero donor_tissue_exceeds_original=accepted gross_negative=REJECTED\n";
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}

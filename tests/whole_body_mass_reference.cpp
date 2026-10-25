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

    const std::vector<NumiHumanTissueMassNode> splitFrames{
        {15u,0u,.6,{.03,.01,-.02}}, {19u,0u,.4,{-.04,.02,.01}}};
    const std::vector<NumiHumanTissueMassContribution> splitSources{
        {15u,1u,.2,{.010,.002,0}}, {15u,2u,.4,{-.006,.003,0}},
        {19u,1u,.2,{.001,-.004,.005}}, {19u,2u,.2,{.003,0,-.007}}};
    const auto splitReference=prepareWholeBodyMassReference(source,splitFrames,splitSources);
    check(splitReference.succeeded(),"split-source whole-body reference failed: "+splitReference.error);
    check(splitReference.nodes.size()==2&&splitReference.contributions.size()==4,
          "split-source reference lost unique frame rows or source contributions");
    check(splitReference.nodes[0].donorBody==0&&splitReference.nodes[0].localPosition==splitFrames[0].localPosition,
          "static registered frame owner or anchor was rewritten by source-mass rebase");
    for(std::size_t i=0;i<splitSources.size();++i) {
        const auto donor=splitSources[i].donorBody;
        for(std::size_t axis=0;axis<3;++axis)
            check(near(splitReference.contributions[i].localPosition[axis],
                       splitSources[i].localPosition[axis]-splitReference.offsets[donor][axis],1e-12),
                  "source-mass contribution did not rebase in its own donor frame");
    }
    const auto splitCompiled=compileNumiHumanTissueMassPartition(
        splitReference.model.bodies,splitReference.nodes,splitReference.contributions);
    check(splitCompiled.succeeded(),"split-source compiler rejected rebased reference: "+splitCompiled.error);
    for(const auto& part:splitCompiled.partitions) {
        for(std::size_t axis=0;axis<3;++axis)
            check(std::abs(splitReference.offsets[part.donorBody][axis]+part.remainingCOMOffsetM[axis])<2e-7,
                  "split-source reference and stock COM offsets do not cancel");
        double mass=0.0;V first{};M secondMoment{};
        for(const auto& contribution:splitReference.contributions)if(contribution.donorBody==part.donorBody) {
            mass+=contribution.massKg;
            for(std::size_t r=0;r<3;++r) {
                first[r]+=contribution.massKg*contribution.localPosition[r];
                for(std::size_t c=0;c<3;++c)
                    secondMoment[3*r+c]+=contribution.massKg*
                        contribution.localPosition[r]*contribution.localPosition[c];
            }
        }
        check(near(part.tissueMassKg,mass,1e-12),
              "split-source donor debit does not equal its contribution mass");
        for(std::size_t r=0;r<3;++r) {
            check(near(part.tissueFirstMomentKgM[r],first[r],1e-12),
                  "split-source donor first moment lost a source-local contribution");
            for(std::size_t c=0;c<3;++c)
                check(near(part.tissueSecondMomentKgM2[3*r+c],secondMoment[3*r+c],1e-12),
                      "split-source donor second moment lost a source-local contribution");
        }
    }
    auto badSplitSources=splitSources;badSplitSources[0].massKg+=.01;
    check(!prepareWholeBodyMassReference(source,splitFrames,badSplitSources).succeeded(),
          "whole-body reference accepted a source ledger that does not close node mass");

    std::vector<NumiHumanTissueMassContribution> oneToOneSources;
    for(const auto& node:result.nodes)
        oneToOneSources.push_back({node.nodeIndex,node.donorBody,node.massKg,node.localPosition});
    const auto legacyCompile=compileNumiHumanTissueMassPartition(result.model.bodies,result.nodes);
    const auto explicitLegacyEquivalent=compileNumiHumanTissueMassPartition(
        result.model.bodies,result.nodes,oneToOneSources);
    check(legacyCompile.succeeded()&&explicitLegacyEquivalent.succeeded(),
          "legacy-equivalent explicit mass partition failed: legacy="+legacyCompile.error+
          " explicit="+explicitLegacyEquivalent.error);
    check(legacyCompile.partitions.size()==explicitLegacyEquivalent.partitions.size(),
          "legacy and explicit one-to-one paths return different donor partitions");
    for(std::size_t i=0;i<legacyCompile.partitions.size();++i) {
        const auto& a=legacyCompile.partitions[i];const auto& b=explicitLegacyEquivalent.partitions[i];
        check(a.donorBody==b.donorBody&&a.nodeCount==b.nodeCount&&near(a.tissueMassKg,b.tissueMassKg,1e-14),
              "legacy and explicit one-to-one partition identity/mass differ");
        for(std::size_t k=0;k<3;++k)check(near(a.tissueFirstMomentKgM[k],b.tissueFirstMomentKgM[k],1e-14)&&
            near(a.remainingCOMOffsetM[k],b.remainingCOMOffsetM[k],1e-14),
            "legacy and explicit one-to-one first moment/COM differ");
        for(std::size_t k=0;k<9;++k)check(near(a.tissueSecondMomentKgM2[k],b.tissueSecondMomentKgM2[k],1e-14),
            "legacy and explicit one-to-one second moment differ");
        check(near(a.remainingBody.massAndInverseMass.x,b.remainingBody.massAndInverseMass.x,1e-7f),
              "legacy and explicit one-to-one cooked remaining mass differs");
    }
    auto expectExplicitReject=[&](auto frames,auto contributions,const std::string& label) {
        check(!compileNumiHumanTissueMassPartition(source.bodies,frames,contributions).succeeded(),
              "explicit compiler accepted "+label);
        check(!prepareWholeBodyMassReference(source,frames,contributions).succeeded(),
              "whole-body rebase accepted "+label);
    };
    auto duplicateFrames=splitFrames;duplicateFrames.push_back(splitFrames.front());
    expectExplicitReject(duplicateFrames,splitSources,"duplicate physical frame binding");
    auto badFrame=splitFrames;badFrame[0].donorBody=static_cast<std::uint32_t>(source.bodies.size());
    expectExplicitReject(badFrame,splitSources,"out-of-range node-frame owner");
    badFrame=splitFrames;badFrame[0].massKg=0.0;
    expectExplicitReject(badFrame,splitSources,"nonpositive physical frame mass");
    badFrame=splitFrames;badFrame[0].localPosition[2]=std::numeric_limits<double>::infinity();
    expectExplicitReject(badFrame,splitSources,"nonfinite node-frame anchor");
    auto missingFrameSource=splitSources;missingFrameSource[0].nodeIndex=999u;
    expectExplicitReject(splitFrames,missingFrameSource,"contribution without a registered frame row");
    auto missingNodeContribution=splitSources;missingNodeContribution.pop_back();
    expectExplicitReject(splitFrames,missingNodeContribution,"missing contribution mass for a physical node");
    auto badDonorSources=splitSources;badDonorSources[0].donorBody=static_cast<std::uint32_t>(source.bodies.size());
    expectExplicitReject(splitFrames,badDonorSources,"out-of-range source donor");
    auto staticDonorSources=splitSources;staticDonorSources[0].donorBody=0u;
    expectExplicitReject(splitFrames,staticDonorSources,"static source mass donor");
    auto negativeSources=splitSources;negativeSources[0].massKg=-.2;
    expectExplicitReject(splitFrames,negativeSources,"negative source contribution");
    auto nanSources=splitSources;nanSources[0].localPosition[1]=std::numeric_limits<double>::quiet_NaN();
    expectExplicitReject(splitFrames,nanSources,"nonfinite source-local position");

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

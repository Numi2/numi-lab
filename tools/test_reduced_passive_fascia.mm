#include "NumiHumanReducedPassiveFascia.hpp"
#include "numi/matter/numi_human.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace numi::human::reduced_passive_fascia;

static void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    constexpr double side = 0.1;
    std::vector<Node> nodes;
    std::vector<Tetrahedron> tetrahedra;
    nodes.reserve(18u);
    tetrahedra.reserve(24u);
    constexpr std::array<Vec3, 9u> unitCubeNodes{{
        {0,0,0},{0,side,0},{0,0,side},{0,side,side},
        {side,0,0},{side,side,0},{side,0,side},{side,side,side},
        {side*0.5,side*0.5,side*0.5}}};
    constexpr std::array<std::array<std::uint32_t,3u>,12u> surfaceTriangles{{
        {{0,1,3}},{{0,3,2}}, {{4,5,7}},{{4,7,6}},
        {{0,2,6}},{{0,6,4}}, {{1,3,7}},{{1,7,5}},
        {{0,1,5}},{{0,5,4}}, {{2,3,7}},{{2,7,6}}}};
    for (std::uint32_t layer=0;layer<2u;++layer) {
        const std::uint32_t offset=layer*9u;
        for (std::uint32_t i=0;i<9u;++i) {
            Vec3 p=unitCubeNodes[i];
            p.z += double(layer)*0.25;
            const std::uint32_t flags=i<4u?1u:(i<8u?2u:0u);
            nodes.push_back({p,500u+offset+i,flags});
        }
        for (const auto& tri:surfaceTriangles)
            tetrahedra.push_back({{{offset+8u,offset+tri[0],
                                    offset+tri[1],offset+tri[2]}}});
    }
    const Material material{1000.0,1840.0,10000.0,2.36,1.0e-4};
    const Vec3 fiber{1,0,0};
    const auto ownerEvaluator=[](const FiberLaw& law,double length,double area,
                                 double stretch,double& stress)->bool {
        NMNumiHumanPassiveLigamentGPU row{};
        row.flags=NM_NUMI_HUMAN_PASSIVE_LIGAMENT_ACTIVE;
        row.material={float(law.c3Pascals),float(law.c4),
                      float(law.c5Pascals),float(law.maximumStretch)};
        row.reference={float(length),float(area),1.0f,0.0f};
        numi::matter::NumiHumanPassiveLigamentFiberEvaluation evaluation;
        if(!numi::matter::evaluateNumiHumanPassiveLigamentFiber(
                row,length*stretch,evaluation)) return false;
        stress=evaluation.fiberStressPascals;
        return std::isfinite(stress);
    };
    Result result;
    std::string error;
    check(cook(nodes,tetrahedra,material,fiber,ownerEvaluator,result,error),
          error.empty()?"valid fixture was rejected":error.c_str());
    check(result.strips.size()==8u,"expected eight strips from two four-anchor layers");
    const double expectedVolume=2.0*side*side*side;
    check(std::abs(result.sourceVolumeCubicMeters-expectedVolume)<1.0e-12,
          "tetrahedra do not recover the two source-layer volumes");
    check(std::abs(result.sourceMassKg-material.densityKgPerCubicMeter*expectedVolume)<1.0e-10,
          "source mass audit is inconsistent with source volume and density");
    check(result.distributedStripMassKg==0.0,
          "reduced force rows must not duplicate source tissue mass");
    std::set<std::uint32_t> anchors;
    double stripVolume=0.0;
    std::array<unsigned,2u> stripsPerLayer{};
    for (const auto& strip:result.strips) {
        check(strip.layer<2u,"strip has invalid layer identity");
        ++stripsPerLayer[strip.layer];
        check(anchors.insert(strip.firstNode).second&&
              anchors.insert(strip.secondNode).second,
              "source attachment endpoint is reused");
        const auto half=std::uint32_t(nodes.size()/2u);
        check(strip.firstNode/half==strip.layer&&strip.secondNode/half==strip.layer,
              "strip crosses source layers");
        check((nodes[strip.firstNode].endpointFlags&3u)==1u&&
              (nodes[strip.secondNode].endpointFlags&3u)==2u,
              "strip endpoint roles do not preserve source attachment flags");
        check(strip.firstSourceVertex==nodes[strip.firstNode].sourceVertexIndex&&
              strip.secondSourceVertex==nodes[strip.secondNode].sourceVertexIndex,
              "strip endpoint source-vertex identity changed");
        check(strip.restLengthMeters>0.0&&strip.sourceVolumeCubicMeters>0.0&&
              strip.areaSquareMeters>0.0,
              "strip geometry or tributary volume is invalid");
        check(std::abs(strip.areaSquareMeters*strip.restLengthMeters-
                       strip.sourceVolumeCubicMeters)<1.0e-12,
              "strip area does not conserve its assigned volume");
        check(std::isfinite(strip.peakNormalizedRmsError)&&
              std::isfinite(strip.peakNormalizedMaximumError)&&
              strip.peakNormalizedMaximumError<=kMaximumPeakNormalizedFitError,
              "existing passive owner curve fit exceeded its peak-normalized bound");
        stripVolume+=strip.sourceVolumeCubicMeters;
    }
    check(stripsPerLayer[0]==4u&&stripsPerLayer[1]==4u,
          "layer-specific strip counts changed");
    check(anchors.size()==16u,"all sixteen source attachment nodes must appear exactly once");
    check(std::abs(stripVolume-result.sourceVolumeCubicMeters)<1.0e-12,
          "strip tributary assignment does not conserve source tetra volume");

    const double acceptedVolume=result.sourceVolumeCubicMeters;
    const double acceptedMass=result.sourceMassKg;
    const auto acceptedFirstStrip=result.strips.front();
    auto malformed=nodes;
    malformed[0].endpointFlags=0u;
    check(!cook(malformed,tetrahedra,material,fiber,ownerEvaluator,result,error),
          "missing attachment node must fail closed");
    check(result.strips.size()==8u &&
          result.sourceVolumeCubicMeters==acceptedVolume &&
          result.sourceMassKg==acceptedMass &&
          result.strips.front().firstSourceVertex==acceptedFirstStrip.firstSourceVertex &&
          result.strips.front().areaSquareMeters==acceptedFirstStrip.areaSquareMeters,
          "failed attachment cook modified the previous output");
    auto degenerate=tetrahedra;
    degenerate[0].node[3]=degenerate[0].node[2];
    check(!cook(nodes,degenerate,material,fiber,ownerEvaluator,result,error),
          "degenerate source tetrahedron must fail closed");
    check(result.strips.size()==8u &&
          result.sourceVolumeCubicMeters==acceptedVolume &&
          result.sourceMassKg==acceptedMass,
          "failed geometry cook modified the previous output");
    std::cout<<"reduced_passive_fascia_fixture=PASS strips=8 anchors=16 "
        <<"layers=2 source_volume_m3="<<result.sourceVolumeCubicMeters
        <<" source_mass_kg="<<result.sourceMassKg
        <<" distributed_strip_mass_kg="<<result.distributedStripMassKg
        <<" max_peak_normalized_fit=";
    double maxFit=0.0;
    for(const auto& strip:result.strips)
        maxFit=std::max(maxFit,strip.peakNormalizedMaximumError);
    std::cout<<maxFit<<" actual_owner_evaluator=true malformed_inputs_fail_closed=true\n";
    return 0;
}

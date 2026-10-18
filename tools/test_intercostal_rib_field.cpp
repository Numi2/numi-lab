#include "NumiHumanRestingIntercostalRibField.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static void require(bool value,const char* message) {
    if(!value){std::cerr<<"FAIL "<<message<<"\n";std::exit(1);}
}

static std::vector<NumiHumanRestingRibFieldChannel> sideChannels(int side) {
    std::vector<NumiHumanRestingRibFieldChannel> result;
    const unsigned base=side<0?0u:12u;
    const unsigned stableBase=side<0?36u:48u;
    for(unsigned i=0;i<12;++i)result.push_back({base+i,stableBase+i,side});
    return result;
}

static double represented(const NumiHumanRestingRibFieldValue& v) {
    return double(v.firstOrdinal)*(1.0-double(v.secondaryWeight))+
        double(v.firstOrdinal+1)*double(v.secondaryWeight);
}

int main() {
    require(numiHumanUseContinuousRibField(317),
        "row 317 must use the measured continuous passive field");
    for(unsigned row:{312u,313u,314u,315u,316u})
        require(!numiHumanUseContinuousRibField(row),
            "rows 312-316 must retain their accepted rib map until separately qualified");

    const auto channels=sideChannels(1);
    const auto below=numiHumanEncodeContinuousRibOrdinal(5.0-1.0e-7,channels,1);
    const auto above=numiHumanEncodeContinuousRibOrdinal(5.0+1.0e-7,channels,1);
    require(below.firstChannel==16&&below.secondChannel==17&&below.secondaryWeight>0.999f,
        "ordinal immediately below an integer must approach the preceding pair's upper rib");
    require(above.firstChannel==17&&above.secondChannel==18&&above.secondaryWeight<1.0e-5f,
        "ordinal immediately above an integer must begin at the same shared rib channel");
    require(std::abs(represented(below)-represented(above))<3.0e-7,
        "encoded adjacent-pair transitions must be continuous at integer boundaries");

    const auto distance=[](const std::array<double,3>& p,unsigned channel) {
        return std::abs(p[0]-double(channel-12));
    };
    const std::array<double,3> sharedPoint{5.25,0,0};
    const auto componentA=numiHumanEvaluateContinuousRibOrdinal(sharedPoint,channels,1,distance);
    const auto componentB=numiHumanEvaluateContinuousRibOrdinal(sharedPoint,channels,1,distance);
    require(componentA.firstChannel==componentB.firstChannel&&
        componentA.secondChannel==componentB.secondChannel&&
        componentA.secondaryWeight==componentB.secondaryWeight,
        "coincident points across disconnected components must receive the same global spatial field");

    std::vector<NumiHumanRestingRibFieldChannel> bilateral=sideChannels(-1);
    const auto right=sideChannels(1);bilateral.insert(bilateral.end(),right.begin(),right.end());
    const auto sideDistance=[](const std::array<double,3>& p,unsigned channel) {
        const double x=channel<12?100.0+double(channel):double(channel-12);
        return std::abs(p[0]-x);
    };
    const int chosen=numiHumanChooseIntercostalRibFieldSide({{5.0,0,0},{6.0,0,0}},bilateral,sideDistance);
    require(chosen==1,"component laterality must select the nearer authenticated same-side chain");
    const auto componentAWithSide=numiHumanEvaluateContinuousRibOrdinal(
        sharedPoint,sideChannels(chosen),chosen,distance);
    const auto componentBWithSide=numiHumanEvaluateContinuousRibOrdinal(
        sharedPoint,sideChannels(chosen),chosen,distance);
    require(componentAWithSide.firstChannel==componentBWithSide.firstChannel&&
        componentAWithSide.secondChannel==componentBWithSide.secondChannel&&
        componentAWithSide.secondaryWeight==componentBWithSide.secondaryWeight,
        "one coincident point must get a component-independent field when sides are selected from typed surfaces");

    const auto ordinalDistance=[](const std::array<double,3>& p,unsigned channel) {
        const double k=double(channel-12);
        return std::exp(-0.5*k*p[0]);
    };
    const auto weighted=numiHumanEvaluateContinuousRibOrdinal({-0.2,0,0},channels,1,ordinalDistance);
    double minDistance=std::numeric_limits<double>::infinity();
    for(unsigned i=0;i<12;++i)minDistance=std::min(minDistance,ordinalDistance({-0.2,0,0},12+i));
    double den=0,num=0;
    for(unsigned i=0;i<12;++i) {
        const double d=ordinalDistance({-0.2,0,0},12+i),w=(minDistance/d)*(minDistance/d);
        den+=w;num+=double(i)*w;
    }
    require(std::abs(weighted.scalarOrdinal-num/den)<1.0e-12,
        "field scalar must equal the normalized inverse-square weighted ordinal over all twelve ribs");

    const auto exactDistance=[](const std::array<double,3>&,unsigned channel) {
        return channel==18?0.0:double(channel>18?channel-18:18-channel)+1.0;
    };
    const auto exact=numiHumanEvaluateContinuousRibOrdinal({0,0,0},channels,1,exactDistance);
    require(exact.firstChannel==18&&exact.secondChannel==19&&exact.secondaryWeight==0.0f,
        "one exact typed-rib hit must map to that rib with finite adjacent-channel encoding");

    bool conflicted=false;
    try {
        const auto ambiguous=[](const std::array<double,3>&,unsigned channel) {
            return channel==17||channel==18?0.0:1.0;
        };
        (void)numiHumanEvaluateContinuousRibOrdinal({0,0,0},channels,1,ambiguous);
    } catch(const std::invalid_argument&) { conflicted=true; }
    require(conflicted,"conflicting exact zero-distance rib hits must fail closed");

    const auto& tinyPolicies=numiHumanRow317OneSidedRibFragmentPolicies();
    require(tinyPolicies.size()==8,"one-sided carriage must be limited to the eight measured row-317 fragments");
    for(std::size_t i=0;i<tinyPolicies.size();++i) {
        const auto& policy=tinyPolicies[i];
        require(policy.fragmentOrdinal==i&&policy.lateralSide==1&&
            policy.sourceFaceOrdinals.size()==(i==3?18u:(i==5?12u:8u)),
            "tiny-fragment policy ordinals, side, or source-face ancestry changed");
        require(std::is_sorted(policy.sourceFaceOrdinals.begin(),policy.sourceFaceOrdinals.end()),
            "tiny-fragment source face ancestry must be ordered");
        require(numiHumanMatchRow317OneSidedRibFragment(policy.sourceFaceOrdinals)==&policy,
            "an exact authenticated row-317 face set must select its one-sided policy");
    }
    const auto& firstTiny=tinyPolicies.front();
    numiHumanValidateRow317OneSidedRibFragment(firstTiny,317,70351,261732,6,1,56,
        firstTiny.measuredDiameterM,firstTiny.measuredPositiveVolumeM3,
        firstTiny.measuredNearestRibDistanceM+0.5*firstTiny.measuredDiameterM,0.017);
    const auto oneRibCarriage=numiHumanEncodeContinuousRibOrdinal(
        double(firstTiny.selectedRibStableId-48u),channels,1);
    require(oneRibCarriage.firstStableId==firstTiny.selectedRibStableId&&
        oneRibCarriage.secondaryWeight==0.0f,
        "one-sided fragment carriage must encode exactly its selected typed rib");
    bool partialTinyAncestry=false;
    try {
        auto partial=firstTiny.sourceFaceOrdinals;
        partial.pop_back();
        (void)numiHumanMatchRow317OneSidedRibFragment(partial);
    } catch(const std::invalid_argument&) { partialTinyAncestry=true; }
    require(partialTinyAncestry,"partial source-face ancestry must fail closed instead of selecting a tiny-fragment exception");
    bool wrongTinyRib=false;
    try {
        numiHumanValidateRow317OneSidedRibFragment(firstTiny,317,70351,261732,6,1,57,
            firstTiny.measuredDiameterM,firstTiny.measuredPositiveVolumeM3,
            firstTiny.measuredNearestRibDistanceM+0.5*firstTiny.measuredDiameterM,0.017);
    } catch(const std::invalid_argument&) { wrongTinyRib=true; }
    require(wrongTinyRib,"a changed one-sided typed-rib identity must fail closed");
    bool oversizedTiny=false;
    try {
        numiHumanValidateRow317OneSidedRibFragment(firstTiny,317,70351,261732,6,1,56,
            firstTiny.maximumDiameterM*1.01,firstTiny.measuredPositiveVolumeM3,
            firstTiny.measuredNearestRibDistanceM,0.017);
    } catch(const std::invalid_argument&) { oversizedTiny=true; }
    require(oversizedTiny,"one-sided carriage must reject fragments beyond their measured size bound");
    bool wrongTinyAllocation=false;
    try {
        numiHumanValidateRow317OneSidedRibFragment(firstTiny,317,70351,261729,6,1,56,
            firstTiny.measuredDiameterM,firstTiny.measuredPositiveVolumeM3,
            firstTiny.measuredNearestRibDistanceM,0.017);
    } catch(const std::invalid_argument&) { wrongTinyAllocation=true; }
    require(wrongTinyAllocation,"a changed source allocation must not inherit a tiny-fragment exception");
    bool reversedTinyVolume=false;
    try {
        numiHumanValidateRow317OneSidedRibFragment(firstTiny,317,70351,261732,6,1,56,
            firstTiny.measuredDiameterM,-firstTiny.measuredPositiveVolumeM3,
            firstTiny.measuredNearestRibDistanceM,0.017);
    } catch(const std::invalid_argument&) { reversedTinyVolume=true; }
    require(reversedTinyVolume,"one-sided carriage must preserve positive signed fragment volume");
    bool nearbyRunnerUp=false;
    try {
        numiHumanValidateRow317OneSidedRibFragment(firstTiny,317,70351,261732,6,1,56,
            firstTiny.measuredDiameterM,firstTiny.measuredPositiveVolumeM3,
            firstTiny.measuredNearestRibDistanceM,0.009);
    } catch(const std::invalid_argument&) { nearbyRunnerUp=true; }
    require(nearbyRunnerUp,"one-sided carriage must reject a fragment with a nearby second typed rib");

    const double free=std::numeric_limits<double>::quiet_NaN();
    const std::vector<std::vector<std::pair<unsigned,double>>> path{
        {{1,1.0}},{{0,1.0},{2,1.0}},{{1,1.0},{3,1.0}},{{2,1.0},{4,1.0}},{{3,1.0}}};
    const auto harmonic=numiHumanSolvePositiveGraphHarmonic(path,{16.0,free,free,free,17.0});
    require(harmonic.values.size()==5&&harmonic.iterations>0&&harmonic.residualInfinity<1.0e-8&&
        harmonic.maximumPrincipleViolation==0.0,
        "positive graph harmonic solve must report bounded residual and maximum principle");
    require(std::abs(harmonic.values[1]-16.25)<1.0e-8&&
        std::abs(harmonic.values[2]-16.5)<1.0e-8&&
        std::abs(harmonic.values[3]-16.75)<1.0e-8,
        "equal positive conductances on a path must yield the linear Dirichlet field");

    const std::vector<std::vector<std::pair<unsigned,double>>> weightedPath{
        {{1,1.0}},{{0,1.0},{2,2.0}},{{1,2.0},{3,4.0}},{{2,4.0},{4,8.0}},{{3,8.0}}};
    const auto weightedHarmonic=numiHumanSolvePositiveGraphHarmonic(
        weightedPath,{0.0,free,free,free,1.0});
    require(std::abs(weightedHarmonic.values[1]-(8.0/15.0))<1.0e-8&&
        std::abs(weightedHarmonic.values[2]-(12.0/15.0))<1.0e-8&&
        std::abs(weightedHarmonic.values[3]-(14.0/15.0))<1.0e-8,
        "positive weighted graph solve must respect inverse-conductance path resistance");

    bool unanchored=false;
    try {
        (void)numiHumanSolvePositiveGraphHarmonic(
            {{{1,1.0}},{{0,1.0}},{}},{0.0,1.0,free});
    } catch(const std::invalid_argument&) { unanchored=true; }
    require(unanchored,"an unanchored graph component must fail closed");
    bool invalidConductance=false;
    try {
        (void)numiHumanSolvePositiveGraphHarmonic(
            {{{1,0.0}},{{0,0.0}}},{0.0,1.0});
    } catch(const std::invalid_argument&) { invalidConductance=true; }
    require(invalidConductance,"nonpositive conductance must fail closed");

    std::cout<<"PASS continuous ordinal fixtures and positive graph harmonic Dirichlet solve, residual, maximum principle, and fail-closed anchors\n";
}

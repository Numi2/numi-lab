#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

// Receipt-ordered typed ribs used for inferred passive visual registration.
// This field is not a force attachment or an interface exemption.
struct NumiHumanRestingRibFieldChannel {
    unsigned channel=0;
    unsigned stableId=0;
    int lateralSide=0;
};

inline bool numiHumanUseContinuousRibField(unsigned rowStableId) noexcept {
    return rowStableId==317;
}

struct NumiHumanRestingRibFieldValue {
    unsigned firstChannel=0;
    unsigned firstStableId=0;
    unsigned secondChannel=0;
    unsigned secondStableId=0;
    unsigned firstOrdinal=0;
    double scalarOrdinal=0;
    float secondaryWeight=0;
    int lateralSide=0;
};

// Exact-source row-317 fragment exceptions: these tiny closed shells have one
// measured typed-rib neighbor and no second rib within 10 mm. Their passive
// display map is constant one-rib carriage, never a force owner or exemption.
struct NumiHumanOneSidedRibFragmentPolicy {
    unsigned fragmentOrdinal=0;
    std::vector<unsigned> sourceFaceOrdinals;
    unsigned sourceVertexCount=0;
    int lateralSide=0;
    unsigned selectedRibStableId=0;
    double measuredDiameterM=0;
    double measuredPositiveVolumeM3=0;
    double measuredNearestRibDistanceM=0;
    double maximumDiameterM=0;
    double maximumNearestRibDistanceM=0;
    double minimumRunnerUpRibDistanceM=0.010;
};

inline const std::array<NumiHumanOneSidedRibFragmentPolicy,8>&
numiHumanRow317OneSidedRibFragmentPolicies() {
    static const std::array<NumiHumanOneSidedRibFragmentPolicy,8> policies{{
        {0,{9091,9092,9093,9094,9095,9096,9097,9098},6,1,56,0.00038333342236964422,1.7093401102563967e-12,0.0004317929240570001,0.000489000,0.001010},
        {1,{39293,39294,39297,39298,39299,39300,39301,39302},6,1,54,0.0003635895360557221,1.4849606919936043e-12,0.0025577477595251524,0.000465000,0.003020},
        {2,{77041,77044,77045,77046,77047,77050,77051,77053},6,1,50,0.00050277665459025887,4.5007125163596211e-12,0.000026812308713988096,0.000638500,0.000680},
        {3,{77048,77049,77052,77054,77055,77056,77057,77058,77059,77061,77063,77064,77065,77066,77069,77070,77071,77092},11,1,50,0.001037298854339965,4.0062498797175097e-12,0.00002565124826860165,0.001306625,0.001347},
        {4,{77060,77062,77067,77068,77072,77073,77074,77075},6,1,50,0.00063964981463565467,3.3046750078425203e-12,0.00005639464937166404,0.000809562,0.000881},
        {5,{79317,79318,79319,79320,79321,79322,79323,79324,79325,79326,79329,79330},8,1,50,0.00071748251464049371,1.1651323268722289e-11,0.00098868362892979775,0.000906853,0.001913},
        {6,{80181,80182,80183,80184,80185,80186,80187,80188},6,1,50,0.00016916412633941584,1.5494462522303623e-13,0.0013312665368981236,0.000221455,0.001568},
        {7,{86083,86084,86085,86086,86087,86088,86089,86090},6,1,51,0.00024292916719171776,2.7915266248038314e-13,0.0015385791333494157,0.000313662,0.001872}
    }};
    return policies;
}

inline const NumiHumanOneSidedRibFragmentPolicy*
numiHumanMatchRow317OneSidedRibFragment(const std::vector<unsigned>& sourceFaceOrdinals) {
    if(!std::is_sorted(sourceFaceOrdinals.begin(),sourceFaceOrdinals.end())||
        std::adjacent_find(sourceFaceOrdinals.begin(),sourceFaceOrdinals.end())!=sourceFaceOrdinals.end())
        throw std::invalid_argument("row-317 one-sided fragment source-face ancestry is not ordered and unique");
    const auto& policies=numiHumanRow317OneSidedRibFragmentPolicies();
    for(const auto& policy:policies)
        if(sourceFaceOrdinals==policy.sourceFaceOrdinals)return &policy;
    for(const auto& policy:policies)
        for(const unsigned expected:policy.sourceFaceOrdinals)
            if(std::binary_search(sourceFaceOrdinals.begin(),sourceFaceOrdinals.end(),expected))
                throw std::invalid_argument("row-317 one-sided fragment face ancestry is only a partial policy match");
    return nullptr;
}

inline void numiHumanValidateRow317OneSidedRibFragment(
    const NumiHumanOneSidedRibFragmentPolicy& policy,unsigned rowStableId,
    unsigned sourceVertexCount,unsigned sourceIndexCount,unsigned quotientVertexCount,
    int lateralSide,unsigned selectedRibStableId,double diameterM,double signedVolumeM3,
    double maximumVertexDistanceToSelectedRibM,double minimumRunnerUpRibDistanceM) {
    if(rowStableId!=317||sourceVertexCount!=70351||sourceIndexCount!=261732||
        quotientVertexCount!=policy.sourceVertexCount||lateralSide!=policy.lateralSide||
        selectedRibStableId!=policy.selectedRibStableId)
        throw std::invalid_argument("row-317 one-sided fragment identity or typed rib association changed");
    if(!std::isfinite(diameterM)||!std::isfinite(signedVolumeM3)||
        !std::isfinite(maximumVertexDistanceToSelectedRibM)||!std::isfinite(minimumRunnerUpRibDistanceM)||
        diameterM<=0||diameterM>policy.maximumDiameterM||
        signedVolumeM3<0.5*policy.measuredPositiveVolumeM3||
        signedVolumeM3>2.0*policy.measuredPositiveVolumeM3||
        maximumVertexDistanceToSelectedRibM>policy.maximumNearestRibDistanceM||
        minimumRunnerUpRibDistanceM<policy.minimumRunnerUpRibDistanceM)
        throw std::invalid_argument("row-317 one-sided fragment geometry escaped its measured bounds");
}

inline std::vector<NumiHumanRestingRibFieldChannel> numiHumanRestingValidateRibSideChannels(
    std::vector<NumiHumanRestingRibFieldChannel> channels,int side) {
    if((side!=-1&&side!=1)||channels.size()!=12)
        throw std::invalid_argument("continuous intercostal field requires exactly twelve channels on one typed side");
    std::sort(channels.begin(),channels.end(),[](const auto& a,const auto& b) {
        return a.channel<b.channel;
    });
    for(std::size_t i=0;i<channels.size();++i) {
        if(!channels[i].stableId||channels[i].lateralSide!=side||
            (i&&channels[i].channel!=channels[i-1].channel+1))
            throw std::invalid_argument("continuous intercostal rib channels are not a valid same-side ordered chain");
        for(std::size_t j=0;j<i;++j)if(channels[j].stableId==channels[i].stableId)
            throw std::invalid_argument("continuous intercostal rib channels contain a duplicate typed identity");
    }
    return channels;
}

inline NumiHumanRestingRibFieldValue numiHumanEncodeContinuousRibOrdinal(
    double scalarOrdinal,const std::vector<NumiHumanRestingRibFieldChannel>& rawChannels,int side) {
    const auto channels=numiHumanRestingValidateRibSideChannels(rawChannels,side);
    if(!std::isfinite(scalarOrdinal)||scalarOrdinal<0||scalarOrdinal>11)
        throw std::invalid_argument("continuous intercostal rib ordinal is outside its twelve-channel chain");
    const unsigned first=std::min(10u,unsigned(std::floor(scalarOrdinal)));
    const double blend=scalarOrdinal==11.0?1.0:scalarOrdinal-double(first);
    const auto& a=channels[first];const auto& b=channels[first+1];
    return {a.channel,a.stableId,b.channel,b.stableId,first,scalarOrdinal,float(blend),side};
}

struct NumiHumanPositiveGraphHarmonicResult {
    std::vector<double> values;
    unsigned iterations=0;
    double residualInfinity=0;
    double maximumPrincipleViolation=0;
};

inline NumiHumanPositiveGraphHarmonicResult numiHumanSolvePositiveGraphHarmonic(
    const std::vector<std::vector<std::pair<unsigned,double>>>& adjacency,
    const std::vector<double>& dirichlet,double relativeTolerance=1.0e-9,
    unsigned maximumIterations=20000) {
    const std::size_t count=adjacency.size();
    if(count==0||dirichlet.size()!=count||!std::isfinite(relativeTolerance)||
        relativeTolerance<=0||relativeTolerance>=1||maximumIterations==0)
        throw std::invalid_argument("positive harmonic graph dimensions or solver bounds are invalid");
    double minimum=std::numeric_limits<double>::infinity();
    double maximum=-std::numeric_limits<double>::infinity();
    unsigned anchorCount=0;
    for(const double value:dirichlet)if(std::isfinite(value)) {
        minimum=std::min(minimum,value);maximum=std::max(maximum,value);++anchorCount;
    }
    if(anchorCount<2||!(maximum>minimum))
        throw std::invalid_argument("positive harmonic graph lacks distinct Dirichlet anchors");
    std::vector<double> diagonal(count,0.0);
    std::vector<bool> reached(count,false);
    std::vector<unsigned> queue;
    for(unsigned i=0;i<count;++i)if(std::isfinite(dirichlet[i])) {
        reached[i]=true;queue.push_back(i);
    }
    for(unsigned i=0;i<count;++i) {
        unsigned previous=0;bool first=true;std::set<unsigned> unique;
        for(const auto& [neighbor,weight]:adjacency[i]) {
            if(neighbor>=count||neighbor==i||!std::isfinite(weight)||weight<=0||
                !unique.insert(neighbor).second)
                throw std::invalid_argument("positive harmonic graph has an invalid edge");
            if(!first&&neighbor<=previous)
                throw std::invalid_argument("positive harmonic adjacency is not deterministically sorted");
            first=false;previous=neighbor;diagonal[i]+=weight;
            const auto& reverse=adjacency[neighbor];
            const auto found=std::lower_bound(reverse.begin(),reverse.end(),i,
                [](const auto& edge,unsigned value){return edge.first<value;});
            if(found==reverse.end()||found->first!=i||found->second!=weight)
                throw std::invalid_argument("positive harmonic graph edge is not symmetric");
        }
        if(diagonal[i]<=0||!std::isfinite(diagonal[i]))
            throw std::invalid_argument("positive harmonic graph contains an isolated node");
    }
    for(std::size_t cursor=0;cursor<queue.size();++cursor)
        for(const auto& [neighbor,weight]:adjacency[queue[cursor]]) {
            (void)weight;
            if(!reached[neighbor]){reached[neighbor]=true;queue.push_back(neighbor);}
        }
    if(queue.size()!=count)
        throw std::invalid_argument("positive harmonic graph has an unanchored connected component");

    std::vector<int> freeIndex(count,-1);
    std::vector<unsigned> freeNodes;
    for(unsigned i=0;i<count;++i)if(!std::isfinite(dirichlet[i])) {
        freeIndex[i]=static_cast<int>(freeNodes.size());freeNodes.push_back(i);
    }
    NumiHumanPositiveGraphHarmonicResult result;result.values=dirichlet;
    if(freeNodes.empty())return result;
    const std::size_t freeCount=freeNodes.size();
    std::vector<double> rhs(freeCount,0.0),x(freeCount,0.0),residual(freeCount),z(freeCount),direction(freeCount),product(freeCount);
    for(std::size_t local=0;local<freeCount;++local) {
        const unsigned node=freeNodes[local];
        for(const auto& [neighbor,weight]:adjacency[node])
            if(std::isfinite(dirichlet[neighbor]))rhs[local]+=weight*dirichlet[neighbor];
        x[local]=(minimum+maximum)*0.5;
    }
    const auto apply=[&](const std::vector<double>& input,std::vector<double>& output) {
        for(std::size_t local=0;local<freeCount;++local) {
            const unsigned node=freeNodes[local];double value=diagonal[node]*input[local];
            for(const auto& [neighbor,weight]:adjacency[node]) {
                const int other=freeIndex[neighbor];if(other>=0)value-=weight*input[static_cast<std::size_t>(other)];
            }
            output[local]=value;
        }
    };
    apply(x,product);double rhsNormSquared=0,residualNormSquared=0,rz=0;
    for(std::size_t i=0;i<freeCount;++i) {
        residual[i]=rhs[i]-product[i];rhsNormSquared+=rhs[i]*rhs[i];
        z[i]=residual[i]/diagonal[freeNodes[i]];direction[i]=z[i];
        residualNormSquared+=residual[i]*residual[i];rz+=residual[i]*z[i];
    }
    const double target=relativeTolerance*std::max(1.0,std::sqrt(rhsNormSquared));
    bool converged=std::sqrt(residualNormSquared)<=target;
    for(unsigned iteration=0;iteration<maximumIterations&&!converged;++iteration) {
        apply(direction,product);double denominator=0;
        for(std::size_t i=0;i<freeCount;++i)denominator+=direction[i]*product[i];
        if(!(denominator>0)||!std::isfinite(denominator)||!(rz>=0)||!std::isfinite(rz))
            throw std::invalid_argument("positive harmonic conjugate-gradient system is not positive definite");
        const double alpha=rz/denominator;residualNormSquared=0;
        for(std::size_t i=0;i<freeCount;++i) {
            x[i]+=alpha*direction[i];residual[i]-=alpha*product[i];residualNormSquared+=residual[i]*residual[i];
        }
        result.iterations=iteration+1;
        if(std::sqrt(residualNormSquared)<=target){converged=true;break;}
        double nextRz=0;
        for(std::size_t i=0;i<freeCount;++i) {
            z[i]=residual[i]/diagonal[freeNodes[i]];nextRz+=residual[i]*z[i];
        }
        if(!(nextRz>=0)||!std::isfinite(nextRz)||!(rz>0))
            throw std::invalid_argument("positive harmonic conjugate-gradient residual is invalid");
        const double beta=nextRz/rz;
        for(std::size_t i=0;i<freeCount;++i)direction[i]=z[i]+beta*direction[i];
        rz=nextRz;
    }
    if(!converged)throw std::invalid_argument("positive harmonic graph did not converge within its bounded iteration count");
    for(std::size_t local=0;local<freeCount;++local)result.values[freeNodes[local]]=x[local];
    const double rangeScale=std::max({1.0,std::abs(minimum),std::abs(maximum)});
    const double allowed=std::max(1.0e-12,relativeTolerance*10.0)*rangeScale;
    for(double& value:result.values) {
        if(!std::isfinite(value))throw std::invalid_argument("positive harmonic field contains a nonfinite value");
        result.maximumPrincipleViolation=std::max(result.maximumPrincipleViolation,
            std::max({0.0,minimum-value,value-maximum}));
        if(value<minimum-allowed||value>maximum+allowed)
            throw std::invalid_argument("positive harmonic field violates the maximum principle");
        value=std::clamp(value,minimum,maximum);
    }
    apply(x,product);result.residualInfinity=0;
    for(std::size_t i=0;i<freeCount;++i)
        result.residualInfinity=std::max(result.residualInfinity,std::abs(product[i]-rhs[i]));
    if(result.residualInfinity>std::max(target*10.0,1.0e-10))
        throw std::invalid_argument("positive harmonic field residual exceeds its bounded tolerance");
    return result;
}

template<class DistanceToChannel>
NumiHumanRestingRibFieldValue numiHumanEvaluateContinuousRibOrdinal(
    const std::array<double,3>& point,
    const std::vector<NumiHumanRestingRibFieldChannel>& rawChannels,
    int side,DistanceToChannel distanceToChannel) {
    const auto channels=numiHumanRestingValidateRibSideChannels(rawChannels,side);
    std::array<double,12> distances{};
    unsigned zeroCount=0,zeroOrdinal=0;
    double minimum=std::numeric_limits<double>::infinity();
    for(unsigned i=0;i<12;++i) {
        const double d=distanceToChannel(point,channels[i].channel);
        if(!std::isfinite(d)||d<0)
            throw std::invalid_argument("continuous intercostal rib field has an invalid surface distance");
        distances[i]=d;
        if(d==0){++zeroCount;zeroOrdinal=i;}
        minimum=std::min(minimum,d);
    }
    if(zeroCount>1)
        throw std::invalid_argument("continuous intercostal rib field has conflicting exact zero-distance rib surfaces");
    if(zeroCount==1)
        return numiHumanEncodeContinuousRibOrdinal(double(zeroOrdinal),channels,side);
    if(!(minimum>0)||!std::isfinite(minimum))
        throw std::invalid_argument("continuous intercostal rib field has no positive finite distance scale");

    // Scale by d_min before squaring to prevent inverse-distance overflow.
    double denominator=0,numerator=0;
    for(unsigned i=0;i<12;++i) {
        const double ratio=minimum/distances[i];
        const double weight=ratio*ratio;
        denominator+=weight;
        numerator+=double(i)*weight;
    }
    if(!(denominator>0)||!std::isfinite(denominator)||!std::isfinite(numerator))
        throw std::invalid_argument("continuous intercostal rib field normalization failed");
    const double scalar=std::clamp(numerator/denominator,0.0,11.0);
    return numiHumanEncodeContinuousRibOrdinal(scalar,channels,side);
}

template<class DistanceToChannel>
int numiHumanChooseIntercostalRibFieldSide(
    const std::vector<std::array<double,3>>& samples,
    const std::vector<NumiHumanRestingRibFieldChannel>& channels,
    DistanceToChannel distanceToChannel) {
    if(samples.empty()||channels.size()!=24)
        throw std::invalid_argument("intercostal rib laterality requires samples and all twenty-four typed ribs");
    std::array<std::vector<NumiHumanRestingRibFieldChannel>,2> sides;
    for(const auto& c:channels) {
        if(c.lateralSide!=-1&&c.lateralSide!=1)
            throw std::invalid_argument("intercostal rib laterality has an invalid typed side");
        sides[c.lateralSide<0?0u:1u].push_back(c);
    }
    sides[0]=numiHumanRestingValidateRibSideChannels(std::move(sides[0]),-1);
    sides[1]=numiHumanRestingValidateRibSideChannels(std::move(sides[1]),1);
    std::array<double,2> score{};
    for(const auto& point:samples) {
        for(unsigned s=0;s<2;++s) {
            double nearest=std::numeric_limits<double>::infinity();
            for(const auto& channel:sides[s]) {
                const double d=distanceToChannel(point,channel.channel);
                if(!std::isfinite(d)||d<0)
                    throw std::invalid_argument("intercostal rib laterality has an invalid surface distance");
                nearest=std::min(nearest,d);
            }
            score[s]+=nearest/double(samples.size());
        }
    }
    if(score[0]==score[1])
        throw std::invalid_argument("intercostal component has ambiguous equal-distance rib laterality");
    return score[0]<score[1]?-1:1;
}

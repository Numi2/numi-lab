#include "NumiHumanConservativeSpatialMassTransport.hpp"
#include "ConservativeSpatialFractionalSemanticSupport.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

void dumpFlowParity(const std::string& label,const ConservativeSpatialMassTransportResult& result) {
    std::cout<<"FLOW_PARITY "<<label<<" receipt="<<result.receiptSHA256
             <<" candidates="<<result.semanticCandidateEdgeCount
             <<" augmentations="<<result.augmentationCount<<"\n";
    std::cout<<std::setprecision(17);
    for(std::size_t c=0;c<result.childMassKg.size();++c)
        std::cout<<"FLOW_CHILD "<<label<<" "<<c<<" "<<result.childMassKg[c]
                 <<" "<<result.childParentOffsets[c]<<" "<<result.childParentOffsets[c+1u]<<"\n";
    for(std::size_t i=0;i<result.childParentRows.size();++i)
        std::cout<<"FLOW_PARENT_ROW "<<label<<" "<<i<<" "<<result.childParentRows[i]<<"\n";
}

#ifndef NUMI_FLOW_PARITY_ONLY
bool runForestPathSelectionControl() {
    using namespace fractional_semantic_transport_detail;
    const auto invalid=std::numeric_limits<std::uint32_t>::max();
    const auto none=std::numeric_limits<std::size_t>::max();
    // 0=source; 1/3=parents; 2/4/5/6=children. Edge 2 is an unrelated
    // reached dead-end branch; targets 4 and 6 require the other three edges.
    const std::vector<std::uint32_t> predecessor{invalid,0,1,2,3,1,1,6};
    const std::vector<std::size_t> omitted{none,none,0,none,1,2,3,none};
    const std::vector<std::uint32_t> targets{4,6,4};
    std::vector<bool> selected(4u,false);
    std::size_t targetCount=0u,edgeCount=0u;
    std::string error;
    if(!selectImplicitResidualForestPathEdges(0u,predecessor,omitted,targets,4u,
            selected,targetCount,edgeCount,error)||targetCount!=2u||edgeCount!=3u||
       !selected[0]||!selected[1]||selected[2]||!selected[3]){
        std::cerr<<"residual forest target-path union control failed: "<<error
                 <<" targets="<<targetCount<<" edges="<<edgeCount<<"\n";return false;
    }
    std::vector<bool> emptySelected(0u,false);
    targetCount=1u;edgeCount=1u;
    if(!selectImplicitResidualForestPathEdges(0u,std::vector<std::uint32_t>{invalid},
            std::vector<std::size_t>{none},std::vector<std::uint32_t>{},0u,
            emptySelected,targetCount,edgeCount,error)||targetCount!=0u||edgeCount!=0u){
        std::cerr<<"residual forest empty-target control failed: "<<error<<"\n";return false;
    }
    std::vector<bool> invalidPathSelected(1u,false);
    if(selectImplicitResidualForestPathEdges(0u,
            std::vector<std::uint32_t>{invalid,0u,invalid},
            std::vector<std::size_t>{none,none,none},std::vector<std::uint32_t>{2u},
            1u,invalidPathSelected,targetCount,edgeCount,error)||
       error.find("no source path")==std::string::npos){
        std::cerr<<"residual forest invalid-path fail-closed control failed: "<<error<<"\n";return false;
    }
    std::vector<bool> invalidIndexSelected(1u,false);
    if(selectImplicitResidualForestPathEdges(0u,predecessor,omitted,{4u},1u,
            invalidIndexSelected,targetCount,edgeCount,error)||
       error.find("edge index")==std::string::npos){
        std::cerr<<"residual forest invalid-edge-index control failed: "<<error<<"\n";return false;
    }
    std::cout<<"residual forest reachable sink-path union, duplicate, empty and fail-closed controls PASS\n";
    return true;
}
#endif

#ifndef NUMI_FLOW_PARITY_ONLY
bool runDirectCapacityGreedyControl() {
    using namespace fractional_semantic_transport_detail;
    std::vector<std::uint32_t> parents{0u,1u,0u,1u,0u};
    std::vector<std::uint32_t> children{0u,0u,1u,1u,0u};
    std::vector<double> source{0.8,0.5};
    std::vector<double> demand{0.6,0.4};
    double finalSink=std::numeric_limits<double>::infinity();
    std::vector<std::size_t> selected;
    double simulated=0.0;
    std::string error;
    if(!simulateDirectCapacityGreedy(parents,children,source,demand,finalSink,8u,
            selected,simulated,error)||selected!=std::vector<std::size_t>{0u,2u,3u}||
       std::abs(simulated-1.0)>1.0e-12||source[0]!=0.0||
       std::abs(source[1]-0.3)>1.0e-12||demand[0]!=0.0||demand[1]!=0.0){
        std::cerr<<"direct capacity greedy shared-budget/duplicate control failed: "<<error
                 <<" selected="<<selected.size()<<" sum="<<simulated<<"\n";return false;
    }
    source={1.0,1.0};demand={1.0,1.0};finalSink=0.75;
    selected.clear();simulated=0.0;
    if(!simulateDirectCapacityGreedy(std::vector<std::uint32_t>{0u,1u},
            std::vector<std::uint32_t>{0u,1u},source,demand,finalSink,2u,
            selected,simulated,error)||selected.size()!=1u||
       std::abs(simulated-0.75)>1.0e-12||finalSink!=0.0){
        std::cerr<<"direct capacity greedy shared final-sink budget control failed: "<<error
                 <<" selected="<<selected.size()<<" sum="<<simulated<<"\n";return false;
    }
    source={-0.1};demand={1.0};finalSink=std::numeric_limits<double>::infinity();
    if(simulateDirectCapacityGreedy(std::vector<std::uint32_t>{0u},
            std::vector<std::uint32_t>{0u},source,demand,finalSink,1u,
            selected,simulated,error)||error.find("invalid residual capacity")==std::string::npos){
        std::cerr<<"direct capacity greedy invalid-capacity fail-closed control failed: "<<error
                 <<"\n";return false;
    }
    std::cout<<"direct capacity greedy duplicate, parent/child shared budgets, final sink, and invalid-capacity controls PASS\n";
    return true;
}
#endif


bool runCrossPathSelectionControl() {
    using namespace fractional_semantic_transport_detail;
    using spatial_mass_transport_detail::MinCostFlow;
    using spatial_mass_transport_detail::EdgeRef;
    const auto invalid=std::numeric_limits<std::uint32_t>::max();
    const auto none=std::numeric_limits<std::size_t>::max();
    // Three reached parents can feed one positive-demand child. The first
    // predecessor path reaches the child through parent A, while parent B's
    // source prefix includes an omitted forest edge. C has an independent path.
    const std::vector<std::uint32_t> predecessor{invalid,0u,1u,2u,0u,1u,5u,6u};
    const std::vector<std::size_t> omitted{none,none,0u,none,none,1u,none,none};
    const std::uint64_t pairA=(std::uint64_t{0}<<32u)|0u;
    const std::uint64_t pairB=(std::uint64_t{1}<<32u)|0u;
    const std::uint64_t pairC=(std::uint64_t{2}<<32u)|0u;
    const std::vector<std::uint64_t> initialTreePairs{pairA};
    const std::vector<std::uint32_t> sinkChildren{5u};
    const std::vector<ImplicitCrossPathSelectionCandidate> candidates{
        {1u,5u,pairA,0.4},
        {3u,5u,pairB,0.4},
        {3u,5u,pairB,0.4}, // duplicate candidate is ignored
        {4u,5u,pairC,0.3}};
    std::vector<bool> selectedForest;
    std::vector<std::size_t> selectedCandidates;
    std::vector<std::uint32_t> pathTargets;
    std::size_t targetCount=0u,forestCount=0u;
    std::string error;
    if(!selectImplicitCrossPathCandidates(0u,predecessor,omitted,initialTreePairs,
            sinkChildren,candidates,2u,2u,8u,selectedForest,selectedCandidates,
            pathTargets,targetCount,forestCount,error)||
       selectedCandidates!=std::vector<std::size_t>{1u,3u}||
       pathTargets!=std::vector<std::uint32_t>{5u,3u,4u}||
       targetCount!=3u||forestCount!=2u||
       selectedForest!=std::vector<bool>{true,true}){
        std::cerr<<"parallel reached-parent cross-path selection failed: "<<error
                 <<" selected="<<selectedCandidates.size()<<" targets="<<targetCount
                 <<" forest="<<forestCount<<"\n";return false;
    }
    const auto parallelSelectedForest=selectedForest;
    const auto parallelSelectedCandidates=selectedCandidates;
    std::vector<std::size_t> budgetCandidates;
    std::vector<std::uint32_t> budgetTargets;
    std::size_t budgetTargetCount=0u,budgetForestCount=0u;
    if(!selectImplicitCrossPathCandidates(0u,predecessor,omitted,initialTreePairs,
            sinkChildren,candidates,2u,2u,2u,selectedForest,budgetCandidates,
            budgetTargets,budgetTargetCount,budgetForestCount,error)||
       budgetCandidates!=std::vector<std::size_t>{3u}||
       budgetForestCount+budgetCandidates.size()!=2u){
        std::cerr<<"cross-path edge-budget control failed: "<<error<<"\n";return false;
    }

    const std::vector<std::uint32_t> cyclicPredecessor{invalid,0u,1u};
    const std::vector<std::size_t> noOmitted(3u,none);
    std::vector<std::size_t> cycleCandidates;
    std::vector<std::uint32_t> cycleTargets;
    std::size_t cycleTargetCount=0u,cycleForestCount=0u;
    if(!selectImplicitCrossPathCandidates(0u,cyclicPredecessor,noOmitted,{},
            {1u},std::vector<ImplicitCrossPathSelectionCandidate>{{2u,1u,7u,0.5}},
            4u,0u,4u,selectedForest,cycleCandidates,cycleTargets,
            cycleTargetCount,cycleForestCount,error)||
       !cycleCandidates.empty()||cycleTargets!=std::vector<std::uint32_t>{1u}||
       cycleForestCount!=0u){
        std::cerr<<"cross-path ancestor-cycle rejection control failed: "<<error<<"\n";return false;
    }

    struct TestArc {std::uint32_t from,to,index;};
    const auto addArc=[](MinCostFlow& net,std::uint32_t from,std::uint32_t to,double cap){
        return TestArc{from,to,net.addEdge(from,to,cap,0.0)};
    };
    const auto arcFlow=[](const MinCostFlow& net,const TestArc& arc){
        return net.reverseFlow(EdgeRef{arc.from,arc.to,arc.from,arc.index,0.0});
    };
    const auto addFixed=[&](MinCostFlow& net){
        std::vector<TestArc> arcs;
        arcs.push_back(addArc(net,0u,1u,0.4));
        arcs.push_back(addArc(net,2u,3u,0.4));
        arcs.push_back(addArc(net,0u,4u,0.3));
        arcs.push_back(addArc(net,5u,6u,0.7));
        arcs.push_back(addArc(net,6u,7u,0.7));
        return arcs;
    };
    MinCostFlow baseline(8u);
    auto baselineArcs=addFixed(baseline);
    const std::array<TestArc,4> optionalBaseline{{
        addArc(baseline,1u,2u,0.4),addArc(baseline,1u,5u,0.4),
        addArc(baseline,3u,5u,0.4),addArc(baseline,4u,5u,0.3)}};
    baselineArcs.insert(baselineArcs.end(),optionalBaseline.begin(),optionalBaseline.end());
    const std::array<bool,4> optionalPresent{
        parallelSelectedForest[0],parallelSelectedForest[1],
        std::find(parallelSelectedCandidates.begin(),parallelSelectedCandidates.end(),1u)!=parallelSelectedCandidates.end(),
        std::find(parallelSelectedCandidates.begin(),parallelSelectedCandidates.end(),3u)!=parallelSelectedCandidates.end()};
    // Rebuild the candidate in canonical arc order so this comparison also
    // checks the complete per-edge flow vector against the full graph.
    MinCostFlow materialized(8u);
    std::vector<TestArc> candidateArcs=addFixed(materialized);
    for(std::size_t i=0;i<optionalPresent.size();++i)if(optionalPresent[i]){
        const std::array<std::uint32_t,4> from{1u,1u,3u,4u};
        const std::array<std::uint32_t,4> to{2u,5u,5u,5u};
        const std::array<double,4> cap{0.4,0.4,0.4,0.3};
        candidateArcs.push_back(addArc(materialized,from[i],to[i],cap[i]));
    }
    double baselineSent=0.0,candidateSent=0.0;
    std::size_t baselineAug=0u,candidateAug=0u;
    if(!baseline.runFeasible(0u,7u,32u,baselineAug,baselineSent,error)||
       !materialized.runFeasible(0u,7u,32u,candidateAug,candidateSent,error)||
       std::abs(baselineSent-0.7)>1.0e-14||std::abs(candidateSent-baselineSent)>1.0e-14||
       baselineArcs.size()!=candidateArcs.size()){
        std::cerr<<"parallel cross-path exact maxflow control failed: "<<error
                 <<" baseline="<<baselineSent<<" candidate="<<candidateSent<<"\n";return false;
    }
    for(std::size_t i=0;i<baselineArcs.size();++i)
        if(std::abs(arcFlow(baseline,baselineArcs[i])-
                    arcFlow(materialized,candidateArcs[i]))>1.0e-14){
            std::cerr<<"parallel cross-path edge-flow parity failed at "<<i<<"\n";return false;
        }
    std::cout<<"parallel reached-parent cross paths, prefix union, cycle/budget controls, exact maxflow/edge-flow parity PASS"
             <<" selected_cross_edges="<<parallelSelectedCandidates.size()
             <<" selected_prefix_edges="<<forestCount
             <<" sent="<<candidateSent<<"\n";
    return true;
}


bool runPositiveDemandChildTerminalCrossPathControl() {
    using namespace fractional_semantic_transport_detail;
    using spatial_mass_transport_detail::MinCostFlow;
    using spatial_mass_transport_detail::EdgeRef;
    const auto invalid=std::numeric_limits<std::uint32_t>::max();
    const auto none=std::numeric_limits<std::size_t>::max();
    if(!isTerminalPositiveDemandChild(true,true,false)||
       !isTerminalPositiveDemandChild(false,true,true)||
       isTerminalPositiveDemandChild(false,true,false)||
       isTerminalPositiveDemandChild(false,false,true)){
        std::cerr<<"positive-demand terminal phase/final-sink threshold controls failed\n";return false;
    }

    struct TestArc {std::uint32_t from,to,index;};
    const auto addArc=[](MinCostFlow& net,std::uint32_t from,std::uint32_t to,double cap){
        return TestArc{from,to,net.addEdge(from,to,cap,0.0)};
    };
    const auto arcFlow=[](const MinCostFlow& net,const TestArc& arc){
        return net.reverseFlow(EdgeRef{arc.from,arc.to,arc.from,arc.index,0.0});
    };
    const double threshold=0.1, massA=0.4, massB=0.3, demand=0.7;
    MinCostFlow discovery(8u);
    const auto sourceA=addArc(discovery,0u,1u,massA);
    const auto sourceX=addArc(discovery,0u,2u,massB);
    const auto xToY=addArc(discovery,2u,3u,massB);
    const auto yToB=addArc(discovery,3u,5u,massB);
    const auto childToB=addArc(discovery,4u,5u,massB);
    const auto childToSink=addArc(discovery,4u,6u,demand);
    const auto sinkToSuper=addArc(discovery,6u,7u,demand);
    (void)sourceA;(void)sourceX;(void)xToY;(void)yToB;(void)childToB;

    std::vector<std::uint32_t> pred(8u,invalid);
    std::vector<std::size_t> predOmitted(8u,none);
    std::vector<double> bottleneck(8u,0.0);
    std::vector<bool> reached(8u,false);
    std::vector<std::uint32_t> queue{0u};
    reached[0]=true;bottleneck[0]=std::numeric_limits<double>::infinity();
    std::vector<ImplicitCrossPathSelectionCandidate> crossCandidates;
    std::vector<std::uint32_t> omittedParents{1u,5u};
    std::size_t omittedEdges=0u;
    bool finalReached=false;
    for(std::size_t cursor=0;cursor<queue.size();++cursor){
        const auto from=queue[cursor];
        if(from==7u)continue;
        if(from==6u&&discovery.hasResidualCapacityAtLeast(6u,sinkToSuper.index,threshold)){
            const double cap=discovery.remainingCapacity(6u,sinkToSuper.index);
            if(!reached[7u]){reached[7u]=true;pred[7u]=6u;bottleneck[7u]=std::min(bottleneck[6u],cap);queue.push_back(7u);}
            finalReached=true;continue;
        }
        if(from==4u){
            const bool childAtLeast=discovery.hasResidualCapacityAtLeast(4u,childToSink.index,threshold);
            const bool finalAtLeast=discovery.hasResidualCapacityAtLeast(6u,sinkToSuper.index,threshold);
            if(isTerminalPositiveDemandChild(false,childAtLeast,finalAtLeast)){
                const double cap=discovery.remainingCapacity(4u,childToSink.index);
                if(!reached[6u]){reached[6u]=true;pred[6u]=4u;bottleneck[6u]=std::min(bottleneck[4u],cap);queue.push_back(6u);}
                continue;
            }
        }
        discovery.forEachResidualEdgeAtLeast(from,threshold,[&](const std::uint32_t to,double cap){
            if(reached[to])return;
            reached[to]=true;pred[to]=from;bottleneck[to]=std::min(bottleneck[from],cap);queue.push_back(to);
            if(to==7u)finalReached=true;
        });
        if(from!=1u&&from!=5u)continue;
        const std::uint32_t parentRow=from==1u?0u:1u;
        const double parentMass=parentRow==0u?massA:massB;
        const bool childPositive=discovery.hasResidualCapacityAtLeast(4u,childToSink.index,threshold);
        const bool finalPositive=discovery.hasResidualCapacityAtLeast(6u,sinkToSuper.index,threshold);
        if(isTerminalPositiveDemandChild(false,childPositive,finalPositive)){
            const double score=std::min({bottleneck[from],parentMass,
                discovery.remainingCapacity(4u,childToSink.index),
                discovery.remainingCapacity(6u,sinkToSuper.index)});
            crossCandidates.push_back({from,4u,(static_cast<std::uint64_t>(parentRow)<<32u),score});
        }
        if(!reached[4u]){
            reached[4u]=true;pred[4u]=from;predOmitted[4u]=omittedEdges;
            bottleneck[4u]=std::min(bottleneck[from],parentMass);queue.push_back(4u);
            ++omittedEdges;
        }else{
            // Parent B's arc is an alternative to a child already reached by A.
            ++omittedEdges;
        }
    }
    if(!finalReached||!reached[5u]||pred[5u]!=3u||crossCandidates.size()!=2u||omittedEdges!=2u){
        std::cerr<<"terminal child traversal did not retain the independent parent route"
                 <<" final="<<finalReached<<" Bpred="<<pred[5u]
                 <<" cross="<<crossCandidates.size()<<" omitted="<<omittedEdges<<"\n";return false;
    }
    const std::uint64_t pairA=0u;
    const std::vector<std::uint64_t> initialTreePairs{pairA};
    std::vector<bool> selectedForest;
    std::vector<std::size_t> selectedCross;
    std::vector<std::uint32_t> targets;
    std::size_t targetCount=0u,forestCount=0u;
    std::string error;
    if(!selectImplicitCrossPathCandidates(0u,pred,predOmitted,initialTreePairs,{4u},
            crossCandidates,4u,omittedEdges,4u,selectedForest,selectedCross,targets,
            targetCount,forestCount,error)||
       selectedCross!=std::vector<std::size_t>{1u}||
       selectedForest!=std::vector<bool>{true,false}||forestCount!=1u||
       targets!=std::vector<std::uint32_t>{4u,5u}){
        std::cerr<<"terminal child route failed to select independent cross path: "<<error
                 <<" selected="<<selectedCross.size()<<" forest="<<forestCount<<"\n";return false;
    }

    std::vector<std::uint32_t> unprunedPred(8u,invalid);
    std::vector<std::size_t> unprunedOmitted(8u,none);
    unprunedPred[1]=0u;unprunedPred[4]=1u;unprunedOmitted[4]=0u;
    unprunedPred[5]=4u;unprunedPred[6]=4u;unprunedPred[7]=6u;
    std::vector<bool> unprunedForest;
    std::vector<std::size_t> unprunedCross;
    std::vector<std::uint32_t> unprunedTargets;
    std::size_t unprunedTargetCount=0u,unprunedForestCount=0u;
    if(!selectImplicitCrossPathCandidates(0u,unprunedPred,unprunedOmitted,
            initialTreePairs,{4u},std::vector<ImplicitCrossPathSelectionCandidate>{crossCandidates[1]},
            4u,2u,4u,unprunedForest,unprunedCross,unprunedTargets,
            unprunedTargetCount,unprunedForestCount,error)||!unprunedCross.empty()){
        std::cerr<<"unpruned reverse-parent ancestry control failed to reject cyclic cross edge: "<<error<<"\n";
        return false;
    }

    MinCostFlow baseline(8u),candidate(8u);
    auto buildFixed=[&](MinCostFlow& net){
        std::vector<TestArc> arcs;
        arcs.push_back(addArc(net,0u,1u,massA));
        arcs.push_back(addArc(net,0u,2u,massB));
        arcs.push_back(addArc(net,2u,3u,massB));
        arcs.push_back(addArc(net,3u,5u,massB));
        arcs.push_back(addArc(net,4u,5u,massB));
        arcs.push_back(addArc(net,4u,6u,demand));
        arcs.push_back(addArc(net,6u,7u,demand));
        return arcs;
    };
    auto baselineArcs=buildFixed(baseline);
    baselineArcs.push_back(addArc(baseline,1u,4u,massA));
    baselineArcs.push_back(addArc(baseline,5u,4u,massB));
    auto candidateArcs=buildFixed(candidate);
    if(selectedForest[0])candidateArcs.push_back(addArc(candidate,1u,4u,massA));
    if(std::find(selectedCross.begin(),selectedCross.end(),1u)!=selectedCross.end())
        candidateArcs.push_back(addArc(candidate,5u,4u,massB));
    double baseSent=0.0,candidateSent=0.0;std::size_t baseAug=0u,candidateAug=0u;
    if(!baseline.runFeasible(0u,7u,32u,baseAug,baseSent,error)||
       !candidate.runFeasible(0u,7u,32u,candidateAug,candidateSent,error)||
       std::abs(baseSent-demand)>1.0e-14||std::abs(candidateSent-baseSent)>1.0e-14||
       baselineArcs.size()!=candidateArcs.size()){
        std::cerr<<"terminal-child full-graph maxflow control failed: "<<error
                 <<" baseline="<<baseSent<<" candidate="<<candidateSent<<"\n";return false;
    }
    for(std::size_t i=0;i<baselineArcs.size();++i)
        if(std::abs(arcFlow(baseline,baselineArcs[i])-arcFlow(candidate,candidateArcs[i]))>1.0e-14){
            std::cerr<<"terminal-child per-edge flow parity failed at "<<i<<"\n";return false;
        }
    std::cout<<"positive-demand child terminal route, reverse-parent ancestry, parallel cross insertion, and exact maxflow parity PASS"
             <<" cross_edges="<<selectedCross.size()<<" flow="<<candidateSent<<"\n";
    return true;
}

bool runMaximumPhaseImplicitForestControl() {
    const std::array<SpatialMassTransportPoint,2> parentPoints{{
        SpatialMassTransportPoint{0.0,0.0,0.0},
        SpatialMassTransportPoint{0.875,0.0,0.0}}};
    const std::array<double,2> parentVolumes{1.0,1.5};
    const std::array<double,2> parentMasses{1.0,1.5};
    const std::array<std::uint32_t,2> parentMaterials{15,15};
    const std::array<std::uint32_t,2> parentDonors{1,1};
    const std::array<std::uint32_t,2> parentRegions{10,10};
    const std::array<std::uint32_t,3> donorOffsets{0,1,2};
    const std::array<std::uint32_t,2> donorRows{1,1};
    const std::array<double,2> donorMasses{1.0,1.5};
    const std::array<SpatialMassTransportPoint,5> childPoints{{
        SpatialMassTransportPoint{0.125,0.0,0.0},
        SpatialMassTransportPoint{0.25,0.0,0.0},
        SpatialMassTransportPoint{0.375,0.0,0.0},
        SpatialMassTransportPoint{0.5,0.0,0.0},
        SpatialMassTransportPoint{0.875,0.0,0.0}}};
    const std::array<double,5> childVolumes{0.125,0.125,0.125,0.125,1.5};
    const std::array<std::uint32_t,5> childMaterials{15,15,15,15,15};
    const std::array<std::uint32_t,5> childDonors{1,1,1,1,1};
    const std::array<std::uint32_t,5> childRegions{10,10,10,10,10};
    std::array<double,16> densityLower{};densityLower.fill(1.0);
    std::array<double,16> densityUpper{};densityUpper.fill(1.5);
    const std::array<std::uint32_t,6> childOffsets{0,1,2,3,4,5};
    const std::array<std::uint32_t,5> slotGroups{0,0,0,0,0};
    const std::array<double,5> slotFractions{1,1,1,1,1};
    const std::array<std::uint32_t,1> groupMaterial{15};
    const std::array<std::uint32_t,1> groupDonor{1};
    const std::array<std::uint32_t,1> groupRegion{10};
    const std::array<std::uint64_t,1> groupIdentity{0x8000000000000010ull};
    const std::array<std::uint32_t,2> parentStableFaces{501,502};
    const std::array<std::uint32_t,2> groupParentOffsets{0,2};
    const std::array<std::uint32_t,2> groupParentRows{0,1};
    const std::array<std::uint32_t,2> groupFaceOffsets{0,2};
    const std::array<std::uint32_t,2> groupFaceRows{501,502};
    const std::string pin(64,'b');

    ConservativeSpatialMassTransportInput input;
    input.parentCentroidsWorld=parentPoints;input.parentReferenceVolumeM3=parentVolumes;
    input.parentMassKg=parentMasses;input.parentMaterialIndices=parentMaterials;
    input.parentDonorBodyIndices=parentDonors;input.parentAnatomicalRegionIndices=parentRegions;
    input.parentSourceDonorOffsets=donorOffsets;input.parentSourceDonorBodyIndices=donorRows;
    input.parentSourceDonorMassKg=donorMasses;input.childCentroidsWorld=childPoints;
    input.childReferenceVolumeM3=childVolumes;input.childMaterialIndices=childMaterials;
    input.childDonorBodyIndices=childDonors;input.childAnatomicalRegionIndices=childRegions;
    input.materialDensityLowerKgM3=densityLower;input.materialDensityUpperKgM3=densityUpper;
    input.parentMassPacketSHA256=pin;input.parentSourceContributionSHA256=pin;
    input.childMeshSHA256=pin;input.regionLabelSHA256=pin;input.coordinateFrameSHA256=pin;
    input.maximumTransportDistanceM=1.0;input.maximumAllowedEdges=100;input.maximumAugmentations=100;

    ConservativeSpatialChildSemanticSupport support;
    support.childOffsets=childOffsets;support.slotGroupIndices=slotGroups;
    support.volumeFractions=slotFractions;support.groupMaterialIndices=groupMaterial;
    support.donorBodyIndices=groupDonor;support.anatomicalRegionIndices=groupRegion;
    support.derivedGroupIdentityIDs=groupIdentity;
    support.parentStableSourceFaceIdentities=parentStableFaces;
    support.sourceParentTetOffsets=groupParentOffsets;support.sourceParentTetRows=groupParentRows;
    support.sourceStableFaceOffsets=groupFaceOffsets;support.sourceStableFaceRows=groupFaceRows;
    support.typedSupportSHA256=pin;

    ConservativeSpatialMassTransportResult result;std::string error;
    std::ostringstream diagnostics;
    auto* savedErr=std::cerr.rdbuf(diagnostics.rdbuf());
    const bool assembled=assembleConservativeSpatialMassTransportWithSemanticSupport(input,support,result,error);
    std::cerr.rdbuf(savedErr);
    if(!assembled){
        std::cerr<<"maximum-phase implicit forest failed: "<<error<<"\n";return false;
    }
    if(!verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(input,support,result,error)){
        std::cerr<<"maximum-phase implicit forest replay failed: "<<error<<"\n";return false;
    }
    const double total=std::accumulate(result.childMassKg.begin(),result.childMassKg.end(),0.0);
    bool p0SuppliesLargeChild=false;
    for(std::uint32_t row=result.childParentOffsets[4];row<result.childParentOffsets[5];++row)
        p0SuppliesLargeChild|=result.childParentRows[row]==0u;
    if(std::abs(total-2.5)>1.0e-12||result.semanticCandidateEdgeCount!=9u||!p0SuppliesLargeChild){
        std::cerr<<"maximum-phase implicit forest did not add the omitted broad-capacity edge"
                 <<" total="<<total<<" edges="<<result.semanticCandidateEdgeCount
                 <<" p0_large_child="<<(p0SuppliesLargeChild?1:0)<<"\n";return false;
    }
#ifndef NUMI_FLOW_PARITY_ONLY
    if(diagnostics.str().find("stage=maximum_flow")==std::string::npos||
       diagnostics.str().find("reached_children=5")==std::string::npos||
       diagnostics.str().find("positive_demand_sink_neighbors=1")==std::string::npos||
       diagnostics.str().find("forest_path_edges_selected=1")==std::string::npos||
       diagnostics.str().find("direct_source_residual_pairs=")==std::string::npos||
       diagnostics.str().find("direct_edges_selected=")==std::string::npos||
       diagnostics.str().find("final_super_sink_reachable=1")==std::string::npos){
        std::cerr<<"maximum-phase predecessor/direct diagnostics missing\n";return false;
    }
#endif
#ifdef NUMI_FLOW_PARITY_ONLY
    dumpFlowParity("maximum",result);
#endif
    std::cout<<"maximum-phase ordinary-sink implicit residual forest PASS\n";
    return true;
}

std::vector<std::string> parseCSVFields(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted=false;
    for(std::size_t i=0;i<line.size();++i){
        const char ch=line[i];
        if(ch=='"'){
            if(quoted&&i+1u<line.size()&&line[i+1u]=='"'){field.push_back('"');++i;}
            else quoted=!quoted;
        }else if(ch==','&&!quoted){
            fields.push_back(field);field.clear();
        }else field.push_back(ch);
    }
    fields.push_back(field);
    return fields;
}

std::vector<std::vector<std::string>> parseWitnessSection(const std::string& witness,
                                                          const std::string& begin,
                                                          const std::string& end) {
    const auto start=witness.find(begin);
    if(start==std::string::npos)return {};
    const auto contentStart=witness.find('\n',start);
    const auto finish=witness.find(end,contentStart);
    if(contentStart==std::string::npos||finish==std::string::npos)return {};
    std::istringstream rows(witness.substr(contentStart+1u,finish-contentStart-1u));
    std::vector<std::vector<std::string>> parsed;
    std::string line;
    while(std::getline(rows,line))
        if(!line.empty())parsed.push_back(parseCSVFields(line));
    return parsed;
}

bool runFullImplicitCutWitnessControl() {
    const std::array<SpatialMassTransportPoint,2> parentPoints{{
        SpatialMassTransportPoint{0.0,0.0,0.0},
        SpatialMassTransportPoint{10.0,0.0,0.0}}};
    const std::array<double,2> parentVolumes{0.0011,0.0001};
    const std::array<double,2> parentMasses{1.1,0.1};
    const std::array<std::uint32_t,2> parentMaterials{15,15};
    const std::array<std::uint32_t,2> parentDonors{1,1};
    const std::array<std::uint32_t,2> parentRegions{10,10};
    const std::array<std::uint32_t,3> donorOffsets{0,1,2};
    const std::array<std::uint32_t,2> donorRows{1,1};
    const std::array<double,2> donorMasses{1.1,0.1};
    const std::array<SpatialMassTransportPoint,2> childPoints{{
        SpatialMassTransportPoint{0.0,0.0,0.0},
        SpatialMassTransportPoint{10.0,0.0,0.0}}};
    const std::array<double,2> childVolumes{0.0005,0.0006};
    const std::array<std::uint32_t,2> childMaterials{15,15};
    const std::array<std::uint32_t,2> childDonors{1,1};
    const std::array<std::uint32_t,2> childRegions{10,10};
    std::array<double,16> densityLower{};densityLower.fill(999.0);
    std::array<double,16> densityUpper{};densityUpper.fill(2000.0);
    const std::array<std::uint32_t,3> childOffsets{0,1,2};
    const std::array<std::uint32_t,2> slotGroups{0,0};
    const std::array<double,2> slotFractions{1,1};
    const std::array<std::uint32_t,1> groupMaterial{15};
    const std::array<std::uint32_t,1> groupDonor{1};
    const std::array<std::uint32_t,1> groupRegion{10};
    const std::array<std::uint64_t,1> groupIdentity{0x8000000000000010ull};
    const std::array<std::uint32_t,2> parentStableFaces{501,502};
    const std::array<std::uint32_t,2> groupParentOffsets{0,2};
    const std::array<std::uint32_t,2> groupParentRows{0,1};
    const std::array<std::uint32_t,2> groupFaceOffsets{0,2};
    const std::array<std::uint32_t,2> groupFaceRows{501,502};
    const std::string pin(64,'c');

    ConservativeSpatialMassTransportInput input;
    input.parentCentroidsWorld=parentPoints;input.parentReferenceVolumeM3=parentVolumes;
    input.parentMassKg=parentMasses;input.parentMaterialIndices=parentMaterials;
    input.parentDonorBodyIndices=parentDonors;input.parentAnatomicalRegionIndices=parentRegions;
    input.parentSourceDonorOffsets=donorOffsets;input.parentSourceDonorBodyIndices=donorRows;
    input.parentSourceDonorMassKg=donorMasses;input.childCentroidsWorld=childPoints;
    input.childReferenceVolumeM3=childVolumes;input.childMaterialIndices=childMaterials;
    input.childDonorBodyIndices=childDonors;input.childAnatomicalRegionIndices=childRegions;
    input.materialDensityLowerKgM3=densityLower;input.materialDensityUpperKgM3=densityUpper;
    input.parentMassPacketSHA256=pin;input.parentSourceContributionSHA256=pin;
    input.childMeshSHA256=pin;input.regionLabelSHA256=pin;input.coordinateFrameSHA256=pin;
    input.maximumTransportDistanceM=1.0;input.maximumAllowedEdges=100;input.maximumAugmentations=100;

    ConservativeSpatialChildSemanticSupport support;
    support.childOffsets=childOffsets;support.slotGroupIndices=slotGroups;
    support.volumeFractions=slotFractions;support.groupMaterialIndices=groupMaterial;
    support.donorBodyIndices=groupDonor;support.anatomicalRegionIndices=groupRegion;
    support.derivedGroupIdentityIDs=groupIdentity;
    support.parentStableSourceFaceIdentities=parentStableFaces;
    support.sourceParentTetOffsets=groupParentOffsets;support.sourceParentTetRows=groupParentRows;
    support.sourceStableFaceOffsets=groupFaceOffsets;support.sourceStableFaceRows=groupFaceRows;
    support.typedSupportSHA256=pin;

    ConservativeSpatialMassTransportResult result;std::string error;
    if(assembleConservativeSpatialMassTransportWithSemanticSupport(input,support,result,error)||
       error.find("full_implicit_residual_closure=exhausted") == std::string::npos||
       error.find("no_augmenting_path=1") == std::string::npos||
       error.find("residual_child_witnesses=") == std::string::npos){
        std::cerr<<"full implicit cut witness control failed: "<<error<<"\n";return false;
    }
    const auto marker=error.find("shared_capacity_full_cut_begin");
    if(marker==std::string::npos){std::cerr<<"full cut witness marker missing\n";return false;}
    const std::string witness=error.substr(marker);
    const auto firstLineEnd=witness.find('\n');
    if(firstLineEnd==std::string::npos||
       witness.find("phase=minimum_flow")==std::string::npos||
       witness.find("cut_capacity_minus_flow_kg=")==std::string::npos||
       witness.find("residual_child_rows=1")==std::string::npos||
       witness.find("unreachable_residual_child_rows=1")==std::string::npos){
        std::cerr<<"full cut witness metadata/newline roundtrip failed\n";return false;
    }
    const auto groupRows=parseWitnessSection(witness,"group_summary_csv_begin","group_summary_csv_end");
    const auto parentRows=parseWitnessSection(witness,"parent_group_rows_csv_begin","parent_group_rows_csv_end");
    const auto childRows=parseWitnessSection(witness,"residual_child_rows_csv_begin","residual_child_rows_csv_end");
    if(groupRows.size()!=2u||groupRows[0].size()!=8u||
       parentRows.size()!=3u||parentRows[0].size()!=13u||parentRows[1].size()!=13u||
       parentRows[1][0]!="0"||parentRows[1][5]!="0"||parentRows[1][9]!="1.1000000000000001"||
       parentRows[1][11]!="0;1"||parentRows[1][12]!="501;502"||
       childRows.size()!=2u||childRows[0].size()!=10u||
       childRows[1][0]!="1"||childRows[1][1]!="15"||
       childRows[1][9].find("0:15:1:10:")==std::string::npos){
        std::cerr<<"full cut witness CSV section roundtrip failed groups="<<groupRows.size()
                 <<" parents="<<parentRows.size()<<" children="<<childRows.size()<<"\n";
        return false;
    }
    std::cout<<"full implicit residual cut certificate and CSV roundtrip control PASS\n";
    return true;
}

int main() {
    static const std::array<SpatialMassTransportPoint, 2> parentPoints{{
        SpatialMassTransportPoint{0.0, 0.0, 0.0},
        SpatialMassTransportPoint{0.85, 0.0, 0.0}}};
    static const std::array<double, 2> parentVolumes{0.002, 0.0004};
    static const std::array<double, 2> parentMasses{2.0, 0.4};
    static const std::array<std::uint32_t, 2> parentMaterials{15, 15};
    static const std::array<std::uint32_t, 2> parentDonors{1, 1};
    static const std::array<std::uint32_t, 2> parentRegions{10, 10};
    static const std::array<std::uint32_t, 3> donorOffsets{0, 1, 2};
    static const std::array<std::uint32_t, 2> donorRows{1, 1};
    static const std::array<double, 2> donorMasses{2.0, 0.4};
    static const std::array<SpatialMassTransportPoint, 6> childPoints{{
        SpatialMassTransportPoint{0.10, 0.0, 0.0},
        SpatialMassTransportPoint{0.20, 0.0, 0.0},
        SpatialMassTransportPoint{0.25, 0.0, 0.0},
        SpatialMassTransportPoint{0.30, 0.0, 0.0},
        SpatialMassTransportPoint{0.80, 0.0, 0.0},
        SpatialMassTransportPoint{0.90, 0.0, 0.0}}};
    static const std::array<double, 6> childVolumes{
        0.0003, 0.0003, 0.0003, 0.0003, 0.0006, 0.0006};
    static const std::array<std::uint32_t, 6> childMaterials{15, 15, 15, 15, 15, 15};
    static const std::array<std::uint32_t, 6> childDonors{1, 1, 1, 1, 1, 1};
    static const std::array<std::uint32_t, 6> childRegions{10, 10, 10, 10, 10, 10};
    static const std::array<double, 16> densityLower=[] {
        std::array<double, 16> out{}; out.fill(999.0); return out;
    }();
    static const std::array<double, 16> densityUpper=[] {
        std::array<double, 16> out{}; out.fill(1001.0); return out;
    }();
    static const std::array<std::uint32_t, 7> childOffsets{0, 1, 2, 3, 4, 5, 6};
    static const std::array<std::uint32_t, 6> slotGroups{0, 0, 0, 0, 0, 0};
    static const std::array<double, 6> slotFractions{1, 1, 1, 1, 1, 1};
    static const std::array<std::uint32_t, 1> groupMaterial{15};
    static const std::array<std::uint32_t, 1> groupDonor{1};
    static const std::array<std::uint32_t, 1> groupRegion{10};
    static const std::array<std::uint64_t, 1> groupIdentity{0x8000000000000010ull};
    static const std::array<std::uint32_t, 2> parentStableFaces{501, 502};
    static const std::array<std::uint32_t, 2> groupParentOffsets{0, 2};
    static const std::array<std::uint32_t, 2> groupParentRows{0, 1};
    static const std::array<std::uint32_t, 2> groupFaceOffsets{0, 2};
    static const std::array<std::uint32_t, 2> groupFaceRows{501, 502};
    const std::string pin(64, 'a');

    ConservativeSpatialMassTransportInput input;
    input.parentCentroidsWorld=parentPoints;
    input.parentReferenceVolumeM3=parentVolumes;
    input.parentMassKg=parentMasses;
    input.parentMaterialIndices=parentMaterials;
    input.parentDonorBodyIndices=parentDonors;
    input.parentAnatomicalRegionIndices=parentRegions;
    input.parentSourceDonorOffsets=donorOffsets;
    input.parentSourceDonorBodyIndices=donorRows;
    input.parentSourceDonorMassKg=donorMasses;
    input.childCentroidsWorld=childPoints;
    input.childReferenceVolumeM3=childVolumes;
    input.childMaterialIndices=childMaterials;
    input.childDonorBodyIndices=childDonors;
    input.childAnatomicalRegionIndices=childRegions;
    input.materialDensityLowerKgM3=densityLower;
    input.materialDensityUpperKgM3=densityUpper;
    input.parentMassPacketSHA256=pin;
    input.parentSourceContributionSHA256=pin;
    input.childMeshSHA256=pin;
    input.regionLabelSHA256=pin;
    input.coordinateFrameSHA256=pin;
    input.maximumTransportDistanceM=1.0;
    input.maximumAllowedEdges=100;
    input.maximumAugmentations=100;

    ConservativeSpatialChildSemanticSupport support;
    support.childOffsets=childOffsets;
    support.slotGroupIndices=slotGroups;
    support.volumeFractions=slotFractions;
    support.groupMaterialIndices=groupMaterial;
    support.donorBodyIndices=groupDonor;
    support.anatomicalRegionIndices=groupRegion;
    support.derivedGroupIdentityIDs=groupIdentity;
    support.parentStableSourceFaceIdentities=parentStableFaces;
    support.sourceParentTetOffsets=groupParentOffsets;
    support.sourceParentTetRows=groupParentRows;
    support.sourceStableFaceOffsets=groupFaceOffsets;
    support.sourceStableFaceRows=groupFaceRows;
    support.typedSupportSHA256=pin;

    ConservativeSpatialMassTransportResult result;
    std::string error;
    if (!assembleConservativeSpatialMassTransportWithSemanticSupport(input, support, result, error)) {
        std::cerr << "optimization trial fixture failed: " << error << "\n";
        return 1;
    }
    const double childMass=std::accumulate(result.childMassKg.begin(),result.childMassKg.end(),0.0);
    if (result.semanticCandidateEdgeCount != 10u || result.childMassKg.size()!=6u ||
        std::abs(childMass-2.4)>1e-12 ||
        !verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(input, support, result, error)) {
        std::cerr << "optimization trial replay mismatch edges=" << result.semanticCandidateEdgeCount
                  << " error=" << error << "\n";
        return 2;
    }
#ifdef NUMI_FLOW_PARITY_ONLY
    dumpFlowParity("minimum",result);
    if(!runMaximumPhaseImplicitForestControl())return 3;
    return 0;
#else
    if(!runForestPathSelectionControl()||!runDirectCapacityGreedyControl()||
       !runCrossPathSelectionControl()||!runPositiveDemandChildTerminalCrossPathControl()||
       !runMaximumPhaseImplicitForestControl()||
       !runFullImplicitCutWitnessControl())return 3;
#endif
    std::cout << "optimized parent-K4/reverse-child-K1/full-implicit-residual-forest exact-capacity fixture PASS\n";
    return 0;
}

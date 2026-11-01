#include <NumiHumanConservativeSpatialMassTransport.hpp>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

using metalrobo::human::coarse_tissue::spatial_mass_transport_detail::EdgeRef;
using metalrobo::human::coarse_tissue::spatial_mass_transport_detail::MinCostFlow;

struct Arc {
    std::uint32_t from=0,to=0,index=0;
};

static void emit(const char* name,MinCostFlow& net,const std::vector<Arc>& arcs,
                 const bool ok,const std::size_t augmentations,const double sent,
                 const std::string& error) {
    std::cout<<name<<" ok="<<(ok?1:0)<<" aug="<<augmentations<<" sent="<<sent<<" flows=";
    for(std::size_t i=0;i<arcs.size();++i) {
        if(i)std::cout<<',';
        const auto& a=arcs[i];
        std::cout<<net.reverseFlow(EdgeRef{a.from,a.to,a.from,a.index,0.0});
    }
    std::cout<<" error="<<error<<'\n';
}

static Arc add(MinCostFlow& net,std::vector<Arc>& arcs,std::uint32_t from,
               std::uint32_t to,double cap) {
    const auto index=net.addEdge(from,to,cap,0.0);
    arcs.push_back({from,to,index});
    return arcs.back();
}

static bool run(MinCostFlow& net,std::uint32_t source,std::uint32_t sink,
                std::size_t& aug,double& sent,std::string& error) {
    return net.runFeasible(source,sink,100000,aug,sent,error);
}

int main() {
    std::cout<<std::hexfloat;
    {
        // A short sink path plus a long dead-end reachable branch exercises
        // the BFS cutoff while preserving a fully reachable residual tail.
        MinCostFlow net(10);std::vector<Arc> a;
        add(net,a,0,1,1.0);add(net,a,1,9,1.0);
        add(net,a,0,2,1.0);add(net,a,2,3,1.0);add(net,a,3,4,1.0);
        add(net,a,4,5,1.0);add(net,a,5,6,1.0);add(net,a,6,7,1.0);
        std::size_t aug=0;double sent=0;std::string error;
        const bool ok=run(net,0,9,aug,sent,error);
        emit("short_sink_deep_tail",net,a,ok,aug,sent,error);
        if(!ok||sent!=1.0||aug!=1u)return 1;
    }
    {
        // Sub-epsilon capacities must remain traversable and observable.
        MinCostFlow net(3);std::vector<Arc> a;
        const double tiny=std::ldexp(1.0,-100);
        add(net,a,0,1,tiny);add(net,a,1,2,tiny);
        std::size_t aug=0;double sent=0;std::string error;
        const bool ok=run(net,0,2,aug,sent,error);
        emit("tiny_only",net,a,ok,aug,sent,error);
        if(!ok||sent!=tiny||aug!=1u||net.reverseFlow(EdgeRef{0,1,0,a[0].index,0.0})!=tiny)return 2;
    }
    {
        // Parallel duplicate-capacity arcs retain deterministic split flow.
        MinCostFlow net(3);std::vector<Arc> a;
        add(net,a,0,1,0.25);add(net,a,0,1,0.75);add(net,a,1,2,1.0);
        std::size_t aug=0;double sent=0;std::string error;
        const bool ok=run(net,0,2,aug,sent,error);
        emit("duplicate_parallel_capacity",net,a,ok,aug,sent,error);
        if(!ok||sent!=1.0||net.reverseFlow(EdgeRef{0,1,0,a[0].index,0.0})!=0.25||
           net.reverseFlow(EdgeRef{0,1,0,a[1].index,0.0})!=0.75)return 3;
    }
    {
        // A saturated child is rerouted through a reverse arc after adding a
        // newly admitted column to the same retained residual network.
        MinCostFlow net(6);std::vector<Arc> a;
        add(net,a,0,1,1.0);add(net,a,0,2,1.0);
        const auto p1c1=add(net,a,1,3,1.0);
        const auto p2c1=add(net,a,2,3,1.0);
        add(net,a,3,5,1.0);add(net,a,4,5,1.0);
        std::size_t aug1=0;double sent1=0;std::string error;
        const bool ok1=run(net,0,5,aug1,sent1,error);
        if(!ok1||sent1!=1.0||net.reverseFlow(EdgeRef{p1c1.from,p1c1.to,p1c1.from,p1c1.index,0.0})!=1.0)return 4;
        const auto p1c2=add(net,a,1,4,1.0);
        std::size_t aug2=0;double sent2=0;
        const bool ok2=run(net,0,5,aug2,sent2,error);
        emit("cycle_reroute_retained_residual",net,a,ok1&&ok2,aug1+aug2,sent2,error);
        if(!ok2||sent2!=2.0||net.reverseFlow(EdgeRef{p1c1.from,p1c1.to,p1c1.from,p1c1.index,0.0})!=0.0||
           net.reverseFlow(EdgeRef{p2c1.from,p2c1.to,p2c1.from,p2c1.index,0.0})!=1.0||
           net.reverseFlow(EdgeRef{p1c2.from,p1c2.to,p1c2.from,p1c2.index,0.0})!=1.0)return 5;
    }
    {
        // A real bottleneck reports the same maximum partial flow and residual.
        MinCostFlow net(5);std::vector<Arc> a;
        add(net,a,0,1,2.0);add(net,a,0,2,3.0);
        add(net,a,1,3,2.0);add(net,a,2,3,3.0);add(net,a,3,4,4.5);
        std::size_t aug=0;double sent=0;std::string error;
        const bool ok=run(net,0,4,aug,sent,error);
        emit("infeasible_sink_bottleneck",net,a,ok,aug,sent,error);
        if(!ok||sent!=4.5||!net.hasResidualSourceCapacity(0))return 6;
    }
    return 0;
}

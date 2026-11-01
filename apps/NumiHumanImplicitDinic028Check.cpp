#include "NumiHumanConservativeSpatialMassTransport.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
using namespace metalrobo::human::coarse_tissue;
struct Cursor { std::size_t index=0u; };
struct Candidate { std::uint32_t parent,child; double capacity,cost; };
static bool require(bool value,const char* message){if(!value)std::cerr<<"FAIL: "<<message<<"\n";return value;}
static bool reroutingParity(){
    using spatial_mass_transport_detail::MinCostFlow;using spatial_mass_transport_detail::EdgeRef;
    MinCostFlow full(6u),implicit(6u);full.addEdge(0u,1u,1.0,0.0);implicit.addEdge(0u,1u,1.0,0.0);
    full.addEdge(0u,2u,1.0,0.0);implicit.addEdge(0u,2u,1.0,0.0);
    const auto f3=full.addEdge(3u,5u,1.0,0.0),f4=full.addEdge(4u,5u,1.0,0.0);
    const auto i3=implicit.addEdge(3u,5u,1.0,0.0),i4=implicit.addEdge(4u,5u,1.0,0.0);
    const std::vector<Candidate> pairs{{1u,3u,1.0,0.1},{1u,4u,1.0,0.2},{2u,3u,1.0,0.3}};
    std::vector<std::uint32_t> fullEdges;for(const auto& x:pairs)fullEdges.push_back(full.addEdge(x.parent,x.child,x.capacity,x.cost));
    std::vector<std::vector<std::uint32_t>> represented(6u);std::vector<EdgeRef> refs;
    std::size_t live=0u,fa=0u,ia=0u;double fs=0.0,is=0.0;std::string error;
    if(!full.runFeasible(0u,5u,100u,fa,fs,error))return require(false,"explicit flow");
    const auto next=[&](std::uint32_t from,Cursor& cur,std::uint32_t& to,double& cap,double& cost){
        while(cur.index<pairs.size()){const auto& x=pairs[cur.index++];if(x.parent!=from)continue;
            if(std::find(represented[from].begin(),represented[from].end(),x.child)!=represented[from].end())continue;
            to=x.child;cap=x.capacity;cost=x.cost;return true;}return false;};
    const auto mat=[&](std::uint32_t from,std::uint32_t to,std::uint32_t edge,double cap,double cost,std::string& why){
        const auto it=std::find_if(pairs.begin(),pairs.end(),[&](const Candidate& x){return x.parent==from&&x.child==to;});
        if(it==pairs.end()||it->capacity!=cap||it->cost!=cost){why="unexpected pair";return false;}
        represented[from].push_back(to);refs.push_back({from,to,from,edge,0.0});return true;};
    if(!implicit.runFeasibleImplicit<Cursor>(0u,5u,100u,ia,is,error,live,pairs.size(),next,mat)){std::cerr<<error<<"\n";return require(false,"implicit flow");}
    bool ok=true;ok&=require(std::abs(fs-2.0)<1e-14&&std::abs(is-fs)<1e-14,"full max flow and residual rerouting");
    ok&=require(refs.size()==3u,"materialize only path-used candidates");
    ok&=require(std::abs(full.remainingCapacity(3u,f3)-implicit.remainingCapacity(3u,i3))<1e-14&&std::abs(full.remainingCapacity(4u,f4)-implicit.remainingCapacity(4u,i4))<1e-14,"sink flow parity");
    for(std::size_t i=0;i<pairs.size();++i){double ir=pairs[i].capacity;
        for(const auto& r:refs)if(r.parent==pairs[i].parent&&r.child==pairs[i].child)ir=implicit.remainingCapacity(r.from,r.edge);
        ok&=require(std::abs(full.remainingCapacity(pairs[i].parent,fullEdges[i])-ir)<1e-14,"per-pair residual parity");}
    return ok;
}
static bool maxPhaseSink(){
    using spatial_mass_transport_detail::MinCostFlow;MinCostFlow n(5u);
    n.addEdge(0,1,1.0,0.0);n.addEdge(2,3,1.0,0.0);n.addEdge(3,4,1.0,0.0);
    std::size_t live=0u,a=0u;double sent=0.0;std::string err;
    const auto next=[](std::uint32_t from,Cursor& c,std::uint32_t& to,double& cap,double& cost){if(from!=1u||c.index++)return false;to=2u;cap=1.0;cost=0.0;return true;};
    const auto mat=[](std::uint32_t,std::uint32_t,std::uint32_t,double,double,std::string&){return true;};
    return require(n.runFeasibleImplicit<Cursor>(0,4,10,a,sent,err,live,2,next,mat)&&std::abs(sent-1.0)<1e-14,"maximum phase ordinary sink reaches final sink");
}
static bool tinyPositive(){
    using spatial_mass_transport_detail::MinCostFlow;constexpr double tiny=1e-13;MinCostFlow n(4u);
    n.addEdge(0,1,tiny,0.0);n.addEdge(2,3,tiny,0.0);std::size_t live=0u,a=0u;double sent=0.0;std::string err;
    const auto next=[](std::uint32_t from,Cursor& c,std::uint32_t& to,double& cap,double& cost){if(from!=1u||c.index++)return false;to=2u;cap=tiny;cost=0.0;return true;};
    const auto mat=[](std::uint32_t,std::uint32_t,std::uint32_t,double,double,std::string&){return true;};
    const bool pass=n.runFeasibleImplicit<Cursor>(0,3,10,a,sent,err,live,2,next,mat);if(!pass)std::cerr<<err<<"\n";return require(pass&&live==1u&&std::abs(sent-tiny)<1e-27,"strict-positive tiny capacity");
}

static bool badProviderFailsClosed(){
    using spatial_mass_transport_detail::MinCostFlow;MinCostFlow n(4u);
    n.addEdge(0,1,1.0,0.0);n.addEdge(2,3,1.0,0.0);
    std::size_t live=0u,a=0u;double sent=0.0;std::string err,providerError;
    const auto next=[&](std::uint32_t from,Cursor&,std::uint32_t&,double&,double&){
        if(from==1u){providerError="synthetic invalid edge cost";return false;}return false;};
    const auto mat=[](std::uint32_t,std::uint32_t,std::uint32_t,double,double,std::string&){return true;};
    const bool pass=n.runFeasibleImplicit<Cursor>(0u,3u,10u,a,sent,err,live,2u,
        next,mat,nullptr,&providerError);
    return require(!pass&&err.find("provider failed")!=std::string::npos&&live==0u&&
        std::abs(n.sentFromSource(0u))<1e-14,
        "provider error cannot be mistaken for exhausted neighbors or a cut");
}
static bool capFailsClosed(){
    using spatial_mass_transport_detail::MinCostFlow;MinCostFlow n(4u);
    n.addEdge(0,1,1.0,0.0);n.addEdge(2,3,1.0,0.0);std::size_t live=0u,a=0u;double sent=0.0;std::string err;
    const auto next=[](std::uint32_t from,Cursor& c,std::uint32_t& to,double& cap,double& cost){if(from!=1u||c.index++)return false;to=2u;cap=1.0;cost=0.0;return true;};
    const auto mat=[](std::uint32_t,std::uint32_t,std::uint32_t,double,double,std::string&){return true;};
    const bool pass=n.runFeasibleImplicit<Cursor>(0,3,10,a,sent,err,live,0,next,mat);
    return require(!pass&&err.find("INCONCLUSIVE")!=std::string::npos&&live==0u&&std::abs(n.sentFromSource(0u))<1e-14,"live candidate cap fails closed before changing flow");
}
int main(){bool ok=true;ok&=reroutingParity();ok&=maxPhaseSink();ok&=tinyPositive();ok&=badProviderFailsClosed();ok&=capFailsClosed();if(!ok)return 1;
std::cout<<"implicit Dinic full-graph parity, residual rerouting, max-phase sink, tiny positive capacity, provider fail-closed, and live-cap controls PASS\n";return 0;}

#include "NumiHumanSkinBoundaryNodeMap.hpp"
#include <cassert>
#include <iostream>
using namespace numi::human;
int main(){
    std::string e;SkinBoundaryNodeMap m;
    assert(makePairedSkinBoundaryNodeMap(4,m,e));
    assert(isPairedSkinBoundaryNodeMap(m));
    assert(m.volumeNodeCount==12 && m.outer[3]==3 && m.skinInner[3]==7 && m.fatDeep[3]==11);
    assert(validateSkinBoundaryNodeMap(m,4,e));
    assert(skinBoundaryNodeRangeFits(m,1666,1678));
    assert(!skinBoundaryNodeRangeFits(m,1666,1677));
    assert(!skinBoundaryNodeRangeFits(m,std::numeric_limits<std::size_t>::max(),1678));
    auto previous=m;
    assert(!makePairedSkinBoundaryNodeMap(0,m,e));
    assert(m.outer==previous.outer);
    assert(!makePairedSkinBoundaryNodeMap(std::numeric_limits<std::uint32_t>::max(),m,e));
    m.volumeNodeCount=20;assert(!isPairedSkinBoundaryNodeMap(m));
    m.outer={19,1,17,3};m.skinInner={10,11,12,13};m.fatDeep={4,6,8,9};
    assert(validateSkinBoundaryNodeMap(m,4,e));
    m.fatDeep[1]=19;assert(!validateSkinBoundaryNodeMap(m,4,e));
    m.fatDeep[1]=20;assert(!validateSkinBoundaryNodeMap(m,4,e));
    m.fatDeep[1]=6;assert(!validateSkinBoundaryNodeMap(m,5,e));
    m.outer.clear();assert(!validateSkinBoundaryNodeMap(m,4,e));
    std::cout<<"paired parity, arbitrary volume rows, disjoint roles, bounds and overflow: PASS\n";
}

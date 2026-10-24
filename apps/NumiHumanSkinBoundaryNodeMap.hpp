#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <string>
#include <vector>

namespace numi::human {
// Local FEM node rows for the registered surface, independently of interior
// meshing. Interior nodes retain their physical state without invented surface IDs.
struct SkinBoundaryNodeMap {
    std::uint32_t volumeNodeCount=0;
    std::vector<std::uint32_t> outer, skinInner, fatDeep;
};
inline bool validateSkinBoundaryNodeMap(const SkinBoundaryNodeMap& map,
        std::size_t surfaceRows, std::string& error) {
    const auto fail=[&](const char* reason){ error=reason; return false; };
    if(surfaceRows==0 || map.volumeNodeCount==0 ||
       map.outer.size()!=surfaceRows || map.skinInner.size()!=surfaceRows ||
       map.fatDeep.size()!=surfaceRows)
        return fail("skin boundary maps do not cover the registered surface");
    std::vector<std::uint8_t> used(map.volumeNodeCount,0u);
    for(const auto* rows:{&map.outer,&map.skinInner,&map.fatDeep})
        for(const auto node:*rows) {
            if(node>=map.volumeNodeCount) return fail("skin boundary node exceeds the volume topology");
            if(used[node]) return fail("skin boundary roles alias a volume node");
            used[node]=1u;
        }
    error.clear(); return true;
}
inline bool makePairedSkinBoundaryNodeMap(std::size_t surfaceRows,
        SkinBoundaryNodeMap& map,std::string& error) {
    if(surfaceRows==0 || surfaceRows>std::numeric_limits<std::uint32_t>::max()/3u) {
        error="paired skin boundary count is empty or overflows"; return false;
    }
    SkinBoundaryNodeMap next;
    next.volumeNodeCount=static_cast<std::uint32_t>(3u*surfaceRows);
    next.outer.resize(surfaceRows);next.skinInner.resize(surfaceRows);next.fatDeep.resize(surfaceRows);
    for(std::size_t i=0;i<surfaceRows;++i) {
        next.outer[i]=static_cast<std::uint32_t>(i);
        next.skinInner[i]=static_cast<std::uint32_t>(surfaceRows+i);
        next.fatDeep[i]=static_cast<std::uint32_t>(2u*surfaceRows+i);
    }
    map=std::move(next);error.clear();return true;
}
inline bool skinBoundaryNodeRangeFits(const SkinBoundaryNodeMap& map,
        std::size_t firstGlobalNode,std::size_t snapshotNodeCount) {
    return firstGlobalNode<=snapshotNodeCount &&
        map.volumeNodeCount<=snapshotNodeCount-firstGlobalNode;
}
} // namespace numi::human

#pragma once
#include "NumiHumanInitialSkinBoneContactWitness.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <tuple>

namespace numi::human::d37_initial_skin_bone_witness {
using Triangle = NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU;
using Node = NMNumiHumanFEMBodyContactBVHNodeGPU;
namespace witness = numi::human::initial_skin_bone_contact_witness;
using Vec3 = witness::Vec3;
using Closest = witness::Closest;

struct Nearest {
    bool valid=false;
    double distanceSquared=std::numeric_limits<double>::infinity();
    double signedFaceDistance=std::numeric_limits<double>::quiet_NaN();
    double signedPseudoDistance=std::numeric_limits<double>::quiet_NaN();
    Vec3 query{}, point{}, faceNormal{}, pseudoNormal{};
    Closest closest{};
    std::uint32_t body=UINT32_MAX, bone=UINT32_MAX, triangle=UINT32_MAX;
};

inline double pointAabbDistanceSquared(Vec3 p,const Node& node) noexcept {
    const double dx=std::max({double(node.boundsMinimum.x)-p.x,
        p.x-double(node.boundsMaximum.x),0.0});
    const double dy=std::max({double(node.boundsMinimum.y)-p.y,
        p.y-double(node.boundsMaximum.y),0.0});
    const double dz=std::max({double(node.boundsMinimum.z)-p.z,
        p.z-double(node.boundsMaximum.z),0.0});
    return dx*dx+dy*dy+dz*dz;
}

// Stackless traversal follows the static asset's preorder/escapeIndex contract.
// It visits only nodes whose AABB can beat the current exact closest distance.
inline Nearest nearestPointOnBodyTree(
    Vec3 query,std::uint32_t expectedBody,
    std::span<const Triangle> triangles,std::span<const Node> nodes,
    std::uint32_t firstNode,std::uint32_t nodeCount
) noexcept {
    Nearest best{};best.query=query;best.body=expectedBody;
    if(!witness::finite(query)||nodeCount==0u||firstNode>nodes.size()||
       nodeCount>nodes.size()-firstNode)return best;
    const std::uint32_t end=firstNode+nodeCount;
    std::uint32_t cursor=firstNode;
    while(cursor<end) {
        const auto& node=nodes[cursor];
        if(node.escapeIndex<=cursor||node.escapeIndex>end)return Nearest{};
        const double lower=pointAabbDistanceSquared(query,node);
        if(!std::isfinite(lower))return Nearest{};
        if(lower>best.distanceSquared) {cursor=node.escapeIndex;continue;}
        if((node.flags&NM_NUMI_HUMAN_FEM_BODY_CONTACT_BVH_NODE_LEAF)!=0u) {
            if(node.triangleIndex>=triangles.size())return Nearest{};
            const auto& tri=triangles[node.triangleIndex];
            if(tri.bodyIndex!=expectedBody)return Nearest{};
            const Vec3 a=witness::read3(tri.bodyLocalVertex0);
            const Vec3 b=witness::read3(tri.bodyLocalVertex1);
            const Vec3 c=witness::read3(tri.bodyLocalVertex2);
            const auto cp=witness::closestPointOnTriangle(query,a,b,c);
            if(!cp.valid)return Nearest{};
            const Vec3 delta=witness::sub(query,cp.point);
            const double d2=witness::norm2(delta);
            const Vec3 raw=witness::cross(witness::sub(b,a),witness::sub(c,a));
            const double rawLength=witness::norm(raw);
            Vec3 pseudo{};
            if(!std::isfinite(d2)||d2<0.0||!std::isfinite(rawLength)||!(rawLength>0.0)||
               !witness::featurePseudoNormal(tri,cp,pseudo))return Nearest{};
            const Vec3 face=witness::mul(raw,1.0/rawLength);
            const auto identity=std::pair{tri.boneStableId,tri.triangleStableId};
            const auto current=std::pair{best.bone,best.triangle};
            if(d2<best.distanceSquared||(d2==best.distanceSquared&&identity<current)) {
                best.valid=true;best.distanceSquared=d2;best.closest=cp;best.point=cp.point;
                best.faceNormal=face;best.pseudoNormal=pseudo;
                best.signedFaceDistance=witness::dot(delta,face);
                best.signedPseudoDistance=witness::dot(delta,pseudo);
                best.bone=tri.boneStableId;best.triangle=tri.triangleStableId;
            }
        }
        ++cursor;
    }
    return best;
}
} // namespace numi::human::d37_initial_skin_bone_witness

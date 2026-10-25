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
    std::uint32_t firstNode,std::uint32_t nodeCount,
    std::uint32_t expectedBone=UINT32_MAX
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
            if(expectedBone!=UINT32_MAX&&tri.boneStableId!=expectedBone) {++cursor;continue;}
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

// AABB culling is a proof of outside for a closed finite bank: its enclosed
// solid lies within the axis-aligned bounds of the bank's triangle vertices.
struct BankAabb {
    Vec3 minimum{},maximum{};
    bool valid=false;
};

inline BankAabb exactTriangleBankAabb(std::span<const Triangle> triangles) noexcept {
    BankAabb result{};
    if(triangles.empty())return result;
    const double inf=std::numeric_limits<double>::infinity();
    result.minimum={inf,inf,inf};result.maximum={-inf,-inf,-inf};
    for(const auto& triangle:triangles) {
        const std::array<Vec3,3> points{{witness::read3(triangle.bodyLocalVertex0),
            witness::read3(triangle.bodyLocalVertex1),witness::read3(triangle.bodyLocalVertex2)}};
        for(const auto point:points) {
            if(!finite(point))return {};
            result.minimum.x=std::min(result.minimum.x,point.x);
            result.minimum.y=std::min(result.minimum.y,point.y);
            result.minimum.z=std::min(result.minimum.z,point.z);
            result.maximum.x=std::max(result.maximum.x,point.x);
            result.maximum.y=std::max(result.maximum.y,point.y);
            result.maximum.z=std::max(result.maximum.z,point.z);
        }
    }
    result.valid=std::isfinite(result.minimum.x)&&std::isfinite(result.minimum.y)&&
        std::isfinite(result.minimum.z)&&std::isfinite(result.maximum.x)&&
        std::isfinite(result.maximum.y)&&std::isfinite(result.maximum.z)&&
        result.minimum.x<=result.maximum.x&&result.minimum.y<=result.maximum.y&&
        result.minimum.z<=result.maximum.z;
    return result;
}

inline bool bankAabbContains(const BankAabb& bounds,Vec3 point) noexcept {
    return bounds.valid&&finite(point)&&
        point.x>=bounds.minimum.x&&point.x<=bounds.maximum.x&&
        point.y>=bounds.minimum.y&&point.y<=bounds.maximum.y&&
        point.z>=bounds.minimum.z&&point.z<=bounds.maximum.z;
}

enum class BankContainmentState : std::uint8_t {
    aabbProvenOutside,
    parityOutside,
    parityInside,
    ambiguous,
    invalid
};

struct BankContainment {
    BankContainmentState state=BankContainmentState::invalid;
    BankAabb bounds{};
    witness::SolidParity parity{};
};

inline BankContainment classifyClosedBank(Vec3 point,std::span<const Triangle> triangles,
                                          const BankAabb& knownBounds) noexcept {
    BankContainment result{};
    if(!finite(point))return result;
    result.bounds=knownBounds;
    if(!result.bounds.valid)return result;
    if(!bankAabbContains(result.bounds,point)) {
        result.state=BankContainmentState::aabbProvenOutside;
        return result;
    }
    result.parity=classifyClosedSolid(point,triangles);
    if(result.parity.ambiguous||
       result.parity.insideRays+result.parity.outsideRays!=3||
       (result.parity.insideRays!=0&&result.parity.outsideRays!=0)) {
        result.state=BankContainmentState::ambiguous;
    } else if(result.parity.insideRays==3&&result.parity.outsideRays==0) {
        result.state=BankContainmentState::parityInside;
    } else if(result.parity.outsideRays==3&&result.parity.insideRays==0) {
        result.state=BankContainmentState::parityOutside;
    }
    return result;
}

inline BankContainment classifyClosedBank(Vec3 point,std::span<const Triangle> triangles) noexcept {
    return classifyClosedBank(point,triangles,exactTriangleBankAabb(triangles));
}

struct AllBankContainmentSummary {
    bool valid=false;
    std::string failure;
    std::size_t nodes=0u,banks=0u,aabbProvenOutsidePairs=0u,
        parityOutsidePairs=0u,insidePairs=0u,ambiguousPairs=0u,invalidPairs=0u;
    double deepestInsideDistanceM=0.0;
    std::uint32_t deepestInsideNode=UINT32_MAX,deepestInsideBody=UINT32_MAX,
        deepestInsideBone=UINT32_MAX,deepestInsideTriangle=UINT32_MAX;
};

// Writes one row for every node / closed bone-bank pair whose exact per-bone
// AABB contains the point. Body banks may contain several bones, so parity is
// never evaluated over the body-wide union (nested closed surfaces could cancel).
// Pairs outside a per-bone AABB are counted as a geometric outside proof.
template<class NodeContainer,class BodyContainer,class TreeContainer,
         class TriangleContainer,class BvhNodeContainer,class OffsetFn>
AllBankContainmentSummary writeAllBankContainment(
    std::ostream& out,
    const NodeContainer& nodes,
    const BodyContainer& bodies,
    const TreeContainer& trees,
    const TriangleContainer& triangles,
    const BvhNodeContainer& bvhNodes,
    std::span<const std::uint8_t> roleMask,
    std::span<const std::uint32_t> roleRow,
    std::uint32_t objectStateOffset,
    std::uint32_t volumeNodeCount,
    OffsetFn offset,
    std::ostream* nodeSummaryOut=nullptr
) {
    struct BoneBank {
        std::uint32_t body=UINT32_MAX,bone=UINT32_MAX;
        std::uint32_t firstNode=0u,nodeCount=0u,firstTriangle=0u,triangleCount=0u;
        BankAabb bounds{};
    };
    AllBankContainmentSummary result{};
    if(roleMask.size()!=volumeNodeCount||roleRow.size()!=volumeNodeCount||
       std::uint64_t(objectStateOffset)+volumeNodeCount>nodes.size()||bodies.empty()||trees.empty()) {
        result.failure="all_bank_input_extent_invalid";return result;
    }
    const std::span<const Triangle> allTriangles(triangles.data(),triangles.size());
    const std::span<const Node> allBvhNodes(bvhNodes.data(),bvhNodes.size());
    std::vector<std::uint8_t> seenBody(bodies.size(),0u);
    std::vector<BoneBank> banks;
    std::size_t expectedTriangle=0u;
    for(const auto& tree:trees) {
        if(tree.body>=bodies.size()||seenBody[tree.body]++||tree.triangleCount==0u||
           tree.firstTriangle!=expectedTriangle||
           std::uint64_t(tree.firstTriangle)+tree.triangleCount>triangles.size()||
           tree.firstNode>=bvhNodes.size()||tree.nodeCount==0u||
           std::uint64_t(tree.firstNode)+tree.nodeCount>bvhNodes.size()||
           bvhNodes[tree.firstNode].escapeIndex!=tree.firstNode+tree.nodeCount) {
            result.failure="all_bank_tree_partition_invalid";return result;
        }
        const std::size_t treeEnd=std::size_t(tree.firstTriangle)+tree.triangleCount;
        std::size_t first=tree.firstTriangle;
        while(first<treeEnd) {
            const std::uint32_t bone=triangles[first].boneStableId;
            if(bone==UINT32_MAX) {result.failure="all_bank_invalid_bone_identity";return result;}
            std::size_t end=first+1u;
            while(end<treeEnd&&triangles[end].boneStableId==bone)++end;
            for(std::size_t i=first;i<end;++i)
                if(triangles[i].bodyIndex!=tree.body||triangles[i].boneStableId!=bone) {
                    result.failure="all_bank_bone_partition_invalid";return result;
                }
            const auto boneTriangles=allTriangles.subspan(first,end-first);
            const auto bounds=exactTriangleBankAabb(boneTriangles);
            if(!bounds.valid) {result.failure="all_bank_triangle_bounds_invalid";return result;}
            banks.push_back({tree.body,bone,tree.firstNode,tree.nodeCount,
                static_cast<std::uint32_t>(first),static_cast<std::uint32_t>(end-first),bounds});
            first=end;
        }
        expectedTriangle=treeEnd;
    }
    if(expectedTriangle!=triangles.size()||banks.empty()) {
        result.failure="all_bank_trees_do_not_cover_complete_triangle_asset";return result;
    }
    out<<"local_node,global_node,layer,row,role_mask,bank_body,bank_stable_id,classification,inside_rays,outside_rays,ambiguous,nearest_bone,nearest_triangle,nearest_distance_m,signed_face_distance_m,query_local_x,query_local_y,query_local_z,closest_x,closest_y,closest_z\n";
    if(nodeSummaryOut) {
        *nodeSummaryOut<<"local_node,global_node,layer,row,role_mask,total_bone_banks,aabb_proven_outside_banks,parity_outside_banks,inside_banks,ambiguous_banks,invalid_banks,parity_query_banks\n";
        *nodeSummaryOut<<std::setprecision(17);
    }
    out<<std::setprecision(17);
    result.nodes=volumeNodeCount;result.banks=banks.size();
    for(std::uint32_t local=0u;local<volumeNodeCount;++local) {
        const std::uint32_t global=objectStateOffset+local;
        const Vec3 world=witness::read3(nodes[global].positionAndMass);
        if(!finite(world)) {result.failure="all_bank_world_node_nonfinite";return result;}
        std::size_t nodeAabbOutside=0u,nodeParityOutside=0u,nodeInside=0u,
            nodeAmbiguous=0u,nodeInvalid=0u,nodeParityQueries=0u;
        for(const auto& bank:banks) {
            const auto& body=bodies[bank.body];
            const std::array<double,4> inverseQ{{-body.orientation.x,-body.orientation.y,
                -body.orientation.z,body.orientation.w}};
            Vec3 query=rotate(inverseQ,sub(world,Vec3{body.position.x,body.position.y,body.position.z}));
            const auto bodyOffset=offset(bank.body);
            query={query.x-double(bodyOffset[0]),query.y-double(bodyOffset[1]),
                query.z-double(bodyOffset[2])};
            if(!finite(query)) {result.failure="all_bank_local_query_nonfinite";return result;}
            if(!bankAabbContains(bank.bounds,query)) {
                ++result.aabbProvenOutsidePairs;++nodeAabbOutside;continue;
            }
            ++nodeParityQueries;
            const auto boneTriangles=allTriangles.subspan(bank.firstTriangle,bank.triangleCount);
            const auto containment=classifyClosedBank(query,boneTriangles,bank.bounds);
            if(containment.state==BankContainmentState::invalid||
               containment.state==BankContainmentState::aabbProvenOutside) {
                ++result.invalidPairs;++nodeInvalid;continue;
            }
            const auto closest=nearestPointOnBodyTree(query,bank.body,allTriangles,allBvhNodes,
                bank.firstNode,bank.nodeCount,bank.bone);
            if(!closest.valid) {++result.invalidPairs;++nodeInvalid;continue;}
            const double distance=std::sqrt(closest.distanceSquared);
            const char* label="ambiguous";
            switch(containment.state) {
                case BankContainmentState::aabbProvenOutside: label="aabb_proven_outside";break;
                case BankContainmentState::parityOutside:
                    label="outside";++result.parityOutsidePairs;++nodeParityOutside;break;
                case BankContainmentState::parityInside:
                    label="inside";++result.insidePairs;++nodeInside;
                    if(distance>result.deepestInsideDistanceM) {
                        result.deepestInsideDistanceM=distance;result.deepestInsideNode=global;
                        result.deepestInsideBody=bank.body;result.deepestInsideBone=closest.bone;
                        result.deepestInsideTriangle=closest.triangle;
                    }
                    break;
                case BankContainmentState::ambiguous:
                    label="ambiguous";++result.ambiguousPairs;++nodeAmbiguous;break;
                case BankContainmentState::invalid:
                    ++result.invalidPairs;++nodeInvalid;continue;
            }
            const std::uint32_t layer=roleMask[local]==1u?0u:roleMask[local]==2u?1u:
                roleMask[local]==4u?2u:roleMask[local]==0u?3u:4u;
            const std::uint32_t row=roleMask[local]!=0u&&roleRow[local]!=UINT32_MAX
                ?roleRow[local]:local;
            out<<local<<","<<global<<","<<layer<<","<<row<<","<<unsigned(roleMask[local])<<","<<
                bank.body<<","<<bank.bone<<","<<label<<","<<containment.parity.insideRays<<","<<
                containment.parity.outsideRays<<","<<(containment.parity.ambiguous?1:0)<<","<<
                closest.bone<<","<<closest.triangle<<","<<distance<<","<<
                closest.signedFaceDistance<<","<<query.x<<","<<query.y<<","<<query.z<<","<<
                closest.point.x<<","<<closest.point.y<<","<<closest.point.z<<"\n";
            if(!out.good()) {result.failure="all_bank_output_stream_failed";return result;}
        }
        if(nodeSummaryOut) {
            const std::uint32_t layer=roleMask[local]==1u?0u:roleMask[local]==2u?1u:
                roleMask[local]==4u?2u:roleMask[local]==0u?3u:4u;
            const std::uint32_t row=roleMask[local]!=0u&&roleRow[local]!=UINT32_MAX
                ?roleRow[local]:local;
            *nodeSummaryOut<<local<<","<<global<<","<<layer<<","<<row<<","<<
                unsigned(roleMask[local])<<","<<banks.size()<<","<<nodeAabbOutside<<","<<
                nodeParityOutside<<","<<nodeInside<<","<<nodeAmbiguous<<","<<nodeInvalid<<","<<
                nodeParityQueries<<"\n";
            if(!nodeSummaryOut->good()) {
                result.failure="all_bank_node_summary_stream_failed";return result;
            }
        }
    }
    if(result.invalidPairs!=0u) {result.failure="all_bank_query_invalid_pairs";return result;}
    result.valid=true;return result;
}

} // namespace numi::human::d37_initial_skin_bone_witness

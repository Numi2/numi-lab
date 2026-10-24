#pragma once

#include "numi/matter/numi_human_shared.h"
#include "numi/matter/fem_body_triangle_contact_sample.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <tuple>
#include <vector>

namespace numi::matter::fem_body_contact_bvh {

inline constexpr std::uint32_t kInvalidIndex = UINT32_MAX;
inline constexpr std::uint32_t kRangeFlagStaticCompleteBVH = 2u;
inline constexpr std::uint32_t kRangeFlagStaticClosedOrientedBVH = 3u;

// Conservative CPU mirror of the shader fast-path predicate. A true result is
// a lower-bound proof that an exterior query is farther than the contact band.
inline bool closedAABBProvesInactive(
    nm_float4 query,nm_float4 minimum,nm_float4 maximum,float maximumDistance
) noexcept {
    const auto finite=[](float x){return std::isfinite(x);};
    if (!finite(query.x)||!finite(query.y)||!finite(query.z)||query.w!=0.0f||
        !finite(minimum.x)||!finite(minimum.y)||!finite(minimum.z)||minimum.w!=0.0f||
        !finite(maximum.x)||!finite(maximum.y)||!finite(maximum.z)||maximum.w!=0.0f||
        minimum.x>maximum.x||minimum.y>maximum.y||minimum.z>maximum.z||
        !finite(maximumDistance)||!(maximumDistance>0.0f)) return false;
    const float dx=std::max({minimum.x-query.x,query.x-maximum.x,0.0f});
    const float dy=std::max({minimum.y-query.y,query.y-maximum.y,0.0f});
    const float dz=std::max({minimum.z-query.z,query.z-maximum.z,0.0f});
    const float distanceSquared=dx*dx+dy*dy+dz*dz;
    const float querySquared=query.x*query.x+query.y*query.y+query.z*query.z;
    const float coordinateScale=std::max(1.0f,querySquared);
    const float roundingGuard=32.0f*std::numeric_limits<float>::epsilon()*coordinateScale;
    const float bandSquared=maximumDistance*maximumDistance;
    return finite(distanceSquared)&&finite(roundingGuard)&&
        distanceSquared>bandSquared+roundingGuard;
}

struct BodyRange {
    std::uint32_t bodyIndex = kInvalidIndex;
    std::uint32_t firstNode = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t firstTriangle = 0u;
    std::uint32_t triangleCount = 0u;
    // Set only after exact-coordinate closed/oriented bank validation.
    bool closedOrientedCertified = false;
};

enum class BuildFailure : std::uint8_t {
    none, emptyInventory, inventoryCountMismatch, flattenedIndexOverflow,
    invalidTriangle, duplicateStableIdentity, nodeIndexOverflow,
    invalidContactSamples, invalidClosedOrientedBank,
};

struct BuildResult {
    BuildFailure failure = BuildFailure::none;
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> triangles;
    std::vector<NMNumiHumanFEMBodyContactBVHNodeGPU> nodes;
    std::vector<BodyRange> bodies;
    [[nodiscard]] bool succeeded() const noexcept {
        return failure == BuildFailure::none;
    }
};

inline bool finite4(nm_float4 v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) &&
        std::isfinite(v.z) && std::isfinite(v.w);
}
inline double dot3(nm_float4 a,nm_float4 b) noexcept {
    return double(a.x)*b.x+double(a.y)*b.y+double(a.z)*b.z;
}
inline bool validTriangle(
    const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU& t
) noexcept {
    constexpr std::uint32_t featureMask =
        NM_NUMI_HUMAN_FEM_BODY_CONTACT_ALL_FEATURE_PSEUDONORMALS;
    const std::array<nm_float4,6> featureNormals{{
        t.edgePseudoNormal01,t.edgePseudoNormal12,t.edgePseudoNormal20,
        t.vertexPseudoNormal0,t.vertexPseudoNormal1,t.vertexPseudoNormal2}};
    if (t.bodyIndex==kInvalidIndex || t.boneStableId==kInvalidIndex ||
        t.triangleStableId==kInvalidIndex ||
        (t.featureNormalFlags & ~featureMask)!=0u ||
        !finite4(t.bodyLocalVertex0) || !finite4(t.bodyLocalVertex1) ||
        !finite4(t.bodyLocalVertex2) || !finite4(t.authoredOutwardNormal) ||
        t.bodyLocalVertex0.w!=0.0f || t.bodyLocalVertex1.w!=0.0f ||
        t.bodyLocalVertex2.w!=0.0f || t.authoredOutwardNormal.w!=0.0f)
        return false;
    for (std::size_t i=0;i<featureNormals.size();++i) {
        const auto v=featureNormals[i];
        if (!finite4(v) || v.w!=0.0f) return false;
        if ((t.featureNormalFlags & (1u<<i))!=0u) {
            const double length=std::sqrt(dot3(v,v));
            if (!std::isfinite(length) || std::abs(length-1.0)>1.0e-4)
                return false;
        } else if (v.x!=0.0f || v.y!=0.0f || v.z!=0.0f) {
            return false;
        }
    }
    const auto a=t.bodyLocalVertex0,b=t.bodyLocalVertex1,c=t.bodyLocalVertex2;
    const double abx=double(b.x)-a.x, aby=double(b.y)-a.y, abz=double(b.z)-a.z;
    const double acx=double(c.x)-a.x, acy=double(c.y)-a.y, acz=double(c.z)-a.z;
    const double nx=aby*acz-abz*acy, ny=abz*acx-abx*acz,
                 nz=abx*acy-aby*acx;
    const double nlen=std::sqrt(nx*nx+ny*ny+nz*nz);
    const double authored=std::sqrt(dot3(t.authoredOutwardNormal,
                                          t.authoredOutwardNormal));
    if (!std::isfinite(nlen) || !(nlen>0.0) ||
        !std::isfinite(authored) || std::abs(authored-1.0)>1.0e-4)
        return false;
    const double alignment=(nx*t.authoredOutwardNormal.x+
        ny*t.authoredOutwardNormal.y+nz*t.authoredOutwardNormal.z)/(nlen*authored);
    return std::isfinite(alignment) && alignment>=0.999;
}

struct ExactPointKey {
    std::array<std::uint32_t,3> bits{};
    friend bool operator<(const ExactPointKey& a,const ExactPointKey& b) noexcept {
        return a.bits<b.bits;
    }
    friend bool operator==(const ExactPointKey& a,const ExactPointKey& b) noexcept {
        return a.bits==b.bits;
    }
};
struct ExactEdgeKey {
    ExactPointKey first{},second{};
    friend bool operator<(const ExactEdgeKey& a,const ExactEdgeKey& b) noexcept {
        return std::tie(a.first.bits,a.second.bits)<
               std::tie(b.first.bits,b.second.bits);
    }
};
struct DVec3 { double x=0.0,y=0.0,z=0.0; };
inline DVec3 add(DVec3 a,DVec3 b) noexcept { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline DVec3 scale(DVec3 a,double s) noexcept { return {a.x*s,a.y*s,a.z*s}; }
inline DVec3 sub(DVec3 a,DVec3 b) noexcept { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline double dot(DVec3 a,DVec3 b) noexcept { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline DVec3 cross(DVec3 a,DVec3 b) noexcept {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
inline double length(DVec3 a) noexcept { return std::sqrt(dot(a,a)); }
inline bool normalize(DVec3 v,DVec3& out) noexcept {
    const double n=length(v);
    if (!std::isfinite(n) || !(n>0.0)) return false;
    out=scale(v,1.0/n);
    return std::isfinite(out.x)&&std::isfinite(out.y)&&std::isfinite(out.z);
}
inline DVec3 point(nm_float4 v) noexcept { return {v.x,v.y,v.z}; }
inline ExactPointKey exactPointKey(nm_float4 p) noexcept {
    const auto bits=[](float v) {
        return v==0.0f ? 0u : std::bit_cast<std::uint32_t>(v);
    };
    return {{{bits(p.x),bits(p.y),bits(p.z)}}};
}
inline ExactEdgeKey exactEdgeKey(ExactPointKey a,ExactPointKey b) noexcept {
    return b<a ? ExactEdgeKey{b,a} : ExactEdgeKey{a,b};
}
struct EdgeIncident { std::size_t triangle=0u; unsigned edge=0u; bool forward=false; };
struct FaceFeatureData {
    std::array<ExactPointKey,3> vertices{};
    DVec3 normal{};
    std::array<double,3> cornerAngles{};
};

// Compute the static signed-distance pseudonormals from exact F32 coordinate
// identity. This deliberately does not repair face winding or weld by a
// tolerance: an open, non-manifold, pinched, inconsistently wound, or
// inward-oriented bank is rejected when requireClosedOriented is true.
inline bool buildExactFeaturePseudonormals(
    std::span<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> triangles,
    bool requireClosedOriented
) {
    std::size_t bankBegin=0u;
    while (bankBegin<triangles.size()) {
        std::size_t bankEnd=bankBegin+1u;
        const auto body=triangles[bankBegin].bodyIndex;
        const auto bone=triangles[bankBegin].boneStableId;
        while (bankEnd<triangles.size() && triangles[bankEnd].bodyIndex==body &&
               triangles[bankEnd].boneStableId==bone) ++bankEnd;
        if (!requireClosedOriented) { bankBegin=bankEnd; continue; }
        const std::size_t count=bankEnd-bankBegin;
        std::vector<FaceFeatureData> faces(count);
        std::map<ExactEdgeKey,std::vector<EdgeIncident>> edges;
        std::map<ExactPointKey,DVec3> vertexSums;
        std::map<ExactPointKey,std::map<ExactPointKey,std::set<ExactPointKey>>> links;
        double signedVolume=0.0;
        for (std::size_t local=0;local<count;++local) {
            const std::size_t ti=bankBegin+local;
            auto& t=triangles[ti];
            const std::array<nm_float4,3> p{{
                t.bodyLocalVertex0,t.bodyLocalVertex1,t.bodyLocalVertex2}};
            auto& f=faces[local];
            for (unsigned k=0;k<3u;++k) f.vertices[k]=exactPointKey(p[k]);
            if (f.vertices[0]==f.vertices[1] || f.vertices[1]==f.vertices[2] ||
                f.vertices[2]==f.vertices[0]) return false;
            const DVec3 a=point(p[0]),b=point(p[1]),c=point(p[2]);
            const DVec3 ab=sub(b,a),ac=sub(c,a);
            if (!normalize(cross(ab,ac),f.normal)) return false;
            const DVec3 authored=point(t.authoredOutwardNormal);
            if (dot(f.normal,authored)<0.999) return false;
            signedVolume+=dot(a,cross(b,c))/6.0;
            for (unsigned k=0;k<3u;++k) {
                const unsigned j=(k+1u)%3u,l=(k+2u)%3u;
                const DVec3 u=sub(point(p[j]),point(p[k]));
                const DVec3 v=sub(point(p[l]),point(p[k]));
                const double angle=std::atan2(length(cross(u,v)),dot(u,v));
                if (!std::isfinite(angle) || !(angle>0.0)) return false;
                f.cornerAngles[k]=angle;
                vertexSums[f.vertices[k]]=add(vertexSums[f.vertices[k]],
                                              scale(f.normal,angle));
                links[f.vertices[k]][f.vertices[j]].insert(f.vertices[l]);
                links[f.vertices[k]][f.vertices[l]].insert(f.vertices[j]);
            }
            constexpr unsigned edgeVertices[3][2]={{0u,1u},{1u,2u},{2u,0u}};
            for (unsigned e=0;e<3u;++e) {
                const auto x=f.vertices[edgeVertices[e][0]];
                const auto y=f.vertices[edgeVertices[e][1]];
                const auto key=exactEdgeKey(x,y);
                edges[key].push_back({ti,e,x<y});
            }
        }
        if (!std::isfinite(signedVolume) || !(signedVolume>0.0)) return false;
        for (const auto& [edge,incident]:edges) {
            (void)edge;
            if (incident.size()!=2u || incident[0].forward==incident[1].forward)
                return false;
        }
        for (const auto& [vertex,adjacency]:links) {
            if (adjacency.size()<3u) return false;
            for (const auto& [neighbor,ring]:adjacency) {
                (void)neighbor;
                if (ring.size()!=2u) return false;
            }
            std::set<ExactPointKey> visited;
            std::vector<ExactPointKey> stack{adjacency.begin()->first};
            while (!stack.empty()) {
                const auto current=stack.back(); stack.pop_back();
                if (!visited.insert(current).second) continue;
                for (const auto& next:adjacency.at(current)) stack.push_back(next);
            }
            if (visited.size()!=adjacency.size()) return false;
            DVec3 pseudo{};
            if (!normalize(vertexSums.at(vertex),pseudo)) return false;
            (void)pseudo;
        }
        // All banks pass their closure/orientation checks before any fields are
        // emitted; caller discards the whole candidate on false.
        for (const auto& [key,incident]:edges) {
            DVec3 pseudo{};
            if (!normalize(add(faces[incident[0].triangle-bankBegin].normal,
                               faces[incident[1].triangle-bankBegin].normal),pseudo))
                return false;
            for (const auto& side:incident) {
                auto& t=triangles[side.triangle];
                const nm_float4 packed{float(pseudo.x),float(pseudo.y),float(pseudo.z),0.0f};
                if (side.edge==0u) t.edgePseudoNormal01=packed;
                else if (side.edge==1u) t.edgePseudoNormal12=packed;
                else t.edgePseudoNormal20=packed;
                t.featureNormalFlags|=1u<<side.edge;
            }
        }
        std::map<ExactPointKey,DVec3> vertexPseudo;
        for (const auto& [key,sum]:vertexSums) {
            DVec3 pseudo{};
            if (!normalize(sum,pseudo)) return false;
            vertexPseudo.emplace(key,pseudo);
        }
        for (std::size_t local=0;local<count;++local) {
            auto& t=triangles[bankBegin+local];
            const auto& f=faces[local];
            for (unsigned k=0;k<3u;++k) {
                const auto pseudo=vertexPseudo.at(f.vertices[k]);
                const nm_float4 packed{float(pseudo.x),float(pseudo.y),float(pseudo.z),0.0f};
                if (k==0u) t.vertexPseudoNormal0=packed;
                else if (k==1u) t.vertexPseudoNormal1=packed;
                else t.vertexPseudoNormal2=packed;
                t.featureNormalFlags|=1u<<(k+3u);
            }
        }
        bankBegin=bankEnd;
    }
    return true;
}

inline BuildResult buildStaticCompleteBodyBVH(
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> input,
    std::size_t expectedTriangleCount,
    bool requireClosedOrientedBanks = false
) {
    BuildResult result{};
    const auto fail=[&](BuildFailure failure) {
        result.failure=failure;
        result.triangles.clear(); result.nodes.clear(); result.bodies.clear();
        return result;
    };
    if (input.empty() || expectedTriangleCount==0u)
        return fail(BuildFailure::emptyInventory);
    if (input.size()!=expectedTriangleCount)
        return fail(BuildFailure::inventoryCountMismatch);
    if (input.size()>std::numeric_limits<std::uint32_t>::max())
        return fail(BuildFailure::flattenedIndexOverflow);
    if (input.size()>(std::numeric_limits<std::uint32_t>::max()-1ull)/2ull)
        return fail(BuildFailure::nodeIndexOverflow);
    for (const auto& t:input)
        if (!validTriangle(t)) return fail(BuildFailure::invalidTriangle);

    result.triangles.assign(input.begin(),input.end());
    std::sort(result.triangles.begin(),result.triangles.end(),
        [](const auto& a,const auto& b) {
            return std::tie(a.bodyIndex,a.boneStableId,a.triangleStableId)<
                   std::tie(b.bodyIndex,b.boneStableId,b.triangleStableId);
        });
    for (std::size_t i=1;i<result.triangles.size();++i) {
        const auto& a=result.triangles[i-1u];
        const auto& b=result.triangles[i];
        if (a.bodyIndex==b.bodyIndex && a.boneStableId==b.boneStableId &&
            a.triangleStableId==b.triangleStableId)
            return fail(BuildFailure::duplicateStableIdentity);
    }
    if (!buildExactFeaturePseudonormals(result.triangles,requireClosedOrientedBanks))
        return fail(BuildFailure::invalidClosedOrientedBank);
    if (result.triangles.size()>(std::numeric_limits<std::uint32_t>::max()+1ull)/2ull)
        return fail(BuildFailure::nodeIndexOverflow);
    result.nodes.reserve(result.triangles.size()*2u-result.bodies.size());
    std::size_t firstTriangle=0u;
    while (firstTriangle<result.triangles.size()) {
        std::size_t endTriangle=firstTriangle+1u;
        const std::uint32_t body=result.triangles[firstTriangle].bodyIndex;
        while (endTriangle<result.triangles.size() &&
               result.triangles[endTriangle].bodyIndex==body) ++endTriangle;
        if (result.nodes.size()>std::numeric_limits<std::uint32_t>::max())
            return fail(BuildFailure::nodeIndexOverflow);
        BodyRange bodyRange{};
        bodyRange.bodyIndex=body;
        bodyRange.firstNode=static_cast<std::uint32_t>(result.nodes.size());
        bodyRange.firstTriangle=static_cast<std::uint32_t>(firstTriangle);
        bodyRange.triangleCount=static_cast<std::uint32_t>(endTriangle-firstTriangle);
        bodyRange.closedOrientedCertified=requireClosedOrientedBanks;
        std::vector<std::uint32_t> ids(bodyRange.triangleCount);
        for (std::uint32_t i=0;i<bodyRange.triangleCount;++i)
            ids[i]=bodyRange.firstTriangle+i;
        const auto coord=[](nm_float4 p,unsigned axis) -> double {
            return axis==0u ? double(p.x) : (axis==1u ? double(p.y) : double(p.z));
        };
        const auto centroid=[&](std::uint32_t tri,unsigned axis) {
            const auto& t=result.triangles[tri];
            return (coord(t.bodyLocalVertex0,axis)+coord(t.bodyLocalVertex1,axis)+
                    coord(t.bodyLocalVertex2,axis))/3.0;
        };
        bool overflow=false;
        const auto build=[&](auto&& self,std::size_t begin,std::size_t finish)->void {
            const std::size_t nodeIndex=result.nodes.size();
            if (nodeIndex>=std::numeric_limits<std::uint32_t>::max()) {
                overflow=true; return;
            }
            NMNumiHumanFEMBodyContactBVHNodeGPU node{};
            std::array<double,3> lo{{INFINITY,INFINITY,INFINITY}};
            std::array<double,3> hi{{-INFINITY,-INFINITY,-INFINITY}};
            std::array<double,3> clo{{INFINITY,INFINITY,INFINITY}};
            std::array<double,3> chi{{-INFINITY,-INFINITY,-INFINITY}};
            for (std::size_t i=begin;i<finish;++i) {
                const auto& t=result.triangles[ids[i]];
                const std::array<nm_float4,3> v{{t.bodyLocalVertex0,t.bodyLocalVertex1,t.bodyLocalVertex2}};
                for (const auto p:v) for (unsigned axis=0;axis<3u;++axis) {
                    lo[axis]=std::min(lo[axis],coord(p,axis));
                    hi[axis]=std::max(hi[axis],coord(p,axis));
                }
                for (unsigned axis=0;axis<3u;++axis) {
                    const double c=centroid(ids[i],axis);
                    clo[axis]=std::min(clo[axis],c); chi[axis]=std::max(chi[axis],c);
                }
            }
            node.boundsMinimum={std::nextafter(float(lo[0]),-INFINITY),
                std::nextafter(float(lo[1]),-INFINITY),
                std::nextafter(float(lo[2]),-INFINITY),0.0f};
            node.boundsMaximum={std::nextafter(float(hi[0]),INFINITY),
                std::nextafter(float(hi[1]),INFINITY),
                std::nextafter(float(hi[2]),INFINITY),0.0f};
            node.triangleIndex=kInvalidIndex;
            if (finish-begin==1u) {
                node.flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_BVH_NODE_LEAF;
                node.triangleIndex=ids[begin];
                result.nodes.push_back(node);
            } else {
                unsigned axis=0u;
                if (chi[1]-clo[1]>chi[axis]-clo[axis]) axis=1u;
                if (chi[2]-clo[2]>chi[axis]-clo[axis]) axis=2u;
                std::sort(ids.begin()+static_cast<std::ptrdiff_t>(begin),
                          ids.begin()+static_cast<std::ptrdiff_t>(finish),
                    [&](std::uint32_t a,std::uint32_t b) {
                        const double ca=centroid(a,axis), cb=centroid(b,axis);
                        if (ca!=cb) return ca<cb;
                        const auto& ta=result.triangles[a];
                        const auto& tb=result.triangles[b];
                        return std::tie(ta.boneStableId,ta.triangleStableId)<
                               std::tie(tb.boneStableId,tb.triangleStableId);
                    });
                node.flags=0u;
                result.nodes.push_back(node);
                const std::size_t middle=begin+(finish-begin)/2u;
                self(self,begin,middle);
                self(self,middle,finish);
            }
            const std::size_t escape=result.nodes.size();
            if (escape>std::numeric_limits<std::uint32_t>::max()) {
                overflow=true; return;
            }
            result.nodes[nodeIndex].escapeIndex=static_cast<std::uint32_t>(escape);
            result.nodes[nodeIndex].reserved0=0u;
        };
        build(build,0u,ids.size());
        if (overflow) return fail(BuildFailure::nodeIndexOverflow);
        const std::size_t nodeCount=result.nodes.size()-bodyRange.firstNode;
        bodyRange.nodeCount=static_cast<std::uint32_t>(nodeCount);
        result.bodies.push_back(bodyRange);
        firstTriangle=endTriangle;
    }
    return result;
}

// In static-complete mode the range count field carries nodeCount (not
// triangleCount); the body-local triangle inventory remains the shared table.
inline bool makeStaticCompleteBodyContactRanges(
    std::span<const NMNumiHumanFEMBodyContactSampleGPU> samples,
    std::span<const BodyRange> bodies,
    float maximumDistanceMeters,
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU>& output
) {
    output.clear();
    if (samples.empty() || bodies.empty() ||
        !std::isfinite(maximumDistanceMeters) ||
        maximumDistanceMeters<=0.0f || maximumDistanceMeters>0.05f)
        return false;
    for (std::size_t i=1u;i<bodies.size();++i)
        if (bodies[i-1u].bodyIndex>=bodies[i].bodyIndex) return false;
    output.reserve(samples.size());
    for (const auto& sample:samples) {
        const bool triangle=sample.flags==
            (NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
             NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT);
        if (!triangle) {
            output.push_back({});
            continue;
        }
        const auto it=std::lower_bound(
            bodies.begin(),bodies.end(),sample.bodyIndex,
            [](const BodyRange& body,std::uint32_t id) {
                return body.bodyIndex<id;
            });
        if (it==bodies.end() || it->bodyIndex!=sample.bodyIndex ||
            it->nodeCount==0u || it->firstNode>kInvalidIndex-it->nodeCount) {
            output.clear();
            return false;
        }
        output.push_back({it->firstNode,it->nodeCount,maximumDistanceMeters,
            it->closedOrientedCertified ? kRangeFlagStaticClosedOrientedBVH
                                        : kRangeFlagStaticCompleteBVH,
            {0.0f,0.0f,0.0f,0.0f}});
    }
    return true;
}

inline bool validStaticBVHRange(
    const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU& range,
    std::span<const NMNumiHumanFEMBodyContactBVHNodeGPU> nodes,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> triangles,
    std::uint32_t expectedBodyIndex
) noexcept {
    if ((range.reserved0!=kRangeFlagStaticCompleteBVH &&
         range.reserved0!=kRangeFlagStaticClosedOrientedBVH) ||
        range.triangleCount==0u ||
        !std::isfinite(range.maximumDistanceMeters) || range.maximumDistanceMeters<=0.0f ||
        range.maximumDistanceMeters>0.05f || range.firstTriangle>nodes.size() ||
        range.triangleCount>nodes.size()-range.firstTriangle ||
        range.restPointAndMotionRadius.x!=0.0f || range.restPointAndMotionRadius.y!=0.0f ||
        range.restPointAndMotionRadius.z!=0.0f || range.restPointAndMotionRadius.w!=0.0f)
        return false;
    const std::uint64_t end=std::uint64_t(range.firstTriangle)+range.triangleCount;
    for (std::uint32_t i=range.firstTriangle;i<end;++i) {
        const auto& n=nodes[i];
        if (!finite4(n.boundsMinimum) || !finite4(n.boundsMaximum) ||
            n.boundsMinimum.w!=0.0f || n.boundsMaximum.w!=0.0f ||
            n.boundsMinimum.x>n.boundsMaximum.x || n.boundsMinimum.y>n.boundsMaximum.y ||
            n.boundsMinimum.z>n.boundsMaximum.z || n.reserved0!=0u ||
            n.escapeIndex<=i || n.escapeIndex>end) return false;
        if (n.flags==NM_NUMI_HUMAN_FEM_BODY_CONTACT_BVH_NODE_LEAF) {
            if (n.triangleIndex>=triangles.size() ||
                triangles[n.triangleIndex].bodyIndex!=expectedBodyIndex) return false;
        } else if (n.flags!=0u || n.triangleIndex!=kInvalidIndex || i+1u>=end) {
            return false;
        }
    }
    const auto& root=nodes[range.firstTriangle];
    return root.escapeIndex==end &&
        (root.flags==NM_NUMI_HUMAN_FEM_BODY_CONTACT_BVH_NODE_LEAF ||
         root.flags==0u);
}

inline bool validCompleteStaticBVHSpans(
    std::span<const NMNumiHumanFEMBodyContactSampleGPU> samples,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> ranges,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> triangles,
    std::span<const NMNumiHumanFEMBodyContactBVHNodeGPU> nodes
) noexcept {
    if (ranges.size()!=samples.size() || samples.empty() || triangles.empty() ||
        nodes.empty() || triangles.size()>std::numeric_limits<std::uint32_t>::max() ||
        nodes.size()>std::numeric_limits<std::uint32_t>::max()) return false;
    for (std::size_t i=0;i<triangles.size();++i) {
        if (!validTriangle(triangles[i])) return false;
        if (i>0u) {
            const auto& a=triangles[i-1u]; const auto& b=triangles[i];
            if (std::tie(a.bodyIndex,a.boneStableId,a.triangleStableId)>=
                std::tie(b.bodyIndex,b.boneStableId,b.triangleStableId)) return false;
        }
    }
    struct TreeRef { std::uint32_t body, firstNode, nodeCount; };
    std::vector<TreeRef> trees;
    std::size_t nodeCursor=0u;
    std::set<std::uint32_t> closedCertifiedBodies;
    const auto finiteBounds=[](const NMNumiHumanFEMBodyContactBVHNodeGPU& n) {
        return finite4(n.boundsMinimum) && finite4(n.boundsMaximum) &&
            n.boundsMinimum.w==0.0f && n.boundsMaximum.w==0.0f &&
            n.boundsMinimum.x<=n.boundsMaximum.x &&
            n.boundsMinimum.y<=n.boundsMaximum.y &&
            n.boundsMinimum.z<=n.boundsMaximum.z && n.reserved0==0u;
    };
    const auto contains=[](const NMNumiHumanFEMBodyContactBVHNodeGPU& parent,
                           const NMNumiHumanFEMBodyContactBVHNodeGPU& child) {
        return parent.boundsMinimum.x<=child.boundsMinimum.x &&
            parent.boundsMinimum.y<=child.boundsMinimum.y &&
            parent.boundsMinimum.z<=child.boundsMinimum.z &&
            parent.boundsMaximum.x>=child.boundsMaximum.x &&
            parent.boundsMaximum.y>=child.boundsMaximum.y &&
            parent.boundsMaximum.z>=child.boundsMaximum.z;
    };
    // Validate every asset tree from the sorted triangle inventory. A body
    // need not have a contact sample this step; its complete tree still must
    // be present and valid so later grouped/query rows cannot expose unchecked
    // inventory. Trees are emitted contiguously in ascending body order.
    for (std::size_t triFirst=0u;triFirst<triangles.size();) {
        const std::uint32_t body=triangles[triFirst].bodyIndex;
        std::size_t triEnd=triFirst+1u;
        while (triEnd<triangles.size() && triangles[triEnd].bodyIndex==body) ++triEnd;
        if (nodeCursor>=nodes.size() || !finiteBounds(nodes[nodeCursor])) return false;
        const std::uint32_t firstNode=static_cast<std::uint32_t>(nodeCursor);
        const auto rootEnd=nodes[nodeCursor].escapeIndex;
        if (rootEnd<=nodeCursor || rootEnd>nodes.size()) return false;
        const std::uint32_t nodeEnd=static_cast<std::uint32_t>(rootEnd);
        const std::uint64_t triCount=triEnd-triFirst;
        if (triCount>(std::numeric_limits<std::uint32_t>::max()+1ull)/2ull ||
            std::uint64_t(nodeEnd)-nodeCursor!=2u*triCount-1u) return false;
        const std::uint32_t triFirst32=static_cast<std::uint32_t>(triFirst);
        const std::uint32_t triCount32=static_cast<std::uint32_t>(triCount);
        std::vector<std::uint8_t> covered(triCount32,0u);
        const auto walk=[&](auto&& self,std::uint32_t index,unsigned depth)->std::uint32_t {
            if (index<firstNode || index>=nodeEnd || depth>64u) return UINT32_MAX;
            const auto& n=nodes[index];
            if (!finiteBounds(n) || n.escapeIndex<=index || n.escapeIndex>nodeEnd)
                return UINT32_MAX;
            if (n.flags==NM_NUMI_HUMAN_FEM_BODY_CONTACT_BVH_NODE_LEAF) {
                if (n.escapeIndex!=index+1u || n.triangleIndex<triFirst32 ||
                    n.triangleIndex-triFirst32>=triCount32) return UINT32_MAX;
                const std::uint32_t local=n.triangleIndex-triFirst32;
                if (covered[local]!=0u || triangles[n.triangleIndex].bodyIndex!=body)
                    return UINT32_MAX;
                covered[local]=1u;
                const auto& t=triangles[n.triangleIndex];
                for (const auto v:{t.bodyLocalVertex0,t.bodyLocalVertex1,t.bodyLocalVertex2})
                    if (v.x<n.boundsMinimum.x || v.y<n.boundsMinimum.y || v.z<n.boundsMinimum.z ||
                        v.x>n.boundsMaximum.x || v.y>n.boundsMaximum.y || v.z>n.boundsMaximum.z)
                        return UINT32_MAX;
                return index+1u;
            }
            if (n.flags!=0u || n.triangleIndex!=kInvalidIndex || index+1u>=nodeEnd)
                return UINT32_MAX;
            const std::uint32_t left=index+1u;
            const std::uint32_t right=self(self,left,depth+1u);
            if (right==UINT32_MAX || right>=nodeEnd || !contains(n,nodes[left]) ||
                !contains(n,nodes[right])) return UINT32_MAX;
            const std::uint32_t after=self(self,right,depth+1u);
            if (after==UINT32_MAX || n.escapeIndex!=after) return UINT32_MAX;
            return after;
        };
        if (walk(walk,firstNode,0u)!=nodeEnd ||
            std::find(covered.begin(),covered.end(),0u)!=covered.end()) return false;
        trees.push_back({body,firstNode,nodeEnd-firstNode});
        nodeCursor=nodeEnd;
        triFirst=triEnd;
    }
    if (nodeCursor!=nodes.size() || trees.empty()) return false;

    for (std::size_t i=0u;i<samples.size();++i) {
        const auto& sample=samples[i];
        const auto& range=ranges[i];
        const bool triangleSample=sample.flags==
            (NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
             NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT);
        if (range.triangleCount==0u) {
            if (triangleSample || range.firstTriangle!=0u ||
                range.maximumDistanceMeters!=0.0f || range.reserved0!=0u ||
                range.restPointAndMotionRadius.x!=0.0f ||
                range.restPointAndMotionRadius.y!=0.0f ||
                range.restPointAndMotionRadius.z!=0.0f ||
                range.restPointAndMotionRadius.w!=0.0f) return false;
            continue;
        }
        if (!triangleSample ||
            (range.reserved0!=kRangeFlagStaticCompleteBVH &&
             range.reserved0!=kRangeFlagStaticClosedOrientedBVH) ||
            !std::isfinite(range.maximumDistanceMeters) ||
            range.maximumDistanceMeters<=0.0f || range.maximumDistanceMeters>0.05f ||
            range.restPointAndMotionRadius.x!=0.0f ||
            range.restPointAndMotionRadius.y!=0.0f ||
            range.restPointAndMotionRadius.z!=0.0f ||
            range.restPointAndMotionRadius.w!=0.0f) return false;
        if (range.reserved0==kRangeFlagStaticClosedOrientedBVH &&
            closedCertifiedBodies.insert(sample.bodyIndex).second) {
            const auto first=std::lower_bound(triangles.begin(),triangles.end(),
                sample.bodyIndex,[](const auto& triangle,std::uint32_t body) {
                    return triangle.bodyIndex<body;
                });
            const auto last=std::upper_bound(first,triangles.end(),
                sample.bodyIndex,[](std::uint32_t body,const auto& triangle) {
                    return body<triangle.bodyIndex;
                });
            std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> bank(
                first,last);
            if (bank.empty() || !buildExactFeaturePseudonormals(bank,true))
                return false;
        }
        const auto tree=std::lower_bound(trees.begin(),trees.end(),sample.bodyIndex,
            [](const TreeRef& x,std::uint32_t body){return x.body<body;});
        if (tree==trees.end() || tree->body!=sample.bodyIndex ||
            range.firstTriangle!=tree->firstNode || range.triangleCount!=tree->nodeCount)
            return false;
    }
    return true;
}

// Expands one triangle-contact template per slave into one deterministic row
// for every complete registered body bank. Non-triangle rows (fascia glide and
// elastic attachments) are preserved once. The generated triangle rows use a
// valid seed face only as ABI storage; the GPU query resolves the live closest
// face from the selected body's complete BVH each step.
inline bool expandTriangleContactSamplesAcrossAllBodies(
    std::span<const NMNumiHumanFEMBodyContactSampleGPU> input,
    std::span<const BodyRange> bodies,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> triangles,
    std::vector<NMNumiHumanFEMBodyContactSampleGPU>& output
) {
    output.clear();
    if (input.empty() || bodies.empty() || triangles.empty()) return false;
    for (std::size_t i=1u;i<bodies.size();++i)
        if (bodies[i-1u].bodyIndex>=bodies[i].bodyIndex) return false;
    std::uint64_t triangleTemplates=0u,nonTriangleRows=0u;
    std::uint32_t priorSlave=0u;
    bool sawTriangleTemplate=false;
    for (const auto& sample:input) {
        const bool triangleSample=sample.flags==
            (NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
             NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT);
        if (!triangleSample) { ++nonTriangleRows; continue; }
        if (!numi::matter::fem_body_triangle_contact::validTriangleContactSample(sample) ||
            (sawTriangleTemplate && sample.slaveNode<=priorSlave)) return false;
        sawTriangleTemplate=true;
        priorSlave=sample.slaveNode;
        ++triangleTemplates;
    }
    if (triangleTemplates==0u ||
        triangleTemplates*std::uint64_t(bodies.size())+nonTriangleRows>
            std::numeric_limits<std::uint32_t>::max()) return false;
    for (const auto& body:bodies) {
        if (body.bodyIndex==kInvalidIndex || body.triangleCount==0u ||
            body.firstTriangle>triangles.size() ||
            body.triangleCount>triangles.size()-body.firstTriangle ||
            triangles[body.firstTriangle].bodyIndex!=body.bodyIndex) return false;
        for (std::uint32_t i=0u;i<body.triangleCount;++i)
            if (triangles[body.firstTriangle+i].bodyIndex!=body.bodyIndex) return false;
    }
    output.reserve(static_cast<std::size_t>(triangleTemplates*bodies.size()+nonTriangleRows));
    for (const auto& sample:input) {
        const bool triangleSample=sample.flags==
            (NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
             NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT);
        if (!triangleSample) {
            output.push_back(sample);
            continue;
        }
        for (const auto& body:bodies) {
            const auto& seed=triangles[body.firstTriangle];
            auto expanded=sample;
            expanded.bodyIndex=body.bodyIndex;
            expanded.bodyLocalPointAndArea.x=seed.bodyLocalVertex0.x;
            expanded.bodyLocalPointAndArea.y=seed.bodyLocalVertex0.y;
            expanded.bodyLocalPointAndArea.z=seed.bodyLocalVertex0.z;
            expanded.bodyLocalNormalAndReferenceSeparation.x=seed.bodyLocalVertex1.x;
            expanded.bodyLocalNormalAndReferenceSeparation.y=seed.bodyLocalVertex1.y;
            expanded.bodyLocalNormalAndReferenceSeparation.z=seed.bodyLocalVertex1.z;
            expanded.stiffnessAndNormalStrainPerPressure.x=seed.bodyLocalVertex2.x;
            expanded.stiffnessAndNormalStrainPerPressure.y=seed.bodyLocalVertex2.y;
            expanded.stiffnessAndNormalStrainPerPressure.z=seed.bodyLocalVertex2.z;
            if (!numi::matter::fem_body_triangle_contact::validTriangleContactSample(expanded)) {
                output.clear();
                return false;
            }
            output.push_back(expanded);
        }
    }
    return true;
}


struct PreparedStaticCompleteBodyBVH {
    BuildFailure failure=BuildFailure::none;
    std::vector<NMNumiHumanFEMBodyContactSampleGPU> samples;
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> triangles;
    std::vector<NMNumiHumanFEMBodyContactBVHNodeGPU> nodes;
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> ranges;
    [[nodiscard]] bool succeeded() const noexcept {
        return failure==BuildFailure::none;
    }
};

struct PreparationOptions {
    bool requireClosedOrientedBanks=false;
    bool expandAllTriangleSamplesAcrossBodies=false;
};

// Asset-load entry point: validates and deterministically flattens the complete
// body-local triangle table, builds per-body trees, and prepares aligned ranges.
inline PreparedStaticCompleteBodyBVH prepareStaticCompleteBodyContactAsset(
    std::span<const NMNumiHumanFEMBodyContactSampleGPU> samples,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> input,
    std::size_t expectedTriangleCount,
    float maximumDistanceMeters,
    PreparationOptions options = {}
) {
    PreparedStaticCompleteBodyBVH out{};
    auto tree=buildStaticCompleteBodyBVH(
        input,expectedTriangleCount,options.requireClosedOrientedBanks);
    if (!tree.succeeded()) {
        out.failure=tree.failure;
        return out;
    }
    if (options.expandAllTriangleSamplesAcrossBodies) {
        if (!expandTriangleContactSamplesAcrossAllBodies(
                samples,tree.bodies,tree.triangles,out.samples)) {
            out.failure=BuildFailure::invalidContactSamples;
            return out;
        }
    } else {
        out.samples.assign(samples.begin(),samples.end());
    }
    if (!makeStaticCompleteBodyContactRanges(
            out.samples,tree.bodies,maximumDistanceMeters,out.ranges)) {
        out.failure=BuildFailure::invalidContactSamples;
        return out;
    }
    if (!validCompleteStaticBVHSpans(out.samples,out.ranges,
            tree.triangles,tree.nodes)) {
        out.failure=BuildFailure::invalidContactSamples;
        return out;
    }
    out.triangles=std::move(tree.triangles);
    out.nodes=std::move(tree.nodes);
    return out;
}

} // namespace numi::matter::fem_body_contact_bvh

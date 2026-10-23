#pragma once

#include "numi/matter/numi_human_shared.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <tuple>
#include <vector>

namespace numi::matter::fem_body_contact_bvh {

inline constexpr std::uint32_t kInvalidIndex = UINT32_MAX;
inline constexpr std::uint32_t kRangeFlagStaticCompleteBVH = 2u;

struct BodyRange {
    std::uint32_t bodyIndex = kInvalidIndex;
    std::uint32_t firstNode = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t firstTriangle = 0u;
    std::uint32_t triangleCount = 0u;
};

enum class BuildFailure : std::uint8_t {
    none, emptyInventory, inventoryCountMismatch, flattenedIndexOverflow,
    invalidTriangle, duplicateStableIdentity, nodeIndexOverflow,
    invalidContactSamples,
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
    if (t.bodyIndex==kInvalidIndex || t.boneStableId==kInvalidIndex ||
        t.triangleStableId==kInvalidIndex || t.reserved0!=0u ||
        !finite4(t.bodyLocalVertex0) || !finite4(t.bodyLocalVertex1) ||
        !finite4(t.bodyLocalVertex2) || !finite4(t.authoredOutwardNormal) ||
        t.bodyLocalVertex0.w!=0.0f || t.bodyLocalVertex1.w!=0.0f ||
        t.bodyLocalVertex2.w!=0.0f || t.authoredOutwardNormal.w!=0.0f)
        return false;
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

inline BuildResult buildStaticCompleteBodyBVH(
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> input,
    std::size_t expectedTriangleCount
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
            kRangeFlagStaticCompleteBVH,{0.0f,0.0f,0.0f,0.0f}});
    }
    return true;
}

inline bool validStaticBVHRange(
    const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU& range,
    std::span<const NMNumiHumanFEMBodyContactBVHNodeGPU> nodes,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> triangles,
    std::uint32_t expectedBodyIndex
) noexcept {
    if (range.reserved0!=kRangeFlagStaticCompleteBVH || range.triangleCount==0u ||
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
    if (ranges.size()!=samples.size() || triangles.empty() || nodes.empty() ||
        triangles.size()>std::numeric_limits<std::uint32_t>::max() ||
        nodes.size()>std::numeric_limits<std::uint32_t>::max()) return false;
    for (std::size_t i=0;i<triangles.size();++i) {
        if (!validTriangle(triangles[i])) return false;
        if (i>0u) {
            const auto& a=triangles[i-1u]; const auto& b=triangles[i];
            if (std::tie(a.bodyIndex,a.boneStableId,a.triangleStableId)>=
                std::tie(b.bodyIndex,b.boneStableId,b.triangleStableId)) return false;
        }
    }
    struct TreeRef {
        std::uint32_t body, firstNode, nodeCount, firstTriangle, triangleCount;
    };
    std::vector<TreeRef> refs;
    for (std::size_t i=0;i<samples.size();++i) {
        const auto& r=ranges[i];
        const bool triangleSample=samples[i].flags==
            (NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
             NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT);
        if (r.triangleCount==0u) {
            if (triangleSample || r.firstTriangle!=0u ||
                r.maximumDistanceMeters!=0.0f || r.reserved0!=0u ||
                r.restPointAndMotionRadius.x!=0.0f ||
                r.restPointAndMotionRadius.y!=0.0f ||
                r.restPointAndMotionRadius.z!=0.0f ||
                r.restPointAndMotionRadius.w!=0.0f) return false;
            continue;
        }
        if (!triangleSample ||
            r.reserved0!=kRangeFlagStaticCompleteBVH ||
            !std::isfinite(r.maximumDistanceMeters) ||
            r.maximumDistanceMeters<=0.0f || r.maximumDistanceMeters>0.05f ||
            r.firstTriangle>nodes.size() ||
            r.triangleCount>nodes.size()-r.firstTriangle ||
            r.restPointAndMotionRadius.x!=0.0f ||
            r.restPointAndMotionRadius.y!=0.0f ||
            r.restPointAndMotionRadius.z!=0.0f ||
            r.restPointAndMotionRadius.w!=0.0f) return false;
        const auto body=samples[i].bodyIndex;
        const auto prior=std::find_if(refs.begin(),refs.end(),
            [body](const TreeRef& x){return x.body==body;});
        if (prior!=refs.end()) {
            if (prior->firstNode!=r.firstTriangle ||
                prior->nodeCount!=r.triangleCount) return false;
            continue;
        }
        const auto first=std::lower_bound(triangles.begin(),triangles.end(),body,
            [](const auto& t,std::uint32_t id){return t.bodyIndex<id;});
        const auto last=std::upper_bound(first,triangles.end(),body,
            [](std::uint32_t id,const auto& t){return id<t.bodyIndex;});
        if (first==last) return false;
        const auto triFirst=static_cast<std::uint32_t>(first-triangles.begin());
        const auto triCount=static_cast<std::uint32_t>(last-first);
        refs.push_back({body,r.firstTriangle,r.triangleCount,triFirst,triCount});
    }
    if (refs.empty()) return false;
    // Static-complete mode guarantees every body in the triangle inventory has
    // a query range, rather than silently leaving a registered body uncovered.
    for (std::size_t first=0u;first<triangles.size();) {
        const std::uint32_t body=triangles[first].bodyIndex;
        if (std::find_if(refs.begin(),refs.end(),
                [body](const TreeRef& x){return x.body==body;})==refs.end())
            return false;
        std::size_t last=first+1u;
        while (last<triangles.size() && triangles[last].bodyIndex==body) ++last;
        first=last;
    }
    for (const auto& ref:refs) {
        const auto& firstRange=ranges[static_cast<std::size_t>(
            std::find_if(samples.begin(),samples.end(),
                [&ref](const auto& s){return s.bodyIndex==ref.body &&
                    s.flags==(NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
                              NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT);})
            -samples.begin())];
        if (!validStaticBVHRange(firstRange,nodes,triangles,ref.body))
            return false;
        const std::uint64_t end64=std::uint64_t(ref.firstNode)+ref.nodeCount;
        if (end64>nodes.size() ||
            ref.triangleCount!=(ref.nodeCount+1u)/2u) return false;
        const std::uint32_t end=static_cast<std::uint32_t>(end64);
        std::vector<std::uint8_t> covered(ref.triangleCount,0u);
        const auto contains=[](const NMNumiHumanFEMBodyContactBVHNodeGPU& parent,
                               const NMNumiHumanFEMBodyContactBVHNodeGPU& child) {
            return parent.boundsMinimum.x<=child.boundsMinimum.x &&
                parent.boundsMinimum.y<=child.boundsMinimum.y &&
                parent.boundsMinimum.z<=child.boundsMinimum.z &&
                parent.boundsMaximum.x>=child.boundsMaximum.x &&
                parent.boundsMaximum.y>=child.boundsMaximum.y &&
                parent.boundsMaximum.z>=child.boundsMaximum.z;
        };
        const auto walk=[&](auto&& self,std::uint32_t index,unsigned depth)->std::uint32_t {
            if (index>=end || depth>64u) return UINT32_MAX;
            const auto& n=nodes[index];
            if (n.flags==NM_NUMI_HUMAN_FEM_BODY_CONTACT_BVH_NODE_LEAF) {
                if (n.escapeIndex!=index+1u || n.triangleIndex<ref.firstTriangle ||
                    n.triangleIndex-ref.firstTriangle>=ref.triangleCount)
                    return UINT32_MAX;
                const std::uint32_t local=n.triangleIndex-ref.firstTriangle;
                if (covered[local]!=0u) return UINT32_MAX;
                covered[local]=1u;
                const auto& t=triangles[n.triangleIndex];
                if (t.bodyIndex!=ref.body) return UINT32_MAX;
                for (const auto v:{t.bodyLocalVertex0,t.bodyLocalVertex1,t.bodyLocalVertex2})
                    if (v.x<n.boundsMinimum.x || v.y<n.boundsMinimum.y || v.z<n.boundsMinimum.z ||
                        v.x>n.boundsMaximum.x || v.y>n.boundsMaximum.y || v.z>n.boundsMaximum.z)
                        return UINT32_MAX;
                return index+1u;
            }
            if (n.flags!=0u || n.triangleIndex!=kInvalidIndex || index+1u>=end) return UINT32_MAX;
            const std::uint32_t left=index+1u;
            const std::uint32_t right=self(self,left,depth+1u);
            if (right==UINT32_MAX || right>=end || !contains(n,nodes[left]) ||
                !contains(n,nodes[right])) return UINT32_MAX;
            const std::uint32_t after=self(self,right,depth+1u);
            if (after==UINT32_MAX || n.escapeIndex!=after) return UINT32_MAX;
            return after;
        };
        if (walk(walk,ref.firstNode,0u)!=end ||
            std::find(covered.begin(),covered.end(),0u)!=covered.end()) return false;
    }
    return true;
}

struct PreparedStaticCompleteBodyBVH {
    BuildFailure failure=BuildFailure::none;
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> triangles;
    std::vector<NMNumiHumanFEMBodyContactBVHNodeGPU> nodes;
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> ranges;
    [[nodiscard]] bool succeeded() const noexcept {
        return failure==BuildFailure::none;
    }
};

// Asset-load entry point: validates and deterministically flattens the complete
// body-local triangle table, builds per-body trees, and prepares aligned ranges.
inline PreparedStaticCompleteBodyBVH prepareStaticCompleteBodyContactAsset(
    std::span<const NMNumiHumanFEMBodyContactSampleGPU> samples,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> input,
    std::size_t expectedTriangleCount,
    float maximumDistanceMeters
) {
    PreparedStaticCompleteBodyBVH out{};
    auto tree=buildStaticCompleteBodyBVH(input,expectedTriangleCount);
    if (!tree.succeeded()) {
        out.failure=tree.failure;
        return out;
    }
    if (!makeStaticCompleteBodyContactRanges(
            samples,tree.bodies,maximumDistanceMeters,out.ranges)) {
        out.failure=BuildFailure::invalidContactSamples;
        return out;
    }
    if (!validCompleteStaticBVHSpans(samples,out.ranges,
            tree.triangles,tree.nodes)) {
        out.failure=BuildFailure::invalidContactSamples;
        return out;
    }
    out.triangles=std::move(tree.triangles);
    out.nodes=std::move(tree.nodes);
    return out;
}

} // namespace numi::matter::fem_body_contact_bvh

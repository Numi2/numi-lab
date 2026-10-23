#include "numi/matter/fem_body_contact_static_bvh.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

using namespace numi::matter::fem_body_contact_bvh;
namespace {
void require(bool ok,std::string_view message) {
    if (!ok) { std::fprintf(stderr,"FAIL: %.*s\n",int(message.size()),message.data()); std::exit(1); }
}
NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU tri(
    std::uint32_t body,std::uint32_t bone,std::uint32_t id,
    float x,float y,float z,float dx,float dy
) {
    NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU t{};
    t.bodyLocalVertex0={x,y,z,0}; t.bodyLocalVertex1={x+dx,y,z,0};
    t.bodyLocalVertex2={x,y+dy,z,0}; t.authoredOutwardNormal={0,0,1,0};
    t.bodyIndex=body; t.boneStableId=bone; t.triangleStableId=id;
    return t;
}
}
int main() {
    // Includes a tied duplicate face at the same site to exercise stable-ID tie-breaking.
    const std::array<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU,7> input{{
        tri(9,4,8, 0,0,0, 1,0.2f), tri(9,4,2, 0,0,0, 1,0.2f),
        tri(9,7,5, 2,0,0, 0.5f,0.5f), tri(9,7,1, 0,2,0, 0.3f,0.4f),
        tri(2,3,6, -1,0,0, 0.4f,0.6f), tri(2,3,3, 0,-2,0, 0.2f,0.2f),
        tri(2,8,9, 1,1,1, 0.1f,0.1f),
    }};
    const auto a=buildStaticCompleteBodyBVH(input,input.size());
    const auto b=buildStaticCompleteBodyBVH(input,input.size());
    require(a.succeeded() && b.succeeded(),"complete two-body inventory builds");
    require(a.triangles.size()==input.size() && a.nodes.size()==2*input.size()-a.bodies.size(),
            "one leaf per complete triangle and binary internal nodes");
    require(a.bodies.size()==2 && a.bodies[0].bodyIndex==2 && a.bodies[1].bodyIndex==9,
            "body ranges have deterministic ascending IDs");
    require(std::memcmp(a.triangles.data(),b.triangles.data(),a.triangles.size()*sizeof(a.triangles[0]))==0 &&
            std::memcmp(a.nodes.data(),b.nodes.data(),a.nodes.size()*sizeof(a.nodes[0]))==0,
            "flattened inventory and BVH are deterministic");
    for (const auto& body:a.bodies) {
        NMNumiHumanFEMBodyContactNeighborhoodRangeGPU range{};
        range.firstTriangle=body.firstNode; range.triangleCount=body.nodeCount;
        range.maximumDistanceMeters=0.05f;
        range.reserved0=kRangeFlagStaticCompleteBVH;
        require(validStaticBVHRange(range,a.nodes,a.triangles,body.bodyIndex),
                "flattened body range validates all node and leaf bounds");
        std::uint32_t leaves=0;
        const auto end=body.firstNode+body.nodeCount;
        for (std::uint32_t i=body.firstNode;i<end;++i) {
            const auto& node=a.nodes[i];
            require(node.escapeIndex>i && node.escapeIndex<=end,
                    "escape links stay forward and within this body tree");
            if (node.flags&NM_NUMI_HUMAN_FEM_BODY_CONTACT_BVH_NODE_LEAF) ++leaves;
        }
        require(leaves==body.triangleCount,"every complete triangle appears once as a leaf");
    }
    std::array<NMNumiHumanFEMBodyContactSampleGPU,2> samples{};
    std::array<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU,2> ranges{};
    for (std::size_t i=0;i<2u;++i) {
        samples[i].bodyIndex=a.bodies[i].bodyIndex;
        samples[i].flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
            NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT;
        ranges[i].firstTriangle=a.bodies[i].firstNode;
        ranges[i].triangleCount=a.bodies[i].nodeCount;
        ranges[i].maximumDistanceMeters=0.05f;
        ranges[i].reserved0=kRangeFlagStaticCompleteBVH;
    }
    require(validCompleteStaticBVHSpans(samples,ranges,a.triangles,a.nodes),
            "complete per-body inventory and sample ranges validate");
    auto brokenNodes=a.nodes; brokenNodes[a.bodies[0].firstNode].escapeIndex++;
    require(!validCompleteStaticBVHSpans(samples,ranges,a.triangles,brokenNodes),
            "malformed escape links are rejected");
    auto duplicate=input; duplicate[1].triangleStableId=duplicate[0].triangleStableId;
    require(buildStaticCompleteBodyBVH(duplicate,duplicate.size()).failure==
            BuildFailure::duplicateStableIdentity,"duplicate stable IDs rejected");
    require(buildStaticCompleteBodyBVH(input,input.size()-1).failure==
            BuildFailure::inventoryCountMismatch,"partial registered inventory rejected");
    auto invalid=input; invalid[0].authoredOutwardNormal.z=-1.0f;
    require(buildStaticCompleteBodyBVH(invalid,invalid.size()).failure==
            BuildFailure::invalidTriangle,"winding/normal mismatch rejected");
    auto nonfinite=input; nonfinite[0].bodyLocalVertex0.x=NAN;
    require(buildStaticCompleteBodyBVH(nonfinite,nonfinite.size()).failure==
            BuildFailure::invalidTriangle,"nonfinite triangle rejected");
    // Convert body node spans into aligned sample ranges. Triangle samples
    // must have a complete body tree; legacy samples retain exact empty rows.
    std::array<NMNumiHumanFEMBodyContactSampleGPU,3> staticSamples{};
    staticSamples[0].bodyIndex=2u;
    staticSamples[0].flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
                     NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT;
    staticSamples[1].bodyIndex=9u;
    staticSamples[1].flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
                     NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT;
    staticSamples[2].flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE;
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> staticRanges;
    require(makeStaticCompleteBodyContactRanges(staticSamples,a.bodies,0.02f,staticRanges),
            "construct per-sample complete BVH ranges");
    require(staticRanges.size()==staticSamples.size() &&
            staticRanges[0].firstTriangle==a.bodies[0].firstNode &&
            staticRanges[0].triangleCount==a.bodies[0].nodeCount &&
            staticRanges[0].reserved0==kRangeFlagStaticCompleteBVH &&
            staticRanges[1].firstTriangle==a.bodies[1].firstNode &&
            staticRanges[1].triangleCount==a.bodies[1].nodeCount &&
            staticRanges[2].triangleCount==0u && staticRanges[2].reserved0==0u,
            "ranges point at each body's full tree and preserve legacy rows");
    require(validCompleteStaticBVHSpans(staticSamples,staticRanges,a.triangles,a.nodes),
            "complete BVH ranges validate against all triangles");
    auto prepared=prepareStaticCompleteBodyContactAsset(
        staticSamples,input,input.size(),0.02f);
    require(prepared.succeeded() && prepared.triangles.size()==a.triangles.size() &&
            prepared.nodes.size()==a.nodes.size() &&
            prepared.ranges.size()==staticSamples.size(),
            "asset-load builder prepares one complete body-local bank");
    require(std::memcmp(prepared.triangles.data(),a.triangles.data(),
                a.triangles.size()*sizeof(a.triangles[0]))==0 &&
            std::memcmp(prepared.nodes.data(),a.nodes.data(),
                a.nodes.size()*sizeof(a.nodes[0]))==0 &&
            validCompleteStaticBVHSpans(staticSamples,prepared.ranges,
                prepared.triangles,prepared.nodes),
            "asset-load output is deterministic and fully validated");
    require(prepareStaticCompleteBodyContactAsset(
                staticSamples,input,input.size()-1u,0.02f).failure==
                BuildFailure::inventoryCountMismatch,
            "asset-load refuses incomplete registered-body inventory");
    auto absentBody=staticSamples;
    absentBody[1].bodyIndex=10u;
    require(prepareStaticCompleteBodyContactAsset(
                absentBody,input,input.size(),0.02f).failure==
                BuildFailure::invalidContactSamples,
            "asset-load refuses triangle samples without a registered body tree");
    auto missing=staticRanges;
    missing[0]={};
    require(!validCompleteStaticBVHSpans(staticSamples,missing,a.triangles,a.nodes),
            "active triangle sample cannot omit its complete BVH range");
    std::puts("static complete body BVH CPU regression PASS");
}

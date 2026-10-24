#include "numi/matter/fem_body_triangle_contact_sample.hpp"
#include "numi/matter/fem_body_contact_static_bvh.hpp"
#include "../../apps/NumiHumanInitialSkinBoneContactWitness.hpp"

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
std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> closedBox(
    std::uint32_t body
) {
    const std::array<std::array<float,3>,8> p{{
        {{-1,-1,-1}},{{1,-1,-1}},{{1,1,-1}},{{-1,1,-1}},
        {{-1,-1,1}},{{1,-1,1}},{{1,1,1}},{{-1,1,1}}}};
    const std::array<std::array<unsigned,3>,12> faces{{
        {{0,2,1}},{{0,3,2}},{{4,5,6}},{{4,6,7}},
        {{0,4,7}},{{0,7,3}},{{1,2,6}},{{1,6,5}},
        {{0,1,5}},{{0,5,4}},{{3,7,6}},{{3,6,2}}}};
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> out;
    for (std::uint32_t i=0;i<faces.size();++i) {
        const auto& f=faces[i];
        auto t=tri(body,77u,i,p[f[0]][0],p[f[0]][1],p[f[0]][2],0.01f,0.01f);
        t.bodyLocalVertex0={p[f[0]][0],p[f[0]][1],p[f[0]][2],0};
        t.bodyLocalVertex1={p[f[1]][0],p[f[1]][1],p[f[1]][2],0};
        t.bodyLocalVertex2={p[f[2]][0],p[f[2]][1],p[f[2]][2],0};
        const double ax=double(t.bodyLocalVertex1.x)-t.bodyLocalVertex0.x;
        const double ay=double(t.bodyLocalVertex1.y)-t.bodyLocalVertex0.y;
        const double az=double(t.bodyLocalVertex1.z)-t.bodyLocalVertex0.z;
        const double bx=double(t.bodyLocalVertex2.x)-t.bodyLocalVertex0.x;
        const double by=double(t.bodyLocalVertex2.y)-t.bodyLocalVertex0.y;
        const double bz=double(t.bodyLocalVertex2.z)-t.bodyLocalVertex0.z;
        const double nx=ay*bz-az*by,ny=az*bx-ax*bz,nz=ax*by-ay*bx;
        const double length=std::sqrt(nx*nx+ny*ny+nz*nz);
        t.authoredOutwardNormal={float(nx/length),float(ny/length),float(nz/length),0};
        out.push_back(t);
    }
    return out;
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
    // The asset owns complete trees for all registered bodies, while a given
    // contact batch may query only a subset. Unqueried trees are still checked.
    std::array<NMNumiHumanFEMBodyContactSampleGPU,1> oneBodySample{};
    oneBodySample[0].bodyIndex=2u;
    oneBodySample[0].flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
        NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT;
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> oneBodyRanges;
    require(makeStaticCompleteBodyContactRanges(oneBodySample,a.bodies,0.02f,
                oneBodyRanges) &&
            validCompleteStaticBVHSpans(oneBodySample,oneBodyRanges,
                a.triangles,a.nodes),
            "one queried body validates while the complete unqueried body tree remains present");
    auto badUnqueriedTree=a.nodes;
    badUnqueriedTree[a.bodies[1].firstNode].boundsMinimum.x+=10.0f;
    require(!validCompleteStaticBVHSpans(oneBodySample,oneBodyRanges,
                a.triangles,badUnqueriedTree),
            "corruption in an unqueried body's complete tree is rejected");
    auto wrongBodyRange=oneBodyRanges;
    wrongBodyRange[0].firstTriangle=a.bodies[1].firstNode;
    wrongBodyRange[0].triangleCount=a.bodies[1].nodeCount;
    require(!validCompleteStaticBVHSpans(oneBodySample,wrongBodyRange,
                a.triangles,a.nodes),
            "sample range cannot query a different body's tree");
    const auto seed9=a.triangles[a.bodies[1].firstTriangle];
    std::array<NMNumiHumanFEMBodyContactSampleGPU,2> pairTemplates{};
    pairTemplates[0]=numi::matter::fem_body_triangle_contact::makeTriangleContactSample(
        100u,9u,seed9.bodyLocalVertex0,seed9.bodyLocalVertex1,
        seed9.bodyLocalVertex2,0.25f,7.0f,0.125f);
    pairTemplates[1]=pairTemplates[0];
    pairTemplates[1].slaveNode=200u;
    numi::matter::fem_body_contact_bvh::PreparationOptions pairOptions{};
    pairOptions.expandAllTriangleSamplesAcrossBodies=true;
    const auto allBodyPrepared=prepareStaticCompleteBodyContactAsset(
        pairTemplates,input,input.size(),0.02f,pairOptions);
    require(allBodyPrepared.succeeded()&&allBodyPrepared.samples.size()==4u&&
            allBodyPrepared.ranges.size()==4u&&
            allBodyPrepared.samples[0].slaveNode==100u&&
            allBodyPrepared.samples[0].bodyIndex==2u&&
            allBodyPrepared.samples[1].bodyIndex==9u&&
            allBodyPrepared.samples[2].slaveNode==200u&&
            allBodyPrepared.samples[2].bodyIndex==2u&&
            allBodyPrepared.samples[3].bodyIndex==9u&&
            validCompleteStaticBVHSpans(allBodyPrepared.samples,
                allBodyPrepared.ranges,allBodyPrepared.triangles,allBodyPrepared.nodes),
            "named all-body option expands deterministic node/body pairs");
    const auto legacyDefaults=prepareStaticCompleteBodyContactAsset(
        pairTemplates,input,input.size(),0.02f);
    require(legacyDefaults.succeeded()&&legacyDefaults.samples.size()==pairTemplates.size(),
            "four-argument preparation preserves legacy one-body template semantics");
    auto strictPairOptions=pairOptions;
    strictPairOptions.requireClosedOrientedBanks=true;
    require(prepareStaticCompleteBodyContactAsset(
                pairTemplates,input,input.size(),0.02f,strictPairOptions).failure==
                BuildFailure::invalidClosedOrientedBank,
            "strict option rejects open/inconsistently oriented complete banks");
    static_assert(NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION == 16u,
                  "feature-normal triangle records require ABI 16");
    static_assert(sizeof(NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU) == 176u,
                  "host triangle stride must match Metal ABI 16");
    static_assert(__builtin_offsetof(
        NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU, edgePseudoNormal01) == 64u);
    static_assert(__builtin_offsetof(
        NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU, bodyIndex) == 160u);
    auto boxTriangles=closedBox(77u);
    NMNumiHumanFEMBodyContactSampleGPU boxSample{};
    boxSample.bodyIndex=77u;
    boxSample.flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
        NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT;
    const auto ordinaryBox=prepareStaticCompleteBodyContactAsset(
        std::span(&boxSample,1u),boxTriangles,boxTriangles.size(),0.02f);
    const auto certifiedBox=prepareStaticCompleteBodyContactAsset(
        std::span(&boxSample,1u),boxTriangles,boxTriangles.size(),0.02f,
        PreparationOptions{.requireClosedOrientedBanks=true});
    require(ordinaryBox.succeeded() &&
            ordinaryBox.ranges[0].reserved0==kRangeFlagStaticCompleteBVH &&
            certifiedBox.succeeded() &&
            certifiedBox.ranges[0].reserved0==kRangeFlagStaticClosedOrientedBVH,
            "only strict closed/oriented preparation emits the per-range certificate");
    require(validCompleteStaticBVHSpans(certifiedBox.samples,certifiedBox.ranges,
                certifiedBox.triangles,certifiedBox.nodes),
            "closed certificate revalidates against the exact body bank");
    auto openCertified=ordinaryBox.ranges;
    openCertified[0].reserved0=kRangeFlagStaticClosedOrientedBVH;
    auto openTriangles=boxTriangles; openTriangles.pop_back();
    const auto openTree=buildStaticCompleteBodyBVH(openTriangles,openTriangles.size());
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> openRanges;
    require(openTree.succeeded() && makeStaticCompleteBodyContactRanges(
                std::span(&boxSample,1u),openTree.bodies,0.02f,openRanges) &&
            openRanges[0].reserved0==kRangeFlagStaticCompleteBVH,
            "open bank keeps uncertified complete-tree range");
    openRanges[0].reserved0=kRangeFlagStaticClosedOrientedBVH;
    require(!validCompleteStaticBVHSpans(std::span(&boxSample,1u),openRanges,
                openTree.triangles,openTree.nodes),
            "forged closed certificate on open bank is rejected");
    const nm_float4 lo{-1,-1,-1,0},hi{1,1,1,0};
    require(closedAABBProvesInactive({1.1f,0,0,0},lo,hi,0.02f) &&
            closedAABBProvesInactive({1.04f,1.04f,0,0},lo,hi,0.02f) &&
            closedAABBProvesInactive({1.04f,1.04f,1.04f,0},lo,hi,0.02f),
            "certified exterior AABB distance covers face, edge, and corner queries");
    require(!closedAABBProvesInactive({1.02f,0,0,0},lo,hi,0.02f) &&
            !closedAABBProvesInactive({std::nextafter(1.0f,2.0f),0,0,0},lo,hi,0.02f) &&
            !closedAABBProvesInactive({0,0,0,0},lo,hi,0.02f),
            "band boundary and one-ULP boundary remain on full-query path");
    require(!closedAABBProvesInactive({0,0,0,0},lo,hi,0.02f),
            "inside-AABB concave exteriors cannot use the fast path");
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
    // Closed outward tetrahedron exercises exact-seam edge and angle-weighted
    // vertex pseudonormals. The feature fields are absent on the legacy open
    // inventory above, so the default path retains byte-identical behavior.
    const std::array<nm_float4,4> v{{
        {0,0,0,0},{1,0,0,0},{0,1,0,0},{0,0,1,0}}};
    const auto face=[&](unsigned a,unsigned b,unsigned c,std::uint32_t id) {
        const double ux=double(v[b].x)-v[a].x, uy=double(v[b].y)-v[a].y,
            uz=double(v[b].z)-v[a].z;
        const double vx=double(v[c].x)-v[a].x, vy=double(v[c].y)-v[a].y,
            vz=double(v[c].z)-v[a].z;
        const double nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
        const double n=std::sqrt(nx*nx+ny*ny+nz*nz);
        NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU out{};
        out.bodyIndex=31u; out.boneStableId=7u; out.triangleStableId=id;
        out.bodyLocalVertex0=v[a]; out.bodyLocalVertex1=v[b]; out.bodyLocalVertex2=v[c];
        out.authoredOutwardNormal={float(nx/n),float(ny/n),float(nz/n),0};
        return out;
    };
    const std::array<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU,4> tetra{{
        face(0,2,1,0), face(0,3,2,1), face(0,1,3,2), face(1,2,3,3)}};
    const auto strict=buildStaticCompleteBodyBVH(tetra,tetra.size(),true);
    require(strict.succeeded(),"closed positively oriented bank emits feature pseudonormals");
    NMNumiHumanFEMBodyContactSampleGPU closedTemplate{};
    closedTemplate.bodyIndex=31u;
    closedTemplate.flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
        NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT;
    closedTemplate.slaveNode=300u;
    closedTemplate.bodyLocalPointAndArea={0.1f,0.1f,0.1f,0.2f};
    closedTemplate.bodyLocalNormalAndReferenceSeparation={0.0f,0.0f,1.0f,0.0f};
    closedTemplate.stiffnessAndNormalStrainPerPressure={1.0f,0.0f,0.0f,1.0f};
    numi::matter::fem_body_contact_bvh::PreparationOptions closedOptions{};
    closedOptions.requireClosedOrientedBanks=true;
    const auto strictPrepared=prepareStaticCompleteBodyContactAsset(
        std::span<const NMNumiHumanFEMBodyContactSampleGPU>(&closedTemplate,1u),
        tetra,tetra.size(),0.02f,closedOptions);
    require(strictPrepared.succeeded()&&strictPrepared.samples.size()==1u&&
            strictPrepared.triangles.size()==tetra.size()&&
            strictPrepared.triangles[0].featureNormalFlags==
                NM_NUMI_HUMAN_FEM_BODY_CONTACT_ALL_FEATURE_PSEUDONORMALS,
            "named strict option validates closure then emits edge and vertex pseudonormals");

    for (const auto& t:strict.triangles)
        require(t.featureNormalFlags==NM_NUMI_HUMAN_FEM_BODY_CONTACT_ALL_FEATURE_PSEUDONORMALS,
                "every closed-bank triangle has all six feature normals");
    const auto edge=std::find_if(strict.triangles.begin(),strict.triangles.end(),
        [](const auto& t) { return t.triangleStableId==0u; });
    require(edge!=strict.triangles.end() && edge->edgePseudoNormal20.y<-.70f &&
            edge->edgePseudoNormal20.z<-.70f &&
            std::abs(edge->edgePseudoNormal20.x)<1.0e-6f,
            "edge pseudo normal averages the two adjacent outward face normals");
    const auto vertex=std::find_if(strict.triangles.begin(),strict.triangles.end(),
        [](const auto& t) { return t.triangleStableId==2u; });
    require(vertex!=strict.triangles.end() && vertex->vertexPseudoNormal0.x<-.45f &&
            vertex->vertexPseudoNormal0.y<-.45f &&
            vertex->vertexPseudoNormal0.z<-.45f,
            "corner-angle weighted vertex pseudo normal points out of convex vertex");
    auto open=tetra; open[3].bodyLocalVertex0={2,2,2,0};
    {
        const auto& a=open[3].bodyLocalVertex0;
        const auto& b=open[3].bodyLocalVertex1;
        const auto& c=open[3].bodyLocalVertex2;
        const double ux=double(b.x)-a.x,uy=double(b.y)-a.y,uz=double(b.z)-a.z;
        const double vx=double(c.x)-a.x,vy=double(c.y)-a.y,vz=double(c.z)-a.z;
        const double nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
        const double n=std::sqrt(nx*nx+ny*ny+nz*nz);
        open[3].authoredOutwardNormal={float(nx/n),float(ny/n),float(nz/n),0};
    }
    require(buildStaticCompleteBodyBVH(open,open.size(),true).failure==
            BuildFailure::invalidClosedOrientedBank,
            "open bank fails strict complete-bank admission");
    auto inward=tetra;
    for (auto& t:inward) {
        std::swap(t.bodyLocalVertex1,t.bodyLocalVertex2);
        t.authoredOutwardNormal.x=-t.authoredOutwardNormal.x;
        t.authoredOutwardNormal.y=-t.authoredOutwardNormal.y;
        t.authoredOutwardNormal.z=-t.authoredOutwardNormal.z;
    }
    require(buildStaticCompleteBodyBVH(inward,inward.size(),true).failure==
            BuildFailure::invalidClosedOrientedBank,
            "closed inward-wound bank is rejected without flipping faces");
    auto pinched=tetra;
    pinched[3].bodyLocalVertex0={0,0,0,0};
    pinched[3].bodyLocalVertex1={-1,0,0,0};
    pinched[3].bodyLocalVertex2={0,-1,0,0};
    pinched[3].authoredOutwardNormal={0,0,1,0};
    require(buildStaticCompleteBodyBVH(pinched,pinched.size(),true).failure==
            BuildFailure::invalidClosedOrientedBank,
            "pinched/nonmanifold vertex bank rejected");

    using numi::human::initial_skin_bone_contact_witness::Closest;
    using numi::human::initial_skin_bone_contact_witness::Feature;
    using numi::human::initial_skin_bone_contact_witness::Vec3;
    using numi::human::initial_skin_bone_contact_witness::closestPointOnTriangle;
    using numi::human::initial_skin_bone_contact_witness::featurePseudoNormal;
    const auto testSign=[&](const auto& t,Vec3 query,Feature expected,double sign,
                            const char* message) {
        const auto cp=closestPointOnTriangle(query,
            {t.bodyLocalVertex0.x,t.bodyLocalVertex0.y,t.bodyLocalVertex0.z},
            {t.bodyLocalVertex1.x,t.bodyLocalVertex1.y,t.bodyLocalVertex1.z},
            {t.bodyLocalVertex2.x,t.bodyLocalVertex2.y,t.bodyLocalVertex2.z});
        Vec3 pseudo{};
        require(cp.valid&&cp.feature==expected&&featurePseudoNormal(t,cp,pseudo),message);
        const double d=(query.x-cp.point.x)*pseudo.x+
            (query.y-cp.point.y)*pseudo.y+(query.z-cp.point.z)*pseudo.z;
        require(sign*d>0.0,message);
    };
    require(edge!=strict.triangles.end(),"convex edge face remains available");
    Closest convexEdge{};convexEdge.feature=Feature::edge;
    convexEdge.bary={0.5,0.0,0.5};convexEdge.point={0.5,0.0,0.0};
    Vec3 convexEdgePseudo{};
    require(featurePseudoNormal(*edge,convexEdge,convexEdgePseudo),
            "convex edge feature resolves its edge pseudonormal");
    require(dot({0.5,-0.02,-0.02},convexEdgePseudo)>0.0 &&
            dot({0.5,0.02,0.02},convexEdgePseudo)<0.0,
            "convex edge pseudonormal signs exterior and interior sides");
    require(vertex!=strict.triangles.end(),"convex vertex face remains available");
    Closest convexVertex{};convexVertex.feature=Feature::vertex;
    convexVertex.bary={1.0,0.0,0.0};convexVertex.point={0.0,0.0,0.0};
    Vec3 convexVertexPseudo{};
    require(featurePseudoNormal(*vertex,convexVertex,convexVertexPseudo),
            "convex vertex feature resolves its vertex pseudonormal");
    require(dot({-0.02,-0.02,-0.02},convexVertexPseudo)>0.0 &&
            dot({0.02,0.02,0.02},convexVertexPseudo)<0.0,
            "convex vertex pseudonormal signs exterior and interior sides");
    const auto facePlane=std::find_if(strict.triangles.begin(),strict.triangles.end(),
        [](const auto& t) { return t.triangleStableId==1u; });
    require(facePlane!=strict.triangles.end(),"convex face sample triangle exists");
    testSign(*facePlane,{-0.02,0.2,0.2},Feature::face,1.0,
             "convex face marks exterior positive");
    testSign(*facePlane,{0.02,0.2,0.2},Feature::face,-1.0,
             "convex face marks interior negative");

    // Triangulate an L-shaped prism by deterministic CCW ear clipping; its
    // reflex vertical edge has the angle-weighted sum of both outward walls.
    const std::array<std::array<double,2>,7> polygon{{
        {{0,0}},{{2,0}},{{2,1}},{{1,1}},{{1,2}},{{0,2}},{{0,1}}}};
    std::vector<std::array<unsigned,3>> topFaces;
    std::vector<unsigned> ring{0,1,2,3,4,5,6};
    const auto cross2=[&](unsigned a,unsigned b,unsigned c) {
        const auto& A=polygon[a];const auto& B=polygon[b];const auto& C=polygon[c];
        return (B[0]-A[0])*(C[1]-A[1])-(B[1]-A[1])*(C[0]-A[0]);
    };
    while(ring.size()>3u) {
        bool clipped=false;
        for(std::size_t i=0;i<ring.size();++i) {
            const unsigned a=ring[(i+ring.size()-1u)%ring.size()],b=ring[i],
                           c=ring[(i+1u)%ring.size()];
            if(cross2(a,b,c)<=0.0)continue;
            bool contains=false;
            for(const unsigned q:ring)if(q!=a&&q!=b&&q!=c) {
                const double x=cross2(a,b,q),y=cross2(b,c,q),z=cross2(c,a,q);
                if(x>=0.0&&y>=0.0&&z>=0.0){contains=true;break;}
            }
            if(contains)continue;
            topFaces.push_back({a,b,c});
            ring.erase(ring.begin()+static_cast<std::ptrdiff_t>(i));
            clipped=true;break;
        }
        require(clipped,"L prism polygon has a deterministic ear");
    }
    topFaces.push_back({ring[0],ring[1],ring[2]});
    std::array<nm_float4,14> prismVertices{};
    for(unsigned i=0;i<7u;++i) {
        prismVertices[i]={float(polygon[i][0]),float(polygon[i][1]),0.0f,0.0f};
        prismVertices[7u+i]={float(polygon[i][0]),float(polygon[i][1]),1.0f,0.0f};
    }
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> prism;
    std::uint32_t prismId=0u;
    const auto emit=[&](unsigned a,unsigned b,unsigned c) {
        NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU t{};
        t.bodyIndex=32u;t.boneStableId=8u;t.triangleStableId=prismId++;
        t.bodyLocalVertex0=prismVertices[a];t.bodyLocalVertex1=prismVertices[b];
        t.bodyLocalVertex2=prismVertices[c];
        const double ux=double(t.bodyLocalVertex1.x)-t.bodyLocalVertex0.x,
            uy=double(t.bodyLocalVertex1.y)-t.bodyLocalVertex0.y,
            uz=double(t.bodyLocalVertex1.z)-t.bodyLocalVertex0.z;
        const double vx=double(t.bodyLocalVertex2.x)-t.bodyLocalVertex0.x,
            vy=double(t.bodyLocalVertex2.y)-t.bodyLocalVertex0.y,
            vz=double(t.bodyLocalVertex2.z)-t.bodyLocalVertex0.z;
        const double nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
        const double n=std::sqrt(nx*nx+ny*ny+nz*nz);
        require(n>0.0,"L prism triangulation emits nondegenerate faces");
        t.authoredOutwardNormal={float(nx/n),float(ny/n),float(nz/n),0.0f};
        prism.push_back(t);
    };
    for(const auto& f:topFaces) {
        emit(7u+f[0],7u+f[1],7u+f[2]);
        emit(f[2],f[1],f[0]);
    }
    for(unsigned i=0;i<7u;++i) {
        const unsigned j=(i+1u)%7u;
        emit(i,j,7u+j);emit(i,7u+j,7u+i);
    }
    const auto concave=numi::matter::fem_body_contact_bvh::
        buildStaticCompleteBodyBVH(prism,prism.size(),true);
    require(concave.succeeded(),"L-shaped concave closed prism passes exact topology/orientation");
    const auto samePoint=[](nm_float4 a,nm_float4 b) {
        return a.x==b.x&&a.y==b.y&&a.z==b.z;
    };
    const nm_float4 reflexBottom=prismVertices[3],reflexTop=prismVertices[10];
    bool testedReflexEdge=false,testedReflexVertex=false;
    for(const auto& t:concave.triangles) {
        const std::array<nm_float4,3> p{{t.bodyLocalVertex0,t.bodyLocalVertex1,t.bodyLocalVertex2}};
        for(unsigned e=0;e<3u;++e) {
            const unsigned j=(e+1u)%3u;
            if((samePoint(p[e],reflexBottom)&&samePoint(p[j],reflexTop))||
               (samePoint(p[j],reflexBottom)&&samePoint(p[e],reflexTop))) {
                Closest cp{};cp.feature=Feature::edge;
                cp.bary=e==0u?std::array<double,3>{0.5,0.5,0.0}:
                    (e==1u?std::array<double,3>{0.0,0.5,0.5}:
                           std::array<double,3>{0.5,0.0,0.5});
                Vec3 pseudo{};
                require(featurePseudoNormal(t,cp,pseudo),"concave reflex edge has pseudo normal");
                const double outside=(1.05-1.0)*pseudo.x+(1.05-1.0)*pseudo.y;
                const double inside=(0.95-1.0)*pseudo.x+(0.95-1.0)*pseudo.y;
                require(outside>0.0&&inside<0.0,
                        "concave edge pseudonormal separates notch exterior from solid interior");
                testedReflexEdge=true;
            }
        }
        for(unsigned vtx=0;vtx<3u;++vtx)if(samePoint(p[vtx],reflexTop)) {
            Closest cp{};cp.feature=Feature::vertex;cp.bary[vtx]=1.0;
            Vec3 pseudo{};
            require(featurePseudoNormal(t,cp,pseudo),"concave reflex vertex has pseudo normal");
            const double outside=0.05*(pseudo.x+pseudo.y+pseudo.z);
            const double inside=-outside;
            require(outside>0.0&&inside<0.0,
                    "concave vertex pseudonormal separates exterior from solid interior");
            testedReflexVertex=true;
        }
    }
    require(testedReflexEdge&&testedReflexVertex,
            "L prism exposes its concave edge and corner feature normals");
    std::puts("static complete body BVH CPU regression PASS");
}

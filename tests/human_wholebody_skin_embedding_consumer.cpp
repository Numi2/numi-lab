#include "NumiHumanWholeBodySkinEmbeddingConsumer.hpp"
#include <cassert>
#include <iostream>

using namespace metalrobo::human::coarse_tissue;
struct Vtx { V3 p{}; float weight[4]{}; std::uint32_t bindingIndex[4]{}; };
struct Binding { std::uint32_t bodyIndex=0; };
struct Skin {
    std::vector<Vtx> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<Binding> bindings;
    std::vector<float> fullWeights;
};
int main() {
    Skin skin;
    skin.vertices.resize(4);
    skin.vertices[0].p={0,0,0}; skin.vertices[1].p={1,0,0};
    skin.vertices[2].p={0,1,0}; skin.vertices[3].p={1,1,0};
    skin.indices={0,1,2, 1,3,2};
    skin.bindings.resize(86);
    for(std::uint32_t b=0;b<86;++b) skin.bindings[b].bodyIndex=b;
    skin.fullWeights.assign(4u*86u,0.0f);
    for(std::size_t v=0;v<4;++v) skin.fullWeights[v*86u]=1.0f;
    // A projected row spans source corners with different skeletal owners.
    skin.fullWeights[3u*86u] = 0.2f;
    skin.fullWeights[3u*86u+1u] = 0.8f;
    RegisteredSurface region;
    region.worldPoints={{0,0,0},{0.5,0.5,0},{0,1,0}};
    region.sourceVertexIndices={0u,NM_INVALID_INDEX,2u};
    region.donorBodyIndices={0u,0u,0u};
    region.sourceEmbeddingFaceIndices={NM_INVALID_INDEX,1u,NM_INVALID_INDEX};
    region.sourceEmbeddingBarycentric={{0,0,0},{0.25,0.5,0.25},{0,0,0}};
    std::vector<std::uint8_t> used{1,1,1,1};
    WholeBodySkinSourceConsumerRows rows;
    std::string error;
    auto positionAt=[](std::uint32_t,const Vtx& v){return v.p;};
    if(!makeWholeBodySkinSourceConsumerRows(
            skin,region,positionAt,std::span<const std::uint8_t>(used),rows,error)) {
        std::cerr<<error<<"\n"; return 1;
    }
    assert(rows.directSourceIDs.size()==3u);
    assert(rows.directSourceIDs[0]==0u);
    assert(rows.directSourceIDs[1]==NM_INVALID_INDEX);
    assert(rows.directSourceIDs[2]==2u);
    assert(rows.requiredSourceIDs==std::vector<std::uint32_t>({0u,1u,2u,3u}));
    assert(rows.fullBodyWeights86[0][0]==1.0);
    assert(std::abs(rows.fullBodyWeights86[1][0]-0.6)<1e-7);
    assert(std::abs(rows.fullBodyWeights86[1][1]-0.4)<1e-7);
    assert(rows.dominantBody==std::vector<std::uint32_t>({0u,0u,0u}));

    // Failed rebuilds must discard prior rows and identify invalid ancestry.
    const auto goodRegion = region;
    region.sourceEmbeddingFaceIndices[1] = 2u;
    assert(!makeWholeBodySkinSourceConsumerRows(skin,region,positionAt,used,rows,error));
    assert(rows.sourceFaces.empty() && !error.empty());
    region = goodRegion;
    region.sourceEmbeddingBarycentric[1] = {0.25,0.5,0.5};
    assert(!makeWholeBodySkinSourceConsumerRows(skin,region,positionAt,used,rows,error));
    assert(rows.sourceFaces.empty() && !error.empty());
    region = goodRegion;
    region.donorBodyIndices[1] = 1u;
    assert(!makeWholeBodySkinSourceConsumerRows(skin,region,positionAt,used,rows,error));
    assert(rows.sourceFaces.empty() && !error.empty());
    region = goodRegion;
    auto inactive = used;
    inactive[3] = 0u;
    assert(!makeWholeBodySkinSourceConsumerRows(skin,region,positionAt,inactive,rows,error));
    assert(rows.sourceFaces.empty() && !error.empty());
    assert(makeWholeBodySkinSourceConsumerRows(skin,region,positionAt,used,rows,error));
    assert(error.empty());

    using Transfer=numi::human::AcceptedFEMSkinTransfer;
    std::vector<Transfer::Point> detailRest{{0,0,0},{1,0,0},{0,1,0},{1,1,0}};
    std::vector<std::uint32_t> detailSource{0,1,2,3}, detailDonor{0,0,0,1};
    std::vector<std::uint8_t> detailUsed{1,1,1,1};
    std::vector<Transfer::Point> coarseRest{{0,0,0},{0.5,0.5,0},{0,1,0}};
    std::vector<std::uint32_t> coarseDonor{0,0,0};
    std::vector<Transfer::Face> faces{{0,1,2}};
    const std::vector<std::uint32_t> representative{0,1,2,1};
    Transfer::BuildReport report;
    Transfer transfer;
    if(!transfer.build(detailRest,detailSource,detailDonor,detailUsed,
                       coarseRest,rows.directSourceIDs,coarseDonor,faces,
                       &report,&error,std::numeric_limits<double>::infinity(),
                       &representative,&rows.sourceFaces,&rows.sourceBarycentric)) {
        std::cerr<<error<<"\n"; return 2;
    }
    std::vector<Transfer::Point> current=detailRest, baseline;
    std::vector<Transfer::Pose> poses(2);
    if(!transfer.makeCurrentCoarseBaseline(detailRest,current,poses,poses,baseline,&error)) {
        std::cerr<<error<<"\n"; return 3;
    }
    assert(baseline.size()==3u);
    assert(std::abs(baseline[0][0]-coarseRest[0][0])<1e-12);
    assert(std::abs(baseline[1][0]-coarseRest[1][0])<1e-12);
    assert(std::abs(baseline[1][1]-coarseRest[1][1])<1e-12);
    std::cout<<"source-ID consumer fixture passed: direct=2 projected=1 required=4 full86=normalized transfer=3\n";
}

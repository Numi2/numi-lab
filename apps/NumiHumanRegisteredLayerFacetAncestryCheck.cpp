#include "NumiHumanCoarseTissuePreparation.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>
using namespace metalrobo::human::coarse_tissue;
// Focused, data-independent regression for exact RegisteredSurface facet identity
// and its boundary-area consumer. The parent-volume CSR contract validates sorted
// contributors, material compatibility, positive fractions, per-child fraction
// closure, and aggregate parent-volume closure. It does NOT recompute geometric
// child/parent overlap. This test therefore makes no geometric-volume-ancestry
// certification claim; that remains the responsibility of the candidate builder
// or a separate overlap audit.
int main() {
    constexpr std::uint32_t none=std::numeric_limits<std::uint32_t>::max();
    constexpr std::size_t n=4u;
    const std::vector<Face> registered{{0u,1u,2u}};
    const std::vector<std::uint32_t> stable{900u};
    // This D-layer child facet is exactly registered row 0 (stable 900), while
    // its cell's independent dominant-volume donor stable identity may be 570.
    const std::vector<std::array<std::uint32_t,4u>> children{{8u,9u,10u,11u}};
    const std::vector<std::uint32_t> facetRows{none,none,none,0u};
    const std::vector<std::uint32_t> facetStable{none,none,none,900u};
    std::string error;
    assert(validateExplicitRegisteredLayerFacetAncestry(children,n,registered,stable,
        facetRows,facetStable,error));
    // Old aggregate callers remain supported when both optional arrays are absent.
    assert(validateExplicitRegisteredLayerFacetAncestry(children,n,registered,stable,{}, {},error));
    auto badStable=facetStable; badStable[3]=570u;
    assert(!validateExplicitRegisteredLayerFacetAncestry(children,n,registered,stable,
        facetRows,badStable,error));
    assert(error.find("child=0 facet=3")!=std::string::npos && error.find("exact_stable_id=900")!=std::string::npos);
    auto badRow=facetRows; badRow[3]=1u;
    assert(!validateExplicitRegisteredLayerFacetAncestry(children,n,registered,stable,
        badRow,facetStable,error));
    assert(error.find("exact_row=0")!=std::string::npos);
    auto omitted=facetRows; omitted[3]=none;
    auto omittedStable=facetStable; omittedStable[3]=none;
    assert(!validateExplicitRegisteredLayerFacetAncestry(children,n,registered,stable,
        omitted,omittedStable,error));
    auto fabricated=facetRows; fabricated[0]=0u;
    auto fabricatedStable=facetStable; fabricatedStable[0]=900u;
    assert(!validateExplicitRegisteredLayerFacetAncestry(children,n,registered,stable,
        fabricated,fabricatedStable,error));
    assert(error.find("no exact RegisteredSurface triangle")!=std::string::npos);

    // Real D36 global-boundary witness from generated/tetrahedra.csv and faces.csv:
    // child 28526 has dominant-volume stable ID 98039, but its D-layer facet is
    // RegisteredSurface row 2413 / stable ID 98567. This exercises the exact
    // boundary-owner selection consumed by addTet, not merely the validator.
    constexpr std::size_t d36NodesPerLayer=2411u;
    std::vector<Face> d36RegisteredFaces;
    d36RegisteredFaces.reserve(2414u);
    for(std::uint32_t i=0u;i<2411u;++i)
        d36RegisteredFaces.push_back({i,(i+1u)%2411u,(i+2u)%2411u});
    d36RegisteredFaces.push_back({0u,2u,4u});
    d36RegisteredFaces.push_back({3u,5u,7u});
    d36RegisteredFaces.push_back({1107u,1233u,1288u});
    assert(d36RegisteredFaces.size()==2414u);
    std::vector<std::uint32_t> d36StableFaces(d36RegisteredFaces.size());
    for(std::uint32_t i=0u;i<d36StableFaces.size();++i)d36StableFaces[i]=1000000u+i;
    d36StableFaces[2413u]=98567u;
    const std::vector<std::array<std::uint32_t,4u>> d36Child{{5929u,6055u,6110u,0u}};
    const std::vector<std::uint32_t> d36FacetRows{none,none,none,2413u};
    const std::vector<std::uint32_t> d36FacetStable{none,none,none,98567u};
    assert(validateExplicitRegisteredLayerFacetAncestry(d36Child,d36NodesPerLayer,
        d36RegisteredFaces,d36StableFaces,d36FacetRows,d36FacetStable,error));
    const std::vector<std::array<std::uint32_t,4u>> d36MaterialChildren{
        d36Child[0],{100u,101u,102u,2200u}};
    const std::vector<std::uint32_t> d36MaterialFacetRows{
        none,none,none,2413u,none,none,none,100u};
    const std::vector<std::uint32_t> d36MaterialFacetStable{
        none,none,none,98567u,none,none,none,1000100u};
    assert(validateExplicitRegisteredLayerFacetAncestry(d36MaterialChildren,d36NodesPerLayer,
        d36RegisteredFaces,d36StableFaces,d36MaterialFacetRows,d36MaterialFacetStable,error));
    const std::vector<std::uint32_t> d36MaterialOrder{1u,0u};
    std::vector<std::uint32_t> d36FilteredRows,d36FilteredStable;
    assert(filterExplicitChildLayerFacetAncestry(d36MaterialFacetRows,d36MaterialFacetStable,
        d36MaterialOrder,d36FilteredRows,d36FilteredStable,error));
    const std::vector<std::array<std::uint32_t,4u>> d36MaterialChildrenReordered{
        d36MaterialChildren[1],d36MaterialChildren[0]};
    assert(validateExplicitRegisteredLayerFacetAncestry(d36MaterialChildrenReordered,
        d36NodesPerLayer,d36RegisteredFaces,d36StableFaces,d36FilteredRows,d36FilteredStable,error));
    assert(d36FilteredRows[7u]==2413u&&d36FilteredStable[7u]==98567u);
    const std::span<const std::uint32_t,4u> d36FacetOwnerRows(d36FilteredRows.data()+4u,4u);
    const auto consumedBoundaryOwner=explicitBoundaryFacetOwnerFace(4304u,d36FacetOwnerRows,3u);
    assert(consumedBoundaryOwner==2413u && consumedBoundaryOwner!=4304u);

    // The consumer must use the exact facet owner during boundary-area coverage.
    // Two disjoint registered triangles form two small prism shells. The second
    // shell's scalar donor intentionally points at row 0, while its exact outer /
    // inner boundary facets belong to RegisteredSurface row 1.
    const std::vector<V3> consumerOuter{
        V3{0.0,0.0,0.0},V3{1.0,0.0,0.0},V3{0.0,1.0,0.0},
        V3{3.0,0.0,0.0},V3{4.0,0.0,0.0},V3{3.0,1.0,0.0}};
    std::vector<V3> consumerInner=consumerOuter;
    for(auto& point:consumerInner)point[2]=-0.1;
    const std::vector<Face> consumerSourceFaces{{0u,1u,2u},{3u,4u,5u}};
    std::map<Face,std::uint32_t> consumerFaceRows;
    for(std::uint32_t row=0u;row<consumerSourceFaces.size();++row){
        auto key=consumerSourceFaces[row];std::sort(key.begin(),key.end());
        consumerFaceRows.emplace(key,row);
    }
    const unsigned tetFacets[4][3]={{1u,2u,3u},{0u,3u,2u},{0u,1u,3u},{0u,2u,1u}};
    auto buildConsumerBoundary=[&](bool useExactFacetOwners,
            std::map<std::array<std::uint32_t,3>,std::uint32_t>& owners,
            std::map<std::array<std::uint32_t,3>,std::array<std::uint32_t,3>>& oriented,
            std::vector<std::array<std::uint32_t,3>>& boundary){
        std::map<std::array<std::uint32_t,3>,unsigned> uses;
        for(std::uint32_t triangle=0u;triangle<2u;++triangle){
            const auto base=3u*triangle;const auto a=base,b=base+1u,c=base+2u;
            std::array<std::array<std::uint32_t,4u>,3u> tets{{
                {a,b,c,static_cast<std::uint32_t>(6u+c)},
                {a,b,static_cast<std::uint32_t>(6u+b),static_cast<std::uint32_t>(6u+c)},
                {a,static_cast<std::uint32_t>(6u+a),static_cast<std::uint32_t>(6u+b),static_cast<std::uint32_t>(6u+c)}}};
            for(auto tet:tets){
                const auto point=[&](std::uint32_t id)->const V3&{return id<6u?consumerOuter[id]:consumerInner[id-6u];};
                if(tetVolume(point(tet[0]),point(tet[1]),point(tet[2]),point(tet[3]))<0.0)std::swap(tet[0],tet[1]);
                std::array<std::uint32_t,4u> perFacetOwners{none,none,none,none};
                if(useExactFacetOwners)for(unsigned omitted=0u;omitted<4u;++omitted){
                    std::array<std::uint32_t,3u> tri{};std::size_t at=0u;bool allOuter=true,allInner=true;
                    for(unsigned corner=0u;corner<4u;++corner)if(corner!=omitted){
                        const auto node=tet[corner];allOuter=allOuter&&node<6u;allInner=allInner&&node>=6u&&node<12u;
                        tri[at++]=node<6u?node:node-6u;
                    }
                    if(allOuter||allInner){std::sort(tri.begin(),tri.end());const auto found=consumerFaceRows.find(tri);
                        assert(found!=consumerFaceRows.end());perFacetOwners[omitted]=found->second;}
                }
                for(unsigned omitted=0u;omitted<4u;++omitted){
                    std::array<std::uint32_t,3u> face{};
                    for(unsigned j=0u;j<3u;++j)face[j]=tet[tetFacets[omitted][j]];
                    auto key=face;std::sort(key.begin(),key.end());++uses[key];
                    const std::span<const std::uint32_t,4u> facetOwners(perFacetOwners);
                    const auto owner=explicitBoundaryFacetOwnerFace(0u,facetOwners,omitted);
                    owners.try_emplace(key,owner);oriented.try_emplace(key,face);
                }
            }
        }
        boundary.clear();for(const auto& [face,count]:uses)if(count==1u)boundary.push_back(face);
    };
    std::map<std::array<std::uint32_t,3>,std::uint32_t> exactOwners,donorOnlyOwners;
    std::map<std::array<std::uint32_t,3>,std::array<std::uint32_t,3>> exactOriented,donorOnlyOriented;
    std::vector<std::array<std::uint32_t,3>> exactBoundary,donorOnlyBoundary;
    buildConsumerBoundary(true,exactOwners,exactOriented,exactBoundary);
    buildConsumerBoundary(false,donorOnlyOwners,donorOnlyOriented,donorOnlyBoundary);
    assert(validateExplicitBoundaryAreaCoverage(consumerOuter,consumerInner,consumerSourceFaces,
        exactOwners,exactOriented,exactBoundary,6u,error));
    assert(!validateExplicitBoundaryAreaCoverage(consumerOuter,consumerInner,consumerSourceFaces,
        donorOnlyOwners,donorOnlyOriented,donorOnlyBoundary,6u,error));
    assert(!error.empty());

    // Skin/fat partitioning can reorder child rows. Filtering preserves all four
    // facet rows/stable IDs in the requested material-local order.
    const std::vector<std::uint32_t> sourceFacetRows{10u,none,12u,13u,20u,21u,none,23u};
    const std::vector<std::uint32_t> sourceFacetStable{110u,none,112u,113u,120u,121u,none,123u};
    const std::vector<std::uint32_t> selectedChildRows{1u,0u};
    std::vector<std::uint32_t> filteredFacetRows,filteredFacetStable;
    assert(filterExplicitChildLayerFacetAncestry(sourceFacetRows,sourceFacetStable,
        selectedChildRows,filteredFacetRows,filteredFacetStable,error));
    assert((filteredFacetRows==std::vector<std::uint32_t>{20u,21u,none,23u,10u,none,12u,13u}));
    assert((filteredFacetStable==std::vector<std::uint32_t>{120u,121u,none,123u,110u,none,112u,113u}));
    const std::span<const std::uint32_t,4u> filteredFirstOwners(filteredFacetRows.data(),4u);
    assert(explicitBoundaryFacetOwnerFace(999u,filteredFirstOwners,0u)==20u);
    assert(explicitBoundaryFacetOwnerFace(999u,filteredFirstOwners,2u)==999u);
    std::vector<std::uint32_t> badFilteredRows,badFilteredStable;
    const std::vector<std::uint32_t> duplicateSelection{1u,1u};
    assert(!filterExplicitChildLayerFacetAncestry(sourceFacetRows,sourceFacetStable,
        duplicateSelection,badFilteredRows,badFilteredStable,error));
    const std::vector<std::uint32_t> outOfRangeSelection{2u};
    assert(!filterExplicitChildLayerFacetAncestry(sourceFacetRows,sourceFacetStable,
        outOfRangeSelection,badFilteredRows,badFilteredStable,error));
    auto badMissingStable=sourceFacetStable;badMissingStable[2]=none;
    assert(!filterExplicitChildLayerFacetAncestry(sourceFacetRows,badMissingStable,
        selectedChildRows,badFilteredRows,badFilteredStable,error));

    std::cout << "registered_layer_facet_ancestry=pass dominant_volume_identity_independent=pass exact_row_and_stable_checks=pass material_filter_order=pass d36_boundary_consumer=pass boundary_area_owner_consumer=pass legacy_empty=pass volume_csr_geometric_overlap=not_checked\n";
}

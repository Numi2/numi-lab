#include "cardboard_shipping_closure_mesh.hpp"
#include "cardboard_shipping_protocol.hpp"
#include "cardboard_shipping_driver.hpp"
#include "cardboard_starch_interface_mesh.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace numi::cardboard;

static std::uint32_t opposite(const std::array<std::uint32_t,4>& tet,
                              const std::array<std::uint32_t,3>& face) {
    auto f=face; std::sort(f.begin(),f.end());
    for (std::uint32_t o=0;o<4;++o) {
        std::array<std::uint32_t,3> candidate{}; std::uint32_t k=0;
        for(std::uint32_t i=0;i<4;++i) if(i!=o) candidate[k++]=tet[i];
        std::sort(candidate.begin(),candidate.end());
        if(candidate==f) return o;
    }
    throw std::runtime_error("HMA face does not match tetrahedron");
}

int main() {
    ShippingBoxDesignDefaults defaults; defaults.subdivisionsPerBand=16u;
    const std::array<double,3> zero{0.,0.,0.};
    const auto selected=selectShippingDesign(defaults,nullptr,&zero,false);
    ShippingCorrugatedRecipe recipe; recipe.nxPerPitch=8u; recipe.thicknessSlices=1u;
    const auto geometry=buildShippingBoxBlankGeometry(selected.design,recipe);
    StarchInterfaceMesh starchSource;
    for(const auto& p:geometry.blank.points) starchSource.referenceNodesM.push_back({p.x,p.y,p.z});
    for(const auto& st:geometry.blank.tetrahedra) {
        StarchInterfaceTet t;
        for(std::size_t i=0;i<4;++i) t.nodeIds[i]=st.nodes[i];
        t.materialIndex=static_cast<std::uint8_t>(st.material);
        t.densityKgPerM3=t.materialIndex==kStarchMaterialLiner?766.0:
            (t.materialIndex==kStarchMaterialMedium?679.0:1500.0);
        starchSource.tetrahedra.push_back(t);
    }
    const auto starch=prepareStarchInterfaceMesh(starchSource);
    std::vector<BoxBlankPoint> nodes;
    for(const auto& p:starch.referenceNodesM) nodes.push_back({p.x,p.y,p.z});
    std::vector<ShippingClosureTet> tets;
    for(const auto& st:starch.tetrahedra) {
        ShippingClosureTet t;
        for(std::size_t i=0;i<4;++i) t.nodes[i]=static_cast<std::uint32_t>(st.nodeIds[i]);
        t.materialIndex=st.materialIndex; tets.push_back(t);
    }
    const auto hma=buildShippingClosureMortarMesh(geometry.blank,nodes,tets,
        0.010,0.0001,1u,static_cast<std::uint32_t>(starch.pairedFaces.size()+1u));
    using Pair=std::pair<std::uint32_t,std::uint32_t>;
    std::map<Pair,std::vector<std::pair<const char*,std::uint32_t>>> ids;
    for(std::uint32_t i=0;i<starch.pairedFaces.size();++i) {
        const auto& p=starch.pairedFaces[i];
        std::array<std::uint32_t,4> aTet{}, bTet{};
        for(std::size_t c=0;c<4;++c) {
            aTet[c]=static_cast<std::uint32_t>(starch.tetrahedra.at(p.paperTetId).nodeIds[c]);
            bTet[c]=static_cast<std::uint32_t>(starch.tetrahedra.at(p.glueTetId).nodeIds[c]);
        }
        std::array<std::uint32_t,3> aFace{}, bFace{};
        for(std::size_t c=0;c<3;++c) {
            aFace[c]=static_cast<std::uint32_t>(p.paperNodeIds[c]);
            bFace[c]=static_cast<std::uint32_t>(p.glueNodeIds[c]);
        }
        const auto a=4u*static_cast<std::uint32_t>(p.paperTetId)+opposite(aTet,aFace);
        const auto b=4u*static_cast<std::uint32_t>(p.glueTetId)+opposite(bTet,bFace);
        ids[std::minmax(a,b)].push_back({"starch",i});
    }
    for(std::uint32_t i=0;i<hma.patches.size();++i) {
        const auto& p=hma.patches[i];
        const auto a=4u*p.firstTetrahedron+opposite(tets.at(p.firstTetrahedron).nodes,p.sideANodes);
        const auto b=4u*p.secondTetrahedron+opposite(tets.at(p.secondTetrahedron).nodes,p.sideBNodes);
        ids[std::minmax(a,b)].push_back({"hma",i});
    }
    std::map<std::size_t,std::size_t> histogram;
    std::size_t maxCount=0; Pair maxPair{};
    for(const auto& [pair,faces]:ids) {
        ++histogram[faces.size()];
        if(faces.size()>maxCount) { maxCount=faces.size(); maxPair=pair; }
    }
    std::cout << "nodes=" << starch.referenceNodesM.size()
              << " tets=" << starch.tetrahedra.size()
              << " starch_faces=" << starch.pairedFaces.size()
              << " hma_patches=" << hma.patches.size()
              << " hma_groups=" << hma.groups.size()
              << " total_mortar_records=" << starch.pairedFaces.size()+hma.patches.size()
              << " unique_parent_pairs=" << ids.size()
              << " max_multiplicity=" << maxCount
              << " max_pair=" << maxPair.first << ',' << maxPair.second << '\n';
    for(const auto& [n,c]:histogram) std::cout << "multiplicity_" << n << '=' << c << '\n';
    for(const auto& [source,index]:ids[maxPair]) {
        std::cout << "max_record_source=" << source << " source_index=" << index << '\n';
    }
}

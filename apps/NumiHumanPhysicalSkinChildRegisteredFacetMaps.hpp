#pragma once
#include "NumiHumanPhysicalSkinChildRegisteredVolumeImport.hpp"
#include <map>
#include <istream>

namespace metalrobo::human::coarse_tissue {

// This row is an existing donor head-sidewall face, not permission to cap an
// arbitrary new hole. Both parent source nodes and the child facet are exact.
struct PhysicalSkinAuthoredHeadWallRow {
    std::uint32_t stableWallFace=NM_INVALID_INDEX,childTet=NM_INVALID_INDEX;
    std::array<std::uint32_t,3> childNodes{};
    std::uint32_t parentTet=NM_INVALID_INDEX,material=NM_INVALID_INDEX;
    std::array<std::uint32_t,3> parentNodes{};
};
struct PhysicalSkinRegisteredFacetMapReport {
    std::array<std::size_t,3> layerFaceUses{};
    std::size_t authoredHeadWallFaces=0,internalFacets=0;
};

// Build incidence and ownership once from exact mapped node IDs. The existing
// unpaired volume owner repeats physical admission independently.
[[nodiscard]] inline bool preparePhysicalSkinRegisteredFacetMaps(
        PhysicalSkinChildMassMappingBundle& bundle,
        const std::span<const PhysicalSkinAuthoredHeadWallRow> authoredWalls,
        const std::uint32_t skinMaterial,const std::uint32_t fatMaterial,
        PhysicalSkinRegisteredFacetMapReport& report,std::string& error) {
    using Key=std::array<std::uint32_t,3>;
    struct RegisteredFace { unsigned layer;std::uint32_t row,stable; };
    struct FacetUse { std::uint32_t cell,omitted;int orientation; };
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const std::size_t nodes=bundle.currentWorldPoints.size(),cells=bundle.childTetrahedra.size();
    if(nodes==0u||nodes>300000u||cells==0u||cells>200000u||
       bundle.referenceWorldPoints.size()!=nodes||bundle.boundaryMap.volumeNodeCount!=nodes||
       bundle.childMaterialIndices.size()!=cells||skinMaterial==fatMaterial||
       bundle.parent.currentWorldPoints.size()!=bundle.parent.referenceWorldPoints.size()||
       bundle.parent.materialIndices.size()!=bundle.parent.tetrahedra.size())
        return fail("registered facet mapping requires complete current/reference nodes, cells, materials, and parent");
    const std::array<const RegisteredSurface*,3> surfaces{
        &bundle.outerSkin,&bundle.independentSkinInner,&bundle.independentFatDeep};
    const std::array<const std::vector<std::uint32_t>*,3> maps{
        &bundle.boundaryMap.outer,&bundle.boundaryMap.skinInner,&bundle.boundaryMap.fatDeep};
    std::map<Key,RegisteredFace> registered;
    std::array<std::vector<std::size_t>,3> counts;
    for(unsigned layer=0;layer<3u;++layer) {
        const auto& surface=*surfaces[layer];const auto& map=*maps[layer];
        if(surface.worldPoints.empty()||map.size()!=surface.worldPoints.size()||
           surface.faces.empty()||(!surface.sourceFaceIndices.empty()&&
           surface.sourceFaceIndices.size()!=surface.faces.size()))
            return fail("registered facet mapping has incomplete O/I/D boundary rows");
        counts[layer].assign(surface.faces.size(),0u);
        for(std::size_t face=0;face<surface.faces.size();++face) {
            Key key{};
            for(unsigned corner=0;corner<3u;++corner) {
                const auto vertex=surface.faces[face][corner];
                if(vertex>=map.size()||map[vertex]>=nodes)
                    return fail("registered facet mapping references a missing exact surface/volume node");
                key[corner]=map[vertex];
            }
            std::sort(key.begin(),key.end());
            const auto stable=surface.sourceFaceIndices.empty()?static_cast<std::uint32_t>(face):
                surface.sourceFaceIndices[face];
            if(key[0]==key[1]||key[1]==key[2]||stable==NM_INVALID_INDEX||
               !registered.emplace(key,RegisteredFace{layer,static_cast<std::uint32_t>(face),stable}).second)
                return fail("registered facet mapping contains repeated/degenerate O/I/D face identities");
        }
    }
    std::map<Key,std::vector<FacetUse>> incidence;
    constexpr std::array<std::array<unsigned,3>,4> corners{{{{1,2,3}},{{0,3,2}},{{0,1,3}},{{0,2,1}}}};
    for(std::size_t cell=0;cell<cells;++cell) {
        const auto tet=bundle.childTetrahedra[cell];auto sorted=tet;std::sort(sorted.begin(),sorted.end());
        if(sorted.back()>=nodes||std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end()||
           (bundle.childMaterialIndices[cell]!=skinMaterial&&bundle.childMaterialIndices[cell]!=fatMaterial))
            return fail("registered facet mapping contains an invalid child tet or material");
        for(unsigned omitted=0;omitted<4u;++omitted) {
            Key key{};for(unsigned c=0;c<3u;++c)key[c]=tet[corners[omitted][c]];
            int sign=1;for(unsigned a=0;a<3u;++a)for(unsigned b=a+1u;b<3u;++b)
                if(key[a]>key[b])sign=-sign;
            std::sort(key.begin(),key.end());
            auto& uses=incidence[key];uses.push_back({static_cast<std::uint32_t>(cell),omitted,sign});
            if(uses.size()>2u)return fail("registered facet mapping has a nonmanifold child facet");
        }
    }
    std::map<std::pair<std::uint32_t,Key>,const PhysicalSkinAuthoredHeadWallRow*> wallByFacet;
    std::set<std::uint32_t> wallIDs;
    for(const auto& wall:authoredWalls) {
        auto key=wall.childNodes;std::sort(key.begin(),key.end());
        if(wall.stableWallFace==NM_INVALID_INDEX||wall.childTet>=cells||
           wall.parentTet>=bundle.parent.tetrahedra.size()||
           wall.material!=bundle.childMaterialIndices[wall.childTet]||
           wall.material!=bundle.parent.materialIndices[wall.parentTet]||
           !wallIDs.insert(wall.stableWallFace).second||
           !wallByFacet.emplace(std::pair{wall.childTet,key},&wall).second)
            return fail("authored head wall has a missing/duplicate source, child, parent, or material identity");
        auto parentNodes=wall.parentNodes;std::sort(parentNodes.begin(),parentNodes.end());
        const auto parentTet=bundle.parent.tetrahedra[wall.parentTet];
        if(std::adjacent_find(parentNodes.begin(),parentNodes.end())!=parentNodes.end())
            return fail("authored head wall repeats parent source nodes");
        for(const auto node:parentNodes)
            if(node>=bundle.parent.currentWorldPoints.size()||
               std::find(parentTet.begin(),parentTet.end(),node)==parentTet.end())
                return fail("authored head wall is not a face of its declared current parent tet");
        // A wall remains the retained parent face, even if local surface
        // topology elsewhere changes. Do not synthesize a new neck closure.
        std::set<std::uint32_t> matched;
        for(const auto childNode:key) {
            if(childNode>=nodes)return fail("authored head wall references an absent child node");
            std::uint32_t match=NM_INVALID_INDEX;
            for(const auto parentNode:parentNodes) {
                V3 emitted{};for(unsigned a=0;a<3u;++a)
                    emitted[a]=double(float(bundle.parent.currentWorldPoints[parentNode][a]));
                if(norm(sub(emitted,bundle.currentWorldPoints[childNode]))<=2.0e-8) {
                    if(match!=NM_INVALID_INDEX)return fail("authored head wall parent geometry is ambiguous");
                    match=parentNode;
                }
            }
            if(match==NM_INVALID_INDEX||!matched.insert(match).second)
                return fail("authored head wall no longer matches its retained current parent face geometry");
        }
    }
    std::vector<std::uint32_t> facetRows(4u*cells,NM_INVALID_INDEX),stableFaces(4u*cells,NM_INVALID_INDEX);
    std::vector<std::uint8_t> wallFlags(4u*cells,0u);
    std::vector<std::uint32_t> stableWalls(4u*cells,NM_INVALID_INDEX);
    std::size_t usedWalls=0u;
    PhysicalSkinRegisteredFacetMapReport measured;
    for(const auto& [key,uses]:incidence) {
        const auto found=registered.find(key);
        if(uses.size()==2u) {
            if(uses[0].orientation+uses[1].orientation!=0)
                return fail("registered facet mapping has equally oriented internal facets");
            ++measured.internalFacets;
        }
        if(found!=registered.end()) {
            const auto face=found->second;
            if(uses.size()!=(face.layer==1u?2u:1u))
                return fail("registered O/I/D facet has incorrect child incidence");
            unsigned skinSides=0u,fatSides=0u;
            for(const auto& use:uses) {
                const auto material=bundle.childMaterialIndices[use.cell];
                skinSides+=material==skinMaterial;fatSides+=material==fatMaterial;
                if((face.layer==0u&&material!=skinMaterial)||(face.layer==2u&&material!=fatMaterial)||
                   wallByFacet.contains(std::pair{use.cell,key}))
                    return fail("registered O/D material or boundary/sidewall ownership conflicts");
                const auto slot=4u*use.cell+use.omitted;
                facetRows[slot]=face.row;stableFaces[slot]=face.stable;
                ++counts[face.layer][face.row];++measured.layerFaceUses[face.layer];
            }
            if(face.layer==1u&&(skinSides!=1u||fatSides!=1u))
                return fail("registered I facet does not connect exactly one skin and one fat cell");
        } else if(uses.size()==1u) {
            const auto& use=uses.front();const auto wall=wallByFacet.find(std::pair{use.cell,key});
            if(wall==wallByFacet.end())
                return fail("external child facet has no exact registered surface or retained authored head wall");
            const auto slot=4u*use.cell+use.omitted;wallFlags[slot]=1u;
            stableWalls[slot]=wall->second->stableWallFace;++usedWalls;
        } else {
            if(bundle.childMaterialIndices[uses[0].cell]!=bundle.childMaterialIndices[uses[1].cell])
                return fail("child material interface is not an exact registered I face");
            for(const auto& use:uses)if(wallByFacet.contains(std::pair{use.cell,key}))
                return fail("internal child facet is incorrectly tagged as an authored head wall");
        }
    }
    for(unsigned layer=0;layer<3u;++layer)
        if(std::any_of(counts[layer].begin(),counts[layer].end(),
            [layer](auto count){return count!=(layer==1u?2u:1u);}))
            return fail("registered facet mapping does not cover every O/I/D face exactly");
    if(usedWalls!=authoredWalls.size())return fail("authored head wall rows include an unused child facet");
    measured.authoredHeadWallFaces=usedWalls;
    bundle.childRegisteredLayerFacetSurfaceFaces=std::move(facetRows);
    bundle.childRegisteredLayerFacetSourceFaceIdentities=std::move(stableFaces);
    bundle.childAuthoredHeadSidewallFacetFlags=std::move(wallFlags);
    bundle.childAuthoredHeadSidewallStableFaceIdentities=std::move(stableWalls);
    bundle.childParentSurfaceFaces.assign(cells,NM_INVALID_INDEX);
    report=measured;error.clear();return true;
}

struct PhysicalSkinRegisteredFacetCSVReport {
    std::size_t fileRows=0,registeredRows=0,authoredWallRows=0;
};

inline std::vector<std::string> physicalSkinRegisteredFacetCSVHeader() {
    return {"source_partition","shell_face_row","face_marker","face_source_role",
        "face_source_row","face_exact_source_row","face_source_original_id",
        "face_source_layer","face_source_body","face_source_bone","face_ancestry_kind",
        "face_run_original_id","face_source_face_id","face_ancestry_source_rows",
        "face_parent_tet_row","face_parent_material_index","parent_source_node_rows",
        "joint_element_id","child_tet_row","joint_node0","joint_node1","joint_node2",
        "face_ancestry_source_original_ids","face_ancestry_parent_face_rows"};
}

namespace physical_skin_registered_facet_csv_detail {
inline bool splitQuotedRow(std::string line,std::vector<std::string>& fields) {
    if(!line.empty()&&line.back()=='\r')line.pop_back();
    fields.clear();std::size_t at=0;
    for(;;) {
        std::string field;
        if(at<line.size()&&line[at]=='"') {
            ++at;bool closed=false;
            while(at<line.size()) {
                const char value=line[at++];
                if(value!='"')field+=value;
                else if(at<line.size()&&line[at]=='"'){field+='"';++at;}
                else{closed=true;break;}
            }
            if(!closed||(at<line.size()&&line[at]!=','))return false;
        } else {
            while(at<line.size()&&line[at]!=',') {
                if(line[at]=='"')return false;
                field+=line[at++];
            }
        }
        fields.push_back(std::move(field));
        if(at==line.size())return true;
        ++at;
    }
}
inline bool parentFaceNodes(const std::string& field,std::array<std::uint32_t,3>& nodes) {
    if(field.size()<7u||field.front()!='['||field.back()!=']')return false;
    const auto numbers=physical_skin_child_mass_detail::splitCSV(field.substr(1,field.size()-2u));
    if(numbers.size()!=3u)return false;
    for(unsigned c=0;c<3u;++c)
        if(!physical_skin_child_mass_detail::parseNumber(numbers[c],nodes[c])||
           nodes[c]==NM_INVALID_INDEX)return false;
    auto sorted=nodes;std::sort(sorted.begin(),sorted.end());
    return sorted[0]!=sorted[1]&&sorted[1]!=sorted[2];
}
}

// Extract only explicitly authored source head walls from the existing full
// facet-ancestry CSV. Quoted JSON ancestry columns remain in the source asset;
// the mapper derives O/I/D ownership using exact registered node identities.
[[nodiscard]] inline bool readPhysicalSkinAuthoredHeadWallsCSV(
        std::istream& input,const std::size_t childCellCount,const std::size_t childNodeCount,
        std::vector<PhysicalSkinAuthoredHeadWallRow>& output,
        PhysicalSkinRegisteredFacetCSVReport& report,std::string& error) {
    using physical_skin_child_mass_detail::parseNumber;
    using physical_skin_registered_facet_csv_detail::splitQuotedRow;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(childCellCount==0u||childCellCount>200000u||childNodeCount<4u||childNodeCount>300000u)
        return fail("facet CSV child domain is invalid or exceeds capacity");
    std::string line;std::vector<std::string> fields;
    if(!std::getline(input,line)||!splitQuotedRow(line,fields)||
       fields!=physicalSkinRegisteredFacetCSVHeader())
        return fail("facet CSV header differs from the existing ancestry contract");
    PhysicalSkinRegisteredFacetCSVReport measured;
    std::vector<PhysicalSkinAuthoredHeadWallRow> next;
    std::set<std::pair<std::uint32_t,std::array<std::uint32_t,3>>> seen;
    std::set<std::uint32_t> stableWallIDs;
    while(std::getline(input,line)) {
        if(line.size()>16384u||measured.fileRows>=4u*childCellCount||
           !splitQuotedRow(line,fields)||fields.size()!=24u)
            return fail("facet CSV row exceeds capacity or has invalid CSV quoting/width");
        const bool skin=fields[0]=="skin",fat=fields[0]=="fat";
        const bool wall=fields[3]==(skin?"donor_skin_OI_headwall":"donor_fat_ID_headwall");
        if((!skin&&!fat)||(!wall&&fields[3]!=(skin?"outer_O":"outer_I")&&
                          fields[3]!=(skin?"inner_I":"inner_D")))
            return fail("facet CSV source partition/role is unsupported");
        PhysicalSkinAuthoredHeadWallRow row;std::uint32_t element=0;
        if(!parseNumber(fields[17],element)||!parseNumber(fields[18],row.childTet)||
           row.childTet>=childCellCount||element!=row.childTet+1u)
            return fail("facet CSV child row and joint element identity disagree");
        for(unsigned c=0;c<3u;++c) {
            std::uint32_t id=0;
            if(!parseNumber(fields[19u+c],id)||id==0u||id>childNodeCount)
                return fail("facet CSV child facet references an invalid volume node ID");
            row.childNodes[c]=id-1u;
        }
        auto key=row.childNodes;std::sort(key.begin(),key.end());
        if(key[0]==key[1]||key[1]==key[2]||!seen.emplace(row.childTet,key).second)
            return fail("facet CSV repeats a child facet or its corner nodes");
        if(wall) {
            unsigned marker=0,kind=0;
            if(!parseNumber(fields[2],marker)||marker!=(skin?3u:6u)||
               !parseNumber(fields[10],kind)||kind!=5u||
               !parseNumber(fields[4],row.stableWallFace)||row.stableWallFace==NM_INVALID_INDEX||
               !stableWallIDs.insert(row.stableWallFace).second||
               !parseNumber(fields[14],row.parentTet)||row.parentTet==NM_INVALID_INDEX||
               !parseNumber(fields[15],row.material)||row.material!=(skin?15u:16u)||
               !physical_skin_registered_facet_csv_detail::parentFaceNodes(fields[16],row.parentNodes))
                return fail("facet CSV authored head wall lacks exact source/material/parent identities");
            next.push_back(row);++measured.authoredWallRows;
        } else ++measured.registeredRows;
        ++measured.fileRows;
    }
    if(!input.eof()||measured.fileRows==0u)
        return fail("facet CSV is empty or unreadable before EOF");
    output=std::move(next);report=measured;error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue

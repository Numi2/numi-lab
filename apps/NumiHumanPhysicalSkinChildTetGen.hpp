#pragma once
#include "NumiHumanPhysicalSkinChildCapacity.hpp"

// Offline import of the existing TetGen node/element tables and explicit native
// cell labels. No equations are evaluated here and no physical admission is
// implied by successful parsing. The unpaired volume owner remains mandatory.
#include "NumiHumanPhysicalSkinChildMassMapping.hpp"
#include <unordered_map>

namespace metalrobo::human::coarse_tissue {

struct PhysicalSkinChildTetGenImportReport {
    std::size_t nodeCount=0u,cellCount=0u;
    double minimumFloat32ReferenceVolumeM3=std::numeric_limits<double>::infinity();
    double maximumFloat32RoundingDistanceM=0.0;
    std::string nodeFileSHA256,elementFileSHA256,labelFileSHA256;
};

namespace physical_skin_child_tetgen_detail {
inline bool tokens(std::ifstream& input,std::vector<std::string>& row) {
    std::string line;
    while(std::getline(input,line)) {
        if(line.size()>16384u)return false;
        line.resize(line.find('#')==std::string::npos?line.size():line.find('#'));
        std::istringstream stream(line);row.clear();std::string word;
        while(stream>>word)row.push_back(std::move(word));
        if(!row.empty())return true;
    }
    return false;
}
}

[[nodiscard]] inline bool loadPhysicalSkinChildTetGenReference(
        const std::filesystem::path& nodePath,const std::filesystem::path& elementPath,
        const std::filesystem::path& labelsPath,const std::string_view coordinateFrameSHA256,
        const std::size_t sourceBodyCount,const double maximumMassTransportDistanceM,
        PhysicalSkinChildMassMappingBundle& output,
        PhysicalSkinChildTetGenImportReport& report,std::string& error) {
    using namespace physical_skin_child_mass_detail;
    using physical_skin_child_tetgen_detail::tokens;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(!validSHA256(coordinateFrameSHA256)||sourceBodyCount==0u||
       !std::isfinite(maximumMassTransportDistanceM)||maximumMassTransportDistanceM<=0.0)
        return fail("child TetGen import requires a pinned coordinate frame, body domain, and mass transport bound");
    if(!output.currentWorldPoints.empty()||!output.referenceWorldPoints.empty()||
       !output.childTetrahedra.empty()||!output.childMaterialIndices.empty())
        return fail("child TetGen import requires an empty child bundle to prevent stale physical metadata");
    std::ifstream nodes(nodePath),elements(elementPath),labels(labelsPath);
    if(!nodes||!elements||!labels)return fail("child TetGen import requires node, element, and explicit label files");
    // Preserve only the caller's immutable registered-surface/frame/ancestry
    // sidecars. Child tables are required empty above and are filled below.
    PhysicalSkinChildMassMappingBundle next=output;
    PhysicalSkinChildTetGenImportReport measured;
    std::vector<std::string> row;
    std::size_t nodeCount=0u,dimension=0u,attributes=0u,markers=0u;
    if(!tokens(nodes,row)||row.size()!=4u||!parseNumber(row[0],nodeCount)||
       !parseNumber(row[1],dimension)||!parseNumber(row[2],attributes)||
       !parseNumber(row[3],markers)||nodeCount<4u||nodeCount>physical_skin_child_capacity_detail::maxNodes||
       dimension!=3u||attributes!=0u||markers>1u)
        return fail("child TetGen node header is invalid or outside its bounded 3D domain");
    std::unordered_map<std::uint32_t,std::uint32_t> nodeRows;
    nodeRows.reserve(nodeCount);
    next.referenceWorldPoints.reserve(nodeCount);
    for(std::size_t i=0u;i<nodeCount;++i) {
        std::uint32_t id=0u;V3 original{},point{};
        if(!tokens(nodes,row)||row.size()!=4u+markers||!parseNumber(row[0],id)||
           !nodeRows.emplace(id,static_cast<std::uint32_t>(i)).second)
            return fail("child TetGen node row has a duplicate identity or invalid width");
        if(markers){std::int64_t marker=0;if(!parseNumber(row[4],marker))
            return fail("child TetGen node boundary marker is invalid");}
        for(unsigned axis=0u;axis<3u;++axis) {
            if(!parseNumber(row[axis+1u],original[axis])||!std::isfinite(original[axis]))
                return fail("child TetGen node coordinate is not finite");
            point[axis]=static_cast<double>(static_cast<float>(original[axis]));
            if(!std::isfinite(point[axis]))return fail("child TetGen node overflows Float32");
        }
        measured.maximumFloat32RoundingDistanceM=std::max(measured.maximumFloat32RoundingDistanceM,
            norm(sub(point,original)));
        next.referenceWorldPoints.push_back(point);
    }
    if(tokens(nodes,row)||!nodes.eof())return fail("child TetGen node table has extra or unreadable rows");
    std::size_t cellCount=0u,corners=0u,cellAttributes=0u;
    if(!tokens(elements,row)||row.size()!=3u||!parseNumber(row[0],cellCount)||
       !parseNumber(row[1],corners)||!parseNumber(row[2],cellAttributes)||
       cellCount==0u||cellCount>physical_skin_child_capacity_detail::maxCells||corners!=4u||cellAttributes!=1u)
        return fail("child TetGen element header exceeds the selected bounded cell capacity or lacks one material attribute");
    std::set<std::uint32_t> elementIDs;
    std::set<std::array<std::uint32_t,4>> canonicalCells;
    next.childTetrahedra.reserve(cellCount);next.childMaterialIndices.reserve(cellCount);
    for(std::size_t cell=0u;cell<cellCount;++cell) {
        std::uint32_t id=0u;std::array<std::uint32_t,4> tet{};
        double materialValue=0.0;
        if(!tokens(elements,row)||row.size()!=6u||!parseNumber(row[0],id)||
           !elementIDs.insert(id).second||!parseNumber(row[5],materialValue)||
           !std::isfinite(materialValue)||materialValue<0.0||materialValue>=double(NM_INVALID_INDEX)||
           std::floor(materialValue)!=materialValue)
            return fail("child TetGen element row has duplicate identity or invalid material");
        for(unsigned corner=0u;corner<4u;++corner) {
            std::uint32_t nodeID=0u;
            if(!parseNumber(row[corner+1u],nodeID)||!nodeRows.contains(nodeID))
                return fail("child TetGen element references a missing node identity");
            tet[corner]=nodeRows.at(nodeID);
        }
        auto sorted=tet;std::sort(sorted.begin(),sorted.end());
        const auto volume=signedTetVolume(next.referenceWorldPoints[tet[0]],next.referenceWorldPoints[tet[1]],
            next.referenceWorldPoints[tet[2]],next.referenceWorldPoints[tet[3]]);
        if(sorted[0]==sorted[1]||sorted[1]==sorted[2]||sorted[2]==sorted[3]||
           !std::isfinite(volume)||volume<=0.0)
            return fail("child TetGen cell is nonpositive after exact Float32 emission; no orientation repair is performed by import");
        if(!canonicalCells.insert(sorted).second)return fail("child TetGen contains a duplicate tetrahedron");
        measured.minimumFloat32ReferenceVolumeM3=std::min(measured.minimumFloat32ReferenceVolumeM3,volume);
        next.childTetrahedra.push_back(tet);
        next.childMaterialIndices.push_back(static_cast<std::uint32_t>(materialValue));
    }
    if(tokens(elements,row)||!elements.eof())return fail("child TetGen element table has extra or unreadable rows");
    if(!readHeader(labels,labelsPath,{"child_tet_row","material_index","geometric_donor_body",
            "anatomical_region_id","stable_source_face_id"},error))return false;
    std::string line;
    for(std::size_t cell=0u;cell<cellCount;++cell) {
        if(!std::getline(labels,line)||line.size()>16384u)return fail("child TetGen explicit labels are truncated or oversized");
        const auto fields=splitCSV(line);std::array<std::uint32_t,5> value{};
        if(fields.size()!=value.size())return fail("child TetGen explicit label row has invalid width");
        for(unsigned column=0u;column<value.size();++column)
            if(!parseNumber(fields[column],value[column]))return fail("child TetGen label row has an invalid identity");
        if(value[0]!=cell||value[1]!=next.childMaterialIndices[cell]||
           value[2]>=sourceBodyCount||value[3]==NM_INVALID_INDEX||value[4]==NM_INVALID_INDEX)
            return fail("child TetGen explicit labels violate row/material/body/region/source domains");
        next.childDonorBodyIndices.push_back(value[2]);
        next.childAnatomicalRegionIndices.push_back(value[3]);
        next.childStableSourceFaceIdentities.push_back(value[4]);
    }
    if(std::getline(labels,line)||!labels.eof())return fail("child TetGen explicit labels have extra or unreadable rows");
    if(!sha256File(nodePath,measured.nodeFileSHA256,error)||
       !sha256File(elementPath,measured.elementFileSHA256,error)||
       !sha256File(labelsPath,measured.labelFileSHA256,error))return false;
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("existing-tetgen-reference-child.v1");
    hash.string(measured.nodeFileSHA256);hash.string(measured.elementFileSHA256);
    hash.u64(next.referenceWorldPoints.size());
    for(const auto& point:next.referenceWorldPoints)for(auto value:point)hash.real(value);
    hash.u64(next.childTetrahedra.size());
    for(const auto& tet:next.childTetrahedra)for(auto node:tet)hash.u32(node);
    next.childMeshSHA256=hash.finish();
    next.regionLabelSHA256=measured.labelFileSHA256;
    next.coordinateFrameSHA256=coordinateFrameSHA256;
    next.maximumSpatialMassTransportDistanceM=maximumMassTransportDistanceM;
    next.currentWorldPoints=next.referenceWorldPoints;
    measured.nodeCount=nodeCount;measured.cellCount=cellCount;
    output=std::move(next);report=std::move(measured);error.clear();return true;
}
} // namespace metalrobo::human::coarse_tissue

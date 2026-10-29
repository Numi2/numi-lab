#pragma once

#include "NumiHumanPhysicalSkinAuthoringRoleConsumer.hpp"

#include <charconv>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace metalrobo::human::coarse_tissue {

struct PhysicalSkinAuthoringDetailBridgeCSVPin {
    std::filesystem::path path;
    std::string expectedSHA256;
};

struct PhysicalSkinAuthoringDetailBridgeCSVFiles {
    PhysicalSkinSourceDomainRegistration detailDomain;
    PhysicalSkinAuthoringDetailBridgeCSVPin detailSourceAsset;
    PhysicalSkinAuthoringDetailBridgeCSVPin detailNodes,detailFaces,detailFull86Weights,
        detailFull86Owners,fineToBoundaryMap,historicalFineToBoundaryMap,
        canonicalTetGenNode,coarseBoundaryFaces;
    double maximumBoundaryProjectionDistanceM=0.02;
    double maximumTetExtrapolationDistanceM=0.05;
    double producerResidualComparisonToleranceM=2.0e-6;
    double maximumPhysicalBoundaryNodeReferenceDifferenceM=2.0e-7;
    // Explicit producer pass only. Final import always requires the canonical
    // native-transfer face rows and never falls back to the historical seed map.
    bool allowPrecanonicalExportSeed=false;
    // Separate source-only permission for loading a canonical map over an
    // authenticated refined child domain. Runtime callers leave this false.
    bool allowAuthenticatedRefinedChildRouteForSourceOnlyPreflight=false;
};

struct PhysicalSkinAuthoringDetailBridgeCSVReport {
    std::size_t detailRows=0,detailFaces=0,coarseBoundaryFaces=0;
    std::size_t ancestryWalkRows=0,donorRemapRows=0;
    std::uint32_t maximumAncestryHops=0;
    double maximumProducerProjectionDistanceM=0.0;
    double maximumRecomputedProjectionDistanceM=0.0;
    double maximumPhysicalBoundaryNodeReferenceDifferenceM=0.0;
    double maximumResidualRebaseDeltaM=0.0;
    std::array<std::string,9> inputSHA256{};
    std::string historicalMapSHA256,canonicalTetGenNodeSHA256;
    std::size_t nativeCanonicalSelectionRows=0,nativeCanonicalFaceChangeRows=0;
    bool physicalAdmission=false;
};

struct PhysicalSkinAuthoringDetailBridgeCanonicalExportReport {
    std::size_t detailRows=0,changedFaceRows=0,representativeAdjustments=0;
    double maximumProjectionDistanceM=0.0;
    std::string historicalMapSHA256,canonicalTetGenNodeSHA256,seedMapSHA256,outputSHA256;
    bool physicalAdmission=false;
};

namespace physical_skin_authoring_detail_bridge_csv_detail {

using Row=std::vector<std::string>;
using Face=std::array<std::uint32_t,3>;

inline bool parseLine(std::string_view line,Row& fields) {
    fields.clear();
    std::string value;
    bool quoted=false,closedQuote=false;
    for(std::size_t i=0;i<line.size();++i) {
        const char c=line[i];
        if(quoted) {
            if(c=='"') {
                if(i+1u<line.size()&&line[i+1u]=='"'){value.push_back('"');++i;}
                else {quoted=false;closedQuote=true;}
            } else value.push_back(c);
        } else if(c==',') {
            fields.push_back(std::move(value));value.clear();closedQuote=false;
        } else if(c=='"'&&value.empty()&&!closedQuote) {
            quoted=true;
        } else {
            if(closedQuote)return false;
            value.push_back(c);
        }
    }
    if(quoted)return false;
    fields.push_back(std::move(value));
    return true;
}

template<class T>
inline bool number(const Row& row,const std::size_t column,T& value) {
    return column<row.size()&&physical_skin_child_mass_detail::parseNumber(row[column],value);
}
inline bool finiteNumber(const Row& row,const std::size_t column,double& value) {
    return number(row,column,value)&&std::isfinite(value);
}
inline bool signedNumber(const Row& row,const std::size_t column,std::int64_t& value) {
    return number(row,column,value);
}
inline bool digest(const PhysicalSkinAuthoringDetailBridgeCSVPin& pin,
                   std::string& actual,std::string& error) {
    using physical_skin_child_mass_detail::sha256File;
    if(pin.path.empty()||!physical_skin_child_mass_detail::validSHA256(pin.expectedSHA256)) {
        error="detail bridge CSV input is missing its path or SHA-256 pin";return false;
    }
    if(!sha256File(pin.path,actual,error))return false;
    if(actual!=pin.expectedSHA256) {
        error="detail bridge CSV SHA-256 pin mismatch: "+pin.path.string();return false;
    }
    return true;
}
template<class Callback>
inline bool readPinnedCSV(const PhysicalSkinAuthoringDetailBridgeCSVPin& pin,
        const std::vector<std::string>& expectedHeader,const std::size_t maximumRows,
        const std::size_t expectedRows,Callback&& callback,std::size_t& rows,
        std::string& actualSHA256,std::string& error,
        const std::vector<std::string>* alternateHeader=nullptr,
        bool* usedAlternateHeader=nullptr) {
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    std::string before;
    if(!digest(pin,before,error))return false;
    std::ifstream input(pin.path);
    if(!input)return fail("cannot open pinned detail bridge CSV: "+pin.path.string());
    std::string line;Row fields;
    const auto stripCRLF=[](std::string& csvLine) {
        if(!csvLine.empty()&&csvLine.back()==static_cast<char>(13))csvLine.pop_back();
    };
    if(!std::getline(input,line))return fail("detail bridge CSV is empty: "+pin.path.string());
    stripCRLF(line);
    if(line.size()>65536u||!parseLine(line,fields))
        return fail("detail bridge CSV header differs from the pinned producer contract: "+pin.path.string());
    const std::vector<std::string>* activeHeader=&expectedHeader;
    if(fields!=expectedHeader) {
        if(!alternateHeader||fields!=*alternateHeader)
            return fail("detail bridge CSV header differs from the pinned producer contract: "+pin.path.string());
        activeHeader=alternateHeader;
    }
    if(usedAlternateHeader)*usedAlternateHeader=activeHeader==alternateHeader;
    rows=0u;
    while(std::getline(input,line)) {
        stripCRLF(line);
        if(line.size()>1048576u||rows>=maximumRows||!parseLine(line,fields)||
           fields.size()!=activeHeader->size())
            return fail("detail bridge CSV has a malformed or excessive row: "+pin.path.string());
        if(!callback(fields,rows,error))return false;
        ++rows;
    }
    if(!input.eof()||(expectedRows!=0u&&rows!=expectedRows))
        return fail("detail bridge CSV row count differs from its pinned source domain: "+pin.path.string());
    std::string after;
    if(!physical_skin_child_mass_detail::sha256File(pin.path,after,error)||after!=before)
        return fail("detail bridge CSV changed while it was being imported: "+pin.path.string());
    actualSHA256=std::move(after);
    return true;
}
inline bool jsonUIntArray(const std::string& text,std::vector<std::uint32_t>& result) {
    result.clear();
    std::size_t i=0u;
    while(i<text.size()&&std::isspace(static_cast<unsigned char>(text[i])))++i;
    if(i==text.size()||text[i++]!='[')return false;
    while(i<text.size()) {
        while(i<text.size()&&std::isspace(static_cast<unsigned char>(text[i])))++i;
        if(i<text.size()&&text[i]==']') {
            ++i;while(i<text.size()&&std::isspace(static_cast<unsigned char>(text[i])))++i;
            return i==text.size();
        }
        const auto begin=i;
        while(i<text.size()&&text[i]>='0'&&text[i]<='9')++i;
        if(begin==i)return false;
        std::uint32_t value=0u;
        const auto parsed=std::from_chars(text.data()+begin,text.data()+i,value);
        if(parsed.ec!=std::errc{}||parsed.ptr!=text.data()+i)return false;
        result.push_back(value);
        while(i<text.size()&&std::isspace(static_cast<unsigned char>(text[i])))++i;
        if(i<text.size()&&text[i]==','){++i;continue;}
        if(i<text.size()&&text[i]==']')continue;
        return false;
    }
    return false;
}
inline Face sortedFace(Face face){std::sort(face.begin(),face.end());return face;}
inline bool validFace(const Face& face) {
    auto sorted=sortedFace(face);
    return sorted[0]!=sorted[1]&&sorted[1]!=sorted[2];
}
inline std::uint32_t majorityOwner(const Face& face,
        const std::vector<std::uint32_t>& owners) {
    std::array<std::uint32_t,3> values{
        owners[face[0]],owners[face[1]],owners[face[2]]};
    std::sort(values.begin(),values.end());
    if(values[0]==values[1]||values[0]==values[2])return values[0];
    if(values[1]==values[2])return values[1];
    return values[0];
}
inline bool pointFrom(const Row& row,const std::size_t first,V3& point) {
    for(unsigned axis=0;axis<3u;++axis)
        if(!finiteNumber(row,first+axis,point[axis]))return false;
    return physical_skin_authoring_detail_bridge_detail::finitePoint(point);
}
inline V3 baryPoint(const std::vector<V3>& points,const Face& face,
                    const std::array<double,3>& bary) {
    V3 result{};
    for(unsigned k=0;k<3u;++k)for(unsigned axis=0;axis<3u;++axis)
        result[axis]+=bary[k]*points[face[k]][axis];
    return result;
}

inline std::vector<std::string> seedMapHeader() {
    return {
        "fine_vertex_row","role","coarse_face_row","coarse_boundary_primary_parent_tet_row",
        "mapped_boundary_side_parent_tet_row","boundary_walk_start_parent_tet_row",
        "boundary_walk_hop_count","boundary_walk_parent_tet_rows_json",
        "volume_node0","volume_node1","volume_node2","coarse_primary_material_index",
        "coarse_primary_geometric_donor_body","coarse_primary_region_id",
        "coarse_primary_stable_source_face_id","mapped_side_material_index",
        "mapped_side_geometric_donor_body","mapped_side_region_id",
        "mapped_side_stable_source_face_id","bary0","bary1","bary2","residual_x_m",
        "residual_y_m","residual_z_m","projection_distance_m","mapping_kind",
        "fine_parent_material_index","fine_parent_geometric_donor_body",
        "fine_parent_region_id","fine_parent_stable_source_face_id",
        "fine_parent_registration_kind","fine_parent_tet_row",
        "fine_parent_tied_tet_rows_json","fine_parent_registration_distance_m",
        "fine_parent_registration_tie_count","fine_frame_owner_body","fine_source_domain",
        "fine_source_mapping_kind","fine_source_vertex_row","fine_source_face_row",
        "fine_source_bary0","fine_source_bary1","fine_source_bary2",
        "fine_incident_detail_face_rows_json","historical_producer_map_sha256",
        "canonical_tetgen_node_sha256","canonical_bary0","canonical_bary1",
        "canonical_bary2","canonical_residual_x_m","canonical_residual_y_m",
        "canonical_residual_z_m","canonical_projection_distance_m"};
}
inline const std::vector<std::string>& nativeSelectionHeader() {
    static const std::vector<std::string> header{
        "canonical_selected_boundary_face_row","canonical_selected_volume_node0",
        "canonical_selected_volume_node1","canonical_selected_volume_node2",
        "canonical_selected_parent_tet_row","canonical_selected_material_index",
        "canonical_selected_geometric_donor_body","canonical_selected_region_id",
        "canonical_selected_stable_source_face_id","canonical_selected_frame_owner_body",
        "canonical_selected_representative_boundary_vertex_row",
        "canonical_selected_walk_start_parent_tet_row","canonical_selected_walk_hop_count",
        "canonical_selected_walk_parent_tet_rows_json","canonical_selected_face_bary0",
        "canonical_selected_face_bary1","canonical_selected_face_bary2",
        "canonical_selected_residual_x_m","canonical_selected_residual_y_m",
        "canonical_selected_residual_z_m","canonical_selected_projection_distance_m"};
    return header;
}
inline std::string csvQuote(std::string_view value) {
    if(value.find_first_of(",\"\r\n")==std::string_view::npos)return std::string(value);
    std::string result;result.reserve(value.size()+2u);result.push_back('"');
    for(const char c:value){if(c=='"')result.push_back('"');result.push_back(c);}
    result.push_back('"');return result;
}
inline std::string realText(const double value) {
    std::ostringstream stream;stream<<std::setprecision(17)<<value;return stream.str();
}
inline std::string uintArrayJSON(const std::vector<std::uint32_t>& values) {
    std::string text="[";
    for(std::size_t i=0;i<values.size();++i){if(i)text.push_back(',');text+=std::to_string(values[i]);}
    text.push_back(']');return text;
}
struct CanonicalNativeSelection {
    std::uint32_t faceRow=NM_INVALID_INDEX,parentTet=NM_INVALID_INDEX,material=NM_INVALID_INDEX;
    std::uint32_t donor=NM_INVALID_INDEX,region=NM_INVALID_INDEX,stableFace=NM_INVALID_INDEX;
    std::uint32_t frameOwner=NM_INVALID_INDEX,representative=NM_INVALID_INDEX;
    std::uint32_t walkStart=NM_INVALID_INDEX,walkHops=0u;
    Face volumeNodes{};
    std::array<double,3> bary{};
    V3 residual{};double distance=0.0;
    std::vector<std::uint32_t> walk;
};
inline bool runNativeFaceSelection(
        const PhysicalSkinAuthoringDetailBridgeInput& input,
        std::vector<std::uint32_t>& representatives,
        std::vector<CanonicalNativeSelection>& selections,
        std::size_t& changedRepresentatives,std::string& error) {
    using Transfer=numi::human::AcceptedFEMSkinTransfer;
    const std::size_t n=input.detailReferenceWorldPoints.size();
    const std::size_t b=input.physicalBoundaryReferenceWorldPoints.size();
    if(n==0u||input.detailDomain.vertexCount!=n||input.detailSourceVertexRows.size()!=n||
       input.detailGeometricDonorBodyIndices.size()!=n||
       input.detailBoundaryProjectionDonorBodyIndices.size()!=n||
       input.detailFaces.empty()||input.detailRepresentativeBoundaryVertexRows.size()!=n||
       input.physicalBoundaryFaces.empty()||input.physicalBoundaryVolumeNodeRows.size()!=b||
       input.physicalBoundaryGeometricDonorBodyIndices.size()!=b||
       input.physicalBoundaryStableFaceIDs.size()!=input.physicalBoundaryFaces.size()||
       input.physicalBoundaryFaceParentTetRows.size()!=input.physicalBoundaryFaces.size()||
       input.physicalBoundaryFaceMaterialIndices.size()!=input.physicalBoundaryFaces.size()||
       input.physicalBoundaryFaceGeometricDonorBodyIndices.size()!=input.physicalBoundaryFaces.size()||
       input.physicalBoundaryFaceAnatomicalRegionIndices.size()!=input.physicalBoundaryFaces.size()||
       input.physicalBoundaryFaceFrameOwnerBodyIndices.size()!=input.physicalBoundaryFaces.size()||
       input.detailMaterialIndices.size()!=n||input.detailAnatomicalRegionIndices.size()!=n||
       input.detailStableSourceFaceIDs.size()!=n||input.detailParentTetRows.size()!=n) {
        error="canonical native-face export received incomplete loaded detail-bridge arrays";return false;
    }
    std::vector<std::uint8_t> used(n,0u);
    for(const auto& face:input.detailFaces)for(const auto vertex:face){
        if(vertex>=n){error="canonical native-face export found an out-of-domain detail face";return false;}
        used[vertex]=1u;
    }
    if(std::find(used.begin(),used.end(),std::uint8_t(0u))!=used.end()){
        error="canonical native-face export detail topology omits a source vertex";return false;
    }
    bool directIdentity=input.detailDomain==input.physicalBoundaryDomain&&n==b&&
        input.detailFaces==input.physicalBoundaryFaces&&
        input.detailSourceVertexRows==input.physicalBoundarySourceVertexRows&&
        input.detailGeometricDonorBodyIndices==input.physicalBoundaryGeometricDonorBodyIndices&&
        input.detailBoundaryProjectionDonorBodyIndices==input.detailGeometricDonorBodyIndices&&
        std::all_of(input.detailRegionScopedDonorFallback.begin(),
                    input.detailRegionScopedDonorFallback.end(),[](auto x){return x==0u;})&&
        (input.detailRequiresParentTetPath.empty()||
         std::all_of(input.detailRequiresParentTetPath.begin(),input.detailRequiresParentTetPath.end(),
                     [](auto x){return x==0u;}));
    if(directIdentity){
        std::set<std::uint32_t> unique;
        for(std::size_t i=0;i<n;++i)
            directIdentity=directIdentity&&unique.insert(input.detailSourceVertexRows[i]).second&&
                input.detailReferenceWorldPoints[i]==input.physicalBoundaryReferenceWorldPoints[i]&&
                input.physicalBoundaryVolumeNodeRows[i]<input.mechanicalReferenceWorldPoints.size()&&
                input.mechanicalReferenceWorldPoints[input.physicalBoundaryVolumeNodeRows[i]]==
                    input.detailReferenceWorldPoints[i];
    }
    std::vector<std::uint32_t> detailSource(n),boundarySource(b,NM_INVALID_INDEX);
    if(directIdentity){
        detailSource=input.detailSourceVertexRows;boundarySource=input.physicalBoundarySourceVertexRows;
    }else for(std::size_t i=0;i<n;++i)detailSource[i]=0x80000000u+static_cast<std::uint32_t>(i);
    representatives=input.detailRepresentativeBoundaryVertexRows;
    constexpr unsigned maximumRepresentativePasses=8u;
    std::vector<std::uint32_t> finalFaceRows;
    std::vector<Transfer::GPUTransferRow> finalRows;
    bool stable=false;
    std::vector<std::vector<std::uint32_t>> parentAdjacency;
    if(input.sourceParentTetNodeRows.empty()||
       input.sourceParentTetMaterialIndices.size()!=input.sourceParentTetNodeRows.size()||
       input.sourceParentTetAnatomicalRegionIndices.size()!=input.sourceParentTetNodeRows.size()||
       input.sourceParentTetStableSourceFaceIDs.size()!=input.sourceParentTetNodeRows.size()||
       !physical_skin_authoring_detail_bridge_detail::buildParentTetFaceAdjacency(
           input.sourceParentTetNodeRows,parentAdjacency,error)){
        error="canonical face export requires valid pinned parent-tet adjacency: "+error;return false;
    }
    PhysicalSkinAuthoringDetailFaceEligibilityContext eligibilityContext{&input,&parentAdjacency};
    const auto faceEligible=+[](const void* opaque,std::size_t row,std::size_t face)->bool {
        return physicalSkinAuthoringDetailFaceEligible(
            *static_cast<const PhysicalSkinAuthoringDetailFaceEligibilityContext*>(opaque),row,face);
    };
    for(unsigned pass=0;pass<maximumRepresentativePasses;++pass){
        Transfer transfer;Transfer::BuildReport transferReport;std::string transferError;
        if(!transfer.build(input.detailReferenceWorldPoints,detailSource,
                input.detailRegisteredSourceDonorBodyIndices,used,
                input.physicalBoundaryReferenceWorldPoints,boundarySource,
                input.physicalBoundaryGeometricDonorBodyIndices,input.physicalBoundaryFaces,
                &transferReport,&transferError,input.maximumBoundaryProjectionDistanceM,
                &representatives,nullptr,nullptr,nullptr,1u,
                &input.physicalBoundaryStableFaceIDs,nullptr,
                faceEligible,&eligibilityContext)){
            error="existing native source-connected transfer builder rejected canonical face export: "+transferError;
            return false;
        }
        std::vector<std::uint32_t> nextRepresentatives=representatives;
        finalFaceRows.resize(n);
        for(std::size_t i=0;i<n;++i){
            const auto faceRow=transfer.selectedFaceIndex(i);
            if(faceRow>=input.physicalBoundaryFaces.size()){
                error="native transfer returned an invalid selected physical face at detail row "+std::to_string(i);return false;
            }
            finalFaceRows[i]=faceRow;
            const auto& face=input.physicalBoundaryFaces[faceRow];
            const auto& row=transfer.rows()[i];
            if(row.coarseVertex[0]!=face[0]||row.coarseVertex[1]!=face[1]||
               row.coarseVertex[2]!=face[2]){
                error="native transfer face-corner order differs from its source boundary topology at detail row "+std::to_string(i);return false;
            }
            if(std::find(face.begin(),face.end(),representatives[i])!=face.end())continue;
            unsigned best=0u;
            for(unsigned corner=1u;corner<3u;++corner)
                if(row.weight[corner]>row.weight[best])best=corner;
            nextRepresentatives[i]=face[best];
        }
        changedRepresentatives=0u;
        for(std::size_t i=0;i<n;++i)changedRepresentatives+=nextRepresentatives[i]!=representatives[i];
        if(changedRepresentatives==0u){
            finalRows=transfer.rows();stable=true;break;
        }
        representatives.swap(nextRepresentatives);
    }
    if(!stable){
        error="native selected-face representative did not reach a fixed point after "+
            std::to_string(maximumRepresentativePasses)+" bounded transfer builds";return false;
    }
    selections.assign(n,{});
    std::size_t failures=0u;std::ostringstream witnesses;
    for(std::size_t i=0;i<n;++i){
        auto& selected=selections[i];
        selected.faceRow=finalFaceRows[i];
        selected.parentTet=input.physicalBoundaryFaceParentTetRows[selected.faceRow];
        selected.material=input.physicalBoundaryFaceMaterialIndices[selected.faceRow];
        selected.donor=input.physicalBoundaryFaceGeometricDonorBodyIndices[selected.faceRow];
        selected.region=input.physicalBoundaryFaceAnatomicalRegionIndices[selected.faceRow];
        selected.stableFace=input.physicalBoundaryStableFaceIDs[selected.faceRow];
        selected.frameOwner=input.physicalBoundaryFaceFrameOwnerBodyIndices[selected.faceRow];
        selected.representative=representatives[i];
        selected.walkStart=input.detailParentTetRows[i];
        const auto& localFace=input.physicalBoundaryFaces[selected.faceRow];
        const auto& transferRow=finalRows[i];
        double sum=0.0;
        for(unsigned k=0;k<3u;++k){
            selected.volumeNodes[k]=input.physicalBoundaryVolumeNodeRows[localFace[k]];
            if(transferRow.coarseVertex[k]!=localFace[k]||
               selected.volumeNodes[k]>=input.mechanicalReferenceWorldPoints.size()){
                ++failures;if(failures<=20u)witnesses<<" row="<<i<<":native face order/node map mismatch;";
                continue;
            }
            selected.bary[k]=double(transferRow.weight[k]);sum+=selected.bary[k];
        }
        if(!(std::isfinite(sum)&&sum>0.0&&std::abs(sum-1.0)<=2.0e-6)){
            ++failures;if(failures<=20u)witnesses<<" row="<<i<<":native barycentric sum="<<sum<<';';
            continue;
        }
        for(double& weight:selected.bary)weight/=sum;
        const bool labels=selected.material==input.detailMaterialIndices[i]&&
            selected.region==input.detailAnatomicalRegionIndices[i]&&
            selected.frameOwner!=NM_INVALID_INDEX&&
            selected.parentTet<input.sourceParentTetStableSourceFaceIDs.size()&&
            input.sourceParentTetStableSourceFaceIDs[selected.parentTet]==selected.stableFace;
        if(!labels){
            ++failures;if(failures<=20u)witnesses<<" row="<<i<<":selected labels material/physical-donor/region/current-stable/owner="
                <<selected.material<<'/'<<selected.donor<<'/'<<selected.region<<'/'
                <<selected.stableFace<<'/'<<selected.frameOwner<<" fine="
                <<input.detailMaterialIndices[i]<<'/'<<selected.donor<<'/'
                <<input.detailAnatomicalRegionIndices[i]<<'/'<<input.detailStableSourceFaceIDs[i]<<';';
        }
        const V3 projected=baryPoint(input.physicalBoundaryReferenceWorldPoints,localFace,selected.bary);
        selected.residual=sub(input.detailReferenceWorldPoints[i],projected);
        selected.distance=norm(selected.residual);
        if(!std::isfinite(selected.distance)||selected.distance>input.maximumBoundaryProjectionDistanceM||
           !std::isfinite(transferRow.restDistance)||
           std::abs(selected.distance-double(transferRow.restDistance))>2.0e-6){
            ++failures;if(failures<=20u)witnesses<<" row="<<i<<":projection="<<selected.distance
                <<" native="<<transferRow.restDistance<<';';
        }
        if(selected.walkStart>=input.sourceParentTetNodeRows.size()||
           selected.parentTet>=input.sourceParentTetNodeRows.size()||
           input.sourceParentTetStableSourceFaceIDs.size()!=input.sourceParentTetNodeRows.size()||
           selected.walkStart>=input.sourceParentTetStableSourceFaceIDs.size()||
           selected.parentTet>=input.sourceParentTetStableSourceFaceIDs.size()||
           input.sourceParentTetStableSourceFaceIDs[selected.parentTet]!=selected.stableFace||
           !physical_skin_authoring_detail_bridge_detail::boundedSameRegionParentTetPath(parentAdjacency,
                input.sourceParentTetMaterialIndices,input.sourceParentTetAnatomicalRegionIndices,
                selected.walkStart,selected.parentTet,selected.material,selected.region,
                input.maximumSourceParentTetHops,selected.walkHops,selected.walk)){
            ++failures;if(failures<=20u)witnesses<<" row="<<i<<":no bounded same-material/region parent-cell path;";
        }
        if(selected.representative>=input.physicalBoundaryReferenceWorldPoints.size()||
           std::find(localFace.begin(),localFace.end(),selected.representative)==localFace.end()){
            ++failures;if(failures<=20u)witnesses<<" row="<<i<<":selected representative is outside selected face;";
        }
    }
    if(failures){
        error="bulk native canonical-face audit rejected "+std::to_string(failures)+
            " of "+std::to_string(n)+" rows; first witnesses:"+witnesses.str();return false;
    }
    return true;
}

} // namespace physical_skin_authoring_detail_bridge_csv_detail

// Loads the detailed typed O asset and its producer correspondence, then
// recomputes all rest offsets against the exact imported physical boundary and
// child volume. The result feeds the existing checked transfer builder; it
// never admits or mutates physical geometry.
[[nodiscard]] inline bool loadPhysicalSkinAuthoringDetailBridgeCSV(
        const PhysicalSkinAuthoringDetailBridgeCSVFiles& files,
        const PhysicalSkinChildMassMappingBundle& bundle,
        const PhysicalSkinAuthoringRoleSurfaceRows& physicalBoundary,
        PhysicalSkinAuthoringDetailBridgeInput& output,
        PhysicalSkinAuthoringDetailBridgeCSVReport& report,
        std::string& error) {
    using namespace physical_skin_authoring_detail_bridge_csv_detail;
    using namespace physical_skin_authoring_detail_bridge_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(!validDomain(files.detailDomain)||
       files.detailDomain.role!=PhysicalSkinSourceDomainRole::outerSurface||
       files.detailDomain.vertexCount==0u||files.detailDomain.vertexCount>200000u||
       files.detailDomain.faceCount==0u||files.detailDomain.faceCount>500000u||
       physicalBoundary.sourceDomain.role!=PhysicalSkinSourceDomainRole::outerSurface||
       physicalBoundary.sourceDomain.vertexCount!=physicalBoundary.currentWorldPoints.size()||
       physicalBoundary.sourceDomain.faceCount!=physicalBoundary.roleLocalFaces.size()||
       physicalBoundary.volumeNodeRows.size()!=physicalBoundary.currentWorldPoints.size()||
       physicalBoundary.authoringVertexRows.size()!=physicalBoundary.currentWorldPoints.size()||
       physicalBoundary.frameOwnerBodyIndices.size()!=physicalBoundary.currentWorldPoints.size()||
       physicalBoundary.parentGeometricDonorBodyIndices.size()!=physicalBoundary.currentWorldPoints.size()||
       bundle.childTetrahedra.empty()||
       bundle.childDominantParentTetrahedra.size()!=bundle.childTetrahedra.size()||
       bundle.childMaterialIndices.size()!=bundle.childTetrahedra.size()||
       bundle.childDonorBodyIndices.size()!=bundle.childTetrahedra.size()||
       bundle.childAnatomicalRegionIndices.size()!=bundle.childTetrahedra.size()||
       bundle.referenceWorldPoints.size()!=bundle.currentWorldPoints.size()||
       bundle.parent.referenceWorldPoints.empty()||
       bundle.parent.tetrahedra.size()!=bundle.parent.materialIndices.size()||
       bundle.parent.tetrahedra.size()!=bundle.parent.donorBodyIndices.size()||
       bundle.parent.tetrahedra.size()!=bundle.parent.anatomicalRegionIndices.size()||
       bundle.parent.tetrahedra.size()!=bundle.parent.stableSourceFaceIdentities.size())
        return fail("detail bridge needs the pinned outer source domain, role boundary, and complete parent-identity volume");
    if(!std::isfinite(files.maximumBoundaryProjectionDistanceM)||
       files.maximumBoundaryProjectionDistanceM<=0.0||
       files.maximumBoundaryProjectionDistanceM>0.05||
       !std::isfinite(files.maximumTetExtrapolationDistanceM)||
       files.maximumTetExtrapolationDistanceM<=0.0||
       files.maximumTetExtrapolationDistanceM>0.05||
       !std::isfinite(files.producerResidualComparisonToleranceM)||
       files.producerResidualComparisonToleranceM<0.0||
       files.producerResidualComparisonToleranceM>1.0e-3||
       !std::isfinite(files.maximumPhysicalBoundaryNodeReferenceDifferenceM)||
       files.maximumPhysicalBoundaryNodeReferenceDifferenceM<0.0||
       files.maximumPhysicalBoundaryNodeReferenceDifferenceM>2.0e-6)
        return fail("detail bridge projection or residual comparison bounds are invalid");
    std::string historicalMapSHA256,canonicalTetGenNodeSHA256;
    if(!digest(files.historicalFineToBoundaryMap,historicalMapSHA256,error)||
       !digest(files.canonicalTetGenNode,canonicalTetGenNodeSHA256,error))
        return false;
    if(files.historicalFineToBoundaryMap.path==files.fineToBoundaryMap.path||
       !physical_skin_child_mass_detail::validSHA256(canonicalTetGenNodeSHA256))
        return fail("canonical detail map requires distinct historical and imported-node pins");
    bool childParentIdentity=bundle.childTetrahedra.size()==bundle.parent.tetrahedra.size();
    if(childParentIdentity)for(std::size_t cell=0;cell<bundle.childTetrahedra.size();++cell) {
        auto child=bundle.childTetrahedra[cell],parent=bundle.parent.tetrahedra[cell];
        std::sort(child.begin(),child.end());std::sort(parent.begin(),parent.end());
        if(bundle.childDominantParentTetrahedra[cell]!=cell||child!=parent||
           bundle.childMaterialIndices[cell]!=bundle.parent.materialIndices[cell]||
           bundle.childDonorBodyIndices[cell]!=bundle.parent.donorBodyIndices[cell]||
           bundle.childAnatomicalRegionIndices[cell]!=bundle.parent.anatomicalRegionIndices[cell]) {
            childParentIdentity=false;break;
        }
    }
    // Runtime rendering remains identity-only. Refined child maps are
    // accepted only under an explicit canonical-export or authenticated
    // source-only preflight permission, with the 002->530->006 route checked below.
    if(!files.allowPrecanonicalExportSeed&&
       !files.allowAuthenticatedRefinedChildRouteForSourceOnlyPreflight&&
       !childParentIdentity)
        return fail("runtime detail bridge requires exact child/parent cell identity");
    std::map<std::uint32_t,std::uint32_t> volumeNodeToBoundaryRow;
    double maxBoundaryNodeReferenceDifference=0.0;
    for(std::size_t row=0;row<physicalBoundary.volumeNodeRows.size();++row) {
        const auto volumeNode=physicalBoundary.volumeNodeRows[row];
        if(volumeNode>=bundle.referenceWorldPoints.size()||
           physicalBoundary.authoringVertexRows[row]>=physicalBoundary.sourceDomain.vertexCount||
           physicalBoundary.frameOwnerBodyIndices[row]==NM_INVALID_INDEX||
           physicalBoundary.parentGeometricDonorBodyIndices[row]==NM_INVALID_INDEX||
           !finitePoint(physicalBoundary.currentWorldPoints[row])||
           !volumeNodeToBoundaryRow.emplace(volumeNode,static_cast<std::uint32_t>(row)).second)
            return fail("detail bridge physical boundary rows do not map to imported child reference nodes");
        const double coordinateDelta=norm(sub(bundle.referenceWorldPoints[volumeNode],physicalBoundary.currentWorldPoints[row]));
        if(!std::isfinite(coordinateDelta)||coordinateDelta>files.maximumPhysicalBoundaryNodeReferenceDifferenceM)
            return fail("typed physical boundary differs from the canonical imported reference node beyond its declared frame-rounding bound");
        maxBoundaryNodeReferenceDifference=std::max(maxBoundaryNodeReferenceDifference,coordinateDelta);
    }

    std::string sourceAssetSHA256;
    if(!digest(files.detailSourceAsset,sourceAssetSHA256,error))return false;
    if(sourceAssetSHA256!=files.detailDomain.sourceSHA256)
        return fail("detail bridge source asset SHA-256 differs from its registered source domain");
    PhysicalSkinAuthoringDetailBridgeInput next;
    next.detailDomain=files.detailDomain;
    next.physicalBoundaryDomain=physicalBoundary.sourceDomain;
    next.maximumBoundaryProjectionDistanceM=files.maximumBoundaryProjectionDistanceM;
    next.maximumTetExtrapolationDistanceM=files.maximumTetExtrapolationDistanceM;
    next.maximumSourceParentTetHops=8u;
    next.sourceParentNodeCount=bundle.parent.referenceWorldPoints.size();
    next.sourceParentTetNodeRows=bundle.parent.tetrahedra;
    next.sourceParentTetMaterialIndices=bundle.parent.materialIndices;
    next.sourceParentTetAnatomicalRegionIndices=bundle.parent.anatomicalRegionIndices;
    next.sourceParentTetStableSourceFaceIDs=bundle.parent.stableSourceFaceIdentities;
    next.physicalBoundaryReferenceWorldPoints.resize(physicalBoundary.volumeNodeRows.size());
    next.physicalBoundarySourceVertexRows=physicalBoundary.authoringVertexRows;
    next.physicalBoundaryGeometricDonorBodyIndices=physicalBoundary.parentGeometricDonorBodyIndices;
    next.physicalBoundaryFrameOwnerBodyIndices=physicalBoundary.frameOwnerBodyIndices;
    next.physicalBoundaryVolumeNodeRows=physicalBoundary.volumeNodeRows;
    next.physicalBoundaryFaces=physicalBoundary.roleLocalFaces;
    for(std::size_t row=0;row<physicalBoundary.volumeNodeRows.size();++row)
        next.physicalBoundaryReferenceWorldPoints[row]=
            bundle.referenceWorldPoints[physicalBoundary.volumeNodeRows[row]];
    next.mechanicalReferenceWorldPoints=bundle.referenceWorldPoints;
    next.mechanicalTetNodeRows=bundle.childTetrahedra;
    next.mechanicalTetMaterialIndices=bundle.childMaterialIndices;
    next.mechanicalTetGeometricDonorBodyIndices=bundle.childDonorBodyIndices;
    next.mechanicalTetAnatomicalRegionIndices=bundle.childAnatomicalRegionIndices;
    std::vector<std::vector<std::uint32_t>> sourceParentTetAdjacency;
    if(next.sourceParentTetNodeRows.empty()||
       next.sourceParentTetMaterialIndices.size()!=next.sourceParentTetNodeRows.size()||
       next.sourceParentTetAnatomicalRegionIndices.size()!=next.sourceParentTetNodeRows.size()||
       next.sourceParentTetStableSourceFaceIDs.size()!=next.sourceParentTetNodeRows.size()||
       !physical_skin_authoring_detail_bridge_detail::buildParentTetFaceAdjacency(
           next.sourceParentTetNodeRows,sourceParentTetAdjacency,error))
        return fail("detail bridge requires valid authenticated source-parent adjacency: "+error);

    static const std::vector<std::string> nodeHeader{
        "vertex","donor","wx","wy","wz","nx","ny","nz","lx","ly","lz",
        "m00","m01","m02","m10","m11","m12","m20","m21","m22",
        "source_domain","source_mapping_kind","source_vertex_row","source_face_row",
        "source_bary0","source_bary1","source_bary2","source_residual_x_m",
        "source_residual_y_m","source_residual_z_m","source_registration_method",
        "frame_owner_method","parent_registration_kind","parent_tet_row",
        "parent_bary0","parent_bary1","parent_bary2","parent_bary3",
        "parent_residual_x_m","parent_residual_y_m","parent_residual_z_m",
        "parent_registration_distance_m","parent_registration_tie_count",
        "parent_tied_tet_rows_json","parent_material_index",
        "parent_geometric_donor_body","parent_anatomical_region_id",
        "parent_stable_source_face_id"};
    std::vector<std::uint32_t> sourceParentNode(files.detailDomain.vertexCount,NM_INVALID_INDEX);
    std::vector<std::int64_t> sourceFaceRows(files.detailDomain.vertexCount,-1);
    std::vector<std::string> sourceMappingKind(files.detailDomain.vertexCount);
    std::vector<std::string> sourceParentRegistrationKind(files.detailDomain.vertexCount);
    std::vector<std::array<double,3>> sourceBaryRows(files.detailDomain.vertexCount);
    std::vector<double> sourceParentRegistrationDistance(files.detailDomain.vertexCount,0.0);
    std::vector<std::uint32_t> sourceParentTieCounts(files.detailDomain.vertexCount,0u);
    std::vector<std::vector<std::uint32_t>> sourceParentTiedTets(files.detailDomain.vertexCount);
    std::vector<std::uint8_t> sourceSeen(files.detailDomain.vertexCount,0u);
    next.detailReferenceWorldPoints.resize(files.detailDomain.vertexCount);
    next.detailReferenceWorldNormals.resize(files.detailDomain.vertexCount);
    next.detailFullBodyWeights86.resize(files.detailDomain.vertexCount);
    next.detailDominantBindingIndices.resize(files.detailDomain.vertexCount,NM_INVALID_INDEX);
    next.detailFaces.reserve(files.detailDomain.faceCount);
    next.detailSourceVertexRows.resize(files.detailDomain.vertexCount);
    next.detailGeometricDonorBodyIndices.resize(files.detailDomain.vertexCount);
    next.detailRegisteredSourceDonorBodyIndices.resize(files.detailDomain.vertexCount,NM_INVALID_INDEX);
    next.detailMaterialIndices.resize(files.detailDomain.vertexCount);
    next.detailAnatomicalRegionIndices.resize(files.detailDomain.vertexCount);
    next.detailFrameOwnerBodyIndices.resize(files.detailDomain.vertexCount);
    next.detailBoundaryProjectionDonorBodyIndices.resize(files.detailDomain.vertexCount);
    next.detailRegionScopedDonorFallback.resize(files.detailDomain.vertexCount);
    next.detailRequiresParentTetPath.resize(files.detailDomain.vertexCount);
    next.detailParentTetRows.resize(files.detailDomain.vertexCount);
    next.detailStableSourceFaceIDs.resize(files.detailDomain.vertexCount);
    next.detailRepresentativeBoundaryVertexRows.resize(files.detailDomain.vertexCount);
    next.detailExpectedBoundaryFaceRows.resize(files.detailDomain.vertexCount);
    next.detailExpectedBoundaryFaceBarycentrics.resize(files.detailDomain.vertexCount);
    next.detailMechanicalTetRows.resize(files.detailDomain.vertexCount);
    std::size_t rows=0u;std::array<std::string,9> hashes{};
    hashes[7]=historicalMapSHA256;hashes[8]=canonicalTetGenNodeSHA256;
    if(!readPinnedCSV(files.detailNodes,nodeHeader,files.detailDomain.vertexCount,
            files.detailDomain.vertexCount,[&](const Row& row,const std::size_t index,std::string& e) {
        auto bad=[&](const char* message){e=std::string(message)+" at fine vertex "+std::to_string(index);return false;};
        std::uint32_t vertex=0u,donor=0u,parentTet=0u,material=0u,parentDonor=0u,region=0u,stable=0u,tieCount=0u;
        std::int64_t sourceVertex=-1,sourceFace=-1;
        V3 point{},normal{};
        double registrationDistance=0.0;
        std::vector<std::uint32_t> ties;
        if(!number(row,0,vertex)||vertex!=index||!number(row,1,donor)||
           !pointFrom(row,2,point)||!pointFrom(row,5,normal)||
           !signedNumber(row,22,sourceVertex)||sourceVertex<0||
           static_cast<std::uint64_t>(sourceVertex)>=files.detailDomain.vertexCount||
           !signedNumber(row,23,sourceFace)||sourceFace>=static_cast<std::int64_t>(files.detailDomain.faceCount)||
           row[20]!=files.detailDomain.domain||row[21].empty()||
           row[31]!="parent_tet_full86_dominant_binding_current_pose"||
           row[32].empty()||!number(row,33,parentTet)||parentTet>=bundle.parent.tetrahedra.size()||
           !finiteNumber(row,41,registrationDistance)||registrationDistance<0.0||
           !number(row,42,tieCount)||!jsonUIntArray(row[43],ties)||ties.empty()||
           !number(row,44,material)||!number(row,45,parentDonor)||!number(row,46,region)||
           !number(row,47,stable))
            return bad("registered detail node has invalid identity, ancestry, or pose fields");
        const double normalLength=std::sqrt(dot(normal,normal));
        if(!std::isfinite(normalLength)||normalLength<=1.0e-12)return bad("registered detail normal is zero or nonfinite");
        for(double& x:normal)x/=normalLength;
        std::array<double,3> sourceBary{};
        double barySum=0.0;
        for(unsigned k=0;k<3u;++k) {
            if(!finiteNumber(row,24u+k,sourceBary[k])||sourceBary[k]<-1.0e-8||sourceBary[k]>1.0+1.0e-8)
                return bad("registered detail source barycentric is invalid");
            barySum+=sourceBary[k];
        }
        if(sourceFace<0) {
            if(row[21]!="vertex"||std::abs(sourceBary[0])+std::abs(sourceBary[1])+std::abs(sourceBary[2])>1.0e-12)
                return bad("direct typed detail source row has inconsistent source-face ancestry");
        } else if(std::abs(barySum-1.0)>1.0e-8)return bad("registered detail source barycentrics do not normalize");
        if(parentTet>=bundle.parent.materialIndices.size()||
           material!=bundle.parent.materialIndices[parentTet]||
           parentDonor!=bundle.parent.donorBodyIndices[parentTet]||
           region!=bundle.parent.anatomicalRegionIndices[parentTet]||
           stable!=bundle.parent.stableSourceFaceIdentities[parentTet])
            return bad("registered detail ancestry labels differ from the pinned parent cell");
        const auto sourceRow=static_cast<std::uint32_t>(sourceVertex);
        if(sourceSeen[sourceRow])return bad("typed source vertex is duplicated");
        sourceSeen[sourceRow]=1u;
        next.detailReferenceWorldPoints[index]=point;
        next.detailReferenceWorldNormals[index]=normal;
        next.detailSourceVertexRows[index]=sourceRow;
        next.detailGeometricDonorBodyIndices[index]=parentDonor;
        next.detailRegisteredSourceDonorBodyIndices[index]=donor;
        next.detailMaterialIndices[index]=material;
        next.detailAnatomicalRegionIndices[index]=region;
        next.detailParentTetRows[index]=parentTet;
        next.detailStableSourceFaceIDs[index]=stable;
        sourceParentNode[index]=parentTet;
        sourceFaceRows[index]=sourceFace;
        sourceMappingKind[index]=row[21];
        sourceParentRegistrationKind[index]=row[32];
        sourceBaryRows[index]=sourceBary;
        sourceParentRegistrationDistance[index]=registrationDistance;
        sourceParentTieCounts[index]=tieCount;
        sourceParentTiedTets[index]=std::move(ties);
        return true;
    },rows,hashes[0],error))return false;

    static const std::vector<std::string> faceHeader{
        "face","v0","v1","v2","face_source_layer","face_source_body","face_source_bone",
        "face_source_face_id","face_source_original_id","face_source_identity0","face_source_identity1"};
    std::vector<std::uint8_t> detailUsed(files.detailDomain.vertexCount,0u);
    if(!readPinnedCSV(files.detailFaces,faceHeader,files.detailDomain.faceCount,
            files.detailDomain.faceCount,[&](const Row& row,const std::size_t index,std::string& e) {
        Face face{};std::uint32_t id=0u;
        if(!number(row,0,id)||id!=index||!number(row,1,face[0])||
           !number(row,2,face[1])||!number(row,3,face[2])||!validFace(face)||
           face[0]>=files.detailDomain.vertexCount||face[1]>=files.detailDomain.vertexCount||
           face[2]>=files.detailDomain.vertexCount) {
            e="registered detail face has invalid typed vertex rows at face "+std::to_string(index);return false;
        }
        next.detailFaces.push_back(face);
        for(const auto v:face)detailUsed[v]=1u;
        return true;
    },rows,hashes[1],error))return false;
    if(std::find(detailUsed.begin(),detailUsed.end(),std::uint8_t(0u))!=detailUsed.end())
        return fail("registered detail face topology does not cover the complete typed vertex domain");

    static const std::vector<std::string> weightHeader=[] {
        std::vector<std::string> h{"vertex"};
        for(unsigned i=0;i<86u;++i)h.push_back("w"+std::to_string(i));
        return h;
    }();
    std::vector<std::uint32_t> maxWeightLane(files.detailDomain.vertexCount,NM_INVALID_INDEX);
    if(!readPinnedCSV(files.detailFull86Weights,weightHeader,files.detailDomain.vertexCount,
            files.detailDomain.vertexCount,[&](const Row& row,const std::size_t index,std::string& e) {
        std::uint32_t vertex=0u;double sum=0.0,maxValue=-1.0;std::uint32_t maxLane=0u;
        if(!number(row,0,vertex)||vertex!=index) {
            e="full86 row identity is invalid at fine vertex "+std::to_string(index);return false;
        }
        for(unsigned lane=0;lane<86u;++lane) {
            double w=0.0;
            if(!finiteNumber(row,1u+lane,w)||w<0.0) {
                e="full86 row has a negative or nonfinite lane at fine vertex "+std::to_string(index);return false;
            }
            next.detailFullBodyWeights86[index][lane]=w;
            sum+=w;
            if(w>maxValue){maxValue=w;maxLane=lane;}
        }
        if(!std::isfinite(sum)||std::abs(sum-1.0)>1.0e-4) {
            e="full86 row does not normalize at fine vertex "+std::to_string(index);return false;
        }
        maxWeightLane[index]=maxLane;return true;
    },rows,hashes[2],error))return false;

    static const std::vector<std::string> ownerHeader{
        "vertex","dominant_binding_index","dominant_binding_body","parent_geometric_donor_body",
        "parent_tet_row","parent_bary0","parent_bary1","parent_bary2","parent_bary3",
        "parent_anatomical_region_id"};
    if(!readPinnedCSV(files.detailFull86Owners,ownerHeader,files.detailDomain.vertexCount,
            files.detailDomain.vertexCount,[&](const Row& row,const std::size_t index,std::string& e) {
        std::uint32_t vertex=0u,lane=0u,body=0u,donor=0u,parentTet=0u,region=0u;
        if(!number(row,0,vertex)||vertex!=index||!number(row,1,lane)||lane>=86u||
           !number(row,2,body)||body==NM_INVALID_INDEX||!number(row,3,donor)||
           !number(row,4,parentTet)||!number(row,9,region)||
           lane!=maxWeightLane[index]||donor!=next.detailGeometricDonorBodyIndices[index]||
           parentTet!=sourceParentNode[index]||region!=next.detailAnatomicalRegionIndices[index]) {
            e="full86 frame-owner row differs from its normalized weights or parent registration at fine vertex "+
              std::to_string(index);return false;
        }
        for(unsigned k=0;k<4u;++k) {
            double bary=0.0;
            if(!finiteNumber(row,5u+k,bary)||bary< -1.0e-8||bary>1.0+1.0e-8) {
                e="full86 owner parent barycentric is invalid at fine vertex "+std::to_string(index);return false;
            }
        }
        next.detailDominantBindingIndices[index]=lane;
        next.detailFrameOwnerBodyIndices[index]=body;
        return true;
    },rows,hashes[3],error))return false;

    static const std::vector<std::string> coarseFaceHeader{
        "role","face_row","parent_tet_row","volume_node0","volume_node1","volume_node2",
        "material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id",
        "opposite_parent_tet_row","opposite_material_index","opposite_geometric_donor_body",
        "opposite_anatomical_region_id","opposite_stable_source_face_id"};
    static const std::vector<std::string> refinedCoarseFaceHeader{
        "role","face_row","child_tet_row","parent_tet_row","volume_node0","volume_node1","volume_node2",
        "material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id",
        "opposite_parent_tet_row","opposite_material_index","opposite_geometric_donor_body",
        "opposite_anatomical_region_id","opposite_stable_source_face_id"};
    struct CoarseFace {
        Face nodes{};
        std::uint32_t child=NM_INVALID_INDEX,parent=NM_INVALID_INDEX;
        std::uint32_t material=0,donor=0,region=0,stable=0;
    };
    std::vector<CoarseFace> coarseFaces;
    std::size_t coarseRoleRows=0u,allCoarseRows=0u;
    bool refinedCoarseSchema=false;
    if(!readPinnedCSV(files.coarseBoundaryFaces,coarseFaceHeader,200000u,0u,
            [&](const Row& row,const std::size_t,std::string& e) {
        if(row[0]!="O"&&row[0]!="I"&&row[0]!="D") {
            e="coarse boundary face uses an unknown role";return false;
        }
        ++allCoarseRows;
        if(row[0]!="O")return true;
        const std::size_t childColumn=refinedCoarseSchema?2u:1u;
        const std::size_t parentColumn=refinedCoarseSchema?3u:2u;
        const std::size_t nodeColumn=parentColumn+1u;
        const std::size_t labelColumn=nodeColumn+3u;
        std::uint32_t faceRow=0u;CoarseFace face;
        if(!number(row,1,faceRow)||faceRow!=coarseRoleRows||
           (refinedCoarseSchema&&!number(row,childColumn,face.child))||
           !number(row,parentColumn,face.parent)||
           !number(row,nodeColumn,face.nodes[0])||
           !number(row,nodeColumn+1u,face.nodes[1])||
           !number(row,nodeColumn+2u,face.nodes[2])||
           !validFace(face.nodes)||
           !number(row,labelColumn,face.material)||
           !number(row,labelColumn+1u,face.donor)||
           !number(row,labelColumn+2u,face.region)||
           !number(row,labelColumn+3u,face.stable)||
           face.parent>=bundle.parent.tetrahedra.size()) {
            e="coarse O boundary-face row is malformed at face "+std::to_string(coarseRoleRows);return false;
        }
        if(!refinedCoarseSchema)face.child=face.parent;
        if(face.child>=bundle.childTetrahedra.size()) {
            e="coarse O boundary face exceeds the imported child-cell domain";return false;
        }
        for(const auto node:face.nodes)if(node>=bundle.referenceWorldPoints.size()) {
            e="coarse O boundary face exceeds the imported child volume-node domain";return false;
        }
        const auto& cell=bundle.childTetrahedra[face.child];
        for(const auto node:face.nodes)
            if(std::find(cell.begin(),cell.end(),node)==cell.end()) {
                e="coarse O boundary face is not a face of its declared child cell";return false;
            }
        const auto parent=face.parent;
        if(bundle.childDominantParentTetrahedra[face.child]!=parent||
           bundle.childMaterialIndices[face.child]!=face.material||
           bundle.childDonorBodyIndices[face.child]!=face.donor||
           bundle.childAnatomicalRegionIndices[face.child]!=face.region||
           bundle.parent.materialIndices[parent]!=face.material||
           bundle.parent.donorBodyIndices[parent]!=face.donor||
           bundle.parent.anatomicalRegionIndices[parent]!=face.region||
           bundle.parent.stableSourceFaceIdentities[parent]!=face.stable||
           (bundle.childStableSourceFaceIdentities.size()==bundle.childTetrahedra.size()&&
            bundle.childStableSourceFaceIdentities[face.child]!=face.stable)) {
            e="coarse O face child-to-parent ancestry or labels differ from the authenticated imported cell route";return false;
        }
        coarseFaces.push_back(face);++coarseRoleRows;return true;
    },rows,hashes[5],error,&refinedCoarseFaceHeader,&refinedCoarseSchema))return false;
    if(refinedCoarseSchema) {
        const auto& route=bundle.conformingLineageMassRoute;
        const auto validSHA=[](const std::string& value) {
            return physical_skin_child_mass_detail::validSHA256(value);
        };
        if((!files.allowPrecanonicalExportSeed&&
            !files.allowAuthenticatedRefinedChildRouteForSourceOnlyPreflight)||
           !bundle.conformingLineageMassRouteRequired||
           !bundle.conformingLineageMassTransportUsed||
           !route.authenticated||
           route.refinedSourceCellCount!=bundle.childTetrahedra.size()||
           route.childRefinedSourceRows.size()!=bundle.childTetrahedra.size()||
           route.childImmutableParentRows!=bundle.childDominantParentTetrahedra||
           route.candidateOperationalReferenceDomainSHA256!=bundle.operationalReferenceDomainSHA256||
           route.candidateCompiledDataSHA256!=
               physical_skin_child_mass_mapping_detail::operationalChildCompiledDataSHA256(bundle)||
           route.verificationSHA256!=
               physical_skin_child_mass_mapping_detail::verifiedConformingMassRouteSHA256(route)||
           bundle.childStableSourceFaceIdentities.size()!=bundle.childTetrahedra.size())
            return fail("refined child-face map lacks the authenticated 002->530->006 parent route");
        for(const auto& value:{route.parentMassPacketSHA256,route.parentSourceContributionSHA256,
                route.sourceVolumeSHA256,route.refinedVolumeSHA256,route.candidateVolumeSHA256,
                route.refinementVerificationSHA256,route.sourceDonorLedgerVerificationSHA256,
                route.pullbackVerificationSHA256,route.chainVerificationSHA256,
                route.candidateOperationalReferenceDomainSHA256,route.candidateCompiledDataSHA256})
            if(!validSHA(value))
                return fail("refined child-face map route has a missing 002->530->006 proof pin");
        std::vector<std::uint8_t> refinedRowsSeen(route.refinedSourceCellCount,0u);
        for(const auto row:route.childRefinedSourceRows) {
            if(row>=refinedRowsSeen.size()||refinedRowsSeen[row])
                return fail("refined child-face map route repeats or exceeds a 530 source row");
            refinedRowsSeen[row]=1u;
        }
    } else if(!childParentIdentity) {
        return fail("refined child volume requires explicit child_tet_row and parent_tet_row columns");
    }
    (void)allCoarseRows;
    if(coarseFaces.size()!=physicalBoundary.roleLocalFaces.size())
        return fail("coarse O boundary-face map count differs from the imported typed boundary surface");
    std::map<Face,std::uint32_t> boundaryFaceRows;
    for(std::size_t row=0;row<physicalBoundary.roleLocalFaces.size();++row) {
        const auto& face=physicalBoundary.roleLocalFaces[row];
        if(!validFace(face)||face[0]>=physicalBoundary.volumeNodeRows.size()||
           face[1]>=physicalBoundary.volumeNodeRows.size()||face[2]>=physicalBoundary.volumeNodeRows.size())
            return fail("imported typed O boundary has an invalid role-local face");
        Face volumeFace{
            physicalBoundary.volumeNodeRows[face[0]],
            physicalBoundary.volumeNodeRows[face[1]],
            physicalBoundary.volumeNodeRows[face[2]]};
        if(!boundaryFaceRows.emplace(sortedFace(volumeFace),static_cast<std::uint32_t>(row)).second)
            return fail("imported typed O boundary repeats a physical triangle");
    }
    std::vector<std::uint32_t> coarseFaceToBoundaryFace(coarseFaces.size(),NM_INVALID_INDEX);
    std::vector<std::uint32_t> physicalBoundaryFaceChildTetRows(
        physicalBoundary.roleLocalFaces.size(),NM_INVALID_INDEX);
    std::vector<std::uint8_t> boundaryFaceSeen(physicalBoundary.roleLocalFaces.size(),0u);
    next.physicalBoundaryFaceParentTetRows.assign(physicalBoundary.roleLocalFaces.size(),NM_INVALID_INDEX);
    next.physicalBoundaryStableFaceIDs.assign(physicalBoundary.roleLocalFaces.size(),NM_INVALID_INDEX);
    next.physicalBoundaryFaceMaterialIndices.assign(physicalBoundary.roleLocalFaces.size(),NM_INVALID_INDEX);
    next.physicalBoundaryFaceGeometricDonorBodyIndices.assign(physicalBoundary.roleLocalFaces.size(),NM_INVALID_INDEX);
    next.physicalBoundaryFaceAnatomicalRegionIndices.assign(physicalBoundary.roleLocalFaces.size(),NM_INVALID_INDEX);
    next.physicalBoundaryFaceFrameOwnerBodyIndices.assign(physicalBoundary.roleLocalFaces.size(),NM_INVALID_INDEX);
    for(std::size_t coarseRow=0;coarseRow<coarseFaces.size();++coarseRow) {
        const auto& coarse=coarseFaces[coarseRow];
        auto found=boundaryFaceRows.find(sortedFace(coarse.nodes));
        if(found==boundaryFaceRows.end()||boundaryFaceSeen[found->second])
            return fail("coarse O face rows do not bijectively match imported physical-boundary triangles");
        const auto boundaryRow=found->second;
        boundaryFaceSeen[boundaryRow]=1u;coarseFaceToBoundaryFace[coarseRow]=boundaryRow;
        next.physicalBoundaryFaceParentTetRows[boundaryRow]=coarse.parent;
        physicalBoundaryFaceChildTetRows[boundaryRow]=coarse.child;
        next.physicalBoundaryStableFaceIDs[boundaryRow]=coarse.stable;
        next.physicalBoundaryFaceMaterialIndices[boundaryRow]=coarse.material;
        next.physicalBoundaryFaceGeometricDonorBodyIndices[boundaryRow]=coarse.donor;
        next.physicalBoundaryFaceAnatomicalRegionIndices[boundaryRow]=coarse.region;
        next.physicalBoundaryFaceFrameOwnerBodyIndices[boundaryRow]=
            majorityOwner(physicalBoundary.roleLocalFaces[boundaryRow],physicalBoundary.frameOwnerBodyIndices);
    }
    if(std::find(boundaryFaceSeen.begin(),boundaryFaceSeen.end(),std::uint8_t(0u))!=boundaryFaceSeen.end())
        return fail("coarse O face map omitted an imported physical-boundary triangle");

    auto mapHeader=seedMapHeader();
    const auto& selectedHeader=nativeSelectionHeader();
    if(!files.allowPrecanonicalExportSeed)
        mapHeader.insert(mapHeader.end(),selectedHeader.begin(),selectedHeader.end());
    double maxProducerDistance=0.0,maxActualDistance=0.0,maxResidualDelta=0.0;
    std::size_t nativeSelectionRows=0u,nativeFaceChanges=0u;
    std::size_t pathRows=0u,donorRemaps=0u;std::uint32_t maxHops=0u;
    std::vector<std::uint8_t> seenFine(files.detailDomain.vertexCount,0u);
    std::ifstream historicalMapInput(files.historicalFineToBoundaryMap.path);
    if(!historicalMapInput)return fail("cannot open separately pinned historical detail map");
    std::string historicalLine;
    Row historicalFields;
    std::vector<std::string> historicalHeader(mapHeader.begin(),mapHeader.begin()+45);
    if(!std::getline(historicalMapInput,historicalLine))
        return fail("historical detail map is empty");
    if(!historicalLine.empty()&&historicalLine.back()==static_cast<char>(13))historicalLine.pop_back();
    if(!parseLine(historicalLine,historicalFields)||historicalFields!=historicalHeader)
        return fail("historical detail map header differs from the preserved 45-column producer contract");
    if(!readPinnedCSV(files.fineToBoundaryMap,mapHeader,files.detailDomain.vertexCount,
            files.detailDomain.vertexCount,[&](const Row& row,const std::size_t index,std::string& e) {
        auto bad=[&](const char* message){e=std::string(message)+" at detail map row "+std::to_string(index);return false;};
        std::uint32_t fine=0u,faceRow=0u,primaryTet=0u,mappedTet=0u,startTet=0u,hops=0u;
        Face faceNodes{};
        std::uint32_t primaryMaterial=0u,primaryDonor=0u,primaryRegion=0u,primaryStable=0u;
        std::uint32_t mappedMaterial=0u,mappedDonor=0u,mappedRegion=0u,mappedStable=0u;
        std::uint32_t fineMaterial=0u,fineDonor=0u,fineRegion=0u,fineStable=0u,fineParent=0u;
        std::uint32_t tieCount=0u,mapSourceVertex=0u;
        std::int64_t mapSourceFace=-1;
        double producerDistance=0.0,registrationDistance=0.0,canonicalDistance=0.0;
        std::array<double,3> historicalBary{},historicalResidual{},bary{},residual{},sourceBary{};
        std::vector<std::uint32_t> path,ties,incident;
        if(!number(row,0,fine)||fine!=index||row[1]!="O"||
           !number(row,2,faceRow)||faceRow>=coarseFaces.size()||
           !number(row,3,primaryTet)||!number(row,4,mappedTet)||!number(row,5,startTet)||
           !number(row,6,hops)||!jsonUIntArray(row[7],path)||
           !number(row,8,faceNodes[0])||!number(row,9,faceNodes[1])||!number(row,10,faceNodes[2])||
           !number(row,11,primaryMaterial)||!number(row,12,primaryDonor)||
           !number(row,13,primaryRegion)||!number(row,14,primaryStable)||
           !number(row,15,mappedMaterial)||!number(row,16,mappedDonor)||
           !number(row,17,mappedRegion)||!number(row,18,mappedStable)||
           !finiteNumber(row,19,historicalBary[0])||!finiteNumber(row,20,historicalBary[1])||
           !finiteNumber(row,21,historicalBary[2])||!finiteNumber(row,22,historicalResidual[0])||
           !finiteNumber(row,23,historicalResidual[1])||!finiteNumber(row,24,historicalResidual[2])||
           !finiteNumber(row,25,producerDistance)||producerDistance<0.0||
           (row[26]!="primary_registered_parent_tet_boundary_stable_face"&&
            row[26]!="registered_parent_tie_boundary_stable_face"&&
            row[26]!="bounded_same_material_region_parent_tet_walk_to_stable_boundary_face")||
           !number(row,27,fineMaterial)||!number(row,28,fineDonor)||
           !number(row,29,fineRegion)||!number(row,30,fineStable)||
           row[31]!=sourceParentRegistrationKind[fine]||
           !number(row,32,fineParent)||!jsonUIntArray(row[33],ties)||ties.empty()||
           !finiteNumber(row,34,registrationDistance)||registrationDistance<0.0||
           !number(row,35,tieCount)||row[36]!="parent_tet_full86_dominant_binding_current_pose"||
           row[37]!=files.detailDomain.domain||
           row[38]!=sourceMappingKind[fine]||!number(row,39,mapSourceVertex)||
           !signedNumber(row,40,mapSourceFace)||!finiteNumber(row,41,sourceBary[0])||
           !finiteNumber(row,42,sourceBary[1])||!finiteNumber(row,43,sourceBary[2])||
           !jsonUIntArray(row[44],incident)||row[45]!=historicalMapSHA256||
           row[46]!=canonicalTetGenNodeSHA256||
           !finiteNumber(row,47,bary[0])||!finiteNumber(row,48,bary[1])||
           !finiteNumber(row,49,bary[2])||!finiteNumber(row,50,residual[0])||
           !finiteNumber(row,51,residual[1])||!finiteNumber(row,52,residual[2])||
           !finiteNumber(row,53,canonicalDistance)||canonicalDistance<0.0)
            return bad("malformed row, changed historical pin, or wrong canonical imported-node pin");
        std::uint32_t selectedFaceRow=NM_INVALID_INDEX,selectedParent=NM_INVALID_INDEX;
        std::uint32_t selectedMaterial=NM_INVALID_INDEX,selectedDonor=NM_INVALID_INDEX;
        std::uint32_t selectedRegion=NM_INVALID_INDEX,selectedStable=NM_INVALID_INDEX;
        std::uint32_t selectedFrameOwner=NM_INVALID_INDEX,selectedRepresentative=NM_INVALID_INDEX;
        std::uint32_t selectedWalkStart=NM_INVALID_INDEX,selectedHops=0u;
        Face selectedVolumeNodes{};
        std::array<double,3> selectedBary{},selectedResidual{};
        double selectedDistance=0.0;
        std::vector<std::uint32_t> selectedPath;
        if(!files.allowPrecanonicalExportSeed&&
           (!number(row,54,selectedFaceRow)||!number(row,55,selectedVolumeNodes[0])||
            !number(row,56,selectedVolumeNodes[1])||!number(row,57,selectedVolumeNodes[2])||
            !number(row,58,selectedParent)||!number(row,59,selectedMaterial)||
            !number(row,60,selectedDonor)||!number(row,61,selectedRegion)||
            !number(row,62,selectedStable)||!number(row,63,selectedFrameOwner)||
            !number(row,64,selectedRepresentative)||!number(row,65,selectedWalkStart)||
            !number(row,66,selectedHops)||!jsonUIntArray(row[67],selectedPath)||
            !finiteNumber(row,68,selectedBary[0])||!finiteNumber(row,69,selectedBary[1])||
            !finiteNumber(row,70,selectedBary[2])||!finiteNumber(row,71,selectedResidual[0])||
            !finiteNumber(row,72,selectedResidual[1])||!finiteNumber(row,73,selectedResidual[2])||
            !finiteNumber(row,74,selectedDistance)||selectedDistance<0.0))
            return bad("canonical native selected-face fields are malformed");
        if(!std::getline(historicalMapInput,historicalLine))
            return bad("historical map is shorter than the canonical map");
        if(!historicalLine.empty()&&historicalLine.back()==static_cast<char>(13))historicalLine.pop_back();
        if(!parseLine(historicalLine,historicalFields)||historicalFields.size()!=45u)
            return bad("historical map has a malformed preserved row");
        for(std::size_t column=0u;column<45u;++column)
            if(historicalFields[column]!=row[column])
                return bad("historical producer columns were altered in the canonical map copy");
        const auto& coarse=coarseFaces[faceRow];
        const auto physicalFaceRow=coarseFaceToBoundaryFace[faceRow];
        if(physicalFaceRow>=next.physicalBoundaryFaces.size())
            return bad("coarse face does not map into the imported physical boundary");
        const auto expectedSourceFace=sourceFaceRows[fine];
        if(seenFine[fine]||fineMaterial!=next.detailMaterialIndices[fine]||
           fineDonor!=next.detailGeometricDonorBodyIndices[fine]||
           fineRegion!=next.detailAnatomicalRegionIndices[fine]||
           fineStable!=next.detailStableSourceFaceIDs[fine]||
           fineParent!=next.detailParentTetRows[fine]||
           mapSourceVertex!=next.detailSourceVertexRows[fine]||mapSourceFace!=expectedSourceFace)
            return bad("fine source, frame-owner, or parent labels differ from the pinned detail node row");
        if(std::find(sourceParentTiedTets[fine].begin(),sourceParentTiedTets[fine].end(),startTet)==
           sourceParentTiedTets[fine].end())
            return bad("boundary ancestry starts outside the pinned parent registration tie witness");
        seenFine[fine]=1u;
        if(ties!=sourceParentTiedTets[fine]||tieCount!=sourceParentTieCounts[fine]||
           std::abs(registrationDistance-sourceParentRegistrationDistance[fine])>1.0e-12)
            return bad("parent registration tie witness differs from the pinned detail node row");
        for(unsigned k=0;k<3u;++k)
            if(!std::isfinite(sourceBary[k])||
               std::abs(sourceBary[k]-sourceBaryRows[fine][k])>1.0e-12)
                return bad("fine source barycentrics differ from the pinned detail node row");
        const bool walkedStableFace=row[26]==
            "bounded_same_material_region_parent_tet_walk_to_stable_boundary_face";
        if(!validFace(faceNodes)||sortedFace(faceNodes)!=sortedFace(coarse.nodes)||
           primaryTet!=coarse.parent||mappedTet!=coarse.parent||
           primaryMaterial!=coarse.material||mappedMaterial!=coarse.material||
           primaryDonor!=coarse.donor||mappedDonor!=coarse.donor||
           primaryRegion!=coarse.region||mappedRegion!=coarse.region||
           primaryStable!=coarse.stable||mappedStable!=coarse.stable||
           (!walkedStableFace&&fineStable!=coarse.stable)||
           fineMaterial!=coarse.material||fineRegion!=coarse.region)
            return bad("selected O face, mapped-side labels, and fine ancestry do not share one pinned boundary identity");
        if(startTet>=bundle.parent.tetrahedra.size()||mappedTet>=bundle.parent.tetrahedra.size()||
           path.size()!=std::size_t(hops)+1u||path.empty()||path.front()!=startTet||
           path.back()!=mappedTet||hops>next.maximumSourceParentTetHops)
            return bad("parent-tet walk endpoints or hop count are invalid");
        for(const auto cell:path)
            if(cell>=bundle.parent.tetrahedra.size()||
               bundle.parent.materialIndices[cell]!=fineMaterial||
               bundle.parent.anatomicalRegionIndices[cell]!=fineRegion)
                return bad("parent-tet walk leaves its same-material/region source patch");
        for(std::size_t step=1u;step<path.size();++step) {
            const auto previous=path[step-1u];
            if(previous>=sourceParentTetAdjacency.size()||
               std::find(sourceParentTetAdjacency[previous].begin(),sourceParentTetAdjacency[previous].end(),path[step])==
                   sourceParentTetAdjacency[previous].end())
                return bad("parent-tet walk contains nonadjacent consecutive source cells");
        }
        if(next.sourceParentTetStableSourceFaceIDs.size()!=bundle.parent.tetrahedra.size()||
           fineParent>=next.sourceParentTetStableSourceFaceIDs.size()||
           mappedTet>=next.sourceParentTetStableSourceFaceIDs.size()||
           next.sourceParentTetStableSourceFaceIDs[fineParent]!=fineStable||
           next.sourceParentTetStableSourceFaceIDs[mappedTet]!=coarse.stable)
            return bad("fine and selected stable-face IDs do not match their authenticated parent tet ancestry");
        if(!std::all_of(incident.begin(),incident.end(),[&](std::uint32_t face) {
                return face<next.detailFaces.size()&&
                    std::find(next.detailFaces[face].begin(),next.detailFaces[face].end(),fine)!=
                        next.detailFaces[face].end();
            })||incident.empty())
            return bad("fine vertex incident-face witnesses do not reference its pinned detail topology");
        double historicalBarySum=0.0;
        for(const double w:historicalBary) {
            if(!std::isfinite(w)||w< -1.0e-8||w>1.0+1.0e-8)
                return bad("historical producer barycentrics are invalid");
            historicalBarySum+=w;
        }
        if(std::abs(historicalBarySum-1.0)>1.0e-8)
            return bad("historical producer barycentrics do not sum to one");
        const V3 historicalResidualPoint{historicalResidual[0],historicalResidual[1],
                                          historicalResidual[2]};
        const double historicalResidualLength=norm(historicalResidualPoint);
        if(!std::isfinite(historicalResidualLength)||
           std::abs(historicalResidualLength-producerDistance)>
               files.producerResidualComparisonToleranceM)
            return bad("historical producer distance differs from its preserved residual columns");
        const V3 sameHistoricalFaceResidual{residual[0],residual[1],residual[2]};
        const double sameHistoricalFaceDistance=norm(sameHistoricalFaceResidual);
        if(!std::isfinite(sameHistoricalFaceDistance)||
           std::abs(sameHistoricalFaceDistance-canonicalDistance)>
               files.producerResidualComparisonToleranceM)
            return bad("canonical same-historical-face distance differs from its named residual fields");
        const V3 historicalFacePoint=baryPoint(bundle.referenceWorldPoints,faceNodes,bary);
        const V3 actualHistoricalFaceResidual=sub(next.detailReferenceWorldPoints[fine],historicalFacePoint);
        const double actualHistoricalFaceDistance=norm(actualHistoricalFaceResidual);
        const double canonicalHistoryResidualDelta=norm(
            sub(actualHistoricalFaceResidual,sameHistoricalFaceResidual));
        const double residualDelta=norm(sub(actualHistoricalFaceResidual,historicalResidualPoint));
        if(!std::isfinite(actualHistoricalFaceDistance)||
           !std::isfinite(canonicalHistoryResidualDelta)||
           canonicalHistoryResidualDelta>files.producerResidualComparisonToleranceM||
           std::abs(actualHistoricalFaceDistance-canonicalDistance)>
               files.producerResidualComparisonToleranceM||
           actualHistoricalFaceDistance>files.maximumBoundaryProjectionDistanceM)
            return bad("same-historical-face canonical projection differs from the imported node pin");

        Face selectedFaceNodes=faceNodes;
        std::uint32_t selectedParentRow=mappedTet,selectedDonorBody=mappedDonor;
        std::uint32_t selectedMaterialIndex=mappedMaterial,selectedRegionID=mappedRegion;
        std::uint32_t selectedStableFace= mappedStable;
        std::uint32_t selectedOwner=NM_INVALID_INDEX,selectedRep=NM_INVALID_INDEX;
        std::uint32_t selectedStart=NM_INVALID_INDEX,selectedHopCount=hops;
        std::vector<std::uint32_t> selectedWalk=path;
        std::array<double,3> selectedFaceBary{};
        V3 selectedCanonicalResidual{};double selectedCanonicalDistance=0.0;
        const std::uint32_t roleFaceRow=files.allowPrecanonicalExportSeed?
            physicalFaceRow:selectedFaceRow;
        if(roleFaceRow>=next.physicalBoundaryFaces.size())
            return bad("canonical selected physical boundary face row is outside the imported O domain");
        const auto& localFace=next.physicalBoundaryFaces[roleFaceRow];
        for(unsigned corner=0;corner<3u;++corner) {
            const auto localVertex=localFace[corner];
            if(localVertex>=physicalBoundary.volumeNodeRows.size())
                return bad("canonical selected face has an invalid role-local vertex");
            const auto volumeNode=physicalBoundary.volumeNodeRows[localVertex];
            if(volumeNode>=bundle.referenceWorldPoints.size())
                return bad("canonical selected face exceeds the imported volume node domain");
            selectedFaceNodes[corner]=volumeNode;
        }
        if(files.allowPrecanonicalExportSeed) {
            selectedParentRow=mappedTet;selectedDonorBody=mappedDonor;
            selectedMaterialIndex=mappedMaterial;selectedRegionID=mappedRegion;
            selectedStableFace=mappedStable;
            selectedOwner=next.physicalBoundaryFaceFrameOwnerBodyIndices[roleFaceRow];
            selectedStart=startTet;selectedHopCount=hops;selectedWalk=path;
            selectedRep=localFace[0];
            double best=-std::numeric_limits<double>::infinity();
            for(unsigned corner=0u;corner<3u;++corner) {
                const auto volumeNode=selectedFaceNodes[corner];
                const auto it=std::find(faceNodes.begin(),faceNodes.end(),volumeNode);
                if(it==faceNodes.end())return bad("historical O face differs from imported role boundary face");
                const auto mapCorner=static_cast<unsigned>(it-faceNodes.begin());
                selectedFaceBary[corner]=bary[mapCorner];
                if(selectedFaceBary[corner]>best) {
                    best=selectedFaceBary[corner];selectedRep=localFace[corner];
                }
            }
        } else {
            if(selectedFaceRow>=next.physicalBoundaryFaces.size()||
               selectedMaterial!=next.physicalBoundaryFaceMaterialIndices[selectedFaceRow]||
               selectedDonor==NM_INVALID_INDEX||
               selectedMaterial!=fineMaterial||selectedRegion!=fineRegion||
               selectedFrameOwner!=
                    next.physicalBoundaryFaceFrameOwnerBodyIndices[selectedFaceRow]||
               selectedParent!=next.physicalBoundaryFaceParentTetRows[selectedFaceRow]||
               selectedRegion!=next.physicalBoundaryFaceAnatomicalRegionIndices[selectedFaceRow]||
               selectedStable!=next.physicalBoundaryStableFaceIDs[selectedFaceRow]||
               selectedDonor!=next.physicalBoundaryFaceGeometricDonorBodyIndices[selectedFaceRow])
                return bad("canonical selected face labels differ from the imported physical O face and source ancestry");
            const Face expectedVolumeNodes{selectedVolumeNodes[0],selectedVolumeNodes[1],selectedVolumeNodes[2]};
            if(expectedVolumeNodes!=selectedFaceNodes)
                return bad("canonical selected volume-node rows differ from its imported role face ordering");
            if(selectedRepresentative>=physicalBoundary.currentWorldPoints.size()||
               std::find(localFace.begin(),localFace.end(),selectedRepresentative)==localFace.end())
                return bad("canonical selected representative is not on its selected imported face");
            if(selectedWalkStart!=fineParent||selectedPath.empty()||
               selectedPath.front()!=selectedWalkStart||selectedPath.back()!=selectedParent||
               selectedPath.size()!=std::size_t(selectedHops)+1u||
               selectedParent!=next.physicalBoundaryFaceParentTetRows[selectedFaceRow])
                return bad("canonical selected ancestry path differs from its selected physical face parent");
            selectedParentRow=selectedParent;selectedDonorBody=selectedDonor;
            selectedMaterialIndex=selectedMaterial;selectedRegionID=selectedRegion;
            selectedStableFace=selectedStable;selectedOwner=selectedFrameOwner;
            selectedRep=selectedRepresentative;selectedStart=selectedWalkStart;
            selectedHopCount=selectedHops;selectedWalk=std::move(selectedPath);
            selectedFaceBary=selectedBary;selectedCanonicalResidual=V3{
                selectedResidual[0],selectedResidual[1],selectedResidual[2]};
            selectedCanonicalDistance=selectedDistance;
            ++nativeSelectionRows;
            if(selectedFaceRow!=physicalFaceRow)++nativeFaceChanges;
        }
        double barySum=0.0;
        for(const double w:selectedFaceBary) {
            if(!std::isfinite(w)||w< -1.0e-8||w>1.0+1.0e-8)
                return bad("native selected-face barycentrics are outside the physical face");
            barySum+=w;
        }
        if(std::abs(barySum-1.0)>2.0e-6)
            return bad("native selected-face barycentrics do not sum to one");
        const V3 selectedPoint=baryPoint(bundle.referenceWorldPoints,selectedFaceNodes,selectedFaceBary);
        const V3 actualSelectedResidual=sub(next.detailReferenceWorldPoints[fine],selectedPoint);
        const double actualSelectedDistance=norm(actualSelectedResidual);
        if(files.allowPrecanonicalExportSeed) {
            selectedCanonicalResidual=actualSelectedResidual;
            selectedCanonicalDistance=actualSelectedDistance;
        }
        const double selectedResidualDelta=norm(sub(actualSelectedResidual,selectedCanonicalResidual));
        if(!std::isfinite(actualSelectedDistance)||!std::isfinite(selectedResidualDelta)||
           selectedResidualDelta>files.producerResidualComparisonToleranceM||
           std::abs(actualSelectedDistance-selectedCanonicalDistance)>
               files.producerResidualComparisonToleranceM||
           actualSelectedDistance>files.maximumBoundaryProjectionDistanceM)
            return bad("canonical native-selected face projection differs from its pinned witness");
        // Seed rows retain a historical producer path whose start is one of
        // the pinned tied registration tets (validated above). It is
        // provenance only; the canonical path is generated by the native
        // selector and is required to start at the primary fine parent.
        if((!files.allowPrecanonicalExportSeed&&selectedStart!=fineParent)||
           selectedParentRow>=bundle.parent.tetrahedra.size()||
           selectedWalk.size()!=std::size_t(selectedHopCount)+1u||selectedWalk.empty()||
           selectedWalk.front()!=selectedStart||selectedWalk.back()!=selectedParentRow||
           selectedHopCount>next.maximumSourceParentTetHops)
            return bad("canonical selected parent-tet ancestry endpoints or hop count are invalid");
        for(const auto cell:selectedWalk)
            if(cell>=bundle.parent.tetrahedra.size()||
               bundle.parent.materialIndices[cell]!=selectedMaterialIndex||
               bundle.parent.anatomicalRegionIndices[cell]!=selectedRegionID)
                return bad("canonical selected parent-tet ancestry leaves its same-material/region source patch");
        if(!files.allowPrecanonicalExportSeed&&
           bundle.parent.stableSourceFaceIdentities[selectedParentRow]!=selectedStableFace)
            return bad("canonical native-selected face lacks matching stable source-face ancestry");
        if(files.allowPrecanonicalExportSeed&&
           (fineParent>=bundle.parent.stableSourceFaceIdentities.size()||
            bundle.parent.stableSourceFaceIdentities[fineParent]!=fineStable||
            bundle.parent.stableSourceFaceIdentities[selectedParentRow]!=selectedStableFace||
            (selectedStableFace!=fineStable&&
             (row[26]!="bounded_same_material_region_parent_tet_walk_to_stable_boundary_face"||
              selectedHopCount==0u||selectedWalk.size()<2u))))
            return bad("precanonical seed lacks authenticated source/target stable-face identities or a bounded walk re-inference");
        if(selectedOwner==NM_INVALID_INDEX)
            return bad("canonical native-selected face lacks a retained frame owner");
        maxProducerDistance=std::max(maxProducerDistance,producerDistance);
        maxActualDistance=std::max(maxActualDistance,actualSelectedDistance);
        maxResidualDelta=std::max(maxResidualDelta,residualDelta);
        const bool donorFallback=next.detailGeometricDonorBodyIndices[fine]!=selectedDonorBody;
        next.detailBoundaryProjectionDonorBodyIndices[fine]=selectedDonorBody;
        if(!files.allowPrecanonicalExportSeed)
            next.detailStableSourceFaceIDs[fine]=selectedStableFace;
        next.detailRegionScopedDonorFallback[fine]=donorFallback?1u:0u;
        next.detailRequiresParentTetPath[fine]=
            (selectedHopCount>0u||donorFallback||selectedStableFace!=fineStable||
             selectedParentRow!=fineParent)?1u:0u;
        next.detailRepresentativeBoundaryVertexRows[fine]=selectedRep;
        next.detailExpectedBoundaryFaceRows[fine]=roleFaceRow;
        next.detailExpectedBoundaryFaceBarycentrics[fine]=selectedFaceBary;
        if(roleFaceRow>=physicalBoundaryFaceChildTetRows.size()||
           physicalBoundaryFaceChildTetRows[roleFaceRow]>=bundle.childTetrahedra.size())
            return bad("selected physical boundary face lacks its authenticated child-cell row");
        next.detailMechanicalTetRows[fine]=physicalBoundaryFaceChildTetRows[roleFaceRow];
        maxHops=std::max(maxHops,selectedHopCount);
        if(selectedHopCount>0u||next.detailRegionScopedDonorFallback[fine]!=0u)++pathRows;
        if(next.detailRegionScopedDonorFallback[fine]!=0u)++donorRemaps;
        return true;
    },rows,hashes[4],error))return false;
    if(std::find(seenFine.begin(),seenFine.end(),std::uint8_t(0u))!=seenFine.end())
        return fail("fine-to-boundary map does not bijectively cover the complete typed detail domain");
    if(std::getline(historicalMapInput,historicalLine))
        return fail("historical detail map has extra rows beyond the canonical map domain");
    std::string historicalMapAfterSHA256,canonicalNodeAfterSHA256;
    if(!physical_skin_child_mass_detail::sha256File(
            files.historicalFineToBoundaryMap.path,historicalMapAfterSHA256,error)||
       historicalMapAfterSHA256!=historicalMapSHA256||
       !physical_skin_child_mass_detail::sha256File(
            files.canonicalTetGenNode.path,canonicalNodeAfterSHA256,error)||
       canonicalNodeAfterSHA256!=canonicalTetGenNodeSHA256)
        return fail("historical map or canonical imported TetGen node file changed during detail import");
    std::string sourceAssetAfterSHA256;
    if(!physical_skin_child_mass_detail::sha256File(
            files.detailSourceAsset.path,sourceAssetAfterSHA256,error)||
       sourceAssetAfterSHA256!=sourceAssetSHA256)
        return fail("detail bridge source asset changed while its CSVs were being imported");
    hashes[6]=sourceAssetAfterSHA256;

    PhysicalSkinAuthoringDetailBridgeCSVReport measured;
    measured.detailRows=files.detailDomain.vertexCount;
    measured.detailFaces=files.detailDomain.faceCount;
    measured.coarseBoundaryFaces=coarseFaces.size();
    measured.ancestryWalkRows=pathRows;measured.donorRemapRows=donorRemaps;
    measured.maximumAncestryHops=maxHops;
    measured.maximumProducerProjectionDistanceM=maxProducerDistance;
    measured.maximumRecomputedProjectionDistanceM=maxActualDistance;
    measured.maximumResidualRebaseDeltaM=maxResidualDelta;
    measured.maximumPhysicalBoundaryNodeReferenceDifferenceM=maxBoundaryNodeReferenceDifference;
    measured.nativeCanonicalSelectionRows=nativeSelectionRows;
    measured.nativeCanonicalFaceChangeRows=nativeFaceChanges;
    measured.inputSHA256=hashes;
    measured.historicalMapSHA256=historicalMapSHA256;
    measured.canonicalTetGenNodeSHA256=canonicalTetGenNodeSHA256;
    measured.physicalAdmission=false;
    output=std::move(next);report=std::move(measured);error.clear();return true;
}


// Runs the same native AcceptedFEMSkinTransfer face selector used by
// preparePhysicalSkinAuthoringDetailBridge and writes its pinned selection
// witness beside the immutable historical producer columns.
[[nodiscard]] inline bool exportPhysicalSkinAuthoringDetailBridgeCanonicalMap(
        const PhysicalSkinAuthoringDetailBridgeCSVFiles& files,
        const PhysicalSkinAuthoringDetailBridgeInput& input,
        const PhysicalSkinAuthoringDetailBridgeCSVReport& loadedReport,
        const std::filesystem::path& outputPath,
        PhysicalSkinAuthoringDetailBridgeCanonicalExportReport& report,
        std::string& error) {
    using namespace physical_skin_authoring_detail_bridge_csv_detail;
    using namespace physical_skin_authoring_detail_bridge_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(!files.allowPrecanonicalExportSeed||outputPath.empty()||
       outputPath==files.fineToBoundaryMap.path||
       outputPath==files.historicalFineToBoundaryMap.path||
       std::filesystem::exists(outputPath))
        return fail("canonical face export requires explicit seed mode and a fresh separate output path");
    if(loadedReport.physicalAdmission||loadedReport.detailRows!=input.detailReferenceWorldPoints.size()||
       loadedReport.historicalMapSHA256!=files.historicalFineToBoundaryMap.expectedSHA256||
       loadedReport.canonicalTetGenNodeSHA256!=files.canonicalTetGenNode.expectedSHA256)
        return fail("canonical face export source report does not match the pinned historical map and imported node");
    const std::array<const PhysicalSkinAuthoringDetailBridgeCSVPin*,9> pins{
        &files.detailNodes,&files.detailFaces,&files.detailFull86Weights,
        &files.detailFull86Owners,&files.fineToBoundaryMap,&files.coarseBoundaryFaces,
        &files.detailSourceAsset,&files.historicalFineToBoundaryMap,&files.canonicalTetGenNode};
    for(std::size_t i=0;i<pins.size();++i) {
        std::string actual;
        if(!digest(*pins[i],actual,error))return false;
        if(actual!=loadedReport.inputSHA256[i])
            return fail("canonical face export input changed after its pinned native load: "+pins[i]->path.string());
    }
    std::vector<std::uint32_t> representatives;
    std::vector<CanonicalNativeSelection> selections;
    std::size_t representativeAdjustments=0u;
    if(!runNativeFaceSelection(input,representatives,selections,
                               representativeAdjustments,error))return false;
    if(selections.size()!=input.detailReferenceWorldPoints.size())
        return fail("native canonical face selection did not cover every detail vertex");
    std::ifstream seed(files.fineToBoundaryMap.path,std::ios::binary);
    if(!seed)return fail("cannot reopen the pinned precanonical map for canonical export");
    std::string headerLine;
    if(!std::getline(seed,headerLine))return fail("precanonical map is empty");
    if(!headerLine.empty()&&headerLine.back()==static_cast<char>(13))headerLine.pop_back();
    Row parsed;
    if(!parseLine(headerLine,parsed)||parsed.size()!=54u)
        return fail("precanonical map header does not match the fixed 54-column seed contract");
    std::error_code filesystemError;
    if(!std::filesystem::is_directory(outputPath.parent_path(),filesystemError)||filesystemError)
        return fail("canonical face export output directory does not exist");
    auto temporaryPath=outputPath;temporaryPath += ".partial";
    if(std::filesystem::exists(temporaryPath))
        return fail("canonical face export refuses an existing partial output");
    std::ofstream output(temporaryPath,std::ios::binary|std::ios::out);
    if(!output)return fail("cannot create canonical face export partial output");
    const auto seedHeader=seedMapHeader();
    if(seedHeader.size()!=54u)return fail("internal canonical seed header has an unexpected field count");
    std::vector<std::string> outputHeader=seedHeader;
    outputHeader.insert(outputHeader.end(),nativeSelectionHeader().begin(),nativeSelectionHeader().end());
    const auto writeFields=[&](const std::vector<std::string>& fields) {
        for(std::size_t i=0;i<fields.size();++i){if(i)output.put(',');output<<csvQuote(fields[i]);}
        output.put('\n');
    };
    writeFields(outputHeader);
    std::size_t rows=0u,changedFaceRows=0u;double maximumDistance=0.0;
    std::string line;Row fields;
    while(std::getline(seed,line)) {
        if(!line.empty()&&line.back()==static_cast<char>(13))line.pop_back();
        if(rows>=selections.size()||line.size()>1048576u||
           !parseLine(line,fields)||fields.size()!=seedHeader.size())
            return fail("precanonical map has a malformed or excessive source row during canonical export");
        std::uint32_t fine=NM_INVALID_INDEX;
        if(!number(fields,0u,fine)||fine!=rows||fields[1]!="O"||
           fields[45]!=loadedReport.historicalMapSHA256||
           fields[46]!=loadedReport.canonicalTetGenNodeSHA256)
            return fail("precanonical row identity or historical/node pins changed during canonical export");
        const auto& selected=selections[rows];
        const auto appendNumber=[](std::vector<std::string>& row,const auto value) {
            row.push_back(std::to_string(value));
        };
        std::vector<std::string> suffix;
        suffix.reserve(21u);
        appendNumber(suffix,selected.faceRow);
        for(const auto node:selected.volumeNodes)appendNumber(suffix,node);
        appendNumber(suffix,selected.parentTet);appendNumber(suffix,selected.material);
        appendNumber(suffix,selected.donor);appendNumber(suffix,selected.region);
        appendNumber(suffix,selected.stableFace);appendNumber(suffix,selected.frameOwner);
        appendNumber(suffix,selected.representative);appendNumber(suffix,selected.walkStart);
        appendNumber(suffix,selected.walkHops);suffix.push_back(uintArrayJSON(selected.walk));
        for(const auto bary:selected.bary)suffix.push_back(realText(bary));
        for(const auto residual:selected.residual)suffix.push_back(realText(residual));
        suffix.push_back(realText(selected.distance));
        if(suffix.size()!=nativeSelectionHeader().size())
            return fail("internal canonical face witness has an unexpected field count");
        std::string outputLine=line;
        for(const auto& field:suffix){outputLine.push_back(',');outputLine+=csvQuote(field);}
        output<<outputLine<<'\n';
        if(!output)return fail("canonical face export write failed");
        if(selected.faceRow!=input.detailExpectedBoundaryFaceRows[rows])++changedFaceRows;
        maximumDistance=std::max(maximumDistance,selected.distance);
        ++rows;
    }
    if(!seed.eof()||rows!=selections.size())
        return fail("precanonical map row count differs from the complete typed detail domain");
    output.flush();output.close();
    if(!output.good())return fail("canonical face export did not flush successfully");
    if(std::filesystem::exists(outputPath))
        return fail("canonical face export target appeared during this run");
    std::filesystem::rename(temporaryPath,outputPath,filesystemError);
    if(filesystemError) {
        std::filesystem::remove(temporaryPath);
        return fail("canonical face export could not atomically publish its fresh output path");
    }
    std::string outputSHA;
    if(!physical_skin_child_mass_detail::sha256File(outputPath,outputSHA,error))return false;
    for(std::size_t i=0;i<pins.size();++i) {
        std::string actual;
        if(!physical_skin_child_mass_detail::sha256File(pins[i]->path,actual,error)||
           actual!=loadedReport.inputSHA256[i])
            return fail("canonical face export input changed before its prepared output was pinned");
    }
    PhysicalSkinAuthoringDetailBridgeCanonicalExportReport measured;
    measured.detailRows=rows;measured.changedFaceRows=changedFaceRows;
    measured.representativeAdjustments=representativeAdjustments;
    measured.maximumProjectionDistanceM=maximumDistance;
    measured.historicalMapSHA256=loadedReport.historicalMapSHA256;
    measured.canonicalTetGenNodeSHA256=loadedReport.canonicalTetGenNodeSHA256;
    measured.seedMapSHA256=loadedReport.inputSHA256[4];
    measured.outputSHA256=outputSHA;measured.physicalAdmission=false;
    report=std::move(measured);error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue

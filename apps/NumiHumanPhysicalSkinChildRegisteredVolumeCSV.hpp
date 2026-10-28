#pragma once

// Reads the existing combined TetGen node-map CSV after one-time gzip
// decompression. This is asset import only; the registered-volume, mass,
// and physical-volume owners still perform their independent admission.
#include "NumiHumanPhysicalSkinChildRegisteredVolumeImport.hpp"
#include <istream>
#include <map>

namespace metalrobo::human::coarse_tissue {

struct PhysicalSkinCombinedNodeMapCSVReport {
    std::size_t fileRows=0,volumeNodes=0,sharedInterfacePairs=0;
};

inline std::vector<std::string> physicalSkinCombinedNodeMapCSVHeader() {
    return {"joint_node_id","source_partition","source_tetgen_node_id",
        "registered_I_source_vertex_row","merge_peer_partition",
        "merge_peer_tetgen_node_id","registered_surface_layer",
        "registered_surface_vertex_row",
        "role","current_x_m","current_y_m","current_z_m",
        "donor_body","donor_binding","donor_local_x_m","donor_local_y_m","donor_local_z_m",
        "source_domain","source_mapping_kind","source_vertex_row","source_face_row",
        "source_bary0","source_bary1","source_bary2",
        "source_residual_x_m","source_residual_y_m","source_residual_z_m",
        "frame_owner_method","parent_registration_kind","parent_tet_row",
        "parent_bary0","parent_bary1","parent_bary2","parent_bary3",
        "parent_current_residual_x_m","parent_current_residual_y_m","parent_current_residual_z_m",
        "parent_registration_distance_m","parent_registration_tie_count",
        "incident_material_mask","geometric_donor_body","anatomical_region_id"};
}

[[nodiscard]] inline bool readPhysicalSkinCombinedNodeMapCSV(
        std::istream& input,const std::size_t volumeNodeCount,
        const std::size_t sourceBodyCount,
        std::vector<PhysicalSkinRegisteredVolumeNodeRow>& output,
        PhysicalSkinCombinedNodeMapCSVReport& report,std::string& error) {
    using physical_skin_child_mass_detail::parseNumber;
    using physical_skin_child_mass_detail::splitCSV;
    using physical_skin_registered_volume_import_detail::finitePoint;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const auto optionalIndex=[](const std::string& value,std::uint32_t& index) {
        if(value=="-1"){index=NM_INVALID_INDEX;return true;}
        return physical_skin_child_mass_detail::parseNumber(value,index);
    };
    if(volumeNodeCount<4u||volumeNodeCount>300000u||sourceBodyCount==0u||
       sourceBodyCount>std::numeric_limits<std::uint32_t>::max())
        return fail("combined-node-map CSV has an invalid bounded node/body domain");
    std::string line;
    if(!std::getline(input,line)||splitCSV(line)!=physicalSkinCombinedNodeMapCSVHeader())
        return fail("combined-node-map CSV header differs from the registered volume contract");
    struct SourceRow {
        std::string partition,peerPartition;
        std::uint32_t sourceID=0,peerID=0,interfaceRow=NM_INVALID_INDEX;
        std::vector<std::string> typedFields;
        unsigned copies=0;
    };
    std::vector<SourceRow> sources(volumeNodeCount);
    std::vector<PhysicalSkinRegisteredVolumeNodeRow> next(volumeNodeCount);
    // A TetGen source node may occur once. The sole merge is an exact
    // reciprocal skin/fat I node, never a coordinate-nearest weld.
    std::map<std::pair<std::string,std::uint32_t>,std::uint32_t> sourceOwners;
    PhysicalSkinCombinedNodeMapCSVReport measured;
    while(std::getline(input,line)) {
        if(line.size()>16384u||measured.fileRows>=2u*volumeNodeCount)
            return fail("combined-node-map CSV exceeds bounded line or row capacity");
        const auto fields=splitCSV(line);
        if(fields.size()!=42u)
            return fail("combined-node-map CSV row has an invalid width");
        std::uint32_t id=0,sourceID=0,interfaceRow=NM_INVALID_INDEX,peerID=0,surfaceRow=NM_INVALID_INDEX;
        if(!parseNumber(fields[0],id)||id==0u||id>volumeNodeCount||
           (fields[1]!="skin"&&fields[1]!="fat")||
           !parseNumber(fields[2],sourceID)||sourceID==0u||
           !optionalIndex(fields[3],interfaceRow)||
           !optionalIndex(fields[7],surfaceRow))
            return fail("combined-node-map CSV has an invalid joint/source identity");
        const auto node=id-1u;
        if(!sourceOwners.emplace(std::pair{fields[1],sourceID},node).second)
            return fail("combined-node-map CSV repeats a source TetGen node identity");
        PhysicalSkinRegisteredVolumeNodeRow row;
        row.volumeNodeRow=node;row.surfaceVertexRow=surfaceRow;
        if(fields[8]=="O")row.role=PhysicalSkinRegisteredVolumeNodeRole::outerO;
        else if(fields[8]=="I")row.role=PhysicalSkinRegisteredVolumeNodeRole::innerI;
        else if(fields[8]=="D")row.role=PhysicalSkinRegisteredVolumeNodeRole::deepD;
        else if(fields[8]=="interior")row.role=PhysicalSkinRegisteredVolumeNodeRole::interior;
        else return fail("combined-node-map CSV declares an unsupported node role");
        const bool shared=row.role==PhysicalSkinRegisteredVolumeNodeRole::innerI;
        if(shared) {
            if(fields[6]!="I"||surfaceRow==NM_INVALID_INDEX||interfaceRow!=surfaceRow||
               fields[4]!=(fields[1]=="skin"?"fat":"skin")||
               !parseNumber(fields[5],peerID)||peerID==0u)
                return fail("combined-node-map I row lacks exact reciprocal source identities");
        } else {
            if(interfaceRow!=NM_INVALID_INDEX||!fields[4].empty()||!fields[5].empty())
                return fail("combined-node-map non-I row claims an interface merge");
            const bool interior=row.role==PhysicalSkinRegisteredVolumeNodeRole::interior;
            if((interior&&(!fields[6].empty()||surfaceRow!=NM_INVALID_INDEX))||
               (!interior&&(fields[6]!=fields[8]||surfaceRow==NM_INVALID_INDEX))||
               (row.role==PhysicalSkinRegisteredVolumeNodeRole::outerO&&fields[1]!="skin")||
               (row.role==PhysicalSkinRegisteredVolumeNodeRole::deepD&&fields[1]!="fat"))
                return fail("combined-node-map surface role differs from its source partition/map");
        }
        for(unsigned axis=0;axis<3u;++axis) {
            if(!parseNumber(fields[9u+axis],row.currentWorldPoint[axis])||
               !parseNumber(fields[14u+axis],row.donorLocalPoint[axis])||
               !parseNumber(fields[21u+axis],row.sourceBarycentric[axis])||
               !parseNumber(fields[24u+axis],row.sourceResidualWorldM[axis])||
               !parseNumber(fields[34u+axis],row.parentCurrentResidualWorldM[axis]))
                return fail("combined-node-map point/source/parent field is not a number");
        }
        for(unsigned corner=0;corner<4u;++corner)
            if(!parseNumber(fields[30u+corner],row.parentBarycentric[corner])||
               !std::isfinite(row.parentBarycentric[corner]))
                return fail("combined-node-map parent barycentric is not finite");
        if(!finitePoint(row.currentWorldPoint)||!finitePoint(row.donorLocalPoint)||
           !finitePoint(row.sourceBarycentric)||!finitePoint(row.sourceResidualWorldM)||
           !finitePoint(row.parentCurrentResidualWorldM)||
           !parseNumber(fields[12],row.donorBody)||row.donorBody>=sourceBodyCount||
           !optionalIndex(fields[13],row.donorBinding)||
           !optionalIndex(fields[19],row.sourceVertex)||!optionalIndex(fields[20],row.sourceFace)||
           !parseNumber(fields[29],row.parentTet)||
           !parseNumber(fields[37],row.parentRegistrationDistanceM)||
           !std::isfinite(row.parentRegistrationDistanceM)||row.parentRegistrationDistanceM<0.0||
           !parseNumber(fields[38],row.parentTieCount)||
           !parseNumber(fields[39],row.incidentMaterialMask)||
           !parseNumber(fields[40],row.geometricDonorBody)||row.geometricDonorBody>=sourceBodyCount||
           !parseNumber(fields[41],row.anatomicalRegion))
            return fail("combined-node-map contains invalid finite values or typed identities");
        for(double value:row.currentWorldPoint)
            if(double(float(value))!=value)
                return fail("combined-node-map current-world target is not exact Float32");
        row.sourceDomain=fields[17];row.sourceMappingKind=fields[18];
        row.frameMethod=fields[27];row.parentRegistrationKind=fields[28];
        const bool direct=row.sourceVertex!=NM_INVALID_INDEX,embedded=row.sourceFace!=NM_INVALID_INDEX;
        if((row.sourceMappingKind=="vertex"&&(!direct||embedded))||
           (row.sourceMappingKind=="face"&&(direct||!embedded))||
           (row.sourceMappingKind=="none"&&(direct||embedded))||
           (row.sourceMappingKind!="vertex"&&row.sourceMappingKind!="face"&&row.sourceMappingKind!="none"))
            return fail("combined-node-map source mapping kind disagrees with its typed source identity");
        std::string sourceError;
        if(!physical_skin_registered_volume_import_detail::validSourceIdentity(row,sourceError))
            return fail(sourceError);
        const std::vector<std::string> typedFields(fields.begin()+6,fields.end());
        auto& source=sources[node];
        if(source.copies==0u) {
            source={fields[1],fields[4],sourceID,peerID,interfaceRow,typedFields,1u};
            next[node]=std::move(row);
        } else {
            if(source.copies!=1u||!shared||source.partition==fields[1]||
               source.peerPartition!=fields[1]||fields[4]!=source.partition||
               source.peerID!=sourceID||peerID!=source.sourceID||
               source.interfaceRow!=interfaceRow||source.typedFields!=typedFields)
                return fail("combined-node-map duplicate joint ID is not an identical reciprocal I node");
            source.copies=2u;++measured.sharedInterfacePairs;
        }
        ++measured.fileRows;
    }
    if(!input.eof())return fail("combined-node-map CSV input failed before EOF");
    for(std::size_t node=0;node<volumeNodeCount;++node) {
        if(sources[node].copies==0u)
            return fail("combined-node-map CSV omits a physical volume node");
        if(next[node].role==PhysicalSkinRegisteredVolumeNodeRole::innerI&&sources[node].copies!=2u)
            return fail("combined-node-map shared I node lacks its reciprocal TetGen source row");
    }
    measured.volumeNodes=volumeNodeCount;output=std::move(next);report=measured;error.clear();return true;
}
} // namespace metalrobo::human::coarse_tissue

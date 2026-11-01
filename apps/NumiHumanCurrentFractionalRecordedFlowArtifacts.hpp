#pragma once

#include "NumiHumanRecordedCarrierAllocationImport.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <string_view>
#include <span>
#include <vector>

namespace metalrobo::human::coarse_tissue {

struct CurrentRecordedRoleFlowArtifacts {
    std::array<std::array<std::filesystem::path,6>,2> csvPaths{};
    std::array<std::array<std::string,6>,2> csvSHA256{};
    std::array<std::filesystem::path,2> receiptPaths{};
    std::array<std::string,2> receiptSHA256{};
};

namespace current_recorded_role_flow_artifact_detail {
inline constexpr std::uint32_t kInvalid=std::numeric_limits<std::uint32_t>::max();
inline constexpr std::array<std::string_view,6> kNames{
    "child-mass.csv","transport-flow.csv","source-carrier-flow.csv",
    "child-source-donor.csv","source-parent.csv","source-parent-donor.csv"};

inline bool validFileSHA(const std::filesystem::path& path,std::string& sha,std::string& error) {
    return physical_skin_child_mass_detail::sha256File(path,sha,error)&&
           physical_skin_child_mass_detail::validSHA256(sha);
}
inline std::string roleCarrierArraySHA(
        const unsigned role,const std::span<const std::uint32_t> roleParents,
        const SourceCellCarrierQuadratureInput& carrier,const std::string_view rolePacketSHA) {
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("numi.human.current-recorded-role-source-carrier-arrays.v1");
    hash.u32(role==0u?15u:16u);hash.string(rolePacketSHA);
    hash.string(carrier.sourceTetTableSHA256);hash.string(carrier.sourceNodeTableSHA256);
    for(const auto parent:roleParents){
        hash.u32(parent);hash.u32(carrier.globalSourceTetRows[parent]);
        const auto begin=carrier.parentOffsets[parent],end=carrier.parentOffsets[parent+1u];
        hash.u32(end-begin);
        for(auto row=begin;row<end;++row){
            hash.real(carrier.carrierWeights[row]);
            for(const auto value:carrier.carrierBarycentric[row])hash.real(value);
            hash.u32(carrier.witnessChildRows[row]);
            hash.real(carrier.witnessChildCentroidDistancesM[row]);
        }
    }
    return hash.finish();
}
inline bool writeRoleFiles(
        const std::filesystem::path& directory,const unsigned role,
        const PhysicalSkinChildMassMappingBundle& bundle,
        const ConservativeSpatialMassTransportResult& result,
        const SourceCellCarrierTransportTrace& trace,
        const SourceCellCarrierQuadratureInput& carrier,
        const std::array<std::vector<std::uint32_t>,2>& roleParents,
        const std::array<std::vector<std::uint32_t>,2>& roleChildren,
        const std::vector<std::uint32_t>& parentLocal,
        const std::vector<std::uint32_t>& virtualLocal,
        const std::vector<std::uint32_t>& carrierLocal,
        const std::vector<std::uint32_t>& sourceDonorOffsets,
        const std::vector<std::uint32_t>& sourceDonorIDs,
        const std::vector<double>& sourceDonorMass,
        const std::array<std::string,2>& carrierPacketSHA256,
        const std::array<std::string,10>& expectedInputSHA256,
        CurrentRecordedRoleFlowArtifacts& output,std::string& error) {
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const std::uint32_t material=role==0u?15u:16u;
    const std::string prefix=role==0u?"skin-":"fat-";
    std::array<std::filesystem::path,6> finalPaths{},tempPaths{};
    for(std::size_t file=0;file<kNames.size();++file){
        finalPaths[file]=directory/(prefix+std::string(kNames[file]));
        tempPaths[file]=finalPaths[file].string()+".tmp";
        std::error_code ec;std::filesystem::remove(tempPaths[file],ec);
        std::filesystem::remove(finalPaths[file],ec);
    }
    const auto sourceCentroid=[&](const std::uint32_t parent){
        V3 centroid{};const auto tet=bundle.parent.tetrahedra[parent];
        for(const auto node:tet)for(unsigned axis=0;axis<3u;++axis)
            centroid[axis]+=bundle.parent.referenceWorldPoints[node][axis]*0.25;
        return centroid;
    };
    std::size_t roleEdges=0u,roleCarrierEdges=0u,roleDonorRows=0u;
    try{
        {
            std::ofstream f(tempPaths[0],std::ios::binary|std::ios::trunc);
            if(!f)return fail("cannot create role child-mass CSV");
            f<<"child_row,reference_volume_m3,mass_kg,density_kg_m3\n"<<std::setprecision(17);
            for(std::size_t local=0;local<roleChildren[role].size();++local){
                const auto child=roleChildren[role][local];
                const auto tet=bundle.childTetrahedra[child];
                const double volume=parentReferenceTetVolumeM3(tet,bundle.referenceWorldPoints);
                if(!(std::isfinite(volume)&&volume>0.0&&result.childMassKg[child]>0.0&&
                     result.childImpliedDensityKgM3[child]>0.0))
                    return fail("current role child mass/volume/density is not positive");
                f<<local<<','<<volume<<','<<result.childMassKg[child]<<','
                 <<result.childImpliedDensityKgM3[child]<<'\n';
            }
            f.flush();if(!f)return fail("failed writing role child-mass CSV");
        }
        {
            std::ofstream f(tempPaths[1],std::ios::binary|std::ios::trunc);
            if(!f)return fail("cannot create role aggregate-flow CSV");
            f<<"child_row,parent_row,mass_kg,source_measure_m3,distance_m\n"<<std::setprecision(17);
            for(std::size_t localChild=0;localChild<roleChildren[role].size();++localChild){
                const auto child=roleChildren[role][localChild];
                for(std::uint32_t edge=result.childParentOffsets[child];
                    edge<result.childParentOffsets[child+1u];++edge){
                    const auto parent=result.childParentRows[edge];
                    if(parent>=parentLocal.size()||parentLocal[parent]==kInvalid)
                        return fail("role aggregate flow crosses the material-specific parent row domain");
                    f<<localChild<<','<<parentLocal[parent]<<','
                     <<result.childParentTransportMassKg[edge]<<','
                     <<result.childParentTransportMeasureM3[edge]<<','
                     <<result.childParentTransportDistanceM[edge]<<'\n';
                    ++roleEdges;
                }
            }
            f.flush();if(!f)return fail("failed writing role aggregate-flow CSV");
        }
        {
            std::ofstream f(tempPaths[2],std::ios::binary|std::ios::trunc);
            if(!f)return fail("cannot create role expanded-carrier-flow CSV");
            f<<"child_row,parent_row,expanded_parent_row,carrier_row,mass_kg,source_measure_m3,distance_m\n"
             <<std::setprecision(17);
            for(std::size_t localChild=0;localChild<roleChildren[role].size();++localChild){
                const auto child=roleChildren[role][localChild];
                for(std::uint32_t edge=result.childParentOffsets[child];
                    edge<result.childParentOffsets[child+1u];++edge){
                    const auto parent=result.childParentRows[edge];
                    if(parent>=parentLocal.size()||parentLocal[parent]==kInvalid||
                       edge+1u>=trace.childParentCarrierOffsets.size())
                        return fail("role carrier flow lacks an exact role-local aggregate edge");
                    const auto begin=trace.childParentCarrierOffsets[edge];
                    const auto end=trace.childParentCarrierOffsets[edge+1u];
                    if(begin>=end||end>trace.childParentCarrierRows.size())
                        return fail("current carrier trace omits a role aggregate edge");
                    for(std::uint32_t row=begin;row<end;++row){
                        const auto virtualParent=trace.childParentCarrierRows[row];
                        if(virtualParent>=virtualLocal.size()||virtualLocal[virtualParent]==kInvalid||
                           virtualParent>=trace.expandedParentToSourceParentRows.size()||
                           trace.expandedParentToSourceParentRows[virtualParent]!=parent||
                           virtualParent>=trace.expandedParentToCarrierRows.size())
                            return fail("carrier trace row has no matching material-local virtual parent");
                        const auto carrierRow=trace.expandedParentToCarrierRows[virtualParent];
                        std::uint32_t roleCarrier=kInvalid;
                        if(carrierRow!=kInvalid){
                            if(carrierRow>=carrierLocal.size()||carrierLocal[carrierRow]==kInvalid)
                                return fail("carrier trace references a carrier outside the role CSR domain");
                            roleCarrier=carrierLocal[carrierRow];
                        }
                        f<<localChild<<','<<parentLocal[parent]<<','<<virtualLocal[virtualParent]<<','
                         <<roleCarrier<<','<<trace.childParentCarrierMassKg[row]<<','
                         <<trace.childParentCarrierMeasureM3[row]<<','
                         <<trace.childParentCarrierDistanceM[row]<<'\n';
                        ++roleCarrierEdges;
                    }
                }
            }
            f.flush();if(!f)return fail("failed writing role expanded-carrier-flow CSV");
        }
        {
            std::ofstream f(tempPaths[3],std::ios::binary|std::ios::trunc);
            if(!f)return fail("cannot create role child-source-donor CSV");
            f<<"child_row,source_donor_body,mass_kg\n"<<std::setprecision(17);
            for(std::size_t local=0;local<roleChildren[role].size();++local){
                const auto child=roleChildren[role][local];
                if(child+1u>=result.childSourceDonorOffsets.size())
                    return fail("result is missing the role child donor CSR");
                const auto begin=result.childSourceDonorOffsets[child];
                const auto end=result.childSourceDonorOffsets[child+1u];
                if(begin>=end||end>result.childSourceDonorBodyIndices.size()||
                   end>result.childSourceDonorMassKg.size())
                    return fail("role child has an empty or malformed donor-mass CSR row");
                for(std::uint32_t row=begin;row<end;++row){
                    f<<local<<','<<result.childSourceDonorBodyIndices[row]<<','
                     <<result.childSourceDonorMassKg[row]<<'\n';
                    ++roleDonorRows;
                }
            }
            f.flush();if(!f)return fail("failed writing role child-source-donor CSV");
        }
        {
            std::ofstream f(tempPaths[4],std::ios::binary|std::ios::trunc);
            if(!f)return fail("cannot create role source-parent CSV");
            f<<"parent_row,global_source330_row,material,geometric_donor,region,stable_source_face,"
              "mass_kg,reference_volume_m3,source_centroid_x_m,source_centroid_y_m,source_centroid_z_m\n"
             <<std::setprecision(17);
            for(std::size_t local=0;local<roleParents[role].size();++local){
                const auto parent=roleParents[role][local];
                const auto centroid=sourceCentroid(parent);
                if(parent>=carrier.globalSourceTetRows.size()||
                   parent>=bundle.parent.tetrahedronMassKg.size()||
                   parent>=bundle.parent.referenceTetVolumeM3.size())
                    return fail("role source-parent row is outside the authenticated current source domain");
                const double volume=parentReferenceTetVolumeM3(
                    bundle.parent.tetrahedra[parent],bundle.parent.referenceWorldPoints);
                f<<local<<','<<carrier.globalSourceTetRows[parent]<<','<<material<<','
                 <<bundle.parent.donorBodyIndices[parent]<<','
                 <<bundle.parent.anatomicalRegionIndices[parent]<<','
                 <<bundle.parent.stableSourceFaceIdentities[parent]<<','
                 <<bundle.parent.tetrahedronMassKg[parent]<<','<<volume<<','
                 <<centroid[0]<<','<<centroid[1]<<','<<centroid[2]<<'\n';
            }
            f.flush();if(!f)return fail("failed writing role source-parent CSV");
        }
        {
            std::ofstream f(tempPaths[5],std::ios::binary|std::ios::trunc);
            if(!f)return fail("cannot create role source-parent-donor CSV");
            f<<"parent_row,source_donor_body,mass_kg\n"<<std::setprecision(17);
            for(std::size_t local=0;local<roleParents[role].size();++local){
                const auto parent=roleParents[role][local];
                const auto begin=sourceDonorOffsets[parent],end=sourceDonorOffsets[parent+1u];
                if(begin>=end||end>sourceDonorIDs.size()||end>sourceDonorMass.size())
                    return fail("role parent has an empty or malformed native source-donor CSR row");
                for(std::uint32_t row=begin;row<end;++row)
                    f<<local<<','<<sourceDonorIDs[row]<<','<<sourceDonorMass[row]<<'\n';
            }
            f.flush();if(!f)return fail("failed writing role source-parent-donor CSV");
        }
        for(std::size_t file=0;file<finalPaths.size();++file){
            std::error_code ec;std::filesystem::rename(tempPaths[file],finalPaths[file],ec);
            if(ec)return fail("cannot atomically publish role-local flow CSV: "+ec.message());
            output.csvPaths[role][file]=finalPaths[file];
            if(!validFileSHA(finalPaths[file],output.csvSHA256[role][file],error))
                return fail("cannot verify role-local flow CSV: "+error);
        }
        if(roleEdges==0u||roleCarrierEdges==0u||roleDonorRows==0u)
            return fail("role-local recorded flow output is unexpectedly empty");
        output.receiptPaths[role]=directory/(prefix+"recorded-flow-receipt.json");
        const auto receiptTemp=std::filesystem::path(output.receiptPaths[role].string()+".tmp");
        std::error_code ec;std::filesystem::remove(receiptTemp,ec);
        std::ofstream receipt(receiptTemp,std::ios::binary|std::ios::trunc);
        if(!receipt)return fail("cannot create role recorded-flow receipt");
        const std::string roleUpper=role==0u?"SKIN":"FAT";
        const auto roleArraySHA=roleCarrierArraySHA(role,roleParents[role],carrier,
                                                     carrierPacketSHA256[role]);
        receipt<<"{\n"
          <<"  \"status\":\"PASS_SOURCE_ONLY_"<<roleUpper<<"_SOURCE_CARRIER_FEASIBLE_ALLOCATION\",\n"
          <<"  \"recorded_flow_feasibility\":\"PASS_SOLVER_FREE\",\n"
          <<"  \"deterministic_optimizer_replay\":\"NOT_RUN\",\n"
          <<"  \"complete_allocation\":true,\n"
          <<"  \"physical_admission\":false,\n"
          <<"  \"native_steps\":0,\n"
          <<"  \"solver_metrics_scope\":\"paired\",\n"
          <<"  \"paired_solver_run_sha256\":\""<<result.receiptSHA256<<"\",\n"
          <<"  \"source_carrier_packet_sha256\":\""<<carrierPacketSHA256[role]<<"\",\n"
          <<"  \"source_carrier_array_sha256\":\""<<roleArraySHA<<"\",\n"
          <<"  \"source_carrier_flow_trace_sha256\":\""<<output.csvSHA256[role][2]<<"\",\n"
          <<"  \"expected_input_sha256\":[";
        for(std::size_t pin=0;pin<expectedInputSHA256.size();++pin){
            receipt<<"\""<<expectedInputSHA256[pin]<<"\""<<(pin+1u==expectedInputSHA256.size()?"":",");
        }
        receipt<<"],\n"

          <<"  \"transport_receipt_sha256\":\""<<result.receiptSHA256<<"\",\n"
          <<"  \"parent_"<<(role==0u?"skin":"fat")<<"_tets\":"<<roleParents[role].size()<<",\n"
          <<"  \"child_"<<(role==0u?"skin":"fat")<<"_tets\":"<<roleChildren[role].size()<<",\n"
          <<"  \"transport_edges\":"<<roleEdges<<",\n"
          <<"  \"candidate_edges\":"<<result.semanticCandidateEdgeCount<<",\n"
          <<"  \"augmentations\":"<<result.augmentationCount<<",\n"
          <<"  \"flow_csv\":{\n";
        for(std::size_t file=0;file<finalPaths.size();++file){
            std::error_code sizeError;const auto size=std::filesystem::file_size(finalPaths[file],sizeError);
            if(sizeError)return fail("cannot measure role-local flow CSV for its receipt");
            receipt<<"    \""<<finalPaths[file].filename().string()<<"\":{\"sha256\":\""
                   <<output.csvSHA256[role][file]<<"\",\"size_bytes\":"<<size<<"}"
                   <<(file+1u==finalPaths.size()?"\n":",\n");
        }
        receipt<<"  }\n}\n";receipt.flush();
        if(!receipt)return fail("failed writing role recorded-flow receipt");
        receipt.close();std::filesystem::rename(receiptTemp,output.receiptPaths[role],ec);
        if(ec)return fail("cannot publish role recorded-flow receipt: "+ec.message());
        if(!validFileSHA(output.receiptPaths[role],output.receiptSHA256[role],error))
            return fail("cannot verify role recorded-flow receipt: "+error);
        error.clear();return true;
    }catch(const std::exception& e){
        return fail(std::string("role recorded-flow serialization exception: ")+e.what());
    }
}
} // namespace current_recorded_role_flow_artifact_detail

[[nodiscard]] inline bool writeCurrentFractionalRecordedRoleFlowArtifacts(
        const std::filesystem::path& directory,
        const PhysicalSkinChildMassMappingBundle& bundle,
        const ConservativeSpatialMassTransportResult& result,
        const SourceCellCarrierTransportTrace& trace,
        const SourceCellCarrierQuadratureInput& carrier,
        const std::array<std::string,2>& supportSHA256,
        const std::array<std::string,2>& carrierPacketSHA256,
        const std::array<std::array<std::string,10>,2>& expectedInputSHA256,
        CurrentRecordedRoleFlowArtifacts& output,std::string& error) {
    using namespace current_recorded_role_flow_artifact_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const auto np=bundle.parent.tetrahedra.size(),nc=bundle.childTetrahedra.size();
    if(np==0u||nc==0u||bundle.parent.materialIndices.size()!=np||
       bundle.parent.tetrahedronMassKg.size()!=np||bundle.parent.donorBodyIndices.size()!=np||
       bundle.parent.anatomicalRegionIndices.size()!=np||bundle.parent.stableSourceFaceIdentities.size()!=np||
       bundle.parent.referenceWorldPoints.empty()||bundle.childMaterialIndices.size()!=nc||
       result.childParentOffsets.size()!=nc+1u||result.childParentOffsets.empty()||
       result.childParentOffsets.front()!=0u||result.childParentOffsets.back()!=result.childParentRows.size()||
       result.childParentRows.size()!=result.childParentTransportMassKg.size()||
       result.childParentRows.size()!=result.childParentTransportMeasureM3.size()||
       result.childParentRows.size()!=result.childParentTransportDistanceM.size()||
       result.childParentRows.size()!=result.childParentVolumeFractions.size()||
       result.childSourceDonorOffsets.size()!=nc+1u||
       result.childSourceDonorOffsets.empty()||result.childSourceDonorOffsets.front()!=0u||
       result.childSourceDonorOffsets.back()!=result.childSourceDonorBodyIndices.size()||
       result.childSourceDonorBodyIndices.size()!=result.childSourceDonorMassKg.size()||
       carrier.parentOffsets.size()!=np+1u||carrier.globalSourceTetRows.size()!=np||
       carrier.parentOffsets.empty()||carrier.parentOffsets.front()!=0u||
       carrier.parentOffsets.back()!=carrier.carrierWeights.size()||
       carrier.carrierWeights.size()!=carrier.carrierBarycentric.size()||
       carrier.carrierWeights.size()!=carrier.witnessChildRows.size()||
       carrier.carrierWeights.size()!=carrier.witnessChildCentroidDistancesM.size()||
       trace.childParentCarrierOffsets.size()!=result.childParentRows.size()+1u||
       trace.childParentCarrierOffsets.empty()||trace.childParentCarrierOffsets.front()!=0u||
       trace.childParentCarrierOffsets.back()!=trace.childParentCarrierRows.size()||
       trace.childParentCarrierRows.size()!=trace.childParentCarrierMassKg.size()||
       trace.childParentCarrierRows.size()!=trace.childParentCarrierMeasureM3.size()||
       trace.childParentCarrierRows.size()!=trace.childParentCarrierDistanceM.size()||
       trace.expandedParentToSourceParentRows.size()!=trace.expandedParentToCarrierRows.size())
        return fail("paired current result, parent donor, or carrier trace arrays are incomplete");
    for(unsigned role=0;role<2u;++role)
        if(!physical_skin_child_mass_detail::validSHA256(supportSHA256[role])||
           !physical_skin_child_mass_detail::validSHA256(carrierPacketSHA256[role]))
            return fail("paired role support/carrier output pins are invalid");
    for(const auto& rolePins:expectedInputSHA256)
        for(const auto& sha:rolePins)
            if(!physical_skin_child_mass_detail::validSHA256(sha))
                return fail("recorded role input file pin is invalid");
    if(result.augmentationCount==0u||result.semanticCandidateEdgeCount<result.transportEdgeCount||
       result.transportEdgeCount==0u||result.semanticCandidateEdgeCount>12000000u||
       result.augmentationCount>4000000u||!physical_skin_child_mass_detail::validSHA256(result.receiptSHA256))
        return fail("paired solver result counters or transport receipt are outside the imported-flow limits");
    if(!std::filesystem::is_directory(directory))
        return fail("current fractional output directory must exist before role artifacts are added");

    std::array<std::vector<std::uint32_t>,2> parents,children;
    std::vector<std::uint32_t> parentLocal(np,kInvalid);
    for(std::uint32_t row=0;row<np;++row){
        const auto material=bundle.parent.materialIndices[row];
        const unsigned role=material==15u?0u:material==16u?1u:2u;
        if(role>1u)return fail("current recorded-flow parent material is outside Skin/Fat");
        parentLocal[row]=static_cast<std::uint32_t>(parents[role].size());parents[role].push_back(row);
    }
    for(std::uint32_t row=0;row<nc;++row){
        const auto material=bundle.childMaterialIndices[row];
        const unsigned role=material==15u?0u:material==16u?1u:2u;
        if(role>1u)return fail("current recorded-flow child material is outside Skin/Fat");
        children[role].push_back(row);
    }
    if(parents[0].empty()||parents[1].empty()||children[0].empty()||children[1].empty())
        return fail("both current Skin and Fat parent/child domains are required");
    std::vector<std::uint32_t> carrierLocal(carrier.carrierWeights.size(),kInvalid);
    std::array<std::uint32_t,2> nextCarrier{};
    for(unsigned role=0;role<2u;++role){
        for(const auto parent:parents[role]){
            const auto begin=carrier.parentOffsets[parent],end=carrier.parentOffsets[parent+1u];
            for(std::uint32_t row=begin;row<end;++row){
                if(row>=carrierLocal.size()||nextCarrier[role]>=carrier.carrierWeights.size())
                    return fail("role carrier-local source CSR domain is incomplete");
                carrierLocal[row]=nextCarrier[role]++;
            }
        }
    }
    // The public trace uses global virtual rows. Build the exact global-to-role
    // virtual map in the same parent/carrier CSR order used by expandParentRows.
    std::vector<std::uint32_t> globalVirtualLocal(trace.expandedParentToSourceParentRows.size(),kInvalid);
    std::array<std::uint32_t,2> nextVirtual{};
    for(std::uint32_t virtualParent=0;virtualParent<trace.expandedParentToSourceParentRows.size();++virtualParent){
        const auto parent=trace.expandedParentToSourceParentRows[virtualParent];
        if(parent>=np)return fail("expanded trace references an absent current source parent");
        const auto material=bundle.parent.materialIndices[parent];
        const unsigned role=material==15u?0u:material==16u?1u:2u;
        if(role>1u)return fail("expanded trace parent has an unsupported material");
        globalVirtualLocal[virtualParent]=nextVirtual[role]++;
        const auto carrierRow=trace.expandedParentToCarrierRows[virtualParent];
        const auto begin=carrier.parentOffsets[parent],end=carrier.parentOffsets[parent+1u];
        if(begin==end){
            if(carrierRow!=kInvalid)return fail("centroid virtual parent unexpectedly names a carrier row");
        }else if(carrierRow<begin||carrierRow>=end||carrierLocal[carrierRow]==kInvalid){
            return fail("expanded trace carrier row is absent from its exact role CSR map");
        }
    }
    std::vector<ConservativeSpatialMassTransportParentNodeSourceMass> sourceRows;
    sourceRows.reserve(bundle.parent.sourceMassContributions.size());
    for(const auto& row:bundle.parent.sourceMassContributions)
        sourceRows.push_back({.parentNodeIndex=row.parentNodeIndex,.materialIndex=row.materialIndex,
            .sourceDonorBodyIndex=row.donorBodyIndex,.massKg=row.massKg});
    std::vector<std::uint32_t> sourceDonorOffsets,sourceDonorIDs;
    std::vector<double> sourceDonorMass;
    std::string splitError;
    if(!assembleConservativeSpatialMassTransportParentSourceDonorSplits(
            bundle.parent.tetrahedronMassKg,bundle.parent.materialIndices,bundle.parent.tetrahedra,
            sourceRows,1.0e-10,5.0e-4,sourceDonorOffsets,sourceDonorIDs,sourceDonorMass,splitError))
        return fail("current parent source-donor ledger could not be rebuilt from its pinned packet: "+splitError);
    for(unsigned role=0;role<2u;++role)
        if(!writeRoleFiles(directory,role,bundle,result,trace,carrier,parents,children,
                parentLocal,globalVirtualLocal,carrierLocal,sourceDonorOffsets,
                sourceDonorIDs,sourceDonorMass,carrierPacketSHA256,
                expectedInputSHA256[role],output,error))
            return false;
    error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue

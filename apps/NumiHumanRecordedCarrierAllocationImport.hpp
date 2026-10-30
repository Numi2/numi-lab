#pragma once

#include "NumiHumanSourceCarrierQuadratureImport.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace metalrobo::human::coarse_tissue {

struct PhysicalSkinRecordedCarrierAllocationCSVAssets {
    // One role-local source-only allocation, using the six established CSV files.
    std::uint32_t materialIndex=NM_INVALID_INDEX;
    std::array<std::filesystem::path,6> csvPaths{};
    std::array<std::string,6> csvSHA256{};
    std::filesystem::path receiptPath;
    std::string receiptSHA256;
    std::vector<std::string> expectedInputSHA256;
};
struct PhysicalSkinRecordedCarrierAllocationCSVImportReport {
    std::array<std::string,6> csvSHA256{};
    std::string receiptSHA256,sourceCarrierPacketSHA256,sourceCarrierArraySHA256,
        sourceCarrierFlowTraceSHA256,transportReceiptSHA256;
    std::size_t sourceParentRows=0,childRows=0,aggregateFlowRows=0,carrierFlowRows=0,
        childSourceDonorRows=0,sourceDonorRows=0;
    std::uint64_t augmentationCount=0,semanticCandidateEdgeCount=0;
    bool verified=false,physicalAdmission=false;
    bool deterministicOptimizerReplay=false,solverFreeFeasibility=false;
    std::uint32_t nativeSteps=0;
};

namespace recorded_carrier_allocation_import_detail {
using namespace source_carrier_quadrature_import_detail;
using namespace physical_skin_child_mass_detail;

inline bool readPinnedReceipt(const std::filesystem::path& path,const std::string& expected,
                              std::string& text,std::string& error) {
    if(!verifyFilePin(path,expected,error))return false;
    std::error_code ec;const auto bytes=std::filesystem::file_size(path,ec);
    if(ec||bytes>16u*1024u*1024u){error="recorded carrier allocation receipt is unreadable or oversized";return false;}
    std::ifstream f(path,std::ios::binary);
    if(!f){error="cannot open recorded carrier allocation receipt";return false;}
    text.assign(std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>());
    if(f.bad()){error="failed reading recorded carrier allocation receipt";return false;}
    std::string after;if(!physical_skin_child_mass_detail::sha256File(path,after,error)||after!=expected){
        error="recorded carrier allocation receipt changed while being read";return false;
    }
    return true;
}
inline bool openCSV(const std::filesystem::path& path,const std::string& expected,
                    std::ifstream& input,std::string& error) {
    if(!verifyFilePin(path,expected,error))return false;
    input.open(path);
    if(!input){error="cannot open pinned recorded carrier CSV: "+path.string();return false;}
    return true;
}
inline bool finishCSV(const std::filesystem::path& path,const std::string& expected,
                      std::ifstream& input,std::string& error) {
    if(!input.eof()||input.bad()){error="recorded carrier CSV read failed: "+path.string();return false;}
    std::string after;if(!physical_skin_child_mass_detail::sha256File(path,after,error)||after!=expected){
        error="recorded carrier CSV changed while being parsed: "+path.string();return false;
    }
    return true;
}
inline bool receiptFlowCSV(std::string_view flowMap,const std::string& name,
                           const std::string& sha,const std::uint64_t size) {
    std::string row,recordedSHA;std::uint64_t recordedSize=0;
    return object(flowMap,name,row)&&str(row,"sha256",recordedSHA)&&recordedSHA==sha&&
        num(row,"size_bytes",recordedSize)&&recordedSize==size;
}
inline bool boolean(std::string_view json,std::string_view key,bool& value) {
    const std::string token="\""+std::string(key)+"\"";
    auto p=json.find(token);
    if(p==json.npos||json.find(token,p+token.size())!=json.npos)return false;
    p=json.find(':',p+token.size());if(p==json.npos)return false;++p;
    while(p<json.size()&&std::isspace(static_cast<unsigned char>(json[p])))++p;
    if(json.substr(p,4u)=="true"&&(p+4u==json.size()||json[p+4u]==','||json[p+4u]=='}'||
       std::isspace(static_cast<unsigned char>(json[p+4u])))){value=true;return true;}
    if(json.substr(p,5u)=="false"&&(p+5u==json.size()||json[p+5u]==','||json[p+5u]=='}'||
       std::isspace(static_cast<unsigned char>(json[p+5u])))){value=false;return true;}
    return false;
}
inline bool validPositive(const double value){return std::isfinite(value)&&value>0.0;}
inline bool validNonnegative(const double value){return std::isfinite(value)&&value>=0.0;}

[[nodiscard]] inline bool loadPhysicalSkinRecordedCarrierAllocationCSVs(
        const PhysicalSkinRecordedCarrierAllocationCSVAssets& assets,
        const std::uint32_t expectedMaterial,const std::size_t expectedParentRows,
        const std::size_t expectedChildRows,
        PhysicalSkinSourceCarrierRoleRecordedAllocationData& output,
        PhysicalSkinRecordedCarrierAllocationCSVImportReport& report,std::string& error) {
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if((expectedMaterial!=15u&&expectedMaterial!=16u)||assets.materialIndex!=expectedMaterial||
       expectedParentRows==0u||expectedChildRows==0u||expectedParentRows>1000000u||
       expectedChildRows>2000000u||!validSHA(assets.receiptSHA256)||assets.receiptPath.empty())
        return fail("recorded carrier allocation import has an invalid parent/child domain or receipt pin");
    for(std::size_t i=0;i<assets.csvPaths.size();++i)
        if(assets.csvPaths[i].empty()||!validSHA(assets.csvSHA256[i]))
            return fail("recorded carrier allocation import requires six explicit CSV path/SHA pins");

    std::string receipt;
    if(!readPinnedReceipt(assets.receiptPath,assets.receiptSHA256,receipt,error))return false;
    std::string status,replay,flowFeasibility,optimizerReplay,flowMap,packetSHA,arraySHA,traceSHA,transportSHA;
    std::uint64_t nativeSteps=std::numeric_limits<std::uint64_t>::max(),parents=0,children=0,
        edges=0,candidates=0,augmentations=0;
    const std::string roleName=expectedMaterial==15u?"SKIN":"FAT";
    const std::string parentCountKey=expectedMaterial==15u?"parent_skin_tets":"parent_fat_tets";
    const std::string childCountKey=expectedMaterial==15u?"child_skin_tets":"child_fat_tets";
    const bool deterministicReplayPath=
        str(receipt,"status",status)&&
        status=="PASS_SOURCE_ONLY_"+roleName+"_SOURCE_CARRIER_SHARED_CAPACITY_ALLOCATION"&&
        str(receipt,"recorded_flow_deterministic_replay",replay)&&replay=="PASS";
    bool completeAllocation=false;
    const bool solverFreeFeasibilityPath=
        status=="PASS_SOURCE_ONLY_"+roleName+"_SOURCE_CARRIER_FEASIBLE_ALLOCATION"&&
        str(receipt,"recorded_flow_feasibility",flowFeasibility)&&flowFeasibility=="PASS_SOLVER_FREE"&&
        str(receipt,"deterministic_optimizer_replay",optimizerReplay)&&optimizerReplay=="NOT_RUN"&&
        boolean(receipt,"complete_allocation",completeAllocation)&&completeAllocation;
    if((!deterministicReplayPath&&!solverFreeFeasibilityPath)||
       !noAdmission(receipt)||!num(receipt,"native_steps",nativeSteps)||nativeSteps!=0u||
       !str(receipt,"source_carrier_packet_sha256",packetSHA)||!validSHA(packetSHA)||
       !str(receipt,"source_carrier_array_sha256",arraySHA)||!validSHA(arraySHA)||
       !str(receipt,"source_carrier_flow_trace_sha256",traceSHA)||!validSHA(traceSHA)||
       !str(receipt,"transport_receipt_sha256",transportSHA)||!validSHA(transportSHA)||
       !num(receipt,parentCountKey,parents)||parents!=expectedParentRows||
       !num(receipt,childCountKey,children)||children!=expectedChildRows||
       !num(receipt,"transport_edges",edges)||edges==0u||edges>12000000u||
       !num(receipt,"candidate_edges",candidates)||candidates<edges||candidates>12000000u||
       !num(receipt,"augmentations",augmentations)||augmentations==0u||augmentations>4000000u||
       !object(receipt,"flow_csv",flowMap))
        return fail("recorded carrier allocation receipt is not a complete source-only replay- or feasibility-verified result");
    for(const auto& sha:assets.expectedInputSHA256)
        if(!validSHA(sha)||receipt.find("\""+sha+"\"")==std::string::npos)
            return fail("recorded carrier allocation receipt omits a required current input SHA pin");
    for(std::size_t i=0;i<assets.csvPaths.size();++i){
        std::error_code ec;const auto size=std::filesystem::file_size(assets.csvPaths[i],ec);
        if(ec||!receiptFlowCSV(flowMap,assets.csvPaths[i].filename().string(),assets.csvSHA256[i],size))
            return fail("recorded carrier allocation receipt does not bind the exact six CSV files");
    }

    PhysicalSkinSourceCarrierRoleRecordedAllocationData next;
    next.materialIndex=expectedMaterial;
    next.allocationReceiptSHA256=assets.receiptSHA256;
    spatial_mass_transport_detail::SHA256Builder expandedReceiptAnchor;
    expandedReceiptAnchor.string("numi.human.recorded-carrier-expanded-receipt-anchor.v1");
    expandedReceiptAnchor.string(assets.receiptSHA256);expandedReceiptAnchor.string(traceSHA);
    next.expandedSolverReceiptSHA256=expandedReceiptAnchor.finish();
    next.csvSHA256=assets.csvSHA256;next.sourceCarrierFlowTraceSHA256=traceSHA;
    next.sourceCarrierArraySHA256=arraySHA;next.sourceCarrierPacketSHA256=packetSHA;
    next.transportReceiptSHA256=transportSHA;
    next.augmentationCount=augmentations;next.semanticCandidateEdgeCount=candidates;
    next.sourceParentRows=expectedParentRows;next.childRows=expectedChildRows;
    auto& result=next.result;auto& trace=next.trace;
    result.method=UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport;
    result.usesFractionalSemanticSupport=true;
    result.transportEdgeCount=static_cast<std::size_t>(edges);
    result.augmentationCount=augmentations;result.semanticCandidateEdgeCount=candidates;
    result.childMassKg.assign(expectedChildRows,0.0);
    result.childImpliedDensityKgM3.assign(expectedChildRows,0.0);
    result.childAncestryMeasureVolumeM3.assign(expectedChildRows,0.0);
    result.childDominantParentRows.assign(expectedChildRows,spatial_mass_transport_detail::kInvalid);
    result.parentMassTransferredKg.assign(expectedParentRows,0.0);
    result.parentMeasureTransferredM3.assign(expectedParentRows,0.0);
    result.childParentOffsets.assign(expectedChildRows+1u,0u);

    // 0. child mass and reference volume
    {
        std::ifstream f;if(!openCSV(assets.csvPaths[0],assets.csvSHA256[0],f,error))return false;
        std::string line;if(!std::getline(f,line)||splitCSV(line)!=std::vector<std::string>{
            "child_row","reference_volume_m3","mass_kg","density_kg_m3"})
            return fail("recorded child-mass CSV header differs from the pinned six-file contract");
        next.childReferenceVolumeM3.reserve(expectedChildRows);
        std::size_t rowIndex=0;while(std::getline(f,line)){
            if(rowIndex>=expectedChildRows||line.size()>4096u)return fail("recorded child-mass CSV exceeds its row/line bound");
            const auto row=splitCSV(line);std::uint32_t child=0;double volume=0,mass=0,density=0;
            if(row.size()!=4u||!parseNumber(row[0],child)||child!=rowIndex||
               !parseNumber(row[1],volume)||!parseNumber(row[2],mass)||!parseNumber(row[3],density)||
               !validPositive(volume)||!validPositive(mass)||!validPositive(density))
                return fail("recorded child-mass CSV row is malformed, noncontiguous, or nonpositive");
            next.childReferenceVolumeM3.push_back(volume);result.childMassKg[rowIndex]=mass;
            result.childImpliedDensityKgM3[rowIndex]=density;++rowIndex;
        }
        if(rowIndex!=expectedChildRows||!finishCSV(assets.csvPaths[0],assets.csvSHA256[0],f,error))
            return fail("recorded child-mass CSV does not cover the full current child row domain");
    }

    // 1. collapsed parent-child flow
    {
        std::ifstream f;if(!openCSV(assets.csvPaths[1],assets.csvSHA256[1],f,error))return false;
        std::string line;if(!std::getline(f,line)||splitCSV(line)!=std::vector<std::string>{
            "child_row","parent_row","mass_kg","source_measure_m3","distance_m"})
            return fail("recorded transport-flow CSV header differs from the pinned six-file contract");
        result.childParentRows.clear();result.childParentTransportMassKg.clear();
        result.childParentTransportMeasureM3.clear();result.childParentTransportDistanceM.clear();
        std::size_t child=0;std::uint32_t priorParent=0;bool haveParent=false;std::size_t rowIndex=0;
        result.childParentOffsets[0]=0;
        while(std::getline(f,line)){
            if(rowIndex>=12000000u||line.size()>4096u)return fail("recorded transport-flow CSV exceeds its row/line bound");
            const auto row=splitCSV(line);std::uint32_t c=0,p=0;double mass=0,measure=0,distance=0;
            if(row.size()!=5u||!parseNumber(row[0],c)||!parseNumber(row[1],p)||
               !parseNumber(row[2],mass)||!parseNumber(row[3],measure)||!parseNumber(row[4],distance)||
               c>=expectedChildRows||p>=expectedParentRows||!validPositive(mass)||!validPositive(measure)||
               !validNonnegative(distance)||distance>0.019353)
                return fail("recorded transport-flow row has an invalid ID, mass, measure, or distance");
            if(rowIndex==0u){child=c;priorParent=p;haveParent=true;}
            else if(c<child||(c==child&&p<=priorParent))
                return fail("recorded transport-flow rows are not sorted and unique by child/parent");
            if(c!=child){
                result.childParentOffsets[child+1u]=static_cast<std::uint32_t>(result.childParentRows.size());
                for(std::size_t empty=child+1u;empty<c;++empty)
                    result.childParentOffsets[empty+1u]=result.childParentOffsets[empty];
                child=c;priorParent=p;haveParent=true;
            }else if(haveParent){priorParent=p;}
            result.childParentRows.push_back(p);result.childParentTransportMassKg.push_back(mass);
            result.childParentTransportMeasureM3.push_back(measure);result.childParentTransportDistanceM.push_back(distance);
            result.childAncestryMeasureVolumeM3[c]+=measure;
            result.parentMassTransferredKg[p]+=mass;result.parentMeasureTransferredM3[p]+=measure;
            ++rowIndex;
        }
        if(rowIndex==0u||!finishCSV(assets.csvPaths[1],assets.csvSHA256[1],f,error))
            return fail("recorded transport-flow CSV is empty or unreadable");
        result.childParentOffsets[child+1u]=static_cast<std::uint32_t>(result.childParentRows.size());
        for(std::size_t empty=child+1u;empty<expectedChildRows;++empty)
            result.childParentOffsets[empty+1u]=result.childParentOffsets[empty];
        for(std::size_t c=0;c<expectedChildRows;++c){
            const auto begin=result.childParentOffsets[c],end=result.childParentOffsets[c+1u];
            if(begin>=end)return fail("recorded transport-flow CSV leaves a current child without a source flow");
            double best=-1.0;std::uint32_t dominant=spatial_mass_transport_detail::kInvalid;
            double fractionSum=0.0;
            for(std::uint32_t e=begin;e<end;++e){
                const double fraction=result.childParentTransportMeasureM3[e]/result.childAncestryMeasureVolumeM3[c];
                result.childParentVolumeFractions.push_back(fraction);fractionSum+=fraction;
                const auto parent=result.childParentRows[e];const auto measure=result.childParentTransportMeasureM3[e];
                if(measure>best||(measure==best&&parent<dominant)){best=measure;dominant=parent;}
            }
            if(std::abs(fractionSum-1.0)>1.0e-12)return fail("recorded aggregate source-measure fractions do not sum to one");
            result.childDominantParentRows[c]=dominant;
        }
        if(rowIndex!=edges)return fail("recorded aggregate flow row count differs from the authenticated receipt");
        next.aggregateFlowRows=rowIndex;
    }

    // 2. expanded carrier-level flow. Every collapsed edge must have at least one row.
    {
        std::ifstream f;if(!openCSV(assets.csvPaths[2],assets.csvSHA256[2],f,error))return false;
        std::string line;if(!std::getline(f,line)||splitCSV(line)!=std::vector<std::string>{
            "child_row","parent_row","expanded_parent_row","carrier_row","mass_kg","source_measure_m3","distance_m"})
            return fail("recorded source-carrier flow CSV header differs from the pinned six-file contract");
        const auto ne=result.childParentRows.size();trace.childParentCarrierOffsets.assign(ne+1u,0u);
        std::map<std::pair<std::uint32_t,std::uint32_t>,std::size_t> edgeIndex;
        for(std::size_t c=0;c<expectedChildRows;++c)
            for(std::uint32_t e=result.childParentOffsets[c];e<result.childParentOffsets[c+1u];++e)
                edgeIndex.emplace(std::pair{static_cast<std::uint32_t>(c),result.childParentRows[e]},e);
        std::map<std::uint32_t,std::uint32_t> carrierMap;std::size_t currentEdge=0,rowIndex=0;
        while(std::getline(f,line)){
            if(rowIndex>=4000000u||line.size()>4096u)return fail("recorded source-carrier flow CSV exceeds its row/line bound");
            const auto row=splitCSV(line);std::uint32_t c=0,p=0,v=0,carrierRow=0;double mass=0,measure=0,distance=0;
            if(row.size()!=7u||!parseNumber(row[0],c)||!parseNumber(row[1],p)||
               !parseNumber(row[2],v)||!parseNumber(row[3],carrierRow)||
               !parseNumber(row[4],mass)||!parseNumber(row[5],measure)||!parseNumber(row[6],distance)||
               c>=expectedChildRows||p>=expectedParentRows||!validPositive(mass)||!validPositive(measure)||
               !validNonnegative(distance)||distance>0.019353)
                return fail("recorded source-carrier flow row has an invalid identity or value");
            const auto found=edgeIndex.find({c,p});
            if(found==edgeIndex.end()||found->second<currentEdge||v>=1000000u)
                return fail("recorded source-carrier flow references an absent or reordered aggregate edge");
            while(currentEdge<found->second){
                if(trace.childParentCarrierOffsets[currentEdge]==trace.childParentCarrierRows.size())
                    return fail("recorded source-carrier flow omits an aggregate parent-child edge");
                trace.childParentCarrierOffsets[currentEdge+1u]=static_cast<std::uint32_t>(trace.childParentCarrierRows.size());
                ++currentEdge;
            }
            if(found->second!=currentEdge)return fail("recorded source-carrier flow order is inconsistent");
            if(!trace.childParentCarrierRows.empty()&&trace.childParentCarrierOffsets[currentEdge]<trace.childParentCarrierRows.size()&&
               v<=trace.childParentCarrierRows.back())
                return fail("recorded source-carrier expanded parent rows are not strictly ordered within an aggregate edge");
            auto [it,inserted]=carrierMap.emplace(v,carrierRow);
            if(!inserted&&it->second!=carrierRow)
                return fail("recorded source-carrier flow assigns conflicting carrier rows to one expanded parent");
            trace.childParentCarrierRows.push_back(v);trace.childParentCarrierMassKg.push_back(mass);
            trace.childParentCarrierMeasureM3.push_back(measure);trace.childParentCarrierDistanceM.push_back(distance);
            ++rowIndex;
        }
        if(rowIndex==0u||!finishCSV(assets.csvPaths[2],assets.csvSHA256[2],f,error))
            return fail("recorded source-carrier flow CSV is empty or unreadable");
        if(trace.childParentCarrierOffsets[currentEdge]==trace.childParentCarrierRows.size())
            return fail("recorded source-carrier flow omits the final aggregate parent-child edge");
        trace.childParentCarrierOffsets[currentEdge+1u]=static_cast<std::uint32_t>(trace.childParentCarrierRows.size());
        for(std::size_t e=currentEdge+1u;e<ne;++e)
            if(trace.childParentCarrierOffsets[e]==trace.childParentCarrierOffsets[e-1u])
                return fail("recorded source-carrier flow omits an aggregate parent-child edge");
            else trace.childParentCarrierOffsets[e+1u]=static_cast<std::uint32_t>(trace.childParentCarrierRows.size());
        const auto maxVirtual=carrierMap.rbegin()->first;
        next.csvExpandedParentToCarrierRows.assign(static_cast<std::size_t>(maxVirtual)+1u,
            std::numeric_limits<std::uint32_t>::max());
        for(const auto& [v,r]:carrierMap)next.csvExpandedParentToCarrierRows[v]=r;
        next.carrierFlowRows=rowIndex;
    }

    // 3. child source-donor ledger
    {
        std::ifstream f;if(!openCSV(assets.csvPaths[3],assets.csvSHA256[3],f,error))return false;
        std::string line;if(!std::getline(f,line)||splitCSV(line)!=std::vector<std::string>{
            "child_row","source_donor_body","mass_kg"})
            return fail("recorded child-source-donor CSV header differs from the pinned six-file contract");
        auto& offsets=result.childSourceDonorOffsets;auto& ids=result.childSourceDonorBodyIndices;
        auto& masses=result.childSourceDonorMassKg;offsets.assign(expectedChildRows+1u,0u);
        std::size_t child=0,rowIndex=0;std::uint32_t priorDonor=0;bool rowOpen=false;
        while(std::getline(f,line)){
            if(rowIndex>=4000000u||line.size()>4096u)return fail("recorded child-source-donor CSV exceeds its row/line bound");
            const auto row=splitCSV(line);std::uint32_t c=0,donor=0;double mass=0;
            if(row.size()!=3u||!parseNumber(row[0],c)||!parseNumber(row[1],donor)||
               !parseNumber(row[2],mass)||c>=expectedChildRows||!validPositive(mass))
                return fail("recorded child-source-donor row has an invalid identity or mass");
            if(rowIndex==0u){child=c;priorDonor=donor;rowOpen=true;}
            else if(c<child||(c==child&&donor<=priorDonor))
                return fail("recorded child-source-donor rows are not sorted and unique");
            if(c!=child){offsets[child+1u]=static_cast<std::uint32_t>(ids.size());
                for(std::size_t empty=child+1u;empty<c;++empty)offsets[empty+1u]=offsets[empty];
                child=c;priorDonor=donor;rowOpen=true;
            }else if(rowOpen)priorDonor=donor;
            ids.push_back(donor);masses.push_back(mass);++rowIndex;
        }
        if(rowIndex==0u||!finishCSV(assets.csvPaths[3],assets.csvSHA256[3],f,error))
            return fail("recorded child-source-donor CSV is empty or unreadable");
        offsets[child+1u]=static_cast<std::uint32_t>(ids.size());
        for(std::size_t empty=child+1u;empty<expectedChildRows;++empty)offsets[empty+1u]=offsets[empty];
        for(std::size_t c=0;c<expectedChildRows;++c)
            if(offsets[c]>=offsets[c+1u])return fail("recorded child-source-donor CSV leaves a child without donor mass");
        next.childSourceDonorRows=rowIndex;
    }

    // 4. exact current source-parent identities and values
    {
        std::ifstream f;if(!openCSV(assets.csvPaths[4],assets.csvSHA256[4],f,error))return false;
        std::string line;if(!std::getline(f,line)||splitCSV(line)!=std::vector<std::string>{
            "parent_row","global_source330_row","material","geometric_donor","region","stable_source_face",
            "mass_kg","reference_volume_m3","source_centroid_x_m","source_centroid_y_m","source_centroid_z_m"})
            return fail("recorded source-parent CSV header differs from the pinned six-file contract");
        std::size_t rowIndex=0;while(std::getline(f,line)){
            if(rowIndex>=expectedParentRows||line.size()>4096u)return fail("recorded source-parent CSV exceeds its row/line bound");
            const auto row=splitCSV(line);std::uint32_t id=0,global=0,material=0,donor=0,region=0,stable=0;
            double mass=0,volume=0;V3 centroid{};
            if(row.size()!=11u||!parseNumber(row[0],id)||id!=rowIndex||!parseNumber(row[1],global)||
               !parseNumber(row[2],material)||!parseNumber(row[3],donor)||!parseNumber(row[4],region)||
               !parseNumber(row[5],stable)||!parseNumber(row[6],mass)||!parseNumber(row[7],volume)||
               !parseNumber(row[8],centroid[0])||!parseNumber(row[9],centroid[1])||!parseNumber(row[10],centroid[2])||
               !validPositive(mass)||!validPositive(volume)||!finitePoint(centroid))
                return fail("recorded source-parent row is malformed or nonpositive");
            next.sourceGlobalTetRows.push_back(global);next.sourceMaterialIndices.push_back(material);
            next.sourceDonorBodyIndices.push_back(donor);next.sourceAnatomicalRegionIndices.push_back(region);
            next.sourceStableFaceIdentities.push_back(stable);next.sourceMassKg.push_back(mass);
            next.sourceReferenceVolumeM3.push_back(volume);next.sourceCentroidsWorld.push_back(centroid);++rowIndex;
        }
        if(rowIndex!=expectedParentRows||!finishCSV(assets.csvPaths[4],assets.csvSHA256[4],f,error))
            return fail("recorded source-parent CSV does not cover the current source row domain");
        next.sourceParentRows=rowIndex;
    }

    // 5. exact parent source-donor ledger
    {
        std::ifstream f;if(!openCSV(assets.csvPaths[5],assets.csvSHA256[5],f,error))return false;
        std::string line;if(!std::getline(f,line)||splitCSV(line)!=std::vector<std::string>{
            "parent_row","source_donor_body","mass_kg"})
            return fail("recorded source-parent-donor CSV header differs from the pinned six-file contract");
        next.sourceDonorOffsets.assign(expectedParentRows+1u,0u);std::size_t parent=0,rowIndex=0;
        std::uint32_t priorDonor=0;while(std::getline(f,line)){
            if(rowIndex>=4000000u||line.size()>4096u)return fail("recorded source-parent-donor CSV exceeds its row/line bound");
            const auto row=splitCSV(line);std::uint32_t p=0,donor=0;double mass=0;
            if(row.size()!=3u||!parseNumber(row[0],p)||!parseNumber(row[1],donor)||
               !parseNumber(row[2],mass)||p>=expectedParentRows||!validPositive(mass))
                return fail("recorded source-parent-donor row has an invalid identity or mass");
            if(rowIndex==0u){parent=p;priorDonor=donor;}
            else if(p<parent||(p==parent&&donor<=priorDonor))
                return fail("recorded source-parent-donor rows are not sorted and unique");
            if(p!=parent){next.sourceDonorOffsets[parent+1u]=static_cast<std::uint32_t>(next.sourceDonorIndices.size());
                for(std::size_t empty=parent+1u;empty<p;++empty)next.sourceDonorOffsets[empty+1u]=next.sourceDonorOffsets[empty];
                parent=p;priorDonor=donor;
            }else priorDonor=donor;
            next.sourceDonorIndices.push_back(donor);next.sourceDonorMassKg.push_back(mass);++rowIndex;
        }
        if(rowIndex==0u||!finishCSV(assets.csvPaths[5],assets.csvSHA256[5],f,error))
            return fail("recorded source-parent-donor CSV is empty or unreadable");
        next.sourceDonorOffsets[parent+1u]=static_cast<std::uint32_t>(next.sourceDonorIndices.size());
        for(std::size_t empty=parent+1u;empty<expectedParentRows;++empty)next.sourceDonorOffsets[empty+1u]=next.sourceDonorOffsets[empty];
        for(std::size_t p=0;p<expectedParentRows;++p)
            if(next.sourceDonorOffsets[p]>=next.sourceDonorOffsets[p+1u])
                return fail("recorded source-parent-donor CSV leaves a parent without source mass");
        next.sourceDonorRows=rowIndex;
    }

    next.importedAndPinned=true;
    PhysicalSkinRecordedCarrierAllocationCSVImportReport measured;
    measured.csvSHA256=assets.csvSHA256;measured.receiptSHA256=assets.receiptSHA256;
    measured.sourceCarrierPacketSHA256=packetSHA;measured.sourceCarrierArraySHA256=arraySHA;
    measured.sourceCarrierFlowTraceSHA256=traceSHA;measured.transportReceiptSHA256=transportSHA;
    measured.sourceParentRows=next.sourceParentRows;measured.childRows=next.childRows;
    measured.aggregateFlowRows=next.aggregateFlowRows;measured.carrierFlowRows=next.carrierFlowRows;
    measured.childSourceDonorRows=next.childSourceDonorRows;measured.sourceDonorRows=next.sourceDonorRows;
    measured.augmentationCount=augmentations;measured.semanticCandidateEdgeCount=candidates;
    measured.verified=true;measured.physicalAdmission=false;measured.nativeSteps=0;
    measured.deterministicOptimizerReplay=deterministicReplayPath;
    measured.solverFreeFeasibility=solverFreeFeasibilityPath;
    output=std::move(next);report=std::move(measured);error.clear();return true;
}

} // namespace recorded_carrier_allocation_import_detail
} // namespace metalrobo::human::coarse_tissue

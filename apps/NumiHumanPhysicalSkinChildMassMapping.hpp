#pragma once
#include "NumiHumanPhysicalSkinChildCapacity.hpp"

// Strict adapter for a real compiled skin/fat parent mass packet and a separately
// prepared positive child volume. Spatial transport remains distinct from
// geometric ancestry; the existing unpaired volume owner performs final admission.
#include "NumiHumanUnpairedSkinFatVolume.hpp"
#include "ConservativeSpatialSourceCarrierQuadrature.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace metalrobo::human::coarse_tissue {

struct PhysicalSkinParentMassPacket {
    std::vector<V3> currentWorldPoints;
    std::vector<V3> referenceWorldPoints;
    std::vector<double> compiledLumpedNodeMassKg;
    std::vector<std::uint32_t> compiledGlobalNodeIndices;
    std::vector<std::uint32_t> nodeFrameOwnerBodyIndices;
    std::vector<V3> nodeFrameOwnerLocalPoints;

    std::vector<std::array<std::uint32_t,4>> tetrahedra;
    std::vector<double> referenceTetVolumeM3;
    std::vector<double> tetrahedronMassKg;
    std::vector<std::uint32_t> materialIndices;
    std::vector<std::uint32_t> donorBodyIndices;
    // These two identity arrays are deliberately supplied by the native owner:
    // the donor CSV packet does not contain anatomical-region or source-face IDs.
    std::vector<std::uint32_t> anatomicalRegionIndices;
    std::vector<std::uint32_t> stableSourceFaceIdentities;
    std::vector<ExplicitUnpairedVolumeSourceMassContribution> sourceMassContributions;

    std::string parentMassPacketSHA256;
    std::string parentSourceContributionSHA256;
    std::string parentTopologySHA256;
    std::string parentLabelSHA256;
    double maximumNodalMassClosureErrorKg=0.0;
    double maximumNodalMassClosureRelativeError=0.0;
    double totalMassKg=0.0;
};

namespace physical_skin_child_mass_detail {

inline std::vector<std::string> splitCSV(std::string line) {
    if(!line.empty()&&line.back()=='\r')line.pop_back();
    std::vector<std::string> fields;
    std::size_t begin=0u;
    for(;;){
        const auto end=line.find(',',begin);
        fields.push_back(line.substr(begin,end==std::string::npos?end:end-begin));
        if(end==std::string::npos)break;
        begin=end+1u;
    }
    return fields;
}

template<class T>
inline bool parseNumber(const std::string& text,T& value) {
    if(text.empty())return false;
    const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
    return parsed.ec==std::errc{}&&parsed.ptr==text.data()+text.size();
}

inline bool finitePoint(const V3& point) {
    return std::all_of(point.begin(),point.end(),[](double value){return std::isfinite(value);});
}

inline bool sha256File(const std::filesystem::path& path,std::string& digest,std::string& error) {
    std::ifstream input(path,std::ios::binary);
    if(!input){error="cannot open parent mass packet file "+path.string();return false;}
    CC_SHA256_CTX context;
    if(CC_SHA256_Init(&context)!=1){error="cannot initialize parent packet SHA-256";return false;}
    std::array<char,65536> buffer{};
    while(input){
        input.read(buffer.data(),static_cast<std::streamsize>(buffer.size()));
        const auto count=input.gcount();
        if(count>0&&CC_SHA256_Update(&context,buffer.data(),static_cast<CC_LONG>(count))!=1){
            error="cannot update parent packet SHA-256";return false;
        }
    }
    if(!input.eof()){error="failed reading parent mass packet file "+path.string();return false;}
    std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> bytes{};
    if(CC_SHA256_Final(bytes.data(),&context)!=1){error="cannot finalize parent packet SHA-256";return false;}
    std::ostringstream out;
    out<<std::hex<<std::setfill('0');
    for(const auto byte:bytes)out<<std::setw(2)<<static_cast<unsigned>(byte);
    digest=out.str();
    return true;
}

inline bool hashJoinedFiles(const std::array<std::string,4>& hashes,std::string& output) {
    CC_SHA256_CTX context;
    if(CC_SHA256_Init(&context)!=1)return false;
    for(const auto& hash:hashes){
        if(hash.size()!=64u||CC_SHA256_Update(&context,hash.data(),static_cast<CC_LONG>(hash.size()))!=1)
            return false;
    }
    std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> bytes{};
    if(CC_SHA256_Final(bytes.data(),&context)!=1)return false;
    std::ostringstream out;
    out<<std::hex<<std::setfill('0');
    for(const auto byte:bytes)out<<std::setw(2)<<static_cast<unsigned>(byte);
    output=out.str();
    return true;
}

inline double signedTetVolume(const V3& a,const V3& b,const V3& c,const V3& d) {
    return dot(sub(b,a),cross(sub(c,a),sub(d,a)))/6.0;
}

inline bool readHeader(std::ifstream& file,const std::filesystem::path& path,
                       const std::vector<std::string>& expected,std::string& error) {
    std::string line;
    if(!std::getline(file,line)||splitCSV(line)!=expected){
        error="parent mass packet has an unexpected header: "+path.string();return false;
    }
    return true;
}

inline bool validSHA256(std::string_view value) {
    return value.size()==64u&&std::all_of(value.begin(),value.end(),[](char c){
        return (c>='0'&&c<='9')||(c>='a'&&c<='f');
    });
}

} // namespace physical_skin_child_mass_detail

// Read the existing four-file native capture format:
//   base.csv, base.csv.nodes.csv, base.csv.source.csv, base.csv.tets.csv
// Parent region and stable surface-face identities are not invented from donor
// IDs or tet rows; the owning app attaches those explicit arrays separately.
[[nodiscard]] inline bool loadCompiledSkinFatParentMassPacketCSV(
        const std::filesystem::path& massCSV,
        PhysicalSkinParentMassPacket& output,
        std::string& error) {
    using namespace physical_skin_child_mass_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const auto nodesCSV=std::filesystem::path(massCSV.string()+".nodes.csv");
    const auto sourceCSV=std::filesystem::path(massCSV.string()+".source.csv");
    const auto tetsCSV=std::filesystem::path(massCSV.string()+".tets.csv");
    std::ifstream massFile(massCSV),nodesFile(nodesCSV),sourceFile(sourceCSV),tetsFile(tetsCSV);
    if(!massFile||!nodesFile||!sourceFile||!tetsFile)
        return fail("compiled parent mass packet requires all four native CSV files");
    if(!readHeader(massFile,massCSV,{"source_tet_row","material_index","mass_kg"},error)||
       !readHeader(nodesFile,nodesCSV,{"parent_node_index","compiled_global_node",
           "current_x_m","current_y_m","current_z_m","rest_x_m","rest_y_m","rest_z_m",
           "compiled_lumped_mass_kg","node_frame_owner_body","donor_local_x_m",
           "donor_local_y_m","donor_local_z_m","mass_bits"},error)||
       !readHeader(sourceFile,sourceCSV,{"object_local_node","material_index","donor_body",
           "mass_kg","local_x_m","local_y_m","local_z_m"},error)||
       !readHeader(tetsFile,tetsCSV,{"parent_tet_row","material_index","donor_body",
           "parent_node0","parent_node1","parent_node2","parent_node3",
           "parent_reference_volume_m3","parent_mass_kg"},error))
        return false;

    PhysicalSkinParentMassPacket next;
    constexpr std::size_t maxNodes=physical_skin_child_capacity_detail::maxNodes,maxSourceRows=1200000u;
    std::string line;
    std::uint32_t expectedNode=0u;
    while(std::getline(nodesFile,line)){
        if(next.referenceWorldPoints.size()>=maxNodes||line.size()>16384u)
            return fail("compiled parent node packet exceeds its bounded domain");
        const auto row=splitCSV(line);
        if(row.size()!=14u)return fail("compiled parent node CSV row has the wrong width");
        std::uint32_t id=0u,global=0u,owner=0u,massBits=0u;
        V3 current{},reference{},local{};
        double lumpedMass=0.0;
        if(!parseNumber(row[0],id)||id!=expectedNode||
           !parseNumber(row[1],global)||!parseNumber(row[8],lumpedMass)||
           !parseNumber(row[9],owner)||!parseNumber(row[13],massBits))
            return fail("compiled parent node CSV has a noncontiguous or invalid identity row "+
                std::to_string(expectedNode));
        for(unsigned axis=0;axis<3u;++axis)
            if(!parseNumber(row[2u+axis],current[axis])||
               !parseNumber(row[5u+axis],reference[axis])||
               !parseNumber(row[10u+axis],local[axis]))
                return fail("compiled parent node CSV has an invalid coordinate at row "+
                    std::to_string(expectedNode));
        if(!finitePoint(current)||!finitePoint(reference)||!finitePoint(local)||
           !std::isfinite(lumpedMass)||lumpedMass<=0.0||
           std::bit_cast<std::uint32_t>(static_cast<float>(lumpedMass))!=massBits||
           (expectedNode>0u&&global!=next.compiledGlobalNodeIndices.back()+1u))
            return fail("compiled parent node CSV has invalid Float32 mass bits, coordinates, or global ordering");
        next.currentWorldPoints.push_back(current);
        next.referenceWorldPoints.push_back(reference);
        next.compiledLumpedNodeMassKg.push_back(lumpedMass);
        next.compiledGlobalNodeIndices.push_back(global);
        next.nodeFrameOwnerBodyIndices.push_back(owner);
        next.nodeFrameOwnerLocalPoints.push_back(local);
        ++expectedNode;
    }
    if(!nodesFile.eof()||next.referenceWorldPoints.size()<4u)
        return fail("compiled parent node CSV is incomplete or empty");

    std::ifstream massRows(massCSV),tetRows(tetsCSV);
    std::string ignored;
    std::getline(massRows,ignored);std::getline(tetRows,ignored);
    std::vector<double> nodalMass(next.referenceWorldPoints.size(),0.0);
    std::set<std::array<std::uint32_t,4>> uniqueTets;
    std::uint32_t expectedTet=0u;
    for(;;){
        std::string massLine,tetLine;
        const bool haveMass=static_cast<bool>(std::getline(massRows,massLine));
        const bool haveTet=static_cast<bool>(std::getline(tetRows,tetLine));
        if(!haveMass&&!haveTet)break;
        if(!haveMass||!haveTet||expectedTet>=physical_skin_child_capacity_detail::maxCells||massLine.size()>4096u||tetLine.size()>4096u)
            return fail("compiled parent cell-mass and tetrahedron ledgers have different or excessive row counts");
        const auto massRow=splitCSV(massLine),tetRow=splitCSV(tetLine);
        if(massRow.size()!=3u||tetRow.size()!=9u)
            return fail("compiled parent cell/tet CSV row has the wrong width");
        std::uint32_t massID=0u,tetID=0u,material=0u,tetMaterial=0u,donor=0u;
        double mass=0.0,tetMass=0.0,referenceVolume=0.0;
        if(!parseNumber(massRow[0],massID)||massID!=expectedTet||
           !parseNumber(massRow[1],material)||!parseNumber(massRow[2],mass)||
           !parseNumber(tetRow[0],tetID)||tetID!=expectedTet||
           !parseNumber(tetRow[1],tetMaterial)||!parseNumber(tetRow[2],donor)||
           !parseNumber(tetRow[7],referenceVolume)||!parseNumber(tetRow[8],tetMass)||
           material!=tetMaterial||mass!=tetMass||!std::isfinite(mass)||mass<=0.0||
           !std::isfinite(referenceVolume)||referenceVolume<=0.0)
            return fail("compiled parent cell-mass and tet identities, mass, or material differ at row "+
                std::to_string(expectedTet));
        std::array<std::uint32_t,4> tet{};
        for(unsigned corner=0;corner<4u;++corner)
            if(!parseNumber(tetRow[3u+corner],tet[corner])||
               tet[corner]>=next.referenceWorldPoints.size())
                return fail("compiled parent tet references an invalid source node at row "+
                    std::to_string(expectedTet));
        auto key=tet;std::sort(key.begin(),key.end());
        if(std::adjacent_find(key.begin(),key.end())!=key.end()||!uniqueTets.insert(key).second)
            return fail("compiled parent tet packet contains repeated node IDs or duplicate cells");
        const auto& points=next.referenceWorldPoints;
        const double computedVolume=signedTetVolume(points[tet[0]],points[tet[1]],
            points[tet[2]],points[tet[3]]);
        if(!std::isfinite(computedVolume)||computedVolume<=0.0||
           std::abs(computedVolume-referenceVolume)>
               std::max(1.0e-15,1.0e-8*referenceVolume))
            return fail("compiled parent tet is nonpositive or its captured reference volume differs at row "+
                std::to_string(expectedTet));
        for(const auto node:tet)nodalMass[node]+=mass*0.25;
        next.tetrahedra.push_back(tet);
        next.referenceTetVolumeM3.push_back(referenceVolume);
        next.tetrahedronMassKg.push_back(mass);
        next.materialIndices.push_back(material);
        next.donorBodyIndices.push_back(donor);
        next.totalMassKg+=mass;
        ++expectedTet;
    }
    if(!massRows.eof()||!tetRows.eof()||next.tetrahedra.empty())
        return fail("compiled parent cell/tet mass packet is incomplete or empty");

    double maxAbsError=0.0,maxRelativeError=0.0;
    for(std::size_t node=0;node<nodalMass.size();++node){
        const double captured=next.compiledLumpedNodeMassKg[node];
        const double difference=std::abs(nodalMass[node]-captured);
        const double relative=difference/std::max({nodalMass[node],captured,1.0e-12});
        maxAbsError=std::max(maxAbsError,difference);
        maxRelativeError=std::max(maxRelativeError,relative);
        if(difference>std::max(2.0e-9,1.0e-6*std::max(nodalMass[node],captured)))
            return fail("compiled parent cell masses do not close their captured nodal mass at node "+
                std::to_string(node));
    }

    std::set<std::tuple<std::uint32_t,std::uint32_t,std::uint32_t>> sourceKeys;
    std::uint32_t sourceRow=0u;
    while(std::getline(sourceFile,line)){
        if(next.sourceMassContributions.size()>=maxSourceRows||line.size()>4096u)
            return fail("compiled parent source contribution packet exceeds its bounded domain");
        const auto row=splitCSV(line);
        if(row.size()!=7u)return fail("compiled parent source contribution CSV row has the wrong width");
        std::uint32_t node=0u,material=0u,donor=0u;
        double mass=0.0;
        V3 local{};
        if(!parseNumber(row[0],node)||node>=next.referenceWorldPoints.size()||
           !parseNumber(row[1],material)||!parseNumber(row[2],donor)||
           !parseNumber(row[3],mass)||!std::isfinite(mass)||mass<=0.0)
            return fail("compiled parent source contribution has an invalid node/material/donor/mass row "+
                std::to_string(sourceRow));
        for(unsigned axis=0;axis<3u;++axis)
            if(!parseNumber(row[4u+axis],local[axis])||!std::isfinite(local[axis]))
                return fail("compiled parent source contribution has a nonfinite donor-local point");
        if(!sourceKeys.emplace(node,material,donor).second)
            return fail("compiled parent source contribution repeats a node/material/donor identity");
        if(material!=next.materialIndices.front()&&
           std::find(next.materialIndices.begin(),next.materialIndices.end(),material)==next.materialIndices.end())
            return fail("compiled parent source contribution references an absent material");
        if(donor==std::numeric_limits<std::uint32_t>::max())
            return fail("compiled parent source contribution uses an invalid donor identity");
        if(donor==next.nodeFrameOwnerBodyIndices[node]&&
           norm(sub(local,next.nodeFrameOwnerLocalPoints[node]))>1.0e-7)
            return fail("compiled parent source donor-local point differs from its exact node-frame owner");
        next.sourceMassContributions.push_back({.parentNodeIndex=node,.materialIndex=material,
            .donorBodyIndex=donor,.massKg=mass,.donorLocalPosition=local});
        ++sourceRow;
    }
    if(!sourceFile.eof()||next.sourceMassContributions.empty())
        return fail("compiled parent source contribution packet is incomplete or empty");

    std::vector<ConservativeSpatialMassTransportParentNodeSourceMass> sourceRows;
    sourceRows.reserve(next.sourceMassContributions.size());
    for(const auto& row:next.sourceMassContributions)
        sourceRows.push_back({.parentNodeIndex=row.parentNodeIndex,
            .materialIndex=row.materialIndex,.sourceDonorBodyIndex=row.donorBodyIndex,
            .massKg=row.massKg});
    std::vector<std::uint32_t> splitOffsets,splitDonors;
    std::vector<double> splitMasses;
    std::string splitError;
    if(!assembleConservativeSpatialMassTransportParentSourceDonorSplits(
            next.tetrahedronMassKg,next.materialIndices,next.tetrahedra,sourceRows,
            1.0e-10,5.0e-4,splitOffsets,splitDonors,splitMasses,splitError))
        return fail("compiled parent donor source rows do not match cooked cell-corner mass splits: "+splitError);

    std::array<std::string,4> fileHashes{};
    const std::array<std::filesystem::path,4> paths{massCSV,nodesCSV,sourceCSV,tetsCSV};
    for(std::size_t i=0;i<paths.size();++i)
        if(!sha256File(paths[i],fileHashes[i],error))return false;
    next.parentMassPacketSHA256=fileHashes[0];
    next.parentSourceContributionSHA256=fileHashes[2];
    if(!hashJoinedFiles(fileHashes,next.parentTopologySHA256))
        return fail("cannot hash the complete compiled parent mass/topology packet");
    next.maximumNodalMassClosureErrorKg=maxAbsError;
    next.maximumNodalMassClosureRelativeError=maxRelativeError;
    output=std::move(next);
    error.clear();
    return true;
}

[[nodiscard]] inline bool attachExplicitParentMassMappingLabels(
        PhysicalSkinParentMassPacket& packet,
        const std::span<const std::uint32_t> anatomicalRegionIndices,
        const std::span<const std::uint32_t> stableSourceFaceIdentities,
        std::string& error) {
    const auto fail=[&](const char* message){error=message;return false;};
    if(anatomicalRegionIndices.size()!=packet.tetrahedra.size()||
       stableSourceFaceIdentities.size()!=packet.tetrahedra.size())
        return fail("parent mass mapping requires explicit region and stable source-face identity for every source tet");
    if(std::any_of(anatomicalRegionIndices.begin(),anatomicalRegionIndices.end(),
            [](auto id){return id==NM_INVALID_INDEX;})||
       std::any_of(stableSourceFaceIdentities.begin(),stableSourceFaceIdentities.end(),
            [](auto id){return id==NM_INVALID_INDEX;}))
        return fail("parent mass mapping contains an invalid anatomical region or source-face identity");
    packet.anatomicalRegionIndices.assign(anatomicalRegionIndices.begin(),anatomicalRegionIndices.end());
    packet.stableSourceFaceIdentities.assign(stableSourceFaceIdentities.begin(),stableSourceFaceIdentities.end());
    error.clear();
    return true;
}

// Optional companion to the existing native parent CSV capture. These labels
// come from RegisteredSkinFatReductionSourceLabels at the same source cook;
// source identities are never inferred from CSV row numbers.
[[nodiscard]] inline bool loadCompiledSkinFatParentLabelsCSV(
        const std::filesystem::path& labelCSV,
        PhysicalSkinParentMassPacket& packet,std::string& error) {
    using namespace physical_skin_child_mass_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    std::ifstream input(labelCSV);
    if(!input)return fail("cannot open native parent label CSV "+labelCSV.string());
    if(!readHeader(input,labelCSV,{"parent_tet_row","material_index",
            "geometric_donor_body","anatomical_region_id","stable_source_face_id"},error))
        return false;
    const auto count=packet.tetrahedra.size();
    if(count==0u||packet.materialIndices.size()!=count||packet.donorBodyIndices.size()!=count)
        return fail("native parent labels require a complete parsed parent packet");
    std::vector<std::uint32_t> regions(count),faces(count);
    std::string line;std::size_t row=0u;
    while(std::getline(input,line)) {
        const auto fields=splitCSV(line);
        std::uint32_t cell=0u,material=0u,donor=0u,region=0u,face=0u;
        if(row>=count||fields.size()!=5u||
           !parseNumber(fields[0],cell)||!parseNumber(fields[1],material)||
           !parseNumber(fields[2],donor)||!parseNumber(fields[3],region)||
           !parseNumber(fields[4],face)||cell!=row||
           material!=packet.materialIndices[row]||donor!=packet.donorBodyIndices[row]||
           region==NM_INVALID_INDEX||face==NM_INVALID_INDEX)
            return fail("native parent label row does not match current parent material/donor identity at row "+
                std::to_string(row));
        regions[row]=region;faces[row]=face;++row;
    }
    if(!input.eof()||row!=count)
        return fail("native parent label CSV is incomplete or unreadable");
    std::string digest;
    if(!sha256File(labelCSV,digest,error))return false;
    // Publish only after every row, identity and source hash has passed.
    packet.anatomicalRegionIndices=std::move(regions);
    packet.stableSourceFaceIdentities=std::move(faces);
    packet.parentLabelSHA256=std::move(digest);
    error.clear();return true;
}

struct PhysicalSkinChildStableDeepAttachment {
    std::uint64_t stableAttachmentID=0u;
    std::uint32_t donorBody=NM_INVALID_INDEX;
    std::uint32_t stableSourceVertex=NM_INVALID_INDEX;
    std::uint32_t deepSurfaceVertexRow=NM_INVALID_INDEX;
    std::uint32_t volumeNodeRow=NM_INVALID_INDEX;
    friend bool operator==(const PhysicalSkinChildStableDeepAttachment&,
                           const PhysicalSkinChildStableDeepAttachment&)=default;
};

// Compact, already-authenticated 002 -> 530 -> fitted-candidate lineage used
// by the one-time mass initializer. This carries immutable-330 cell rows and
// 534 donor shares; it does not replace the parent packet or represent runtime
// mass advection.
struct PhysicalSkinVerifiedConformingMassRoute {
    bool authenticated=false;
    std::size_t refinedSourceCellCount=0u;
    std::vector<std::uint32_t> childRefinedSourceRows,childImmutableParentRows;
    std::vector<double> childImmutableParentFractions;
    std::vector<double> childImmutableParentReferenceMeasureM3,childMassKg;
    std::vector<std::uint32_t> childSourceDonorOffsets,childSourceDonorBodyIndices;
    std::vector<double> childSourceDonorMassKg;
    std::string parentMassPacketSHA256,parentSourceContributionSHA256;
    std::string sourceVolumeSHA256,refinedVolumeSHA256,candidateVolumeSHA256;
    std::string refinementVerificationSHA256,sourceDonorLedgerVerificationSHA256;
    std::string pullbackVerificationSHA256,chainVerificationSHA256;
    std::string candidateOperationalReferenceDomainSHA256,candidateCompiledDataSHA256,verificationSHA256;
};

struct PhysicalSkinSourceCarrierRoleRecordedAllocationData {
    bool importedAndPinned=false;
    std::uint32_t materialIndex=NM_INVALID_INDEX;
    std::array<std::string,6> csvSHA256{};
    std::string allocationReceiptSHA256,expandedSolverReceiptSHA256,sourceCarrierPacketSHA256,
        sourceCarrierArraySHA256,sourceCarrierFlowTraceSHA256,transportReceiptSHA256;
    std::uint64_t augmentationCount=0,semanticCandidateEdgeCount=0;
    std::size_t sourceParentRows=0,childRows=0,aggregateFlowRows=0,carrierFlowRows=0,
        childSourceDonorRows=0,sourceDonorRows=0;
    std::vector<std::uint32_t> sourceGlobalTetRows,sourceMaterialIndices,sourceDonorBodyIndices,
        sourceAnatomicalRegionIndices,sourceStableFaceIdentities,csvExpandedParentToCarrierRows;
    std::vector<double> childReferenceVolumeM3,sourceMassKg,sourceReferenceVolumeM3;
    std::vector<V3> sourceCentroidsWorld;
    std::vector<std::uint32_t> sourceDonorOffsets,sourceDonorIndices;
    std::vector<double> sourceDonorMassKg;
    ConservativeSpatialMassTransportResult result;
    SourceCellCarrierTransportTrace trace;
};
struct PhysicalSkinSourceCarrierRecordedAllocationData {
    bool importedAndPinned=false;
    std::array<PhysicalSkinSourceCarrierRoleRecordedAllocationData,2> roles{};
    std::array<std::string,2> roleReceiptSHA256{},roleSourceCarrierPacketSHA256{},
        roleSourceCarrierArraySHA256{},roleSourceCarrierFlowTraceSHA256{},roleTransportReceiptSHA256{};
    std::array<std::string,12> csvSHA256{};
    std::string allocationReceiptSHA256,expandedSolverReceiptSHA256,
        sourceCarrierPacketSHA256,sourceCarrierArraySHA256,sourceCarrierFlowTraceSHA256,
        transportReceiptSHA256;
    std::uint64_t augmentationCount=0,semanticCandidateEdgeCount=0;
    std::size_t sourceParentRows=0,childRows=0,aggregateFlowRows=0,carrierFlowRows=0,
        childSourceDonorRows=0,sourceDonorRows=0;
    ConservativeSpatialMassTransportResult result;
    SourceCellCarrierTransportTrace trace;
};

struct PhysicalSkinSourceCarrierQuadratureData {
    bool importedAndVerified=false;
    std::string verificationSHA256,sourceCarrierPacketSHA256,sourceTetTableSHA256,sourceNodeTableSHA256;
    std::array<std::string,2> roleBinarySHA256{},roleReceiptSHA256{};
    std::vector<std::uint32_t> parentOffsets,globalSourceTetRows,witnessChildRows;
    std::vector<double> carrierWeights,witnessChildCentroidDistancesM;
    std::vector<std::array<double,4>> carrierBarycentric;
    SourceCellCarrierTransportTrace transportTrace;
    PhysicalSkinSourceCarrierRecordedAllocationData recordedAllocation;
    [[nodiscard]] bool hasCarriers() const { return !carrierWeights.empty(); }
};

struct PhysicalSkinChildMassMappingBundle {
    PhysicalSkinParentMassPacket parent;
    RegisteredSurface outerSkin;
    RegisteredSurface independentSkinInner;
    RegisteredSurface independentFatDeep;
    numi::human::SkinBoundaryNodeMap boundaryMap;

    std::vector<V3> currentWorldPoints,referenceWorldPoints;
    std::vector<std::array<std::uint32_t,4>> childTetrahedra;
    std::vector<std::uint32_t> childMaterialIndices,childDominantParentTetrahedra;
    std::vector<std::uint32_t> childParentSurfaceFaces,childStableSourceFaceIdentities;
    std::vector<std::uint32_t> childDonorBodyIndices,childAnatomicalRegionIndices;
    // Optional global semantic-group table plus child-to-group volume support.
    // Populated only from the authenticated current-atlas producer.
    bool fractionalSemanticMassRouteRequired=false;
    ConservativeSpatialChildSemanticSupportData fractionalSemanticSupport;
    // Optional source-tetra quadrature is an authenticated spatial-support
    // extension. It partitions unchanged parent mass/measure and never changes
    // source geometry, labels, density, or the legacy empty-carrier solver path.
    PhysicalSkinSourceCarrierQuadratureData sourceCarrierQuadrature;
    std::vector<std::uint32_t> childRegisteredLayerFacetSurfaceFaces;
    std::vector<std::uint32_t> childRegisteredLayerFacetSourceFaceIdentities;
    std::vector<std::uint8_t> childAuthoredHeadSidewallFacetFlags;
    std::vector<std::uint32_t> childAuthoredHeadSidewallStableFaceIdentities;
    // Stable attachment identities are retained as typed D-domain rows.
    // They are source constraints, not mass or a permission to reuse old volume nodes.
    std::vector<PhysicalSkinChildStableDeepAttachment> stableDeepAttachments;

    std::vector<std::uint32_t> volumeNodeDonorBodyIndices;
    std::vector<V3> volumeNodeDonorLocalPoints,volumeNodeDonorOriginWorld;
    std::vector<std::array<double,9>> volumeNodeDonorLocalToWorldLinear;
    std::vector<std::array<double,9>> registeredBodyLocalToWorldLinear;
    std::vector<V3> registeredBodyOriginWorld;
    std::size_t detailedSourceVertexCount=0u,detailedSourceFaceCount=0u;
    // Typed source namespaces preserve mixed O/I/D ancestry without treating
    // role-local IDs as one canonical detailed payload index space.
    std::vector<PhysicalSkinSourceDomainRegistration> sourceDomains;
    std::vector<std::uint32_t> volumeNodeSourceDomainIndices;
    std::vector<std::uint32_t> volumeNodeSourceVertexIndices,volumeNodeSourceFaceIndices;
    std::vector<std::array<double,3>> volumeNodeSourceBarycentric;
    std::size_t retainedOriginalNodeCount=0u;
    // Historical node-to-330 registration remains immutable provenance. In
    // the explicit operational-reference route, the separate candidate-local
    // map below owns the repaired support coordinates and retains the same
    // declared 0.4 mm registration bound.
    std::vector<std::uint32_t> volumeNodeParentTetrahedra;
    std::vector<std::array<double,4>> volumeNodeParentBarycentric;
    std::vector<V3> volumeNodeParentResidualWorldM;
    std::vector<std::uint32_t> volumeNodeSourceParentNodeRows;
    std::vector<double> volumeNodeSourceParentRegistrationDistanceM;
    std::vector<std::uint32_t> volumeNodeOperationalReferenceTetrahedra;
    std::vector<std::array<double,4>> volumeNodeOperationalReferenceBarycentric;
    std::vector<V3> volumeNodeOperationalReferenceResidualWorldM;
    std::vector<double> volumeNodeOperationalReferenceDistanceM;
    bool usesSeparatedOperationalReference=false;
    std::string operationalReferenceDomainSHA256;
    std::array<std::uint64_t,4> referenceSourceIdentity{},materialSourceIdentity{};

    std::string childMeshSHA256,regionLabelSHA256,coordinateFrameSHA256;
    double maximumSpatialMassTransportDistanceM=0.0;
    ConservativeSpatialMassTransportResult spatialMassTransportResult;
    // Audit metadata for the deterministic exact-topology shortcut. The
    // transport receipt remains the existing receipt over physical inputs and
    // flows; this separate digest binds which mapping algorithm was used.
    bool exactTopologyIdentityTransportUsed=false;
    // The proof-required bit is set only by the transactional registered-file
    // importer. When set, a missing or invalid lineage route must fail closed
    // instead of falling back to global spatial transport.
    bool conformingLineageMassRouteRequired=false;
    bool conformingLineageMassTransportUsed=false;
    PhysicalSkinVerifiedConformingMassRoute conformingLineageMassRoute;
    // Set only by authenticated operational-reference file import. The existing
    // transport result is recomputed/verified against current inputs before use.
    // Historical source ancestry remains separate from spatial mass transport.
    bool operationalSpatialMassRouteRequired=false;
    std::string operationalSpatialReferenceVerificationSHA256;
    std::string operationalSpatialTransportReceiptSHA256;
    std::string massMappingPathSHA256;

    [[nodiscard]] ExplicitUnpairedVolumeTopology topologyView() const {
        ExplicitUnpairedVolumeTopology topology;
        topology.independentSkinInnerSurface=&independentSkinInner;
        topology.independentFatDeepSurface=&independentFatDeep;
        topology.currentWorldPoints=currentWorldPoints;
        topology.referenceWorldPoints=referenceWorldPoints;
        topology.childTetrahedra=childTetrahedra;
        topology.childMaterialIndices=childMaterialIndices;
        topology.childDominantParentTetrahedra=childDominantParentTetrahedra;
        topology.childParentSurfaceFaces=childParentSurfaceFaces;
        topology.childStableSourceFaceIdentities=childStableSourceFaceIdentities;
        topology.childDonorBodyIndices=childDonorBodyIndices;
        topology.childAnatomicalRegionIndices=childAnatomicalRegionIndices;
        topology.fractionalSemanticMassRouteRequired=fractionalSemanticMassRouteRequired;
        topology.fractionalSemanticSupport=fractionalSemanticSupport.view();

        topology.parentReferenceWorldPoints=parent.referenceWorldPoints;
        topology.parentTetrahedra=parent.tetrahedra;
        topology.parentMaterialIndices=parent.materialIndices;
        topology.parentStableSourceFaceIdentities=parent.stableSourceFaceIdentities;
        topology.parentDonorBodyIndices=parent.donorBodyIndices;
        topology.parentAnatomicalRegionIndices=parent.anatomicalRegionIndices;
        topology.parentTetrahedronMassKg=parent.tetrahedronMassKg;
        topology.parentNodeSourceMassContributions=parent.sourceMassContributions;

        topology.childRegisteredLayerFacetSurfaceFaces=childRegisteredLayerFacetSurfaceFaces;
        topology.childRegisteredLayerFacetSourceFaceIdentities=childRegisteredLayerFacetSourceFaceIdentities;
        topology.childAuthoredHeadSidewallFacetFlags=childAuthoredHeadSidewallFacetFlags;
        topology.childAuthoredHeadSidewallStableFaceIdentities=childAuthoredHeadSidewallStableFaceIdentities;

        topology.volumeNodeDonorBodyIndices=volumeNodeDonorBodyIndices;
        topology.volumeNodeDonorLocalPoints=volumeNodeDonorLocalPoints;
        topology.volumeNodeDonorLocalToWorldLinear=volumeNodeDonorLocalToWorldLinear;
        topology.volumeNodeDonorOriginWorld=volumeNodeDonorOriginWorld;
        topology.registeredBodyLocalToWorldLinear=registeredBodyLocalToWorldLinear;
        topology.registeredBodyOriginWorld=registeredBodyOriginWorld;
        topology.detailedSourceVertexCount=detailedSourceVertexCount;
        topology.detailedSourceFaceCount=detailedSourceFaceCount;
        topology.sourceDomains=sourceDomains;
        topology.volumeNodeSourceDomainIndices=volumeNodeSourceDomainIndices;
        topology.volumeNodeSourceVertexIndices=volumeNodeSourceVertexIndices;
        topology.volumeNodeSourceFaceIndices=volumeNodeSourceFaceIndices;
        topology.volumeNodeSourceBarycentric=volumeNodeSourceBarycentric;
        topology.retainedOriginalNodeCount=retainedOriginalNodeCount;
        topology.volumeNodeParentTetrahedra=volumeNodeParentTetrahedra;
        topology.volumeNodeParentBarycentric=volumeNodeParentBarycentric;
        topology.volumeNodeParentResidualWorldM=volumeNodeParentResidualWorldM;

        topology.ancestryVerificationKind=VolumeAncestryVerificationKind::unverified;
        topology.massTransferKind=UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport;
        topology.spatialMassTransportResult=&spatialMassTransportResult;
        topology.maximumSpatialMassTransportDistanceM=maximumSpatialMassTransportDistanceM;
        topology.referenceSourceIdentity=referenceSourceIdentity;
        topology.materialSourceIdentity=materialSourceIdentity;
        return topology;
    }

};

namespace physical_skin_child_mass_mapping_detail {

[[nodiscard]] inline bool sourceCarrierRelativeMassClose(const double actual,const double expected) {
    return std::isfinite(actual)&&std::isfinite(expected)&&expected>0.0&&
        std::abs(actual-expected)<=1.0e-10*std::abs(expected);
}

[[nodiscard]] inline bool recordedRoleChildDonorLedgerMatches(
        const ConservativeSpatialMassTransportResult& roleResult,const std::size_t roleChild,
        const ConservativeSpatialMassTransportResult& pairedResult,const std::size_t pairedChild) {
    if(roleChild+1u>=roleResult.childSourceDonorOffsets.size()||
       pairedChild+1u>=pairedResult.childSourceDonorOffsets.size())return false;
    const auto rb=roleResult.childSourceDonorOffsets[roleChild];
    const auto re=roleResult.childSourceDonorOffsets[roleChild+1u];
    const auto pb=pairedResult.childSourceDonorOffsets[pairedChild];
    const auto pe=pairedResult.childSourceDonorOffsets[pairedChild+1u];
    if(rb>re||re>roleResult.childSourceDonorBodyIndices.size()||
       re>roleResult.childSourceDonorMassKg.size()||pb>pe||
       pe>pairedResult.childSourceDonorBodyIndices.size()||pe>pairedResult.childSourceDonorMassKg.size()||
       re-rb!=pe-pb)return false;
    for(std::uint32_t i=0;i<re-rb;++i)
        if(roleResult.childSourceDonorBodyIndices[rb+i]!=pairedResult.childSourceDonorBodyIndices[pb+i]||
           !sourceCarrierRelativeMassClose(roleResult.childSourceDonorMassKg[rb+i],
                                           pairedResult.childSourceDonorMassKg[pb+i]))return false;
    return true;
}

[[nodiscard]] inline bool appendRecordedCarrierFlowEdge(
        const SourceCellCarrierTransportTrace& roleTrace,const std::size_t roleEdge,
        const std::span<const std::uint32_t> roleVirtualToCurrent,
        SourceCellCarrierTransportTrace& combinedTrace,std::string& error) {
    const auto fail=[&](const char* message){error=message;return false;};
    const auto& offsets=roleTrace.childParentCarrierOffsets;
    if(offsets.size()<2u||offsets.front()!=0u||offsets.back()!=roleTrace.childParentCarrierRows.size()||
       roleTrace.childParentCarrierMassKg.size()!=roleTrace.childParentCarrierRows.size()||
       roleTrace.childParentCarrierMeasureM3.size()!=roleTrace.childParentCarrierRows.size()||
       roleTrace.childParentCarrierDistanceM.size()!=roleTrace.childParentCarrierRows.size()||
       roleEdge+1u>=offsets.size()||offsets[roleEdge]>=offsets[roleEdge+1u]||
       offsets[roleEdge+1u]>roleTrace.childParentCarrierRows.size())
        return fail("role carrier edge has an invalid or empty CSR range");
    if(combinedTrace.childParentCarrierOffsets.empty()||combinedTrace.childParentCarrierOffsets.front()!=0u||
       combinedTrace.childParentCarrierOffsets.back()!=combinedTrace.childParentCarrierRows.size()||
       combinedTrace.childParentCarrierRows.size()!=combinedTrace.childParentCarrierMassKg.size()||
       combinedTrace.childParentCarrierRows.size()!=combinedTrace.childParentCarrierMeasureM3.size()||
       combinedTrace.childParentCarrierRows.size()!=combinedTrace.childParentCarrierDistanceM.size())
        return fail("paired carrier trace is not a valid CSR prefix");
    std::vector<std::uint32_t> mappedRows;std::vector<double> masses,measures,distances;
    for(auto row=offsets[roleEdge];row<offsets[roleEdge+1u];++row){
        const auto local=roleTrace.childParentCarrierRows[row];
        if(local>=roleVirtualToCurrent.size())return fail("role carrier row is outside its paired virtual-parent map");
        const auto mapped=roleVirtualToCurrent[local];
        if(!mappedRows.empty()&&mapped<=mappedRows.back())
            return fail("mapped role carrier rows are not strictly ordered");
        mappedRows.push_back(mapped);masses.push_back(roleTrace.childParentCarrierMassKg[row]);
        measures.push_back(roleTrace.childParentCarrierMeasureM3[row]);
        distances.push_back(roleTrace.childParentCarrierDistanceM[row]);
    }
    combinedTrace.childParentCarrierRows.insert(combinedTrace.childParentCarrierRows.end(),mappedRows.begin(),mappedRows.end());
    combinedTrace.childParentCarrierMassKg.insert(combinedTrace.childParentCarrierMassKg.end(),masses.begin(),masses.end());
    combinedTrace.childParentCarrierMeasureM3.insert(combinedTrace.childParentCarrierMeasureM3.end(),measures.begin(),measures.end());
    combinedTrace.childParentCarrierDistanceM.insert(combinedTrace.childParentCarrierDistanceM.end(),distances.begin(),distances.end());
    combinedTrace.childParentCarrierOffsets.push_back(static_cast<std::uint32_t>(combinedTrace.childParentCarrierRows.size()));
    error.clear();return true;
}

using ParentCellKey=std::array<std::uint32_t,4>;
using GroupKey=spatial_mass_transport_detail::GroupKey;
using SourceKey=std::pair<std::uint32_t,std::uint32_t>;

inline bool evenTetNodePermutation(
        const std::array<std::uint32_t,4>& parent,
        const std::array<std::uint32_t,4>& child) {
    std::array<unsigned,4> permutation{};
    for(unsigned i=0;i<4u;++i) {
        const auto found=std::find(parent.begin(),parent.end(),child[i]);
        if(found==parent.end())return false;
        permutation[i]=static_cast<unsigned>(found-parent.begin());
    }
    unsigned inversions=0u;
    for(unsigned i=0;i<4u;++i)
        for(unsigned j=i+1u;j<4u;++j)
            inversions+=permutation[i]>permutation[j];
    return (inversions&1u)==0u;
}

inline bool exactIdentityParentRows(
        const PhysicalSkinChildMassMappingBundle& bundle,
        const std::span<const SpatialMassTransportPoint> parentCentroids,
        const std::span<const SpatialMassTransportPoint> childCentroids,
        const double maximumTransportDistanceM,
        std::vector<std::uint32_t>& childParentRows) {
    const auto& parent=bundle.parent;
    const auto parentCount=parent.tetrahedra.size();
    const auto childCount=bundle.childTetrahedra.size();
    const auto nodeCount=bundle.referenceWorldPoints.size();
    if(parentCount==0u||parentCount!=childCount||
       parentCentroids.size()!=parentCount||childCentroids.size()!=childCount||
       bundle.currentWorldPoints.size()!=nodeCount||
       parent.currentWorldPoints.size()!=nodeCount||
       parent.referenceWorldPoints.size()!=nodeCount)
        return false;
    // Exact-topology transport has two explicit coordinate contracts:
    // preserve the parent's reference frame, or rebase child reference onto the
    // parent's current frame. In either case, child current coordinates remain
    // the exact parent-current rows and the parent current/reference arrays are
    // retained separately for centroid, density, and receipt checks.
    bool referenceMatchesParentReference=true;
    bool referenceMatchesParentCurrent=true;
    bool currentMatchesParentCurrent=true;
    bool currentCompilesToParentCurrent=true;
    for(std::size_t node=0;node<nodeCount;++node) {
        currentMatchesParentCurrent=currentMatchesParentCurrent&&
            bundle.currentWorldPoints[node]==parent.currentWorldPoints[node];
        for(unsigned axis=0;axis<3u;++axis) {
            const double emitted=static_cast<double>(static_cast<float>(bundle.currentWorldPoints[node][axis]));
            currentCompilesToParentCurrent=currentCompilesToParentCurrent&&
                std::isfinite(emitted)&&emitted==parent.currentWorldPoints[node][axis];
        }
        referenceMatchesParentReference=referenceMatchesParentReference&&
            bundle.referenceWorldPoints[node]==parent.referenceWorldPoints[node];
        referenceMatchesParentCurrent=referenceMatchesParentCurrent&&
            bundle.referenceWorldPoints[node]==parent.currentWorldPoints[node];
    }
    if(!referenceMatchesParentReference&&!referenceMatchesParentCurrent)
        return false;
    // Exact native source doubles can differ from their compiled GPU current
    // positions. This representation contract is allowed only when every
    // current row emits the captured parent bits and reference remains exact.
    if(!currentMatchesParentCurrent&&
       !(referenceMatchesParentReference&&currentCompilesToParentCurrent))
        return false;

    std::map<ParentCellKey,std::uint32_t> parentByNodes;
    for(std::size_t row=0;row<parentCount;++row) {
        auto key=parent.tetrahedra[row];
        std::sort(key.begin(),key.end());
        if(key.front()==key.back()||
           std::adjacent_find(key.begin(),key.end())!=key.end()||
           !parentByNodes.emplace(key,static_cast<std::uint32_t>(row)).second)
            return false;
    }
    std::vector<std::uint32_t> rows(childCount,NM_INVALID_INDEX);
    std::vector<std::uint8_t> usedParent(parentCount,0u);
    for(std::size_t child=0;child<childCount;++child) {
        auto key=bundle.childTetrahedra[child];
        std::sort(key.begin(),key.end());
        if(key.front()==key.back()||std::adjacent_find(key.begin(),key.end())!=key.end())
            return false;
        const auto found=parentByNodes.find(key);
        if(found==parentByNodes.end()||usedParent[found->second])
            return false;
        const auto parentRow=found->second;
        if(!evenTetNodePermutation(parent.tetrahedra[parentRow],bundle.childTetrahedra[child])||
           parent.materialIndices[parentRow]!=bundle.childMaterialIndices[child]||
           parent.donorBodyIndices[parentRow]!=bundle.childDonorBodyIndices[child]||
           parent.anatomicalRegionIndices[parentRow]!=bundle.childAnatomicalRegionIndices[child]||
           parent.stableSourceFaceIdentities[parentRow]!=bundle.childStableSourceFaceIdentities[child])
            return false;
        const auto distance=spatial_mass_transport_detail::distance(
            parentCentroids[parentRow],childCentroids[child]);
        if(!std::isfinite(distance)||distance>maximumTransportDistanceM)
            return false;
        usedParent[parentRow]=1u;
        rows[child]=parentRow;
    }
    if(std::any_of(usedParent.begin(),usedParent.end(),[](std::uint8_t used){return used==0u;}))
        return false;
    childParentRows=std::move(rows);
    return true;
}

inline bool buildExactIdentityTransport(
        const ConservativeSpatialMassTransportInput& input,
        const std::span<const std::uint32_t> childParentRows,
        ConservativeSpatialMassTransportResult& output,
        std::string& error) {
    using namespace spatial_mass_transport_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const auto parentCount=input.parentMassKg.size();
    const auto childCount=input.childReferenceVolumeM3.size();
    if(parentCount==0u||childCount==0u||childParentRows.size()!=childCount||
       childCount>input.maximumAllowedEdges||childCount>input.maximumAugmentations||
       input.parentSourceDonorOffsets.size()!=parentCount+1u||
       input.parentSourceDonorOffsets.empty()||
       input.parentSourceDonorOffsets.front()!=0u||
       input.parentSourceDonorOffsets.back()!=input.parentSourceDonorBodyIndices.size()||
       input.parentSourceDonorBodyIndices.size()!=input.parentSourceDonorMassKg.size())
        return fail("exact identity transport arrays or pinned edge/augmentation budget are invalid");

    ConservativeSpatialMassTransportResult next;
    next.method=UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport;
    next.parentMassPacketSHA256=std::string(input.parentMassPacketSHA256);
    next.parentSourceContributionSHA256=std::string(input.parentSourceContributionSHA256);
    next.childMeshSHA256=std::string(input.childMeshSHA256);
    next.regionLabelSHA256=std::string(input.regionLabelSHA256);
    next.coordinateFrameSHA256=std::string(input.coordinateFrameSHA256);
    next.maximumTransportDistanceM=input.maximumTransportDistanceM;
    next.relativeClosureTolerance=input.relativeClosureTolerance;
    next.absoluteMassToleranceKg=input.absoluteMassToleranceKg;
    next.absoluteMeasureToleranceM3=input.absoluteMeasureToleranceM3;
    next.maximumAllowedEdges=input.maximumAllowedEdges;
    next.maximumAugmentations=input.maximumAugmentations;
    hashInputArrays(input,next.parentDataSHA256,next.childDataSHA256,next.materialBoundsSHA256);

    next.childParentOffsets.assign(childCount+1u,0u);
    next.childParentRows.reserve(childCount);
    next.childParentTransportMassKg.reserve(childCount);
    next.childParentTransportMeasureM3.reserve(childCount);
    next.childParentTransportDistanceM.reserve(childCount);
    next.childParentVolumeFractions.reserve(childCount);
    next.parentMassTransferredKg.assign(parentCount,0.0);
    next.parentMeasureTransferredM3.assign(parentCount,0.0);
    next.childAncestryMeasureVolumeM3.assign(childCount,0.0);
    next.childMassKg.assign(childCount,0.0);
    next.childImpliedDensityKgM3.assign(childCount,0.0);
    next.childDominantParentRows.assign(childCount,NM_INVALID_INDEX);
    next.childSourceDonorOffsets.assign(childCount+1u,0u);

    std::map<GroupKey,ConservativeSpatialMassTransportGroupClosure> groups;
    std::map<SourceKey,ConservativeSpatialMassTransportSourceDonorClosure> sourceGroups;
    long double weightedDistance=0.0L,totalMass=0.0L;
    next.minimumChildImpliedDensityKgM3=std::numeric_limits<double>::infinity();
    next.maximumChildImpliedDensityKgM3=0.0;
    for(std::size_t parent=0;parent<parentCount;++parent) {
        const auto material=input.parentMaterialIndices[parent];
        const GroupKey key{material,input.parentDonorBodyIndices[parent],
                           input.parentAnatomicalRegionIndices[parent]};
        auto& group=groups[key];
        group.materialIndex=std::get<0>(key);
        group.donorBodyIndex=std::get<1>(key);
        group.anatomicalRegionIndex=std::get<2>(key);
        group.parentMassKg+=input.parentMassKg[parent];
        for(std::uint32_t source=input.parentSourceDonorOffsets[parent];
            source<input.parentSourceDonorOffsets[parent+1u];++source) {
            const SourceKey sourceKey{material,input.parentSourceDonorBodyIndices[source]};
            auto& closure=sourceGroups[sourceKey];
            closure.materialIndex=sourceKey.first;
            closure.sourceDonorBodyIndex=sourceKey.second;
            closure.parentMassKg+=input.parentSourceDonorMassKg[source];
        }
    }

    for(std::size_t child=0;child<childCount;++child) {
        const auto parent=childParentRows[child];
        if(parent>=parentCount||next.parentMassTransferredKg[parent]!=0.0)
            return fail("exact identity transport parent rows are not a bijection");
        const auto material=input.childMaterialIndices[child];
        const double mass=input.parentMassKg[parent];
        const double parentDensity=mass/input.parentReferenceVolumeM3[parent];
        const double measure=mass/parentDensity;
        const double distance=spatial_mass_transport_detail::distance(
            input.parentCentroidsWorld[parent],input.childCentroidsWorld[child]);
        const double density=mass/input.childReferenceVolumeM3[child];
        const double lower=input.materialDensityLowerKgM3[material];
        const double upper=input.materialDensityUpperKgM3[material];
        if(!std::isfinite(mass)||mass<=0.0||!std::isfinite(parentDensity)||
           parentDensity<=0.0||!std::isfinite(measure)||measure<=0.0||
           !std::isfinite(distance)||distance>input.maximumTransportDistanceM||
           !std::isfinite(density)||!std::isfinite(lower)||!std::isfinite(upper)||
           lower<=0.0||lower>upper||
           density<lower-densityBoundRoundoff(lower,upper,density)||
           density>upper+densityBoundRoundoff(lower,upper,density))
            return fail("exact identity transport fails current centroid, source density, child density, or pinned distance checks");

        const auto edge=static_cast<std::uint32_t>(next.childParentRows.size());
        next.childParentOffsets[child]=edge;
        next.childParentRows.push_back(parent);
        next.childParentTransportMassKg.push_back(mass);
        next.childParentTransportMeasureM3.push_back(measure);
        next.childParentTransportDistanceM.push_back(distance);
        next.childParentVolumeFractions.push_back(1.0);
        next.parentMassTransferredKg[parent]=mass;
        next.parentMeasureTransferredM3[parent]=measure;
        next.childAncestryMeasureVolumeM3[child]=measure;
        next.childMassKg[child]=mass;
        next.childImpliedDensityKgM3[child]=density;
        next.childDominantParentRows[child]=parent;
        next.maximumTransportDistanceUsedM=std::max(next.maximumTransportDistanceUsedM,distance);
        weightedDistance+=static_cast<long double>(mass)*distance;
        totalMass+=mass;

        const GroupKey groupKey{material,input.childDonorBodyIndices[child],
                                input.childAnatomicalRegionIndices[child]};
        auto foundGroup=groups.find(groupKey);
        if(foundGroup==groups.end())
            return fail("exact identity transport child group has no exact parent supply");
        foundGroup->second.childMassKg+=mass;
        foundGroup->second.childMinimumCapacityKg+=
            input.childReferenceVolumeM3[child]*lower;
        foundGroup->second.childMaximumCapacityKg+=
            input.childReferenceVolumeM3[child]*upper;

        next.childSourceDonorOffsets[child]=
            static_cast<std::uint32_t>(next.childSourceDonorBodyIndices.size());
        double sourceMassForChild=0.0;
        for(std::uint32_t source=input.parentSourceDonorOffsets[parent];
            source<input.parentSourceDonorOffsets[parent+1u];++source) {
            const auto donor=input.parentSourceDonorBodyIndices[source];
            const auto sourceMass=input.parentSourceDonorMassKg[source];
            next.childSourceDonorBodyIndices.push_back(donor);
            next.childSourceDonorMassKg.push_back(sourceMass);
            sourceMassForChild+=sourceMass;
            auto& closure=sourceGroups[{material,donor}];
            closure.childMassKg+=sourceMass;
        }
        next.childSourceDonorOffsets[child+1u]=
            static_cast<std::uint32_t>(next.childSourceDonorBodyIndices.size());
        if(std::abs(sourceMassForChild-mass)>
           std::max(input.absoluteMassToleranceKg,input.relativeClosureTolerance*mass))
            return fail("exact identity transport source-donor rows do not close child mass");
        next.minimumChildImpliedDensityKgM3=
            std::min(next.minimumChildImpliedDensityKgM3,density);
        next.maximumChildImpliedDensityKgM3=
            std::max(next.maximumChildImpliedDensityKgM3,density);
    }
    next.childParentOffsets[childCount]=
        static_cast<std::uint32_t>(next.childParentRows.size());
    next.transportEdgeCount=next.childParentRows.size();
    next.augmentationCount=childCount; // one direct exact assignment per child cell
    next.massWeightedMeanTransportDistanceM=totalMass>0.0L?
        static_cast<double>(weightedDistance/totalMass):0.0L;
    next.maximumChildFractionSumError=0.0;
    for(const auto& [key,group]:groups) {
        (void)key;
        next.groupClosure.push_back(group);
    }
    for(const auto& [key,source]:sourceGroups) {
        (void)key;
        next.sourceDonorClosure.push_back(source);
        const auto errorMass=std::abs(source.parentMassKg-source.childMassKg);
        next.maximumSourceDonorMassClosureErrorKg=
            std::max(next.maximumSourceDonorMassClosureErrorKg,errorMass);
    }
    for(std::size_t parent=0;parent<parentCount;++parent) {
        next.maximumParentMassClosureErrorKg=std::max(next.maximumParentMassClosureErrorKg,
            std::abs(next.parentMassTransferredKg[parent]-input.parentMassKg[parent]));
        next.maximumParentMeasureClosureErrorM3=std::max(next.maximumParentMeasureClosureErrorM3,
            std::abs(next.parentMeasureTransferredM3[parent]-input.parentReferenceVolumeM3[parent]));
    }
    next.receiptSHA256=receiptSHA256(input,next);
    output=std::move(next);
    error.clear();
    return true;
}

inline std::string operationalChildCompiledDataSHA256(
        const PhysicalSkinChildMassMappingBundle& bundle) {
    using spatial_mass_transport_detail::SHA256Builder;
    SHA256Builder hash;hash.string("physical-skin-operational-child-compiled-data.v1");
    hash.u64(bundle.currentWorldPoints.size());
    for(const auto& point:bundle.currentWorldPoints)for(const auto value:point)hash.real(value);
    hash.u64(bundle.referenceWorldPoints.size());
    for(const auto& point:bundle.referenceWorldPoints)for(const auto value:point)hash.real(value);
    hash.u64(bundle.childTetrahedra.size());
    for(const auto& tet:bundle.childTetrahedra)for(const auto node:tet)hash.u32(node);
    const auto hashIndices=[&](const char* name,const auto& rows){
        hash.string(name);hash.u64(rows.size());for(const auto value:rows)hash.u32(value);
    };
    hashIndices("material",bundle.childMaterialIndices);
    hashIndices("geometric_donor",bundle.childDonorBodyIndices);
    hashIndices("region",bundle.childAnatomicalRegionIndices);
    hashIndices("stable_source_face",bundle.childStableSourceFaceIdentities);
    return hash.finish();
}

inline std::string verifiedConformingMassRouteSHA256(
        const PhysicalSkinVerifiedConformingMassRoute& route) {
    using spatial_mass_transport_detail::SHA256Builder;
    SHA256Builder hash;
    hash.string("physical-skin-verified-conforming-mass-route.v1");
    hash.u32(route.authenticated?1u:0u);hash.u64(route.refinedSourceCellCount);
    for(const auto& value:{route.parentMassPacketSHA256,route.parentSourceContributionSHA256,
                           route.sourceVolumeSHA256,route.refinedVolumeSHA256,
                           route.candidateVolumeSHA256,route.refinementVerificationSHA256,
                           route.sourceDonorLedgerVerificationSHA256,
                           route.pullbackVerificationSHA256,route.chainVerificationSHA256,
                           route.candidateOperationalReferenceDomainSHA256,
                           route.candidateCompiledDataSHA256}) hash.string(value);
    const auto hashU32=[&](const char* name,const auto& values){
        hash.string(name);hash.u64(values.size());for(const auto value:values)hash.u32(value);
    };
    const auto hashDouble=[&](const char* name,const auto& values){
        hash.string(name);hash.u64(values.size());for(const auto value:values)hash.real(value);
    };
    hashU32("candidate_to_refined_rows",route.childRefinedSourceRows);
    hashU32("candidate_to_immutable_parent_rows",route.childImmutableParentRows);
    hashDouble("child_immutable_parent_fractions",route.childImmutableParentFractions);
    hashDouble("child_immutable_parent_reference_measure_m3",route.childImmutableParentReferenceMeasureM3);
    hashDouble("child_mass_kg",route.childMassKg);
    hashU32("child_source_donor_offsets",route.childSourceDonorOffsets);
    hashU32("child_source_donor_body_indices",route.childSourceDonorBodyIndices);
    hashDouble("child_source_donor_mass_kg",route.childSourceDonorMassKg);
    return hash.finish();
}

// Construct a one-edge-per-child record from a receipt-authenticated exact
// refinement lineage. The generic spatial assembler remains the fallback for
// bundles without this explicit chain; every edge still runs through the same
// recorded-flow validator and frozen centroid-distance policy.
inline bool buildVerifiedConformingAncestryTransport(
        const ConservativeSpatialMassTransportInput& input,
        const PhysicalSkinChildMassMappingBundle& bundle,
        ConservativeSpatialMassTransportResult& output,std::string& error) {
    using namespace spatial_mass_transport_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const auto& route=bundle.conformingLineageMassRoute;
    const auto parentCount=input.parentMassKg.size(),childCount=input.childReferenceVolumeM3.size();
    if(!bundle.conformingLineageMassRouteRequired||!bundle.usesSeparatedOperationalReference||
       !route.authenticated||parentCount==0u||childCount==0u||
       route.refinedSourceCellCount!=childCount||
       route.childRefinedSourceRows.size()!=childCount||
       route.childImmutableParentRows.size()!=childCount||
       route.childImmutableParentFractions.size()!=childCount||
       route.childImmutableParentReferenceMeasureM3.size()!=childCount||
       route.childMassKg.size()!=childCount||route.childSourceDonorOffsets.size()!=childCount+1u||
       route.childSourceDonorOffsets.empty()||route.childSourceDonorOffsets.front()!=0u||
       route.childSourceDonorOffsets.back()!=route.childSourceDonorBodyIndices.size()||
       route.childSourceDonorBodyIndices.size()!=route.childSourceDonorMassKg.size())
        return fail("verified conforming ancestry route is incomplete or detached from the operational-reference import");
    for(const auto& value:{route.parentMassPacketSHA256,route.parentSourceContributionSHA256,
                          route.sourceVolumeSHA256,route.refinedVolumeSHA256,
                          route.candidateVolumeSHA256,route.refinementVerificationSHA256,
                          route.sourceDonorLedgerVerificationSHA256,route.pullbackVerificationSHA256,
                          route.chainVerificationSHA256,route.candidateOperationalReferenceDomainSHA256,
                          route.candidateCompiledDataSHA256})
        if(!validSHA256(value))return fail("verified conforming ancestry route is missing a pinned source or proof identity");
    if(route.parentMassPacketSHA256!=input.parentMassPacketSHA256||
       route.parentSourceContributionSHA256!=input.parentSourceContributionSHA256||
       route.candidateOperationalReferenceDomainSHA256!=bundle.operationalReferenceDomainSHA256||
       route.candidateCompiledDataSHA256!=operationalChildCompiledDataSHA256(bundle)||
       route.verificationSHA256!=verifiedConformingMassRouteSHA256(route))
        return fail("verified conforming ancestry route proof or immutable source packet changed");
    if(input.parentCentroidsWorld.size()!=parentCount||input.parentReferenceVolumeM3.size()!=parentCount||
       input.parentMaterialIndices.size()!=parentCount||input.parentDonorBodyIndices.size()!=parentCount||
       input.parentAnatomicalRegionIndices.size()!=parentCount||
       input.parentSourceDonorOffsets.size()!=parentCount+1u||
       input.parentSourceDonorOffsets.empty()||input.parentSourceDonorOffsets.front()!=0u||
       input.parentSourceDonorOffsets.back()!=input.parentSourceDonorBodyIndices.size()||
       input.parentSourceDonorBodyIndices.size()!=input.parentSourceDonorMassKg.size()||
       input.childCentroidsWorld.size()!=childCount||input.childMaterialIndices.size()!=childCount||
       input.childDonorBodyIndices.size()!=childCount||input.childAnatomicalRegionIndices.size()!=childCount||
       input.materialDensityLowerKgM3.empty()||
       input.materialDensityUpperKgM3.size()!=input.materialDensityLowerKgM3.size()||
       bundle.parent.materialIndices.size()!=parentCount||
       bundle.parent.donorBodyIndices.size()!=parentCount||
       bundle.parent.anatomicalRegionIndices.size()!=parentCount||
       bundle.parent.stableSourceFaceIdentities.size()!=parentCount||
       bundle.childStableSourceFaceIdentities.size()!=childCount)
        return fail("verified conforming ancestry route does not match current mass-mapping input domains");

    const auto withinMass=[&](const double a,const double b){
        return std::isfinite(a)&&std::isfinite(b)&&
            std::abs(a-b)<=std::max(input.absoluteMassToleranceKg,
                input.relativeClosureTolerance*std::max(std::abs(a),std::abs(b)));
    };
    const auto withinMeasure=[&](const double a,const double b){
        return std::isfinite(a)&&std::isfinite(b)&&
            std::abs(a-b)<=std::max(input.absoluteMeasureToleranceM3,
                input.relativeClosureTolerance*std::max(std::abs(a),std::abs(b)));
    };
    std::vector<std::uint8_t> refinedRowsSeen(childCount,0u);
    ConservativeSpatialMassTransportResult next;
    next.method=UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport;
    next.parentMassPacketSHA256=std::string(input.parentMassPacketSHA256);
    next.parentSourceContributionSHA256=std::string(input.parentSourceContributionSHA256);
    next.childMeshSHA256=std::string(input.childMeshSHA256);
    next.regionLabelSHA256=std::string(input.regionLabelSHA256);
    next.coordinateFrameSHA256=std::string(input.coordinateFrameSHA256);
    next.maximumTransportDistanceM=input.maximumTransportDistanceM;
    next.relativeClosureTolerance=input.relativeClosureTolerance;
    next.absoluteMassToleranceKg=input.absoluteMassToleranceKg;
    next.absoluteMeasureToleranceM3=input.absoluteMeasureToleranceM3;
    next.maximumAllowedEdges=input.maximumAllowedEdges;
    next.maximumAugmentations=input.maximumAugmentations;
    if(childCount>input.maximumAllowedEdges||childCount>input.maximumAugmentations)
        return fail("verified conforming ancestry edge count exceeds the existing pinned mass-transport budget");
    hashInputArrays(input,next.parentDataSHA256,next.childDataSHA256,next.materialBoundsSHA256);
    next.childParentOffsets.assign(childCount+1u,0u);
    next.parentMassTransferredKg.assign(parentCount,0.0);
    next.parentMeasureTransferredM3.assign(parentCount,0.0);
    next.childAncestryMeasureVolumeM3.assign(childCount,0.0);
    next.childMassKg.assign(childCount,0.0);
    next.childImpliedDensityKgM3.assign(childCount,0.0);
    next.childDominantParentRows.assign(childCount,kInvalid);
    next.childSourceDonorOffsets.assign(childCount+1u,0u);

    std::map<GroupKey,ConservativeSpatialMassTransportGroupClosure> groups;
    std::map<SourceKey,ConservativeSpatialMassTransportSourceDonorClosure> sourceGroups;
    for(std::size_t parent=0u;parent<parentCount;++parent){
        const auto material=input.parentMaterialIndices[parent];
        if(material>=input.materialDensityLowerKgM3.size()||
           material>=input.materialDensityUpperKgM3.size()||
           parent>=input.parentDonorBodyIndices.size()||parent>=input.parentAnatomicalRegionIndices.size()||
           !std::isfinite(input.materialDensityLowerKgM3[material])||
           !std::isfinite(input.materialDensityUpperKgM3[material])||
           input.materialDensityLowerKgM3[material]<=0.0||
           input.materialDensityLowerKgM3[material]>input.materialDensityUpperKgM3[material]||
           !std::isfinite(input.parentReferenceVolumeM3[parent])||input.parentReferenceVolumeM3[parent]<=0.0||
           !std::isfinite(input.parentMassKg[parent])||input.parentMassKg[parent]<=0.0)
            return fail("verified conforming ancestry source parent row or material bounds are invalid");
        const GroupKey groupKey{material,input.parentDonorBodyIndices[parent],
                                input.parentAnatomicalRegionIndices[parent]};
        auto& group=groups[groupKey];group.materialIndex=std::get<0>(groupKey);
        group.donorBodyIndex=std::get<1>(groupKey);group.anatomicalRegionIndex=std::get<2>(groupKey);
        group.parentMassKg+=input.parentMassKg[parent];
        const auto begin=input.parentSourceDonorOffsets[parent],end=input.parentSourceDonorOffsets[parent+1u];
        if(begin>=end||end>input.parentSourceDonorBodyIndices.size())
            return fail("verified conforming ancestry source parent donor split is empty or invalid");
        std::uint32_t prior=0u;double total=0.0;
        for(std::uint32_t row=begin;row<end;++row){
            const auto donor=input.parentSourceDonorBodyIndices[row];
            const double mass=input.parentSourceDonorMassKg[row];
            if((row>begin&&donor<=prior)||!std::isfinite(mass)||mass<=0.0)
                return fail("verified conforming ancestry source parent donor split is not canonical");
            prior=donor;total+=mass;
            auto& closure=sourceGroups[{material,donor}];
            closure.materialIndex=material;closure.sourceDonorBodyIndex=donor;
            closure.parentMassKg+=mass;
        }
        if(!withinMass(total,input.parentMassKg[parent]))
            return fail("verified conforming ancestry source parent donor split does not close");
    }

    long double weightedDistance=0.0L,totalMass=0.0L;
    next.minimumChildImpliedDensityKgM3=std::numeric_limits<double>::infinity();
    next.maximumChildImpliedDensityKgM3=0.0;
    for(std::size_t child=0u;child<childCount;++child){
        const auto refinedRow=route.childRefinedSourceRows[child];
        const auto parent=route.childImmutableParentRows[child];
        const double fraction=route.childImmutableParentFractions[child];
        const double mass=route.childMassKg[child];
        if(refinedRow>=childCount||refinedRowsSeen[refinedRow]!=0u||parent>=parentCount||
           !std::isfinite(fraction)||fraction<=0.0||fraction>1.0||
           !std::isfinite(mass)||mass<=0.0)
            return fail("verified conforming ancestry child row or immutable parent fraction is invalid");
        refinedRowsSeen[refinedRow]=1u;
        const auto material=input.childMaterialIndices[child];
        if(material>=input.materialDensityLowerKgM3.size()||
           material>=input.materialDensityUpperKgM3.size()||
           !std::isfinite(input.materialDensityLowerKgM3[material])||
           !std::isfinite(input.materialDensityUpperKgM3[material])||
           input.materialDensityLowerKgM3[material]<=0.0||
           input.materialDensityLowerKgM3[material]>input.materialDensityUpperKgM3[material])
            return fail("verified conforming ancestry child material density bounds are missing or invalid");
        if(material!=input.parentMaterialIndices[parent]||
           input.childDonorBodyIndices[child]!=input.parentDonorBodyIndices[parent]||
           input.childAnatomicalRegionIndices[child]!=input.parentAnatomicalRegionIndices[parent]||
           bundle.childStableSourceFaceIdentities[child]!=bundle.parent.stableSourceFaceIdentities[parent])
            return fail("verified conforming ancestry edge changes material, geometric donor, region, or stable source face");
        const double sourceMass=input.parentMassKg[parent];
        const double sourceVolume=input.parentReferenceVolumeM3[parent];
        const double expectedMass=sourceMass*fraction;
        const double expectedImmutableMeasure=sourceVolume*fraction;
        if(!withinMass(mass,expectedMass)||
           !withinMeasure(route.childImmutableParentReferenceMeasureM3[child],expectedImmutableMeasure))
            return fail("verified conforming ancestry mass or immutable reference measure differs from its 330 parent fraction");
        const double parentDensity=sourceMass/sourceVolume;
        const double measure=mass/parentDensity;
        if(!std::isfinite(parentDensity)||parentDensity<=0.0||!withinMeasure(measure,expectedImmutableMeasure))
            return fail("verified conforming ancestry mass does not reproduce its immutable 330 reference measure");
        const double distance=spatial_mass_transport_detail::distance(
            input.parentCentroidsWorld[parent],input.childCentroidsWorld[child]);
        if(!std::isfinite(distance)||distance>input.maximumTransportDistanceM)
            return fail("verified conforming ancestry edge exceeds the existing frozen centroid-distance bound at child "+
                std::to_string(child));
        const double lower=input.materialDensityLowerKgM3[material];
        const double upper=input.materialDensityUpperKgM3[material];
        const double density=mass/input.childReferenceVolumeM3[child];
        const double densityTol=densityBoundRoundoff(lower,upper,density);
        if(!std::isfinite(density)||density<lower-densityTol||density>upper+densityTol)
            return fail("verified conforming ancestry child mass is outside the actual material density bounds");
        const auto donorBegin=route.childSourceDonorOffsets[child];
        const auto donorEnd=route.childSourceDonorOffsets[child+1u];
        const auto parentDonorBegin=input.parentSourceDonorOffsets[parent];
        const auto parentDonorEnd=input.parentSourceDonorOffsets[parent+1u];
        if(donorBegin>=donorEnd||donorEnd>route.childSourceDonorBodyIndices.size()||
           donorEnd-donorBegin!=parentDonorEnd-parentDonorBegin)
            return fail("verified conforming ancestry 534 donor shares do not match the immutable 330 parent row");
        double donorSum=0.0;std::uint32_t priorDonor=0u;
        for(std::uint32_t row=donorBegin;row<donorEnd;++row){
            const auto parentSource=parentDonorBegin+(row-donorBegin);
            const auto donor=route.childSourceDonorBodyIndices[row];
            const double donorMass=route.childSourceDonorMassKg[row];
            const double expectedDonorMass=input.parentSourceDonorMassKg[parentSource]*fraction;
            if(donor!=input.parentSourceDonorBodyIndices[parentSource]||
               (row>donorBegin&&donor<=priorDonor)||!std::isfinite(donorMass)||donorMass<=0.0||
               !withinMass(donorMass,expectedDonorMass))
                return fail("verified conforming ancestry 534 donor share differs from immutable 330 source contribution");
            priorDonor=donor;donorSum+=donorMass;
        }
        if(!withinMass(donorSum,mass))
            return fail("verified conforming ancestry child donor shares do not close child mass");

        next.childParentOffsets[child]=static_cast<std::uint32_t>(next.childParentRows.size());
        next.childParentRows.push_back(parent);
        next.childParentTransportMassKg.push_back(mass);
        next.childParentTransportMeasureM3.push_back(measure);
        next.childParentTransportDistanceM.push_back(distance);
        next.childParentVolumeFractions.push_back(1.0);
        next.parentMassTransferredKg[parent]+=mass;
        next.parentMeasureTransferredM3[parent]+=measure;
        next.childAncestryMeasureVolumeM3[child]=measure;
        next.childMassKg[child]=mass;
        next.childImpliedDensityKgM3[child]=density;
        next.childDominantParentRows[child]=parent;
        next.childSourceDonorOffsets[child]=static_cast<std::uint32_t>(next.childSourceDonorBodyIndices.size());
        for(std::uint32_t row=donorBegin;row<donorEnd;++row){
            const auto donor=route.childSourceDonorBodyIndices[row];
            const double donorMass=route.childSourceDonorMassKg[row];
            next.childSourceDonorBodyIndices.push_back(donor);
            next.childSourceDonorMassKg.push_back(donorMass);
            sourceGroups[{material,donor}].childMassKg+=donorMass;
        }
        next.childSourceDonorOffsets[child+1u]=static_cast<std::uint32_t>(next.childSourceDonorBodyIndices.size());
        auto group=groups.find({material,input.childDonorBodyIndices[child],
                                input.childAnatomicalRegionIndices[child]});
        if(group==groups.end())return fail("verified conforming ancestry child has no immutable parent mass group");
        group->second.childMassKg+=mass;
        group->second.childMinimumCapacityKg+=input.childReferenceVolumeM3[child]*lower;
        group->second.childMaximumCapacityKg+=input.childReferenceVolumeM3[child]*upper;
        next.minimumChildImpliedDensityKgM3=std::min(next.minimumChildImpliedDensityKgM3,density);
        next.maximumChildImpliedDensityKgM3=std::max(next.maximumChildImpliedDensityKgM3,density);
        next.maximumTransportDistanceUsedM=std::max(next.maximumTransportDistanceUsedM,distance);
        weightedDistance+=static_cast<long double>(mass)*distance;totalMass+=mass;
    }
    if(std::any_of(refinedRowsSeen.begin(),refinedRowsSeen.end(),[](std::uint8_t seen){return seen!=1u;}))
        return fail("verified conforming ancestry candidate does not cover every 530 refined source row exactly once");
    next.childParentOffsets[childCount]=static_cast<std::uint32_t>(next.childParentRows.size());
    next.childSourceDonorOffsets[childCount]=static_cast<std::uint32_t>(next.childSourceDonorBodyIndices.size());
    next.transportEdgeCount=next.childParentRows.size();
    next.augmentationCount=childCount; // one authenticated lineage edge per candidate cell
    next.massWeightedMeanTransportDistanceM=totalMass>0.0L?
        static_cast<double>(weightedDistance/totalMass):0.0;
    next.maximumChildFractionSumError=0.0;
    for(const auto& [key,value]:groups){(void)key;next.groupClosure.push_back(value);}
    for(const auto& [key,value]:sourceGroups){(void)key;next.sourceDonorClosure.push_back(value);}
    for(std::size_t parent=0u;parent<parentCount;++parent){
        next.maximumParentMassClosureErrorKg=std::max(next.maximumParentMassClosureErrorKg,
            std::abs(next.parentMassTransferredKg[parent]-input.parentMassKg[parent]));
        next.maximumParentMeasureClosureErrorM3=std::max(next.maximumParentMeasureClosureErrorM3,
            std::abs(next.parentMeasureTransferredM3[parent]-input.parentReferenceVolumeM3[parent]));
    }
    for(const auto& closure:next.sourceDonorClosure)
        next.maximumSourceDonorMassClosureErrorKg=std::max(next.maximumSourceDonorMassClosureErrorKg,
            std::abs(closure.parentMassKg-closure.childMassKg));
    next.receiptSHA256=receiptSHA256(input,next);
    output=std::move(next);error.clear();return true;
}

inline std::string massMappingPathSHA256(
        const PhysicalSkinChildMassMappingBundle& bundle,
        const bool exactTopologyIdentity,
        const bool verifiedConformingAncestry=false) {
    using spatial_mass_transport_detail::SHA256Builder;
    SHA256Builder hash;
    const auto hashPoints=[&](const std::string_view name,const std::vector<V3>& points) {
        hash.string(name);hash.u64(points.size());
        for(const auto& point:points)for(const auto value:point)hash.real(value);
    };
    const auto hashIndices=[&](const std::string_view name,const std::vector<std::uint32_t>& values) {
        hash.string(name);hash.u64(values.size());
        for(const auto value:values)hash.u32(value);
    };
    const auto hashScalars=[&](const std::string_view name,const std::vector<double>& values) {
        hash.string(name);hash.u64(values.size());
        for(const auto value:values)hash.real(value);
    };
    const auto hashTetrahedra=[&](const std::string_view name,
                                  const std::vector<std::array<std::uint32_t,4>>& values) {
        hash.string(name);hash.u64(values.size());
        for(const auto& tet:values)for(const auto node:tet)hash.u32(node);
    };
    const auto hashFaces=[&](const std::string_view name,
                             const std::vector<Face>& values) {
        hash.string(name);hash.u64(values.size());
        for(const auto& face:values)for(const auto node:face)hash.u32(node);
    };
    const auto hashBary3=[&](const std::string_view name,
                             const std::vector<std::array<double,3>>& values) {
        hash.string(name);hash.u64(values.size());
        for(const auto& row:values)for(const auto value:row)hash.real(value);
    };
    const auto hashBary4=[&](const std::string_view name,
                             const std::vector<std::array<double,4>>& values) {
        hash.string(name);hash.u64(values.size());
        for(const auto& row:values)for(const auto value:row)hash.real(value);
    };
    const auto hashMatrices=[&](const std::string_view name,
                                const std::vector<std::array<double,9>>& values) {
        hash.string(name);hash.u64(values.size());
        for(const auto& row:values)for(const auto value:row)hash.real(value);
    };
    const auto hashSurface=[&](const std::string_view name,const RegisteredSurface& surface) {
        hash.string(name);hash.string(surface.name);hash.string(surface.provider);
        hash.string(surface.sourceMember);hash.string(surface.sourceMeshSHA256);
        hash.string(surface.sourcePayloadSHA256);hash.u32(surface.stableIdentifier);
        hash.u32(surface.sourceBodyIndex);hash.u32(surface.sourceLayer);
        hash.u32(surface.aggregateIdentity?1u:0u);
        hashPoints("surface.donor_local",surface.donorLocalPoints);
        hashPoints("surface.world",surface.worldPoints);
        hashFaces("surface.faces",surface.faces);
        hashIndices("surface.source_vertices",surface.sourceVertexIndices);
        hashIndices("surface.source_faces",surface.sourceFaceIndices);
        hashIndices("surface.source_domains",surface.sourceDomainIndices);
        hashIndices("surface.embedding_faces",surface.sourceEmbeddingFaceIndices);
        hashBary3("surface.embedding_barycentric",surface.sourceEmbeddingBarycentric);
        hashPoints("surface.embedding_residuals",surface.sourceEmbeddingResidualWorldM);
        hashIndices("surface.donor_bodies",surface.donorBodyIndices);
        hashMatrices("surface.world_to_donor",surface.worldToDonorLocalLinear);
        hashPoints("surface.outward_normals",surface.outwardWorldNormals);
    };

    if(bundle.fractionalSemanticMassRouteRequired){
        hash.string("physicalSkinChildMassMapping.path.v4");
        hash.string("authenticated_fractional_semantic_support");
        hash.string(fractional_semantic_transport_detail::supportDigest(
            bundle.fractionalSemanticSupport.view()));
    }else if(verifiedConformingAncestry){
        hash.string("physicalSkinChildMassMapping.path.v3");
        hash.string("authenticated_conforming_ancestry");
        hash.u32(bundle.conformingLineageMassRouteRequired?1u:0u);
        hash.u32(bundle.conformingLineageMassTransportUsed?1u:0u);
        hash.string(bundle.conformingLineageMassRoute.verificationSHA256);
    }else{
        // Preserve the exact v2 hash stream for legacy identity/spatial routes.
        hash.string("physicalSkinChildMassMapping.path.v2");
        hash.string(exactTopologyIdentity?"exact_topology_identity":"bounded_conservative_spatial");
    }
    if(bundle.operationalSpatialMassRouteRequired){
        hash.string("authenticated_operational_spatial_reference.v1");
        hash.string(bundle.operationalSpatialReferenceVerificationSHA256);
        hash.string(bundle.operationalSpatialTransportReceiptSHA256);
    }
    hash.string(bundle.spatialMassTransportResult.receiptSHA256);
    if(bundle.sourceCarrierQuadrature.importedAndVerified){
        const auto& carrier=bundle.sourceCarrierQuadrature;
        hash.string("authenticated_source_carrier_quadrature.v1");
        hash.string(carrier.verificationSHA256);hash.string(carrier.sourceCarrierPacketSHA256);
        hash.string(carrier.sourceTetTableSHA256);hash.string(carrier.sourceNodeTableSHA256);
        for(const auto& value:carrier.roleBinarySHA256)hash.string(value);
        for(const auto& value:carrier.roleReceiptSHA256)hash.string(value);
        hash.u64(carrier.parentOffsets.size());for(auto value:carrier.parentOffsets)hash.u32(value);
        hash.u64(carrier.globalSourceTetRows.size());for(auto value:carrier.globalSourceTetRows)hash.u32(value);
        hash.u64(carrier.carrierWeights.size());for(auto value:carrier.carrierWeights)hash.real(value);
        for(const auto& row:carrier.carrierBarycentric)for(auto value:row)hash.real(value);
        for(auto value:carrier.witnessChildRows)hash.u32(value);
        for(auto value:carrier.witnessChildCentroidDistancesM)hash.real(value);
        hash.string(carrier.transportTrace.receiptSHA256);
        const auto& recorded=carrier.recordedAllocation;
        if(recorded.importedAndPinned){
            hash.string("authenticated_recorded_paired_carrier_flow_csv.v1");
            hash.string(recorded.allocationReceiptSHA256);
            hash.string(recorded.expandedSolverReceiptSHA256);
            hash.string(recorded.sourceCarrierFlowTraceSHA256);
            hash.string(recorded.transportReceiptSHA256);
            for(const auto& sha:recorded.csvSHA256)hash.string(sha);
            for(const auto& sha:recorded.roleReceiptSHA256)hash.string(sha);
        }
    }
    hash.string(bundle.parent.parentMassPacketSHA256);
    hash.string(bundle.parent.parentSourceContributionSHA256);
    hash.string(bundle.parent.parentTopologySHA256);
    hash.string(bundle.parent.parentLabelSHA256);
    hash.string(bundle.childMeshSHA256);
    hash.string(bundle.regionLabelSHA256);
    hash.string(bundle.coordinateFrameSHA256);
    hash.real(bundle.maximumSpatialMassTransportDistanceM);

    hashPoints("parent.current_world_points",bundle.parent.currentWorldPoints);
    hashPoints("parent.reference_world_points",bundle.parent.referenceWorldPoints);
    hashPoints("parent.node_frame_owner_local_points",bundle.parent.nodeFrameOwnerLocalPoints);
    hashIndices("parent.compiled_global_node_indices",bundle.parent.compiledGlobalNodeIndices);
    hashIndices("parent.node_frame_owner_body_indices",bundle.parent.nodeFrameOwnerBodyIndices);
    hashTetrahedra("parent.tetrahedra",bundle.parent.tetrahedra);
    hashScalars("parent.reference_tet_volumes_m3",bundle.parent.referenceTetVolumeM3);
    hashScalars("parent.tetrahedron_mass_kg",bundle.parent.tetrahedronMassKg);
    hashIndices("parent.material_indices",bundle.parent.materialIndices);
    hashIndices("parent.donor_body_indices",bundle.parent.donorBodyIndices);
    hashIndices("parent.anatomical_region_indices",bundle.parent.anatomicalRegionIndices);
    hashIndices("parent.stable_source_face_identities",bundle.parent.stableSourceFaceIdentities);
    hash.u64(bundle.parent.sourceMassContributions.size());
    for(const auto& source:bundle.parent.sourceMassContributions) {
        hash.u32(source.parentNodeIndex);hash.u32(source.materialIndex);
        hash.u32(source.donorBodyIndex);hash.real(source.massKg);
        for(const auto value:source.donorLocalPosition)hash.real(value);
    }

    hashSurface("outer_surface",bundle.outerSkin);
    hashSurface("independent_inner_surface",bundle.independentSkinInner);
    hashSurface("independent_deep_surface",bundle.independentFatDeep);
    hash.u64(bundle.boundaryMap.volumeNodeCount);
    hashIndices("boundary.outer",bundle.boundaryMap.outer);
    hashIndices("boundary.inner",bundle.boundaryMap.skinInner);
    hashIndices("boundary.deep",bundle.boundaryMap.fatDeep);

    hashPoints("child.current_world_points",bundle.currentWorldPoints);
    hashPoints("child.reference_world_points",bundle.referenceWorldPoints);
    hashTetrahedra("child.tetrahedra",bundle.childTetrahedra);
    hashIndices("child.material_indices",bundle.childMaterialIndices);
    hashIndices("child.donor_body_indices",bundle.childDonorBodyIndices);
    hashIndices("child.anatomical_region_indices",bundle.childAnatomicalRegionIndices);
    hashIndices("child.stable_source_face_identities",bundle.childStableSourceFaceIdentities);
    hashIndices("child.dominant_parent_tetrahedra",bundle.childDominantParentTetrahedra);
    hashIndices("child.parent_surface_faces",bundle.childParentSurfaceFaces);
    hashIndices("child.registered_layer_facet_surface_faces",bundle.childRegisteredLayerFacetSurfaceFaces);
    hashIndices("child.registered_layer_facet_source_faces",bundle.childRegisteredLayerFacetSourceFaceIdentities);
    std::vector<std::uint32_t> sidewallFlags(bundle.childAuthoredHeadSidewallFacetFlags.begin(),
                                              bundle.childAuthoredHeadSidewallFacetFlags.end());
    hashIndices("child.authored_head_sidewall_flags",sidewallFlags);
    hashIndices("child.authored_head_sidewall_stable_faces",bundle.childAuthoredHeadSidewallStableFaceIdentities);
    hashIndices("volume.node_donors",bundle.volumeNodeDonorBodyIndices);
    hashPoints("volume.node_donor_local",bundle.volumeNodeDonorLocalPoints);
    hashPoints("volume.node_donor_origin",bundle.volumeNodeDonorOriginWorld);
    hashMatrices("volume.node_donor_local_to_world",bundle.volumeNodeDonorLocalToWorldLinear);
    hashMatrices("volume.registered_body_local_to_world",bundle.registeredBodyLocalToWorldLinear);
    hashPoints("volume.registered_body_origin",bundle.registeredBodyOriginWorld);
    hash.u64(bundle.detailedSourceVertexCount);hash.u64(bundle.detailedSourceFaceCount);
    hash.u64(bundle.sourceDomains.size());
    for(const auto& domain:bundle.sourceDomains) {
        hash.string(domain.domain);hash.u32(static_cast<std::uint32_t>(domain.role));
        hash.string(domain.sourceSHA256);hash.u32(domain.vertexCount);hash.u32(domain.faceCount);
    }
    hashIndices("volume.source_domain_indices",bundle.volumeNodeSourceDomainIndices);
    hashIndices("volume.source_vertex_indices",bundle.volumeNodeSourceVertexIndices);
    hashIndices("volume.source_face_indices",bundle.volumeNodeSourceFaceIndices);
    hashBary3("volume.source_barycentric",bundle.volumeNodeSourceBarycentric);
    hash.u64(bundle.retainedOriginalNodeCount);
    hashIndices("volume.parent_tetrahedra",bundle.volumeNodeParentTetrahedra);
    hashBary4("volume.parent_barycentric",bundle.volumeNodeParentBarycentric);
    hashPoints("volume.parent_residuals",bundle.volumeNodeParentResidualWorldM);
    if(bundle.usesSeparatedOperationalReference){
        hash.string("separated_operational_reference.v1");
        hash.string(bundle.operationalReferenceDomainSHA256);
        hashIndices("volume.source_parent_node_rows",bundle.volumeNodeSourceParentNodeRows);
        hashScalars("volume.source_parent_registration_distances",bundle.volumeNodeSourceParentRegistrationDistanceM);
        hashIndices("volume.operational_reference_tetrahedra",bundle.volumeNodeOperationalReferenceTetrahedra);
        hashBary4("volume.operational_reference_barycentric",bundle.volumeNodeOperationalReferenceBarycentric);
        hashPoints("volume.operational_reference_residuals",bundle.volumeNodeOperationalReferenceResidualWorldM);
        hashScalars("volume.operational_reference_distances",bundle.volumeNodeOperationalReferenceDistanceM);
    }
    hash.u64(bundle.stableDeepAttachments.size());
    for(const auto& row:bundle.stableDeepAttachments) {
        hash.u64(row.stableAttachmentID);hash.u32(row.donorBody);
        hash.u32(row.stableSourceVertex);hash.u32(row.deepSurfaceVertexRow);
        hash.u32(row.volumeNodeRow);
    }
    for(const auto word:bundle.referenceSourceIdentity)hash.u64(word);
    for(const auto word:bundle.materialSourceIdentity)hash.u64(word);
    return hash.finish();
}

inline bool validatePreparedPhysicalSkinChildMassMappingPath(
        const PhysicalSkinChildMassMappingBundle& bundle,std::string& error) {
    if((bundle.operationalSpatialMassRouteRequired&&
        (!bundle.usesSeparatedOperationalReference||bundle.conformingLineageMassRouteRequired||
         bundle.conformingLineageMassTransportUsed||bundle.exactTopologyIdentityTransportUsed||
         !physical_skin_child_mass_detail::validSHA256(bundle.operationalSpatialReferenceVerificationSHA256)||
         bundle.operationalSpatialTransportReceiptSHA256!=bundle.spatialMassTransportResult.receiptSHA256))||
       bundle.conformingLineageMassRouteRequired!=bundle.conformingLineageMassTransportUsed||
       (bundle.conformingLineageMassTransportUsed&&bundle.exactTopologyIdentityTransportUsed)||
       !physical_skin_child_mass_detail::validSHA256(
           bundle.spatialMassTransportResult.receiptSHA256)||
       !physical_skin_child_mass_detail::validSHA256(bundle.massMappingPathSHA256)||
       bundle.massMappingPathSHA256!=massMappingPathSHA256(
           bundle,bundle.exactTopologyIdentityTransportUsed,
           bundle.conformingLineageMassTransportUsed)) {
        error="physical child mass mapping path receipt is missing or no longer binds its current inputs";
        return false;
    }
    error.clear();return true;
}
} // namespace physical_skin_child_mass_mapping_detail

[[nodiscard]] inline bool preparePhysicalSkinChildMassMapping(
        PhysicalSkinChildMassMappingBundle& bundle,
        const std::span<const double> materialDensityLowerKgM3,
        const std::span<const double> materialDensityUpperKgM3,
        const double maximumTransportDistanceM,
        const std::size_t sourceBodyCount,
        std::string& error) {
    using namespace physical_skin_child_mass_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const auto& parent=bundle.parent;
    const std::size_t parentCount=parent.tetrahedra.size();
    const std::size_t childCount=bundle.childTetrahedra.size();
    const std::size_t nodeCount=bundle.referenceWorldPoints.size();
    if(parentCount==0u||childCount==0u||nodeCount<4u||sourceBodyCount==0u||
       parent.currentWorldPoints.size()!=parent.referenceWorldPoints.size()||
       parent.nodeFrameOwnerBodyIndices.size()!=parent.referenceWorldPoints.size()||
       parent.nodeFrameOwnerLocalPoints.size()!=parent.referenceWorldPoints.size()||
       parent.compiledGlobalNodeIndices.size()!=parent.referenceWorldPoints.size()||
       parent.materialIndices.size()!=parentCount||parent.donorBodyIndices.size()!=parentCount||
       parent.tetrahedronMassKg.size()!=parentCount||parent.referenceTetVolumeM3.size()!=parentCount||
       parent.anatomicalRegionIndices.size()!=parentCount||
       parent.stableSourceFaceIdentities.size()!=parentCount||
       bundle.currentWorldPoints.size()!=nodeCount||
       bundle.childMaterialIndices.size()!=childCount||
       bundle.childDonorBodyIndices.size()!=childCount||
       bundle.childAnatomicalRegionIndices.size()!=childCount||
       bundle.childStableSourceFaceIdentities.size()!=childCount||
       materialDensityLowerKgM3.size()!=materialDensityUpperKgM3.size()||
       materialDensityLowerKgM3.empty()||bundle.maximumSpatialMassTransportDistanceM<=0.0)
        return fail("physical child mass mapping is missing explicit parent/child geometry, identity labels, or density bounds");
    if(!std::isfinite(maximumTransportDistanceM)||maximumTransportDistanceM<=0.0||
       !std::isfinite(bundle.maximumSpatialMassTransportDistanceM)||
       maximumTransportDistanceM!=bundle.maximumSpatialMassTransportDistanceM)
        return fail("physical child mass mapping distance limit is invalid or differs from the pinned bundle");
    if(!validSHA256(parent.parentMassPacketSHA256)||
       !validSHA256(parent.parentSourceContributionSHA256)||
       !validSHA256(bundle.childMeshSHA256)||!validSHA256(bundle.regionLabelSHA256)||
       !validSHA256(bundle.coordinateFrameSHA256))
        return fail("physical child mass mapping requires five pinned native input SHA-256 values");
    const auto semanticSupport=bundle.fractionalSemanticSupport.view();
    const bool hasAnySemanticSupport=!bundle.fractionalSemanticSupport.empty()||
        !bundle.fractionalSemanticSupport.volumeFractions.empty()||
        !bundle.fractionalSemanticSupport.groupMaterialIndices.empty()||
        !bundle.fractionalSemanticSupport.donorBodyIndices.empty()||
        !bundle.fractionalSemanticSupport.anatomicalRegionIndices.empty()||
        !bundle.fractionalSemanticSupport.derivedGroupIdentityIDs.empty()||
        !bundle.fractionalSemanticSupport.parentStableSourceFaceIdentities.empty()||
        !bundle.fractionalSemanticSupport.sourceParentTetOffsets.empty()||
        !bundle.fractionalSemanticSupport.sourceParentTetRows.empty()||
        !bundle.fractionalSemanticSupport.sourceStableFaceOffsets.empty()||
        !bundle.fractionalSemanticSupport.sourceStableFaceRows.empty()||
        !bundle.fractionalSemanticSupport.typedSupportSHA256.empty();
    if(bundle.fractionalSemanticMassRouteRequired){
        if(!hasAnySemanticSupport||bundle.conformingLineageMassRouteRequired||
           bundle.conformingLineageMassRoute.authenticated)
            return fail("fractional semantic mass routing requires its complete explicit support and cannot be mixed with conforming ancestry");
    }else if(hasAnySemanticSupport){
        return fail("fractional semantic support is present without its explicit registered-import route flag");
    }
    const auto& carrierOwned=bundle.sourceCarrierQuadrature;
    if(carrierOwned.importedAndVerified){
        if(!bundle.fractionalSemanticMassRouteRequired||
           !validSHA256(carrierOwned.verificationSHA256)||
           !validSHA256(carrierOwned.sourceCarrierPacketSHA256)||
           !validSHA256(carrierOwned.sourceTetTableSHA256)||
           !validSHA256(carrierOwned.sourceNodeTableSHA256))
            return fail("source-carrier quadrature lacks its authenticated fractional route or source pins");
        if(carrierOwned.hasCarriers()&&
           (carrierOwned.parentOffsets.size()!=parentCount+1u||
            carrierOwned.globalSourceTetRows.size()!=parentCount||
            carrierOwned.parentOffsets.empty()||carrierOwned.parentOffsets.front()!=0u||
            carrierOwned.parentOffsets.back()!=carrierOwned.carrierWeights.size()||
            carrierOwned.carrierBarycentric.size()!=carrierOwned.carrierWeights.size()||
            carrierOwned.witnessChildRows.size()!=carrierOwned.carrierWeights.size()||
            carrierOwned.witnessChildCentroidDistancesM.size()!=carrierOwned.carrierWeights.size()))
            return fail("source-carrier quadrature arrays do not cover the current full parent row domain");
        for(std::size_t row=1u;row<carrierOwned.parentOffsets.size();++row)
            if(carrierOwned.parentOffsets[row]<carrierOwned.parentOffsets[row-1u]||
               carrierOwned.parentOffsets[row]>carrierOwned.carrierWeights.size())
                return fail("source-carrier parent CSR is decreasing or out of range");
    }else if(carrierOwned.hasCarriers()||!carrierOwned.parentOffsets.empty()||
             !carrierOwned.globalSourceTetRows.empty()||!carrierOwned.verificationSHA256.empty()||
             !carrierOwned.sourceCarrierPacketSHA256.empty()){
        return fail("source-carrier data is present without the authenticated import result");
    }

    std::vector<SpatialMassTransportPoint> parentCentroids(parentCount),childCentroids(childCount);
    std::vector<double> parentVolumes(parentCount),childVolumes(childCount);
    for(const auto owner:parent.nodeFrameOwnerBodyIndices)
        if(owner>=sourceBodyCount)
            return fail("physical child mapping parent node-frame owner is outside the registered body domain");
    std::vector<ConservativeSpatialMassTransportParentNodeSourceMass> sourceRows;
    sourceRows.reserve(parent.sourceMassContributions.size());
    for(const auto& row:parent.sourceMassContributions){
        if(row.donorBodyIndex>=sourceBodyCount||row.materialIndex>=materialDensityLowerKgM3.size()||
           row.parentNodeIndex>=parent.referenceWorldPoints.size())
            return fail("physical child mapping parent source donor/material/node is outside its owning domain");
        sourceRows.push_back({.parentNodeIndex=row.parentNodeIndex,.materialIndex=row.materialIndex,
            .sourceDonorBodyIndex=row.donorBodyIndex,.massKg=row.massKg});
    }
    std::vector<std::uint32_t> sourceOffsets,sourceDonors;
    std::vector<double> sourceMasses;
    std::string splitError;
    if(!assembleConservativeSpatialMassTransportParentSourceDonorSplits(
            parent.tetrahedronMassKg,parent.materialIndices,parent.tetrahedra,sourceRows,
            1.0e-10,5.0e-4,sourceOffsets,sourceDonors,sourceMasses,splitError))
        return fail("physical child mapping rejected the native cooked parent donor packet: "+splitError);

    for(std::size_t cell=0;cell<parentCount;++cell){
        const auto tet=parent.tetrahedra[cell];
        const auto volume=signedTetVolume(parent.referenceWorldPoints[tet[0]],
            parent.referenceWorldPoints[tet[1]],parent.referenceWorldPoints[tet[2]],
            parent.referenceWorldPoints[tet[3]]);
        if(!std::isfinite(volume)||volume<=0.0)
            return fail("physical child mapping parent contains a nonpositive reference tetrahedron");
        // The replay owner hashes this exact determinant evaluation. Equivalent
        // algebra can round differently under FMA and is not receipt-identical.
        parentVolumes[cell]=parentReferenceTetVolumeM3(tet,parent.referenceWorldPoints);
        for(const auto node:tet)for(unsigned axis=0;axis<3u;++axis)
            parentCentroids[cell][axis]+=parent.referenceWorldPoints[node][axis]*0.25;
        const auto material=parent.materialIndices[cell];
        const auto donor=parent.donorBodyIndices[cell];
        if(material>=materialDensityLowerKgM3.size()||donor>=sourceBodyCount||
           parent.anatomicalRegionIndices[cell]==NM_INVALID_INDEX||
           parent.stableSourceFaceIdentities[cell]==NM_INVALID_INDEX||
           !std::isfinite(parent.tetrahedronMassKg[cell])||parent.tetrahedronMassKg[cell]<=0.0)
            return fail("physical child mapping parent labels or mass are outside their owning domains");
    }
    bool requiresNativeIdentityRepresentation=false;
    for(std::size_t cell=0;cell<childCount;++cell){
        const auto tet=bundle.childTetrahedra[cell];
        std::array<std::uint32_t,4> sorted=tet;std::sort(sorted.begin(),sorted.end());
        if(sorted[0]==sorted[1]||sorted[1]==sorted[2]||sorted[2]==sorted[3]||sorted[3]>=nodeCount)
            return fail("physical child mapping candidate contains repeated or out-of-range tet nodes");
        const auto volume=signedTetVolume(bundle.referenceWorldPoints[tet[0]],
            bundle.referenceWorldPoints[tet[1]],bundle.referenceWorldPoints[tet[2]],
            bundle.referenceWorldPoints[tet[3]]);
        const auto currentVolume=signedTetVolume(bundle.currentWorldPoints[tet[0]],
            bundle.currentWorldPoints[tet[1]],bundle.currentWorldPoints[tet[2]],
            bundle.currentWorldPoints[tet[3]]);
        if(!std::isfinite(volume)||volume<=0.0||!std::isfinite(currentVolume)||currentVolume<=0.0)
            return fail("physical child mapping candidate contains a nonpositive current or reference tetrahedron");
        for(const auto node:tet)for(unsigned axis=0;axis<3u;++axis) {
            if(!std::isfinite(bundle.referenceWorldPoints[node][axis])||
               !std::isfinite(bundle.currentWorldPoints[node][axis])||
               !std::isfinite(static_cast<float>(bundle.referenceWorldPoints[node][axis]))||
               !std::isfinite(static_cast<float>(bundle.currentWorldPoints[node][axis])))
                return fail("physical child mapping coordinates are not finite in their native and Float32 domains");
            if(double(static_cast<float>(bundle.referenceWorldPoints[node][axis]))!=bundle.referenceWorldPoints[node][axis]||
               double(static_cast<float>(bundle.currentWorldPoints[node][axis]))!=bundle.currentWorldPoints[node][axis])
                requiresNativeIdentityRepresentation=true;
        }
        childVolumes[cell]=parentReferenceTetVolumeM3(tet,bundle.referenceWorldPoints);
        for(const auto node:tet)for(unsigned axis=0;axis<3u;++axis)
            childCentroids[cell][axis]+=bundle.referenceWorldPoints[node][axis]*0.25;
        const auto material=bundle.childMaterialIndices[cell];
        const auto donor=bundle.childDonorBodyIndices[cell];
        if(material>=materialDensityLowerKgM3.size()||donor>=sourceBodyCount||
           bundle.childAnatomicalRegionIndices[cell]==NM_INVALID_INDEX||
           bundle.childStableSourceFaceIdentities[cell]==NM_INVALID_INDEX)
            return fail("physical child mapping candidate omits an explicit material/donor/region/source-face label");
        if(!std::isfinite(materialDensityLowerKgM3[material])||
           !std::isfinite(materialDensityUpperKgM3[material])||
           materialDensityLowerKgM3[material]<=0.0||
           materialDensityLowerKgM3[material]>materialDensityUpperKgM3[material])
            return fail("physical child mapping material has invalid density bounds");
    }

    std::vector<std::array<std::uint32_t,4>> carrierSourceTetNodeRows;
    std::vector<std::array<std::array<double,3>,4>> carrierParentReferenceTetWorldVertices;
    SourceCellCarrierQuadratureInput carrierInput{};
    if(carrierOwned.importedAndVerified&&carrierOwned.hasCarriers()){
        carrierSourceTetNodeRows=parent.tetrahedra;
        carrierParentReferenceTetWorldVertices.resize(parentCount);
        for(std::size_t row=0;row<parentCount;++row){
            const auto tet=parent.tetrahedra[row];
            for(std::size_t corner=0;corner<4u;++corner){
                if(tet[corner]>=parent.referenceWorldPoints.size())
                    return fail("source-carrier parent tet references a missing immutable source node");
                carrierParentReferenceTetWorldVertices[row][corner]=parent.referenceWorldPoints[tet[corner]];
            }
        }
        carrierInput={
            .parentOffsets=carrierOwned.parentOffsets,
            .globalSourceTetRows=carrierOwned.globalSourceTetRows,
            .sourceTetNodeRows=carrierSourceTetNodeRows,
            .parentReferenceTetWorldVertices=carrierParentReferenceTetWorldVertices,
            .carrierWeights=carrierOwned.carrierWeights,
            .carrierBarycentric=carrierOwned.carrierBarycentric,
            .witnessChildRows=carrierOwned.witnessChildRows,
            .witnessChildCentroidDistancesM=carrierOwned.witnessChildCentroidDistancesM,
            .sourceCarrierPacketSHA256=carrierOwned.sourceCarrierPacketSHA256,
            .sourceTetTableSHA256=carrierOwned.sourceTetTableSHA256,
            .sourceNodeTableSHA256=carrierOwned.sourceNodeTableSHA256};
    }

    ConservativeSpatialMassTransportInput input;
    input.parentCentroidsWorld=parentCentroids;
    input.parentReferenceVolumeM3=parentVolumes;
    input.parentMassKg=parent.tetrahedronMassKg;
    input.parentMaterialIndices=parent.materialIndices;
    input.parentDonorBodyIndices=parent.donorBodyIndices;
    input.parentAnatomicalRegionIndices=parent.anatomicalRegionIndices;
    input.parentSourceDonorOffsets=sourceOffsets;
    input.parentSourceDonorBodyIndices=sourceDonors;
    input.parentSourceDonorMassKg=sourceMasses;
    input.childCentroidsWorld=childCentroids;
    input.childReferenceVolumeM3=childVolumes;
    input.childMaterialIndices=bundle.childMaterialIndices;
    input.childDonorBodyIndices=bundle.childDonorBodyIndices;
    input.childAnatomicalRegionIndices=bundle.childAnatomicalRegionIndices;
    input.materialDensityLowerKgM3=materialDensityLowerKgM3;
    input.materialDensityUpperKgM3=materialDensityUpperKgM3;
    input.parentMassPacketSHA256=parent.parentMassPacketSHA256;
    input.parentSourceContributionSHA256=parent.parentSourceContributionSHA256;
    input.childMeshSHA256=bundle.childMeshSHA256;
    input.regionLabelSHA256=bundle.regionLabelSHA256;
    input.coordinateFrameSHA256=bundle.coordinateFrameSHA256;
    input.maximumTransportDistanceM=maximumTransportDistanceM;
    input.relativeClosureTolerance=1.0e-10;
    input.absoluteMassToleranceKg=1.0e-13;
    input.absoluteMeasureToleranceM3=1.0e-15;
    input.maximumAllowedEdges=3'000'000u;
    input.maximumAugmentations=300'000u;
#if defined(NUMI_HUMAN_FULL_REFERENCE_SKIN_CAPACITY)
    if(bundle.fractionalSemanticMassRouteRequired){
        input.maximumAllowedEdges=12'000'000u;
        input.maximumAugmentations=4'000'000u;
    }
#endif


    auto& recordedAllocation=bundle.sourceCarrierQuadrature.recordedAllocation;
    const bool roleAllocation0=recordedAllocation.roles[0].importedAndPinned;
    const bool roleAllocation1=recordedAllocation.roles[1].importedAndPinned;
    if(roleAllocation0!=roleAllocation1)
        return fail("recorded source-carrier allocation is incomplete: both Skin and Fat role receipts are required");
    if(roleAllocation0){
        if(!bundle.fractionalSemanticMassRouteRequired||!carrierOwned.importedAndVerified||
           !carrierOwned.hasCarriers()||carrierOwned.globalSourceTetRows.size()!=parentCount||
           carrierOwned.parentOffsets.size()!=parentCount+1u)
            return fail("recorded role flows require complete paired semantic and source-carrier inputs");
        const std::array<std::uint32_t,2> roleMaterials{15u,16u};
        std::map<std::uint32_t,std::uint32_t> sourceGlobalToCurrentParent;
        for(std::uint32_t p=0;p<parentCount;++p){
            const auto global=carrierOwned.globalSourceTetRows[p];
            if(!sourceGlobalToCurrentParent.emplace(global,p).second)
                return fail("paired source-carrier map has duplicate global source330 parent identities");
        }
        std::array<std::vector<std::uint32_t>,2> roleParentToCurrent,roleChildToCurrent,
            roleVirtualToCurrent;
        std::map<std::pair<std::uint32_t,std::uint32_t>,std::uint32_t> globalVirtualByParentCarrier;
        std::vector<std::uint32_t> virtualToParent,virtualToCarrier;
        std::vector<std::vector<std::uint32_t>> parentToVirtual(parentCount);
        std::vector<SpatialMassTransportPoint> virtualCentroids;
        std::vector<double> virtualVolumes,virtualMasses,virtualDonorMass;
        std::vector<std::uint32_t> virtualMaterials,virtualDonors,virtualRegions,virtualDonorOffsets{0u},virtualDonorIDs;
        ConservativeSpatialMassTransportInput expandedInput;
        ConservativeSpatialChildSemanticSupportData expandedSupport;
        std::vector<std::uint32_t> parentGroup,carrierParents;
        std::vector<std::map<std::uint32_t,double>> preferences;
        std::vector<std::vector<std::uint32_t>> admissibleRegions;
        std::vector<SpatialMassTransportPoint> carrierPoints;
        std::vector<double> strictNearest;
        const auto carrierArraySHA=source_carrier_transport_detail::carrierArrayDigest(input,carrierInput);
        if(!source_carrier_transport_detail::validateCarrierInput(input,semanticSupport,carrierInput,
                parentGroup,preferences,admissibleRegions,carrierPoints,carrierParents,strictNearest,error)||
           !source_carrier_transport_detail::expandParentRows(input,semanticSupport,carrierInput,
                virtualToParent,virtualToCarrier,parentToVirtual,virtualCentroids,virtualVolumes,
                virtualMasses,virtualMaterials,virtualDonors,virtualRegions,virtualDonorOffsets,
                virtualDonorIDs,virtualDonorMass,expandedInput,expandedSupport,carrierArraySHA,error))
            return fail("recorded role allocation failed current source-carrier preflight: "+error);
        for(std::uint32_t v=0;v<virtualToParent.size();++v)
            globalVirtualByParentCarrier.emplace(
                std::pair{virtualToParent[v],virtualToCarrier[v]},v);
        for(unsigned role=0;role<2u;++role){
            auto& roleData=recordedAllocation.roles[role];
            const auto material=roleMaterials[role];
            const auto expectedParentRows=static_cast<std::size_t>(std::count(
                parent.materialIndices.begin(),parent.materialIndices.end(),material));
            const auto expectedChildRows=static_cast<std::size_t>(std::count(
                bundle.childMaterialIndices.begin(),bundle.childMaterialIndices.end(),material));
            if(roleData.materialIndex!=material||roleData.sourceParentRows!=expectedParentRows||
               roleData.childRows!=expectedChildRows||
               roleData.sourceGlobalTetRows.size()!=expectedParentRows||
               roleData.sourceMaterialIndices.size()!=expectedParentRows||
               roleData.sourceDonorBodyIndices.size()!=expectedParentRows||
               roleData.sourceAnatomicalRegionIndices.size()!=expectedParentRows||
               roleData.sourceStableFaceIdentities.size()!=expectedParentRows||
               roleData.sourceMassKg.size()!=expectedParentRows||
               roleData.sourceReferenceVolumeM3.size()!=expectedParentRows||
               roleData.sourceCentroidsWorld.size()!=expectedParentRows||
               roleData.sourceDonorOffsets.size()!=expectedParentRows+1u||
               roleData.sourceDonorOffsets.front()!=0u||
               roleData.sourceDonorOffsets.back()!=roleData.sourceDonorIndices.size()||
               roleData.sourceDonorIndices.size()!=roleData.sourceDonorMassKg.size()||
               roleData.childReferenceVolumeM3.size()!=expectedChildRows||
               roleData.result.childParentOffsets.size()!=expectedChildRows+1u||
               roleData.result.childMassKg.size()!=expectedChildRows||
               roleData.result.childImpliedDensityKgM3.size()!=expectedChildRows||
               roleData.result.childSourceDonorOffsets.size()!=expectedChildRows+1u)
                return fail("role-local recorded flow arrays do not cover their authenticated material domains");
            roleParentToCurrent[role].resize(expectedParentRows);
            for(std::size_t local=0;local<expectedParentRows;++local){
                const auto found=sourceGlobalToCurrentParent.find(roleData.sourceGlobalTetRows[local]);
                if(found==sourceGlobalToCurrentParent.end())
                    return fail("role source-parent CSV names a source330 tetrahedron absent from the current paired parent");
                const auto global=found->second;
                if(parent.materialIndices[global]!=material||
                   roleData.sourceMaterialIndices[local]!=parent.materialIndices[global]||
                   roleData.sourceDonorBodyIndices[local]!=parent.donorBodyIndices[global]||
                   roleData.sourceAnatomicalRegionIndices[local]!=parent.anatomicalRegionIndices[global]||
                   roleData.sourceStableFaceIdentities[local]!=parent.stableSourceFaceIdentities[global]||
                   roleData.sourceMassKg[local]!=parent.tetrahedronMassKg[global]||
                   roleData.sourceReferenceVolumeM3[local]!=parentVolumes[global]||
                   roleData.sourceCentroidsWorld[local]!=parentCentroids[global]||
                   carrierOwned.globalSourceTetRows[global]!=roleData.sourceGlobalTetRows[local])
                    return fail("role source-parent CSV differs from the exact current source330 row, labels, mass, measure, or centroid");
                roleParentToCurrent[role][local]=global;
                const auto rb=roleData.sourceDonorOffsets[local],re=roleData.sourceDonorOffsets[local+1u];
                const auto gb=sourceOffsets[global],ge=sourceOffsets[global+1u];
                if(re-rb!=ge-gb)
                    return fail("role source-parent donor ledger row count differs from the current source donor split");
                for(std::uint32_t j=0;j<ge-gb;++j)
                    if(roleData.sourceDonorIndices[rb+j]!=sourceDonors[gb+j]||
                       roleData.sourceDonorMassKg[rb+j]!=sourceMasses[gb+j])
                        return fail("role source-parent donor CSV differs from the current source donor mass ledger");
                if(local>0u&&roleParentToCurrent[role][local]<=roleParentToCurrent[role][local-1u])
                    return fail("role source-parent rows do not retain the current ascending parent-row order");
            }
            roleChildToCurrent[role].clear();roleChildToCurrent[role].reserve(expectedChildRows);
            for(std::uint32_t child=0;child<childCount;++child)
                if(bundle.childMaterialIndices[child]==material)
                    roleChildToCurrent[role].push_back(child);
            if(roleChildToCurrent[role].size()!=expectedChildRows)
                return fail("role child row crosswalk does not cover its exact material subsequence");
            for(std::size_t local=0;local<expectedChildRows;++local){
                const auto child=roleChildToCurrent[role][local];
                if(roleData.childReferenceVolumeM3[local]!=childVolumes[child])
                    return fail("role child-mass CSV reference volume differs from the current tetrahedron determinant");
            }
            std::size_t localVirtual=0u,localCarrier=0u;
            roleVirtualToCurrent[role].clear();
            for(std::size_t localParent=0;localParent<expectedParentRows;++localParent){
                const auto globalParent=roleParentToCurrent[role][localParent];
                const auto cb=carrierOwned.parentOffsets[globalParent],ce=carrierOwned.parentOffsets[globalParent+1u];
                if(cb==ce){
                    const auto found=globalVirtualByParentCarrier.find(
                        {globalParent,spatial_mass_transport_detail::kInvalid});
                    if(found==globalVirtualByParentCarrier.end()||
                       localVirtual>=roleData.csvExpandedParentToCarrierRows.size()||
                       roleData.csvExpandedParentToCarrierRows[localVirtual]!=spatial_mass_transport_detail::kInvalid)
                        return fail("role centroid virtual-parent map differs from the authenticated paired carrier expansion");
                    roleVirtualToCurrent[role].push_back(found->second);++localVirtual;
                }else{
                    for(std::uint32_t carrierRow=cb;carrierRow<ce;++carrierRow){
                        const auto found=globalVirtualByParentCarrier.find({globalParent,carrierRow});
                        if(found==globalVirtualByParentCarrier.end()||
                           localVirtual>=roleData.csvExpandedParentToCarrierRows.size()||
                           roleData.csvExpandedParentToCarrierRows[localVirtual]!=localCarrier)
                            return fail("role carrier virtual-parent map differs from the authenticated paired carrier expansion");
                        roleVirtualToCurrent[role].push_back(found->second);++localVirtual;++localCarrier;
                    }
                }
            }
            if(localVirtual!=roleData.csvExpandedParentToCarrierRows.size())
                return fail("role expanded-parent CSV map has missing or extra virtual rows");
            std::size_t expectedRoleCarriers=0u;
            for(const auto globalParent:roleParentToCurrent[role])
                expectedRoleCarriers+=carrierOwned.parentOffsets[globalParent+1u]-carrierOwned.parentOffsets[globalParent];
            if(localCarrier!=expectedRoleCarriers)
                return fail("role carrier row map does not cover the complete paired carrier CSR");
        }

        ConservativeSpatialMassTransportResult combined;
        combined.method=UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport;
        combined.usesFractionalSemanticSupport=true;
        combined.childParentOffsets.assign(childCount+1u,0u);
        combined.childMassKg.assign(childCount,0.0);
        combined.childAncestryMeasureVolumeM3.assign(childCount,0.0);
        combined.childImpliedDensityKgM3.assign(childCount,0.0);
        combined.childDominantParentRows.assign(childCount,spatial_mass_transport_detail::kInvalid);
        combined.parentMassTransferredKg.assign(parentCount,0.0);
        combined.parentMeasureTransferredM3.assign(parentCount,0.0);
        combined.childSourceDonorOffsets.assign(childCount+1u,0u);
        SourceCellCarrierTransportTrace combinedTrace;
        combinedTrace.childParentCarrierOffsets.assign(1u,0u);
        combinedTrace.expandedParentToSourceParentRows=virtualToParent;
        combinedTrace.expandedParentToCarrierRows=virtualToCarrier;
        combinedTrace.carrierSourceParentRows=carrierParents;
        combinedTrace.originalCentroidNearestAdmissibleDistanceM=strictNearest;
        combinedTrace.sourceCarrierPacketSHA256=std::string(carrierInput.sourceCarrierPacketSHA256);
        combinedTrace.sourceCarrierArraySHA256=carrierArraySHA;
        spatial_mass_transport_detail::SHA256Builder allocationHash,expandedHash,flowHash,transportHash;
        allocationHash.string("numi.human.paired-recorded-carrier-allocation.v1");
        expandedHash.string("numi.human.paired-recorded-carrier-expanded-anchor.v1");
        flowHash.string("numi.human.paired-recorded-carrier-flow-traces.v1");
        transportHash.string("numi.human.paired-recorded-carrier-transport-receipts.v1");
        std::uint64_t augmentations=0u,candidateEdges=0u;
        for(unsigned role=0;role<2u;++role){
            const auto& roleData=recordedAllocation.roles[role];
            augmentations+=roleData.augmentationCount;candidateEdges+=roleData.semanticCandidateEdgeCount;
            allocationHash.string(roleData.allocationReceiptSHA256);
            expandedHash.string(roleData.expandedSolverReceiptSHA256);
            flowHash.string(roleData.sourceCarrierFlowTraceSHA256);
            transportHash.string(roleData.transportReceiptSHA256);
            recordedAllocation.roleReceiptSHA256[role]=roleData.allocationReceiptSHA256;
            recordedAllocation.roleSourceCarrierPacketSHA256[role]=roleData.sourceCarrierPacketSHA256;
            recordedAllocation.roleSourceCarrierArraySHA256[role]=roleData.sourceCarrierArraySHA256;
            recordedAllocation.roleSourceCarrierFlowTraceSHA256[role]=roleData.sourceCarrierFlowTraceSHA256;
            recordedAllocation.roleTransportReceiptSHA256[role]=roleData.transportReceiptSHA256;
            for(const auto& sha:roleData.csvSHA256)allocationHash.string(sha);
        }
        std::array<std::uint32_t,2> roleChildCursor{};
        for(std::uint32_t child=0;child<childCount;++child){
            const unsigned role=bundle.childMaterialIndices[child]==15u?0u:1u;
            const auto& roleData=recordedAllocation.roles[role];
            const auto local=roleChildCursor[role]++;
            if(local>=roleChildToCurrent[role].size()||roleChildToCurrent[role][local]!=child)
                return fail("paired role child row order differs from the current global child domain");
            if(local==spatial_mass_transport_detail::kInvalid)
                return fail("paired role flow lacks the current child-row crosswalk");
            combined.childParentOffsets[child]=static_cast<std::uint32_t>(combined.childParentRows.size());
            combined.childMassKg[child]=roleData.result.childMassKg[local];
            combined.childImpliedDensityKgM3[child]=roleData.result.childImpliedDensityKgM3[local];
            combined.childAncestryMeasureVolumeM3[child]=roleData.result.childAncestryMeasureVolumeM3[local];
            combined.childDominantParentRows[child]=roleParentToCurrent[role][roleData.result.childDominantParentRows[local]];
            const auto edgeBegin=roleData.result.childParentOffsets[local];
            const auto edgeEnd=roleData.result.childParentOffsets[local+1u];
            std::vector<std::uint32_t> localEdges;
            for(std::uint32_t e=edgeBegin;e<edgeEnd;++e)localEdges.push_back(e);
            std::sort(localEdges.begin(),localEdges.end(),[&](std::uint32_t a,std::uint32_t b){
                return roleParentToCurrent[role][roleData.result.childParentRows[a]]<
                       roleParentToCurrent[role][roleData.result.childParentRows[b]];
            });
            std::uint32_t previousParent=spatial_mass_transport_detail::kInvalid;
            for(const auto e:localEdges){
                const auto localParent=roleData.result.childParentRows[e];
                if(localParent>=roleParentToCurrent[role].size())
                    return fail("role aggregate flow references a missing role-local parent row");
                const auto globalParent=roleParentToCurrent[role][localParent];
                if(globalParent==previousParent)
                    return fail("role aggregate flow repeats a mapped global parent-child edge");
                previousParent=globalParent;
                combined.childParentRows.push_back(globalParent);
                combined.childParentTransportMassKg.push_back(roleData.result.childParentTransportMassKg[e]);
                combined.childParentTransportMeasureM3.push_back(roleData.result.childParentTransportMeasureM3[e]);
                combined.childParentTransportDistanceM.push_back(roleData.result.childParentTransportDistanceM[e]);
                combined.childParentVolumeFractions.push_back(roleData.result.childParentVolumeFractions[e]);
                if(!physical_skin_child_mass_mapping_detail::appendRecordedCarrierFlowEdge(
                       roleData.trace,e,roleVirtualToCurrent[role],combinedTrace,error))
                    return fail("paired role carrier flow cannot be mapped into the global CSR: "+error);
            }
            const auto db=roleData.result.childSourceDonorOffsets[local];
            const auto de=roleData.result.childSourceDonorOffsets[local+1u];
            if(db>=de||de>roleData.result.childSourceDonorBodyIndices.size())
                return fail("role child source-donor ledger is empty or out of range");
            for(std::uint32_t d=db;d<de;++d){
                combined.childSourceDonorBodyIndices.push_back(roleData.result.childSourceDonorBodyIndices[d]);
                combined.childSourceDonorMassKg.push_back(roleData.result.childSourceDonorMassKg[d]);
            }
            combined.childSourceDonorOffsets[child+1u]=
                static_cast<std::uint32_t>(combined.childSourceDonorBodyIndices.size());
        }
        combined.childParentOffsets[childCount]=static_cast<std::uint32_t>(combined.childParentRows.size());
        combined.transportEdgeCount=combined.childParentRows.size();
        combined.augmentationCount=augmentations;combined.semanticCandidateEdgeCount=candidateEdges;
        combined.maximumAllowedEdges=input.maximumAllowedEdges;combined.maximumAugmentations=input.maximumAugmentations;
        if(combined.transportEdgeCount>input.maximumAllowedEdges||
           combined.augmentationCount>input.maximumAugmentations||
           combined.semanticCandidateEdgeCount>input.maximumAllowedEdges)
            return fail("paired role flow totals exceed the current full-domain bounded solver capacities");
        for(unsigned role=0;role<2u;++role){
            const auto& roleData=recordedAllocation.roles[role];
            for(std::size_t local=0;local<roleParentToCurrent[role].size();++local){
                const auto global=roleParentToCurrent[role][local];
                combined.parentMassTransferredKg[global]=roleData.result.parentMassTransferredKg[local];
                combined.parentMeasureTransferredM3[global]=roleData.result.parentMeasureTransferredM3[local];
            }
        }
        recordedAllocation.allocationReceiptSHA256=allocationHash.finish();
        recordedAllocation.expandedSolverReceiptSHA256=expandedHash.finish();
        recordedAllocation.sourceCarrierFlowTraceSHA256=flowHash.finish();
        recordedAllocation.transportReceiptSHA256=transportHash.finish();
        recordedAllocation.sourceCarrierPacketSHA256=carrierInput.sourceCarrierPacketSHA256;
        recordedAllocation.sourceCarrierArraySHA256=carrierArraySHA;
        recordedAllocation.augmentationCount=augmentations;
        recordedAllocation.semanticCandidateEdgeCount=candidateEdges;
        recordedAllocation.sourceParentRows=parentCount;recordedAllocation.childRows=childCount;
        recordedAllocation.aggregateFlowRows=combined.transportEdgeCount;
        recordedAllocation.carrierFlowRows=combinedTrace.childParentCarrierRows.size();
        recordedAllocation.childSourceDonorRows=combined.childSourceDonorBodyIndices.size();
        recordedAllocation.importedAndPinned=true;
        combined.parentMassPacketSHA256=std::string(input.parentMassPacketSHA256);
        combined.parentSourceContributionSHA256=std::string(input.parentSourceContributionSHA256);
        combined.childMeshSHA256=std::string(input.childMeshSHA256);
        combined.regionLabelSHA256=std::string(input.regionLabelSHA256);
        combined.coordinateFrameSHA256=std::string(input.coordinateFrameSHA256);
        combined.maximumTransportDistanceM=input.maximumTransportDistanceM;
        combined.relativeClosureTolerance=input.relativeClosureTolerance;
        combined.absoluteMassToleranceKg=input.absoluteMassToleranceKg;
        combined.absoluteMeasureToleranceM3=input.absoluteMeasureToleranceM3;
        combined.usesFractionalSemanticSupport=true;
        combinedTrace.expandedSolverReceiptSHA256=recordedAllocation.expandedSolverReceiptSHA256;
        recordedAllocation.result=std::move(combined);
        recordedAllocation.trace=std::move(combinedTrace);
        auto& imported=recordedAllocation.result;
        auto& importedTrace=recordedAllocation.trace;
        const auto importedChildMass=imported.childMassKg;
        const auto importedChildDensity=imported.childImpliedDensityKgM3;
        std::vector<double> childVolumesFromCSV=std::vector<double>(childCount,0.0);
        std::array<std::vector<std::uint32_t>,2> finalRoleChildRows=roleChildToCurrent;
        for(unsigned role=0;role<2u;++role)
            for(std::size_t local=0;local<finalRoleChildRows[role].size();++local)
                childVolumesFromCSV[finalRoleChildRows[role][local]]=
                    recordedAllocation.roles[role].childReferenceVolumeM3[local];
        if(childVolumesFromCSV!=childVolumes)
            return fail("paired role child-mass CSV volumes do not exactly cover the current child determinant domain");
        imported.method=UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport;
        imported.usesFractionalSemanticSupport=true;
        if(!canonicalizeRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
                input,semanticSupport,carrierInput,imported,importedTrace,error))
            return fail("pinned six-CSV paired carrier flow failed solver-free current-input feasibility: "+error);
        std::vector<double> transportedVirtualMass(virtualMasses.size(),0.0);
        for(std::size_t edge=0;edge<imported.childParentRows.size();++edge){
            const auto begin=importedTrace.childParentCarrierOffsets[edge];
            const auto end=importedTrace.childParentCarrierOffsets[edge+1u];
            if(begin>=end||end>importedTrace.childParentCarrierRows.size())
                return fail("canonicalized paired carrier trace omits an aggregate flow edge");
            for(std::uint32_t k=begin;k<end;++k){
                const auto virtualParent=importedTrace.childParentCarrierRows[k];
                if(virtualParent>=transportedVirtualMass.size())
                    return fail("canonicalized paired carrier trace names an absent expanded source parent");
                transportedVirtualMass[virtualParent]+=importedTrace.childParentCarrierMassKg[k];
            }
        }
        for(std::size_t virtualParent=0;virtualParent<virtualMasses.size();++virtualParent)
            if(!physical_skin_child_mass_mapping_detail::sourceCarrierRelativeMassClose(
                   transportedVirtualMass[virtualParent],virtualMasses[virtualParent]))
                return fail("paired recorded flow fails relative-only per-carrierized-source mass closure");
        for(unsigned role=0;role<2u;++role){
            const auto& roleData=recordedAllocation.roles[role];
            for(std::size_t localChild=0;localChild<roleChildToCurrent[role].size();++localChild){
                const auto child=roleChildToCurrent[role][localChild];
                if(!physical_skin_child_mass_mapping_detail::sourceCarrierRelativeMassClose(
                       importedChildMass[child],imported.childMassKg[child])||
                   std::abs(importedChildDensity[child]-imported.childImpliedDensityKgM3[child])>1.0e-8)
                    return fail("paired role child-mass CSV does not match mass and density derived from carrier flows");
                const auto localBegin=roleData.result.childParentOffsets[localChild];
                const auto localEnd=roleData.result.childParentOffsets[localChild+1u];
                for(std::uint32_t e=localBegin;e<localEnd;++e){
                    const auto globalParent=roleParentToCurrent[role][roleData.result.childParentRows[e]];
                    const auto begin=imported.childParentOffsets[child],end=imported.childParentOffsets[child+1u];
                    auto found=std::lower_bound(imported.childParentRows.begin()+begin,
                        imported.childParentRows.begin()+end,globalParent);
                    if(found==imported.childParentRows.begin()+end||*found!=globalParent)
                        return fail("paired collapsed flow CSV edge is absent after carrier-flow aggregation");
                    const auto at=static_cast<std::size_t>(found-imported.childParentRows.begin());
                    if(imported.childParentTransportMassKg[at]!=roleData.result.childParentTransportMassKg[e]||
                       imported.childParentTransportMeasureM3[at]!=roleData.result.childParentTransportMeasureM3[e]||
                       imported.childParentTransportDistanceM[at]!=roleData.result.childParentTransportDistanceM[e])
                        return fail("paired collapsed flow CSV differs from its exact expanded carrier-flow aggregation");
                }
                if(!physical_skin_child_mass_mapping_detail::recordedRoleChildDonorLedgerMatches(
                       roleData.result,localChild,imported,child))
                    return fail("paired role child donor CSV differs from canonicalized carrier-flow-derived donor mass");
            }
        }
        bundle.sourceCarrierQuadrature.transportTrace=importedTrace;
    }

    ConservativeSpatialMassTransportResult transfer;
    bool usedExactTopologyIdentity=false;
    bool usedVerifiedConformingAncestry=false;
    bool usedVerifiedOperationalSpatial=false;
    if(recordedAllocation.importedAndPinned){
        transfer=recordedAllocation.result;
        usedVerifiedOperationalSpatial=true;
    }
    if(bundle.operationalSpatialMassRouteRequired){
        if(!bundle.usesSeparatedOperationalReference||bundle.conformingLineageMassRouteRequired||
           bundle.conformingLineageMassRoute.authenticated||
           !validSHA256(bundle.operationalSpatialReferenceVerificationSHA256)||
           !validSHA256(bundle.operationalSpatialTransportReceiptSHA256)||
           bundle.operationalSpatialTransportReceiptSHA256!=bundle.spatialMassTransportResult.receiptSHA256)
            return fail("operational spatial-reference proof or cached receipt is missing or conflicts with another route");
        const bool verified=recordedAllocation.importedAndPinned?
            verifyRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
                input,semanticSupport,carrierInput,bundle.spatialMassTransportResult,
                bundle.sourceCarrierQuadrature.transportTrace,error):
            (bundle.sourceCarrierQuadrature.importedAndVerified&&carrierOwned.hasCarriers()?
                verifyRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
                    input,semanticSupport,carrierInput,bundle.spatialMassTransportResult,
                    bundle.sourceCarrierQuadrature.transportTrace,error):
            (bundle.fractionalSemanticMassRouteRequired?
                verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
                    input,semanticSupport,bundle.spatialMassTransportResult,error):
                verifyRecordedConservativeSpatialMassTransport(input,bundle.spatialMassTransportResult,error)));
        if(!verified)
            return fail("operational spatial-reference transport failed current-input verification: "+error);
        transfer=bundle.spatialMassTransportResult;
        usedVerifiedOperationalSpatial=true;
    }else if(!bundle.operationalSpatialReferenceVerificationSHA256.empty()||
             !bundle.operationalSpatialTransportReceiptSHA256.empty()){
        return fail("operational spatial-reference proof is present without its required import flag");
    }
    if(bundle.conformingLineageMassRouteRequired) {
        if(!bundle.conformingLineageMassRoute.authenticated||
           !physical_skin_child_mass_mapping_detail::buildVerifiedConformingAncestryTransport(
               input,bundle,transfer,error))
            return fail("authenticated conforming ancestry mass mapping rejected: "+error);
        if(!verifyRecordedConservativeSpatialMassTransport(input,transfer,error))
            return fail("authenticated conforming ancestry transport receipt failed current-input verification: "+error);
        usedVerifiedConformingAncestry=true;
    } else if(bundle.conformingLineageMassRoute.authenticated||
              !bundle.conformingLineageMassRoute.verificationSHA256.empty()) {
        return fail("conforming ancestry mass route is present without its required registered-import proof flag");
    }
    std::vector<std::uint32_t> exactChildParentRows;
    if(!bundle.fractionalSemanticMassRouteRequired&&
       !usedVerifiedConformingAncestry&&!usedVerifiedOperationalSpatial&&
       parentCount<=input.maximumAugmentations&&childCount<=input.maximumAugmentations&&
       childCount<=input.maximumAllowedEdges&&
       physical_skin_child_mass_mapping_detail::exactIdentityParentRows(bundle,
           parentCentroids,childCentroids,maximumTransportDistanceM,exactChildParentRows)) {
        ConservativeSpatialMassTransportResult exact;
        std::string exactError;
        if(physical_skin_child_mass_mapping_detail::buildExactIdentityTransport(
               input,exactChildParentRows,exact,exactError)&&
           verifyRecordedConservativeSpatialMassTransport(input,exact,exactError)) {
            std::vector<std::uint32_t> revalidatedRows;
            if(!physical_skin_child_mass_mapping_detail::exactIdentityParentRows(bundle,
                   parentCentroids,childCentroids,maximumTransportDistanceM,revalidatedRows)||
               revalidatedRows!=exactChildParentRows||
               exact.childParentRows!=exactChildParentRows||
               exact.transportEdgeCount!=childCount)
                return fail("exact topology identity eligibility changed before receipt-bound mass mapping commit");
            transfer=std::move(exact);
            usedExactTopologyIdentity=true;
        }
    }
    // Retain exact native source doubles only for a fully certified identity:
    // current coordinates compile to identical parent GPU bits and reference
    // coordinates remain exact. Changed topology keeps both Float32 gates.
    if(requiresNativeIdentityRepresentation &&
       (!usedExactTopologyIdentity || bundle.referenceWorldPoints!=parent.referenceWorldPoints))
        return fail("non-Float32 source coordinates require certified compiler-equivalent exact parent-reference identity");
    if(!usedExactTopologyIdentity&&!usedVerifiedConformingAncestry&&!usedVerifiedOperationalSpatial) {
        bool assembled=false,verified=false;
        if(bundle.sourceCarrierQuadrature.recordedAllocation.importedAndPinned){
            transfer=bundle.sourceCarrierQuadrature.recordedAllocation.result;
            assembled=true;verified=true;
        }else if(bundle.sourceCarrierQuadrature.importedAndVerified&&carrierOwned.hasCarriers()){
            SourceCellCarrierTransportTrace trace;
            assembled=assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
                input,semanticSupport,carrierInput,transfer,trace,error);
            if(assembled)bundle.sourceCarrierQuadrature.transportTrace=std::move(trace);
            verified=assembled&&verifyRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
                input,semanticSupport,carrierInput,transfer,bundle.sourceCarrierQuadrature.transportTrace,error);
        }else{
            assembled=bundle.fractionalSemanticMassRouteRequired?
                assembleConservativeSpatialMassTransportWithSemanticSupport(
                    input,semanticSupport,transfer,error):
                assembleConservativeSpatialMassTransport(input,transfer,error);
            verified=assembled&&(bundle.fractionalSemanticMassRouteRequired?
                verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
                    input,semanticSupport,transfer,error):
                verifyRecordedConservativeSpatialMassTransport(input,transfer,error));
        }
        if(!assembled)return fail("physical child mass transport was not feasible: "+error);
        if(!verified)return fail("physical child mass transport receipt failed exact current-input verification: "+error);
    }
    if(transfer.childDominantParentRows.size()!=childCount||
       transfer.childMassKg.size()!=childCount||
       std::any_of(transfer.childDominantParentRows.begin(),transfer.childDominantParentRows.end(),
           [&](std::uint32_t parentRow){return parentRow>=parentCount;}))
        return fail("physical child mass transport result has incomplete child-parent rows");
    std::vector<std::uint32_t> semanticGroupForParent;
    const bool hasOneHopSemanticSupport=!semanticSupport.cellAdmissibleRegionOffsets.empty();
    if(bundle.fractionalSemanticMassRouteRequired&&
       !fractional_semantic_transport_detail::buildSourceParentGroupMap(
           semanticSupport,parentCount,semanticGroupForParent))
        return fail("fractional semantic source-parent group map is incomplete or ambiguous");
    for(std::size_t child=0;child<childCount;++child){
        const auto parentRow=transfer.childDominantParentRows[child];
        if(parent.materialIndices[parentRow]!=bundle.childMaterialIndices[child]||
           parent.stableSourceFaceIdentities[parentRow]==NM_INVALID_INDEX)
            return fail("physical child dominant parent violates material or source-face eligibility");
        if(!bundle.fractionalSemanticMassRouteRequired){
            if(parent.donorBodyIndices[parentRow]!=bundle.childDonorBodyIndices[child]||
               parent.anatomicalRegionIndices[parentRow]!=bundle.childAnatomicalRegionIndices[child])
                return fail("physical child dominant parent violates donor or region eligibility");
        }else{
            const auto group=semanticGroupForParent[parentRow];
            if(group>=semanticSupport.groupMaterialIndices.size()||
               semanticSupport.groupMaterialIndices[group]!=parent.materialIndices[parentRow]||
               semanticSupport.donorBodyIndices[group]!=parent.donorBodyIndices[parentRow]||
               semanticSupport.anatomicalRegionIndices[group]!=parent.anatomicalRegionIndices[parentRow])
                return fail("fractional semantic dominant parent is absent from its authenticated source group");
            const bool admissible=hasOneHopSemanticSupport?
                (semanticSupport.groupMaterialIndices[group]==bundle.childMaterialIndices[child]&&
                 fractional_semantic_transport_detail::childAllowsSourceRegion(
                     semanticSupport,static_cast<std::uint32_t>(child),parent.anatomicalRegionIndices[parentRow])):
                fractional_semantic_transport_detail::childPrefersSemanticGroup(
                    semanticSupport,static_cast<std::uint32_t>(child),group);
            if(!admissible)
                return fail("fractional semantic dominant parent is outside the child's authenticated one-hop support or legacy P1 preference");
        }
    }
    bundle.childDominantParentTetrahedra=transfer.childDominantParentRows;
    bundle.spatialMassTransportResult=std::move(transfer);
    bundle.exactTopologyIdentityTransportUsed=usedExactTopologyIdentity;
    bundle.conformingLineageMassTransportUsed=usedVerifiedConformingAncestry;
    bundle.massMappingPathSHA256=physical_skin_child_mass_mapping_detail::massMappingPathSHA256(
        bundle,usedExactTopologyIdentity,usedVerifiedConformingAncestry);
    error.clear();
    return true;
}

// The app can call this on its scratch-prefix WorldSource before later objects
// are appended; the established owner performs all detailed O/I/D, facet,
// frame, source ancestry, mass, and report checks transactionally.
[[nodiscard]] inline bool appendPreparedPhysicalSkinChildCandidate(
        numi::matter::WorldSource& world,
        const PhysicalSkinChildMassMappingBundle& bundle,
        const std::uint32_t skinMaterialIndex,const std::string_view skinMaterialName,
        const std::string_view skinMaterialProvenance,const double skinDonorDebitKg,
        const std::uint32_t fatMaterialIndex,const std::string_view fatMaterialName,
        const std::string_view fatMaterialProvenance,const double fatDonorDebitKg,
        const std::size_t sourceBodyCount,const std::string_view objectName,
        PreparedOrganicFEM& preparedSkin,PreparedOrganicFEM& preparedFat,
        ExplicitUnpairedVolumeReport& report,std::string& error) {
    if(!physical_skin_child_mass_mapping_detail::validatePreparedPhysicalSkinChildMassMappingPath(
           bundle,error))return false;
    const auto topology=bundle.topologyView();
    return appendRegisteredSkinFatUnpairedVolume(world,bundle.outerSkin,bundle.boundaryMap,
        topology,skinMaterialIndex,skinMaterialName,skinMaterialProvenance,skinDonorDebitKg,
        fatMaterialIndex,fatMaterialName,fatMaterialProvenance,fatDonorDebitKg,
        sourceBodyCount,objectName,preparedSkin,preparedFat,report,error);
}

} // namespace metalrobo::human::coarse_tissue


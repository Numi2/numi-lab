#pragma once

// Strict adapter for a real compiled skin/fat parent mass packet and a separately
// prepared positive child volume. Spatial transport remains distinct from
// geometric ancestry; the existing unpaired volume owner performs final admission.
#include "NumiHumanUnpairedSkinFatVolume.hpp"

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
    constexpr std::size_t maxNodes=300000u,maxSourceRows=1200000u;
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
        if(!haveMass||!haveTet||expectedTet>=200000u||massLine.size()>4096u||tetLine.size()>4096u)
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
    std::vector<std::uint32_t> childRegisteredLayerFacetSurfaceFaces;
    std::vector<std::uint32_t> childRegisteredLayerFacetSourceFaceIdentities;
    std::vector<std::uint8_t> childAuthoredHeadSidewallFacetFlags;
    std::vector<std::uint32_t> childAuthoredHeadSidewallStableFaceIdentities;

    std::vector<std::uint32_t> volumeNodeDonorBodyIndices;
    std::vector<V3> volumeNodeDonorLocalPoints,volumeNodeDonorOriginWorld;
    std::vector<std::array<double,9>> volumeNodeDonorLocalToWorldLinear;
    std::vector<std::array<double,9>> registeredBodyLocalToWorldLinear;
    std::vector<V3> registeredBodyOriginWorld;
    std::size_t detailedSourceVertexCount=0u,detailedSourceFaceCount=0u;
    std::vector<std::uint32_t> volumeNodeSourceVertexIndices,volumeNodeSourceFaceIndices;
    std::vector<std::array<double,3>> volumeNodeSourceBarycentric;
    std::size_t retainedOriginalNodeCount=0u;
    std::vector<std::uint32_t> volumeNodeParentTetrahedra;
    std::vector<std::array<double,4>> volumeNodeParentBarycentric;
    std::vector<V3> volumeNodeParentResidualWorldM;
    std::array<std::uint64_t,4> referenceSourceIdentity{},materialSourceIdentity{};

    std::string childMeshSHA256,regionLabelSHA256,coordinateFrameSHA256;
    double maximumSpatialMassTransportDistanceM=0.0;
    ConservativeSpatialMassTransportResult spatialMassTransportResult;

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
        parentVolumes[cell]=volume;
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
        for(const auto node:tet)for(unsigned axis=0;axis<3u;++axis)
            if(!std::isfinite(bundle.referenceWorldPoints[node][axis])||
               !std::isfinite(bundle.currentWorldPoints[node][axis])||
               double(static_cast<float>(bundle.referenceWorldPoints[node][axis]))!=bundle.referenceWorldPoints[node][axis]||
               double(static_cast<float>(bundle.currentWorldPoints[node][axis]))!=bundle.currentWorldPoints[node][axis])
                return fail("physical child mapping coordinates are not exact finite Float32 owner points");
        childVolumes[cell]=volume;
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

    ConservativeSpatialMassTransportResult transfer;
    if(!assembleConservativeSpatialMassTransport(input,transfer,error))
        return fail("physical child mass transport was not feasible: "+error);
    if(!verifyRecordedConservativeSpatialMassTransport(input,transfer,error))
        return fail("physical child mass transport receipt failed exact current-input verification: "+error);
    if(transfer.childDominantParentRows.size()!=childCount||
       transfer.childMassKg.size()!=childCount||
       std::any_of(transfer.childDominantParentRows.begin(),transfer.childDominantParentRows.end(),
           [&](std::uint32_t parentRow){return parentRow>=parentCount;}))
        return fail("physical child mass transport result has incomplete child-parent rows");
    for(std::size_t child=0;child<childCount;++child){
        const auto parentRow=transfer.childDominantParentRows[child];
        if(parent.materialIndices[parentRow]!=bundle.childMaterialIndices[child]||
           parent.donorBodyIndices[parentRow]!=bundle.childDonorBodyIndices[child]||
           parent.anatomicalRegionIndices[parentRow]!=bundle.childAnatomicalRegionIndices[child]||
           parent.stableSourceFaceIdentities[parentRow]==NM_INVALID_INDEX)
            return fail("physical child dominant parent violates material, donor, region, or source-face eligibility");
    }
    bundle.childDominantParentTetrahedra=transfer.childDominantParentRows;
    bundle.spatialMassTransportResult=std::move(transfer);
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
    if(bundle.spatialMassTransportResult.receiptSHA256.empty()){
        error="physical child mass mapping must pass the exact spatial transport gate before volume compilation";
        return false;
    }
    const auto topology=bundle.topologyView();
    return appendRegisteredSkinFatUnpairedVolume(world,bundle.outerSkin,bundle.boundaryMap,
        topology,skinMaterialIndex,skinMaterialName,skinMaterialProvenance,skinDonorDebitKg,
        fatMaterialIndex,fatMaterialName,fatMaterialProvenance,fatDonorDebitKg,
        sourceBodyCount,objectName,preparedSkin,preparedFat,report,error);
}

} // namespace metalrobo::human::coarse_tissue


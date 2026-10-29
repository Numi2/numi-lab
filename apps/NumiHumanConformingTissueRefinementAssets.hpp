#pragma once

// Authenticated loader for the existing 530 lineage CSV/NPZ and TetGen
// reference files. This proves source construction and immutable mass
// subdivision; it does not admit physical geometry or start runtime physics.
#include "NumiHumanConformingTissueRefinement.hpp"
#include "NumiHumanPhysicalSkinChildTetGen.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace metalrobo::human::coarse_tissue {

struct PhysicalSkinConformingTissueRefinementAssets {
    std::filesystem::path lineagePacketNPZ;
    std::filesystem::path refinedVolumeNPZ;
    std::filesystem::path refinedTetgenNode,refinedTetgenElements,childParentLedgerCSV;
    std::filesystem::path childSourceDonorLedgerCSV,nodeSourceAncestryCSV,orderedSplitEdgesCSV;
    std::filesystem::path producerReceipt,selfAuditReceipt;
    std::filesystem::path sourceConstructionProofReceipt;
    std::string sourceConstructionProofReceiptSHA256;
    std::string lineagePacketSHA256,refinedVolumeSHA256,sourceVolumeSHA256;
    std::string refinedNodeSHA256,refinedElementsSHA256,childParentLedgerSHA256;
    std::string childSourceDonorLedgerSHA256,nodeSourceAncestrySHA256,orderedSplitEdgesSHA256;
    std::string producerReceiptSHA256,selfAuditReceiptSHA256;
};

struct PhysicalSkinVerifiedConformingReference {
    ConformingTissueRefinementResult refinement;
    std::vector<V3> referenceWorldPoints;
    std::vector<std::array<std::uint32_t,4>> tetrahedra;
    std::vector<std::uint32_t> materialIndices,geometricDonorBodyIndices;
    std::vector<std::uint32_t> anatomicalRegionIndices,stableSourceFaceIds;
    std::vector<std::uint32_t> sourceDonorOffsets,sourceDonorBodyIndices;
    std::vector<double> sourceDonorMassKg;
    std::string refinedVolumeSHA256;
    std::string sourceDonorLedgerVerificationSHA256;
};

namespace physical_skin_conforming_refinement_assets_detail {

inline bool pinnedText(const std::filesystem::path& path,const std::string& expected,
                       const std::size_t maximumBytes,std::string& text,std::string& error) {
    using physical_skin_child_mass_detail::validSHA256;
    if(path.empty()||!validSHA256(expected)){error="conforming-refinement asset lacks a path or lowercase SHA-256 pin";return false;}
    std::error_code sizeError;const auto size=std::filesystem::file_size(path,sizeError);
    if(sizeError||size>maximumBytes){error="conforming-refinement asset is unreadable or exceeds its bounded file size: "+path.string();return false;}
    std::ifstream in(path,std::ios::binary);
    if(!in){error="cannot open pinned conforming-refinement asset: "+path.string();return false;}
    std::string next((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());
    if(in.bad()){error="failed while reading pinned conforming-refinement asset: "+path.string();return false;}
    spatial_mass_transport_detail::SHA256Builder hash;
    if(!next.empty())hash.bytes(next.data(),next.size());
    if(hash.finish()!=expected){error="conforming-refinement asset SHA-256 differs from its pin: "+path.string();return false;}
    text=std::move(next);return true;
}

inline bool pinnedFile(const std::filesystem::path& path,const std::string& expected,std::string& error) {
    std::string actual;
    if(path.empty()||!physical_skin_child_mass_detail::validSHA256(expected)||
       !physical_skin_child_mass_detail::sha256File(path,actual,error)) {
        if(error.empty())error="conforming-refinement NPZ is missing a valid pinned file";
        return false;
    }
    if(actual!=expected){error="conforming-refinement NPZ SHA-256 differs from its pin: "+path.string();return false;}
    return true;
}

inline bool jsonValueDelimiter(const std::string_view json,const std::size_t pos) {
    return pos<json.size()&&
        (std::isspace(static_cast<unsigned char>(json[pos]))||json[pos]==','||
         json[pos]=='}'||json[pos]==']');
}

inline bool jsonStringFieldMatches(const std::string_view json,const std::string_view key,
                                   const std::string_view expected,std::string& error) {
    const std::string token="\""+std::string(key)+"\"";
    std::size_t count=0u,pos=0u;
    while((pos=json.find(token,pos))!=std::string_view::npos) {
        ++count;pos+=token.size();
        while(pos<json.size()&&std::isspace(static_cast<unsigned char>(json[pos])))++pos;
        if(pos>=json.size()||json[pos++]!=':'){error="malformed conforming-refinement receipt field "+std::string(key);return false;}
        while(pos<json.size()&&std::isspace(static_cast<unsigned char>(json[pos])))++pos;
        if(pos>=json.size()||json[pos++]!='\"'){error="conforming-refinement receipt field is not a string: "+std::string(key);return false;}
        std::string value;bool closed=false;
        while(pos<json.size()) {
            const char c=json[pos++];
            if(c=='\\') {
                if(pos>=json.size()){error="truncated JSON escape in conforming-refinement receipt";return false;}
                value.push_back(json[pos++]);
            } else if(c=='\"'){closed=true;break;}
            else value.push_back(c);
        }
        if(!closed||!jsonValueDelimiter(json,pos)||value!=expected){
            error="conforming-refinement receipt field differs from its pinned identity: "+std::string(key);return false;
        }
    }
    if(count!=1u){error="conforming-refinement receipt field is missing or repeated: "+std::string(key);return false;}
    return true;
}

inline bool jsonBooleanFieldMatches(const std::string_view json,const std::string_view key,
                                    const bool expected,std::string& error) {
    const std::string token="\""+std::string(key)+"\"";
    std::size_t count=0u,pos=0u;
    while((pos=json.find(token,pos))!=std::string_view::npos) {
        ++count;auto cursor=pos+token.size();
        while(cursor<json.size()&&std::isspace(static_cast<unsigned char>(json[cursor])))++cursor;
        if(cursor>=json.size()||json[cursor++]!=':'){error="malformed conforming-refinement receipt boolean";return false;}
        while(cursor<json.size()&&std::isspace(static_cast<unsigned char>(json[cursor])))++cursor;
        const auto literal=expected?std::string_view("true"):std::string_view("false");
        if(json.substr(cursor,literal.size())!=literal){error="conforming-refinement receipt boolean differs from its contract: "+std::string(key);return false;}
        cursor+=literal.size();
        if(!jsonValueDelimiter(json,cursor)){
            error="conforming-refinement receipt boolean is not a complete JSON token: "+std::string(key);return false;
        }
        pos=cursor;
    }
    if(count!=1u){error="conforming-refinement receipt boolean is missing or repeated: "+std::string(key);return false;}
    return true;
}

inline bool jsonUnsignedFieldMatches(const std::string_view json,const std::string_view key,
                                     const std::uint64_t expected,std::string& error) {
    const std::string token="\""+std::string(key)+"\"";
    std::size_t count=0u,pos=0u;
    while((pos=json.find(token,pos))!=std::string_view::npos) {
        ++count;auto cursor=pos+token.size();
        while(cursor<json.size()&&std::isspace(static_cast<unsigned char>(json[cursor])))++cursor;
        if(cursor>=json.size()||json[cursor++]!=':'){error="malformed conforming-refinement receipt integer";return false;}
        while(cursor<json.size()&&std::isspace(static_cast<unsigned char>(json[cursor])))++cursor;
        const auto begin=cursor;
        while(cursor<json.size()&&std::isdigit(static_cast<unsigned char>(json[cursor])))++cursor;
        std::uint64_t value=0u;
        if(begin==cursor||(cursor-begin>1u&&json[begin]=='0')||
           !physical_skin_child_mass_detail::parseNumber(
                std::string(json.substr(begin,cursor-begin)),value)||value!=expected||
           !jsonValueDelimiter(json,cursor)) {
            error="conforming-refinement receipt count differs from its imported files or is not an exact JSON integer: "+std::string(key);return false;
        }
        pos=cursor;
    }
    if(count!=1u){error="conforming-refinement receipt count is missing or repeated: "+std::string(key);return false;}
    return true;
}

inline bool jsonZeroFields(const std::string_view json,const std::string_view key,
                           const std::size_t expectedCount,std::string& error) {
    const std::string token="\""+std::string(key)+"\"";
    std::size_t count=0u,pos=0u;
    while((pos=json.find(token,pos))!=std::string_view::npos) {
        ++count;pos+=token.size();
        while(pos<json.size()&&std::isspace(static_cast<unsigned char>(json[pos])))++pos;
        if(pos>=json.size()||json[pos++]!=':'){error="malformed conforming-refinement receipt zero count";return false;}
        while(pos<json.size()&&std::isspace(static_cast<unsigned char>(json[pos])))++pos;
        if(pos>=json.size()||json[pos++]!='0'||!jsonValueDelimiter(json,pos)) {
            error="conforming-refinement self-audit receipt contains a nonzero or malformed intersection count";return false;
        }
    }
    if(count!=expectedCount){error="conforming-refinement self-audit has a missing layer count";return false;}
    return true;
}

inline bool jsonStringValueInObject(const std::string_view object,const std::string_view key,
                                    std::string& value,std::string& error) {
    std::string token(1u,static_cast<char>(34));token.append(key);token.push_back(static_cast<char>(34));
    std::size_t pos=object.find(token),count=0u;
    if(pos==std::string_view::npos){error="source-construction receipt is missing field "+std::string(key);return false;}
    ++count;pos+=token.size();
    if(object.find(token,pos)!=std::string_view::npos){error="source-construction receipt repeats field "+std::string(key);return false;}
    while(pos<object.size()&&std::isspace(static_cast<unsigned char>(object[pos])))++pos;
    if(pos>=object.size()||object[pos++]!=':'){error="malformed source-construction receipt field "+std::string(key);return false;}
    while(pos<object.size()&&std::isspace(static_cast<unsigned char>(object[pos])))++pos;
    if(pos>=object.size()||object[pos++]!=static_cast<char>(34)){error="source-construction receipt field is not a string: "+std::string(key);return false;}
    std::string next;bool closed=false;
    while(pos<object.size()){
        const char c=object[pos++];
        if(c==static_cast<char>(92)){
            if(pos>=object.size()){error="truncated source-construction receipt string";return false;}
            const char escaped=object[pos++];
            switch(escaped){
                case static_cast<char>(34):next.push_back(static_cast<char>(34));break;
                case static_cast<char>(92):next.push_back(static_cast<char>(92));break;
                case '/':next.push_back('/');break;
                case 'b':next.push_back(static_cast<char>(8));break;
                case 'f':next.push_back(static_cast<char>(12));break;
                case 'n':next.push_back(static_cast<char>(10));break;
                case 'r':next.push_back(static_cast<char>(13));break;
                case 't':next.push_back(static_cast<char>(9));break;
                default:error="unsupported escape in source-construction receipt field";return false;
            }
        }else if(c==static_cast<char>(34)){closed=true;break;}
        else next.push_back(c);
    }
    if(!closed||!jsonValueDelimiter(object,pos)){
        error="unterminated or malformed source-construction receipt string";return false;
    }
    value=std::move(next);(void)count;return true;
}

using NamedInputFiles=std::map<std::string,std::pair<std::filesystem::path,std::string>>;
inline bool readNamedInputFiles(const std::string_view json,NamedInputFiles& output,std::string& error) {
    std::string token(1u,static_cast<char>(34));token.append("input_files");token.push_back(static_cast<char>(34));
    const auto key=json.find(token);
    if(key==std::string_view::npos||json.find(token,key+token.size())!=std::string_view::npos){
        error="actual source-construction proof must contain one pinned input_files array";return false;
    }
    auto pos=json.find('[',key+token.size());
    if(pos==std::string_view::npos){error="actual source-construction proof input_files is not an array";return false;}
    ++pos;NamedInputFiles next;
    while(pos<json.size()){
        while(pos<json.size()&&(std::isspace(static_cast<unsigned char>(json[pos]))||json[pos]==','))++pos;
        if(pos<json.size()&&json[pos]==']'){++pos;break;}
        if(pos>=json.size()||json[pos]!='{'){error="actual source-construction input file entry is not an object";return false;}
        const auto begin=pos++;bool inString=false,escaped=false;std::size_t depth=1u;
        for(;pos<json.size()&&depth!=0u;++pos){
            const char c=json[pos];
            if(inString){
                if(escaped)escaped=false;
                else if(c==static_cast<char>(92))escaped=true;
                else if(c==static_cast<char>(34))inString=false;
            }else if(c==static_cast<char>(34))inString=true;
            else if(c=='{')++depth;
            else if(c=='}')--depth;
        }
        if(depth!=0u){error="actual source-construction input file object is unterminated";return false;}
        const auto object=json.substr(begin,pos-begin);
        std::string name,path,sha;
        if(!jsonStringValueInObject(object,"name",name,error)||
           !jsonStringValueInObject(object,"path",path,error)||
           !jsonStringValueInObject(object,"sha256",sha,error)||
           !physical_skin_child_mass_detail::validSHA256(sha)||
           !next.emplace(name,std::pair{std::filesystem::path(path),sha}).second){
            if(error.empty())error="actual source-construction receipt has a duplicate or invalid named input";
            return false;
        }
    }
    if(next.empty()){error="actual source-construction proof input_files array is empty or malformed";return false;}
    output=std::move(next);return true;
}

inline bool resolveAssetsFromActualProofReceipt(
        const PhysicalSkinConformingTissueRefinementAssets& declared,
        PhysicalSkinConformingTissueRefinementAssets& resolved,std::string& error) {
    std::string proof;
    if(!pinnedText(declared.sourceConstructionProofReceipt,declared.sourceConstructionProofReceiptSHA256,
                   8u*1024u*1024u,proof,error)||
       !jsonStringFieldMatches(proof,"status","SOURCE_CONSTRUCTION_PROOF_PASS",error)||
       !jsonBooleanFieldMatches(proof,"physical_admission",false,error)||
       !jsonUnsignedFieldMatches(proof,"runtime",0u,error)||
       !jsonUnsignedFieldMatches(proof,"accepted_steps",0u,error)||
       !jsonUnsignedFieldMatches(proof,"source_nodes",7233u,error)||
       !jsonUnsignedFieldMatches(proof,"source_tets",28632u,error)||
       !jsonUnsignedFieldMatches(proof,"ordered_edge_splits",258u,error)||
       !jsonUnsignedFieldMatches(proof,"refined_nodes",7491u,error)||
       !jsonUnsignedFieldMatches(proof,"refined_tets",29642u,error)||
       !jsonUnsignedFieldMatches(proof,"source_body_count",157u,error))return false;
    NamedInputFiles inputs;
    if(!readNamedInputFiles(proof,inputs,error))return false;
    const auto bind=[&](const std::string_view name,std::filesystem::path& path,std::string& sha){
        const auto it=inputs.find(std::string(name));
        if(it==inputs.end()){error="actual source-construction receipt is missing named input "+std::string(name);return false;}
        path=it->second.first;sha=it->second.second;return true;
    };
    PhysicalSkinConformingTissueRefinementAssets next=declared;
    if(!bind("refinement_lineage_npz",next.lineagePacketNPZ,next.lineagePacketSHA256)||
       !bind("unprojected_refined_control_npz",next.refinedVolumeNPZ,next.refinedVolumeSHA256)||
       !bind("refined_node",next.refinedTetgenNode,next.refinedNodeSHA256)||
       !bind("refined_element",next.refinedTetgenElements,next.refinedElementsSHA256)||
       !bind("refined_child_parent_ledger",next.childParentLedgerCSV,next.childParentLedgerSHA256)||
       !bind("534_donor_split_reference",next.childSourceDonorLedgerCSV,next.childSourceDonorLedgerSHA256)||
       !bind("node_source_csr_csv",next.nodeSourceAncestryCSV,next.nodeSourceAncestrySHA256)||
       !bind("ordered_split_edge_csv",next.orderedSplitEdgesCSV,next.orderedSplitEdgesSHA256)||
       !bind("refinement_export_receipt",next.producerReceipt,next.producerReceiptSHA256))return false;
    const auto source=inputs.find("source_candidate_npz");
    if(source==inputs.end()){error="actual source-construction receipt is missing source candidate NPZ";return false;}
    const auto immutableMass=inputs.find("immutable_330_parent_mass_packet");
    const auto immutableContributions=inputs.find("immutable_330_source_contributions");
    if(immutableMass==inputs.end()||immutableContributions==inputs.end()){
        error="actual source-construction proof omits immutable 330 mass/source-contribution pins";return false;
    }
    next.sourceVolumeSHA256=source->second.second;
    resolved=std::move(next);error.clear();return true;
}

inline bool nextWords(std::istringstream& input,std::vector<std::string>& words) {
    std::string line;
    while(std::getline(input,line)) {
        if(line.size()>16384u)return false;
        const auto comment=line.find('#');if(comment!=std::string::npos)line.resize(comment);
        std::istringstream row(line);words.clear();std::string word;
        while(row>>word)words.push_back(std::move(word));
        if(!words.empty())return true;
    }
    return false;
}

inline bool readRefinedNodes(const std::string& text,std::vector<V3>& points,
        std::unordered_map<std::uint32_t,std::uint32_t>& nodeIDToRow,std::string& error) {
    using physical_skin_child_mass_detail::parseNumber;
    std::istringstream input(text);std::vector<std::string> words;
    std::size_t count=0u,dimension=0u,attributes=0u,markers=0u;
    if(!nextWords(input,words)||words.size()!=4u||!parseNumber(words[0],count)||
       !parseNumber(words[1],dimension)||!parseNumber(words[2],attributes)||
       !parseNumber(words[3],markers)||count<4u||count>300000u||dimension!=3u||
       attributes!=0u||markers>1u) {
        error="refined conforming TetGen node header is invalid or outside the existing bounded domain";return false;
    }
    std::vector<V3> next(count);std::unordered_map<std::uint32_t,std::uint32_t> rows;rows.reserve(count);
    for(std::size_t row=0u;row<count;++row) {
        std::uint32_t id=0u;V3 original{},canonical{};
        if(!nextWords(input,words)||words.size()!=4u+markers||!parseNumber(words[0],id)||id!=row+1u||
           !rows.emplace(id,static_cast<std::uint32_t>(row)).second) {
            error="refined conforming TetGen node has a duplicate identity or invalid row";return false;
        }
        if(markers){std::int64_t marker=0;if(!parseNumber(words[4],marker)){error="invalid refined node marker";return false;}}
        for(unsigned axis=0u;axis<3u;++axis) {
            if(!parseNumber(words[axis+1u],original[axis])||!std::isfinite(original[axis])) {
                error="refined conforming TetGen node coordinate is not finite";return false;
            }
            canonical[axis]=double(static_cast<float>(original[axis]));
            if(!std::isfinite(canonical[axis])){error="refined node coordinate overflows Float32";return false;}
        }
        next[row]=canonical;
    }
    if(nextWords(input,words)){error="refined conforming TetGen node table has extra rows";return false;}
    points=std::move(next);nodeIDToRow=std::move(rows);return true;
}

inline bool readRefinedTets(const std::string& text,
                            const std::unordered_map<std::uint32_t,std::uint32_t>& nodeIDToRow,
                            std::vector<std::array<std::uint32_t,4>>& tets,
                            std::vector<std::uint32_t>& materials,std::string& error) {
    using physical_skin_child_mass_detail::parseNumber;
    std::istringstream input(text);std::vector<std::string> words;
    std::size_t count=0u,corners=0u,attributes=0u;
    if(!nextWords(input,words)||words.size()!=3u||!parseNumber(words[0],count)||
       !parseNumber(words[1],corners)||!parseNumber(words[2],attributes)||count==0u||
       count>200000u||corners!=4u||attributes!=1u) {
        error="refined conforming TetGen element header is invalid or exceeds the existing 200000-cell limit";return false;
    }
    std::vector<std::array<std::uint32_t,4>> cells(count);
    std::vector<std::uint32_t> mats(count);std::set<std::uint32_t> ids;
    for(std::size_t row=0u;row<count;++row) {
        std::uint32_t id=0u;double materialValue=0.0;std::array<std::uint32_t,4> tet{};
        if(!nextWords(input,words)||words.size()!=6u||!parseNumber(words[0],id)||id!=row+1u||
           !ids.insert(id).second||!parseNumber(words[5],materialValue)||!std::isfinite(materialValue)||
           materialValue<0.0||materialValue>=double(NM_INVALID_INDEX)||std::floor(materialValue)!=materialValue) {
            error="refined conforming TetGen element row has invalid identity or material";return false;
        }
        for(unsigned corner=0u;corner<4u;++corner) {
            std::uint32_t nodeID=0u;
            if(!parseNumber(words[corner+1u],nodeID)||!nodeIDToRow.contains(nodeID)) {
                error="refined conforming TetGen cell references a missing node";return false;
            }
            tet[corner]=nodeIDToRow.at(nodeID);
        }
        cells[row]=tet;mats[row]=static_cast<std::uint32_t>(materialValue);
    }
    if(nextWords(input,words)){error="refined conforming TetGen element table has extra rows";return false;}
    tets=std::move(cells);materials=std::move(mats);return true;
}

inline bool readRefinementLedger(const std::string& text,
        const PhysicalSkinChildMassMappingBundle& source,
        const std::vector<V3>& refinedPoints,
        const std::vector<std::array<std::uint32_t,4>>& refinedTets,
        const std::vector<std::uint32_t>& refinedMaterials,
        std::vector<std::uint32_t>& parentRows,std::vector<double>& fractions,
        std::vector<std::uint32_t>& material,std::vector<std::uint32_t>& donor,
        std::vector<std::uint32_t>& region,std::vector<std::uint32_t>& stable,
        std::string& error) {
    using physical_skin_child_mass_detail::parseNumber;
    using physical_skin_child_mass_detail::splitCSV;
    std::istringstream input(text);std::string line;
    const std::string header="child_tet_row,parent_tet_row,source_fraction,parent_reference_measure_m3,child_mass_kg,child_reference_volume_m3,material_index,geometric_donor_body,anatomical_region_id,stable_source_face_id";
    if(!std::getline(input,line)||splitCSV(line)!=std::vector<std::string>{
        "child_tet_row","parent_tet_row","source_fraction","parent_reference_measure_m3","child_mass_kg",
        "child_reference_volume_m3","material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"}) {
        error="refined 530 parent ledger header differs from the existing ten-column source ancestry contract";return false;
    }
    (void)header;
    std::vector<std::uint32_t> parents,mats,donors,regions,stableFaces;
    std::vector<double> sourceFractions;
    while(std::getline(input,line)) {
        if(line.empty())continue;
        if(line.size()>16384u){error="refined 530 ledger row exceeds the bounded line size";return false;}
        const auto f=splitCSV(line);
        if(f.size()!=10u){error="refined 530 ledger row has the wrong width";return false;}
        std::uint32_t row=0u,parent=0u,mat=0u,don=0u,reg=0u,face=0u;
        double fraction=0.0,immutableMeasure=0.0,childMass=0.0,childVolume=0.0;
        if(!parseNumber(f[0],row)||row!=parents.size()||!parseNumber(f[1],parent)||
           !parseNumber(f[2],fraction)||!parseNumber(f[3],immutableMeasure)||
           !parseNumber(f[4],childMass)||!parseNumber(f[5],childVolume)||
           !parseNumber(f[6],mat)||!parseNumber(f[7],don)||!parseNumber(f[8],reg)||
           !parseNumber(f[9],face)||row>=refinedTets.size()||parent>=source.childTetrahedra.size()||
           mat!=refinedMaterials[row]||mat!=source.childMaterialIndices[parent]||
           don!=source.childDonorBodyIndices[parent]||reg!=source.childAnatomicalRegionIndices[parent]||
           face!=source.childStableSourceFaceIdentities[parent]||
           !std::isfinite(fraction)||fraction<=0.0||!std::isfinite(childMass)||childMass<=0.0||
           !std::isfinite(immutableMeasure)||immutableMeasure<=0.0||
           !std::isfinite(childVolume)||childVolume<=0.0) {
            error="refined 530 ledger has invalid row, parent, labels, fraction, or measure";return false;
        }
        const double expectedMeasure=source.parent.referenceTetVolumeM3[parent]*fraction;
        const double expectedMass=source.parent.tetrahedronMassKg[parent]*fraction;
        const double geometric=signedReferenceTetVolumeM3(refinedTets[row],refinedPoints);
        if(std::abs(immutableMeasure-expectedMeasure)>std::max(1.0e-25,2.0e-11*expectedMeasure)||
           std::abs(childMass-expectedMass)>1.0e-13||
           !std::isfinite(geometric)||geometric<=0.0||
           std::abs(childVolume-geometric)>std::max(1.0e-25,2.0e-11*geometric)) {
            error="refined 530 ledger measure differs from immutable 330 mass or emitted reference geometry";return false;
        }
        parents.push_back(parent);sourceFractions.push_back(fraction);
        mats.push_back(mat);donors.push_back(don);regions.push_back(reg);stableFaces.push_back(face);
    }
    if(parents.size()!=refinedTets.size()){error="refined 530 ledger does not cover every output cell";return false;}
    parentRows=std::move(parents);fractions=std::move(sourceFractions);
    material=std::move(mats);donor=std::move(donors);region=std::move(regions);stable=std::move(stableFaces);
    return true;
}

inline bool verifyChildSourceDonorLedger(const std::string& text,
        const PhysicalSkinChildMassMappingBundle& source,
        const std::vector<std::uint32_t>& childParents,const std::vector<double>& fractions,
        const std::vector<std::uint32_t>& childMaterials,const std::vector<double>& childMassKg,
        std::vector<std::uint32_t>& offsets,std::vector<std::uint32_t>& donors,
        std::vector<double>& masses,std::string& error) {
    using physical_skin_child_mass_detail::parseNumber;
    using physical_skin_child_mass_detail::splitCSV;
    if(childParents.empty()||childParents.size()!=fractions.size()||
       childParents.size()!=childMaterials.size()||childParents.size()!=childMassKg.size()||
       source.parent.sourceMassContributions.empty()) {
        error="530 donor split requires complete child ancestry and immutable 330 source contributions";return false;
    }
    std::vector<ConservativeSpatialMassTransportParentNodeSourceMass> sourceRows;
    sourceRows.reserve(source.parent.sourceMassContributions.size());
    for(const auto& row:source.parent.sourceMassContributions)
        sourceRows.push_back({.parentNodeIndex=row.parentNodeIndex,.materialIndex=row.materialIndex,
            .sourceDonorBodyIndex=row.donorBodyIndex,.massKg=row.massKg});
    std::vector<std::uint32_t> parentOffsets,parentDonors;
    std::vector<double> parentDonorMass;
    if(!assembleConservativeSpatialMassTransportParentSourceDonorSplits(
            source.parent.tetrahedronMassKg,source.parent.materialIndices,source.parent.tetrahedra,
            sourceRows,1.0e-10,5.0e-4,parentOffsets,parentDonors,parentDonorMass,error)) {
        error="immutable 330 source donor split could not be reconstructed: "+error;return false;
    }
    std::vector<std::uint32_t> nextOffsets;nextOffsets.reserve(childParents.size()+1u);nextOffsets.push_back(0u);
    std::vector<std::uint32_t> nextDonors;std::vector<double> nextMasses;
    double maximumChildSplitError=0.0;
    for(std::size_t child=0u;child<childParents.size();++child) {
        const auto parent=childParents[child];
        if(parent+1u>=parentOffsets.size()||!std::isfinite(fractions[child])||fractions[child]<=0.0) {
            error="530 child donor ledger references an invalid immutable source cell or fraction";return false;
        }
        long double childTotal=0.0;
        for(std::uint32_t row=parentOffsets[parent];row<parentOffsets[parent+1u];++row) {
            const double mass=parentDonorMass[row]*fractions[child];
            if(!std::isfinite(mass)||mass<=0.0) {error="530 source donor split produced invalid child mass";return false;}
            nextDonors.push_back(parentDonors[row]);nextMasses.push_back(mass);childTotal+=mass;
        }
        maximumChildSplitError=std::max(maximumChildSplitError,
            std::abs(static_cast<double>(childTotal)-childMassKg[child]));
        if(maximumChildSplitError>1.0e-13) {
            error="530 child source donor shares do not close to the verified child mass";return false;
        }
        nextOffsets.push_back(static_cast<std::uint32_t>(nextDonors.size()));
    }
    std::istringstream input(text);std::string line;
    if(!std::getline(input,line)||splitCSV(line)!=std::vector<std::string>{
        "child_tet_row","material_index","source_donor_body","mass_kg"}) {
        error="534 child-source donor ledger header differs from its existing four-column source contract";return false;
    }
    std::size_t expectedRow=0u;
    for(std::size_t child=0u;child<childParents.size();++child) {
        for(std::uint32_t row=nextOffsets[child];row<nextOffsets[child+1u];++row,++expectedRow) {
            if(!std::getline(input,line)||line.size()>4096u) {error="534 source donor ledger is truncated";return false;}
            const auto fields=splitCSV(line);std::uint32_t cell=0u,material=0u,donor=0u;double mass=0.0;
            if(fields.size()!=4u||!parseNumber(fields[0],cell)||!parseNumber(fields[1],material)||
               !parseNumber(fields[2],donor)||!parseNumber(fields[3],mass)||
               cell!=child||material!=childMaterials[child]||donor!=nextDonors[row]||
               !std::isfinite(mass)||mass<=0.0||mass!=nextMasses[row]) {
                error="534 child-source donor row differs from the recomputed immutable 330 contribution split";return false;
            }
        }
    }
    if(std::getline(input,line)){error="534 source donor ledger has extra rows";return false;}
    offsets=std::move(nextOffsets);donors=std::move(nextDonors);masses=std::move(nextMasses);
    return true;
}

inline bool readNodeSourceCSR(const std::string& text,const std::size_t nodeCount,
        const std::size_t sourceNodeCount,std::vector<std::uint32_t>& offsets,
        std::vector<std::uint32_t>& rows,std::vector<double>& weights,std::string& error) {
    using physical_skin_child_mass_detail::parseNumber;
    using physical_skin_child_mass_detail::splitCSV;
    std::istringstream input(text);std::string line;
    if(!std::getline(input,line)||splitCSV(line)!=std::vector<std::string>{
        "volume_node_row","source_node_row","source_weight"}) {
        error="530 node-source CSR header differs from its existing exported field names";return false;
    }
    std::vector<std::vector<std::pair<std::uint32_t,double>>> byNode(nodeCount);
    std::size_t entries=0u;std::uint32_t previousNode=0u,previousSource=0u;bool first=true;
    while(std::getline(input,line)) {
        if(line.empty())continue;
        if(line.size()>4096u||++entries>5'000'000u){error="530 node-source CSR exceeds bounded input limits";return false;}
        const auto f=splitCSV(line);std::uint32_t node=0u,source=0u;double weight=0.0;
        if(f.size()!=3u||!parseNumber(f[0],node)||!parseNumber(f[1],source)||
           !parseNumber(f[2],weight)||node>=nodeCount||source>=sourceNodeCount||
           !std::isfinite(weight)||weight<=0.0||
           (!first&&(node<previousNode||(node==previousNode&&source<=previousSource)))) {
            error="530 node-source CSR has an invalid, duplicate, or unsorted row";return false;
        }
        byNode[node].push_back({source,weight});
        previousNode=node;previousSource=source;first=false;
    }
    offsets.clear();rows.clear();weights.clear();offsets.reserve(nodeCount+1u);offsets.push_back(0u);
    for(std::size_t node=0u;node<nodeCount;++node) {
        if(byNode[node].empty()){error="530 node-source CSR omits a volume node";return false;}
        double sum=0.0;
        for(const auto& [source,weight]:byNode[node]){rows.push_back(source);weights.push_back(weight);sum+=weight;}
        if(std::abs(sum-1.0)>1.0e-15){error="530 node-source CSR weights do not close to one";return false;}
        offsets.push_back(static_cast<std::uint32_t>(rows.size()));
    }
    return true;
}

inline bool readOrderedEdges(const std::string& text,const std::size_t sourceNodeCount,
        const std::size_t splitCount,std::vector<std::array<std::uint32_t,2>>& edges,std::string& error) {
    using physical_skin_child_mass_detail::parseNumber;
    using physical_skin_child_mass_detail::splitCSV;
    std::istringstream input(text);std::string line;
    if(!std::getline(input,line)||splitCSV(line)!=std::vector<std::string>{
        "volume_node_row","edge_node0","edge_node1"}) {
        error="530 ordered edge CSV header differs from its existing exported field names";return false;
    }
    std::vector<std::array<std::uint32_t,2>> next;next.reserve(splitCount);
    while(std::getline(input,line)) {
        if(line.empty())continue;
        const auto f=splitCSV(line);std::uint32_t row=0u,a=0u,b=0u;
        if(f.size()!=3u||!parseNumber(f[0],row)||!parseNumber(f[1],a)||!parseNumber(f[2],b)||
           row!=sourceNodeCount+next.size()||a>=b||b>=row) {
            error="530 ordered edge history has a noncanonical insertion row or edge endpoint";return false;
        }
        next.push_back({a,b});
    }
    if(next.size()!=splitCount){error="530 ordered edge history does not match the refined node count";return false;}
    edges=std::move(next);return true;
}

inline bool authenticateReceipts(const PhysicalSkinConformingTissueRefinementAssets& assets,
        const std::string_view sourceVolumeSHA256,std::string& error) {
    std::string exporter,self;
    if(!pinnedText(assets.producerReceipt,assets.producerReceiptSHA256,4u*1024u*1024u,exporter,error)||
       !pinnedText(assets.selfAuditReceipt,assets.selfAuditReceiptSHA256,4u*1024u*1024u,self,error))return false;
    if(!jsonStringFieldMatches(exporter,"status","SOURCE_530_NODE_CSR_AND_ORDERED_HISTORY_EXPORT",error)||
       !jsonStringFieldMatches(exporter,"source_npz_sha256",sourceVolumeSHA256,error)||
       !jsonStringFieldMatches(exporter,"refinement_npz_sha256",assets.lineagePacketSHA256,error)||
       !jsonBooleanFieldMatches(exporter,"source_reference_canonical_F32",true,error)||
       !jsonUnsignedFieldMatches(exporter,"nodes",7491u,error)||
       !jsonUnsignedFieldMatches(exporter,"cells",29642u,error)||
       !jsonUnsignedFieldMatches(exporter,"ordered_edges",258u,error)||
       !jsonUnsignedFieldMatches(exporter,"node_ancestry_entries",7749u,error)||
       !jsonStringFieldMatches(exporter,assets.nodeSourceAncestryCSV.string(),assets.nodeSourceAncestrySHA256,error)||
       !jsonStringFieldMatches(exporter,assets.orderedSplitEdgesCSV.string(),assets.orderedSplitEdgesSHA256,error))return false;
    if(!jsonStringFieldMatches(self,"status","exact_refined_material_boundary_self_audit",error)||
       !jsonStringFieldMatches(self,"candidate_sha256",assets.refinedVolumeSHA256,error)||
       !jsonStringFieldMatches(self,"refinement_source_sha256",assets.lineagePacketSHA256,error)||
       !jsonBooleanFieldMatches(self,"physical_admission",false,error)||
       !jsonZeroFields(self,"self_intersection_pairs",2u,error))return false;
    return true;
}
} // namespace physical_skin_conforming_refinement_assets_detail

[[nodiscard]] inline bool verifyPhysicalSkinConformingTissueRefinementAssets(
        const PhysicalSkinConformingTissueRefinementAssets& declaredAssets,
        const PhysicalSkinChildMassMappingBundle& source,
        const std::string_view sourceVolumeSHA256,const std::size_t sourceBodyCount,
        PhysicalSkinVerifiedConformingReference& output,std::string& error) {
    using namespace physical_skin_conforming_refinement_assets_detail;
    using physical_skin_child_mass_detail::validSHA256;
    const auto fail=[&](std::string text){error=std::move(text);return false;};
    PhysicalSkinConformingTissueRefinementAssets resolvedAssets;
    if(!resolveAssetsFromActualProofReceipt(declaredAssets,resolvedAssets,error))return false;
    const auto& assets=resolvedAssets;
    if(assets.sourceVolumeSHA256!=sourceVolumeSHA256)
        return fail("actual 530 proof source NPZ pin differs from the authenticated operational source");
    if(!validSHA256(sourceVolumeSHA256)||sourceBodyCount==0u||
       sourceBodyCount>std::numeric_limits<std::uint32_t>::max()||
       !validSHA256(source.parent.parentMassPacketSHA256)||
       !validSHA256(source.parent.parentSourceContributionSHA256)||
       source.referenceWorldPoints.empty()||source.childTetrahedra.empty()||
       source.parent.referenceTetVolumeM3.size()!=source.childTetrahedra.size()||
       source.parent.tetrahedronMassKg.size()!=source.childTetrahedra.size()||
       source.parent.tetrahedra!=source.childTetrahedra||
       source.parent.materialIndices!=source.childMaterialIndices||
       source.parent.donorBodyIndices!=source.childDonorBodyIndices||
       source.parent.anatomicalRegionIndices!=source.childAnatomicalRegionIndices||
       source.parent.stableSourceFaceIdentities!=source.childStableSourceFaceIdentities)
        return fail("530 refinement requires the authenticated source 002 mesh and immutable 330 cell ledger in exact row correspondence");
    if(!pinnedFile(assets.lineagePacketNPZ,assets.lineagePacketSHA256,error)||
       !pinnedFile(assets.refinedVolumeNPZ,assets.refinedVolumeSHA256,error))return false;
    std::string actualProofText;
    if(!pinnedText(declaredAssets.sourceConstructionProofReceipt,
            declaredAssets.sourceConstructionProofReceiptSHA256,8u*1024u*1024u,actualProofText,error))return false;
    NamedInputFiles actualInputs;
    if(!readNamedInputFiles(actualProofText,actualInputs,error))return false;
    const auto massPin=actualInputs.find("immutable_330_parent_mass_packet");
    const auto contributionPin=actualInputs.find("immutable_330_source_contributions");
    if(massPin==actualInputs.end()||contributionPin==actualInputs.end()||
       massPin->second.second!=source.parent.parentMassPacketSHA256||
       contributionPin->second.second!=source.parent.parentSourceContributionSHA256)
        return fail("actual 530 proof does not bind the immutable 330 mass and source-contribution packets used by this importer");
    std::string nodeText,elementText,ledgerText,donorLedgerText,nodeCSRTxt,edgeText;
    if(!pinnedText(assets.refinedTetgenNode,assets.refinedNodeSHA256,64u*1024u*1024u,nodeText,error)||
       !pinnedText(assets.refinedTetgenElements,assets.refinedElementsSHA256,64u*1024u*1024u,elementText,error)||
       !pinnedText(assets.childParentLedgerCSV,assets.childParentLedgerSHA256,64u*1024u*1024u,ledgerText,error)||
       !pinnedText(assets.childSourceDonorLedgerCSV,assets.childSourceDonorLedgerSHA256,64u*1024u*1024u,donorLedgerText,error)||
       !pinnedText(assets.nodeSourceAncestryCSV,assets.nodeSourceAncestrySHA256,64u*1024u*1024u,nodeCSRTxt,error)||
       !pinnedText(assets.orderedSplitEdgesCSV,assets.orderedSplitEdgesSHA256,64u*1024u*1024u,edgeText,error)||
       !authenticateReceipts(assets,sourceVolumeSHA256,error))return false;

    PhysicalSkinVerifiedConformingReference next;
    std::unordered_map<std::uint32_t,std::uint32_t> refinedNodeRows;
    if(!readRefinedNodes(nodeText,next.referenceWorldPoints,refinedNodeRows,error)||
       !readRefinedTets(elementText,refinedNodeRows,next.tetrahedra,next.materialIndices,error))return false;
    const auto refinedTetMaterials=next.materialIndices;
    std::vector<std::uint32_t> childParents,nodeCSRRows;
    std::vector<double> childFractions,nodeCSRWeights;
    if(!readRefinementLedger(ledgerText,source,next.referenceWorldPoints,next.tetrahedra,
            refinedTetMaterials,childParents,childFractions,next.materialIndices,
            next.geometricDonorBodyIndices,next.anatomicalRegionIndices,
            next.stableSourceFaceIds,error))return false;
    std::vector<std::uint32_t> nodeOffsets;
    if(!readNodeSourceCSR(nodeCSRTxt,next.referenceWorldPoints.size(),source.referenceWorldPoints.size(),
            nodeOffsets,nodeCSRRows,nodeCSRWeights,error))return false;
    if(next.referenceWorldPoints.size()<source.referenceWorldPoints.size())
        return fail("530 refined volume has fewer nodes than its pinned source topology");
    std::vector<std::array<std::uint32_t,2>> edges;
    if(!readOrderedEdges(edgeText,source.referenceWorldPoints.size(),
            next.referenceWorldPoints.size()-source.referenceWorldPoints.size(),edges,error))return false;

    std::vector<double> lower(17u,0.0),upper(17u,0.0);
    lower[15u]=900.0;upper[15u]=1100.0;lower[16u]=850.0;upper[16u]=950.0;
    const auto& parent=source.parent;
    ConformingTissueRefinementInput input{
        .sourceReferencePoints=source.referenceWorldPoints,
        .sourceTetrahedra=source.childTetrahedra,
        .orderedSplitEdges=edges,
        .refinedReferencePoints=next.referenceWorldPoints,
        .refinedTetrahedra=next.tetrahedra,
        .childSourceTetRows=childParents,
        .childSourceMeasureFractions=childFractions,
        .nodeSourceOffsets=nodeOffsets,
        .nodeSourceRows=nodeCSRRows,
        .nodeSourceWeights=nodeCSRWeights,
        .sourceMaterialIds=source.childMaterialIndices,
        .refinedMaterialIds=next.materialIndices,
        .sourceGeometricDonorIds=source.childDonorBodyIndices,
        .refinedGeometricDonorIds=next.geometricDonorBodyIndices,
        .sourceRegionIds=source.childAnatomicalRegionIndices,
        .refinedRegionIds=next.anatomicalRegionIndices,
        .sourceStableFaceIds=source.childStableSourceFaceIdentities,
        .refinedStableFaceIds=next.stableSourceFaceIds,
        .immutableParentReferenceVolumeM3=parent.referenceTetVolumeM3,
        .immutableParentMassKg=parent.tetrahedronMassKg,
        .materialDensityLowerKgM3=lower,.materialDensityUpperKgM3=upper,
        .sourceVolumeSHA256=sourceVolumeSHA256,
        .refinedVolumeSHA256=assets.refinedVolumeSHA256,
        .lineagePacketSHA256=assets.lineagePacketSHA256,
        .parentMassPacketSHA256=parent.parentMassPacketSHA256,
        .parentSourceContributionSHA256=parent.parentSourceContributionSHA256,
        .sourceBodyCount=static_cast<std::uint32_t>(sourceBodyCount)};
    if(!verifyConformingTissueRefinement(input,next.refinement,error))return false;
    if(!verifyChildSourceDonorLedger(donorLedgerText,source,childParents,childFractions,
            next.materialIndices,next.refinement.childMassKg,next.sourceDonorOffsets,
            next.sourceDonorBodyIndices,next.sourceDonorMassKg,error))return false;
    spatial_mass_transport_detail::SHA256Builder donorHash;
    donorHash.string("NumiHumanConformingTissueRefinementSourceDonor/v1");
    donorHash.string(assets.childSourceDonorLedgerSHA256);
    donorHash.string(next.refinement.verificationSHA256);
    for(std::size_t cell=0u;cell<next.refinement.childMassKg.size();++cell) {
        donorHash.u32(next.sourceDonorOffsets[cell]);donorHash.u32(next.sourceDonorOffsets[cell+1u]);
        for(auto row=next.sourceDonorOffsets[cell];row<next.sourceDonorOffsets[cell+1u];++row) {
            donorHash.u32(next.sourceDonorBodyIndices[row]);donorHash.real(next.sourceDonorMassKg[row]);
        }
    }
    next.refinedVolumeSHA256=assets.refinedVolumeSHA256;
    next.sourceDonorLedgerVerificationSHA256=donorHash.finish();
    output=std::move(next);error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue


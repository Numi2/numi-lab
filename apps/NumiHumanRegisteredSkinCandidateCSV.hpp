#pragma once
// Loading only: existing registered-surface CSV rows, never a simulation path.
#include "NumiHumanCoarseTissuePreparation.hpp"
#include <charconv>
#include <filesystem>
#include <fstream>
#include <map>
#include <string_view>

namespace metalrobo::human::coarse_tissue {
struct RegisteredSkinCandidateCSVReport {
    std::size_t nodeRows=0,faceRows=0,directPredecessorRows=0,derivedPredecessorRows=0;
    double maximumPredecessorProjectionDistanceM=0.0;
};
namespace registered_skin_candidate_csv_detail {
inline std::vector<std::string> split(std::string line) {
    if(!line.empty()&&line.back()=='\r')line.pop_back();
    std::vector<std::string> values;std::size_t begin=0;
    for(;;){const auto end=line.find(',',begin);values.push_back(line.substr(begin,end-begin));
        if(end==std::string::npos)break;begin=end+1;}
    return values;
}
template<class T> inline bool number(std::string_view s,T& v) {
    const auto [end,ec]=std::from_chars(s.data(),s.data()+s.size(),v);
    return ec==std::errc{}&&end==s.data()+s.size();
}
}
template<class SHA256OfPath> inline bool loadRegisteredSkinCandidateCSV(
    const std::filesystem::path& nodesPath,const std::filesystem::path& facesPath,
    const std::size_t bodyCount,const RegisteredSurface& identity,
    const std::string_view facesSHA256,const SHA256OfPath& sha256OfPath,
    RegisteredSurface& output,RegisteredSkinCandidateCSVReport& report,std::string& error) {
    using namespace registered_skin_candidate_csv_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(bodyCount==0||identity.sourcePayloadSHA256.size()!=64||identity.sourceMeshSHA256.size()!=64||
       identity.provider.empty()||identity.sourceMember.empty()||!identity.aggregateIdentity)
        return fail("registered skin CSV has incomplete separately pinned identity");
    if(facesSHA256.size()!=64||sha256OfPath(nodesPath)!=identity.sourceMeshSHA256||
       sha256OfPath(facesPath)!=facesSHA256)
        return fail("registered skin CSV source hash mismatch");
    RegisteredSurface next=identity;RegisteredSkinCandidateCSVReport measured;
    if(!next.worldPoints.empty()||!next.faces.empty()||!next.sourceVertexIndices.empty())
        return fail("registered skin CSV identity contains previous geometry");
    std::ifstream nodes(nodesPath),faces(facesPath);std::string line;
    if(!nodes||!faces||!std::getline(nodes,line))return fail("cannot read registered skin CSV");
    const auto names=split(line);std::map<std::string,std::size_t> columns;
    for(std::size_t i=0;i<names.size();++i)if(!columns.emplace(names[i],i).second)
        return fail("registered skin CSV repeats a column");
    for(const auto* key:{"vertex","donor","wx","wy","wz","nx","ny","nz","lx","ly","lz",
                        "m00","m01","m02","m10","m11","m12","m20","m21","m22",
                        "d37_source_vertex_row","d37_source_face_row","beta0","beta1","beta2","projection_distance_m"})
        if(!columns.contains(key))return fail("registered skin CSV misses required column "+std::string(key));
    while(std::getline(nodes,line)) {
        if(measured.nodeRows>=300000u||line.size()>16384)return fail("registered skin CSV capacity exceeded");
        const auto row=split(line);if(row.size()!=names.size())return fail("registered skin CSV row width mismatch");
        const auto get=[&](const char* key,auto& v){return number(row[columns.at(key)],v);};
        std::uint32_t id=0,donor=0;std::int64_t parentVertex=-1,parentFace=-1;
        V3 world{},local{},normal{},beta{};std::array<double,9> matrix{};double distance=0;
        if(!get("vertex",id)||id!=measured.nodeRows||!get("donor",donor)||donor>=bodyCount||
           !get("d37_source_vertex_row",parentVertex)||!get("d37_source_face_row",parentFace)||
           !get("projection_distance_m",distance)||!std::isfinite(distance)||distance<0)
            return fail("registered skin CSV invalid row identity or ancestry");
        for(unsigned a=0;a<3;++a){
            const auto axis=std::string(1,"xyz"[a]);
            if(!number(row[columns.at("w"+axis)],world[a])||!number(row[columns.at("l"+axis)],local[a])||
               !number(row[columns.at("n"+axis)],normal[a])||!get(("beta"+std::to_string(a)).c_str(),beta[a])||
               !std::isfinite(world[a])||double(float(world[a]))!=world[a]||
               !std::isfinite(local[a])||!std::isfinite(normal[a])||!std::isfinite(beta[a]))
                return fail("registered skin CSV nonfinite or invalid coordinate");
            for(unsigned b=0;b<3;++b)if(!number(row[columns.at("m"+std::to_string(a)+std::to_string(b))],matrix[3*a+b])||!std::isfinite(matrix[3*a+b]))
                return fail("registered skin CSV invalid donor frame");
        }
        if(std::abs(norm(normal)-1.0)>1e-5||parentVertex < -1||parentFace < -1||
           parentVertex>=61950||parentFace>=108987||(parentVertex>=0)==(parentFace>=0))return fail("registered skin CSV invalid normal or missing predecessor ancestry");
        if(parentFace>=0&&!validateBarycentricCoordinates(beta,error))return false;
        measured.directPredecessorRows+=parentVertex>=0;measured.derivedPredecessorRows+=parentVertex<0;
        measured.maximumPredecessorProjectionDistanceM=std::max(measured.maximumPredecessorProjectionDistanceM,distance);
        next.worldPoints.push_back(world);next.donorLocalPoints.push_back(local);next.outwardWorldNormals.push_back(normal);
        next.donorBodyIndices.push_back(donor);next.worldToDonorLocalLinear.push_back(matrix);
        // Current NHSKIN row IDs are distinct from recorded predecessor IDs.
        next.sourceVertexIndices.push_back(id);++measured.nodeRows;
    }
    if(!nodes.eof()||measured.nodeRows<4||!std::getline(faces,line)||
       split(line)!=std::vector<std::string>{"face","v0","v1","v2","baseline806_source_face_row"})
        return fail("registered skin CSV missing rows or wrong face schema");
    while(std::getline(faces,line)) {
        const auto row=split(line);Face f{};std::uint32_t id=0;std::int64_t predecessor=-1;
        if(row.size()!=5||!number(row[0],id)||id!=measured.faceRows||
           !number(row[4],predecessor)||predecessor < -1||measured.faceRows>=600000u)
            return fail("registered skin face CSV invalid identity or capacity");
        for(unsigned c=0;c<3;++c)if(!number(row[c+1],f[c])||f[c]>=measured.nodeRows)
            return fail("registered skin face CSV invalid corner");
        if(f[0]==f[1]||f[0]==f[2]||f[1]==f[2]||
           norm(cross(sub(next.worldPoints[f[1]],next.worldPoints[f[0]]),sub(next.worldPoints[f[2]],next.worldPoints[f[0]])))<=0)
            return fail("registered skin face CSV degenerate face");
        next.faces.push_back(f);next.sourceFaceIndices.push_back(id);++measured.faceRows;
    }
    if(!faces.eof()||measured.faceRows==0)return fail("registered skin face CSV incomplete");
    output=std::move(next);report=measured;error.clear();return true;
}
} // namespace metalrobo::human::coarse_tissue

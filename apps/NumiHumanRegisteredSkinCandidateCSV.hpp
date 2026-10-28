#pragma once
// Loading only: existing registered-surface CSV rows, never a simulation path.
#include "NumiHumanCoarseTissuePreparation.hpp"
#include "NumiHumanSkinRigidPlacement.hpp"
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
    if(!nodes.eof()||measured.nodeRows<4||!std::getline(faces,line))
        return fail("registered skin CSV missing rows or wrong face schema");
    const auto faceColumns=split(line);
    const bool legacyFaces=faceColumns==std::vector<std::string>{
        "face","v0","v1","v2","baseline806_source_face_row"};
    const bool conditionedFaces=faceColumns==std::vector<std::string>{
        "face","v0","v1","v2","O033_source_face_parent_row","baseline806_source_face_row_exact"};
    if(!legacyFaces&&!conditionedFaces)
        return fail("registered skin CSV missing rows or wrong face schema");
    while(std::getline(faces,line)) {
        if(line.size()>16384u)return fail("registered skin face CSV oversized row");
        const auto row=split(line);Face f{};std::uint32_t id=0;
        if(row.size()!=faceColumns.size()||!number(row[0],id)||id!=measured.faceRows||
           measured.faceRows>=600000u)
            return fail("registered skin face CSV invalid identity or capacity");
        // The conditioned format retains both the immediate O033 parent and
        // the exact older source row. Full union ancestry remains hash-pinned
        // separately; neither row is silently promoted to a unique source.
        for(std::size_t column=4;column<row.size();++column){
            std::int64_t predecessor=-1;
            if(!number(row[column],predecessor)||predecessor < -1)
                return fail("registered skin face CSV invalid predecessor identity");
        }
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
// Source-only admission uses the same common rigid placement as the physical
// skin compiler. Support first-touch placement may translate the entire body;
// a donor-specific change must still fail the original two-micrometre gate.
struct RegisteredSkinCandidatePlacementReport {
    RegisteredSkinRigidPlacement placement;
    double maximumLocalResidualM=0,maximumSkinningResidualM=0,maximumFrameResidual=0;
    std::size_t maximumLocalRow=0,maximumSkinningRow=0,maximumFrameRow=0;
};
template<class CurrentPoint,class CurrentLocal,class CurrentFrame>
inline bool validateRegisteredSkinCandidateNativePlacement(
    const RegisteredSurface& reference,const CurrentPoint& pointAt,
    const CurrentLocal& localAt,const CurrentFrame& frameAt,
    RegisteredSkinCandidatePlacementReport& output,std::string& error) {
    const auto n=reference.worldPoints.size();
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(n==0||reference.donorLocalPoints.size()!=n||reference.donorBodyIndices.size()!=n||
       reference.worldToDonorLocalLinear.size()!=n)
        return fail("candidate skin placement has incomplete source domains");
    RegisteredSkinCandidatePlacementReport next;
    if(!deriveRegisteredSkinRigidPlacement(reference.worldPoints[0],
        reference.worldToDonorLocalLinear[0],pointAt(0),frameAt(0),next.placement,error))return false;
    for(std::size_t row=0;row<n;++row) {
        const auto expected=next.placement.point(reference.worldPoints[row]);
        const auto point=pointAt(row),local=localAt(row,expected);
        const auto frame=frameAt(row);
        for(unsigned c=0;c<3;++c)
            if(!std::isfinite(point[c])||!std::isfinite(local[c]))
                return fail("candidate skin native placement has a nonfinite point");
        const double localResidual=norm(sub(local,reference.donorLocalPoints[row]));
        const double skinningResidual=norm(sub(point,expected));
        if(localResidual>next.maximumLocalResidualM){next.maximumLocalResidualM=localResidual;next.maximumLocalRow=row;}
        if(skinningResidual>next.maximumSkinningResidualM){next.maximumSkinningResidualM=skinningResidual;next.maximumSkinningRow=row;}
        const auto expectedFrame=next.placement.donorFrame(reference.worldToDonorLocalLinear[row]);
        for(unsigned c=0;c<9;++c) {
            if(!std::isfinite(frame[c]))return fail("candidate skin native placement has a nonfinite donor frame");
            const double residual=std::abs(frame[c]-expectedFrame[c]);
            if(residual>next.maximumFrameResidual){next.maximumFrameResidual=residual;next.maximumFrameRow=row;}
        }
    }
    if(next.maximumLocalResidualM>2.0e-6||next.maximumSkinningResidualM>2.0e-6||
       next.maximumFrameResidual>2.0e-6)
        return fail("candidate skin reference is not a common rigid placement of the native parent cook");
    next.placement.validatedVertices=n;
    output=next;error.clear();return true;
}
} // namespace metalrobo::human::coarse_tissue

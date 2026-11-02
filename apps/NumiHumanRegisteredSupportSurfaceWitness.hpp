#pragma once
#include "numi/matter/fem_body_triangle_contact_sample.hpp"
#include "NumiHumanInitialSkinBoneContactWitness.hpp"
#include <array>
#include <algorithm>
#include <span>
#include <cmath>
#include <limits>
#include <cstdint>
// Observational NHCNT-to-current-O registration. Physical bed contact continues
// to cover every current O FEM node and does not consume this representative.
namespace metalrobo::human::coarse_tissue {
struct RegisteredSupportSourceInfluence {std::uint32_t body=NM_INVALID_INDEX;float weight=0.0f;};
// Use the source's existing four-influence support at blended joints. Full86
// payloads select their four strongest lanes deterministically; tiny positive
// tails cannot authorize an unrelated body. This changes no physical owner.
template<class Skin> inline bool registeredSupportSourceInfluences(const Skin& skin,
    std::size_t vertex,std::array<RegisteredSupportSourceInfluence,4>& output){
    if(vertex>=skin.vertices.size()||skin.bindings.empty())return false;
    const bool full=!skin.fullWeights.empty();
    if(full&&skin.fullWeights.size()!=skin.vertices.size()*skin.bindings.size())return false;
    std::vector<std::pair<RegisteredSupportSourceInfluence,std::size_t>> rows;
    const auto& v=skin.vertices[vertex];const std::size_t count=full?skin.bindings.size():4u;
    for(std::size_t lane=0;lane<count;++lane){const auto index=full?lane:v.bindingIndex[lane];
        if(index>=skin.bindings.size())return false;
        const float weight=full?skin.fullWeights[vertex*count+lane]:v.weight[lane];
        if(!std::isfinite(weight)||weight<0.0f||weight>1.0f)return false;
        if(weight>0.0f)rows.push_back({{skin.bindings[index].bodyIndex,weight},lane});}
    std::sort(rows.begin(),rows.end(),[](const auto& a,const auto& b){return a.first.weight>b.first.weight||(a.first.weight==b.first.weight&&a.second<b.second);});
    std::array<RegisteredSupportSourceInfluence,4> result{};
    for(std::size_t i=0;i<std::min(result.size(),rows.size());++i)result[i]=rows[i].first;
    if(rows.empty())return false;output=result;return true;
}

struct RegisteredSupportSurfaceWitness {
    std::uint32_t faceRow=NM_INVALID_INDEX,representativeRow=NM_INVALID_INDEX;
    std::array<double,3> point{},barycentric{};
    double surfaceDistanceM=std::numeric_limits<double>::infinity();
    double representativeDistanceM=std::numeric_limits<double>::infinity();
};
inline bool findRegisteredSupportSurfaceWitness(
    std::span<const std::array<double,3>> points,
    std::span<const std::array<std::uint32_t,3>> faces,
    std::span<const std::uint32_t> donors,std::uint32_t donor,
    const std::array<double,3>& query,double maximumDistanceM,
    RegisteredSupportSurfaceWitness& output){
    namespace witness=numi::human::initial_skin_bone_contact_witness;
    if(points.size()!=donors.size()||donor==NM_INVALID_INDEX||
       !std::isfinite(maximumDistanceM)||maximumDistanceM<=0.0||
       !std::all_of(query.begin(),query.end(),[](double v){return std::isfinite(v);}))return false;
    RegisteredSupportSurfaceWitness best;
    const witness::Vec3 p{query[0],query[1],query[2]};
    for(std::size_t fi=0;fi<faces.size();++fi){
        const auto f=faces[fi];
        if(std::any_of(f.begin(),f.end(),[&](auto v){return v>=points.size();}))return false;
        if(std::any_of(f.begin(),f.end(),[&](auto v){return donors[v]!=donor;}))continue;
        const auto xyz=[&](auto v){const auto& x=points[v];return witness::Vec3{x[0],x[1],x[2]};};
        const auto a=xyz(f[0]),b=xyz(f[1]),c=xyz(f[2]);
        if(!witness::finite(a)||!witness::finite(b)||!witness::finite(c)||
           witness::norm2(witness::cross(witness::sub(b,a),witness::sub(c,a)))<=0.0)return false;
        const auto closest=witness::closestPointOnTriangle(p,a,b,c);
        if(!closest.valid||!witness::finite(closest.point))return false;
        const double distance=witness::norm(witness::sub(closest.point,p));
        if(!std::isfinite(distance)||distance>maximumDistanceM||distance>=best.surfaceDistanceM)continue;
        best.faceRow=static_cast<std::uint32_t>(fi);best.point={closest.point.x,closest.point.y,closest.point.z};
        best.barycentric=closest.bary;best.surfaceDistanceM=distance;
        best.representativeRow=NM_INVALID_INDEX;best.representativeDistanceM=std::numeric_limits<double>::infinity();
        for(auto v:f){const double d=witness::norm(witness::sub(xyz(v),p));
            if(d<best.representativeDistanceM||(d==best.representativeDistanceM&&v<best.representativeRow)){
                best.representativeRow=v;best.representativeDistanceM=d;}}
    }
    if(best.faceRow==NM_INVALID_INDEX||best.representativeRow==NM_INVALID_INDEX)return false;
    output=best;return true;
}
} // namespace metalrobo::human::coarse_tissue

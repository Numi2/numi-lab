#pragma once

#include "numi/matter/fem_body_triangle_contact_neighborhood_validation.hpp"
#include "numi/matter/fem_body_contact_static_bvh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include <ostream>
#include <span>
#include <tuple>

namespace numi::human::initial_skin_bone_contact_witness {

struct Vec3 { double x=0.0, y=0.0, z=0.0; };
enum class Feature : std::uint8_t { face, edge, vertex };
struct Closest { Vec3 point{}; std::array<double,3> bary{}; Feature feature=Feature::face; bool valid=true; };
struct Summary {
    std::size_t samples=0, penetrating=0, invalidSamples=0, invalidTriangles=0, envelopeEscapes=0;
    std::size_t outerPenetrating=0, deepPenetrating=0;
    double maxClosureM=0.0, maxForceN=0.0, sumForceMagnitudeN=0.0;
    double maxLocalDriftM=0.0, maxDistanceSquaredAgreement=0.0;
    std::uint32_t maxSample=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t maxNode=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t maxRow=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t maxLayer=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t maxBody=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t maxBone=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t maxTriangle=std::numeric_limits<std::uint32_t>::max();
    double maxSignedDistanceM=0.0;
    double maxAreaM2=0.0;
    double maxStiffnessPaPerM=0.0;
    Vec3 maxPoint{}, maxClosest{}, maxNormal{}, maxForce{};
};

inline Vec3 add(Vec3 a,Vec3 b) noexcept { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline Vec3 sub(Vec3 a,Vec3 b) noexcept { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline Vec3 mul(Vec3 a,double s) noexcept { return {a.x*s,a.y*s,a.z*s}; }
inline double dot(Vec3 a,Vec3 b) noexcept { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline Vec3 cross(Vec3 a,Vec3 b) noexcept {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
inline double norm2(Vec3 a) noexcept { return dot(a,a); }
inline double norm(Vec3 a) noexcept { return std::sqrt(norm2(a)); }
inline bool finite(Vec3 a) noexcept {
    return std::isfinite(a.x)&&std::isfinite(a.y)&&std::isfinite(a.z);
}

// Same double-precision Voronoi-region closest-point sequence as
// tests/numi_human_fem_body_triangle_contact/support/
// fem_body_triangle_contact_math.hpp::closestPointOnTriangle.
inline Closest closestPointOnTriangle(Vec3 p,Vec3 a,Vec3 b,Vec3 c) noexcept {
    const Vec3 ab=sub(b,a),ac=sub(c,a),ap=sub(p,a);
    const double d1=dot(ab,ap),d2=dot(ac,ap);
    if(d1<=0.0&&d2<=0.0)return {a,{1.0,0.0,0.0},Feature::vertex};
    const Vec3 bp=sub(p,b);
    const double d3=dot(ab,bp),d4=dot(ac,bp);
    if(d3>=0.0&&d4<=d3)return {b,{0.0,1.0,0.0},Feature::vertex};
    const double vc=d1*d4-d3*d2;
    if(vc<=0.0&&d1>=0.0&&d3<=0.0) {
        const double v=d1/(d1-d3);
        return {add(a,mul(ab,v)),{1.0-v,v,0.0},Feature::edge};
    }
    const Vec3 cp=sub(p,c);
    const double d5=dot(ab,cp),d6=dot(ac,cp);
    if(d6>=0.0&&d5<=d6)return {c,{0.0,0.0,1.0},Feature::vertex};
    const double vb=d5*d2-d1*d6;
    if(vb<=0.0&&d2>=0.0&&d6<=0.0) {
        const double w=d2/(d2-d6);
        return {add(a,mul(ac,w)),{1.0-w,0.0,w},Feature::edge};
    }
    const double va=d3*d6-d5*d4;
    if(va<=0.0&&(d4-d3)>=0.0&&(d5-d6)>=0.0) {
        const double w=(d4-d3)/((d4-d3)+(d5-d6));
        return {add(b,mul(sub(c,b),w)),{0.0,1.0-w,w},Feature::edge};
    }
    const double denom=va+vb+vc;
    if(!(denom>0.0)||!std::isfinite(denom))
        return {{},{},Feature::face,false};
    const double inv=1.0/denom,v=vb*inv,w=vc*inv;
    return {add(a,add(mul(ab,v),mul(ac,w))),{1.0-v-w,v,w},Feature::face};
}

inline Vec3 read3(nm_float4 p) noexcept { return {p.x,p.y,p.z}; }
inline Vec3 rotate(const std::array<double,4>& q,Vec3 v) noexcept {
    const Vec3 xyz{q[0],q[1],q[2]};
    const Vec3 t=mul(cross(xyz,v),2.0);
    return add(v,add(mul(t,q[3]),cross(xyz,t)));
}

template<class NodeState,class BodyState>
Summary write(
    std::ostream& out,
    std::span<const NMNumiHumanFEMBodyContactSampleGPU> samples,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> ranges,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> triangles,
    std::span<const NodeState> nodes,
    std::span<const BodyState> bodies,
    std::uint32_t objectStateOffset,
    std::uint32_t nodesPerLayer
) {
    Summary result{};
    out<<"sample,node,layer,row,body,bone,triangle,feature,distance_m,signed_distance_m,closure_m,area_m2,stiffness_pa_per_m,force_n,force_x_n,force_y_n,force_z_n,actual_local_x,actual_local_y,actual_local_z,range_local_x,range_local_y,range_local_z,local_drift_m,envelope_m,inside_envelope,closest_x,closest_y,closest_z,normal_x,normal_y,normal_z,bary0,bary1,bary2,distance2_contract_error_m2\\n";
    if(samples.size()!=ranges.size()||nodesPerLayer==0u)return result;
    out<<std::setprecision(17);
    for(std::size_t si=0;si<samples.size();++si) {
        const auto& range=ranges[si];
        if(range.reserved0!=NM_NUMI_HUMAN_FEM_BODY_CONTACT_NEIGHBORHOOD_MOTION_ENVELOPE||
           range.triangleCount==0u)continue;
        ++result.samples;
        const auto& sample=samples[si];
        if(sample.slaveNode>=nodes.size()||sample.bodyIndex>=bodies.size()||
           std::uint64_t(range.firstTriangle)+range.triangleCount>triangles.size()) {
            ++result.invalidSamples; continue;
        }
        const std::uint64_t local64=std::uint64_t(sample.slaveNode)-objectStateOffset;
        if(sample.slaveNode<objectStateOffset||local64>=3ull*nodesPerLayer) {
            ++result.invalidSamples; continue;
        }
        const auto layer=static_cast<std::uint32_t>(local64/nodesPerLayer);
        const auto row=static_cast<std::uint32_t>(local64%nodesPerLayer);
        const auto& body=bodies[sample.bodyIndex];
        const auto& node=nodes[sample.slaveNode];
        const Vec3 world=read3(node.positionAndMass);
        const Vec3 bodyPosition=read3(body.position);
        const std::array<double,4> inverseQ{{-body.orientation.x,-body.orientation.y,
            -body.orientation.z,body.orientation.w}};
        const Vec3 exactLocal=rotate(inverseQ,sub(world,bodyPosition));
        const Vec3 actualLocal{float(exactLocal.x),float(exactLocal.y),float(exactLocal.z)};
        const Vec3 restLocal{range.restPointAndMotionRadius.x,
            range.restPointAndMotionRadius.y,range.restPointAndMotionRadius.z};
        const double drift=norm(sub(actualLocal,restLocal));
        result.maxLocalDriftM=std::max(result.maxLocalDriftM,drift);
        if(!finite(actualLocal)||!finite(world)||!std::isfinite(drift)) {
            ++result.invalidSamples;continue;
        }
        if(drift>double(range.restPointAndMotionRadius.w)+1.0e-7)
            ++result.envelopeEscapes;
        double bestD2=std::numeric_limits<double>::infinity();
        Closest best{};
        Vec3 bestNormal{};
        Vec3 bestDelta{};
        std::uint32_t bestBone=std::numeric_limits<std::uint32_t>::max();
        std::uint32_t bestTri=std::numeric_limits<std::uint32_t>::max();
        for(std::uint32_t local=0;local<range.triangleCount;++local) {
            const auto& tri=triangles[range.firstTriangle+local];
            if(tri.bodyIndex!=sample.bodyIndex) {++result.invalidSamples;continue;}
            const Vec3 a=read3(tri.bodyLocalVertex0),b=read3(tri.bodyLocalVertex1),
                       c=read3(tri.bodyLocalVertex2);
            const Vec3 raw=cross(sub(b,a),sub(c,a));
            const double area2=norm(raw);
            if(!finite(a)||!finite(b)||!finite(c)||!(area2>0.0)||!std::isfinite(area2)) {
                ++result.invalidSamples;continue;
            }
            const auto closest=closestPointOnTriangle(actualLocal,a,b,c);
            if(!closest.valid) {++result.invalidTriangles;continue;}
            const Vec3 delta=sub(actualLocal,closest.point);
            const double d2=norm2(delta);
            if(!finite(closest.point)||!std::isfinite(d2)||d2<0.0) {++result.invalidSamples;continue;}
            const auto identity=std::pair{tri.boneStableId,tri.triangleStableId};
            const auto bestIdentity=std::pair{bestBone,bestTri};
            if(d2<bestD2||(d2==bestD2&&identity<bestIdentity)) {
                bestD2=d2;best=closest;bestDelta=delta;bestNormal=mul(raw,1.0/area2);
                bestBone=tri.boneStableId;bestTri=tri.triangleStableId;
            }
        }
        if(!std::isfinite(bestD2)) {++result.invalidSamples;continue;}
        const double distance=std::sqrt(bestD2);
        const double orientation=dot(bestDelta,bestNormal)<0.0?-1.0:1.0;
        const double signedDistance=distance>1.0e-12?orientation*distance:0.0;
        const double closure=std::max(0.0,-signedDistance-1.0e-7);
        const double area=sample.bodyLocalPointAndArea.w;
        const double stiffness=sample.bodyLocalNormalAndReferenceSeparation.w;
        const double force=stiffness*closure*area;
        if(!std::isfinite(distance)||!std::isfinite(signedDistance)||!std::isfinite(closure)||
           !std::isfinite(area)||!(area>0.0)||!std::isfinite(stiffness)||!(stiffness>0.0)||
           !std::isfinite(force)) {++result.invalidSamples;continue;}
        const Vec3 normal=distance>1.0e-12?mul(bestDelta,orientation/distance):bestNormal;
        const Vec3 forceVector=mul(normal,force);
                const nm_float4 queryF32{float(actualLocal.x),float(actualLocal.y),float(actualLocal.z),0.0f};
                // Cross-check the selected distance against the same CPU double
        // distance helper used to build the certified closest-point envelope.
        NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU selected{};
        for(std::uint32_t local=0;local<range.triangleCount;++local) {
            const auto& tri=triangles[range.firstTriangle+local];
            if(tri.boneStableId==bestBone&&tri.triangleStableId==bestTri) {selected=tri;break;}
        }
        const double contractD2=numi::matter::fem_body_contact_neighborhood::pointTriangleDistanceSquared(queryF32,selected);
        const double agreement=std::abs(contractD2-bestD2);
        result.maxDistanceSquaredAgreement=std::max(result.maxDistanceSquaredAgreement,agreement);
        if(closure>0.0) {
            ++result.penetrating;
            if(layer==0u)++result.outerPenetrating; else if(layer==2u)++result.deepPenetrating;
            result.sumForceMagnitudeN+=force;
            if(closure>result.maxClosureM) {
                result.maxClosureM=closure;result.maxForceN=force;
                result.maxSample=static_cast<std::uint32_t>(si);result.maxNode=sample.slaveNode;
                result.maxRow=row;result.maxLayer=layer;result.maxBody=sample.bodyIndex;
                result.maxBone=bestBone;result.maxTriangle=bestTri;
                result.maxSignedDistanceM=signedDistance;result.maxAreaM2=area;
                result.maxStiffnessPaPerM=stiffness;result.maxPoint=actualLocal;
                result.maxClosest=best.point;result.maxNormal=normal;result.maxForce=forceVector;
            }
        }
        out<<si<<","<<sample.slaveNode<<","<<layer<<","<<row<<","<<sample.bodyIndex<<","
           <<bestBone<<","<<bestTri<<","<<static_cast<unsigned>(best.feature)<<","
           <<distance<<","<<signedDistance<<","<<closure<<","<<area<<","<<stiffness<<","
           <<force<<","<<forceVector.x<<","<<forceVector.y<<","<<forceVector.z<<","
           <<actualLocal.x<<","<<actualLocal.y<<","<<actualLocal.z<<","
           <<restLocal.x<<","<<restLocal.y<<","<<restLocal.z<<","<<drift<<","
           <<range.restPointAndMotionRadius.w<<","
           <<(drift<=double(range.restPointAndMotionRadius.w)+1.0e-7?1:0)<<","
           <<best.point.x<<","<<best.point.y<<","<<best.point.z<<","
           <<normal.x<<","<<normal.y<<","<<normal.z<<","
           <<best.bary[0]<<","<<best.bary[1]<<","<<best.bary[2]<<","<<agreement<<"\n";
    }
    return result;
}

struct SolidParity { int insideRays=0; int outsideRays=0; bool ambiguous=false; };
template<class TriangleRange>
SolidParity classifyClosedSolid(Vec3 point,const TriangleRange& triangles) noexcept {
    constexpr std::array<Vec3,3> directions{{
        {1.0,0.3713906763541037,0.217831},
        {-0.231117,1.0,0.417219},
        {0.113117,-0.287119,1.0}}};
    SolidParity result{};
    for(const Vec3 direction:directions) {
        std::size_t intersections=0u;bool ambiguous=false;
        for(const auto& triangle:triangles) {
            const Vec3 a=read3(triangle.bodyLocalVertex0),
                b=read3(triangle.bodyLocalVertex1),c=read3(triangle.bodyLocalVertex2);
            const Vec3 e1=sub(b,a),e2=sub(c,a),h=cross(direction,e2);
            const double det=dot(e1,h),detScale=norm(e1)*norm(e2)*norm(direction);
            if(!std::isfinite(det)||!(detScale>0.0)){ambiguous=true;continue;}
            if(std::abs(det)<=1.0e-14*detScale)continue;
            const double inv=1.0/det;const Vec3 s=sub(point,a);
            const double u=dot(s,h)*inv;const Vec3 q=cross(s,e1);
            const double v=dot(direction,q)*inv,t=dot(e2,q)*inv;
            if(!(t>1.0e-10)||u<0.0||v<0.0||u+v>1.0)continue;
            constexpr double eps=1.0e-10;
            if(u<=eps||v<=eps||1.0-u-v<=eps){ambiguous=true;continue;}
            ++intersections;
        }
        if(ambiguous){result.ambiguous=true;continue;}
        if((intersections&1u)!=0u)++result.insideRays;else ++result.outsideRays;
    }
    return result;
}

struct FullBankComparison {
    bool valid=false;
    std::size_t bankTriangleCount=0u;
    std::uint32_t sample=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t slaveNode=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t body=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t boundedBone=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t boundedTriangle=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t fullBone=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t fullTriangle=std::numeric_limits<std::uint32_t>::max();
    Feature boundedFeature=Feature::face, fullFeature=Feature::face;
    double boundedDistanceM=0.0, fullDistanceM=0.0;
    double boundedFaceDot=0.0, boundedPseudoDot=0.0;
    double fullFaceDot=0.0, fullPseudoDot=0.0;
    SolidParity fullSolid{};
    std::string failure;
};

inline const char* featureName(Feature feature) noexcept {
    switch(feature) {
        case Feature::face:return "face";
        case Feature::edge:return "edge";
        case Feature::vertex:return "vertex";
    }
    return "invalid";
}

inline bool featurePseudoNormal(
    const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU& triangle,
    const Closest& closest,Vec3& result
) noexcept {
    if(closest.feature==Feature::face) {
        result=read3(triangle.authoredOutwardNormal);
        return finite(result)&&std::abs(norm(result)-1.0)<1.0e-4;
    }
    unsigned featureIndex=0u;
    if(closest.feature==Feature::edge) {
        if(closest.bary[2]==0.0)featureIndex=0u;
        else if(closest.bary[0]==0.0)featureIndex=1u;
        else if(closest.bary[1]==0.0)featureIndex=2u;
        else return false;
    } else {
        featureIndex=static_cast<unsigned>(std::distance(
            closest.bary.begin(),std::max_element(closest.bary.begin(),closest.bary.end())));
    }
    if((triangle.featureNormalFlags&(1u<<featureIndex))==0u)return false;
    if(closest.feature==Feature::edge) {
        if(featureIndex==0u)result=read3(triangle.edgePseudoNormal01);
        else if(featureIndex==1u)result=read3(triangle.edgePseudoNormal12);
        else result=read3(triangle.edgePseudoNormal20);
    } else {
        if(featureIndex==0u)result=read3(triangle.vertexPseudoNormal0);
        else if(featureIndex==1u)result=read3(triangle.vertexPseudoNormal1);
        else result=read3(triangle.vertexPseudoNormal2);
    }
    return finite(result)&&std::abs(norm(result)-1.0)<1.0e-4;
}

template<class NodeState,class BodyState,class BodyBanks,class Eligibility>
FullBankComparison writeFullBankComparison(
    std::ostream& out,
    std::span<const NMNumiHumanFEMBodyContactSampleGPU> samples,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> ranges,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> triangles,
    std::span<const NodeState> nodes,
    std::span<const BodyState> bodies,
    const BodyBanks& bodyBanks,
    const Eligibility& bodyBankEligible,
    std::uint32_t objectStateOffset,
    std::uint32_t nodesPerLayer,
    std::uint32_t sampleIndex,
    bool writeHeader=true
) {
    FullBankComparison result{};
    if(writeHeader)out<<"sample,slave_node,layer,row,body,query_x,query_y,query_z,"
          "bounded_bone,bounded_triangle,bounded_feature,bounded_d_m,bounded_face_dot,"
          "bounded_pseudonormal_dot,bounded_closest_x,bounded_closest_y,bounded_closest_z,"
          "full_triangle_count,full_bone,full_triangle,full_feature,full_d_m,full_face_dot,"
          "full_pseudonormal_dot,full_closest_x,full_closest_y,full_closest_z,"
          "full_inside_rays,full_outside_rays,full_ambiguous,same_winner\n";
    if(sampleIndex>=samples.size()||sampleIndex>=ranges.size()||nodesPerLayer==0u) {
        result.failure="sample_index_or_layer_extent_invalid"; return result;
    }
    const auto& sample=samples[sampleIndex];
    const auto& range=ranges[sampleIndex];
    if(range.reserved0!=NM_NUMI_HUMAN_FEM_BODY_CONTACT_NEIGHBORHOOD_MOTION_ENVELOPE||
       range.triangleCount==0u||sample.slaveNode>=nodes.size()||sample.bodyIndex>=bodies.size()||
       sample.bodyIndex>=bodyBanks.size()||sample.bodyIndex>=bodyBankEligible.size()||
       !bodyBankEligible[sample.bodyIndex]||
       std::uint64_t(range.firstTriangle)+range.triangleCount>triangles.size()) {
        result.failure="sample_range_or_body_bank_invalid";return result;
    }
    const std::uint64_t local64=std::uint64_t(sample.slaveNode)-objectStateOffset;
    if(sample.slaveNode<objectStateOffset||local64>=3ull*nodesPerLayer) {
        result.failure="sample_node_outside_object_layers";return result;
    }
    const auto layer=static_cast<std::uint32_t>(local64/nodesPerLayer);
    const auto row=static_cast<std::uint32_t>(local64%nodesPerLayer);
    const auto& body=bodies[sample.bodyIndex];
    const auto& node=nodes[sample.slaveNode];
    const Vec3 world{node.positionAndMass.x,node.positionAndMass.y,node.positionAndMass.z};
    const Vec3 bodyPosition{body.position.x,body.position.y,body.position.z};
    const std::array<double,4> inverseQ{{-body.orientation.x,-body.orientation.y,
        -body.orientation.z,body.orientation.w}};
    const Vec3 exactLocal=rotate(inverseQ,sub(world,bodyPosition));
    const Vec3 query{float(exactLocal.x),float(exactLocal.y),float(exactLocal.z)};
    if(!finite(query)) {result.failure="query_local_nonfinite";return result;}
    using Triangle=NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU;
    std::vector<Triangle> full=bodyBanks[sample.bodyIndex].triangles;
    std::sort(full.begin(),full.end(),[](const Triangle& a,const Triangle& b) {
        return std::tie(a.bodyIndex,a.boneStableId,a.triangleStableId)<
               std::tie(b.bodyIndex,b.boneStableId,b.triangleStableId);
    });
    result.bankTriangleCount=full.size();
    if(full.empty()||!numi::matter::fem_body_contact_bvh::
            buildExactFeaturePseudonormals(std::span<Triangle>(full),true)) {
        result.failure="complete_body_bank_failed_closed_oriented_pseudonormal_build";
        return result;
    }
    std::map<std::pair<std::uint32_t,std::uint32_t>,const Triangle*> featureByIdentity;
    for(const auto& t:full) {
        if(t.bodyIndex!=sample.bodyIndex||
           !featureByIdentity.emplace(std::pair{t.boneStableId,t.triangleStableId},&t).second) {
            result.failure="full_bank_identity_mismatch_or_duplicate";return result;
        }
    }
    const auto findClosest=[&](const Triangle& t,const Triangle& pseudoSource,
        Closest& closest,Vec3& delta,double& d2,double& faceDot,double& pseudoDot)->bool {
        const Vec3 a=read3(t.bodyLocalVertex0),b=read3(t.bodyLocalVertex1),
                   c=read3(t.bodyLocalVertex2);
        closest=closestPointOnTriangle(query,a,b,c);
        if(!closest.valid)return false;
        delta=sub(query,closest.point);d2=norm2(delta);
        const Vec3 rawFace=cross(sub(b,a),sub(c,a));
        const double faceLength=norm(rawFace);
        Vec3 face{},pseudo{};
        if(!std::isfinite(faceLength)||!(faceLength>0.0)||
           !featurePseudoNormal(pseudoSource,closest,pseudo))return false;
        face=mul(rawFace,1.0/faceLength);
        faceDot=dot(delta,face);pseudoDot=dot(delta,pseudo);
        return finite(closest.point)&&finite(delta)&&std::isfinite(d2)&&
            std::isfinite(faceDot)&&std::isfinite(pseudoDot)&&d2>=0.0;
    };
    double boundedD2=std::numeric_limits<double>::infinity();
    Closest bounded{};double boundedFaceDot=0.0,boundedPseudoDot=0.0;
    const Triangle* boundedTriangle=nullptr;
    for(std::uint32_t local=0u;local<range.triangleCount;++local) {
        const auto& t=triangles[range.firstTriangle+local];
        if(t.bodyIndex!=sample.bodyIndex)continue;
        const auto it=featureByIdentity.find({t.boneStableId,t.triangleStableId});
        if(it==featureByIdentity.end())continue;
        const auto& source=*it->second;
        const auto samePoint=[](nm_float4 a,nm_float4 b) {
            return a.x==b.x&&a.y==b.y&&a.z==b.z;
        };
        if(!samePoint(t.bodyLocalVertex0,source.bodyLocalVertex0)||
           !samePoint(t.bodyLocalVertex1,source.bodyLocalVertex1)||
           !samePoint(t.bodyLocalVertex2,source.bodyLocalVertex2)) {
            result.failure="bounded_and_complete_bank_geometry_disagree";return result;
        }
        Closest closest{};Vec3 delta{};double d2=0.0,fd=0.0,pd=0.0;
        if(!findClosest(t,source,closest,delta,d2,fd,pd))continue;
        const auto identity=std::pair{t.boneStableId,t.triangleStableId};
        const auto best=boundedTriangle?std::pair{boundedTriangle->boneStableId,
            boundedTriangle->triangleStableId}:std::pair{UINT32_MAX,UINT32_MAX};
        if(d2<boundedD2||(d2==boundedD2&&identity<best)) {
            boundedD2=d2;bounded=closest;boundedFaceDot=fd;
            boundedPseudoDot=pd;boundedTriangle=&t;
        }
    }
    double fullD2=std::numeric_limits<double>::infinity();
    Closest bestFull{};double fullFaceDot=0.0,fullPseudoDot=0.0;
    const Triangle* fullTriangle=nullptr;
    for(const auto& t:full) {
        Closest closest{};Vec3 delta{};double d2=0.0,fd=0.0,pd=0.0;
        if(!findClosest(t,t,closest,delta,d2,fd,pd)) {
            result.failure="full_bank_closest_query_invalid";return result;
        }
        const auto identity=std::pair{t.boneStableId,t.triangleStableId};
        const auto best=fullTriangle?std::pair{fullTriangle->boneStableId,
            fullTriangle->triangleStableId}:std::pair{UINT32_MAX,UINT32_MAX};
        if(d2<fullD2||(d2==fullD2&&identity<best)) {
            fullD2=d2;bestFull=closest;fullFaceDot=fd;fullPseudoDot=pd;fullTriangle=&t;
        }
    }
    if(!boundedTriangle||!fullTriangle) {result.failure="no_finite_winner";return result;}
    result.sample=sampleIndex;result.slaveNode=sample.slaveNode;result.body=sample.bodyIndex;
    result.boundedBone=boundedTriangle->boneStableId;
    result.boundedTriangle=boundedTriangle->triangleStableId;
    result.fullBone=fullTriangle->boneStableId;result.fullTriangle=fullTriangle->triangleStableId;
    result.boundedFeature=bounded.feature;result.fullFeature=bestFull.feature;
    result.boundedDistanceM=std::sqrt(boundedD2);result.fullDistanceM=std::sqrt(fullD2);
    result.boundedFaceDot=boundedFaceDot;result.boundedPseudoDot=boundedPseudoDot;
    result.fullFaceDot=fullFaceDot;result.fullPseudoDot=fullPseudoDot;
    result.fullSolid=classifyClosedSolid(query,bodyBanks[sample.bodyIndex].triangles);
    const bool sameWinner=result.boundedBone==result.fullBone&&
                          result.boundedTriangle==result.fullTriangle;
    out<<std::setprecision(17)<<sampleIndex<<","<<sample.slaveNode<<","<<layer<<","<<row<<","<<
        sample.bodyIndex<<","<<query.x<<","<<query.y<<","<<query.z<<","<<
        result.boundedBone<<","<<result.boundedTriangle<<","<<featureName(bounded.feature)<<","<<
        result.boundedDistanceM<<","<<boundedFaceDot<<","<<boundedPseudoDot<<","<<
        bounded.point.x<<","<<bounded.point.y<<","<<bounded.point.z<<","<<
        result.bankTriangleCount<<","<<result.fullBone<<","<<result.fullTriangle<<","<<
        featureName(bestFull.feature)<<","<<result.fullDistanceM<<","<<fullFaceDot<<","<<
        fullPseudoDot<<","<<bestFull.point.x<<","<<bestFull.point.y<<","<<bestFull.point.z<<","<<
        result.fullSolid.insideRays<<","<<result.fullSolid.outsideRays<<","<<
        (result.fullSolid.ambiguous?1:0)<<","<<(sameWinner?1:0)<<"\n";
    result.valid=out.good();
    if(!result.valid)result.failure="comparison_output_failed";
    return result;
}

} // namespace numi::human::initial_skin_bone_contact_witness

#pragma once

// Source-bound right-kidney fragment release for the existing Human GPU owner.
// NHANAT stable ID 4 is registered on body 20, sourced from BodyParts3D core
// body 7. Both fragment and reference mass use body 20; the cut plane is mapped
// from core body 7 through the admitted rest FK. The right-kidney mass is inferred
// (0.140 kg Swiss autopsy median; 0.120-0.170 kg reference range), not a
// subject measurement. The attachment is an idealized rigid fascia weld; the
// residual bulk COM is held fixed as an explicit reference approximation.
// The collider is a convex proxy; cavity-wall contact is not represented.

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/GeometryCooker.hpp"
#include "metalrobo/numi_human_joint_equality_gpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <iomanip>
#include <sstream>
#include <map>
#include <span>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo::organ_release {

inline constexpr std::uint32_t kRightKidneyStableId = 4u;
inline constexpr std::uint32_t kDerivedRuntimeDisplayIdBase = 0xf1000000u;
inline constexpr std::uint32_t runtimeDisplaySurfaceId(
    const std::uint32_t sourceStableId,const std::uint32_t childOrdinal) noexcept {
    return kDerivedRuntimeDisplayIdBase + 2u * sourceStableId + childOrdinal;
}
inline constexpr std::uint32_t kRegisteredGeometryBodySource = 20u;
inline constexpr std::uint32_t kCoreAnatomyBodySource = 7u;
inline constexpr std::uint32_t kMassOwnerBodySource = 20u;
inline constexpr std::uint32_t kCaudalFragment = 0u;
inline constexpr std::uint32_t kCranialFragment = 1u;
inline constexpr std::uint32_t kCapFaceAncestry = MR_INVALID_INDEX;
inline constexpr double kReferenceRightKidneyMassKg = 0.140;
inline constexpr double kWeldToleranceM = 1.0e-7;
inline constexpr std::array<std::uint8_t, 32> kExpectedNhaSha256{
    0x1a,0x19,0x1f,0xe2,0xf6,0xc4,0xa1,0xcb,
    0x03,0xee,0x29,0x0b,0xc8,0x50,0xb8,0x00,
    0x11,0xcc,0x26,0x75,0x77,0x99,0x2e,0xa8,
    0x6a,0xbb,0x3a,0x4c,0xb3,0x9f,0x41,0x08,
};
inline constexpr std::array<std::uint8_t, 32> kExpectedSourceMemberSha256{
    0xe2,0x1e,0x2e,0xbd,0x52,0x24,0xb1,0xcb,
    0xf8,0xca,0xbb,0xcf,0x4f,0xa7,0x88,0xd8,
    0x94,0x81,0x9a,0x32,0x8a,0x00,0x6d,0xcd,
    0xb2,0xeb,0x8a,0xa6,0xee,0xe5,0x87,0x5a,
};
// Waist plane from the original source body-7 frame. Runtime admission maps
// it into the registered NHANAT geometry frame on body 20. Body-7 local +y is
// cranial, so the caudal half has signed distance <= 0.
inline constexpr std::array<double, 3> kCutPlanePointBody7{
    -0.03372473683, 0.05810004997, -0.00946755672};
inline constexpr std::array<double, 3> kCutPlaneNormalBody7{0.0, 1.0, 0.0};

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<double, 9>;
struct TriangleMesh {
    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> indices;
    // One entry per output triangle. UINT32_MAX marks a newly generated cap.
    std::vector<std::uint32_t> sourceFace;
};
struct MassProperties {
    double volumeM3 = 0.0;
    double massKg = 0.0;
    Vec3 centerOfMass{};
    Mat3 inertiaAboutCom{};
};
struct Fragment {
    std::uint32_t sourceStableId = kRightKidneyStableId;
    std::uint32_t childOrdinal = 0u;
    TriangleMesh surface;
    MassProperties mass;
};
struct SplitResult {
    std::array<Fragment, 2> fragments{}; // caudal, cranial
    std::uint32_t inputFaceCount = 0u;
    std::uint32_t crossingFaceCount = 0u;
    std::uint32_t capTriangleCount = 0u;
    std::uint32_t cutContourEdgeCount = 0u;
    double cutAreaM2 = 0.0;
    double referenceDensityKgM3 = 0.0;
};

// One pre-reserved support row is rebound at scene admission to this
// source-derived organ witness. The row is excluded from the skin partition
// before simulation and uses the same articulated point/Jacobian GPU owner.
struct RuntimeSupportWitness {
    bool active = false;
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    std::uint32_t contactIndex = MR_INVALID_INDEX;
    mr_float4 localPoint{};
};
inline RuntimeSupportWitness activeRuntimeSupportWitness{};

inline Vec3 add(const Vec3& a,const Vec3& b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
inline Vec3 sub(const Vec3& a,const Vec3& b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
inline Vec3 mul(const Vec3& a,double s){return {a[0]*s,a[1]*s,a[2]*s};}
inline double dot(const Vec3& a,const Vec3& b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline Vec3 cross(const Vec3& a,const Vec3& b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
inline double norm(const Vec3& a){return std::sqrt(dot(a,a));}
inline Vec3 normalized(Vec3 a){const double d=norm(a);return d>0?mul(a,1.0/d):Vec3{};}
inline Mat3 zero3(){return {};}
inline Mat3 addMat(const Mat3&a,const Mat3&b){Mat3 r{};for(std::size_t i=0;i<9;++i)r[i]=a[i]+b[i];return r;}
inline Mat3 subMat(const Mat3&a,const Mat3&b){Mat3 r{};for(std::size_t i=0;i<9;++i)r[i]=a[i]-b[i];return r;}
inline double& m(Mat3& a,int r,int c){return a[std::size_t(r)*3u+std::size_t(c)];}
inline double m(const Mat3& a,int r,int c){return a[std::size_t(r)*3u+std::size_t(c)];}
inline Mat3 parallelAxis(double mass,const Vec3& d){
    Mat3 a{};const double r2=dot(d,d);
    for(int r=0;r<3;++r)for(int c=0;c<3;++c)m(a,r,c)=mass*((r==c?r2:0.0)-d[r]*d[c]);
    return a;
}
struct Key3 { std::array<std::int64_t,3> q{}; friend bool operator<(const Key3&a,const Key3&b){return a.q<b.q;} };
inline Key3 key(const Vec3&p,double tol=kWeldToleranceM){
    return {{{std::llround(p[0]/tol),std::llround(p[1]/tol),std::llround(p[2]/tol)}}};
}
inline Vec3 toVec3(const mr_float4& p){return {double(p.x),double(p.y),double(p.z)};}
inline mr_float4 toFloat4(const Vec3& p,float w=0.0f){return {float(p[0]),float(p[1]),float(p[2]),w};}

struct EdgeKey { std::uint32_t a=0,b=0; friend bool operator<(const EdgeKey&x,const EdgeKey&y){return x.a<y.a||(x.a==y.a&&x.b<y.b);} };
inline EdgeKey undirected(std::uint32_t a,std::uint32_t b){return a<b?EdgeKey{a,b}:EdgeKey{b,a};}
struct EdgeUse { std::uint32_t count=0; int orientation=0; };
inline bool validateClosedOriented(const TriangleMesh& mesh,std::string& error){
    if(mesh.indices.size()<12u||mesh.indices.size()%3u!=0u){error="cut source has too few complete faces";return false;}
    std::map<EdgeKey,EdgeUse> edges;
    for(std::size_t i=0;i<mesh.indices.size();i+=3u){
        const auto a=mesh.indices[i],b=mesh.indices[i+1],c=mesh.indices[i+2];
        if(a>=mesh.vertices.size()||b>=mesh.vertices.size()||c>=mesh.vertices.size()||a==b||b==c||c==a){error="cut source contains an invalid or degenerate triangle";return false;}
        for(const auto e:std::array<EdgeKey,3>{{undirected(a,b),undirected(b,c),undirected(c,a)}}){
            auto& u=edges[e];++u.count;
        }
        auto orient=[&](std::uint32_t x,std::uint32_t y){auto&u=edges[undirected(x,y)];u.orientation+=(x<y?1:-1);};
        orient(a,b);orient(b,c);orient(c,a);
    }
    for(const auto& [edge,use]:edges)if(use.count!=2u||use.orientation!=0){
        error="source kidney is not a closed consistently oriented two-manifold after coordinate welding";return false;
    }
    return true;
}

inline bool triangulateLoop(std::vector<std::uint32_t> loop,const std::vector<Vec3>& vertices,
    const Vec3& planeNormal,bool wantPositiveNormal,std::vector<std::uint32_t>& triangles,std::string& error){
    if(loop.size()<3u){error="cut contour loop has fewer than three vertices";return false;}
    Vec3 u=normalized(std::abs(planeNormal[0])<0.8?cross(planeNormal,{1,0,0}):cross(planeNormal,{0,1,0}));
    Vec3 v=cross(planeNormal,u);
    std::vector<std::array<double,2>> p; p.reserve(loop.size());
    for(auto i:loop)p.push_back({dot(vertices[i],u),dot(vertices[i],v)});
    auto orient2=[](const auto&a,const auto&b,const auto&c){return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);};
    double area=0;for(std::size_t i=0;i<p.size();++i){const auto& a=p[i];const auto& b=p[(i+1u)%p.size()];area+=a[0]*b[1]-b[0]*a[1];}
    if(std::abs(area)<1e-14){error="cut contour has zero projected area";return false;}
    // Ear clipping below uses the positive projected orientation for its
    // convexity predicate. Keep that invariant independent of the requested
    // physical cap normal; reverse emitted triangle winding only at the end.
    const bool positive=area>0;
    if(!positive){std::reverse(loop.begin(),loop.end());p.clear();for(auto i:loop)p.push_back({dot(vertices[i],u),dot(vertices[i],v)});area=-area;}
    std::vector<std::size_t> polygon(loop.size());for(std::size_t i=0;i<polygon.size();++i)polygon[i]=i;
    std::size_t guard=0;
    while(polygon.size()>3u&&guard++<loop.size()*loop.size()){
        bool clipped=false;
        for(std::size_t j=0;j<polygon.size();++j){
            const auto ia=polygon[(j+polygon.size()-1u)%polygon.size()],ib=polygon[j],ic=polygon[(j+1u)%polygon.size()];
            if(orient2(p[ia],p[ib],p[ic])<=1e-14)continue;
            bool contains=false;
            for(auto id:polygon){if(id==ia||id==ib||id==ic)continue;
                const double a=orient2(p[ia],p[ib],p[id]),b=orient2(p[ib],p[ic],p[id]),c=orient2(p[ic],p[ia],p[id]);
                if(a>=-1e-14&&b>=-1e-14&&c>=-1e-14){contains=true;break;}}
            if(contains)continue;
            if(wantPositiveNormal)triangles.insert(triangles.end(),{loop[ia],loop[ib],loop[ic]});
            else triangles.insert(triangles.end(),{loop[ic],loop[ib],loop[ia]});
            polygon.erase(polygon.begin()+j);clipped=true;break;
        }
        if(!clipped){error="cut contour is self-intersecting or not a simple polygon";return false;}
    }
    if(polygon.size()!=3u){error="cut contour ear clipping did not terminate";return false;}
    if(wantPositiveNormal)triangles.insert(triangles.end(),{loop[polygon[0]],loop[polygon[1]],loop[polygon[2]]});
    else triangles.insert(triangles.end(),{loop[polygon[2]],loop[polygon[1]],loop[polygon[0]]});
    return true;
}

inline bool makeCutLoops(TriangleMesh& mesh,const Vec3& point,const Vec3& normal,bool capPositive,
    std::uint32_t& contourEdges,std::uint32_t& capCount,double& areaM2,std::string& error){
    std::map<EdgeKey,std::uint32_t> edgeCounts;
    for(std::size_t i=0;i<mesh.indices.size();i+=3u){const auto a=mesh.indices[i],b=mesh.indices[i+1],c=mesh.indices[i+2];
        for(auto e:std::array<EdgeKey,3>{{undirected(a,b),undirected(b,c),undirected(c,a)}})++edgeCounts[e];}
    std::map<std::uint32_t,std::vector<std::uint32_t>> graph;
    for(const auto& [e,count]:edgeCounts)if(count==1u&&std::abs(dot(sub(mesh.vertices[e.a],point),normal))<2e-7&&
            std::abs(dot(sub(mesh.vertices[e.b],point),normal))<2e-7){graph[e.a].push_back(e.b);graph[e.b].push_back(e.a);}
    if(graph.empty()){error="cut produced no open plane boundary to cap";return false;}
    for(const auto& [v,adj]:graph)if(adj.size()!=2u){error="cut contour is branched or has an open endpoint";return false;}
    contourEdges+=static_cast<std::uint32_t>(graph.size());
    std::set<std::uint32_t> visited;
    std::vector<std::uint32_t> capIndices;
    for(const auto& [start,adj]:graph){if(visited.contains(start))continue;
        std::vector<std::uint32_t> loop;std::uint32_t previous=MR_INVALID_INDEX,current=start;
        for(std::size_t step=0;step<=graph.size();++step){
            if(visited.contains(current)){if(current!=start){error="cut contour loops overlap";return false;}break;}
            visited.insert(current);loop.push_back(current);const auto& nexts=graph.at(current);
            const std::uint32_t next=nexts[0]==previous?nexts[1]:nexts[0];previous=current;current=next;
            if(current==start)break;
        }
        if(current!=start){error="cut contour did not close";return false;}
        Vec3 basisU=normalized(std::abs(normal[0])<0.8?cross(normal,{1,0,0}):cross(normal,{0,1,0}));Vec3 basisV=cross(normal,basisU);double signedArea=0.0;
        for(std::size_t k=0;k<loop.size();++k){const auto& a=mesh.vertices[loop[k]];const auto& b=mesh.vertices[loop[(k+1u)%loop.size()]];signedArea+=dot(a,basisU)*dot(b,basisV)-dot(b,basisU)*dot(a,basisV);}
        areaM2+=0.5*std::abs(signedArea);
        if(!triangulateLoop(loop,mesh.vertices,normal,capPositive,capIndices,error))return false;
    }
    for(auto index:capIndices)mesh.indices.push_back(index);
    capCount+=static_cast<std::uint32_t>(capIndices.size()/3u);
    mesh.sourceFace.insert(mesh.sourceFace.end(),capIndices.size()/3u,kCapFaceAncestry);
    return true;
}

inline bool computeMass(TriangleMesh& mesh,double density,MassProperties& result,std::string& error){
    double volume=0.0;Vec3 first{};Mat3 second{};
    for(std::size_t i=0;i<mesh.indices.size();i+=3u){const Vec3 a=mesh.vertices[mesh.indices[i]],b=mesh.vertices[mesh.indices[i+1]],c=mesh.vertices[mesh.indices[i+2]];
        const double dv=dot(a,cross(b,c))/6.0;volume+=dv;const Vec3 s=add(add(a,b),c);first=add(first,mul(s,dv/4.0));
        for(int r=0;r<3;++r)for(int col=0;col<3;++col)
            m(second,r,col)+=dv/20.0*(s[r]*s[col]+a[r]*a[col]+b[r]*b[col]+c[r]*c[col]);
    }
    if(volume<0.0){for(std::size_t i=0;i<mesh.indices.size();i+=3u)std::swap(mesh.indices[i+1u],mesh.indices[i+2u]);volume=-volume;first=mul(first,-1.0);for(double&x:second)x=-x;}
    if(!(volume>1e-10)||!std::isfinite(volume)){error="closed fragment has zero or non-finite signed volume";return false;}
    const Vec3 center=mul(first,1.0/volume);Mat3 inertia{};
    for(int r=0;r<3;++r)for(int col=0;col<3;++col){const double iOrigin=density*((r==col?(m(second,0,0)+m(second,1,1)+m(second,2,2)):0.0)-m(second,r,col));
        const Mat3 shift=parallelAxis(density*volume,center);m(inertia,r,col)=iOrigin-m(shift,r,col);}
    result.volumeM3=volume;result.massKg=density*volume;result.centerOfMass=center;result.inertiaAboutCom=inertia;
    if(!std::isfinite(result.massKg)||result.massKg<=0){error="fragment mass is invalid";return false;}return true;
}

inline bool splitClosedSurfaceByPlane(std::span<const mr_float4> sourceVertices,
    std::span<const std::uint32_t> sourceIndices,const Vec3& planePoint,const Vec3& planeNormal,
    double referenceMassKg,SplitResult& output,std::string& error){
    error.clear();if(sourceIndices.empty()||sourceIndices.size()%3u||sourceVertices.empty()||referenceMassKg<=0){error="invalid organ cut input";return false;}
    const Vec3 n=normalized(planeNormal);if(norm(n)<0.99){error="cut normal is invalid";return false;}
    TriangleMesh welded;std::map<Key3,std::uint32_t> map;std::vector<std::uint32_t> oldToNew(sourceVertices.size());
    for(std::size_t i=0;i<sourceVertices.size();++i){const Vec3 p=toVec3(sourceVertices[i]);const Key3 k=key(p);auto [it,inserted]=map.emplace(k,static_cast<std::uint32_t>(welded.vertices.size()));
        if(inserted)welded.vertices.push_back(p);oldToNew[i]=it->second;}
    welded.indices.reserve(sourceIndices.size());
    for(auto index:sourceIndices){if(index>=oldToNew.size()){error="organ index exceeds source vertex range";return false;}welded.indices.push_back(oldToNew[index]);}
    std::vector<std::uint32_t> clean;clean.reserve(welded.indices.size());
    std::vector<std::uint32_t> cleanSourceFace;cleanSourceFace.reserve(welded.indices.size()/3u);
    for(std::size_t i=0;i<welded.indices.size();i+=3u){auto a=welded.indices[i],b=welded.indices[i+1],c=welded.indices[i+2];
        if(a==b||b==c||c==a)continue;if(norm(cross(sub(welded.vertices[b],welded.vertices[a]),sub(welded.vertices[c],welded.vertices[a])))<1e-12)continue;
        clean.insert(clean.end(),{a,b,c});cleanSourceFace.push_back(static_cast<std::uint32_t>(i/3u));}
    welded.indices=std::move(clean);
    if(!validateClosedOriented(welded,error)){error="unclipped source: "+error;return false;}
    output={};output.inputFaceCount=static_cast<std::uint32_t>(welded.indices.size()/3u);
    for(unsigned side=0;side<2;++side){
        auto& dst=output.fragments[side].surface;output.fragments[side].childOrdinal=side;
        std::map<Key3,std::uint32_t> outMap;
        auto emitVertex=[&](const Vec3&p){const Key3 k=key(p);auto [it,inserted]=outMap.emplace(k,static_cast<std::uint32_t>(dst.vertices.size()));if(inserted)dst.vertices.push_back(p);return it->second;};
        auto intersection=[&](const Vec3&a,const Vec3&b){const double da=dot(sub(a,planePoint),n),db=dot(sub(b,planePoint),n);const double t=std::clamp(da/(da-db),0.0,1.0);return add(a,mul(sub(b,a),t));};
        for(std::uint32_t face=0;face<output.inputFaceCount;++face){const auto ia=welded.indices[3u*face],ib=welded.indices[3u*face+1u],ic=welded.indices[3u*face+2u];
            const std::array<std::uint32_t,3> ids{ia,ib,ic};std::vector<Vec3> poly;
            for(unsigned edge=0;edge<3;++edge){const auto a=ids[edge],b=ids[(edge+1u)%3u];const Vec3 pa=welded.vertices[a],pb=welded.vertices[b];
                const double da=dot(sub(pa,planePoint),n),db=dot(sub(pb,planePoint),n);const bool inA=side==0?da<=1e-10:da>=-1e-10;const bool inB=side==0?db<=1e-10:db>=-1e-10;
                if(inA&&inB)poly.push_back(pb);else if(inA&&!inB)poly.push_back(intersection(pa,pb));else if(!inA&&inB){poly.push_back(intersection(pa,pb));poly.push_back(pb);}
            }
            std::vector<Vec3> compact;for(const auto&p:poly)if(compact.empty()||norm(sub(p,compact.back()))>kWeldToleranceM)compact.push_back(p);
            if(compact.size()>1u&&norm(sub(compact.front(),compact.back()))<=kWeldToleranceM)compact.pop_back();
            if(compact.size()<3u)continue;
            if(compact.size()>3u)++output.crossingFaceCount;
            std::vector<std::uint32_t> vi;for(const auto&p:compact)vi.push_back(emitVertex(p));
            for(std::size_t j=1;j+1<vi.size();++j){dst.indices.insert(dst.indices.end(),{vi[0],vi[j],vi[j+1]});dst.sourceFace.push_back(cleanSourceFace[face]);}
        }
        if(dst.indices.empty()){error="cut plane removed one entire organ half";return false;}
        // Caudal cap normal is +n; cranial cap normal is -n.
        std::uint32_t capCount=0u,edgeCount=0u;double sideArea=0.0;if(!makeCutLoops(dst,planePoint,n,side==0u,edgeCount,capCount,sideArea,error))return false;
        output.cutContourEdgeCount+=edgeCount;output.capTriangleCount+=capCount;output.cutAreaM2+=sideArea;
        if(!validateClosedOriented(dst,error)){error=(side==0u?"caudal clipped fragment: ":"cranial clipped fragment: ")+error;return false;}
        std::map<EdgeKey,std::uint32_t> components;
        (void)components;
        // Recompute vertex normals from the closed clipped mesh; source face
        // ancestry remains per triangle and cap faces are explicitly marked.
    }
    // First compute geometric volumes using unit density, then scale the
    // reference kidney mass over both closed source-derived fragments.
    MassProperties unit0{},unit1{};if(!computeMass(output.fragments[0].surface,1.0,unit0,error)||!computeMass(output.fragments[1].surface,1.0,unit1,error))return false;
    const double volume=unit0.volumeM3+unit1.volumeM3;if(!(volume>0)){error="organ split has no positive total volume";return false;}
    output.referenceDensityKgM3=referenceMassKg/volume;output.cutAreaM2*=0.5;
    if(!computeMass(output.fragments[0].surface,output.referenceDensityKgM3,output.fragments[0].mass,error)||
       !computeMass(output.fragments[1].surface,output.referenceDensityKgM3,output.fragments[1].mass,error))return false;
    return std::abs(output.fragments[0].mass.massKg+output.fragments[1].mass.massKg-referenceMassKg)<=1e-9;
}

inline Mat3 readInertia(const MRBodyPropertiesGPU& b){Mat3 a{};for(int r=0;r<3;++r)for(int c=0;c<3;++c){const mr_float4 row=r==0?b.inertiaRow0:(r==1?b.inertiaRow1:b.inertiaRow2);m(a,r,c)=c==0?row.x:(c==1?row.y:row.z);}return a;}
// An SPD rotational inertia alone is insufficient: its principal moments
// must satisfy the triangle inequalities. The equivalent second-moment
// matrix C = trace(I)/2 * identity - I must be positive semidefinite.
// Normalize only for scale-independent roundoff checks; never repair I.
inline bool physicallyRealizableInertia(const Mat3& inertia) {
    double scale=0.0;
    for(double value:inertia) {
        if(!std::isfinite(value)) return false;
        scale=std::max(scale,std::abs(value));
    }
    if(!(scale>0.0)) return false;
    Mat3 a{};
    for(int r=0;r<3;++r)for(int col=0;col<3;++col)
        m(a,r,col)=m(inertia,r,col)/scale;
    constexpr double tolerance=64.0*std::numeric_limits<double>::epsilon();
    for(int r=0;r<3;++r)for(int col=r+1;col<3;++col)
        if(std::abs(m(a,r,col)-m(a,col,r))>tolerance) return false;
    const auto det=[](const Mat3& v) {
        return m(v,0,0)*(m(v,1,1)*m(v,2,2)-m(v,1,2)*m(v,2,1))
             - m(v,0,1)*(m(v,1,0)*m(v,2,2)-m(v,1,2)*m(v,2,0))
             + m(v,0,2)*(m(v,1,0)*m(v,2,1)-m(v,1,1)*m(v,2,0));
    };
    if(!(m(a,0,0)>0.0 &&
         m(a,0,0)*m(a,1,1)-m(a,0,1)*m(a,1,0)>0.0 &&
         det(a)>0.0)) return false;
    Mat3 second{};const double halfTrace=0.5*(m(a,0,0)+m(a,1,1)+m(a,2,2));
    for(int r=0;r<3;++r)for(int col=0;col<3;++col)
        m(second,r,col)=(r==col?halfTrace:0.0)-m(a,r,col);
    for(int r=0;r<3;++r) {
        if(m(second,r,r)<-tolerance) return false;
        for(int col=r+1;col<3;++col)
            if(m(second,r,r)*m(second,col,col)-
               m(second,r,col)*m(second,col,r)<-tolerance) return false;
    }
    return det(second)>=-tolerance;
}

enum class StoredInertiaClass { realizable, float32RoundingBoundary, invalid };
// Source payloads contain rounded FP32 tensors. For a diagonal tensor only,
// prove whether a triangle-equality failure lies entirely inside the exact
// round-to-nearest input bins. This classifies uncertainty; it never changes
// the tensor or admits a non-diagonal unresolved tensor.
inline StoredInertiaClass classifyStoredFloat32Inertia(const Mat3& inertia) {
    if(physicallyRealizableInertia(inertia)) return StoredInertiaClass::realizable;
    std::array<double,3> lo{},hi{};
    for(int r=0;r<3;++r) {
        for(int col=0;col<3;++col)
            if(r!=col && m(inertia,r,col)!=0.0) return StoredInertiaClass::invalid;
        const double value=m(inertia,r,r);const float f=static_cast<float>(value);
        if(!std::isfinite(value)||!(value>0.0)||double(f)!=value)
            return StoredInertiaClass::invalid;
        const float previous=std::nextafter(f,-std::numeric_limits<float>::infinity());
        const float next=std::nextafter(f,std::numeric_limits<float>::infinity());
        lo[r]=0.5*(double(previous)+value);
        hi[r]=0.5*(value+double(next));
    }
    for(int r=0;r<3;++r)
        if(lo[r]>hi[(r+1)%3]+hi[(r+2)%3])
            return StoredInertiaClass::invalid;
    return StoredInertiaClass::float32RoundingBoundary;
}

inline bool inverse3(const Mat3&a,Mat3&out){const double d=m(a,0,0)*(m(a,1,1)*m(a,2,2)-m(a,1,2)*m(a,2,1))-m(a,0,1)*(m(a,1,0)*m(a,2,2)-m(a,1,2)*m(a,2,0))+m(a,0,2)*(m(a,1,0)*m(a,2,1)-m(a,1,1)*m(a,2,0));if(!std::isfinite(d)||d<=1e-18)return false;
    m(out,0,0)=(m(a,1,1)*m(a,2,2)-m(a,1,2)*m(a,2,1))/d;m(out,0,1)=(m(a,0,2)*m(a,2,1)-m(a,0,1)*m(a,2,2))/d;m(out,0,2)=(m(a,0,1)*m(a,1,2)-m(a,0,2)*m(a,1,1))/d;
    m(out,1,0)=(m(a,1,2)*m(a,2,0)-m(a,1,0)*m(a,2,2))/d;m(out,1,1)=(m(a,0,0)*m(a,2,2)-m(a,0,2)*m(a,2,0))/d;m(out,1,2)=(m(a,0,2)*m(a,1,0)-m(a,0,0)*m(a,1,2))/d;
    m(out,2,0)=(m(a,1,0)*m(a,2,1)-m(a,1,1)*m(a,2,0))/d;m(out,2,1)=(m(a,0,1)*m(a,2,0)-m(a,0,0)*m(a,2,1))/d;m(out,2,2)=(m(a,0,0)*m(a,1,1)-m(a,0,1)*m(a,1,0))/d;return true;}
inline void storeInertia(MRBodyPropertiesGPU&b,const Mat3&i){if(!physicallyRealizableInertia(i))throw std::runtime_error("organ partition inertia is not physically realizable");Mat3 inv{};if(!inverse3(i,inv))throw std::runtime_error("organ partition produced singular host inertia");b.inertiaRow0=toFloat4({m(i,0,0),m(i,0,1),m(i,0,2)});b.inertiaRow1=toFloat4({m(i,1,0),m(i,1,1),m(i,1,2)});b.inertiaRow2=toFloat4({m(i,2,0),m(i,2,1),m(i,2,2)});b.inverseInertiaRow0=toFloat4({m(inv,0,0),m(inv,0,1),m(inv,0,2)});b.inverseInertiaRow1=toFloat4({m(inv,1,0),m(inv,1,1),m(inv,1,2)});b.inverseInertiaRow2=toFloat4({m(inv,2,0),m(inv,2,1),m(inv,2,2)});}

struct FragmentChain { std::array<std::uint32_t,6> body{},joint{},q{},v{};std::uint32_t terminalBody=MR_INVALID_INDEX; };
inline bool appendFragmentBodies(EngineModel& model,std::uint32_t hostBody,
    std::span<const Fragment> fragments,std::vector<FragmentChain>& chains,
    std::vector<MRNumiHumanJointEqualityGPU>& attachmentRows,std::string& error){
    error.clear();if(model.articulations.size()!=1u||hostBody>=model.bodies.size()||fragments.empty()) {error="organ attachment requires the current one-articulation Human graph and valid registered organ host";return false;}
    auto candidate=model;std::vector<FragmentChain> newChains;auto rows=attachmentRows;const std::uint32_t ai=candidate.bodies[hostBody].articulationIndex;
    // Match the existing free-sphere EngineModel contact defaults as an
    // explicit passive proxy record. These are solver reference coefficients,
    // not measured tissue friction; this adds no force or hidden support.
    MRMaterialGPU organContactMaterial{};
    organContactMaterial.friction={0.8f,0.6f,0.0f,0.0f};
    organContactMaterial.response={0.1f,0.5f,0.0f,0.0f};
    organContactMaterial.geometry={0.01f,0.0f,0.0f,0.0f};
    const std::uint32_t organMaterialIndex=static_cast<std::uint32_t>(candidate.materials.size());
    candidate.materials.push_back(organContactMaterial);
    candidate.world.materialCount=static_cast<std::uint32_t>(candidate.materials.size());
    if(ai>=candidate.articulations.size()||ai+1u!=candidate.articulations.size()){error="organ host does not belong to the trailing Human articulation";return false;}
    constexpr std::uint32_t kBodiesPerFragment=6u,kDofsPerFragment=6u;
    if(candidate.world.bodyCount+fragments.size()*kBodiesPerFragment>192u||candidate.world.nv+fragments.size()*kDofsPerFragment>160u){error="Human GPU articulated capacity is insufficient for both organ fragments";return false;}
    auto& articulation=candidate.articulations[ai];newChains.reserve(fragments.size());
    const std::array<Vec3,6> axes{{{1,0,0},{0,1,0},{0,0,1},{1,0,0},{0,1,0},{0,0,1}}};
    for(std::size_t f=0;f<fragments.size();++f){const auto& fragment=fragments[f];if(fragment.mass.massKg<=0||fragment.surface.vertices.size()<4u){error="organ fragment has no admitted closed mass geometry";return false;}
        FragmentChain chain;std::uint32_t parent=hostBody;const std::uint32_t baseBody=static_cast<std::uint32_t>(candidate.bodies.size()),baseJoint=static_cast<std::uint32_t>(candidate.joints.size()),baseQ=static_cast<std::uint32_t>(candidate.defaultQ.size()),baseV=static_cast<std::uint32_t>(candidate.defaultV.size());
        for(std::uint32_t k=0;k<6u;++k){const std::uint32_t bi=baseBody+k,ji=baseJoint+k,qi=baseQ+k,vi=baseV+k;chain.body[k]=bi;chain.joint[k]=ji;chain.q[k]=qi;chain.v[k]=vi;
            MRBodyPropertiesGPU body{};body.articulationIndex=ai;body.parentBody=parent;body.inboundJoint=ji;body.motionType=k==5u?MR_MOTION_DYNAMIC:MR_MOTION_STATIC;
            if(k==5u){body.massAndInverseMass={float(fragment.mass.massKg),float(1.0/fragment.mass.massKg),0,0};body.centerOfMass={0,0,0,0};const auto&I=fragment.mass.inertiaAboutCom;Mat3 inv{};if(!physicallyRealizableInertia(I)||!inverse3(I,inv)){error="organ fragment inertia is singular or not physically realizable";return false;}body.inertiaRow0=toFloat4({m(I,0,0),m(I,0,1),m(I,0,2)});body.inertiaRow1=toFloat4({m(I,1,0),m(I,1,1),m(I,1,2)});body.inertiaRow2=toFloat4({m(I,2,0),m(I,2,1),m(I,2,2)});body.inverseInertiaRow0=toFloat4({m(inv,0,0),m(inv,0,1),m(inv,0,2)});body.inverseInertiaRow1=toFloat4({m(inv,1,0),m(inv,1,1),m(inv,1,2)});body.inverseInertiaRow2=toFloat4({m(inv,2,0),m(inv,2,1),m(inv,2,2)});}
            candidate.bodies.push_back(body);MRJointDescriptorGPU joint{};joint.parentBody=parent;joint.childBody=bi;joint.jointType=k<3u?MR_JOINT_PRISMATIC:MR_JOINT_REVOLUTE;joint.qOffset=qi;joint.nq=1u;joint.vOffset=vi;joint.nv=1u;joint.axis0=toFloat4(axes[k]);joint.parentRotation={0,0,0,1};joint.childRotation={0,0,0,1};candidate.joints.push_back(joint);
            MRDofPropertiesGPU dof{};dof.articulationIndex=ai;dof.jointIndex=ji;dof.qIndex=qi;dof.vIndex=vi;dof.localDof=0u;candidate.dofs.push_back(dof);if(!candidate.actuatorProfiles.empty()){MRActuatorProfileGPU profile{};profile.identity={vi,0,0,0};candidate.actuatorProfiles.push_back(profile);}candidate.defaultQ.push_back(k<3u?float(fragment.mass.centerOfMass[k]):0.0f);candidate.defaultV.push_back(0.0f);
            MRNumiHumanJointEqualityGPU row{};row.indices={qi,vi,MR_INVALID_INDEX,MR_INVALID_INDEX};row.referencesAndCoefficients0.x=candidate.defaultQ.back();rows.push_back(row);parent=bi;}
        chain.terminalBody=chain.body.back();std::vector<mr_float4> hull;hull.reserve(fragment.surface.vertices.size());for(const auto&p:fragment.surface.vertices)hull.push_back(toFloat4(sub(p,fragment.mass.centerOfMass),1.0));auto cooked=cookConvexHullGeometry(candidate,hull);if(!cooked.succeeded()){error="organ fragment convex contact proxy failed: "+cooked.message;return false;}
        MRShapeGPU shape{};shape.bodyIndex=chain.terminalBody;shape.shapeType=MR_SHAPE_CONVEX;shape.materialIndex=organMaterialIndex;shape.collisionGroup=1u;shape.collisionMask=0xffffffffu;shape.geometryOffset=cooked.geometryIndex;shape.geometryCount=1u;shape.localRotation={0,0,0,1};shape.dimensions={1,1,1,0};double maxRadius=0.0;for(const auto& p:fragment.surface.vertices)maxRadius=std::max(maxRadius,norm(sub(p,fragment.mass.centerOfMass)));shape.contactRestAndBoundingRadius={0.001f,0.0f,float(maxRadius),0.0f};candidate.shapes.push_back(shape);
        if(!candidate.bodyNames.empty())for(unsigned k=0;k<6;++k)candidate.bodyNames.push_back("right_kidney_fragment_"+std::to_string(fragment.childOrdinal)+"_carrier_"+std::to_string(k));
        if(!candidate.jointNames.empty())for(unsigned k=0;k<6;++k)candidate.jointNames.push_back("right_kidney_fragment_"+std::to_string(fragment.childOrdinal)+"_joint_"+std::to_string(k));
        if(!candidate.dofNames.empty())for(unsigned k=0;k<6;++k)candidate.dofNames.push_back("right_kidney_fragment_"+std::to_string(fragment.childOrdinal)+"_dof_"+std::to_string(k));
        if(!candidate.shapeNames.empty())candidate.shapeNames.push_back("right_kidney_fragment_"+std::to_string(fragment.childOrdinal)+"_convex_proxy");
        articulation.bodyCount+=kBodiesPerFragment;articulation.jointCount+=kBodiesPerFragment;articulation.nq+=kDofsPerFragment;articulation.nv+=kDofsPerFragment;candidate.world.bodyCount+=kBodiesPerFragment;candidate.world.jointCount+=kBodiesPerFragment;candidate.world.shapeCount+=1u;candidate.world.nq+=kDofsPerFragment;candidate.world.nv+=kDofsPerFragment;newChains.push_back(chain);
    }
    std::string modelError;
    if(!candidate.valid(&modelError)) {
        const auto rangeValid=[](std::uint32_t offset,std::uint32_t count,std::size_t size) {
            return std::size_t(offset)<=size&&std::size_t(count)<=size-std::size_t(offset);
        };
        const auto finite4=[](mr_float4 v) {
            return std::isfinite(double(v.x))&&std::isfinite(double(v.y))&&
                std::isfinite(double(v.z))&&std::isfinite(double(v.w));
        };
        std::size_t badBodyRefs=0u,badMaterialRefs=0u,badGeometryRefs=0u,badTransforms=0u;
        for(const auto& shape:candidate.shapes) {
            badBodyRefs+=shape.bodyIndex>=candidate.bodies.size();
            badMaterialRefs+=shape.materialIndex>=candidate.materials.size();
            badGeometryRefs+=shape.geometryCount==0u||
                !rangeValid(shape.geometryOffset,shape.geometryCount,candidate.geometryHeaders.size());
            const double q2=double(shape.localRotation.x)*shape.localRotation.x+
                double(shape.localRotation.y)*shape.localRotation.y+
                double(shape.localRotation.z)*shape.localRotation.z+
                double(shape.localRotation.w)*shape.localRotation.w;
            badTransforms+=!finite4(shape.localPosition)||!finite4(shape.localRotation)||
                !finite4(shape.dimensions)||!finite4(shape.contactRestAndBoundingRadius)||
                !(q2>1.0e-12);
        }
        std::ostringstream detail;
        detail<<std::setprecision(10)<<" organ_candidate_counts{world_body="<<candidate.world.bodyCount
            <<",vector_body="<<candidate.bodies.size()<<",world_joint="<<candidate.world.jointCount
            <<",vector_joint="<<candidate.joints.size()<<",world_shape="<<candidate.world.shapeCount
            <<",vector_shape="<<candidate.shapes.size()<<",world_nq="<<candidate.world.nq
            <<",vector_nq="<<candidate.defaultQ.size()<<",world_nv="<<candidate.world.nv
            <<",vector_nv="<<candidate.defaultV.size()<<",material_count="<<candidate.materials.size()
            <<",geometry_header_count="<<candidate.geometryHeaders.size()<<",geometry_vertex_count="<<candidate.geometryVertices.size()
            <<",geometry_index_count="<<candidate.geometryIndices.size()<<",convex_face_count="<<candidate.convexFaces.size()
            <<",convex_half_edge_count="<<candidate.convexHalfEdges.size()<<"}"
            <<" shape_suspects{bad_body_refs="<<badBodyRefs<<",bad_material_refs="<<badMaterialRefs
            <<",bad_geometry_refs="<<badGeometryRefs<<",bad_transforms="<<badTransforms<<"}";
        if(!candidate.shapes.empty()) {
            const auto index=static_cast<std::uint32_t>(candidate.shapes.size()-1u);
            const auto& shape=candidate.shapes.back();
            detail<<" appended_shape{index="<<index<<",body="<<shape.bodyIndex
                <<",body_ok="<<(shape.bodyIndex<candidate.bodies.size())
                <<",type="<<shape.shapeType<<",material="<<shape.materialIndex
                <<",material_ok="<<(shape.materialIndex<candidate.materials.size())
                <<",geometry_offset="<<shape.geometryOffset<<",geometry_count="<<shape.geometryCount
                <<",geometry_range_ok="<<(shape.geometryCount>0u&&rangeValid(shape.geometryOffset,shape.geometryCount,candidate.geometryHeaders.size()))
                <<",local_position=["<<shape.localPosition.x<<","<<shape.localPosition.y<<","<<shape.localPosition.z<<","<<shape.localPosition.w<<"]"
                <<",local_rotation=["<<shape.localRotation.x<<","<<shape.localRotation.y<<","<<shape.localRotation.z<<","<<shape.localRotation.w<<"]"
                <<",dimensions=["<<shape.dimensions.x<<","<<shape.dimensions.y<<","<<shape.dimensions.z<<","<<shape.dimensions.w<<"]"
                <<",contact=["<<shape.contactRestAndBoundingRadius.x<<","<<shape.contactRestAndBoundingRadius.y<<","<<shape.contactRestAndBoundingRadius.z<<","<<shape.contactRestAndBoundingRadius.w<<"]";
            if(shape.geometryCount>0u&&shape.geometryOffset<candidate.geometryHeaders.size()) {
                const auto& g=candidate.geometryHeaders[shape.geometryOffset];
                detail<<",geometry_header{kind="<<g.kind<<",flags="<<g.flags
                    <<",vertices="<<g.vertexOffset<<"/"<<g.vertexCount<<"/"<<rangeValid(g.vertexOffset,g.vertexCount,candidate.geometryVertices.size())
                    <<",indices="<<g.indexOffset<<"/"<<g.indexCount<<"/"<<rangeValid(g.indexOffset,g.indexCount,candidate.geometryIndices.size())
                    <<",faces="<<g.faceOffset<<"/"<<g.faceCount<<"/"<<rangeValid(g.faceOffset,g.faceCount,candidate.convexFaces.size())
                    <<",half_edges="<<g.halfEdgeOffset<<"/"<<g.halfEdgeCount<<"/"<<rangeValid(g.halfEdgeOffset,g.halfEdgeCount,candidate.convexHalfEdges.size())
                    <<",bvh="<<g.bvhOffset<<"/"<<g.bvhCount<<"/"<<rangeValid(g.bvhOffset,g.bvhCount,candidate.meshBvhNodes.size())
                    <<",triangles="<<g.triangleOffset<<"/"<<g.triangleCount<<"/"<<rangeValid(g.triangleOffset,g.triangleCount,candidate.meshTriangles.size())
                    <<",materials="<<g.materialOffset<<"/"<<g.materialCount<<"/"<<rangeValid(g.materialOffset,g.materialCount,candidate.materials.size())<<"}";
            }
            detail<<"}";
        }
        error="organ augmented EngineModel invalid: "+modelError+detail.str();
        return false;
    }
    model=std::move(candidate);chains=std::move(newChains);attachmentRows=std::move(rows);return true;
}

inline std::array<double,4> quaternionConjugate(const std::array<double,4>& q) {
    return {-q[0],-q[1],-q[2],q[3]};
}
inline std::array<double,4> quaternionProduct(const std::array<double,4>& a,const std::array<double,4>& b) {
    return {a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],
            a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],
            a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],
            a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]};
}
inline Vec3 rotateVector(const Vec3& p,const std::array<double,4>& q) {
    const Vec3 u{q[0],q[1],q[2]};
    const Vec3 uv=cross(u,p),uuv=cross(u,uv);
    return add(p,add(mul(uv,2.0*q[3]),mul(uuv,2.0)));
}
inline Mat3 rotationMatrix(const std::array<double,4>& q) {
    Mat3 r{};for(int c=0;c<3;++c){Vec3 e{};e[c]=1.0;const auto v=rotateVector(e,q);for(int row=0;row<3;++row)m(r,row,c)=v[row];}return r;
}
inline MassProperties transformMassProperties(const MassProperties& source,const std::array<double,4>& rotation,const Vec3& translation) {
    MassProperties out=source;const Mat3 r=rotationMatrix(rotation);out.centerOfMass=add(translation,rotateVector(source.centerOfMass,rotation));
    Mat3 tmp{},rotated{};for(int row=0;row<3;++row)for(int col=0;col<3;++col)for(int k=0;k<3;++k)m(tmp,row,col)+=m(r,row,k)*m(source.inertiaAboutCom,k,col);
    for(int row=0;row<3;++row)for(int col=0;col<3;++col)for(int k=0;k<3;++k)m(rotated,row,col)+=m(tmp,row,k)*m(r,col,k);
    out.inertiaAboutCom=rotated;return out;
}
inline bool subtractMassFromFixedFrameHost(MRBodyPropertiesGPU& host,std::span<const Fragment> fragments,std::string& error) {
    error.clear();const double oldMass=host.massAndInverseMass.x;if(!(oldMass>0)){error="organ mass owner has no positive dynamic mass";return false;}
    const Vec3 storedCom{host.centerOfMass.x,host.centerOfMass.y,host.centerOfMass.z};
    if(norm(storedCom)>1e-7){error="fixed-frame organ mass partition requires the source mass-owner COM at its engine-frame origin";return false;}
    double newMass=oldMass;Mat3 newI=readInertia(host);
    for(const auto& f:fragments){const auto& p=f.mass;newMass-=p.massKg;newI=subMat(newI,addMat(p.inertiaAboutCom,parallelAxis(p.massKg,p.centerOfMass)));}
    if(!(newMass>0)){error="reference organ mass exhausts its source mass owner";return false;}
    const double minor2=m(newI,0,0)*m(newI,1,1)-m(newI,0,1)*m(newI,1,0);
    const double determinant=m(newI,0,0)*(m(newI,1,1)*m(newI,2,2)-m(newI,1,2)*m(newI,2,1))-
        m(newI,0,1)*(m(newI,1,0)*m(newI,2,2)-m(newI,1,2)*m(newI,2,0))+
        m(newI,0,2)*(m(newI,1,0)*m(newI,2,1)-m(newI,1,1)*m(newI,2,0));
    if(!(m(newI,0,0)>0.0&&minor2>0.0&&determinant>0.0)){error="mass-owner inertia is not positive definite after organ subtraction";return false;}
    host.massAndInverseMass={float(newMass),float(1.0/newMass),0,0};
    try{storeInertia(host,newI);}catch(const std::exception&e){error=e.what();return false;}return true;
}

inline std::vector<MRNumiHumanJointEqualityGPU> activeAttachmentRows(
    std::span<const MRNumiHumanJointEqualityGPU> allRows,std::span<const std::uint32_t> caudalQ,
    std::uint32_t completedSteps,std::uint32_t releaseStep){
    if(completedSteps<releaseStep)return {allRows.begin(),allRows.end()};std::vector<MRNumiHumanJointEqualityGPU> active;active.reserve(allRows.size());
    for(const auto& row:allRows)if(std::find(caudalQ.begin(),caudalQ.end(),row.indices.x)==caudalQ.end())active.push_back(row);return active;
}

// Build a render-only NHANAT payload. The immutable functional/source payload
// is never modified. Clipped children receive unique runtime display IDs while
// sourceStableIdByRuntimeDisplayId and per-face sourceFace retain source lineage.
template<class Anatomy>
inline Anatomy makeVisualAnatomyPayload(
    const Anatomy& source,const SplitResult& split,
    std::span<const FragmentChain> chains) {
    if(chains.size()!=1u)
        throw std::runtime_error("right-kidney release expects only its caudal child to become dynamic");
    Anatomy output=source;
    using Record=typename decltype(output.records)::value_type;
    using Vertex=typename decltype(output.vertices)::value_type;
    output.records.clear();output.vertices.clear();output.indices.clear();output.sourceStableIdByRuntimeDisplayId.clear();
    auto appendOrdinary=[&](const Record& record) {
        Record copy=record;copy.firstVertex=static_cast<std::uint32_t>(output.vertices.size());
        copy.firstIndex=static_cast<std::uint32_t>(output.indices.size());
        for(std::uint32_t i=0;i<record.vertexCount;++i)
            output.vertices.push_back(source.vertices[record.firstVertex+i]);
        for(std::uint32_t i=0;i<record.indexCount;++i)
            output.indices.push_back(copy.firstVertex+
                source.indices[record.firstIndex+i]-record.firstVertex);
        output.records.push_back(copy);
    };
    auto appendFragment=[&](const Record& sourceRecord,const Fragment& fragment,
                            std::uint32_t bodyIndex,bool recenter) {
        Record copy=sourceRecord;copy.bodyIndex=bodyIndex;copy.stableId=runtimeDisplaySurfaceId(fragment.sourceStableId,fragment.childOrdinal);
        copy.firstVertex=static_cast<std::uint32_t>(output.vertices.size());
        copy.vertexCount=static_cast<std::uint32_t>(fragment.surface.vertices.size());
        copy.firstIndex=static_cast<std::uint32_t>(output.indices.size());
        copy.indexCount=static_cast<std::uint32_t>(fragment.surface.indices.size());
        std::vector<Vec3> normals(fragment.surface.vertices.size());
        for(std::size_t i=0;i<fragment.surface.indices.size();i+=3u) {
            const auto a=fragment.surface.indices[i],b=fragment.surface.indices[i+1u],c=fragment.surface.indices[i+2u];
            const Vec3 n=cross(sub(fragment.surface.vertices[b],fragment.surface.vertices[a]),
                               sub(fragment.surface.vertices[c],fragment.surface.vertices[a]));
            normals[a]=add(normals[a],n);normals[b]=add(normals[b],n);normals[c]=add(normals[c],n);
        }
        for(std::size_t i=0;i<fragment.surface.vertices.size();++i) {
            const Vec3 local=recenter
                ? sub(fragment.surface.vertices[i],fragment.mass.centerOfMass)
                : fragment.surface.vertices[i];
            const Vec3 normal=normalized(normals[i]);Vertex v{};
            v.positionX=float(local[0]);v.positionY=float(local[1]);v.positionZ=float(local[2]);
            v.normalX=float(normal[0]);v.normalY=float(normal[1]);v.normalZ=float(normal[2]);
            output.vertices.push_back(v);
        }
        for(const auto index:fragment.surface.indices)output.indices.push_back(copy.firstVertex+index);
        output.records.push_back(copy);output.sourceStableIdByRuntimeDisplayId[copy.stableId]=fragment.sourceStableId;
    };
    for(const auto& record:source.records) {
        if(record.stableId!=kRightKidneyStableId){appendOrdinary(record);continue;}
        // The caudal clipped piece becomes an independently moving body after
        // the event. The cranial capped piece remains in the registered
        // abdomen body frame, whose mass retains this fraction.
        appendFragment(record,split.fragments[0],chains[0].terminalBody,true);
        appendFragment(record,split.fragments[1],record.bodyIndex,false);
    }
    std::set<std::uint32_t> uniqueRuntimeIds;
    for(const auto& record:output.records)
        if(!uniqueRuntimeIds.insert(record.stableId).second)
            throw std::runtime_error("derived anatomy runtime display IDs are not unique");
    output.header.surfaceCount=static_cast<std::uint32_t>(output.records.size());
    output.header.vertexCount=static_cast<std::uint32_t>(output.vertices.size());
    output.header.indexCount=static_cast<std::uint32_t>(output.indices.size());
    return output;
}

} // namespace metalrobo::organ_release

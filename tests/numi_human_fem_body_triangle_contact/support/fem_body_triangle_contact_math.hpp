#pragma once

#include "numi/matter/fem_body_triangle_contact_sample.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace numi::matter::fem_body_triangle_contact {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

using Matrix3 = std::array<std::array<double, 3>, 3>;

enum class Feature : std::uint8_t { face, edge, vertex };

struct ClosestPoint {
    Vec3 point{};
    std::array<double, 3> barycentric{};
    Vec3 edgeAxis{};
    Feature feature = Feature::face;
};

struct Evaluation {
    bool valid = false;
    bool active = false;
    Feature feature = Feature::face;
    Vec3 closestPoint{};
    Vec3 normal{};
    double signedDistance = 0.0;
    double closure = 0.0;
    double area = 0.0;
    double pressure = 0.0;
    double energy = 0.0;
    Vec3 slaveForce{};
    Vec3 bodyForce{};
    Vec3 bodyTorque{};
    Matrix3 positiveStiffness{}; // -d(slaveForce)/d(slavePosition)
};

inline Vec3 add(Vec3 a, Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline Vec3 sub(Vec3 a, Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline Vec3 mul(Vec3 a, double s) { return {a.x*s,a.y*s,a.z*s}; }
inline double dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
inline double norm2(Vec3 a) { return dot(a,a); }
inline double norm(Vec3 a) { return std::sqrt(norm2(a)); }
inline bool finite(Vec3 a) {
    return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
}
inline Vec3 unit(Vec3 a) { return mul(a, 1.0 / norm(a)); }
inline Vec3 read(nm_float4 v) { return {v.x,v.y,v.z}; }

inline nm_float4 packVec3(Vec3 v) {
    return {float(v.x),float(v.y),float(v.z),0.0f};
}

inline NMNumiHumanFEMBodyContactSampleGPU makeSample(
    std::uint32_t slaveNode,
    std::uint32_t bodyIndex,
    Vec3 v0,
    Vec3 v1,
    Vec3 v2,
    float slaveTributaryAreaM2,
    float pressureOverclosureStiffnessPaPerMeter,
    float normalStrainPerPressurePerPa
) {
    return makeTriangleContactSample(
        slaveNode,bodyIndex,packVec3(v0),packVec3(v1),packVec3(v2),
        slaveTributaryAreaM2,pressureOverclosureStiffnessPaPerMeter,
        normalStrainPerPressurePerPa);
}

inline std::array<Vec3,3> decodeTriangle(
    const NMNumiHumanFEMBodyContactSampleGPU& sample
) {
    const auto packed=decodeTriangleContactSample(sample);
    return {{{packed[0].x,packed[0].y,packed[0].z},
             {packed[1].x,packed[1].y,packed[1].z},
             {packed[2].x,packed[2].y,packed[2].z}}};
}

inline bool isUnitQuaternion(const std::array<double,4>& q) {
    double squaredNorm = 0.0;
    for (double component : q) {
        if (!std::isfinite(component)) return false;
        squaredNorm += component * component;
    }
    return std::abs(std::sqrt(squaredNorm) - 1.0) <= 1.0e-5;
}

inline Vec3 quaternionRotate(const std::array<double,4>& q, Vec3 value) {
    const Vec3 xyz{q[0],q[1],q[2]};
    const Vec3 twiceCross = mul(cross(xyz,value),2.0);
    return add(value,add(mul(twiceCross,q[3]),cross(xyz,twiceCross)));
}

inline bool scaledFaceNormal(Vec3 a,Vec3 b,Vec3 c,Vec3& normal) {
    const Vec3 ab=sub(b,a),ac=sub(c,a),bc=sub(c,b);
    const double scale=std::max({std::abs(ab.x),std::abs(ab.y),std::abs(ab.z),
        std::abs(ac.x),std::abs(ac.y),std::abs(ac.z),
        std::abs(bc.x),std::abs(bc.y),std::abs(bc.z)});
    if (!std::isfinite(scale) || !(scale>0.0)) return false;
    const Vec3 n=cross(mul(ab,1.0/scale),mul(ac,1.0/scale));
    const double nLength=norm(n);
    if (!std::isfinite(nLength) || !(nLength>0.0)) return false;
    normal=mul(n,1.0/nLength);
    return finite(normal);
}

inline ClosestPoint closestPointOnTriangle(
    Vec3 p, Vec3 a, Vec3 b, Vec3 c
) {
    const Vec3 ab=sub(b,a),ac=sub(c,a),bc=sub(c,b);
    const double scale=std::max({std::abs(ab.x),std::abs(ab.y),std::abs(ab.z),
        std::abs(ac.x),std::abs(ac.y),std::abs(ac.z),
        std::abs(bc.x),std::abs(bc.y),std::abs(bc.z)});
    if (!std::isfinite(scale) || !(scale>0.0)) return {};
    const Vec3 e0=mul(ab,1.0/scale),e1=mul(ac,1.0/scale);
    const Vec3 ap=mul(sub(p,a),1.0/scale);
    const Vec3 n=cross(e0,e1);
    const double n2=norm2(n);
    if (!std::isfinite(n2) || !(n2>0.0)) return {};

    const double heightNumerator=dot(ap,n);
    const Vec3 projected=sub(ap,mul(n,heightNumerator/n2));
    const double v=dot(cross(projected,e1),n)/n2;
    const double w=dot(cross(e0,projected),n)/n2;
    const double u=1.0-v-w;
    if (u>=0.0 && v>=0.0 && w>=0.0) {
        const Vec3 point=add(a,mul(add(mul(e0,v),mul(e1,w)),scale));
        return {point,{u,v,w},{},Feature::face};
    }

    ClosestPoint best{};
    double bestDistanceSquared=std::numeric_limits<double>::infinity();
    const auto consider=[&](Vec3 start,Vec3 finish,
                            std::array<double,3> baryStart,
                            std::array<double,3> baryEnd) {
        const Vec3 edge=mul(sub(finish,start),1.0/scale);
        const Vec3 rel=sub(ap,mul(sub(start,a),1.0/scale));
        const double lengthSquared=norm2(edge);
        if (!std::isfinite(lengthSquared) || !(lengthSquared>0.0)) return;
        const double t=std::clamp(dot(rel,edge)/lengthSquared,0.0,1.0);
        const Vec3 delta=sub(rel,mul(edge,t));
        const double distanceSquared=norm2(delta);
        if (!std::isfinite(distanceSquared) || distanceSquared>=bestDistanceSquared)
            return;
        std::array<double,3> bary{};
        for (std::size_t i=0;i<3;++i)
            bary[i]=baryStart[i]+t*(baryEnd[i]-baryStart[i]);
        const Vec3 point=add(start,mul(sub(finish,start),t));
        const Feature feature=(t>0.0 && t<1.0)?Feature::edge:Feature::vertex;
        const Vec3 axis=feature==Feature::edge
            ? mul(edge,1.0/std::sqrt(lengthSquared)) : Vec3{};
        best={point,bary,axis,feature};
        bestDistanceSquared=distanceSquared;
    };
    consider(a,b,{1,0,0},{0,1,0});
    consider(b,c,{0,1,0},{0,0,1});
    consider(c,a,{0,0,1},{1,0,0});
    return best;
}

// CPU mirror of the production feature-normal selection contract. A zero
// flags word is the legacy API and must fall back to the face normal. Face
// projections always use that normal and have no pseudonormal bit; edge and
// vertex projections in strict mode require their exact authored feature bit.
inline bool featureOrientationNormal(
    const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU& triangle,
    const ClosestPoint& closest,
    Vec3 faceNormal,
    Vec3& selectedNormal
) {
    selectedNormal=faceNormal;
    if (triangle.featureNormalFlags==0u || closest.feature==Feature::face)
        return finite(selectedNormal);
    Vec3 candidate{};
    std::uint32_t bit=0u;
    if (closest.feature==Feature::edge) {
        if (closest.barycentric[2]==0.0) {
            candidate=read(triangle.edgePseudoNormal01);
            bit=NM_NUMI_HUMAN_FEM_BODY_CONTACT_EDGE_01_PSEUDONORMAL;
        } else if (closest.barycentric[0]==0.0) {
            candidate=read(triangle.edgePseudoNormal12);
            bit=NM_NUMI_HUMAN_FEM_BODY_CONTACT_EDGE_12_PSEUDONORMAL;
        } else {
            candidate=read(triangle.edgePseudoNormal20);
            bit=NM_NUMI_HUMAN_FEM_BODY_CONTACT_EDGE_20_PSEUDONORMAL;
        }
    } else if (closest.feature==Feature::vertex) {
        if (closest.barycentric[0]>=closest.barycentric[1] &&
            closest.barycentric[0]>=closest.barycentric[2]) {
            candidate=read(triangle.vertexPseudoNormal0);
            bit=NM_NUMI_HUMAN_FEM_BODY_CONTACT_VERTEX_0_PSEUDONORMAL;
        } else if (closest.barycentric[1]>=closest.barycentric[2]) {
            candidate=read(triangle.vertexPseudoNormal1);
            bit=NM_NUMI_HUMAN_FEM_BODY_CONTACT_VERTEX_1_PSEUDONORMAL;
        } else {
            candidate=read(triangle.vertexPseudoNormal2);
            bit=NM_NUMI_HUMAN_FEM_BODY_CONTACT_VERTEX_2_PSEUDONORMAL;
        }
    } else {
        return false;
    }
    if ((triangle.featureNormalFlags & bit)==0u) return false;
    const double candidateLength=norm(candidate);
    if (!finite(candidate) || !std::isfinite(candidateLength) ||
        std::abs(candidateLength-1.0)>1.0e-3 ||
        dot(candidate,faceNormal)<-1.0e-4) return false;
    selectedNormal=candidate;
    return true;
}

inline Evaluation evaluate(
    Vec3 slave,
    Vec3 bodyOrigin,
    const std::array<Vec3,3>& triangle,
    double slaveTributaryAreaM2,
    double stiffnessPaPerMeter,
    double normalStrainPerPressurePerPa
) {
    Evaluation out{};
    const Vec3 a=triangle[0], b=triangle[1], c=triangle[2];
    Vec3 faceNormal{};
    if (!finite(slave) || !finite(bodyOrigin) || !finite(a) || !finite(b) ||
        !finite(c) || !scaledFaceNormal(a,b,c,faceNormal) ||
        !std::isfinite(slaveTributaryAreaM2) || slaveTributaryAreaM2<=0.0 ||
        !std::isfinite(stiffnessPaPerMeter) || stiffnessPaPerMeter<=0.0 ||
        !std::isfinite(normalStrainPerPressurePerPa) ||
        normalStrainPerPressurePerPa<=0.0) return out;
    const ClosestPoint q=closestPointOnTriangle(slave,a,b,c);
    if (!finite(q.point) || !std::all_of(q.barycentric.begin(),
            q.barycentric.end(),[](double value){return std::isfinite(value);}))
        return out;
    const Vec3 delta=sub(slave,q.point);
    const double distance=norm(delta);
    if (!std::isfinite(distance)) return out;
    const double orientation=dot(delta,faceNormal)<0.0 ? -1.0 : 1.0;
    out.closestPoint=q.point;
    out.feature=q.feature;
    out.normal=distance>1.0e-12 ? mul(delta,orientation/distance) : faceNormal;
    out.signedDistance=distance>1.0e-12 ? orientation*distance : 0.0;
    out.closure=std::max(0.0,-out.signedDistance-1.0e-7);
    out.area=slaveTributaryAreaM2;
    out.pressure=stiffnessPaPerMeter*out.closure;
    out.active=out.closure>0.0;
    const double scale=stiffnessPaPerMeter*out.area;
    if (!std::isfinite(out.pressure) || !std::isfinite(scale) ||
        !std::isfinite(out.pressure*out.area)) return Evaluation{};
    out.valid=true;
    if (!out.active) return out;
    out.slaveForce=mul(out.normal,scale*out.closure);
    out.bodyForce=mul(out.slaveForce,-1.0);
    out.bodyTorque=cross(sub(q.point,bodyOrigin),out.bodyForce);
    out.energy=0.5*scale*out.closure*out.closure;

    // The signed-distance Hessian is zero for a face projection. For edge and
    // vertex projections it is the exact curvature of distance to that affine
    // feature; this avoids the common but incorrect n*n^T-only edge tangent.
    const double r=std::max(distance,1.0e-12);
    const double sign=distance>1.0e-12 ? orientation : 1.0;
    for (std::size_t i=0;i<3;++i) {
        for (std::size_t j=0;j<3;++j) {
            const double ni = i==0 ? out.normal.x : (i==1 ? out.normal.y : out.normal.z);
            const double nj = j==0 ? out.normal.x : (j==1 ? out.normal.y : out.normal.z);
            double projection=0.0;
            if (q.feature==Feature::face) {
                const double fi = i==0 ? faceNormal.x : (i==1 ? faceNormal.y : faceNormal.z);
                const double fj = j==0 ? faceNormal.x : (j==1 ? faceNormal.y : faceNormal.z);
                projection=(i==j ? 1.0 : 0.0)-fi*fj;
            } else if (q.feature==Feature::edge) {
                const double ei = i==0 ? q.edgeAxis.x : (i==1 ? q.edgeAxis.y : q.edgeAxis.z);
                const double ej = j==0 ? q.edgeAxis.x : (j==1 ? q.edgeAxis.y : q.edgeAxis.z);
                projection=ei*ej;
            }
            const double signedDistanceHessian = q.feature==Feature::face ? 0.0 :
                sign/r*((i==j ? 1.0 : 0.0)-projection-ni*nj);
            out.positiveStiffness[i][j]=scale*(ni*nj-out.closure*signedDistanceHessian);
        }
    }
    for (const auto& row : out.positiveStiffness)
        for (double value : row)
            if (!std::isfinite(value)) return Evaluation{};
    return out;
}

} // namespace numi::matter::fem_body_triangle_contact

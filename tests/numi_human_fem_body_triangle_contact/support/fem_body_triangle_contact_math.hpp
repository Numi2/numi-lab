#pragma once

#include "numi/matter/fem_body_triangle_contact_sample.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

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

inline ClosestPoint closestPointOnTriangle(
    Vec3 p, Vec3 a, Vec3 b, Vec3 c
) {
    const Vec3 ab = sub(b,a), ac = sub(c,a), ap = sub(p,a);
    const double d1 = dot(ab,ap), d2 = dot(ac,ap);
    if (d1 <= 0.0 && d2 <= 0.0)
        return {a,{1.0,0.0,0.0},{},Feature::vertex};
    const Vec3 bp = sub(p,b);
    const double d3=dot(ab,bp), d4=dot(ac,bp);
    if (d3 >= 0.0 && d4 <= d3)
        return {b,{0.0,1.0,0.0},{},Feature::vertex};
    const double vc=d1*d4-d3*d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v=d1/(d1-d3);
        return {add(a,mul(ab,v)),{1.0-v,v,0.0},unit(ab),Feature::edge};
    }
    const Vec3 cp=sub(p,c);
    const double d5=dot(ab,cp), d6=dot(ac,cp);
    if (d6 >= 0.0 && d5 <= d6)
        return {c,{0.0,0.0,1.0},{},Feature::vertex};
    const double vb=d5*d2-d1*d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w=d2/(d2-d6);
        return {add(a,mul(ac,w)),{1.0-w,0.0,w},unit(ac),Feature::edge};
    }
    const double va=d3*d6-d5*d4;
    if (va <= 0.0 && (d4-d3) >= 0.0 && (d5-d6) >= 0.0) {
        const double w=(d4-d3)/((d4-d3)+(d5-d6));
        return {add(b,mul(sub(c,b),w)),{0.0,1.0-w,w},
                unit(sub(c,b)),Feature::edge};
    }
    const double inverse=1.0/(va+vb+vc);
    const double v=vb*inverse,w=vc*inverse;
    return {add(a,add(mul(ab,v),mul(ac,w))),{1.0-v-w,v,w},{},Feature::face};
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
    const Vec3 rawNormal=cross(sub(b,a),sub(c,a));
    const double rawNormalLength=norm(rawNormal);
    if (!finite(slave) || !finite(bodyOrigin) || !finite(a) || !finite(b) ||
        !finite(c) || !std::isfinite(rawNormalLength) || rawNormalLength<=1.0e-10 ||
        !std::isfinite(slaveTributaryAreaM2) || slaveTributaryAreaM2<=0.0 ||
        !std::isfinite(stiffnessPaPerMeter) || stiffnessPaPerMeter<=0.0 ||
        !std::isfinite(normalStrainPerPressurePerPa) ||
        normalStrainPerPressurePerPa<=0.0) return out;
    const Vec3 faceNormal=mul(rawNormal,1.0/rawNormalLength);
    const ClosestPoint q=closestPointOnTriangle(slave,a,b,c);
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

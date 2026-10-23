#include "support/fem_body_triangle_contact_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <limits>

using namespace numi::matter::fem_body_triangle_contact;

namespace {
void require(bool value, std::string_view message) {
    if (!value) {
        std::fprintf(stderr, "FAIL: %.*s\n",
                     int(message.size()), message.data());
        std::exit(1);
    }
}
double maxAbs(Vec3 a) { return std::max({std::abs(a.x),std::abs(a.y),std::abs(a.z)}); }
double component(Vec3 a, std::size_t i) {
    return i==0 ? a.x : (i==1 ? a.y : a.z);
}
double maxTangentError(const Evaluation& e, Vec3 p,
                       Vec3 bodyOrigin,
                       const std::array<Vec3,3>& tri,
                       double area, double k, double strain) {
    constexpr double h=2.0e-7;
    double maximum=0.0;
    for (std::size_t j=0;j<3;++j) {
        Vec3 plus=p, minus=p;
        if (j==0) { plus.x+=h; minus.x-=h; }
        if (j==1) { plus.y+=h; minus.y-=h; }
        if (j==2) { plus.z+=h; minus.z-=h; }
        const auto fp=evaluate(plus,bodyOrigin,tri,area,k,strain).slaveForce;
        const auto fm=evaluate(minus,bodyOrigin,tri,area,k,strain).slaveForce;
        for (std::size_t i=0;i<3;++i) {
            const double fd=-(component(fp,i)-component(fm,i))/(2.0*h);
            maximum=std::max(maximum,
                std::abs(fd-e.positiveStiffness[i][j])/
                std::max({1.0,std::abs(fd),std::abs(e.positiveStiffness[i][j])}));
        }
    }
    return maximum;
}
}

int main() {
    constexpr double area=0.25, k=1000.0, strain=1.0e-6;
    const std::array<Vec3,3> tri{{{0,0,0},{2,0,0},{0,2,0}}};
    const Vec3 bodyOrigin{0.1,-0.2,0.3};
    const auto sample=makeSample(7,12,tri[0],tri[1],tri[2],
                                 float(area),float(k),float(strain));
    require(sizeof(sample)==64, "triangle packing preserves 64-byte sample ABI");
    require(sample.flags==(NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
                           NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT),
            "explicit triangle flag combination");
    require(validTriangleContactSample(sample),
            "valid explicit triangle sample passes host validation");
    auto illegalFlags=sample;
    illegalFlags.flags |= NM_NUMI_HUMAN_FEM_BODY_CONTACT_ELASTIC_ATTACHMENT;
    require(!validTriangleContactSample(illegalFlags),
            "illegal combined contact modes fail before upload");
    auto zeroArea=sample;
    zeroArea.bodyLocalPointAndArea.w=0.0f;
    require(!validTriangleContactSample(zeroArea),
            "zero slave tributary area fails before upload");
    auto degenerate=sample;
    degenerate.bodyLocalNormalAndReferenceSeparation.x=0.0f;
    degenerate.bodyLocalNormalAndReferenceSeparation.y=0.0f;
    degenerate.bodyLocalNormalAndReferenceSeparation.z=0.0f;
    require(!validTriangleContactSample(degenerate),
            "degenerate master triangle fails before upload");
    auto negativeArea=sample;
    negativeArea.bodyLocalPointAndArea.w=-0.25f;
    require(!validTriangleContactSample(negativeArea),
            "negative slave tributary area fails before upload");
    auto nonFinite=sample;
    nonFinite.stiffnessAndNormalStrainPerPressure.y=
        std::numeric_limits<float>::quiet_NaN();
    require(!validTriangleContactSample(nonFinite),
            "non-finite packed coordinate fails before upload");
    auto invalidStiffness=sample;
    invalidStiffness.bodyLocalNormalAndReferenceSeparation.w=-1.0f;
    require(!validTriangleContactSample(invalidStiffness),
            "non-positive stiffness fails before upload");
    auto invalidReserved=sample;
    invalidReserved.reserved0=1u;
    require(!validTriangleContactSample(invalidReserved),
            "nonzero reserved field fails before upload");
    auto overflowProduct=sample;
    overflowProduct.bodyLocalPointAndArea.w=1.0e30f;
    overflowProduct.bodyLocalNormalAndReferenceSeparation.w=1.0e30f;
    require(!validTriangleContactSample(overflowProduct),
            "overflowing stiffness-area product fails before upload");
    require(sample.bodyLocalPointAndArea.w==0.25f,
            "slave quadrature area is explicit and differs from master area");
    require(isUnitQuaternion({0.0,0.0,0.0,1.0}) &&
            !isUnitQuaternion({0.0,0.0,0.0,2.0}),
            "pose quaternion validation");
    const auto decoded=decodeTriangle(sample);
    require(decoded[0].x==tri[0].x && decoded[1].x==tri[1].x &&
            decoded[2].y==tri[2].y,
            "packed triangle round-trips exactly");

    const auto outside=evaluate({0.4,0.4,0.3},bodyOrigin,tri,area,k,strain);
    require(outside.valid && !outside.active && outside.closure==0.0 &&
            maxAbs(outside.slaveForce)==0.0,
            "outside the zero-gap surface produces exactly zero force");

    const Vec3 facePoint{0.4,0.4,-0.05};
    const auto face=evaluate(facePoint,bodyOrigin,tri,area,k,strain);
    require(face.valid && face.active && face.feature==Feature::face &&
            face.normal.z>0.999999 && face.slaveForce.z>0.0,
            "interior contact uses outward oriented face normal");
    require(maxTangentError(face,facePoint,bodyOrigin,tri,area,k,strain)<2.0e-5,
            "face force tangent matches finite differences");

    const Vec3 edgePoint{0.7,-0.03,-0.04};
    const auto edge=evaluate(edgePoint,bodyOrigin,tri,area,k,strain);
    require(edge.valid && edge.active && edge.feature==Feature::edge &&
            std::abs(edge.closestPoint.y)<1.0e-12 &&
            std::abs(edge.closestPoint.x-0.7)<1.0e-12 &&
            edge.normal.z>0.0,
            "closest edge point and outward normal");
    require(maxTangentError(edge,edgePoint,bodyOrigin,tri,area,k,strain)<3.0e-5,
            "edge curvature tangent matches finite differences");

    const Vec3 vertexPoint{-0.03,-0.04,-0.02};
    const auto vertex=evaluate(vertexPoint,bodyOrigin,tri,area,k,strain);
    require(vertex.valid && vertex.active && vertex.feature==Feature::vertex &&
            maxAbs(vertex.closestPoint)==0.0 && vertex.normal.z>0.0,
            "closest vertex point and outward normal");
    require(maxTangentError(vertex,vertexPoint,bodyOrigin,tri,area,k,strain)<4.0e-5,
            "vertex curvature tangent matches finite differences");

    const auto backside=evaluate({0.4,0.4,-0.02},bodyOrigin,tri,area,k,strain);
    require(backside.valid && backside.active && backside.normal.z>0.999999 &&
            backside.slaveForce.z>0.0,
            "penetration cannot flip force behind authored triangle winding");

    const Vec3 forceResidual=add(face.slaveForce,face.bodyForce);
    const Vec3 momentResidual=add(cross(facePoint,face.slaveForce),
        add(cross(bodyOrigin,face.bodyForce),face.bodyTorque));
    require(maxAbs(forceResidual)<1.0e-10 &&
            maxAbs(momentResidual)<1.0e-10,
            "equal/opposite force and wrench close about world origin");

    std::printf("{\"status\":\"PASS\",\"abi_bytes\":%zu,"
                "\"outside_force\":%.9g,\"face_force\":%.9g,"
                "\"edge_force\":%.9g,\"vertex_force\":%.9g,"
                "\"tangent_rel_error\":[%.3g,%.3g,%.3g],"
                "\"action_reaction_residual\":%.3g,"
                "\"moment_residual\":%.3g}\n",
                sizeof(sample),maxAbs(outside.slaveForce),
                maxAbs(face.slaveForce),maxAbs(edge.slaveForce),
                maxAbs(vertex.slaveForce),
                maxTangentError(face,facePoint,bodyOrigin,tri,area,k,strain),
                maxTangentError(edge,edgePoint,bodyOrigin,tri,area,k,strain),
                maxTangentError(vertex,vertexPoint,bodyOrigin,tri,area,k,strain),
                maxAbs(forceResidual),maxAbs(momentResidual));
    return 0;
}

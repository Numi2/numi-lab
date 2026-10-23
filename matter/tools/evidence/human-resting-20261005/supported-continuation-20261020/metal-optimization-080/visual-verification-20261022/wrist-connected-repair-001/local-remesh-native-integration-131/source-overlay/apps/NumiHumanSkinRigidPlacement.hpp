#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <string>

namespace metalrobo::human::coarse_tissue {
// Initialization-only common SE(3) placement of an already admitted material
// reference. This never changes attachment coordinates, material, or history.
struct RegisteredSkinRigidPlacement {
    using Point=std::array<double,3>;
    using Matrix=std::array<double,9>;
    Matrix rotation{1,0,0,0,1,0,0,0,1};
    Point translation{};
    std::size_t validatedVertices=0;
    double maximumPointResidualM=0,maximumLocalResidualM=0;
    double maximumNormalResidual=0,maximumFrameResidual=0;
    Point vector(const Point& p) const {
        Point out{};
        for(unsigned r=0;r<3;++r)
            for(unsigned c=0;c<3;++c)out[r]+=rotation[3*r+c]*p[c];
        return out;
    }
    Point point(const Point& p) const {
        auto out=vector(p);
        for(unsigned c=0;c<3;++c)out[c]+=translation[c];
        return out;
    }
    Matrix donorFrame(const Matrix& referenceWorldToDonor) const {
        Matrix out{};
        // M_current = M_reference R^T.
        for(unsigned r=0;r<3;++r)for(unsigned c=0;c<3;++c)
            for(unsigned k=0;k<3;++k)
                out[3*r+c]+=referenceWorldToDonor[3*r+k]*rotation[3*c+k];
        return out;
    }
};

inline bool deriveRegisteredSkinRigidPlacement(
        const RegisteredSkinRigidPlacement::Point& referencePoint,
        const RegisteredSkinRigidPlacement::Matrix& referenceWorldToDonor,
        const RegisteredSkinRigidPlacement::Point& currentPoint,
        const RegisteredSkinRigidPlacement::Matrix& currentWorldToDonor,
        RegisteredSkinRigidPlacement& result,std::string& error) {
    using P=RegisteredSkinRigidPlacement::Point;
    using M=RegisteredSkinRigidPlacement::Matrix;
    const auto finite=[](const auto& a){return std::all_of(a.begin(),a.end(),
        [](double x){return std::isfinite(x);});};
    const auto dot=[](P a,P b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];};
    const auto cross=[](P a,P b){return P{a[1]*b[2]-a[2]*b[1],
        a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};};
    if(!finite(referencePoint)||!finite(currentPoint)||
       !finite(referenceWorldToDonor)||!finite(currentWorldToDonor)){
        error="skin placement has a nonfinite reference/current donor frame";return false;
    }
    const auto rigidFrame=[&](M m){
        P a{m[0],m[1],m[2]},b{m[3],m[4],m[5]},c{m[6],m[7],m[8]};
        return std::abs(dot(a,a)-1)<2e-5&&std::abs(dot(b,b)-1)<2e-5&&
            std::abs(dot(c,c)-1)<2e-5&&std::abs(dot(a,b))<2e-5&&
            std::abs(dot(a,c))<2e-5&&std::abs(dot(b,c))<2e-5&&
            std::abs(dot(cross(a,b),c)-1)<2e-5;
    };
    if(!rigidFrame(referenceWorldToDonor)||!rigidFrame(currentWorldToDonor)){
        error="skin placement requires proper orthonormal donor frames";return false;
    }
    RegisteredSkinRigidPlacement next;
    M raw{};
    // R = M_current^T M_reference.
    for(unsigned r=0;r<3;++r)for(unsigned c=0;c<3;++c)
        for(unsigned k=0;k<3;++k)
            raw[3*r+c]+=currentWorldToDonor[3*k+r]*referenceWorldToDonor[3*k+c];
    // Remove only Float32 basis roundoff, never a scale/shear or reflection.
    P a{raw[0],raw[3],raw[6]},b{raw[1],raw[4],raw[7]};
    const double aLength=std::sqrt(dot(a,a));
    for(double& x:a)x/=aLength;
    const double projection=dot(a,b);
    for(unsigned c=0;c<3;++c)b[c]-=projection*a[c];
    const double bLength=std::sqrt(dot(b,b));
    for(double& x:b)x/=bLength;
    const P c=cross(a,b);
    for(unsigned r=0;r<3;++r){
        next.rotation[3*r]=a[r];next.rotation[3*r+1]=b[r];next.rotation[3*r+2]=c[r];
    }
    for(unsigned i=0;i<9;++i)if(std::abs(raw[i]-next.rotation[i])>4e-5){
        error="skin placement donor composition is not a rigid rotation";return false;
    }
    const auto rotated=next.vector(referencePoint);
    for(unsigned i=0;i<3;++i)next.translation[i]=currentPoint[i]-rotated[i];
    result=next;return true;
}

inline bool validateRegisteredSkinPlacementVertex(
        const RegisteredSkinRigidPlacement::Point& referencePoint,
        const RegisteredSkinRigidPlacement::Point& referenceLocal,
        const RegisteredSkinRigidPlacement::Point& referenceNormal,
        const RegisteredSkinRigidPlacement::Matrix& referenceFrame,
        const RegisteredSkinRigidPlacement::Point& currentPoint,
        const RegisteredSkinRigidPlacement::Point& currentLocal,
        const RegisteredSkinRigidPlacement::Point& currentNormal,
        const RegisteredSkinRigidPlacement::Matrix& currentFrame,
        RegisteredSkinRigidPlacement& placement,std::string& error) {
    const auto predicted=placement.point(referencePoint);
    const auto normal=placement.vector(referenceNormal);
    const auto frame=placement.donorFrame(referenceFrame);
    for(unsigned c=0;c<3;++c){
        const double p=std::abs(currentPoint[c]-predicted[c]);
        const double l=std::abs(currentLocal[c]-referenceLocal[c]);
        const double n=std::abs(currentNormal[c]-normal[c]);
        if(!std::isfinite(p)||!std::isfinite(l)||!std::isfinite(n)||
           p>2e-5||l>2e-5||n>2e-4){
            error="skin pose is not a common rigid transfer of its admitted donor reference";return false;
        }
        placement.maximumPointResidualM=std::max(placement.maximumPointResidualM,p);
        placement.maximumLocalResidualM=std::max(placement.maximumLocalResidualM,l);
        placement.maximumNormalResidual=std::max(placement.maximumNormalResidual,n);
    }
    for(unsigned c=0;c<9;++c){
        const double d=std::abs(currentFrame[c]-frame[c]);
        if(!std::isfinite(d)||d>2e-5){
            error="skin pose donor rotations do not share the admitted rigid placement";return false;
        }
        placement.maximumFrameResidual=std::max(placement.maximumFrameResidual,d);
    }
    ++placement.validatedVertices;return true;
}
} // namespace metalrobo::human::coarse_tissue

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../include/numi/matter/shared.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

#ifndef NUMI_CARDBOARD_GEOMETRY_METALLIB
#error NUMI_CARDBOARD_GEOMETRY_METALLIB must identify the built geometry library
#endif

namespace {
using Vector = std::array<double, 3>;
constexpr std::size_t kRows = 133u;
constexpr std::size_t kWordsPerRow = 72u;
constexpr std::size_t kBytes = kRows * kWordsPerRow * sizeof(float);
void require(bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
}
Vector add(Vector a, Vector b) {
    for (unsigned i = 0u; i < 3u; ++i) a[i] += b[i];
    return a;
}
Vector subtract(Vector a, Vector b) {
    for (unsigned i = 0u; i < 3u; ++i) a[i] -= b[i];
    return a;
}
Vector scale(Vector a, double s) {
    for (double& x : a) x *= s;
    return a;
}
double dot(Vector a, Vector b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
double norm(Vector a) { return std::sqrt(dot(a, a)); }
Vector vectorAt(const float* row, unsigned slot) {
    return {row[slot*4u], row[slot*4u+1u], row[slot*4u+2u]};
}
double maximumDifference(Vector a, Vector b) {
    double value = 0.0;
    for (unsigned i = 0u; i < 3u; ++i)
        value = std::max(value, std::abs(a[i]-b[i]));
    return value;
}
double nearestError(const float* row) {
    const unsigned kind = static_cast<unsigned>(row[0]);
    const double size = row[1];
    require(kind < 6u && size > 0.0, "bad nearest-feature case");
    const Vector query = vectorAt(row, 1u);
    const Vector a = vectorAt(row, 2u), b = vectorAt(row, 3u);
    const Vector c = vectorAt(row, 4u), d = vectorAt(row, 5u);
    Vector parameters{}, first{}, second{};
    if (kind == 0u) {
        const double v = (query[0]-a[0])/(b[0]-a[0]);
        const double w = (query[1]-a[1])/(c[1]-a[1]);
        parameters = {1.0-v-w, v, w};
        first = {query[0], query[1], a[2]};
    } else if (kind == 1u) {
        const double v = (query[0]-a[0])/(b[0]-a[0]);
        parameters = {1.0-v, v, 0.0};
        first = {query[0], a[1], a[2]};
    } else if (kind == 2u) {
        parameters = {0.0, 0.0, 1.0};
        first = c;
    } else {
        double s = 0.0, t = 0.0;
        if (kind == 3u) {
            s = (c[0]-a[0])/(b[0]-a[0]);
            t = (a[1]-c[1])/(d[1]-c[1]);
        } else if (kind == 4u) {
            s = (c[0]-a[0])/(b[0]-a[0]);
        } else {
            t = (a[0]-c[0])/(d[0]-c[0]);
        }
        parameters = {s, t, 0.0};
        first = add(a, scale(subtract(b,a),s));
        second = add(c, scale(subtract(d,c),t));
    }
    double error = std::max(maximumDifference(vectorAt(row,7u),parameters),
        norm(subtract(vectorAt(row,9u),first))/size);
    if (kind >= 3u)
        error = std::max(error,norm(subtract(vectorAt(row,11u),second))/size);
    return error;
}

struct SegmentOracle {
    double s=0.0, t=0.0, distance=0.0;
    Vector relative{};
};
SegmentOracle segmentOracle(Vector p0, Vector p1, Vector q0, Vector q1) {
    const Vector u=subtract(p1,p0), v=subtract(q1,q0);
    const double uu=dot(u,u), vv=dot(v,v);
    SegmentOracle best;
    best.distance=INFINITY;
    auto consider=[&](double s,double t) {
        s=std::clamp(s,0.0,1.0); t=std::clamp(t,0.0,1.0);
        const Vector d=subtract(add(p0,scale(u,s)),add(q0,scale(v,t)));
        const double distance=norm(d);
        if(distance<best.distance) best={s,t,distance,d};
    };
    if(uu>0.0) {
        const double s0=dot(subtract(q0,p0),u)/uu;
        consider(std::clamp(s0,0.0,1.0),0.0);
        const double s1=dot(subtract(q1,p0),u)/uu;
        consider(std::clamp(s1,0.0,1.0),1.0);
    }
    if(vv>0.0) {
        const double t0=dot(subtract(p0,q0),v)/vv;
        consider(0.0,std::clamp(t0,0.0,1.0));
        const double t1=dot(subtract(p1,q0),v)/vv;
        consider(1.0,std::clamp(t1,0.0,1.0));
    }
    const double uv=dot(u,v), uw=dot(u,subtract(p0,q0)), vw=dot(v,subtract(p0,q0));
    const double determinant=uu*vv-uv*uv;
    if(determinant>0.0) {
        const double s=(uv*vw-vv*uw)/determinant;
        const double t=(uu*vw-uv*uw)/determinant;
        if(s>=0.0 && s<=1.0 && t>=0.0 && t<=1.0) consider(s,t);
    }
    require(std::isfinite(best.distance),"segment oracle produced no candidate");
    return best;
}
double barrierEnergy(double gap,double thickness,double stiffness) {
    if(!(thickness>0.0) || !(stiffness>0.0) || gap>=thickness) return 0.0;
    const double floor=1.0e-4*thickness;
    const double distance=std::max(gap,floor);
    const double d2=distance*distance, h2=thickness*thickness;
    const double offset=d2-h2, logRatio=std::log(d2/h2);
    const double at=-stiffness*offset*offset*logRatio;
    if(gap>=floor) return std::max(at,0.0);
    const double first=-stiffness*(2.0*offset*logRatio+offset*offset/d2);
    return std::max(at+2.0*distance*first*(gap-floor),0.0);
}
std::array<double,5> segmentErrors(const float* row) {
    const Vector p0=vectorAt(row,1u),p1=vectorAt(row,2u);
    const Vector q0=vectorAt(row,3u),q1=vectorAt(row,4u);
    const auto oracle=segmentOracle(p0,p1,q0,q1);
    const double reported=row[13u*4u+3u];
    const Vector relative=add(vectorAt(row,13u),vectorAt(row,14u));
    const double distanceError=std::abs(reported-oracle.distance);
    const double vectorError=norm(subtract(relative,oracle.relative));
    const double oldDistance=row[15u*4u];
    const double oldError=std::abs(oldDistance-oracle.distance);
    const Vector parameterRelative=subtract(
        add(p0,scale(subtract(p1,p0),row[7u*4u])),
        add(q0,scale(subtract(q1,q0),row[7u*4u+1u])));
    const double arithmeticVectorError=norm(subtract(relative,parameterRelative));
    const double distanceTolerance=row[0]<3.0f ? 2.0e-10 :
        std::max(2.0e-10,0.5*std::numeric_limits<float>::epsilon()*oracle.distance);
    require(distanceError<=distanceTolerance,
        "compensated EE gap exceeds its scale-aware numerical bound");
    require(row[7u*4u+2u]==1.0f,"compensated EE witness rejected valid geometry");
    require(row[16u*4u+2u]==1.0f,"EE barrier-energy evaluation rejected witness");
    require(row[16u*4u+1u]==reported,"barrier energy used a different EE separation");
    require(std::abs(row[17u*4u+3u]-1.0f)<=2.0e-6,
        "compensated EE normal is not finite and unit length");
    const double expectedEnergy=barrierEnergy(oracle.distance,1.0e-5,1.0)*row[16u*4u+3u];
    const double actualEnergy=row[16u*4u];
    const double energyError=std::abs(actualEnergy-expectedEnergy)/
        std::max(std::abs(expectedEnergy),1.0e-30);
    if(row[0]<5.0f) {
        require(std::abs(row[7u*4u]-oracle.s)<=2.0e-3 &&
                std::abs(row[7u*4u+1u]-oracle.t)<=2.0e-3,
            "EE segment parameters disagree with the double oracle");
    }
    const double parameterRoundingBound=
        std::abs(row[7u*4u]-oracle.s)*norm(subtract(p1,p0))+
        std::abs(row[7u*4u+1u]-oracle.t)*norm(subtract(q1,q0))+2.0e-10;
    require(vectorError<=parameterRoundingBound,
        "EE witness-vector error exceeds the reported FP32 parameter rounding bound");
    return {distanceError,vectorError,oldError,energyError,arithmeticVectorError};
}
std::array<double, 2> mollifierError(const float* row) {
    const Vector first = vectorAt(row,1u), second = vectorAt(row,2u);
    const double firstLength = norm(first), secondLength = norm(second);
    require(firstLength > 0.0 && secondLength > 0.0, "invalid mollifier case");
    const Vector u = scale(first,1.0/firstLength), v = scale(second,1.0/secondLength);
    const double cosine = dot(u,v);
    const Vector cross{u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]};
    const double ratio = std::clamp(dot(cross,cross)/1.0e-3,0.0,1.0);
    const double expected = ratio*(2.0-ratio);
    Vector gradientFirst{}, gradientSecond{};
    if (ratio > 0.0 && ratio < 1.0) {
        const double multiplier = 4.0*(1.0-ratio)*cosine/1.0e-3;
        gradientFirst = scale(subtract(scale(u,cosine),v),multiplier/firstLength);
        gradientSecond = scale(subtract(scale(v,cosine),u),multiplier/secondLength);
    }
    const double gradientError = std::max(
        norm(subtract(vectorAt(row,3u),gradientFirst))/std::max(norm(gradientFirst),1.0/firstLength),
        norm(subtract(vectorAt(row,4u),gradientSecond))/std::max(norm(gradientSecond),1.0/secondLength));
    return {std::abs(row[20u]-expected),gradientError};
}
struct TriangleOracle {
    std::array<double,3> bary{};
    Vector closest{}, relative{};
    double separation=0.0;
};
TriangleOracle triangleOracle(Vector p,Vector a,Vector b,Vector c) {
    TriangleOracle best; best.separation=INFINITY;
    const Vector ab=subtract(b,a),ac=subtract(c,a),ap=subtract(p,a);
    const double d00=dot(ab,ab),d01=dot(ab,ac),d11=dot(ac,ac);
    const double d20=dot(ap,ab),d21=dot(ap,ac);
    const double determinant=d00*d11-d01*d01;
    auto consider=[&](std::array<double,3> bary) {
        const Vector q=add(a,add(scale(ab,bary[1]),scale(ac,bary[2])));
        const Vector r=subtract(p,q); const double distance=norm(r);
        if(distance<best.separation) best={bary,q,r,distance};
    };
    if(determinant>std::numeric_limits<double>::epsilon()*d00*d11) {
        const double v=(d11*d20-d01*d21)/determinant;
        const double w=(d00*d21-d01*d20)/determinant;
        if(v>=0.0 && w>=0.0 && v+w<=1.0) consider({1.0-v-w,v,w});
    }
    const std::array<Vector,3> vertices={a,b,c};
    for(unsigned edge=0u;edge<3u;++edge) {
        const unsigned i=edge,j=(edge+1u)%3u;
        const Vector direction=subtract(vertices[j],vertices[i]);
        const double length2=dot(direction,direction);
        const double t=length2>0.0?std::clamp(dot(subtract(p,vertices[i]),direction)/length2,0.0,1.0):0.0;
        std::array<double,3> bary{}; bary[i]=1.0-t; bary[j]=t; consider(bary);
    }
    require(std::isfinite(best.separation),"triangle oracle produced no candidate");
    return best;
}
double triangleMollifier(Vector a,Vector b,Vector c) {
    const Vector u=subtract(b,a),v=subtract(c,a);
    const Vector crossUV={u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]};
    const double uu=dot(u,u),vv=dot(v,v);
    if(!(uu>0.0)||!(vv>0.0)) return 0.0;
    const double ratio=std::clamp((dot(crossUV,crossUV)/(uu*vv))/1.0e-3,0.0,1.0);
    return ratio*(2.0-ratio);
}
double vertexTrianglePotential(const std::array<Vector,4>& position,
                               double thickness,double stiffness) {
    const auto oracle=triangleOracle(
        position[0],position[1],position[2],position[3]);
    return triangleMollifier(position[1],position[2],position[3])*
        barrierEnergy(oracle.separation,thickness,stiffness);
}
struct ContactSlopeErrors {
    double finiteDifference;
    double analytic;
    double used;
    double rounded;
    double relativeError;
    double absoluteError;
    double roundedChordLength;
};
ContactSlopeErrors contactSlopeErrors(const float* row) {
    const unsigned test=static_cast<unsigned>(row[0]);
    require(test<8u,"invalid contact-slope fixture id");
    std::array<Vector,4> base{};
    for(unsigned i=0u;i<4u;++i)base[i]=vectorAt(row,1u+i);
    std::array<Vector,4> plus=base,minus=base;
    std::array<Vector,6> direction{};
    const double epsilon=row[3u];
    const double dt=row[2u];
    const double thickness=row[11u*4u];
    const double stiffness=row[11u*4u+1u];
    require(epsilon>0.0&&dt>0.0&&thickness>0.0&&stiffness>0.0,
        "invalid contact-slope potential parameters");
    for(unsigned i=0u;i<6u;++i)direction[i]=vectorAt(row,5u+i);
    for(unsigned i=0u;i<4u;++i) {
        plus[i]=add(base[i],scale(direction[i],epsilon));
        minus[i]=subtract(base[i],scale(direction[i],epsilon));
    }
    const double finiteDifference=(
        vertexTrianglePotential(plus,thickness,stiffness)-
        vertexTrianglePotential(minus,thickness,stiffness))/(2.0*epsilon);
    const double analytic=row[12u*4u];
    const double used=row[1u];
    const double rounded=row[12u*4u+1u];
    const double magnitude=std::max({std::abs(finiteDifference),
        std::abs(analytic),1.0e-12});
    const double absoluteError=std::abs(analytic-finiteDifference);
    const double relativeError=absoluteError/magnitude;
    const double roundedChordLength=row[12u*4u+2u];
    require(row[12u*4u+3u]==1.0f,"contact-slope fixture rejected its VT witness");
    if(test==0u)require(finiteDifference<0.0&&analytic<0.0,
        "positive normal displacement did not lower the barrier");
    if(test==1u)require(finiteDifference>0.0&&analytic>0.0,
        "negative normal displacement did not raise the barrier");
    if(test==2u||test==3u)
        require(std::abs(finiteDifference)>1.0e-8,
            "mixed or mollifier direction failed to exercise the potential");
    if(test==4u||test==5u)
        require(std::abs(used)<=1.0e-5&&std::abs(analytic)<=1.0e-5,
            "translation or fixed direction produced a contact slope");
    if(test==6u) {
        require(roundedChordLength==0.0&&std::abs(analytic)>1.0e-5,
            "below-ULP analytic direction did not retain its nonzero slope");
        require(std::abs(rounded)<=1.0e-5,
            "rounded endpoint chord unexpectedly retained a below-ULP slope");
    }
    if(test==7u) {
        require(roundedChordLength==0.0&&std::abs(finiteDifference)>1.0e-5,
            "rounded-chord negative control did not erase a nonzero oracle slope");
        require(std::abs(rounded)<=1.0e-5&&std::abs(used-rounded)<=1.0e-7,
            "negative control did not feed the rounded endpoint chord to the helper");
        require(std::abs(used-finiteDifference)>0.99*std::abs(finiteDifference),
            "rounded-chord negative control did not lose the analytic slope");
    } else {
        require(std::abs(used-analytic)<=1.0e-7,
            "contact slope fixture did not exercise its recorded analytic helper result");
    }
    if(test!=4u&&test!=5u)
        require(relativeError<=5.0e-3||absoluteError<=2.0e-5,
            "contact barrier directional derivative disagrees with independent FP64 centered energy difference");
    return {finiteDifference,analytic,used,rounded,relativeError,
        absoluteError,roundedChordLength};
}
struct FEMPrecisionErrors {
    double referencePositionErrorM;
    double trialPositionErrorM;
    double trialVelocityErrorMPerS;
    double trialReferenceDisplacementErrorM;
    double trialDeltaVelocityErrorMPerS;
    double trialNoOpChangedComponents;
    double contactGapErrorM;
    double contactEnergyRelativeError;
    double contactGapIncreaseM;
    double contactEnergyChangeJ;
    double roundedWorldGapChangeM;
    double roundedWorldEnergyChangeJ;
};
FEMPrecisionErrors femPrecisionErrors(const float* row) {
    const unsigned test=static_cast<unsigned>(row[0u]);
    require(test<5u,"invalid FEM reference-position fixture id");
    if(test==0u) {
        const Vector origin=vectorAt(row,1u),rest=vectorAt(row,2u),
            displacement=vectorAt(row,3u);
        const Vector actual=vectorAt(row,5u);
        const Vector expected=add(subtract(rest,origin),displacement);
        const Vector shiftedOrigin=vectorAt(row,7u),shiftedRest=vectorAt(row,8u),
            shiftedU=vectorAt(row,9u),shiftedActual=vectorAt(row,11u);
        const Vector shiftedExpected=add(subtract(shiftedRest,shiftedOrigin),shiftedU);
        const Vector worldControl=vectorAt(row,12u);
        const Vector shiftedWorldControl=vectorAt(row,13u);
        const double directError=norm(subtract(actual,expected));
        const double translatedError=norm(subtract(shiftedActual,shiftedExpected));
        require(directError<=5.0e-9&&translatedError<=5.0e-9,
            "reference-relative helper disagrees with the FP64 rest-origin plus displacement oracle");
        require(norm(subtract(actual,shiftedActual))<=5.0e-9,
            "common translation changed the reference-relative contact position");
        require(norm(subtract(actual,worldControl))>1.0e-6&&
                norm(subtract(shiftedActual,shiftedWorldControl))>1.0e-6,
            "sub-world-ULP reference displacement was not distinguished from rounded world subtraction");
        return {std::max(directError,translatedError),0.0,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,0.0};
    }
    if(test==1u||test==3u) {
        const float alpha=row[1u],dt=row[2u];
        const Vector rest=vectorAt(row,1u),origin=vectorAt(row,2u);
        const Vector acceptedPosition=vectorAt(row,3u);
        const Vector acceptedVelocity=vectorAt(row,4u);
        const Vector acceptedU=vectorAt(row,5u);
        const Vector correction=vectorAt(row,6u);
        const Vector trialPosition=vectorAt(row,7u);
        const Vector trialVelocity=vectorAt(row,8u);
        const Vector trialU=vectorAt(row,9u);
        const Vector trialRelative=vectorAt(row,10u);
        const Vector trialDelta=vectorAt(row,14u);
        Vector expectedVelocity{},expectedU{},expectedPosition{},expectedDelta{};
        for(unsigned axis=0u;axis<3u;++axis) {
            const float velocity=static_cast<float>(acceptedVelocity[axis])+
                alpha*static_cast<float>(correction[axis]);
            const float increment=dt*velocity;
            expectedVelocity[axis]=velocity;
            expectedDelta[axis]=velocity-static_cast<float>(acceptedVelocity[axis]);
            if(test==1u) {
                expectedU[axis]=static_cast<float>(
                    static_cast<float>(acceptedU[axis])+increment);
                expectedPosition[axis]=static_cast<float>(
                    static_cast<float>(rest[axis])+static_cast<float>(expectedU[axis]));
            } else {
                expectedU[axis]=acceptedU[axis];
                expectedPosition[axis]=static_cast<float>(
                    static_cast<float>(acceptedPosition[axis])+increment);
            }
        }
        double positionError=norm(subtract(trialPosition,expectedPosition));
        const double velocityError=norm(subtract(trialVelocity,expectedVelocity));
        const double displacementError=norm(subtract(trialU,expectedU));
        const double deltaVelocityError=norm(subtract(trialDelta,expectedDelta));
        positionError=std::max(positionError,
            norm(subtract(vectorAt(row,15u),vectorAt(row,1u))));
        require(row[7u*4u+3u]==row[3u*4u+3u]&&
                row[15u*4u+3u]==row[1u*4u+3u],
            "FEM trial changed node mass or fixed marker");
        const double velocityFieldError=std::abs(row[8u*4u+3u]-row[4u*4u+3u]);
        const double deltaFieldError=std::abs(row[14u*4u+3u]-row[16u*4u+3u]);
        const double displacementFieldError=std::abs(row[9u*4u+3u]-row[5u*4u+3u]);
        if(test==1u) {
            const Vector expectedRelative=add(subtract(rest,origin),trialU);
            const double relativeError=norm(subtract(trialRelative,expectedRelative));
            require(row[9u*4u+3u]==1.0f&&row[17u*4u+2u]==1.0f,
                "reference trial did not retain reference-displacement mode");
            require(relativeError<=5.0e-9,
                "reference trial contact position differs from the FP64 rest-origin plus displacement oracle");
            positionError=std::max(positionError,relativeError);
        } else {
            require(row[9u*4u+3u]==0.0f&&norm(trialU)<=1.0e-12&&
                    norm(trialRelative)==0.0,
                "legacy trial modified reference displacement or called the reference-relative path");
        }
        require(positionError<=2.0e-7&&displacementError<=2.0e-8&&
                velocityError<=2.0e-8&&deltaVelocityError<=2.0e-8&&
                velocityFieldError==0.0&&deltaFieldError==0.0&&
                displacementFieldError==0.0,
            "FEM trial state differs from the exact dt-times-velocity update");
        return {0.0,positionError,velocityError,displacementError,
            deltaVelocityError,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    }
    if(test==2u) {
        double changed=0.0;
        const unsigned beforeAfter[][2]={{1u,2u},{3u,4u},{5u,6u},{7u,8u},
            {9u,10u},{11u,12u},{13u,14u},{15u,16u}};
        for(const auto& pair:beforeAfter)
            for(unsigned axis=0u;axis<4u;++axis)
                if(row[pair[0]*4u+axis]!=row[pair[1]*4u+axis])changed+=1.0;
        require(changed==0.0,
            "fixed or zero-alpha FEM trial changed a state component");
        require(row[7u*4u+3u]==row[8u*4u+3u]&&
                row[15u*4u+3u]==row[16u*4u+3u]&&
                row[17u*4u]==1.0f&&row[17u*4u+1u]==1.0f,
            "fixed or zero-alpha FEM trial changed reference mode or rest state");
        return {0.0,0.0,0.0,0.0,0.0,changed,0.0,0.0,0.0,0.0,0.0,0.0};
    }
    const Vector basePoint=vectorAt(row,2u),baseA=vectorAt(row,3u),
        baseB=vectorAt(row,4u),baseC=vectorAt(row,5u);
    const Vector trialPoint=vectorAt(row,6u),trialA=vectorAt(row,7u),
        trialB=vectorAt(row,8u),trialC=vectorAt(row,9u);
    const Vector worldPoint=vectorAt(row,10u),worldA=vectorAt(row,11u),
        worldB=vectorAt(row,12u),worldC=vectorAt(row,13u);
    const auto baseOracle=triangleOracle(basePoint,baseA,baseB,baseC);
    const auto trialOracle=triangleOracle(trialPoint,trialA,trialB,trialC);
    const auto worldOracle=triangleOracle(worldPoint,worldA,worldB,worldC);
    const double reportedBaseGap=row[14u*4u];
    const double reportedTrialGap=row[14u*4u+1u];
    const double reportedWorldGap=row[14u*4u+2u];
    const double reportedBaseEnergy=row[14u*4u+3u];
    const double reportedTrialEnergy=row[15u*4u];
    const double reportedWorldEnergy=row[15u*4u+1u];
    const double thickness=row[15u*4u+2u],stiffness=row[15u*4u+3u];
    const double baseEnergy=vertexTrianglePotential(
        {basePoint,baseA,baseB,baseC},thickness,stiffness);
    const double trialEnergy=vertexTrianglePotential(
        {trialPoint,trialA,trialB,trialC},thickness,stiffness);
    const double worldEnergy=vertexTrianglePotential(
        {worldPoint,worldA,worldB,worldC},thickness,stiffness);
    const double gapError=std::max({
        std::abs(reportedBaseGap-baseOracle.separation),
        std::abs(reportedTrialGap-trialOracle.separation),
        std::abs(reportedWorldGap-worldOracle.separation)});
    const double energyError=std::max({
        std::abs(reportedBaseEnergy-baseEnergy),
        std::abs(reportedTrialEnergy-trialEnergy),
        std::abs(reportedWorldEnergy-worldEnergy)})/
        std::max({std::abs(baseEnergy),std::abs(trialEnergy),1.0e-30});
    const double gapIncrease=reportedTrialGap-reportedBaseGap;
    const double energyChange=reportedTrialEnergy-reportedBaseEnergy;
    const double roundedWorldGapChange=reportedWorldGap-reportedBaseGap;
    const double roundedWorldEnergyChange=reportedWorldEnergy-reportedBaseEnergy;
    require(row[0u*4u+1u]==1.0f&&row[0u*4u+2u]==1.0f&&
            row[0u*4u+3u]==1.0f&&row[17u*4u+3u]==1.0f,
        "reference-trial VF witness or barrier evaluation failed");
    require(gapError<=5.0e-9&&energyError<=5.0e-4,
        "reference-trial VF gap or barrier energy disagrees with the independent FP64 oracle");
    require(gapIncrease>1.0e-6&&energyChange<0.0,
        "actual femTrialNodeAtAlpha relative contact did not descend the barrier");
    require(std::abs(roundedWorldGapChange)<=5.0e-9&&
            std::abs(roundedWorldEnergyChange)<=
                5.0e-4*std::max(std::abs(reportedBaseEnergy),1.0e-30),
        "rounded-world negative control did not erase the sub-world-ULP trial movement");
    require(row[16u*4u+3u]>0.0f&&row[17u*4u+2u]==1.0f,
        "reference trial displacement or mode was not retained");
    return {0.0,0.0,0.0,0.0,0.0,0.0,gapError,energyError,
        gapIncrease,energyChange,roundedWorldGapChange,roundedWorldEnergyChange};
}

struct FEMExpansionErrors {
    double coordinateErrorM;
    double barycentricError;
    double gapErrorM;
    double energyRelativeError;
    double deltaEnergyRelativeError;
    double gapIncreaseM;
    double energyChangeJ;
    double translationErrorM;
    double legacyGapChangeM;
    double legacyEnergyChangeJ;
};
FEMExpansionErrors femExpansionErrors(const float* row) {
    const unsigned test=static_cast<unsigned>(row[0u]);
    require(test<5u,"invalid compensated FEM expansion fixture id");
    if(test<4u) {
        const float scales[4]={0.0008f,0.0016f,0.0032f,0.008f};
        const float scaleValue=scales[test];
        const float basePointValues[4][3]={
            {0.00022f,0.25f*scaleValue,0.25f*scaleValue},
            {0.0002f,0.0f,0.0f},
            {0.0002f,scaleValue,0.0f},
            {0.0002f,0.0f,scaleValue}};
        std::array<Vector,4> base{},trial{},expected{};
        const float localRestX=0.0048828125f;
        double coordinateError=0.0;
        for(unsigned i=0u;i<4u;++i) {
            const unsigned highSlot=i==0u?1u:5u+2u*(i-1u);
            const unsigned lowSlot=highSlot+1u;
            base[i]=add(vectorAt(row,highSlot),vectorAt(row,lowSlot));
            expected[i]={double(localRestX)+double(basePointValues[i][0]),
                         double(basePointValues[i][1]),double(basePointValues[i][2])};
            coordinateError=std::max(coordinateError,
                norm(subtract(base[i],expected[i])));
        }
        const float dt=0.0009765625f;
        const float correction=1.490116119384765625e-8f;
        const float expectedTrialU=basePointValues[0][0]+correction*dt;
        require(row[17u*4u+3u]==1.0f&&
                row[17u*4u]==expectedTrialU&&
                row[17u*4u+1u]==basePointValues[0][1]&&
                row[17u*4u+2u]==basePointValues[0][2],
            "femTrialNodeAtAlpha did not preserve the authored reference displacement update");
        for(unsigned i=0u;i<4u;++i)trial[i]=base[i];
        trial[0][0]=double(localRestX)+double(row[17u*4u]);
        const Vector trialPoint=add(vectorAt(row,3u),vectorAt(row,4u));
        coordinateError=std::max(coordinateError,norm(subtract(trial[0],trialPoint)));
        const auto baseOracle=triangleOracle(base[0],base[1],base[2],base[3]);
        const auto trialOracle=triangleOracle(trial[0],trial[1],trial[2],trial[3]);
        const Vector baseReportedBary=vectorAt(row,11u);
        const Vector trialReportedBary=vectorAt(row,12u);
        const double baryError=std::max({
            maximumDifference(baseReportedBary,baseOracle.bary),
            maximumDifference(trialReportedBary,trialOracle.bary)});
        const double baseGapError=std::abs(row[11u*4u+3u]-baseOracle.separation);
        const double trialGapError=std::abs(row[12u*4u+3u]-trialOracle.separation);
        const double gapError=std::max(baseGapError,trialGapError);
        constexpr double thickness=1.0e-4f;
        constexpr double stiffness=1.8e13f;
        const double baseEnergy=barrierEnergy(baseOracle.separation,thickness,stiffness)*
            triangleMollifier(base[1],base[2],base[3]);
        const double trialEnergy=barrierEnergy(trialOracle.separation,thickness,stiffness)*
            triangleMollifier(trial[1],trial[2],trial[3]);
        const double reportedBaseEnergy=row[13u*4u];
        const double reportedTrialEnergy=row[13u*4u+1u];
        const double energyError=std::max(
            std::abs(reportedBaseEnergy-baseEnergy),
            std::abs(reportedTrialEnergy-trialEnergy))/
            std::max({std::abs(baseEnergy),std::abs(trialEnergy),1.0e-30});
        const double expectedEnergyChange=trialEnergy-baseEnergy;
        const double energyChange=reportedTrialEnergy-reportedBaseEnergy;
        const double deltaEnergyError=std::abs(energyChange-expectedEnergyChange)/
            std::max(std::abs(expectedEnergyChange),1.0e-30);
        const double gapIncrease=row[13u*4u+3u]-row[13u*4u+2u];
        require(row[0u*4u+1u]==1.0f&&row[0u*4u+2u]==1.0f&&
                row[0u*4u+3u]==1.0f,
            "compensated FEM witness or barrier path rejected a source row");
        if(!(coordinateError<=2.0e-15&&baryError<=5.0e-7&&gapError<=1.0e-11)) {
            std::fprintf(stderr,
                "fem_expansion_failure row=%u coordinate_error_m=%.17g bary_error=%.17g gap_error_m=%.17g "
                "base_bary_actual=(%.9g,%.9g,%.9g) base_bary_fp64=(%.17g,%.17g,%.17g) "
                "base_gap_actual_m=%.17g base_gap_fp64_m=%.17g "
                "trial_bary_actual=(%.9g,%.9g,%.9g) trial_bary_fp64=(%.17g,%.17g,%.17g) "
                "trial_gap_actual_m=%.17g trial_gap_fp64_m=%.17g\n",
                test,coordinateError,baryError,gapError,
                baseReportedBary[0],baseReportedBary[1],baseReportedBary[2],
                baseOracle.bary[0],baseOracle.bary[1],baseOracle.bary[2],
                double(row[11u*4u+3u]),baseOracle.separation,
                trialReportedBary[0],trialReportedBary[1],trialReportedBary[2],
                trialOracle.bary[0],trialOracle.bary[1],trialOracle.bary[2],
                double(row[12u*4u+3u]),trialOracle.separation);
            for(unsigned i=0u;i<4u;++i)
                std::fprintf(stderr,
                    "fem_expansion_row=%u point=%u actual=(%.17g,%.17g,%.17g) fp64=(%.17g,%.17g,%.17g)\n",
                    test,i,base[i][0],base[i][1],base[i][2],
                    expected[i][0],expected[i][1],expected[i][2]);
            std::fprintf(stderr,
                "fem_expansion_row=%u trial_point_actual=(%.17g,%.17g,%.17g) fp64=(%.17g,%.17g,%.17g)\n",
                test,trialPoint[0],trialPoint[1],trialPoint[2],
                trial[0][0],trial[0][1],trial[0][2]);
        }
        require(coordinateError<=2.0e-15&&baryError<=5.0e-7&&gapError<=1.0e-11,
            "compensated FEM VF frame, barycentrics, or gap differs from the FP64 source-coordinate oracle");
        require(energyError<=5.0e-4&&gapIncrease>1.0e-11&&
                expectedEnergyChange<0.0&&energyChange<0.0&&
                deltaEnergyError<=0.25,
            "compensated FEM VF potential or sub-world-ULP trial descent differs from the FP64 oracle");
        require(row[17u*4u+3u]==1.0f&&
                std::abs(row[14u*4u]-row[14u*4u+1u])==0.0&&
                std::abs(row[14u*4u+2u]-row[14u*4u+3u])==0.0,
            "rounded-world negative control did not erase the reference-only trial movement");
        for(unsigned axis=0u;axis<3u;++axis)
            require(row[15u*4u+axis]==row[16u*4u+axis],
                "rounded-world negative control changed the projected point endpoint");
        return {coordinateError,baryError,gapError,energyError,deltaEnergyError,
            gapIncrease,energyChange,0.0,
            row[14u*4u+3u]-row[14u*4u+2u],
            row[14u*4u+1u]-row[14u*4u]};
    }
    require(row[0u*4u+1u]==1.0f&&row[0u*4u+2u]==1.0f&&
            row[0u*4u+3u]==1.0f,
        "common-translation compensated witness or energy path rejected a source row");
    double translationError=0.0;
    for(unsigned i=0u;i<5u;++i) {
        const unsigned slot=1u+2u*i;
        const Vector delta=add(vectorAt(row,slot),vectorAt(row,slot+1u));
        translationError=std::max(translationError,norm(delta));
    }
    const Vector baseBary=vectorAt(row,13u),translatedBaseBary=vectorAt(row,14u);
    const Vector trialBary=vectorAt(row,15u),translatedTrialBary=vectorAt(row,16u);
    const double baryError=std::max(
        maximumDifference(baseBary,translatedBaseBary),
        maximumDifference(trialBary,translatedTrialBary));
    const double baseGapError=std::abs(row[11u*4u]-row[12u*4u]);
    const double trialGapError=std::abs(row[11u*4u+1u]-row[12u*4u+1u]);
    const double energyError=std::max(
        std::abs(row[17u*4u]-row[17u*4u+2u]),
        std::abs(row[17u*4u+1u]-row[17u*4u+3u]))/
        std::max({std::abs(double(row[17u*4u])),
                  std::abs(double(row[17u*4u+1u])),1.0e-30});
    const double gapIncrease=row[11u*4u+1u]-row[11u*4u];
    const double energyChange=row[17u*4u+1u]-row[17u*4u];
    require(translationError<=2.0e-15&&baryError<=5.0e-7&&
            std::max(baseGapError,trialGapError)<=1.0e-11,
        "common translation changed compensated VF coordinates, barycentrics, or gap");
    require(energyError<=5.0e-4&&gapIncrease>1.0e-11&&energyChange<0.0,
        "common translation changed compensated barrier energy or trial descent");
    return {translationError,baryError,std::max(baseGapError,trialGapError),
        energyError,0.0,gapIncrease,energyChange,translationError,0.0,0.0};
}

struct TriangleErrors { double gap,relative,bary,energy,legacy,oracleGap,oracleNormal; };
TriangleErrors triangleErrors(const float* row) {
    const Vector p=vectorAt(row,1u),a=vectorAt(row,2u),b=vectorAt(row,3u),c=vectorAt(row,4u);
    const auto oracle=triangleOracle(p,a,b,c);
    const Vector bary=vectorAt(row,5u),relative=add(vectorAt(row,7u),vectorAt(row,8u));
    const double baryError=std::max({std::abs(bary[0]-oracle.bary[0]),std::abs(bary[1]-oracle.bary[1]),std::abs(bary[2]-oracle.bary[2])});
    const Vector ab=subtract(b,a),ac=subtract(c,a);
    const Vector closestFromBary=add(a,add(scale(ab,bary[1]),scale(ac,bary[2])));
    const Vector expectedRelative=subtract(p,closestFromBary);
    const double parameterBound=std::abs(bary[1]-oracle.bary[1])*norm(ab)+std::abs(bary[2]-oracle.bary[2])*norm(ac);
    const double relativeError=norm(subtract(relative,expectedRelative));
    const double gapError=std::abs(row[1u]-norm(expectedRelative));
    const double oracleGapError=std::abs(row[1u]-oracle.separation);
    const double roundingBound=2.0e-9+5.0e-7*(norm(ab)+norm(ac));
    require(row[2u]==1.0f,"VF compensated witness rejected valid geometry");
    require(row[10u*4u+3u]==1.0f&&row[14u*4u+2u]==1.0f,"VF barrier evaluation rejected witness");
    require(baryError<=3.0e-5,"VF barycentrics disagree with independent FP64 closest point");
    require(relativeError<=parameterBound+roundingBound,"VF relative vector disagrees with independent FP64 geometry");
    require(gapError<=parameterBound+roundingBound,"VF gap disagrees with independent FP64 geometry");
    const Vector normal=vectorAt(row,9u);
    require(std::abs(norm(normal)-1.0)<=2.0e-5&&norm(subtract(normal,scale(relative,1.0/std::max(double(row[1u]),1.0e-30))))<=3.0e-4,
        "VF normal does not share the compensated witness direction");
    const double oracleNormalError=norm(subtract(normal,scale(oracle.relative,1.0/std::max(oracle.separation,1.0e-30))));
    if(row[0u]<2.0f) require(oracleGapError<=2.0e-10&&oracleNormalError<=1.0e-6,
        "captured VF gap or normal differs from the independent FP64 witness");
    const Vector cp=vectorAt(row,6u); double worldError=0.0,worldUlp=0.0;
    for(unsigned axis=0u;axis<3u;++axis) {
        worldError=std::max(worldError,std::abs(cp[axis]-oracle.closest[axis]));
        const float f=static_cast<float>(oracle.closest[axis]);
        worldUlp=std::max(worldUlp,static_cast<double>(std::nextafterf(f,INFINITY)-f));
    }
    require(worldError<=worldUlp+parameterBound+2.0e-9,"VF closest-point presentation exceeds FP32 rounding bound");
    const double thickness=row[10u*4u],stiffness=row[10u*4u+1u];
    const double mollifier=triangleMollifier(a,b,c);
    const double expectedEnergy=barrierEnergy(row[1u],thickness,stiffness)*mollifier;
    const double energyError=std::abs(row[14u*4u]-expectedEnergy)/std::max(std::abs(expectedEnergy),1.0e-30);
    require(std::abs(row[14u*4u+1u]-row[1u])<=1.0e-12,"VF barrier energy used a different gap");
    require(std::abs(row[14u*4u+3u]-mollifier)<=3.0e-4,"VF barrier mollifier differs from FP64");
    require(energyError<=3.0e-3,"VF barrier energy differs from FP64");
    const Vector legacyRelative=subtract(p,vectorAt(row,12u));
    return {gapError,relativeError,baryError,energyError,norm(subtract(legacyRelative,oracle.relative)),oracleGapError,oracleNormalError};
}
std::array<double,4> mappedWitnessErrors(const float* row) {
    const Vector p=vectorAt(row,1u),a=vectorAt(row,2u),b=vectorAt(row,3u),c=vectorAt(row,4u);
    const auto oracle=triangleOracle(p,a,b,c);
    const unsigned side=static_cast<unsigned>(row[3u]); const double separation=row[1u];
    require(row[2u]==1.0f&&(side==0u||side==1u),"mapped VF helper rejected witness or invalid side");
    Vector weighted{}; double weightSum=0.0;
    for(unsigned i=0u;i<6u;++i) {
        const double w=i<4u?row[11u*4u+i]:row[12u*4u+i-4u];
        const Vector node=i<3u?vectorAt(row,5u+i):vectorAt(row,8u+i-3u);
        weightSum+=w; weighted=add(weighted,scale(node,w));
    }
    const Vector normal=vectorAt(row,14u);
    require(norm(subtract(vectorAt(row,15u),oracle.closest))<=5.0e-9,
        "mapped VF contact point is not the triangle projection");
    const Vector torque={weighted[1]*normal[2]-weighted[2]*normal[1],weighted[2]*normal[0]-weighted[0]*normal[2],weighted[0]*normal[1]-weighted[1]*normal[0]};
    const double witnessError=norm(subtract(weighted,oracle.relative));
    const bool fallback=row[0u]==2.0f;
    const double directionError=fallback
        ? std::abs(dot(weighted,normal)-dot(oracle.relative,normal))
        : std::abs(dot(weighted,normal)-separation);
    const Vector expectedTorque=fallback
        ? Vector{oracle.relative[1]*normal[2]-oracle.relative[2]*normal[1],
            oracle.relative[2]*normal[0]-oracle.relative[0]*normal[2],
            oracle.relative[0]*normal[1]-oracle.relative[1]*normal[0]}
        : Vector{};
    const double torqueError=norm(subtract(torque,expectedTorque));
    require(std::abs(weightSum)<=3.0e-6,"mapped VF support weights do not conserve net force");
    require(witnessError<=2.0e-6+3.0e-5*norm(oracle.relative),"mapped VF weights do not reproduce independent relative witness");
    require(directionError<=2.0e-6+3.0e-5*separation,"mapped VF force direction does not recover physical gap");
    require(torqueError<=2.0e-6+3.0e-5*separation,"mapped VF torque differs from analytical force moment");
    require(norm(subtract(weighted,vectorAt(row,13u)))<=2.0e-7,"mapped VF weighted-position diagnostic mismatch");
    if(fallback) require(separation<1.0e-12&&norm(subtract(normal,vectorAt(row,16u)))<=1.0e-6,
        "tiny-gap mapped VF did not use configured fallback normal");
    else require(std::abs(norm(normal)-1.0)<=2.0e-5,"mapped VF normal is not unit length");
    return {std::abs(weightSum),witnessError,directionError,torqueError};
}
void writeExclusive(const std::string& path,const void* data,std::size_t bytes) {
    const int fd = ::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    require(fd >= 0,"output must be fresh");
    const auto* cursor = static_cast<const char*>(data);
    while (bytes != 0u) {
        const ssize_t count = ::write(fd,cursor,bytes);
        require(count > 0,"output write failed");
        cursor += count;
        bytes -= static_cast<std::size_t>(count);
    }
    require(::fsync(fd)==0,"output fsync failed");
    require(::close(fd)==0,"output close failed");
}

constexpr std::uint32_t kSortEnvironmentCount = 2u;
constexpr std::uint32_t kSortSentinel = 0xcafef00du;
struct SurfaceSortCase {
    std::uint32_t id=0u, primitiveCount=0u, surfaceFaceCount=0u, gridNodeCount=0u, useCache=0u;
    std::array<std::uint32_t,kSortEnvironmentCount> cacheFlags{};
    unsigned activityMode=0u;
    std::vector<NMContinuumSurfacePrimitiveGPU> primitives;
};
struct SurfaceSortExpected {
    std::vector<std::uint32_t> keys, indices;
};
std::uint32_t spreadMortonBits(std::uint32_t value) {
    value &= 0x3ffu;
    value = (value | (value << 16u)) & 0x030000ffu;
    value = (value | (value << 8u)) & 0x0300f00fu;
    value = (value | (value << 4u)) & 0x030c30c3u;
    return (value | (value << 2u)) & 0x09249249u;
}
std::uint32_t referenceMortonKey(const std::array<float,3>& center,
    const std::array<float,3>& minimum, const std::array<float,3>& inverseExtent) {
    std::array<std::uint32_t,3> q{};
    for (unsigned axis=0u; axis<3u; ++axis) {
        const float normalized=(center[axis]-minimum[axis])*inverseExtent[axis];
        q[axis]=static_cast<std::uint32_t>(std::clamp(normalized*1023.0f,0.0f,1023.0f));
    }
    return spreadMortonBits(q[0]) | (spreadMortonBits(q[1])<<1u) |
        (spreadMortonBits(q[2])<<2u);
}
bool sortCaseActive(unsigned mode, unsigned environment, std::uint32_t index) {
    if (mode==0u) return false;
    if (mode==1u) return true;
    if (index<2u) return true;
    if (mode==2u)
        return environment==0u ? (index%4u!=1u && index%9u!=3u)
                               : (index%5u!=0u && index%7u!=2u);
    if (mode==3u) return index%17u!=0u && (index+environment*11u)%23u!=0u;
    return false;
}
std::array<float,3> sortCaseCentroid(unsigned environment, std::uint32_t index) {
    if (index==0u) return {0.0f,0.0f,0.0f};
    if (index==1u) return {1.0f,1.0f,1.0f};
    if (index%7u==2u) return {0.5f,0.25f,0.75f};
    const float x=static_cast<float>(index%8u)/8.0f;
    return {environment==0u ? x : 1.0f-x,
        static_cast<float>((index/8u)%8u)/8.0f,
        static_cast<float>((index/64u)%4u)/4.0f};
}
SurfaceSortCase makeSurfaceSortCase(std::uint32_t id, std::uint32_t count,
    std::uint32_t faceCount, unsigned activityMode, std::uint32_t useCache,
    std::array<std::uint32_t,kSortEnvironmentCount> cacheFlags) {
    require(faceCount<=count,"surface sort face count exceeds primitive count");
    SurfaceSortCase result;
    result.id=id; result.primitiveCount=count; result.surfaceFaceCount=faceCount;
    result.gridNodeCount=count-faceCount; result.activityMode=activityMode;
    result.useCache=useCache; result.cacheFlags=cacheFlags;
    result.primitives.resize(static_cast<std::size_t>(kSortEnvironmentCount)*count);
    for (std::uint32_t env=0u; env<kSortEnvironmentCount; ++env) {
        for (std::uint32_t local=0u; local<count; ++local) {
            const auto center=sortCaseCentroid(env,local);
            constexpr float hx=1.0f/1024.0f, hy=1.0f/512.0f, hz=1.0f/256.0f;
            NMContinuumSurfacePrimitiveGPU primitive{};
            primitive.nodesAndObject={local,local+1u,local+2u,env};
            primitive.identity={local,
                local<faceCount ? NM_CONTINUUM_SURFACE_TRIANGLE : NM_CONTINUUM_SURFACE_POINT,
                1u,sortCaseActive(activityMode,env,local) ? NM_TOPOLOGY_ACTIVE : 0u};
            primitive.boundsMinimum={center[0]-hx,center[1]-hy,center[2]-hz,1.0f};
            primitive.boundsMaximum={center[0]+hx,center[1]+hy,center[2]+hz,1.0f/64.0f};
            result.primitives[static_cast<std::size_t>(env)*count+local]=primitive;
        }
    }
    return result;
}
std::vector<SurfaceSortExpected> surfaceSortReference(const SurfaceSortCase& test) {
    std::vector<SurfaceSortExpected> result(kSortEnvironmentCount);
    for (std::uint32_t env=0u; env<kSortEnvironmentCount; ++env) {
        if (test.useCache!=0u && test.cacheFlags[env]!=0u) continue;
        std::array<float,3> minimum={INFINITY,INFINITY,INFINITY};
        std::array<float,3> maximum={-INFINITY,-INFINITY,-INFINITY};
        std::vector<std::pair<std::uint32_t,std::array<float,3>>> active;
        for (std::uint32_t local=0u; local<test.primitiveCount; ++local) {
            const auto& p=test.primitives[static_cast<std::size_t>(env)*test.primitiveCount+local];
            if ((p.identity.w&NM_TOPOLOGY_ACTIVE)==0u) continue;
            const std::array<float,3> center={
                0.5f*(p.boundsMinimum.x+p.boundsMaximum.x),
                0.5f*(p.boundsMinimum.y+p.boundsMaximum.y),
                0.5f*(p.boundsMinimum.z+p.boundsMaximum.z)};
            for (unsigned axis=0u; axis<3u; ++axis) {
                minimum[axis]=std::min(minimum[axis],center[axis]);
                maximum[axis]=std::max(maximum[axis],center[axis]);
            }
            active.emplace_back(local,center);
        }
        std::array<float,3> inverseExtent{};
        for (unsigned axis=0u; axis<3u; ++axis)
            inverseExtent[axis]=1.0f/std::max(maximum[axis]-minimum[axis],1.0e-12f);
        std::vector<std::pair<std::uint32_t,std::uint32_t>> keyed;
        keyed.reserve(active.size());
        for (const auto& item:active)
            keyed.emplace_back(referenceMortonKey(item.second,minimum,inverseExtent),item.first);
        std::stable_sort(keyed.begin(),keyed.end(),
            [](const auto& a,const auto& b){return a.first<b.first;});
        for (const auto& item:keyed) {
            result[env].keys.push_back(item.first);
            result[env].indices.push_back(item.second);
        }
    }
    return result;
}
id<MTLBuffer> sharedSortBuffer(id<MTLDevice> device,const void* data,std::size_t bytes) {
    const std::size_t allocation=std::max<std::size_t>(bytes,sizeof(std::uint32_t));
    id<MTLBuffer> buffer=[device newBufferWithLength:allocation options:MTLResourceStorageModeShared];
    require(buffer!=nil,"surface sort buffer allocation failed");
    std::memset(buffer.contents,0,allocation);
    if (data!=nullptr && bytes!=0u) std::memcpy(buffer.contents,data,bytes);
    return buffer;
}
struct FEMReferenceAccumulationAudit {
    double displacementErrorM=0.0;
    double accumulatedLowM=0.0;
    double expectedIncrementM=0.0;
    double eightStepErrorM=0.0;
    double eightStepIncrementM=0.0;
    double expectedEightStepIncrementM=0.0;
    double legacyPositionChangeM=0.0;
    unsigned updates=0u;
    unsigned eightStepUpdates=0u;
    bool fixedExact=false;
    bool nondueExact=false;
    bool legacyExact=false;
};
NMFEMNodeStateGPU makeReferenceFixtureNode(bool referenceMode, bool fixed) {
    NMFEMNodeStateGPU state{};
    const nm_float4 origin={8192.0f,-4096.0f,2048.0f,0.0f};
    const float queryU=40.0f*0x1p-20f;
    state.restAndFixed={origin.x+0.001953125f,origin.y+0.001953125f,
        origin.z, fixed?1.0f:0.0f};
    const float displacement=referenceMode?queryU:0.0f;
    state.referenceDisplacementAndMode={displacement,displacement,displacement,
        referenceMode?1.0f:0.0f};
    state.referenceDisplacementLow={0.0f,0.0f,0.0f,0.0f};
    state.positionAndMass={
        state.restAndFixed.x+queryU,
        state.restAndFixed.y+queryU,
        state.restAndFixed.z+queryU,1.0f};
    state.velocityAndInverseMass={0x1p-31f,0.0f,0.0f,1.0f};
    return state;
}
FEMReferenceAccumulationAudit runReferenceAccumulationAudit(
    id<MTLDevice> device,id<MTLLibrary> library,id<MTLCommandQueue> queue) {
    NSError* error=nil;
    id<MTLFunction> integrateFunction=[library newFunctionWithName:
        @"numi_matter_metal::nm_fem_integrate"];
    id<MTLFunction> commitFunction=[library newFunctionWithName:
        @"numi_matter_metal::nm_fem_commit_microstep"];
    require(integrateFunction!=nil && commitFunction!=nil,
        "missing production FEM integration/commit kernels");
    id<MTLComputePipelineState> integratePipeline=[device
        newComputePipelineStateWithFunction:integrateFunction error:&error];
    require(integratePipeline!=nil,"cannot create production FEM integrate pipeline");
    id<MTLComputePipelineState> commitPipeline=[device
        newComputePipelineStateWithFunction:commitFunction error:&error];
    require(commitPipeline!=nil,"cannot create production FEM commit pipeline");
    NMMatterDispatchGPU dispatch{};
    dispatch.abiVersion=NM_MATTER_ABI_VERSION;
    dispatch.environmentCount=1u;
    dispatch.objectCount=1u;
    dispatch.femNodeCount=1u;
    dispatch.materialStateStride=1u;
    dispatch.gravityAndTimestep={0.0f,0.0f,0.0f,0x1p-10f};
    NMContinuumObjectGPU object{};
    object.representation=NM_REPRESENTATION_FEM;
    object.flags=NM_OBJECT_ACTIVE;
    object.schedulerIndex=0u;
    object.solver={0u,0u,0u,0u};
    NMSchedulerStateGPU scheduler{};
    scheduler.activeExponent=0u;
    scheduler.numerical.w=1.0f;
    NMAdaptiveStateGPU adaptive{};
    adaptive.activeRepresentation=NM_REPRESENTATION_FEM;
    NMIncidenceRangeGPU nodeRange{};
    nodeRange.objectIndex=0u;
    NMMicrostepGPU microstep{};
    microstep.microtickCount=1u;
    microstep.time={0x1p-10f,0x1p10f,0.0f,0x1p-10f};
    NMMatterStatusGPU status{};
    float dummyMaterialState=0.0f;
    auto dispatchBuffer=sharedSortBuffer(device,&dispatch,sizeof(dispatch));
    auto objectBuffer=sharedSortBuffer(device,&object,sizeof(object));
    auto nodeRangeBuffer=sharedSortBuffer(device,&nodeRange,sizeof(nodeRange));
    auto schedulerBuffer=sharedSortBuffer(device,&scheduler,sizeof(scheduler));
    auto adaptiveBuffer=sharedSortBuffer(device,&adaptive,sizeof(adaptive));
    auto microstepBuffer=sharedSortBuffer(device,&microstep,sizeof(microstep));
    auto statusBuffer=sharedSortBuffer(device,&status,sizeof(status));
    auto materialAcceptedBuffer=sharedSortBuffer(device,&dummyMaterialState,sizeof(float));
    auto materialCandidateBuffer=sharedSortBuffer(device,&dummyMaterialState,sizeof(float));
    auto initializeNodeBuffers=[&](NMFEMNodeStateGPU initial,
        id<MTLBuffer> __strong& acceptedBuffer,id<MTLBuffer> __strong& candidateBuffer) {
        acceptedBuffer=sharedSortBuffer(device,&initial,sizeof(initial));
        candidateBuffer=sharedSortBuffer(device,&initial,sizeof(initial));
    };
    auto encodeIntegrate=[&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setComputePipelineState:integratePipeline];
        [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];
        [encoder setBuffer:microstepBuffer offset:0 atIndex:1];
        [encoder setBuffer:objectBuffer offset:0 atIndex:2];
        [encoder setBuffer:nodeRangeBuffer offset:0 atIndex:3];
        [encoder setBuffer:schedulerBuffer offset:0 atIndex:4];
        [encoder setBuffer:adaptiveBuffer offset:0 atIndex:5];
    };
    auto dispatchIntegrate=[&](id<MTLComputeCommandEncoder> encoder,
        id<MTLBuffer> acceptedBuffer,id<MTLBuffer> candidateBuffer) {
        encodeIntegrate(encoder);
        [encoder setBuffer:acceptedBuffer offset:0 atIndex:6];
        [encoder setBuffer:candidateBuffer offset:0 atIndex:7];
        [encoder dispatchThreads:MTLSizeMake(1u,1u,1u)
            threadsPerThreadgroup:MTLSizeMake(1u,1u,1u)];
    };
    auto dispatchCommit=[&](id<MTLComputeCommandEncoder> encoder,
        id<MTLBuffer> acceptedBuffer,id<MTLBuffer> candidateBuffer) {
        [encoder setComputePipelineState:commitPipeline];
        [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];
        [encoder setBuffer:statusBuffer offset:0 atIndex:1];
        [encoder setBuffer:acceptedBuffer offset:0 atIndex:2];
        [encoder setBuffer:candidateBuffer offset:0 atIndex:3];
        [encoder setBuffer:materialAcceptedBuffer offset:0 atIndex:4];
        [encoder setBuffer:materialCandidateBuffer offset:0 atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(1u,1u,1u)
            threadsPerThreadgroup:MTLSizeMake(1u,1u,1u)];
    };
    auto runSingle=[&](NMFEMNodeStateGPU initial,NMContinuumObjectGPU oneObject,
        NMMicrostepGPU oneMicrostep) {
        id<MTLBuffer> acceptedBuffer=nil,candidateBuffer=nil;
        initializeNodeBuffers(initial,acceptedBuffer,candidateBuffer);
        std::memcpy(objectBuffer.contents,&oneObject,sizeof(oneObject));
        std::memcpy(microstepBuffer.contents,&oneMicrostep,sizeof(oneMicrostep));
        id<MTLCommandBuffer> command=[queue commandBuffer];
        require(command!=nil,"cannot create FEM integration control command");
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        require(encoder!=nil,"cannot create FEM integration control encoder");
        dispatchIntegrate(encoder,acceptedBuffer,candidateBuffer);
        [encoder endEncoding];
        [command commit]; [command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted,
            "production FEM integration control failed");
        NMFEMNodeStateGPU result{};
        std::memcpy(&result,candidateBuffer.contents,sizeof(result));
        return std::pair<NMFEMNodeStateGPU,NMFEMNodeStateGPU>{initial,result};
    };
    FEMReferenceAccumulationAudit audit;
    constexpr unsigned updateCount=256u;
    const NMFEMNodeStateGPU initial=makeReferenceFixtureNode(true,false);
    id<MTLBuffer> acceptedBuffer=nil,candidateBuffer=nil;
    initializeNodeBuffers(initial,acceptedBuffer,candidateBuffer);
    id<MTLCommandBuffer> command=[queue commandBuffer];
    require(command!=nil,"cannot create repeated FEM integration command");
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    require(encoder!=nil,"cannot create repeated FEM integration encoder");
    for(unsigned step=0u;step<updateCount;++step) {
        dispatchIntegrate(encoder,acceptedBuffer,candidateBuffer);
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        dispatchCommit(encoder,acceptedBuffer,candidateBuffer);
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    require(command.status==MTLCommandBufferStatusCompleted,
        "repeated production FEM integration failed");
    NMFEMNodeStateGPU finalState{};
    std::memcpy(&finalState,acceptedBuffer.contents,sizeof(finalState));
    const double initialDisplacement=static_cast<double>(initial.referenceDisplacementAndMode.x);
    const double expectedIncrement=static_cast<double>(updateCount)*0x1p-41;
    const double finalDisplacement=static_cast<double>(finalState.referenceDisplacementAndMode.x)+
        static_cast<double>(finalState.referenceDisplacementLow.x);
    audit.displacementErrorM=std::abs(finalDisplacement-(initialDisplacement+expectedIncrement));
    audit.accumulatedLowM=finalDisplacement-initialDisplacement;
    audit.expectedIncrementM=expectedIncrement;
    audit.updates=updateCount;
    audit.legacyPositionChangeM=std::abs(
        static_cast<double>(finalState.positionAndMass.x)-initial.positionAndMass.x);
    constexpr unsigned eightStepCount=8u;
    id<MTLBuffer> acceptedEight=nil,candidateEight=nil;
    initializeNodeBuffers(initial,acceptedEight,candidateEight);
    id<MTLCommandBuffer> eightCommand=[queue commandBuffer];
    require(eightCommand!=nil,"cannot create eight-step FEM integration command");
    id<MTLComputeCommandEncoder> eightEncoder=[eightCommand computeCommandEncoder];
    require(eightEncoder!=nil,"cannot create eight-step FEM integration encoder");
    for(unsigned step=0u;step<eightStepCount;++step) {
        dispatchIntegrate(eightEncoder,acceptedEight,candidateEight);
        [eightEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        dispatchCommit(eightEncoder,acceptedEight,candidateEight);
        [eightEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [eightEncoder endEncoding]; [eightCommand commit]; [eightCommand waitUntilCompleted];
    require(eightCommand.status==MTLCommandBufferStatusCompleted,
        "eight-step production FEM integration failed");
    NMFEMNodeStateGPU eightState{};
    std::memcpy(&eightState,acceptedEight.contents,sizeof(eightState));
    const double eightFinalDisplacement=
        static_cast<double>(eightState.referenceDisplacementAndMode.x)+
        static_cast<double>(eightState.referenceDisplacementLow.x);
    audit.expectedEightStepIncrementM=static_cast<double>(eightStepCount)*0x1p-41;
    audit.eightStepIncrementM=eightFinalDisplacement-initialDisplacement;
    audit.eightStepErrorM=std::abs(eightFinalDisplacement-
        (initialDisplacement+audit.expectedEightStepIncrementM));
    audit.eightStepUpdates=eightStepCount;
    auto fixedObject=object;
    auto fixedInitial=makeReferenceFixtureNode(true,true);
    const auto fixedPair=runSingle(fixedInitial,fixedObject,microstep);
    audit.fixedExact=std::memcmp(&fixedPair.first,&fixedPair.second,sizeof(fixedPair.first))==0;
    auto nondueObject=object;
    nondueObject.solver.y=1u;
    auto nondueMicrostep=microstep;
    nondueMicrostep.microtick=1u;
    nondueMicrostep.microtickCount=2u;
    const auto nonduePair=runSingle(initial,nondueObject,nondueMicrostep);
    audit.nondueExact=std::memcmp(&nonduePair.first,&nonduePair.second,sizeof(nonduePair.first))==0;
    auto legacyInitial=makeReferenceFixtureNode(false,false);
    const auto legacyPair=runSingle(legacyInitial,object,microstep);
    audit.legacyExact=std::memcmp(&legacyPair.first,&legacyPair.second,sizeof(legacyPair.first))==0;
    require(audit.displacementErrorM<=2.0e-15&&
        audit.accumulatedLowM>0.99*expectedIncrement&&
        audit.fixedExact&&audit.nondueExact&&audit.legacyExact,
        "production FEM reference-displacement persistence/fixed/nondue/legacy control failed");
    return audit;
}
void appendLittleEndian(std::vector<std::uint8_t>& output,std::uint32_t value) {
    for (unsigned shift=0u; shift<32u; shift+=8u)
        output.push_back(static_cast<std::uint8_t>(value>>shift));
}
int runSurfaceSortOnly(int argc,const char** argv) {
    require(argc==3 || (argc==5 && std::string(argv[3])=="--output"),
        "usage: cardboard-contact-geometry-check --surface-sort-only /absolute/NumiMatter.metallib [--output fresh.bin]");
    const std::string metallibPath=argv[2];
    require(!metallibPath.empty() && metallibPath.front()=='/',
        "surface sort requires an absolute metallib path");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    require(device!=nil,"no Metal device");
    NSError* error=nil;
    NSString* path=[NSString stringWithUTF8String:metallibPath.c_str()];
    id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    require(library!=nil,"cannot load requested surface sort metallib");
    id<MTLFunction> function=[library newFunctionWithName:
        @"numi_matter_metal::nm_contact_sort_surface_primitives"];
    require(function!=nil,"requested metallib is missing nm_contact_sort_surface_primitives");
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline!=nil,"cannot create surface sort pipeline");
    id<MTLCommandQueue> queue=[device newCommandQueue];
    require(queue!=nil,"cannot create surface sort command queue");
    std::vector<SurfaceSortCase> cases;
    cases.push_back(makeSurfaceSortCase(0u,0u,0u,0u,0u,{0u,0u}));
    cases.push_back(makeSurfaceSortCase(1u,17u,13u,0u,0u,{0u,0u}));
    cases.push_back(makeSurfaceSortCase(2u,37u,29u,1u,0u,{0u,0u}));
    cases.push_back(makeSurfaceSortCase(3u,35u,27u,2u,0u,{0u,0u}));
    cases.push_back(makeSurfaceSortCase(4u,31u,23u,1u,1u,{1u,0u}));
    cases.push_back(makeSurfaceSortCase(5u,31u,23u,1u,0u,{1u,1u}));
    cases.push_back(makeSurfaceSortCase(6u,20000u,19200u,3u,0u,{0u,0u}));
    std::vector<std::uint8_t> raw;
    const char magic[8]={'N','M','S','O','R','T','1','\0'};
    raw.insert(raw.end(),magic,magic+8);
    appendLittleEndian(raw,1u);
    appendLittleEndian(raw,static_cast<std::uint32_t>(cases.size()));
    appendLittleEndian(raw,kSortEnvironmentCount);
    appendLittleEndian(raw,0u);
    std::printf("{\"mode\":\"surface_sort\",\"cases\":[");
    for (std::size_t ci=0u; ci<cases.size(); ++ci) {
        const auto& test=cases[ci];
        NMMatterDispatchGPU dispatch{};
        dispatch.abiVersion=NM_MATTER_ABI_VERSION;
        dispatch.flags=NM_MATTER_STATIC_FEM_SURFACE_MAP;
        dispatch.environmentCount=kSortEnvironmentCount;
        dispatch.surfaceFaceCount=test.surfaceFaceCount;
        dispatch.gridNodeCount=test.gridNodeCount;
        const std::size_t total=static_cast<std::size_t>(kSortEnvironmentCount)*test.primitiveCount;
        const std::size_t bytes=total*sizeof(std::uint32_t);
        std::vector<std::uint32_t> sentinels(total,kSortSentinel);
        const std::array<std::uint32_t,kSortEnvironmentCount> sentinelCounts={kSortSentinel,kSortSentinel};
        id<MTLBuffer> primitives=sharedSortBuffer(device,test.primitives.data(),
            test.primitives.size()*sizeof(NMContinuumSurfacePrimitiveGPU));
        id<MTLBuffer> keysA=sharedSortBuffer(device,sentinels.data(),bytes);
        id<MTLBuffer> keysB=sharedSortBuffer(device,sentinels.data(),bytes);
        id<MTLBuffer> indicesA=sharedSortBuffer(device,sentinels.data(),bytes);
        id<MTLBuffer> indicesB=sharedSortBuffer(device,sentinels.data(),bytes);
        id<MTLBuffer> counts=sharedSortBuffer(device,sentinelCounts.data(),sizeof(sentinelCounts));
        id<MTLBuffer> cacheFlags=sharedSortBuffer(device,test.cacheFlags.data(),sizeof(test.cacheFlags));
        id<MTLBuffer> useCache=sharedSortBuffer(device,&test.useCache,sizeof(test.useCache));
        id<MTLCommandBuffer> command=[queue commandBuffer];
        require(command!=nil,"cannot create surface sort command buffer");
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        require(encoder!=nil,"cannot create surface sort encoder");
        [encoder setComputePipelineState:pipeline];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0];
        [encoder setBuffer:primitives offset:0 atIndex:1];
        [encoder setBuffer:keysA offset:0 atIndex:2];
        [encoder setBuffer:keysB offset:0 atIndex:3];
        [encoder setBuffer:indicesA offset:0 atIndex:4];
        [encoder setBuffer:indicesB offset:0 atIndex:5];
        [encoder setBuffer:counts offset:0 atIndex:6];
        [encoder setBuffer:cacheFlags offset:0 atIndex:7];
        [encoder setBuffer:useCache offset:0 atIndex:8];
        [encoder dispatchThreadgroups:MTLSizeMake(kSortEnvironmentCount,1,1)
            threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted,"surface sort GPU command failed");
        require(test.primitives.empty() || std::memcmp(test.primitives.data(),primitives.contents,
            test.primitives.size()*sizeof(NMContinuumSurfacePrimitiveGPU))==0,
            "surface sort modified immutable source primitives");
        const auto expected=surfaceSortReference(test);
        const auto* outKeys=static_cast<const std::uint32_t*>(keysA.contents);
        const auto* outIndices=static_cast<const std::uint32_t*>(indicesA.contents);
        const auto* outCounts=static_cast<const std::uint32_t*>(counts.contents);
        for (std::uint32_t env=0u; env<kSortEnvironmentCount; ++env) {
            const std::size_t base=static_cast<std::size_t>(env)*test.primitiveCount;
            if (test.useCache!=0u && test.cacheFlags[env]!=0u) {
                require(outCounts[env]==kSortSentinel,"cached environment count was unexpectedly overwritten");
                for (std::uint32_t i=0u; i<test.primitiveCount; ++i)
                    require(outKeys[base+i]==kSortSentinel && outIndices[base+i]==kSortSentinel,
                        "cached environment arrays were unexpectedly overwritten");
                continue;
            }
            require(outCounts[env]==expected[env].keys.size(),"surface active count differs from CPU reference");
            require(expected[env].keys.size()<=test.primitiveCount,"CPU active count exceeds primitive capacity");
            for (std::size_t i=0u; i<expected[env].keys.size(); ++i) {
                require(outKeys[base+i]==expected[env].keys[i],"surface Morton key differs from exact lattice reference");
                require(outIndices[base+i]==expected[env].indices[i],"surface permutation differs from stable CPU reference");
                if (i!=0u) {
                    require(outKeys[base+i-1u]<=outKeys[base+i],"surface keys are not nondecreasing");
                    if (outKeys[base+i-1u]==outKeys[base+i])
                        require(outIndices[base+i-1u]<outIndices[base+i],
                            "equal Morton keys lost stable source-index ordering");
                }
            }
            for (std::uint32_t i=static_cast<std::uint32_t>(expected[env].keys.size());
                 i<test.primitiveCount; ++i)
                require(outKeys[base+i]==kSortSentinel && outIndices[base+i]==kSortSentinel,
                    "surface sort wrote beyond active prefix");
        }
        appendLittleEndian(raw,test.id);
        appendLittleEndian(raw,test.primitiveCount);
        appendLittleEndian(raw,test.surfaceFaceCount);
        appendLittleEndian(raw,test.gridNodeCount);
        appendLittleEndian(raw,test.useCache);
        appendLittleEndian(raw,test.cacheFlags[0]);
        appendLittleEndian(raw,test.cacheFlags[1]);
        for (std::uint32_t env=0u; env<kSortEnvironmentCount; ++env) {
            appendLittleEndian(raw,outCounts[env]);
            const std::size_t base=static_cast<std::size_t>(env)*test.primitiveCount;
            for (std::uint32_t i=0u; i<test.primitiveCount; ++i) appendLittleEndian(raw,outKeys[base+i]);
            for (std::uint32_t i=0u; i<test.primitiveCount; ++i) appendLittleEndian(raw,outIndices[base+i]);
        }
        if (ci!=0u) std::printf(",");
        const double gpuMS=std::max(0.0,command.GPUEndTime-command.GPUStartTime)*1000.0;
        std::printf("{\"id\":%u,\"primitive_count\":%u,\"surface_face_count\":%u,"
            "\"grid_node_count\":%u,\"active_env0\":%u,\"active_env1\":%u,\"gpu_ms\":%.6f}",
            test.id,test.primitiveCount,test.surfaceFaceCount,test.gridNodeCount,
            outCounts[0],outCounts[1],gpuMS);
    }
    std::printf("],\"physical_transactions\":0}\n");
    if (argc==5) writeExclusive(argv[4],raw.data(),raw.size());
    return 0;
}


struct ExactVFJVPCase {
    std::array<Vector,4> position{};
    std::array<Vector,4> direction{};
    std::array<double,4> inverseMass={1.0,1.0,1.0,1.0};
    double dt=0.001;
    double thickness=1.0e-4;
    double finiteDifferenceStep=1.0e-4;
};
struct VFResponseOracle {
    std::array<double,3> bary{};
    Vector normal{};
    double gap=0.0;
    double inverseResponse=0.0;
    double stiffness=0.0;
    double impulse=0.0;
    double energy=0.0;
    std::array<Vector,4> residual{};
};
VFResponseOracle vfResponseOracle(const std::array<Vector,4>& position,
    const ExactVFJVPCase& test,double forcedStiffness=0.0) {
    const TriangleOracle witness=triangleOracle(
        position[0],position[1],position[2],position[3]);
    require(witness.bary[0]>0.0&&witness.bary[1]>0.0&&
            witness.bary[2]>0.0&&witness.bary[0]+witness.bary[1]+
            witness.bary[2]<=1.0+1.0e-12,
        "exact VT JVP oracle left the positive interior face");
    const std::array<double,4> weights={
        1.0,-witness.bary[0],-witness.bary[1],-witness.bary[2]};
    VFResponseOracle result;
    result.bary=witness.bary;
    result.gap=witness.separation;
    result.normal=scale(witness.relative,1.0/witness.separation);
    for(unsigned i=0u;i<4u;++i)
        result.inverseResponse+=weights[i]*weights[i]*test.inverseMass[i];
    require(result.inverseResponse>0.0,"exact VT JVP oracle has no inverse response");
    result.stiffness=forcedStiffness>0.0?forcedStiffness:
        10.0/(result.inverseResponse*test.dt*test.dt*
              test.thickness*test.thickness);
    const double distance=std::max(result.gap,1.0e-4*test.thickness);
    const double distance2=distance*distance;
    const double thickness2=test.thickness*test.thickness;
    const double offset=distance2-thickness2;
    const double logarithm=std::log(distance2/thickness2);
    const double firstSquared=-result.stiffness*
        (2.0*offset*logarithm+offset*offset/distance2);
    result.impulse=-2.0*distance*firstSquared*test.dt;
    result.energy=barrierEnergy(result.gap,test.thickness,result.stiffness)*
        triangleMollifier(position[1],position[2],position[3]);
    const double mollifier=triangleMollifier(
        position[1],position[2],position[3]);
    for(unsigned i=0u;i<4u;++i)
        result.residual[i]=scale(result.normal,
            -mollifier*weights[i]*result.impulse);
    return result;
}
std::array<Vector,4> vfPerturbedPosition(
    const std::array<Vector,4>& position,const ExactVFJVPCase& test,
    double epsilon) {
    std::array<Vector,4> result=position;
    for(unsigned i=0u;i<4u;++i)
        result[i]=add(result[i],scale(test.direction[i],
            epsilon*test.dt));
    return result;
}
struct ExactVFJVPResult {
    double relativeErrorMaximum;
    double absoluteErrorMaximum;
    double nearEdgeBaseMinimumBary;
    double nearEdgeSampleMinimumBary;
    double inverseResponseStiffnessDerivativeNs;
};
ExactVFJVPResult runExactVFJVP(
    id<MTLDevice> device,id<MTLLibrary> library,id<MTLCommandQueue> queue) {
    constexpr std::uint32_t environmentCount=2u;
    constexpr std::uint32_t femNodeCount=4u;
    constexpr std::uint32_t slotsPerEnvironment=1u;
    const float dt=0.001f,thickness=1.0e-4f,edgeLength=0.004f;
    const float worldOrigin[3]={8192.0f,-4096.0f,2048.0f};
    const std::array<Vector,4> directions={{
        {0.11,-0.03,0.14},
        {-0.02,0.04,-0.015},
        {0.03,-0.01,0.025},
        {-0.01,0.02,-0.02}}};
    std::array<ExactVFJVPCase,environmentCount> tests{};
    for(unsigned env=0u;env<environmentCount;++env) {
        ExactVFJVPCase& test=tests[env];
        test.dt=dt;test.thickness=thickness;
        test.finiteDifferenceStep=env==0u?1.0e-4:1.0e-5;
        const float baryA=env==0u?0.27f:3.0e-6f;
        const float baryB=env==0u?0.31f:0.41f;
        const float baryC=1.0f-baryA-baryB;
        const float pointX=baryB*edgeLength;
        const float pointY=baryC*edgeLength;
        test.position={{
            {pointX,pointY,0.4*thickness},
            {0.0,0.0,0.0},
            {edgeLength,0.0,0.0},
            {0.0,edgeLength,0.0}}};
        for(unsigned local=0u;local<4u;++local)
            for(unsigned axis=0u;axis<3u;++axis)
                test.direction[local][axis]=static_cast<float>(
                    directions[local][axis]);
    }

    NMMatterDispatchGPU dispatch{};
    dispatch.abiVersion=NM_MATTER_ABI_VERSION;
    dispatch.environmentCount=environmentCount;
    dispatch.objectCount=1u;
    dispatch.gridNodeCount=0u;
    dispatch.femNodeCount=femNodeCount;
    dispatch.maximumRateExponent=0u;
    dispatch.gravityAndTimestep={0.0f,0.0f,0.0f,dt};
    std::array<NMSchedulerStateGPU,environmentCount> schedulers{};
    std::vector<NMFEMNodeStateGPU> candidate(
        environmentCount*femNodeCount);
    std::vector<nm_float4> direction(
        environmentCount*femNodeCount);
    std::vector<NMDeformableContactLimitTraceGPU> traces(environmentCount);
    for(unsigned env=0u;env<environmentCount;++env) {
        const ExactVFJVPCase& test=tests[env];
        const VFResponseOracle initial=vfResponseOracle(test.position,test);
        for(unsigned local=0u;local<femNodeCount;++local) {
            NMFEMNodeStateGPU& state=candidate[env*femNodeCount+local];
            const Vector& u=test.position[local];
            state.restAndFixed={worldOrigin[0],worldOrigin[1],worldOrigin[2],0.0f};
            state.referenceDisplacementAndMode={
                static_cast<float>(u[0]),static_cast<float>(u[1]),
                static_cast<float>(u[2]),1.0f};
            state.positionAndMass={
                worldOrigin[0]+static_cast<float>(u[0]),
                worldOrigin[1]+static_cast<float>(u[1]),
                worldOrigin[2]+static_cast<float>(u[2]),1.0f};
            state.velocityAndInverseMass={0.0f,0.0f,0.0f,
                static_cast<float>(test.inverseMass[local])};
            state.deltaVelocity={0.0f,0.0f,0.0f,0.0f};
            direction[env*femNodeCount+local]={
                static_cast<float>(test.direction[local][0]),
                static_cast<float>(test.direction[local][1]),
                static_cast<float>(test.direction[local][2]),0.0f};
        }
        NMSchedulerStateGPU& scheduler=schedulers[env];
        scheduler.activeExponent=0u;
        scheduler.baseExponent=0u;
        NMDeformableContactLimitTraceGPU& trace=traces[env];
        trace.lineSearch={0.0f,0.0f,0.0f,1.0f};
        trace.result={0u,0u,0u,0u};
        NMDeformableContactGPU& contact=trace.contact;
        contact.nodes={0u,1u,2u,3u};
        contact.identity={0u,0u,NM_DEFORMABLE_CONTACT_VERTEX_TRIANGLE,
                          NM_CONTACT_VALID};
        contact.weights={1.0f,
            static_cast<float>(-initial.bary[0]),
            static_cast<float>(-initial.bary[1]),
            static_cast<float>(-initial.bary[2])};
        contact.timeScales={1.0f,1.0f,1.0f,1.0f};
        contact.normalAndTOI={
            static_cast<float>(initial.normal[0]),
            static_cast<float>(initial.normal[1]),
            static_cast<float>(initial.normal[2]),0.0f};
        const TriangleOracle initialWitness=triangleOracle(
            test.position[0],test.position[1],test.position[2],test.position[3]);
        contact.pointAndSeparation={
            worldOrigin[0]+static_cast<float>(initialWitness.closest[0]),
            worldOrigin[1]+static_cast<float>(initialWitness.closest[1]),
            worldOrigin[2]+static_cast<float>(initialWitness.closest[2]),
            static_cast<float>(initial.gap)};
        contact.velocityAndResponse={0.0f,0.0f,0.0f,
            static_cast<float>(initial.inverseResponse)};
        contact.impulseAndFriction={static_cast<float>(initial.impulse),
                                     0.0f,0.0f,0.0f};
        const double distance=std::max(initial.gap,1.0e-4*test.thickness);
        const double distance2=distance*distance;
        const double h2=test.thickness*test.thickness;
        const double offset=distance2-h2;
        const double logarithm=std::log(distance2/h2);
        const double secondSquared=-initial.stiffness*
            (2.0*logarithm+4.0*offset/distance2-
             offset*offset/(distance2*distance2));
        const double firstSquared=-initial.stiffness*
            (2.0*offset*logarithm+offset*offset/distance2);
        const double radialCurvature=4.0*distance2*secondSquared+
            2.0*firstSquared;
        const double positiveCurvature=std::max(radialCurvature,0.0);
        contact.barrier={static_cast<float>(initial.impulse),
            static_cast<float>(test.dt*test.dt*positiveCurvature),
            static_cast<float>(test.thickness),
            static_cast<float>(initial.stiffness)};
        contact.mollifier={1.0f,
            static_cast<float>(test.dt*initial.energy),
            static_cast<float>(test.dt*test.dt*positiveCurvature),
            static_cast<float>(initial.stiffness)};
        contact.extraNodes={NM_INVALID_INDEX,NM_INVALID_INDEX,
                            NM_INVALID_INDEX,NM_INVALID_INDEX};
    }

    id<MTLFunction> actionFunction=[library newFunctionWithName:
        @"numi_matter_metal::nm_fgmres_capture_exact_deformable_actions"];
    id<MTLFunction> smoothFunction=[library newFunctionWithName:
        @"numi_matter_metal::nm_contact_vf_smooth_guard"];
    require(actionFunction!=nil&&smoothFunction!=nil,
        "geometry metallib lacks the production exact VT JVP or smooth-guard kernel");
    NSError* error=nil;
    id<MTLComputePipelineState> actionPipeline=
        [device newComputePipelineStateWithFunction:actionFunction error:&error];
    require(actionPipeline!=nil,"cannot create production exact VT action pipeline");
    id<MTLComputePipelineState> smoothPipeline=
        [device newComputePipelineStateWithFunction:smoothFunction error:&error];
    require(smoothPipeline!=nil,"cannot create exact VT smooth-guard pipeline");
    id<MTLBuffer> dispatchBuffer=sharedSortBuffer(device,&dispatch,sizeof(dispatch));
    id<MTLBuffer> schedulerBuffer=sharedSortBuffer(device,schedulers.data(),
        sizeof(schedulers));
    id<MTLBuffer> candidateBuffer=sharedSortBuffer(device,candidate.data(),
        candidate.size()*sizeof(NMFEMNodeStateGPU));
    id<MTLBuffer> directionBuffer=sharedSortBuffer(device,direction.data(),
        direction.size()*sizeof(nm_float4));
    id<MTLBuffer> traceBuffer=sharedSortBuffer(device,traces.data(),
        traces.size()*sizeof(NMDeformableContactLimitTraceGPU));
    const std::uint32_t slotsPerEnv=slotsPerEnvironment;
    id<MTLBuffer> slotBuffer=sharedSortBuffer(device,&slotsPerEnv,
        sizeof(slotsPerEnv));
    std::vector<nm_uint4> smoothMasks(environmentCount);
    id<MTLBuffer> smoothBuffer=sharedSortBuffer(device,smoothMasks.data(),
        smoothMasks.size()*sizeof(nm_uint4));

    id<MTLCommandBuffer> actionCommand=[queue commandBuffer];
    require(actionCommand!=nil,"cannot create exact VT action command buffer");
    id<MTLComputeCommandEncoder> actionEncoder=[actionCommand computeCommandEncoder];
    require(actionEncoder!=nil,"cannot create exact VT action encoder");
    [actionEncoder setComputePipelineState:actionPipeline];
    [actionEncoder setBuffer:dispatchBuffer offset:0 atIndex:0];
    [actionEncoder setBuffer:schedulerBuffer offset:0 atIndex:1];
    [actionEncoder setBuffer:candidateBuffer offset:0 atIndex:2];
    [actionEncoder setBuffer:directionBuffer offset:0 atIndex:3];
    [actionEncoder setBuffer:slotBuffer offset:0 atIndex:4];
    [actionEncoder setBuffer:traceBuffer offset:0 atIndex:5];
    [actionEncoder dispatchThreads:MTLSizeMake(environmentCount,1,1)
        threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [actionEncoder endEncoding];
    [actionCommand commit];
    [actionCommand waitUntilCompleted];
    require(actionCommand.status==MTLCommandBufferStatusCompleted,
        "production exact VT action command failed");

    id<MTLCommandBuffer> smoothCommand=[queue commandBuffer];
    require(smoothCommand!=nil,"cannot create exact VT smooth-guard command buffer");
    id<MTLComputeCommandEncoder> smoothEncoder=[smoothCommand computeCommandEncoder];
    require(smoothEncoder!=nil,"cannot create exact VT smooth-guard encoder");
    [smoothEncoder setComputePipelineState:smoothPipeline];
    [smoothEncoder setBuffer:dispatchBuffer offset:0 atIndex:0];
    [smoothEncoder setBuffer:schedulerBuffer offset:0 atIndex:1];
    [smoothEncoder setBuffer:candidateBuffer offset:0 atIndex:2];
    [smoothEncoder setBuffer:directionBuffer offset:0 atIndex:3];
    [smoothEncoder setBuffer:slotBuffer offset:0 atIndex:4];
    [smoothEncoder setBuffer:traceBuffer offset:0 atIndex:5];
    [smoothEncoder setBuffer:smoothBuffer offset:0 atIndex:6];
    [smoothEncoder dispatchThreads:MTLSizeMake(environmentCount,1,1)
        threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [smoothEncoder endEncoding];
    [smoothCommand commit];
    [smoothCommand waitUntilCompleted];
    require(smoothCommand.status==MTLCommandBufferStatusCompleted,
        "exact VT smooth-guard command failed");

    const auto* outputTraces=static_cast<const NMDeformableContactLimitTraceGPU*>(
        traceBuffer.contents);
    const auto* outputSmooth=static_cast<const nm_uint4*>(smoothBuffer.contents);
    double relativeErrorMaximum=0.0,absoluteErrorMaximum=0.0;
    double nearEdgeBaseMinimum=INFINITY,nearEdgeSampleMinimum=INFINITY;
    double inverseResponseStiffnessDerivative=0.0;
    constexpr double smoothMargin=64.0*1.192092896e-7;
    for(unsigned env=0u;env<environmentCount;++env) {
        const ExactVFJVPCase& test=tests[env];
        const VFResponseOracle base=vfResponseOracle(test.position,test);
        const auto plusPosition=vfPerturbedPosition(
            test.position,test,test.finiteDifferenceStep);
        const auto minusPosition=vfPerturbedPosition(
            test.position,test,-test.finiteDifferenceStep);
        const VFResponseOracle plus=vfResponseOracle(plusPosition,test);
        const VFResponseOracle minus=vfResponseOracle(minusPosition,test);
        require(outputTraces[env].productionExactActionSupported.x==1u&&
                outputTraces[env].productionExactActionSupported.y==1u&&
                outputTraces[env].productionExactActionSupported.z==1u&&
                outputTraces[env].productionExactActionSupported.w==1u,
            "production exact VT JVP rejected a smooth or positive near-edge face");
        const nm_uint4 smooth=outputSmooth[env];
        const bool expectedSmooth=env==0u
            ? smooth.x==1u&&smooth.y==1u&&smooth.z==1u&&smooth.w==1u
            : smooth.x==0u&&smooth.y==0u&&smooth.z==0u&&smooth.w==0u;
        require(expectedSmooth,
            "exact VT smooth-feature gate did not accept the interior and reject the sub-margin positive face");
        std::array<Vector,4> fullFD{},fixedKFD{};
        for(unsigned local=0u;local<4u;++local) {
            fullFD[local]=scale(subtract(plus.residual[local],
                minus.residual[local]),1.0/(2.0*test.finiteDifferenceStep));
            const VFResponseOracle plusFixed=vfResponseOracle(
                plusPosition,test,base.stiffness);
            const VFResponseOracle minusFixed=vfResponseOracle(
                minusPosition,test,base.stiffness);
            fixedKFD[local]=scale(subtract(plusFixed.residual[local],
                minusFixed.residual[local]),1.0/(2.0*test.finiteDifferenceStep));
            const nm_float4 gpu=outputTraces[env].productionExactAction[local];
            const Vector actual={gpu.x,gpu.y,gpu.z};
            const double absolute=norm(subtract(actual,fullFD[local]));
            const double relative=absolute/std::max(norm(fullFD[local]),1.0e-3);
            require(std::isfinite(relative)&&norm(fullFD[local])>1.0e-3,
                "exact VT JVP finite-difference oracle produced an uninformative action");
            require(absolute<=std::max(1.0e-2,0.02*norm(fullFD[local])),
                "production exact VT JVP differs from the FP64 residual finite-difference oracle");
            relativeErrorMaximum=std::max(relativeErrorMaximum,relative);
            absoluteErrorMaximum=std::max(absoluteErrorMaximum,absolute);
        }
        if(env==1u) {
            nearEdgeBaseMinimum=*std::min_element(base.bary.begin(),base.bary.end());
            nearEdgeSampleMinimum=*std::min_element(plus.bary.begin(),plus.bary.end());
            nearEdgeSampleMinimum=std::min(nearEdgeSampleMinimum,
                *std::min_element(minus.bary.begin(),minus.bary.end()));
            require(nearEdgeBaseMinimum>0.0&&nearEdgeBaseMinimum<smoothMargin&&
                    nearEdgeSampleMinimum>0.0&&nearEdgeSampleMinimum<smoothMargin,
                "near-edge finite-difference samples crossed or left the intended sub-margin face");
            require(std::abs(plus.bary[0]-minus.bary[0])>1.0e-8,
                "near-edge direction failed to move the closest-point barycentrics tangentially");
            for(unsigned local=0u;local<4u;++local)
                inverseResponseStiffnessDerivative=std::max(
                    inverseResponseStiffnessDerivative,
                    norm(subtract(fullFD[local],fixedKFD[local])));
            require(inverseResponseStiffnessDerivative>1.0e-2,
                "near-edge finite-difference oracle did not exercise inertial stiffness variation through inverse response");
        }
    }
    return {relativeErrorMaximum,absoluteErrorMaximum,
        nearEdgeBaseMinimum,nearEdgeSampleMinimum,
        inverseResponseStiffnessDerivative};
}

}

int main(int argc,const char** argv) {
    @autoreleasepool {
        try {
            if (argc>=2 && std::string(argv[1])=="--surface-sort-only")
                return runSurfaceSortOnly(argc,argv);
            require(argc==1 || (argc==3 && std::string(argv[1])=="--output"),
                "usage: cardboard-contact-geometry-check [--output fresh.bin]");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device!=nil,"no Metal device");
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:
                @NUMI_CARDBOARD_GEOMETRY_METALLIB] error:&error];
            require(library!=nil,"cannot load geometry library");
            id<MTLFunction> function = [library newFunctionWithName:
                @"numi_matter_metal::nm_contact_scaled_geometry_replay"];
            require(function!=nil,"missing geometry kernel");
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            require(pipeline!=nil,"cannot create geometry pipeline");
            id<MTLBuffer> output = [device newBufferWithLength:kBytes options:MTLResourceStorageModeShared];
            require(output!=nil,"geometry arena allocation failed");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue!=nil,"cannot create command queue");
            id<MTLCommandBuffer> command = [queue commandBuffer];
            require(command!=nil,"cannot create command buffer");
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            require(encoder!=nil,"cannot create encoder");
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:output offset:0 atIndex:0];
            [encoder dispatchThreads:MTLSizeMake(kRows,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            require(command.status==MTLCommandBufferStatusCompleted,"geometry GPU command failed");
            const auto* words = static_cast<const float*>(output.contents);
            if (argc==3) writeExclusive(argv[2],words,kBytes);
            double nearestMaximum = 0.0, valueMaximum = 0.0, gradientMaximum = 0.0;
            double eeDistanceMaximum=0.0,eeVectorMaximum=0.0,eeEnergyMaximum=0.0;
            double eeArithmeticVectorMaximum=0.0,eeControlDistanceRelativeMaximum=0.0;
            double baselineOldWitnessError=0.0;
            std::array<double,6> eeOracleDistances{};
            double vfGapMaximum=0.0,vfRelativeMaximum=0.0,vfBaryMaximum=0.0,vfEnergyMaximum=0.0,vfLegacyMaximum=0.0;
            double vfCapturedGapMaximum=0.0,vfCapturedNormalMaximum=0.0,vfCapturedLegacyMaximum=0.0;
            double mappedWeightMaximum=0.0,mappedWitnessMaximum=0.0,mappedDirectionMaximum=0.0,mappedTorqueMaximum=0.0;
            double contactSlopeRelativeMaximum=0.0,contactSlopeAbsoluteMaximum=0.0;
            std::array<double,8> contactSlopeOracle{};
            std::array<bool,5> femPrecisionCases{};
            std::array<bool,5> femExpansionCases{};
            double femExpansionCoordinateError=0.0,femExpansionBaryError=0.0;
            double femExpansionGapError=0.0,femExpansionEnergyError=0.0;
            double femExpansionDeltaEnergyError=0.0,femExpansionGapIncrease=INFINITY;
            double femExpansionTranslationError=0.0,femExpansionLegacyGapChange=0.0;
            double femExpansionLegacyEnergyChange=0.0;
            double femPairCoordinateError=0.0,femPairBaryError=0.0;
            double femPairGapError=0.0,femPairEnergyRelativeError=0.0;
            double femPairDeltaEnergyRelativeError=0.0,femPairResponseRelativeError=0.0;
            double femPairGapIncrease=INFINITY,femPairEnergyChange=0.0;
            double femPairLegacyGapChange=0.0,femPairLegacyEnergyChange=0.0;
            std::array<bool,3> femPairCases{};
            double femReferencePositionError=0.0,femTrialPositionError=0.0;
            double femTrialVelocityError=0.0,femTrialDisplacementError=0.0;
            double femTrialDeltaVelocityError=0.0,femTrialNoOpChanged=0.0;
            double femContactGapError=0.0,femContactEnergyError=0.0;
            double femContactGapIncrease=0.0,femContactEnergyChange=0.0;
            double femRoundedWorldGapChange=0.0,femRoundedWorldEnergyChange=0.0;
            const ExactVFJVPResult exactVFJvp=runExactVFJVP(device,library,queue);
            const FEMReferenceAccumulationAudit femAccumulation=
                runReferenceAccumulationAudit(device,library,queue);
            for (std::size_t i = 0u; i < kRows; ++i) {
                const float* row = words+i*kWordsPerRow;
                for (unsigned j = 0u; j < kWordsPerRow; ++j)
                    require(std::isfinite(row[j]),"nonfinite geometry output");
                if (i < 48u) nearestMaximum = std::max(nearestMaximum,nearestError(row));
                else if (i < 96u) {
                    const auto errors = mollifierError(row);
                    valueMaximum = std::max(valueMaximum,errors[0]);
                    gradientMaximum = std::max(gradientMaximum,errors[1]);
                } else if(i>=120u&&i<125u) {
                    const auto errors=femPrecisionErrors(row);
                    const unsigned test=static_cast<unsigned>(row[0u]);
                    femPrecisionCases[test]=true;
                    femReferencePositionError=std::max(
                        femReferencePositionError,errors.referencePositionErrorM);
                    femTrialPositionError=std::max(
                        femTrialPositionError,errors.trialPositionErrorM);
                    femTrialVelocityError=std::max(
                        femTrialVelocityError,errors.trialVelocityErrorMPerS);
                    femTrialDisplacementError=std::max(
                        femTrialDisplacementError,errors.trialReferenceDisplacementErrorM);
                    femTrialDeltaVelocityError=std::max(
                        femTrialDeltaVelocityError,errors.trialDeltaVelocityErrorMPerS);
                    femTrialNoOpChanged=std::max(
                        femTrialNoOpChanged,errors.trialNoOpChangedComponents);
                    femContactGapError=std::max(
                        femContactGapError,errors.contactGapErrorM);
                    femContactEnergyError=std::max(
                        femContactEnergyError,errors.contactEnergyRelativeError);
                    if(test==4u) {
                        femContactGapIncrease=errors.contactGapIncreaseM;
                        femContactEnergyChange=errors.contactEnergyChangeJ;
                        femRoundedWorldGapChange=errors.roundedWorldGapChangeM;
                        femRoundedWorldEnergyChange=errors.roundedWorldEnergyChangeJ;
                    }
                } else if(i>=130u&&i<133u) {
                    const unsigned test=static_cast<unsigned>(row[0u]);
                    require(test<3u && row[1u]==1.0f && row[2u]==1.0f &&
                        row[3u]==1.0f && row[17u*4u+2u]==1.0f &&
                        row[17u*4u+3u]==1.0f,
                        "compensated FEM pair trial/witness/fixed/zero control flags failed");
                    femPairCases[test]=true;
                    const double scales[3]={0.0078125,0.015625,0.03125};
                    const double scale=scales[test];
                    const double queryU=40.0*std::ldexp(1.0,-20);
                    const double triangleU=33.0*std::ldexp(1.0,-20);
                    const Vector expectedBase[4]={
                        {0.25*scale+queryU,0.25*scale+queryU,queryU},
                        {0.0,0.0,triangleU},{scale,0.0,triangleU},{0.0,scale,triangleU}};
                    std::array<Vector,4> base{},trial{};
                    auto expandedAt=[&](unsigned highSlot,unsigned lowSlot) {
                        return add(vectorAt(row,highSlot),vectorAt(row,lowSlot));
                    };
                    for(unsigned point=0u;point<4u;++point) {
                        base[point]=expandedAt(1u+2u*point,2u+2u*point);
                        trial[point]=base[point];
                        femPairCoordinateError=std::max(femPairCoordinateError,
                            maximumDifference(base[point],expectedBase[point]));
                    }
                    trial[0]=expandedAt(9u,10u);
                    const Vector expectedTrialPoint={expectedBase[0][0]+0x1p-39,
                        expectedBase[0][1],expectedBase[0][2]+0x1p-40};
                    femPairCoordinateError=std::max(femPairCoordinateError,
                        maximumDifference(trial[0],expectedTrialPoint));
                    const Vector trialDisplacement={
                        static_cast<double>(row[15u*4u])+row[16u*4u],
                        static_cast<double>(row[15u*4u+1u])+row[16u*4u+1u],
                        static_cast<double>(row[15u*4u+2u])+row[16u*4u+2u]};
                    const Vector expectedDisplacement={queryU+0x1p-39,
                        queryU,queryU+0x1p-40};
                    femPairCoordinateError=std::max(femPairCoordinateError,
                        maximumDifference(trialDisplacement,expectedDisplacement));
                    require(row[15u*4u+3u]==1.0f&&row[16u*4u+3u]==0.0f,
                        "paired FEM trial changed its reference-mode tag or reserved low limb");
                    const auto baseOracle=triangleOracle(base[0],base[1],base[2],base[3]);
                    const auto trialOracle=triangleOracle(trial[0],trial[1],trial[2],trial[3]);
                    const Vector gpuBaseBary={row[11u*4u],row[11u*4u+1u],row[11u*4u+2u]};
                    const Vector gpuTrialBary={row[12u*4u],row[12u*4u+1u],row[12u*4u+2u]};
                    femPairBaryError=std::max({femPairBaryError,
                        maximumDifference(baseOracle.bary,gpuBaseBary),
                        maximumDifference(trialOracle.bary,gpuTrialBary)});
                    femPairGapError=std::max({femPairGapError,
                        std::abs(baseOracle.separation-row[11u*4u+3u]),
                        std::abs(trialOracle.separation-row[12u*4u+3u]),
                        std::abs(baseOracle.separation-row[13u*4u+2u]),
                        std::abs(trialOracle.separation-row[13u*4u+3u])});
                    constexpr double thickness=10.0e-6,stiffness=1.0e14;
                    const double floorTestGap=row[2u*4u+3u];
                    const double floorThickness=double(float(10.0e-6f));
                    const double floorStiffness=double(float(1.0e14f));
                    const double floorEnergyOracle=barrierEnergy(
                        floorTestGap,floorThickness,floorStiffness);
                    const double floorDistance=std::max(floorTestGap,1.0e-4*floorThickness);
                    const double floorD2=floorDistance*floorDistance;
                    const double floorOffset=floorD2-floorThickness*floorThickness;
                    const double floorDerivative=-2.0*floorDistance*floorStiffness*(
                        2.0*floorOffset*std::log(floorD2/(floorThickness*floorThickness))+
                        floorOffset*floorOffset/floorD2);
                    require(std::abs(double(row[1u*4u+3u])-floorEnergyOracle)/
                                std::max(std::abs(floorEnergyOracle),1.0e-30)<=5.0e-4&&
                            std::abs(double(row[3u*4u+3u])-floorDerivative)/
                                std::max(std::abs(floorDerivative),1.0e-30)<=5.0e-3&&
                            std::abs(double(row[4u*4u+3u])-floorDerivative)/
                                std::max(std::abs(floorDerivative),1.0e-30)<=5.0e-3,
                        "compensated FEM floor energy/derivative control failed");
                    const double mollifier=triangleMollifier(base[1],base[2],base[3]);
                    const double oracleBaseEnergy=mollifier*barrierEnergy(
                        baseOracle.separation,thickness,stiffness);
                    const double oracleTrialEnergy=mollifier*barrierEnergy(
                        trialOracle.separation,thickness,stiffness);
                    const double gpuBaseEnergy=row[13u*4u];
                    const double gpuTrialEnergy=row[13u*4u+1u];
                    femPairEnergyRelativeError=std::max({femPairEnergyRelativeError,
                        std::abs(gpuBaseEnergy-oracleBaseEnergy)/
                            std::max(std::abs(oracleBaseEnergy),1.0e-30),
                        std::abs(gpuTrialEnergy-oracleTrialEnergy)/
                            std::max(std::abs(oracleTrialEnergy),1.0e-30)});
                    const double oracleDelta=oracleTrialEnergy-oracleBaseEnergy;
                    const double gpuDelta=gpuTrialEnergy-gpuBaseEnergy;
                    femPairDeltaEnergyRelativeError=std::max(
                        femPairDeltaEnergyRelativeError,
                        std::abs(gpuDelta-oracleDelta)/std::max(std::abs(oracleDelta),1.0e-30));
                    if(test==0u) {
                        femPairGapIncrease=trialOracle.separation-baseOracle.separation;
                        femPairEnergyChange=gpuDelta;
                        femPairLegacyGapChange=static_cast<double>(row[14u*4u+3u])-row[14u*4u+2u];
                        femPairLegacyEnergyChange=static_cast<double>(row[14u*4u+1u])-row[14u*4u];
                    }
                    const double dt=0x1p-10;
                    auto oracleImpulse=[&](double gap) {
                        const double distance=std::max(gap,1.0e-4*thickness);
                        const double d2=distance*distance,h2=thickness*thickness;
                        const double offset=d2-h2,logRatio=std::log(d2/h2);
                        const double first=-stiffness*(2.0*offset*logRatio+offset*offset/d2);
                        return std::max(-2.0*distance*first,0.0)*dt;
                    };
                    const double expectedBaseImpulse=oracleImpulse(baseOracle.separation);
                    const double expectedTrialImpulse=oracleImpulse(trialOracle.separation);
                    require(expectedBaseImpulse>expectedTrialImpulse&&
                        row[17u*4u]>0.0f&&row[17u*4u+1u]>0.0f,
                        "independent opening response sign or positive GPU response failed");
                    femPairResponseRelativeError=std::max({femPairResponseRelativeError,
                        std::abs(static_cast<double>(row[17u*4u])-expectedBaseImpulse)/
                            std::max(expectedBaseImpulse,1.0e-30),
                        std::abs(static_cast<double>(row[17u*4u+1u])-expectedTrialImpulse)/
                            std::max(expectedTrialImpulse,1.0e-30)});
                    require(row[14u*4u]==row[14u*4u+1u] &&
                        row[14u*4u+2u]==row[14u*4u+3u],
                        "legacy rounded-world projection did not preserve its exact unchanged control");
                } else if(i>=125u&&i<130u) {
                    const auto errors=femExpansionErrors(row);
                    const unsigned test=static_cast<unsigned>(row[0u]);
                    femExpansionCases[test]=true;
                    femExpansionCoordinateError=std::max(
                        femExpansionCoordinateError,errors.coordinateErrorM);
                    femExpansionBaryError=std::max(
                        femExpansionBaryError,errors.barycentricError);
                    femExpansionGapError=std::max(
                        femExpansionGapError,errors.gapErrorM);
                    femExpansionEnergyError=std::max(
                        femExpansionEnergyError,errors.energyRelativeError);
                    femExpansionDeltaEnergyError=std::max(
                        femExpansionDeltaEnergyError,errors.deltaEnergyRelativeError);
                    femExpansionGapIncrease=std::min(
                        femExpansionGapIncrease,errors.gapIncreaseM);
                    femExpansionTranslationError=std::max(
                        femExpansionTranslationError,errors.translationErrorM);
                    femExpansionLegacyGapChange=std::max(
                        femExpansionLegacyGapChange,std::abs(errors.legacyGapChangeM));
                    femExpansionLegacyEnergyChange=std::max(
                        femExpansionLegacyEnergyChange,std::abs(errors.legacyEnergyChangeJ));
                } else if(i>=112u&&i<120u) {
                    const auto errors=contactSlopeErrors(row);
                    const unsigned test=static_cast<unsigned>(row[0u]);
                    contactSlopeOracle[test]=errors.finiteDifference;
                    if(test!=4u&&test!=5u) {
                        contactSlopeRelativeMaximum=std::max(
                            contactSlopeRelativeMaximum,errors.relativeError);
                        contactSlopeAbsoluteMaximum=std::max(
                            contactSlopeAbsoluteMaximum,errors.absoluteError);
                    }
                } else if(i>=102u&&i<109u) {
                    const auto errors=triangleErrors(row);
                    vfGapMaximum=std::max(vfGapMaximum,errors.gap); vfRelativeMaximum=std::max(vfRelativeMaximum,errors.relative);
                    vfBaryMaximum=std::max(vfBaryMaximum,errors.bary); vfEnergyMaximum=std::max(vfEnergyMaximum,errors.energy);
                    vfLegacyMaximum=std::max(vfLegacyMaximum,errors.legacy);
                    if(row[0u]<2.0f) { vfCapturedGapMaximum=std::max(vfCapturedGapMaximum,errors.oracleGap);
                        vfCapturedNormalMaximum=std::max(vfCapturedNormalMaximum,errors.oracleNormal);
                        vfCapturedLegacyMaximum=std::max(vfCapturedLegacyMaximum,errors.legacy); }
                } else if(i>=109u) {
                    const auto errors=mappedWitnessErrors(row);
                    mappedWeightMaximum=std::max(mappedWeightMaximum,errors[0]); mappedWitnessMaximum=std::max(mappedWitnessMaximum,errors[1]);
                    mappedDirectionMaximum=std::max(mappedDirectionMaximum,errors[2]); mappedTorqueMaximum=std::max(mappedTorqueMaximum,errors[3]);
                } else {
                    const auto errors=segmentErrors(row);
                    const auto oracle=segmentOracle(vectorAt(row,1u),vectorAt(row,2u),
                        vectorAt(row,3u),vectorAt(row,4u));
                    eeOracleDistances[i-96u]=oracle.distance;
                    if(i<99u) eeDistanceMaximum=std::max(eeDistanceMaximum,errors[0]);
                    else eeControlDistanceRelativeMaximum=std::max(
                        eeControlDistanceRelativeMaximum,errors[0]/
                            std::max(eeOracleDistances[i-96u],1.0e-30));
                    eeVectorMaximum=std::max(eeVectorMaximum,errors[1]);
                    eeEnergyMaximum=std::max(eeEnergyMaximum,errors[3]);
                    eeArithmeticVectorMaximum=std::max(
                        eeArithmeticVectorMaximum,errors[4]);
                    if(i==96u) baselineOldWitnessError=errors[2];
                }
            }
            require(std::all_of(femPrecisionCases.begin(),femPrecisionCases.end(),
                    [](bool value){return value;}),
                "FEM relative-position fixture did not visit every declared row");
            require(std::all_of(femExpansionCases.begin(),femExpansionCases.end(),
                    [](bool value){return value;}),
                "compensated FEM expansion fixture did not visit all five rows");
            require(std::all_of(femPairCases.begin(),femPairCases.end(),
                    [](bool value){return value;}),
                "paired FEM displacement fixture did not visit all three triangle scales");
            std::fprintf(stderr,
                "paired_displacement_diagnostic coordinate_m=%.17g bary=%.17g gap_m=%.17g energy_relative=%.17g delta_energy_relative=%.17g gap_increase_m=%.17g delta_energy_J=%.17g response_relative=%.17g legacy_gap_m=%.17g legacy_energy_J=%.17g\n",
                femPairCoordinateError,femPairBaryError,femPairGapError,
                femPairEnergyRelativeError,femPairDeltaEnergyRelativeError,
                femPairGapIncrease,femPairEnergyChange,femPairResponseRelativeError,
                femPairLegacyGapChange,femPairLegacyEnergyChange);
            require(femPairCoordinateError<=2.0e-15&&femPairBaryError<=5.0e-7&&
                    femPairGapError<=1.0e-11&&femPairEnergyRelativeError<=5.0e-4&&
                    femPairDeltaEnergyRelativeError<=0.25&&femPairGapIncrease>0.0&&
                    femPairEnergyChange<0.0&&femPairResponseRelativeError<=5.0e-4&&
                    femPairLegacyGapChange==0.0&&femPairLegacyEnergyChange==0.0,
                "paired FEM expansion geometry/energy/response oracle exceeded declared gates");
            require(femAccumulation.updates==256u&&
                    femAccumulation.eightStepUpdates==8u&&
                    femAccumulation.eightStepErrorM<=2.0e-15&&
                    femAccumulation.eightStepIncrementM>
                        0.99*femAccumulation.expectedEightStepIncrementM&&
                    femAccumulation.displacementErrorM<=2.0e-15&&
                    femAccumulation.accumulatedLowM>0.99*femAccumulation.expectedIncrementM&&
                    femAccumulation.fixedExact&&femAccumulation.nondueExact&&
                    femAccumulation.legacyExact,
                "production FEM paired displacement persistence controls failed");
            require(femExpansionCoordinateError<=2.0e-15&&
                    femExpansionBaryError<=5.0e-7&&femExpansionGapError<=1.0e-11&&
                    femExpansionEnergyError<=5.0e-4&&
                    femExpansionDeltaEnergyError<=0.25&&
                    femExpansionGapIncrease>1.0e-11&&
                    femExpansionTranslationError<=2.0e-15&&
                    femExpansionLegacyGapChange==0.0&&
                    femExpansionLegacyEnergyChange==0.0,
                "compensated FEM expansion or common-translation regression exceeded its declared FP64 gate");
            require(exactVFJvp.relativeErrorMaximum<=0.02&&
                    exactVFJvp.absoluteErrorMaximum<=0.25&&
                    exactVFJvp.nearEdgeBaseMinimumBary>0.0&&
                    exactVFJvp.nearEdgeSampleMinimumBary>0.0&&
                    exactVFJvp.inverseResponseStiffnessDerivativeNs>1.0e-2,
                "production exact VT JVP or sub-margin face control exceeded its independent FP64 gate");
            require(contactSlopeOracle[0]<0.0&&contactSlopeOracle[1]>0.0&&
                    contactSlopeOracle[6]<0.0&&contactSlopeOracle[7]<0.0,
                "contact-slope cases do not cover the expected signed slopes");
            require(contactSlopeRelativeMaximum<=5.0e-3 ||
                    contactSlopeAbsoluteMaximum<=2.0e-5,
                "contact-slope analytical regression exceeded its independent FP64 gate");
            require(baselineOldWitnessError>5.0e-9,
                "legacy rounded-world-point control did not reproduce the witness error");
            require(std::abs((eeOracleDistances[1]-eeOracleDistances[0])-8.754313078e-9)<=2.0e-10 &&
                    std::abs((eeOracleDistances[2]-eeOracleDistances[0])-0.352480818e-9)<=2.0e-10,
                "captured EE trial gaps do not reproduce the independent FP64 increase");
            require(vfCapturedLegacyMaximum>1.0e-9,"captured legacy rounded-world-point VF control did not reproduce the measured nanometer-scale witness error");
            require(femReferencePositionError<=2.0e-7&&
                    femTrialPositionError<=2.0e-7&&
                    femTrialVelocityError<=2.0e-8&&
                    femTrialDisplacementError<=2.0e-8&&
                    femTrialDeltaVelocityError<=2.0e-8&&
                    femTrialNoOpChanged==0.0&&femContactGapError<=2.0e-7&&
                    femContactEnergyError<=5.0e-4&&
                    femContactGapIncrease>1.0e-6&&femContactEnergyChange<0.0&&
                    std::abs(femRoundedWorldGapChange)<=5.0e-9,
                "FEM relative-position, exact trial, or synthetic contact gate failed");
            std::printf("{\"nearest_cases\":48,\"mollifier_cases\":48,\"ee_witness_cases\":6,\"vf_witness_cases\":7,\"mapped_vf_cases\":3,\"contact_slope_cases\":8,\"nearest_error\":%.17g,\"mollifier_value_error\":%.17g,\"mollifier_gradient_error\":%.17g,\"ee_witness_gap_error_max_m\":%.17g,\"ee_nearest_vector_deviation_max_m\":%.17g,\"ee_relative_arithmetic_error_max_m\":%.17g,\"ee_control_gap_relative_error_max\":%.17g,\"ee_energy_relative_error\":%.17g,\"vf_witness_gap_error_max_m\":%.17g,\"vf_relative_vector_error_max_m\":%.17g,\"vf_barycentric_error_max\":%.17g,\"vf_energy_relative_error_max\":%.17g,\"vf_legacy_control_vector_error_max_m\":%.17g,\"vf_captured_gap_error_max_m\":%.17g,\"vf_captured_normal_error_max\":%.17g,\"mapped_vf_weight_sum_error_max\":%.17g,\"mapped_vf_relative_error_max_m\":%.17g,\"mapped_vf_direction_error_max_m\":%.17g,\"mapped_vf_torque_max_Nm_per_unit_force\":%.17g,\"legacy_ee_base_world_delta_error_m\":%.17g,\"contact_slope_fp64_relative_error_max\":%.17g,\"contact_slope_fp64_absolute_error_max_J\":%.17g,\"contact_slope_oracle_world_offset_analytic_J\":%.17g,\"contact_slope_oracle_rounded_control_J\":%.17g,\"fem_reference_position_cases\":3,\"fem_trial_state_cases\":3,\"fem_contact_trial_cases\":1,\"fem_reference_position_error_max_m\":%.17g,\"fem_trial_position_error_max_m\":%.17g,\"fem_trial_velocity_error_max_m_per_s\":%.17g,\"fem_trial_reference_displacement_error_max_m\":%.17g,\"fem_trial_delta_velocity_error_max_m_per_s\":%.17g,\"fem_trial_noop_changed_components\":%.17g,\"fem_contact_gap_error_max_m\":%.17g,\"fem_contact_energy_relative_error_max\":%.17g,\"fem_contact_trial_gap_increase_m\":%.17g,\"fem_contact_trial_energy_change_J\":%.17g,\"fem_contact_rounded_world_gap_change_m\":%.17g,\"fem_contact_rounded_world_energy_change_J\":%.17g,\"fem_compensated_pair_cases\":3,\"fem_compensated_pair_coordinate_error_max_m\":%.17g,\"fem_compensated_pair_barycentric_error_max\":%.17g,\"fem_compensated_pair_gap_error_max_m\":%.17g,\"fem_compensated_pair_energy_relative_error_max\":%.17g,\"fem_compensated_pair_delta_energy_relative_error_max\":%.17g,\"fem_compensated_pair_response_relative_error_max\":%.17g,\"fem_compensated_pair_gap_increase_m\":%.17g,\"fem_compensated_pair_energy_change_J\":%.17g,\"fem_compensated_pair_legacy_gap_change_m\":%.17g,\"fem_compensated_pair_legacy_energy_change_J\":%.17g,\"fem_reference_accumulation_updates\":%u,\"fem_reference_accumulation_error_m\":%.17g,\"fem_reference_accumulation_expected_increment_m\":%.17g,\"fem_reference_accumulation_actual_increment_m\":%.17g,\"fem_reference_accumulation_world_projection_change_m\":%.17g,\"fem_reference_accumulation_after8_error_m\":%.17g,\"fem_reference_accumulation_after8_increment_m\":%.17g,\"fem_reference_accumulation_after8_expected_increment_m\":%.17g,\"fem_reference_fixed_exact\":%s,\"fem_reference_nondue_exact\":%s,\"fem_reference_legacy_exact\":%s,\"fem_compensated_expansion_cases\":5,\"fem_compensated_coordinate_error_max_m\":%.17g,\"fem_compensated_barycentric_error_max\":%.17g,\"fem_compensated_gap_error_max_m\":%.17g,\"fem_compensated_energy_relative_error_max\":%.17g,\"fem_compensated_delta_energy_relative_error_max\":%.17g,\"fem_compensated_gap_increase_min_m\":%.17g,\"fem_compensated_translation_error_max_m\":%.17g,\"fem_rounded_legacy_gap_change_max_m\":%.17g,\"fem_rounded_legacy_energy_change_max_J\":%.17g,\"exact_vf_jvp_cases\":2,\"exact_vf_jvp_relative_error_max\":%.17g,\"exact_vf_jvp_absolute_error_max_Ns_per_alpha\":%.17g,\"exact_vf_jvp_near_edge_base_bary_min\":%.17g,\"exact_vf_jvp_near_edge_fd_bary_min\":%.17g,\"exact_vf_jvp_inverse_response_k_effect_Ns_per_alpha\":%.17g,\"physical_transactions\":0}\n",nearestMaximum,valueMaximum,gradientMaximum,eeDistanceMaximum,eeVectorMaximum,eeArithmeticVectorMaximum,eeControlDistanceRelativeMaximum,eeEnergyMaximum,vfGapMaximum,vfRelativeMaximum,vfBaryMaximum,vfEnergyMaximum,vfLegacyMaximum,vfCapturedGapMaximum,vfCapturedNormalMaximum,mappedWeightMaximum,mappedWitnessMaximum,mappedDirectionMaximum,mappedTorqueMaximum,baselineOldWitnessError,contactSlopeRelativeMaximum,contactSlopeAbsoluteMaximum,contactSlopeOracle[6],contactSlopeOracle[7],femReferencePositionError,femTrialPositionError,femTrialVelocityError,femTrialDisplacementError,femTrialDeltaVelocityError,femTrialNoOpChanged,femContactGapError,femContactEnergyError,femContactGapIncrease,femContactEnergyChange,femRoundedWorldGapChange,femRoundedWorldEnergyChange,femPairCoordinateError,femPairBaryError,femPairGapError,femPairEnergyRelativeError,femPairDeltaEnergyRelativeError,femPairResponseRelativeError,femPairGapIncrease,femPairEnergyChange,femPairLegacyGapChange,femPairLegacyEnergyChange,femAccumulation.updates,femAccumulation.displacementErrorM,femAccumulation.expectedIncrementM,femAccumulation.accumulatedLowM,femAccumulation.legacyPositionChangeM,femAccumulation.eightStepErrorM,femAccumulation.eightStepIncrementM,femAccumulation.expectedEightStepIncrementM,femAccumulation.fixedExact?"true":"false",femAccumulation.nondueExact?"true":"false",femAccumulation.legacyExact?"true":"false",femExpansionCoordinateError,femExpansionBaryError,femExpansionGapError,femExpansionEnergyError,femExpansionDeltaEnergyError,femExpansionGapIncrease,femExpansionTranslationError,femExpansionLegacyGapChange,femExpansionLegacyEnergyChange,exactVFJvp.relativeErrorMaximum,exactVFJvp.absoluteErrorMaximum,exactVFJvp.nearEdgeBaseMinimumBary,exactVFJvp.nearEdgeSampleMinimumBary,exactVFJvp.inverseResponseStiffnessDerivativeNs);
            require(nearestMaximum <= 1.0e-3 && valueMaximum <= 1.0e-3 && gradientMaximum <= 1.0e-3 &&
                    eeDistanceMaximum <= 2.0e-10 && eeVectorMaximum <= 2.0e-9 &&
                    eeArithmeticVectorMaximum <= 2.0e-10 &&
                    eeControlDistanceRelativeMaximum <= 5.0e-8 && eeEnergyMaximum <= 2.0e-3 &&
                    vfGapMaximum<=1.0e-6 && vfRelativeMaximum<=1.0e-6 && vfBaryMaximum<=3.0e-5 && vfEnergyMaximum<=3.0e-3 &&
                    mappedWeightMaximum<=3.0e-6 && mappedWitnessMaximum<=2.0e-6 && mappedDirectionMaximum<=2.0e-6 && mappedTorqueMaximum<=2.0e-6,
                "native contact geometry exceeded the analytical gate");
            return 0;
        } catch (const std::exception& ex) {
            std::fprintf(stderr,"cardboard contact geometry check failed: %s\n",ex.what());
            return 2;
        }
    }
}

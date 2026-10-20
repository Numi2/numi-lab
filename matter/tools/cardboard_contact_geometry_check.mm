#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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
constexpr std::size_t kRows = 112u;
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
}

int main(int argc,const char** argv) {
    @autoreleasepool {
        try {
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
            for (std::size_t i = 0u; i < kRows; ++i) {
                const float* row = words+i*kWordsPerRow;
                for (unsigned j = 0u; j < kWordsPerRow; ++j)
                    require(std::isfinite(row[j]),"nonfinite geometry output");
                if (i < 48u) nearestMaximum = std::max(nearestMaximum,nearestError(row));
                else if (i < 96u) {
                    const auto errors = mollifierError(row);
                    valueMaximum = std::max(valueMaximum,errors[0]);
                    gradientMaximum = std::max(gradientMaximum,errors[1]);
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
            require(baselineOldWitnessError>5.0e-9,
                "legacy rounded-world-point control did not reproduce the witness error");
            require(std::abs((eeOracleDistances[1]-eeOracleDistances[0])-8.754313078e-9)<=2.0e-10 &&
                    std::abs((eeOracleDistances[2]-eeOracleDistances[0])-0.352480818e-9)<=2.0e-10,
                "captured EE trial gaps do not reproduce the independent FP64 increase");
            require(vfCapturedLegacyMaximum>1.0e-9,"captured legacy rounded-world-point VF control did not reproduce the measured nanometer-scale witness error");
            std::printf("{\"nearest_cases\":48,\"mollifier_cases\":48,\"ee_witness_cases\":6,\"vf_witness_cases\":7,\"mapped_vf_cases\":3,\"nearest_error\":%.17g,\"mollifier_value_error\":%.17g,\"mollifier_gradient_error\":%.17g,\"ee_witness_gap_error_max_m\":%.17g,\"ee_nearest_vector_deviation_max_m\":%.17g,\"ee_relative_arithmetic_error_max_m\":%.17g,\"ee_control_gap_relative_error_max\":%.17g,\"ee_energy_relative_error\":%.17g,\"vf_witness_gap_error_max_m\":%.17g,\"vf_relative_vector_error_max_m\":%.17g,\"vf_barycentric_error_max\":%.17g,\"vf_energy_relative_error_max\":%.17g,\"vf_legacy_control_vector_error_max_m\":%.17g,\"vf_captured_gap_error_max_m\":%.17g,\"vf_captured_normal_error_max\":%.17g,\"mapped_vf_weight_sum_error_max\":%.17g,\"mapped_vf_relative_error_max_m\":%.17g,\"mapped_vf_direction_error_max_m\":%.17g,\"mapped_vf_torque_max_Nm_per_unit_force\":%.17g,\"legacy_ee_base_world_delta_error_m\":%.17g,\"physical_transactions\":0}\n",nearestMaximum,valueMaximum,gradientMaximum,eeDistanceMaximum,eeVectorMaximum,eeArithmeticVectorMaximum,eeControlDistanceRelativeMaximum,eeEnergyMaximum,vfGapMaximum,vfRelativeMaximum,vfBaryMaximum,vfEnergyMaximum,vfLegacyMaximum,vfCapturedGapMaximum,vfCapturedNormalMaximum,mappedWeightMaximum,mappedWitnessMaximum,mappedDirectionMaximum,mappedTorqueMaximum,baselineOldWitnessError);
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

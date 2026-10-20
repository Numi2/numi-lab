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
constexpr std::size_t kRows = 102u;
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
            for (std::size_t i = 0u; i < kRows; ++i) {
                const float* row = words+i*kWordsPerRow;
                for (unsigned j = 0u; j < kWordsPerRow; ++j)
                    require(std::isfinite(row[j]),"nonfinite geometry output");
                if (i < 48u) nearestMaximum = std::max(nearestMaximum,nearestError(row));
                else if (i < 96u) {
                    const auto errors = mollifierError(row);
                    valueMaximum = std::max(valueMaximum,errors[0]);
                    gradientMaximum = std::max(gradientMaximum,errors[1]);
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
            std::printf("{\"nearest_cases\":48,\"mollifier_cases\":48,\"ee_witness_cases\":6,\"nearest_error\":%.17g,\"mollifier_value_error\":%.17g,\"mollifier_gradient_error\":%.17g,\"ee_witness_gap_error_max_m\":%.17g,\"ee_nearest_vector_deviation_max_m\":%.17g,\"ee_relative_arithmetic_error_max_m\":%.17g,\"ee_control_gap_relative_error_max\":%.17g,\"ee_energy_relative_error\":%.17g,\"legacy_base_world_delta_error_m\":%.17g,\"physical_transactions\":0}\n",nearestMaximum,valueMaximum,gradientMaximum,eeDistanceMaximum,eeVectorMaximum,eeArithmeticVectorMaximum,eeControlDistanceRelativeMaximum,eeEnergyMaximum,baselineOldWitnessError);
            require(nearestMaximum <= 1.0e-3 && valueMaximum <= 1.0e-3 && gradientMaximum <= 1.0e-3 &&
                    eeDistanceMaximum <= 2.0e-10 && eeVectorMaximum <= 2.0e-9 &&
                    eeArithmeticVectorMaximum <= 2.0e-10 &&
                    eeControlDistanceRelativeMaximum <= 5.0e-8 && eeEnergyMaximum <= 2.0e-3,
                "native contact geometry exceeded the analytical gate");
            return 0;
        } catch (const std::exception& ex) {
            std::fprintf(stderr,"cardboard contact geometry check failed: %s\n",ex.what());
            return 2;
        }
    }
}

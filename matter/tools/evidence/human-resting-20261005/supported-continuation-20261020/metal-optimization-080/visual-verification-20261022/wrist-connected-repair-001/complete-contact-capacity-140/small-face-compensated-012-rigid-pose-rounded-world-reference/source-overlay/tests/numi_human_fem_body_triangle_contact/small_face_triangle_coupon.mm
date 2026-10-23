#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "support/fem_body_triangle_contact_math.hpp"
#include "support/small_triangles_107.hpp"
#include "metalrobo/engine_types.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <limits>
#include <cstdlib>
#include <string_view>
#include <vector>

using namespace numi::matter::fem_body_triangle_contact;

namespace {
void require(bool value,std::string_view message) {
    if (!value) {
        std::fprintf(stderr,"FAIL: %.*s\n",int(message.size()),message.data());
        std::exit(1);
    }
}
nm_float4 f4(float x,float y,float z,float w) { return {x,y,z,w}; }
double component(nm_float4 v,std::size_t i) {
    return i==0u?v.x:(i==1u?v.y:v.z);
}
std::uint32_t orderedFloatBits(float value) {
    const auto bits=std::bit_cast<std::uint32_t>(value);
    return (bits&0x80000000u)?~bits:(bits|0x80000000u);
}
std::uint32_t floatUlpDistance(float a,float b) {
    const auto x=orderedFloatBits(a),y=orderedFloatBits(b);
    return x>y?x-y:y-x;
}
Vec3 rotate(const std::array<double,4>& q,Vec3 v) {
    return quaternionRotate(q,v);
}
struct FVec3 { float x,y,z; };
FVec3 rotateFloat(const std::array<float,4>& q,FVec3 v) {
    const float cx=q[1]*v.z-q[2]*v.y;
    const float cy=q[2]*v.x-q[0]*v.z;
    const float cz=q[0]*v.y-q[1]*v.x;
    const float dx=2.0f*cx,dy=2.0f*cy,dz=2.0f*cz;
    return {v.x+q[3]*dx+(q[1]*dz-q[2]*dy),
            v.y+q[3]*dy+(q[2]*dx-q[0]*dz),
            v.z+q[3]*dz+(q[0]*dy-q[1]*dx)};
}
struct Case {
    NMNumiHumanFEMBodyContactSampleGPU sample{};
    NMFEMNodeStateGPU node{};
    MRArticulatedBodyPoseGPU pose{};
    Evaluation expected{};
    Vec3 query{};
    Vec3 bodyOrigin{};
    double edgeScale=0.0;
    double pointRoundoffM=0.0;
    double queryRoundoffM=0.0;
    std::uint32_t sourceFace=0u;
    std::uint32_t caseKind=0u;
    std::uint32_t transformIndex=0u;
};
}

int main(int argc,const char* argv[]) {
    @autoreleasepool {
        require(argc==2,"expected isolated metallib path");
        NSError* error=nil;
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        require(device!=nil,"Metal device");
        NSURL* url=[NSURL fileURLWithPath:@(argv[1])];
        id<MTLLibrary> library=[device newLibraryWithURL:url error:&error];
        if (!library) {
            std::fprintf(stderr,"Metal library: %s\n",error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLFunction> function=[library
            newFunctionWithName:@"nm_numi_human_fem_body_triangle_small_face_coupon"];
        require(function!=nil,"small-face Metal kernel");
        id<MTLComputePipelineState> pipeline=
            [device newComputePipelineStateWithFunction:function error:&error];
        require(pipeline!=nil,"small-face Metal pipeline");

        constexpr double area=0.25,k=1000.0,strain=1.0e-6;
        constexpr std::size_t faceCount=
            test_data::kRegisteredSmallTriangles.size();
        constexpr std::size_t casesPerFace=3u;
        constexpr std::size_t transformCount=5u;
        constexpr std::size_t count=faceCount*casesPerFace*transformCount;
        constexpr std::size_t rejectedCount=2u;
        constexpr std::size_t dispatchCount=count+rejectedCount;
        std::array<Case,count> cases{};
        // Exercise identity, realistic translations, oblique rotation, an
        // axis rotation, and the recorded supine root transform. The CPU
        // reference is built from the same rounded world triangles/queries
        // that the Float32 Metal owner actually receives.
        struct Transform { std::array<float,4> q; std::array<float,3> t; };
        const std::array<Transform,transformCount> transforms{{
            {{{0.0f,0.0f,0.0f,1.0f}},{{0.0f,0.0f,0.0f}}},
            {{{0.0f,0.0f,0.0f,1.0f}},{{0.375f,-0.25f,1.125f}}},
            {{{1.0f,0.0f,0.0f,0.0f}},{{0.35f,0.0f,-0.25f}}},
            {{{0.0f,0.382683426f,0.0f,0.923879504f}},
              {{-0.625f,0.3125f,0.875f}}},
            {{{-0.703164279f,-0.0014300436f,-0.0014054211f,0.711024582f}},
              {{-0.05f,0.2f,0.098855719f}}}}};
        std::size_t row=0u;
        for (const auto& transform:transforms) {
          const std::array<double,4> q{{transform.q[0],transform.q[1],
              transform.q[2],transform.q[3]}};
          const Vec3 translation{transform.t[0],transform.t[1],transform.t[2]};
          for (const auto& face:test_data::kRegisteredSmallTriangles) {
            std::array<Vec3,3> tri{};
            for (std::size_t v=0;v<3u;++v)
                tri[v]={double(face.points[v][0]),double(face.points[v][1]),
                        double(face.points[v][2])};
            Vec3 n{};
            require(scaledFaceNormal(tri[0],tri[1],tri[2],n),
                    "fixture triangle has nonzero scaled area");
            const double scale=std::max({norm(sub(tri[1],tri[0])),
                norm(sub(tri[2],tri[0])),norm(sub(tri[2],tri[1]))});
            const double h=std::max(1.0e-6,0.02*scale);
            const Vec3 center=mul(add(add(tri[0],tri[1]),tri[2]),1.0/3.0);
            const Vec3 edgeMid=mul(add(tri[0],tri[1]),0.5);
            const Vec3 edgeOutside=add(edgeMid,mul(sub(edgeMid,center),0.5));
            const Vec3 vertexDirection=add(
                unit(sub(tri[1],tri[0])),unit(sub(tri[2],tri[0])));
            const std::array<Vec3,3> queries{{
                sub(center,mul(n,h)),
                sub(edgeOutside,mul(n,h)),
                sub(add(tri[0],mul(vertexDirection,-0.1*scale)),mul(n,h))}};
            for (std::uint32_t kind=0;kind<casesPerFace;++kind,++row) {
                // Match the exact Float32 positions the GPU receives.
                const Vec3 query{double(float(queries[kind].x)),
                    double(float(queries[kind].y)),double(float(queries[kind].z))};
                auto& c=cases[row];
                c.edgeScale=scale;c.sourceFace=face.sourceFaceId;
                c.caseKind=kind;
                c.transformIndex=static_cast<std::uint32_t>(&transform-transforms.data());
                c.sample=makeSample(static_cast<std::uint32_t>(row),1u,
                    tri[0],tri[1],tri[2],float(area),float(k),float(strain));
                const FVec3 localQueryF{float(query.x),float(query.y),float(query.z)};
                const FVec3 rotatedQueryF=rotateFloat(transform.q,localQueryF);
                const FVec3 worldQueryF{
                    rotatedQueryF.x+transform.t[0],
                    rotatedQueryF.y+transform.t[1],
                    rotatedQueryF.z+transform.t[2]};
                c.query={double(worldQueryF.x),double(worldQueryF.y),
                         double(worldQueryF.z)};
                c.bodyOrigin=translation;
                c.node.positionAndMass=f4(float(c.query.x),float(c.query.y),
                                           float(c.query.z),1.0f);
                c.node.velocityAndInverseMass=f4(0,0,0,1);
                c.node.restAndFixed=f4(0,0,0,0);
                c.pose.position={transform.t[0],transform.t[1],transform.t[2],0};
                c.pose.orientation={transform.q[0],transform.q[1],
                                    transform.q[2],transform.q[3]};
                const Vec3 idealQuery=add(translation,rotate(q,queries[kind]));
                c.queryRoundoffM=norm(sub(c.query,idealQuery));
                std::array<Vec3,3> worldTriangle{};
                for (std::size_t v=0;v<3u;++v) {
                    const FVec3 localVertex{float(tri[v].x),float(tri[v].y),
                                            float(tri[v].z)};
                    const FVec3 rotatedVertex=rotateFloat(transform.q,localVertex);
                    worldTriangle[v]={
                        double(rotatedVertex.x+transform.t[0]),
                        double(rotatedVertex.y+transform.t[1]),
                        double(rotatedVertex.z+transform.t[2])};
                }
                c.expected=evaluate(c.query,translation,worldTriangle,
                                    area,k,strain);
                require(c.expected.valid && c.expected.active &&
                    c.expected.closure>0.0,
                    "CPU double reference accepts rounded world tiny-face probes");
                const Vec3 idealPoint=c.expected.closestPoint;
                const Vec3 roundedPoint{double(float(idealPoint.x)),
                    double(float(idealPoint.y)),double(float(idealPoint.z))};
                c.pointRoundoffM=norm(sub(roundedPoint,idealPoint));
                c.expected.closestPoint=roundedPoint;
            }
          }
        }
        require(row==count,"all 1605 rounded-world rigid-pose cases constructed");
        // Case has interleaved state; provide the packed owner arrays separately.
        std::array<NMNumiHumanFEMBodyContactSampleGPU,dispatchCount> samples{};
        std::array<NMFEMNodeStateGPU,dispatchCount> nodes{};
        std::array<MRArticulatedBodyPoseGPU,dispatchCount> poses{};
        std::array<nm_float4,dispatchCount*8u> output{};
        for (std::size_t i=0;i<count;++i) {
            samples[i]=cases[i].sample;nodes[i]=cases[i].node;poses[i]=cases[i].pose;
        }
        // A truly collinear triangle and a nonfinite triangle must fail closed.
        samples[count]=samples[0];
        samples[count].bodyLocalPointAndArea={0,0,0,0.25f};
        samples[count].bodyLocalNormalAndReferenceSeparation={1,0,0,1000};
        samples[count].stiffnessAndNormalStrainPerPressure={2,0,0,strain};
        nodes[count]=nodes[0];poses[count]=poses[0];
        samples[count+1u]=samples[0];
        samples[count+1u].bodyLocalPointAndArea.x=
            std::numeric_limits<float>::quiet_NaN();
        nodes[count+1u]=nodes[0];poses[count+1u]=poses[0];
        auto makeBuffer=[&](const void* data,NSUInteger bytes) {
            return [device newBufferWithBytes:data length:bytes
                options:MTLResourceStorageModeShared];
        };
        auto samplesBuffer=makeBuffer(samples.data(),sizeof(samples));
        auto nodesBuffer=makeBuffer(nodes.data(),sizeof(nodes));
        auto posesBuffer=makeBuffer(poses.data(),sizeof(poses));
        auto outputBuffer=[device newBufferWithLength:sizeof(output)
            options:MTLResourceStorageModeShared];
        require(samplesBuffer&&nodesBuffer&&posesBuffer&&outputBuffer,
                "small-face Metal buffers");
        id<MTLCommandQueue> queue=[device newCommandQueue];
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:samplesBuffer offset:0 atIndex:0];
        [encoder setBuffer:nodesBuffer offset:0 atIndex:1];
        [encoder setBuffer:posesBuffer offset:0 atIndex:2];
        [encoder setBuffer:outputBuffer offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(dispatchCount,1,1)
            threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted,
                "small-face command completes");
        const auto* got=static_cast<const nm_float4*>(outputBuffer.contents);
        double maxForceRel=0.0,maxNormalErr=0.0,maxPointRel=0.0;
        double maxPointErrorM=0.0,maxPointRoundoffM=0.0;
        double maxQueryRoundoffM=0.0,maxTangentRel=0.0;
        std::uint32_t maxPointUlp=0u;
        std::array<double,transformCount> maxPointByTransform{};
        std::array<double,transformCount> maxPointRelByTransform{};
        std::array<std::uint32_t,transformCount> maxPointUlpByTransform{};
        for (std::size_t i=0;i<count;++i) {
            const auto& expected=cases[i].expected;
            const nm_float4 force=got[i*8u],reaction=got[i*8u+1u];
            const nm_float4 point=got[i*8u+2u],normal=got[i*8u+7u];
            require(std::isfinite(force.x)&&std::isfinite(force.y)&&
                std::isfinite(force.z)&&std::isfinite(force.w)&&
                std::isfinite(reaction.x)&&std::isfinite(reaction.y)&&
                std::isfinite(reaction.z)&&std::isfinite(point.x)&&
                std::isfinite(point.y)&&std::isfinite(point.z)&&
                std::isfinite(normal.x)&&std::isfinite(normal.y)&&
                std::isfinite(normal.z),
                "all exact small-face outputs are finite");
            const Vec3 fExpected=expected.slaveForce;
            const double forceError=std::sqrt(
                std::pow(double(force.x)-fExpected.x,2)+
                std::pow(double(force.y)-fExpected.y,2)+
                std::pow(double(force.z)-fExpected.z,2));
            const double forceScale=std::max(1.0e-12,norm(fExpected));
            maxForceRel=std::max(maxForceRel,forceError/forceScale);
            require(forceError<=std::max(2.0e-6,forceScale*0.02),
                "Metal force direction and magnitude match CPU reference");
            const double normalError=norm(sub(
                Vec3{normal.x,normal.y,normal.z},expected.normal));
            maxNormalErr=std::max(maxNormalErr,normalError);
            if (!(normalError<2.0e-3)) {
                std::fprintf(stderr,
                    "NORMAL_WITNESS row=%zu source_face=%u feature=%u "
                    "query=(%.9g,%.9g,%.9g) gpu=(%.9g,%.9g,%.9g) "
                    "cpu=(%.17g,%.17g,%.17g) point=(%.9g,%.9g,%.9g) "
                    "err=%.9g edge_scale=%.9g\n",
                    i,cases[i].sourceFace,cases[i].caseKind,
                    cases[i].query.x,cases[i].query.y,cases[i].query.z,
                    normal.x,normal.y,normal.z,
                    expected.normal.x,expected.normal.y,expected.normal.z,
                    point.x,point.y,point.z,normalError,cases[i].edgeScale);
                require(false,
                    "Metal contact direction matches double closest-point direction");
            }
            const double pointError=norm(sub(
                Vec3{point.x,point.y,point.z},expected.closestPoint));
            const double pointTolerance=std::max(5.0e-8,cases[i].edgeScale*0.02);
            maxPointRel=std::max(maxPointRel,pointError/cases[i].edgeScale);
            maxPointErrorM=std::max(maxPointErrorM,pointError);
            maxPointRoundoffM=std::max(maxPointRoundoffM,cases[i].pointRoundoffM);
            maxQueryRoundoffM=std::max(maxQueryRoundoffM,cases[i].queryRoundoffM);
            const auto pointUlp=std::max({
                floatUlpDistance(point.x,float(expected.closestPoint.x)),
                floatUlpDistance(point.y,float(expected.closestPoint.y)),
                floatUlpDistance(point.z,float(expected.closestPoint.z))});
            maxPointUlp=std::max(maxPointUlp,pointUlp);
            const auto transformIndex=cases[i].transformIndex;
            maxPointByTransform[transformIndex]=std::max(
                maxPointByTransform[transformIndex],pointError);
            maxPointRelByTransform[transformIndex]=std::max(
                maxPointRelByTransform[transformIndex],
                pointError/cases[i].edgeScale);
            maxPointUlpByTransform[transformIndex]=std::max(
                maxPointUlpByTransform[transformIndex],pointUlp);
            if (!(pointError<=pointTolerance)) {
                std::fprintf(stderr,
                    "POINT_WITNESS row=%zu source_face=%u case=%u "
                    "feature=%u point_error=%.9g tol=%.9g edge_scale=%.9g "
                    "query=(%.9g,%.9g,%.9g) gpu=(%.9g,%.9g,%.9g) "
                    "cpu=(%.17g,%.17g,%.17g)\n",
                    i,cases[i].sourceFace,cases[i].caseKind,
                    static_cast<unsigned>(expected.feature),pointError,
                    pointTolerance,cases[i].edgeScale,cases[i].query.x,
                    cases[i].query.y,cases[i].query.z,point.x,point.y,point.z,
                    expected.closestPoint.x,expected.closestPoint.y,
                    expected.closestPoint.z);
                require(false,
                    "Metal closest point matches the small-face double reference");
            }
            require(std::abs(double(force.w)-expected.closure)<
                    std::max(5.0e-8,expected.closure*0.02),
                "Metal penetration closure matches reference");
            require(std::abs(double(reaction.x)+force.x)<1.0e-7&&
                    std::abs(double(reaction.y)+force.y)<1.0e-7&&
                    std::abs(double(reaction.z)+force.z)<1.0e-7,
                "small-face action and reaction balance");
            const Vec3 expectedMoment=cross(
                sub(expected.closestPoint,cases[i].bodyOrigin),
                expected.bodyForce);
            const nm_float4 moment=got[i*8u+6u];
            const double momentError=norm(sub(
                Vec3{moment.x,moment.y,moment.z},expectedMoment));
            require(momentError<=std::max(2.0e-8,norm(expectedMoment)*0.03),
                "small-face body moment matches closest-point reference");
            for (std::size_t col=0;col<3u;++col) {
                const nm_float4 rowTangent=got[i*8u+3u+col];
                for (std::size_t r=0;r<3u;++r) {
                    const double reference=expected.positiveStiffness[r][col];
                    const double actual=component(
                        {rowTangent.x,rowTangent.y,rowTangent.z,0},r);
                    const double tol=std::max(2.0e-2,std::abs(reference)*0.02);
                    if (!(std::abs(actual-reference)<=tol)) {
                        const nm_float4 g0=got[i*8u+3u];
                        const nm_float4 g1=got[i*8u+4u];
                        const nm_float4 g2=got[i*8u+5u];
                        std::fprintf(stderr,
                            "TANGENT_WITNESS row=%zu source_face=%u case=%u "
                            "cpu_feature=%u gpu_feature=%.0f "
                            "matrix_col=%zu row=%zu gpu=%.9g cpu=%.17g "
                            "error=%.9g tol=%.9g edge_scale=%.9g "
                            "closure_gpu=%.9g closure_cpu=%.17g "
                            "area=%.9g stiffness=%.9g "
                            "normal_gpu=(%.9g,%.9g,%.9g) "
                            "normal_cpu=(%.17g,%.17g,%.17g) "
                            "gpu_diag=(%.9g,%.9g,%.9g) "
                            "gpu_bary=(%.9g,%.9g,%.9g) "
                            "cpu_diag=(%.9g,%.9g,%.9g) "
                            "query=(%.9g,%.9g,%.9g) point=(%.9g,%.9g,%.9g)\n",
                            i,cases[i].sourceFace,cases[i].caseKind,
                            static_cast<unsigned>(expected.feature),point.w,
                            col,r,actual,reference,std::abs(actual-reference),tol,
                            cases[i].edgeScale,force.w,expected.closure,
                            cases[i].sample.bodyLocalPointAndArea.w,
                            cases[i].sample.bodyLocalNormalAndReferenceSeparation.w,
                            normal.x,normal.y,normal.z,expected.normal.x,
                            expected.normal.y,expected.normal.z,g0.x,g1.y,g2.z,
                            got[i*8u+1u].w,g0.w,g1.w,
                            expected.positiveStiffness[0][0],
                            expected.positiveStiffness[1][1],
                            expected.positiveStiffness[2][2],
                            cases[i].query.x,cases[i].query.y,cases[i].query.z,
                            point.x,point.y,point.z);
                        require(false,
                            "small-face feature tangent matches CPU double reference");
                    }
                    maxTangentRel=std::max(maxTangentRel,
                        std::abs(actual-reference)/
                        std::max(1.0,std::abs(reference)));
                }
            }
        }
        require(std::isnan(got[count*8u].x) &&
                std::isnan(got[(count+1u)*8u].x),
                "collinear and nonfinite triangles fail closed");
        std::printf("{\"status\":\"PASS\",\"device\":\"%s\","
            "\"source_faces\":%zu,\"contact_cases\":%zu,"
            "\"rejected_degenerate_nonfinite\":%zu,"
            "\"max_force_relative_error\":%.9g,"
            "\"max_normal_error\":%.9g,\"max_point_error_m\":%.9g,"
            "\"max_world_point_ulp_error\":%u,"
            "\"max_world_point_quantization_m\":%.9g,"
            "\"max_local_query_rounding_m\":%.9g,"
            "\"max_point_error_by_transform_m\":[%.9g,%.9g,%.9g,%.9g,%.9g],"
            "\"max_point_relative_by_transform\":[%.9g,%.9g,%.9g,%.9g,%.9g],"
            "\"max_point_ulp_by_transform\":[%u,%u,%u,%u,%u],"
            "\"max_point_relative_error\":%.9g,"
            "\"point_error_bound_m\":\"max(5e-8,0.02*edgeScale)\","
            "\"max_tangent_relative_error\":%.9g}\n",
            device.name.UTF8String,faceCount,count,rejectedCount,maxForceRel,
            maxNormalErr,maxPointErrorM,maxPointUlp,maxPointRoundoffM,
            maxQueryRoundoffM,
            maxPointByTransform[0],maxPointByTransform[1],maxPointByTransform[2],
            maxPointByTransform[3],maxPointByTransform[4],
            maxPointRelByTransform[0],maxPointRelByTransform[1],
            maxPointRelByTransform[2],maxPointRelByTransform[3],
            maxPointRelByTransform[4],maxPointUlpByTransform[0],
            maxPointUlpByTransform[1],maxPointUlpByTransform[2],
            maxPointUlpByTransform[3],maxPointUlpByTransform[4],
            maxPointRel,maxTangentRel);
    }
    return 0;
}

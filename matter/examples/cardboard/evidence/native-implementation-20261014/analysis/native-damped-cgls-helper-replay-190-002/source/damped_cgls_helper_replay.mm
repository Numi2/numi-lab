#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

struct F4 { float x, y, z, w; };
static_assert(sizeof(F4) == 16, "Metal float4 ABI");

static id<MTLBuffer> bytesBuffer(id<MTLDevice> device, const void* data,
                                 NSUInteger size) {
    id<MTLBuffer> result = [device newBufferWithBytes:data length:size
        options:MTLResourceStorageModeShared];
    if (result == nil) throw std::runtime_error("newBufferWithBytes failed");
    return result;
}
static id<MTLBuffer> zeroBuffer(id<MTLDevice> device, NSUInteger size) {
    std::vector<std::uint8_t> zeros(size, 0u);
    return bytesBuffer(device, zeros.data(), size);
}
static id<MTLComputePipelineState> pipeline(
    id<MTLDevice> device, id<MTLLibrary> library, const char* name) {
    NSString* functionName = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    if (function == nil)
        throw std::runtime_error(std::string("missing Metal function: ") + name);
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (result == nil) {
        std::string message = error == nil ? "unknown pipeline error" :
            [[error localizedDescription] UTF8String];
        throw std::runtime_error(std::string("pipeline creation failed: ") +
            name + ": " + message);
    }
    return result;
}

template<class Binder>
static void dispatch(id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> state,
                     NSUInteger count, bool oneGroup, Binder bind) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state];
    bind(encoder);
    if (oneGroup) {
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    } else {
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        std::string message = command.error == nil ? "unknown command error" :
            [[command.error localizedDescription] UTF8String];
        throw std::runtime_error(std::string("command failed: ") + message);
    }
}
static void setU(id<MTLComputeCommandEncoder> e, std::uint32_t value,
                 NSUInteger index) {
    [e setBytes:&value length:sizeof(value) atIndex:index];
}
static void setF(id<MTLComputeCommandEncoder> e, float value,
                 NSUInteger index) {
    [e setBytes:&value length:sizeof(value) atIndex:index];
}
static void setB(id<MTLComputeCommandEncoder> e, id<MTLBuffer> b,
                 NSUInteger index) {
    [e setBuffer:b offset:0 atIndex:index];
}
static std::array<double, 3> solveReference(const std::array<float, 9>& A,
                                            const std::array<float, 3>& b,
                                            double mu) {
    double M[3][4]{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k)
                M[i][j] += static_cast<double>(A[k * 3 + i]) * A[k * 3 + j];
            if (i == j) M[i][j] += mu * mu;
        }
        for (int k = 0; k < 3; ++k)
            M[i][3] += static_cast<double>(A[k * 3 + i]) * b[k];
    }
    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 3; ++row)
            if (std::fabs(M[row][col]) > std::fabs(M[pivot][col])) pivot = row;
        if (M[pivot][col] == 0.0) throw std::runtime_error("singular CPU oracle");
        for (int j = col; j < 4; ++j) std::swap(M[col][j], M[pivot][j]);
        const double d = M[col][col];
        for (int j = col; j < 4; ++j) M[col][j] /= d;
        for (int row = 0; row < 3; ++row) if (row != col) {
            const double f = M[row][col];
            for (int j = col; j < 4; ++j) M[row][j] -= f * M[col][j];
        }
    }
    return {M[0][3], M[1][3], M[2][3]};
}

static std::string statusName(std::uint32_t s) {
    switch (s) {
        case 1u: return "running";
        case 2u: return "iteration_cap";
        case 3u: return "zero_normal_residual";
        case 4u: return "breakdown";
        case 5u: return "invalid_input";
        default: return "unselected_or_unknown";
    }
}

struct Result {
    double x[3]{};
    std::array<double, 3> expected{};
    double residualNorm = 0.0;
    double referenceError = 0.0;
    double objective = 0.0;
    double recursiveResidualNorm = 0.0;
    double recursiveResidualMismatch = 0.0;
    double finalGamma = 0.0;
    double finalAlpha = 0.0;
    double finalBeta = 0.0;
    std::uint32_t status = 0u;
    std::uint32_t iterations = 0u;
    std::uint32_t errorBits = 0u;
};

static Result runCase(id<MTLDevice> device, id<MTLCommandQueue> queue,
                      const std::map<std::string, id<MTLComputePipelineState>>& ps,
                      double scale, float mu) {
    constexpr std::uint32_t U = 2u; // one xyz mechanics row + zero field row
    constexpr std::uint32_t E = 1u;
    constexpr std::uint32_t cap = 32u;
    const std::array<float, 9> A{
        static_cast<float>(2.0e-3 * scale), static_cast<float>(-0.75e-3 * scale), 0.0f,
        static_cast<float>(0.5e-3 * scale), static_cast<float>(3.0e-3 * scale), 0.0f,
        0.0f, 0.0f, static_cast<float>(1.25e-3 * scale)};
    const std::array<float, 3> b{1.25e-4f, -2.0e-4f, 0.75e-4f};
    std::array<float, 3> atb{};
    for (int col = 0; col < 3; ++col)
        for (int row = 0; row < 3; ++row)
            atb[col] += A[row * 3 + col] * b[row];
    std::array<F4, U> masks{{F4{1,1,1,0}, F4{0,0,0,0}}};
    std::array<std::uint32_t, U> rowEnv{0u,0u};
    std::array<std::uint32_t, E> selected{1u};
    std::array<F4, U> rhs{{F4{b[0],b[1],b[2],0}, F4{0,0,0,0}}};
    std::array<F4, U> atb4{{F4{atb[0],atb[1],atb[2],0}, F4{0,0,0,0}}};
    std::array<F4, 3> rightScale{{F4{1,0,0,1},F4{0,1,0,0},F4{0,0,1,0}}};
    auto ABuffer = bytesBuffer(device, A.data(), sizeof(A));
    auto maskBuffer = bytesBuffer(device, masks.data(), sizeof(masks));
    auto rowEnvBuffer = bytesBuffer(device, rowEnv.data(), sizeof(rowEnv));
    auto selectedBuffer = bytesBuffer(device, selected.data(), sizeof(selected));
    auto rhsBuffer = bytesBuffer(device, rhs.data(), sizeof(rhs));
    auto atbBuffer = bytesBuffer(device, atb4.data(), sizeof(atb4));
    auto rightScaleBuffer = bytesBuffer(device, rightScale.data(), sizeof(rightScale));
    auto xBuffer = zeroBuffer(device, sizeof(F4) * U);
    auto rBuffer = zeroBuffer(device, sizeof(F4) * U);
    auto sBuffer = zeroBuffer(device, sizeof(F4) * U);
    auto pBuffer = zeroBuffer(device, sizeof(F4) * U);
    auto qBuffer = zeroBuffer(device, sizeof(F4) * U);
    auto scalarBuffer = zeroBuffer(device, sizeof(F4) * E);
    auto controlBuffer = zeroBuffer(device, sizeof(std::uint32_t) * 4 * E);
    auto errorBuffer = zeroBuffer(device, sizeof(std::uint32_t) * E);
    const float dampingSquared = mu * mu;
    const std::uint32_t noScale = 0u;

    dispatch(queue, ps.at("nm_fgmres_base_cgls_initialize_state"), E, false,
        [&](id<MTLComputeCommandEncoder> e) {
            setU(e,E,0); setU(e,cap,1); setB(e,selectedBuffer,2);
            setB(e,scalarBuffer,3); setB(e,controlBuffer,4); setB(e,errorBuffer,5);
        });
    dispatch(queue, ps.at("nm_fgmres_base_cgls_initialize_vectors"), U, false,
        [&](id<MTLComputeCommandEncoder> e) {
            setU(e,U,0); setU(e,E,1); setB(e,maskBuffer,2); setB(e,rowEnvBuffer,3);
            setB(e,selectedBuffer,4); setB(e,rhsBuffer,5); setB(e,atbBuffer,6);
            setB(e,xBuffer,7); setB(e,rBuffer,8); setB(e,sBuffer,9);
            setB(e,pBuffer,10); setB(e,qBuffer,11); setB(e,errorBuffer,12);
            setU(e,noScale,13); setB(e,rightScaleBuffer,14);
        });
    dispatch(queue, ps.at("nm_fgmres_base_cgls_measure_initial"), E, true,
        [&](id<MTLComputeCommandEncoder> e) {
            setU(e,U,0); setU(e,E,1); setB(e,maskBuffer,2); setB(e,rowEnvBuffer,3);
            setB(e,rhsBuffer,4); setB(e,rBuffer,5); setB(e,sBuffer,6);
            setB(e,scalarBuffer,7); setB(e,controlBuffer,8); setB(e,errorBuffer,9);
        });

    for (std::uint32_t step = 0; step < 6u; ++step) {
        dispatch(queue, ps.at("damped_fixture_apply"), U, false,
            [&](id<MTLComputeCommandEncoder> e) {
                setB(e,ABuffer,0); setB(e,pBuffer,1); setB(e,qBuffer,2);
                setU(e,U,3); const std::uint32_t trans=0u; setU(e,trans,4);
            });
        dispatch(queue, ps.at("nm_fgmres_base_cgls_form_alpha"), E, true,
            [&](id<MTLComputeCommandEncoder> e) {
                setU(e,U,0); setU(e,E,1); setB(e,maskBuffer,2); setB(e,rowEnvBuffer,3);
                setB(e,qBuffer,4); setB(e,scalarBuffer,5); setB(e,controlBuffer,6);
                setB(e,errorBuffer,7); setB(e,pBuffer,8); setF(e,dampingSquared,9);
            });
        dispatch(queue, ps.at("nm_fgmres_base_cgls_update_xr"), U, false,
            [&](id<MTLComputeCommandEncoder> e) {
                setU(e,U,0); setU(e,E,1); setB(e,maskBuffer,2); setB(e,rowEnvBuffer,3);
                setB(e,scalarBuffer,4); setB(e,controlBuffer,5); setB(e,pBuffer,6);
                setB(e,qBuffer,7); setB(e,xBuffer,8); setB(e,rBuffer,9); setB(e,errorBuffer,10);
            });
        dispatch(queue, ps.at("nm_fgmres_base_cgls_complete_iteration"), E, false,
            [&](id<MTLComputeCommandEncoder> e) {
                setU(e,E,0); setB(e,controlBuffer,1); setB(e,errorBuffer,2);
            });
        dispatch(queue, ps.at("damped_fixture_apply"), U, false,
            [&](id<MTLComputeCommandEncoder> e) {
                setB(e,ABuffer,0); setB(e,rBuffer,1); setB(e,sBuffer,2);
                setU(e,U,3); const std::uint32_t trans=1u; setU(e,trans,4);
            });
        dispatch(queue, ps.at("nm_fgmres_base_cgls_form_beta"), E, true,
            [&](id<MTLComputeCommandEncoder> e) {
                setU(e,U,0); setU(e,E,1); setB(e,maskBuffer,2); setB(e,rowEnvBuffer,3);
                setB(e,sBuffer,4); setB(e,scalarBuffer,5); setB(e,controlBuffer,6);
                setB(e,errorBuffer,7); setB(e,xBuffer,8); setF(e,dampingSquared,9);
            });
        const auto* control = static_cast<const std::uint32_t*>(controlBuffer.contents);
        const std::uint32_t status = control[0];
        if (status != 1u) break;
        dispatch(queue, ps.at("nm_fgmres_base_cgls_update_p"), U, false,
            [&](id<MTLComputeCommandEncoder> e) {
                setU(e,U,0); setU(e,E,1); setB(e,maskBuffer,2); setB(e,rowEnvBuffer,3);
                setB(e,scalarBuffer,4); setB(e,controlBuffer,5); setB(e,sBuffer,6);
                setB(e,pBuffer,7); setB(e,errorBuffer,8);
            });
    }
    const auto* x = static_cast<const F4*>(xBuffer.contents);
    const auto* residual = static_cast<const F4*>(rBuffer.contents);
    const auto* scalars = static_cast<const F4*>(scalarBuffer.contents);
    const auto* control = static_cast<const std::uint32_t*>(controlBuffer.contents);
    const auto* errors = static_cast<const std::uint32_t*>(errorBuffer.contents);
    Result result;
    result.x[0]=x[0].x; result.x[1]=x[0].y; result.x[2]=x[0].z;
    result.status=control[0]; result.iterations=control[1]; result.errorBits=errors[0];
    result.expected=solveReference(A,b,mu);
    double r2=0.0, objective=0.0, error2=0.0;
    for (int i=0;i<3;++i) {
        double ax=0.0; for (int j=0;j<3;++j) ax += A[i*3+j]*result.x[j];
        const double ri=b[i]-ax;
        r2 += ri*ri; objective += ri*ri + mu*mu*result.x[i]*result.x[i];
        const double d=result.x[i]-result.expected[i]; error2+=d*d;
    }
    result.residualNorm=std::sqrt(r2); result.objective=objective;
    const double dr0=static_cast<double>(residual[0].x)-b[0];
    const double dr1=static_cast<double>(residual[0].y)-b[1];
    const double dr2=static_cast<double>(residual[0].z)-b[2];
    result.recursiveResidualNorm=std::sqrt(dr0*dr0+dr1*dr1+dr2*dr2);
    result.recursiveResidualMismatch=std::fabs(result.recursiveResidualNorm-result.residualNorm);
    result.finalGamma=scalars[0].x; result.finalAlpha=scalars[0].y; result.finalBeta=scalars[0].z;
    result.referenceError=std::sqrt(error2);
    if (result.errorBits != 0u || (result.status != 1u && result.status != 2u && result.status != 3u))
        throw std::runtime_error("production CGLS kernels returned an invalid status/error");
    const double tol=2.0e-6 + 5.0e-4*std::sqrt(
        result.expected[0]*result.expected[0]+result.expected[1]*result.expected[1]+result.expected[2]*result.expected[2]);
    if (result.referenceError > tol)
        throw std::runtime_error("native CGLS candidate exceeds independent CPU normal-equation oracle tolerance");
    return result;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc != 2) {
                std::fprintf(stderr,"usage: damped_cgls_helper_replay <metallib>\n"); return 2;
            }
            id<MTLDevice> device=MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("no Metal device");
            id<MTLCommandQueue> queue=[device newCommandQueue];
            NSURL* url=[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
            NSError* error=nil;
            id<MTLLibrary> library=[device newLibraryWithURL:url error:&error];
            if (!library) {
                std::string message=error==nil?"unknown library error":
                    [[error localizedDescription] UTF8String];
                throw std::runtime_error("metallib load failed: "+message);
            }
            std::map<std::string,id<MTLComputePipelineState>> ps;
            const char* names[]={
                "damped_fixture_apply",
                "nm_fgmres_base_cgls_initialize_state",
                "nm_fgmres_base_cgls_initialize_vectors",
                "nm_fgmres_base_cgls_measure_initial",
                "nm_fgmres_base_cgls_form_alpha",
                "nm_fgmres_base_cgls_update_xr",
                "nm_fgmres_base_cgls_complete_iteration",
                "nm_fgmres_base_cgls_form_beta",
                "nm_fgmres_base_cgls_update_p"};
            for (const char* name:names) ps.emplace(name,pipeline(device,library,name));
            struct CaseSpec { double scale; float mu; } specs[]={
                {1.0,0.0f},{1.0,1.0e-3f},{2.0,0.0f},{2.0,1.0e-3f}};
            Result results[4];
            for (int i=0;i<4;++i) results[i]=runCase(device,queue,ps,specs[i].scale,specs[i].mu);
            double dampingDifference[2]{};
            for (int scaleIndex=0;scaleIndex<2;++scaleIndex) {
                const Result& a=results[scaleIndex*2];
                const Result& b=results[scaleIndex*2+1];
                for(int k=0;k<3;++k) dampingDifference[scaleIndex]+=(a.x[k]-b.x[k])*(a.x[k]-b.x[k]);
                dampingDifference[scaleIndex]=std::sqrt(dampingDifference[scaleIndex]);
                if (dampingDifference[scaleIndex] < 1.0e-3)
                    throw std::runtime_error("damping did not produce a resolved direction change at the chosen operator scale");
            }
            std::cout<<std::setprecision(10)<<"{\"schema\":\"numi.cardboard.damped-cgls-production-kernel-replay.v1\",\"native_metal_executed\":true,\"production_kernels\":[\"nm_fgmres_base_cgls_form_alpha\",\"nm_fgmres_base_cgls_form_beta\",\"nm_fgmres_base_cgls_update_xr\",\"nm_fgmres_base_cgls_update_p\",\"nm_fgmres_base_cgls_complete_iteration\"],\"scope\":\"private 3-DOF nonsymmetric matrix fixture; test A/A^T only; no FEM, accepted history, contact, Newton certificate, Armijo trajectory, or physics qualification\",\"cases\":[";
            for(int i=0;i<4;++i){
                const double scale=specs[i].scale; const float mu=specs[i].mu; const Result& r=results[i];
                if(i) std::cout<<",";
                std::cout<<"{\"scale\":"<<scale<<",\"mu_kg\":"<<mu<<",\"status\":\""<<statusName(r.status)<<"\",\"iterations\":"<<r.iterations<<",\"error_bits\":"<<r.errorBits<<",\"x\":["<<r.x[0]<<","<<r.x[1]<<","<<r.x[2]<<"],\"cpu_normal_equation_reference\":["<<r.expected[0]<<","<<r.expected[1]<<","<<r.expected[2]<<"],\"reference_error_l2\":"<<r.referenceError<<",\"physical_residual_norm\":"<<r.residualNorm<<",\"recursive_residual_norm\":"<<r.recursiveResidualNorm<<",\"recursive_residual_mismatch\":"<<r.recursiveResidualMismatch<<",\"gamma\":"<<r.finalGamma<<",\"alpha\":"<<r.finalAlpha<<",\"beta\":"<<r.finalBeta<<",\"regularized_objective\":"<<r.objective<<"}";
            }
            std::cout<<"],\"damping_direction_delta_l2\":["<<dampingDifference[0]<<","<<dampingDifference[1]<<"],\"compiled_production_recurrence\":true,\"default_off_runtime_path_tested\":false,\"full_coupon_qualified\":false}\n";
            return 0;
        } catch(const std::exception& e) {
            std::fprintf(stderr,"damped_cgls_helper_replay_error=%s\n",e.what()); return 1;
        }
    }
}

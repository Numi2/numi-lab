#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_HUMAN_SKIN_SPRING_LINEARIZATION_METALLIB
#error NUMI_HUMAN_SKIN_SPRING_LINEARIZATION_METALLIB must name the test metallib
#endif

namespace {

struct Dispatch {
    std::uint32_t nv;
    float stiffness;
    std::uint32_t reserved0;
    std::uint32_t reserved1;
};

using Vec3 = std::array<double, 3u>;
using Vec4 = std::array<double, 4u>;

Vec3 bodyPoint(const Vec4& q) {
    return {q[0] + std::cos(q[3]), q[1] + std::sin(q[3]), q[2]};
}

std::array<float, 12u> pointJacobian(const Vec4& q) {
    const double c = std::cos(q[3]);
    const double s = std::sin(q[3]);
    return {1.0f, 0.0f, 0.0f, static_cast<float>(-s),
            0.0f, 1.0f, 0.0f, static_cast<float>(c),
            0.0f, 0.0f, 1.0f, 0.0f};
}

std::array<float, 48u> pointHessian(const Vec4& q) {
    std::array<float, 48u> result{};
    result[3u * 4u + 3u] = static_cast<float>(-std::cos(q[3]));
    result[16u + 3u * 4u + 3u] = static_cast<float>(-std::sin(q[3]));
    return result;
}

std::array<double, 7u> gradient(
    const Vec3& x, const Vec4& q, const double stiffness) {
    const Vec3 y = bodyPoint(q);
    const Vec3 d{x[0] - y[0], x[1] - y[1], x[2] - y[2]};
    const auto jacobian = pointJacobian(q);
    std::array<double, 7u> result{};
    for (std::size_t axis = 0u; axis < 3u; ++axis)
        result[axis] = stiffness * d[axis];
    for (std::size_t dof = 0u; dof < 4u; ++dof) {
        double product = 0.0;
        for (std::size_t axis = 0u; axis < 3u; ++axis)
            product += jacobian[axis * 4u + dof] * d[axis];
        result[3u + dof] = -stiffness * product;
    }
    return result;
}

id<MTLBuffer> makeBuffer(id<MTLDevice> device, const void* data,
                         const NSUInteger bytes) {
    if (data == nullptr || bytes == 0u) return nil;
    return [device newBufferWithBytes:data length:bytes
                              options:MTLResourceStorageModeShared];
}

void requireNear(const double actual, const double expected,
                 const double tolerance, const char* label) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance * (1.0 + std::abs(expected))) {
        throw std::runtime_error(std::string(label) + " mismatch");
    }
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) throw std::runtime_error("Metal device unavailable");
            NSString* path = [NSString stringWithUTF8String:
                NUMI_HUMAN_SKIN_SPRING_LINEARIZATION_METALLIB];
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithURL:
                [NSURL fileURLWithPath:path] error:&error];
            if (library == nil) throw std::runtime_error(
                error.localizedDescription.UTF8String != nullptr ? error.localizedDescription.UTF8String : "test metallib unavailable");
            id<MTLFunction> function = [library
                newFunctionWithName:@"nm_human_skin_spring_linearization_check"];
            if (function == nil) throw std::runtime_error("test kernel missing");
            id<MTLComputePipelineState> pipeline = [device
                newComputePipelineStateWithFunction:function error:&error];
            if (pipeline == nil) throw std::runtime_error(
                error.localizedDescription.UTF8String != nullptr ? error.localizedDescription.UTF8String : "test pipeline unavailable");

            constexpr double stiffness = 230.0;
            constexpr double epsilon = 2.0e-3;
            const Vec4 q{0.2, -0.1, 0.05, 0.41};
            const Vec3 y = bodyPoint(q);
            const Vec3 x{y[0] + 0.004, y[1] - 0.002, y[2] + 0.003};
            const Vec3 dx{0.003, -0.006, 0.001};
            const Vec4 dq{0.12, -0.08, 0.03, 0.2};
            const auto jacobian = pointJacobian(q);
            const auto hessian = pointHessian(q);
            const float nodePoint[4]{static_cast<float>(x[0]),
                static_cast<float>(x[1]), static_cast<float>(x[2]), 0.0f};
            const float bodyPointValue[4]{static_cast<float>(y[0]),
                static_cast<float>(y[1]), static_cast<float>(y[2]), 0.0f};
            const float nodeDirection[4]{static_cast<float>(dx[0]),
                static_cast<float>(dx[1]), static_cast<float>(dx[2]), 0.0f};
            const float bodyDirection[4]{static_cast<float>(dq[0]),
                static_cast<float>(dq[1]), static_cast<float>(dq[2]),
                static_cast<float>(dq[3])};
            const Dispatch dispatch{4u, static_cast<float>(stiffness), 0u, 0u};
            std::array<float, 16u> zeroOutput{};
            id<MTLBuffer> dispatchBuffer = makeBuffer(device, &dispatch,
                sizeof(dispatch));
            id<MTLBuffer> nodeBuffer = makeBuffer(device, nodePoint,
                sizeof(nodePoint));
            id<MTLBuffer> bodyBuffer = makeBuffer(device, bodyPointValue,
                sizeof(bodyPointValue));
            id<MTLBuffer> jacobianBuffer = makeBuffer(device, jacobian.data(),
                sizeof(jacobian));
            id<MTLBuffer> hessianBuffer = makeBuffer(device, hessian.data(),
                sizeof(hessian));
            id<MTLBuffer> nodeDirectionBuffer = makeBuffer(device,
                nodeDirection, sizeof(nodeDirection));
            id<MTLBuffer> bodyDirectionBuffer = makeBuffer(device,
                bodyDirection, sizeof(bodyDirection));
            id<MTLBuffer> outputBuffer = makeBuffer(device, zeroOutput.data(),
                sizeof(zeroOutput));
            if (dispatchBuffer == nil || nodeBuffer == nil || bodyBuffer == nil ||
                jacobianBuffer == nil || hessianBuffer == nil ||
                nodeDirectionBuffer == nil || bodyDirectionBuffer == nil ||
                outputBuffer == nil)
                throw std::runtime_error("test buffers unavailable");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
            [encoder setBuffer:nodeBuffer offset:0u atIndex:1u];
            [encoder setBuffer:bodyBuffer offset:0u atIndex:2u];
            [encoder setBuffer:jacobianBuffer offset:0u atIndex:3u];
            [encoder setBuffer:hessianBuffer offset:0u atIndex:4u];
            [encoder setBuffer:nodeDirectionBuffer offset:0u atIndex:5u];
            [encoder setBuffer:bodyDirectionBuffer offset:0u atIndex:6u];
            [encoder setBuffer:outputBuffer offset:0u atIndex:7u];
            [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error(command.error.localizedDescription.UTF8String != nullptr ? command.error.localizedDescription.UTF8String : "GPU check failed");
            const auto* output = static_cast<const float*>(outputBuffer.contents);
            if (output[15] != 1.0f || !std::isfinite(output[0]))
                throw std::runtime_error("GPU spring linearization rejected valid input");

            const Vec3 d{x[0] - y[0], x[1] - y[1], x[2] - y[2]};
            const double expectedEnergy = 0.5 * stiffness *
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            requireNear(output[0], expectedEnergy, 2.0e-5, "energy");
            const auto baseGradient = gradient(x, q, stiffness);
            for (std::size_t i = 0u; i < 3u; ++i)
                requireNear(output[1u + i], baseGradient[i], 2.0e-5,
                    "node gradient");
            for (std::size_t i = 0u; i < 4u; ++i)
                requireNear(output[4u + i], baseGradient[3u + i], 3.0e-5,
                    "body gradient");

            Vec3 xPlus{}, xMinus{};
            Vec4 qPlus{}, qMinus{};
            for (std::size_t i = 0u; i < 3u; ++i) {
                xPlus[i] = x[i] + epsilon * dx[i];
                xMinus[i] = x[i] - epsilon * dx[i];
            }
            for (std::size_t i = 0u; i < 4u; ++i) {
                qPlus[i] = q[i] + epsilon * dq[i];
                qMinus[i] = q[i] - epsilon * dq[i];
            }
            const auto gradientPlus = gradient(xPlus, qPlus, stiffness);
            const auto gradientMinus = gradient(xMinus, qMinus, stiffness);
            for (std::size_t i = 0u; i < 3u; ++i) {
                const double fd = (gradientPlus[i] - gradientMinus[i]) /
                    (2.0 * epsilon);
                requireNear(output[8u + i], fd, 4.0e-4,
                    "node Hessian action");
            }
            for (std::size_t i = 0u; i < 4u; ++i) {
                const double fd = (gradientPlus[3u + i] -
                    gradientMinus[3u + i]) / (2.0 * epsilon);
                requireNear(output[11u + i], fd, 6.0e-4,
                    "body Hessian action");
            }
            // The body virtual work must carry the exact opposite translational
            // spring force, and the rotational generalized load must equal the
            // moment of that force about the articulated body origin.
            for (std::size_t axis = 0u; axis < 3u; ++axis)
                requireNear(output[1u + axis], -output[4u + axis],
                    3.0e-5, "action-reaction");
            const Vec3 bodyForce{-output[4], -output[5], -output[6]};
            const Vec3 arm{std::cos(q[3]), std::sin(q[3]), 0.0};
            const double momentZ = arm[0] * bodyForce[1] -
                arm[1] * bodyForce[0];
            requireNear(-output[7], momentZ, 3.0e-5,
                "opposite body moment");
            std::cout << "human_skin_spring_linearization=passed"
                << " energy_gradient_hessian=finite_difference"
                << " action_reaction=passed"
                << " tangent=exact_point_hessian_required"
                << std::endl;
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "human_skin_spring_linearization=failed reason=\""
                << exception.what() << "\"" << std::endl;
            return 1;
        }
    }
}

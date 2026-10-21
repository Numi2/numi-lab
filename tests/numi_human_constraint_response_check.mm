#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr unsigned kDofs = 3u;
constexpr unsigned kRows = 2u;
using Vector = std::array<double, kDofs>;
using Matrix = std::array<double, kDofs * kDofs>;

Matrix denseFromFactor(const std::array<float, 9>& factor) {
    Matrix matrix{};
    for (unsigned row = 0; row < kDofs; ++row) {
        for (unsigned column = 0; column < kDofs; ++column) {
            double value = 0.0;
            for (unsigned k = 0; k < kDofs; ++k)
                value += static_cast<double>(factor[row * kDofs + k]) *
                    static_cast<double>(factor[column * kDofs + k]);
            matrix[row * kDofs + column] = value;
        }
    }
    return matrix;
}

Vector solve(const Matrix& matrix, const Vector& rhs) {
    double augmented[kDofs][kDofs + 1u]{};
    for (unsigned row = 0; row < kDofs; ++row) {
        for (unsigned column = 0; column < kDofs; ++column)
            augmented[row][column] = matrix[row * kDofs + column];
        augmented[row][kDofs] = rhs[row];
    }
    for (unsigned pivot = 0; pivot < kDofs; ++pivot) {
        unsigned best = pivot;
        for (unsigned row = pivot + 1u; row < kDofs; ++row)
            if (std::abs(augmented[row][pivot]) >
                std::abs(augmented[best][pivot])) best = row;
        require(std::abs(augmented[best][pivot]) > 1.0e-12,
            "CPU reference matrix is singular");
        for (unsigned column = pivot; column <= kDofs; ++column)
            std::swap(augmented[pivot][column], augmented[best][column]);
        const double diagonal = augmented[pivot][pivot];
        for (unsigned column = pivot; column <= kDofs; ++column)
            augmented[pivot][column] /= diagonal;
        for (unsigned row = 0; row < kDofs; ++row) {
            if (row == pivot) continue;
            const double scale = augmented[row][pivot];
            for (unsigned column = pivot; column <= kDofs; ++column)
                augmented[row][column] -= scale * augmented[pivot][column];
        }
    }
    Vector result{};
    for (unsigned row = 0; row < kDofs; ++row)
        result[row] = augmented[row][kDofs];
    return result;
}

void near(float actual, double expected, const char* label) {
    const double error = std::abs(static_cast<double>(actual) - expected);
    const double tolerance = 2.0e-5 * std::max(1.0, std::abs(expected));
    if (!std::isfinite(actual) || error > tolerance)
        throw std::runtime_error(label);
}
}

int main(int argc, char** argv) {
    try {
        @autoreleasepool {
            require(argc == 2,
                "usage: numi-human-constraint-response-check METALLIB");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "Metal device unavailable");
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithURL:
                [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
                error:&error];
            require(library != nil, "cannot load Metal library");
            id<MTLFunction> function = [library newFunctionWithName:
                @"numi_human_constraint_response_check"];
            require(function != nil, "constraint response probe kernel missing");
            id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:function error:&error];
            require(pipeline != nil, "cannot compile constraint response probe");

            const std::array<float, 9> factor{
                2.0f, 0.0f, 0.0f,
                0.5f, 1.5f, 0.0f,
                -0.25f, 0.2f, 1.2f
            };
            const std::array<float, 9> pointJacobians{
                1.0f, 0.2f, -0.1f,
                0.3f, 1.4f, 0.5f,
                -0.2f, 0.1f, 1.1f
            };
            id<MTLBuffer> factorBuffer = [device newBufferWithBytes:
                factor.data() length:sizeof(factor)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> jacobianBuffer = [device newBufferWithBytes:
                pointJacobians.data() length:sizeof(pointJacobians)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> rowBuffer = [device newBufferWithLength:
                kRows * kDofs * sizeof(float)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> responseBuffer = [device newBufferWithLength:
                kRows * kDofs * sizeof(float)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> delassusBuffer = [device newBufferWithLength:
                kRows * kRows * sizeof(float)
                options:MTLResourceStorageModeShared];
            const std::array<float, 9> singularFactor{};
            id<MTLBuffer> singularFactorBuffer = [device newBufferWithBytes:
                singularFactor.data() length:sizeof(singularFactor)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> resultBuffer = [device newBufferWithLength:
                4u * sizeof(unsigned) options:MTLResourceStorageModeShared];
            require(factorBuffer && jacobianBuffer && rowBuffer &&
                responseBuffer && delassusBuffer && singularFactorBuffer &&
                resultBuffer,
                "constraint response test buffers unavailable");

            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "Metal command queue unavailable");
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            require(command && encoder, "cannot create Metal command");
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:factorBuffer offset:0 atIndex:0];
            [encoder setBuffer:jacobianBuffer offset:0 atIndex:1];
            [encoder setBuffer:rowBuffer offset:0 atIndex:2];
            [encoder setBuffer:responseBuffer offset:0 atIndex:3];
            [encoder setBuffer:delassusBuffer offset:0 atIndex:4];
            [encoder setBuffer:resultBuffer offset:0 atIndex:5];
            [encoder setBuffer:singularFactorBuffer offset:0 atIndex:6];
            [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            require(command.status == MTLCommandBufferStatusCompleted,
                "constraint response Metal dispatch failed");
            const auto* result =
                static_cast<const unsigned*>(resultBuffer.contents);
            require(result[0] == 0u,
                "shared response algebra rejected the manufactured rows");
            require(result[1] == 0u,
                "invalid or boundary equality row was mishandled");
            require(result[2] == 0u,
                "unit or degenerate contact normal was mishandled");
            require(result[3] == 0u,
                "singular A0 factor was not rejected");

            const float normalZ = std::sqrt(0.75f);
            const std::array<Vector, kRows> rows{{
                Vector{1.0, 0.0, -0.35},
                Vector{
                    0.3 * static_cast<double>(pointJacobians[0]) +
                        0.4 * static_cast<double>(pointJacobians[3]) +
                        static_cast<double>(normalZ) *
                            static_cast<double>(pointJacobians[6]),
                    0.3 * static_cast<double>(pointJacobians[1]) +
                        0.4 * static_cast<double>(pointJacobians[4]) +
                        static_cast<double>(normalZ) *
                            static_cast<double>(pointJacobians[7]),
                    0.3 * static_cast<double>(pointJacobians[2]) +
                        0.4 * static_cast<double>(pointJacobians[5]) +
                        static_cast<double>(normalZ) *
                            static_cast<double>(pointJacobians[8])
                }
            }};
            const Matrix mass = denseFromFactor(factor);
            std::array<Vector, kRows> responses{};
            for (unsigned row = 0; row < kRows; ++row)
                responses[row] = solve(mass, rows[row]);
            const auto* actualRows =
                static_cast<const float*>(rowBuffer.contents);
            const auto* actualResponses =
                static_cast<const float*>(responseBuffer.contents);
            const auto* actualDelassus =
                static_cast<const float*>(delassusBuffer.contents);
            unsigned checks = 1u;
            for (unsigned row = 0; row < kRows; ++row) {
                for (unsigned dof = 0; dof < kDofs; ++dof) {
                    near(actualRows[row * kDofs + dof], rows[row][dof],
                        "source constraint Jacobian differs from reference");
                    near(actualResponses[row * kDofs + dof],
                        responses[row][dof],
                        "A0^-1 J^T response differs from reference");
                    ++checks;
                }
            }
            double expectedDelassus[kRows][kRows]{};
            for (unsigned row = 0; row < kRows; ++row) {
                for (unsigned column = 0; column < kRows; ++column) {
                    for (unsigned dof = 0; dof < kDofs; ++dof)
                        expectedDelassus[row][column] += rows[row][dof] *
                            responses[column][dof];
                    near(actualDelassus[row * kRows + column],
                        expectedDelassus[row][column],
                        "Delassus product differs from reference");
                    ++checks;
                }
            }
            require(std::abs(actualDelassus[1] - actualDelassus[2]) < 2.0e-5f,
                "bilateral/unilateral Delassus matrix lost symmetry");
            std::printf(
                "{\"passed\":true,\"mode\":\"metal\",\"rows\":2,"
                "\"dofs\":3,\"checks\":%u,"
                "\"edge_checks\":4,"
                "\"constraint_types\":[\"bilateral\",\"unilateral_normal\"]}\n",
                checks);
        }
        return 0;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "constraint response check failed: %s\n",
            exception.what());
        return 1;
    }
}

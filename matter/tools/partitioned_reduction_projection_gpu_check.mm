#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/partitioned_reduction_basis.hpp"
#include "numi/matter/partitioned_reduction_projection.h"
#include "numi/matter/partitioned_reduction_fem_tangent.h"
#include "numi/matter/partitioned_reduction_sparse_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_PARTITIONED_REDUCTION_PROJECTION_METALLIB
#define NUMI_PARTITIONED_REDUCTION_PROJECTION_METALLIB ""
#endif

namespace {

using namespace numi::matter;
using Vec3 = std::array<double, 3u>;
using Tet = std::array<std::uint32_t, 4u>;
constexpr std::uint32_t kNX = 8u;
constexpr std::uint32_t kNY = 7u;
constexpr std::uint32_t kNZ = 6u;
constexpr std::uint32_t kGlobalNodeOffset = 7u;
constexpr float kAbsoluteTolerance = 2.0e-6f;
constexpr float kRelativeTolerance = 4.0e-4f;

void need(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool close(const double a, const double b,
           const double absolute = kAbsoluteTolerance,
           const double relative = kRelativeTolerance) {
    return std::abs(a - b) <= absolute + relative *
        std::max(std::abs(a), std::abs(b));
}

template<class T>
id<MTLBuffer> makeBuffer(id<MTLDevice> device, const std::vector<T>& values) {
    const T zero{};
    const void* bytes = values.empty() ? static_cast<const void*>(&zero) :
        static_cast<const void*>(values.data());
    return [device newBufferWithBytes:bytes
        length:std::max<std::size_t>(1u, values.size()) * sizeof(T)
        options:MTLResourceStorageModeShared];
}

struct Fixture {
    std::vector<Vec3> positions;
    std::vector<double> masses;
    std::vector<std::uint8_t> fixed;
    std::vector<Tet> tetrahedra;
    std::vector<std::uint32_t> labels;
    std::vector<std::uint32_t> anchors;
};

std::uint32_t node(const std::uint32_t x, const std::uint32_t y,
                   const std::uint32_t z) {
    return (z * kNY + y) * kNX + x;
}

Fixture makeFixture() {
    Fixture fixture;
    const std::uint32_t nodeCount = kNX * kNY * kNZ;
    fixture.positions.resize(nodeCount);
    fixture.masses.resize(nodeCount);
    fixture.fixed.assign(nodeCount, 0u);
    for (std::uint32_t z = 0u; z < kNZ; ++z) {
        for (std::uint32_t y = 0u; y < kNY; ++y) {
            for (std::uint32_t x = 0u; x < kNX; ++x) {
                const std::uint32_t index = node(x, y, z);
                fixture.positions[index] = {
                    0.001 * static_cast<double>(x),
                    0.0013 * static_cast<double>(y),
                    0.0008 * static_cast<double>(z)};
                // Binary fractions are exactly representable by the GPU test.
                fixture.masses[index] =
                    0.5 + 0.125 * static_cast<double>(index % 7u);
                if (x == 0u) fixture.fixed[index] = 1u;
            }
        }
    }
    fixture.anchors = {node(3u, 3u, 3u)};
    constexpr std::array<std::array<std::uint32_t, 4u>, 6u> split{{
        {{0u, 1u, 2u, 6u}}, {{0u, 2u, 3u, 6u}},
        {{0u, 3u, 7u, 6u}}, {{0u, 7u, 4u, 6u}},
        {{0u, 4u, 5u, 6u}}, {{0u, 5u, 1u, 6u}},
    }};
    for (std::uint32_t z = 0u; z + 1u < kNZ; ++z) {
        for (std::uint32_t y = 0u; y + 1u < kNY; ++y) {
            for (std::uint32_t x = 0u; x + 1u < kNX; ++x) {
                const std::array<std::uint32_t, 8u> corners{
                    node(x, y, z), node(x + 1u, y, z),
                    node(x + 1u, y + 1u, z), node(x, y + 1u, z),
                    node(x, y, z + 1u), node(x + 1u, y, z + 1u),
                    node(x + 1u, y + 1u, z + 1u),
                    node(x, y + 1u, z + 1u)};
                for (const auto& local : split) {
                    fixture.tetrahedra.push_back({
                        corners[local[0]], corners[local[1]],
                        corners[local[2]], corners[local[3]]});
                    fixture.labels.push_back(x < kNX / 2u ? 101u : 202u);
                }
            }
        }
    }
    return fixture;
}

PartitionedReductionBasis buildBasis(const Fixture& fixture) {
    PartitionedReductionBasisInput input{
        std::span<const Vec3>(fixture.positions),
        std::span<const double>(fixture.masses),
        std::span<const std::uint8_t>(fixture.fixed),
        std::span<const Tet>(fixture.tetrahedra),
        std::span<const std::uint32_t>(fixture.labels),
        std::span<const std::uint32_t>(fixture.anchors)};
    PartitionedReductionBasisOptions options;
    options.minimumCoordinateReductionFactor = 4u;
    options.maximumModeCount = 32u;
    auto basis = buildPartitionedReductionBasis(input, options);
    need(basis.valid(), "CPU basis fixture failed: " + basis.error);
    need(basis.nodeCount > NM_PARTITIONED_REDUCTION_TILE_WIDTH &&
         basis.modeCount > 0u &&
         basis.modeCount * 4u <= basis.freeCoordinateCount,
         "CPU fixture does not exercise tiled reduction and coordinate reduction");
    bool hasFixedZero = false;
    for (std::uint32_t row = 0u; row < basis.nodeCount; ++row) {
        if (fixture.fixed[row] == 0u) continue;
        bool allZero = true;
        for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
            const auto& value =
                basis.values[static_cast<std::size_t>(mode) * basis.nodeCount + row];
            allZero = allZero && value.x == 0.0f && value.y == 0.0f &&
                value.z == 0.0f;
        }
        need(allZero, "provider emitted a nonzero fixed-node coefficient");
        hasFixedZero = true;
    }
    need(hasFixedZero, "fixture has no fixed rows");
    bool hasAnchorMode = false;
    for (const auto& mode : basis.modes)
        hasAnchorMode = hasAnchorMode ||
            mode.kind == PartitionedReductionModeKind::anchorTranslation;
    need(hasAnchorMode, "provider omitted the anchor identity modes");
    return basis;
}

struct DeviceKernels {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> copy = nil;
    id<MTLComputePipelineState> prolongate = nil;
    id<MTLComputePipelineState> reduceTiles = nil;
    id<MTLComputePipelineState> reducePartials = nil;
    id<MTLComputePipelineState> invertNodeBlocks = nil;
    id<MTLComputePipelineState> reduceBlockTiles = nil;
    id<MTLComputePipelineState> reduceBlockPartials = nil;
    id<MTLComputePipelineState> regionValidityTiles = nil;
    id<MTLComputePipelineState> regionValidityPartials = nil;
    id<MTLComputePipelineState> factorRegionBlocks = nil;
    id<MTLComputePipelineState> solveRegionBlocks = nil;
    id<MTLComputePipelineState> addRegionCorrection = nil;
    id<MTLComputePipelineState> reduceSparseModeTiles = nil;
    id<MTLComputePipelineState> reduceSparseModePartials = nil;
    id<MTLComputePipelineState> reduceSparseBlockTiles = nil;
    id<MTLComputePipelineState> reduceSparseBlockPartials = nil;
    id<MTLComputePipelineState> regionSparseValidityTiles = nil;
    id<MTLComputePipelineState> regionSparseValidityPartials = nil;
    id<MTLComputePipelineState> addSparseRegionCorrection = nil;
    id<MTLComputePipelineState> femTangentFDCheck = nil;
    id<MTLComputePipelineState> hybridCopy = nil;
    id<MTLComputePipelineState> hybridLift = nil;
    id<MTLComputePipelineState> hybridZeroSelected = nil;
    id<MTLComputePipelineState> hybridPackModes = nil;
    id<MTLComputePipelineState> hybridClearInactiveField = nil;

    DeviceKernels() {
        device = MTLCreateSystemDefaultDevice();
        need(device != nil, "Apple Metal device is unavailable");
        const std::string name = [[device name] UTF8String];
        need(name.find("Apple") != std::string::npos &&
             name.find("Paravirtual") == std::string::npos,
             "physical Apple GPU required");
        queue = [device newCommandQueue];
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:
            [NSURL fileURLWithPath:@NUMI_PARTITIONED_REDUCTION_PROJECTION_METALLIB]
            error:&error];
        need(library != nil, "projection test metallib unavailable");
        const auto pipeline = [&](NSString* name) {
            NSString* qualified = [NSString stringWithFormat:
                @"numi_matter_metal::%@", name];
            const char* qualifiedUTF8 = [qualified UTF8String];
            if (qualifiedUTF8 == nullptr)
                throw std::runtime_error("projection kernel name is not UTF-8");
            id<MTLFunction> function = [library newFunctionWithName:qualified];
            if (function == nil)
                throw std::runtime_error(std::string("projection kernel missing: ") +
                                         qualifiedUTF8);
            error = nil;
            id<MTLComputePipelineState> result =
                [device newComputePipelineStateWithFunction:function error:&error];
            if (result == nil) {
                const char* description = error.localizedDescription != nil
                    ? [error.localizedDescription UTF8String] : nullptr;
                std::string message = std::string("projection pipeline failed: ") +
                    qualifiedUTF8;
                message += description != nullptr ? ": " + std::string(description)
                                                  : ": no NSError description";
                throw std::runtime_error(message);
            }
            std::fprintf(stderr, "projection_pso_ready=%s\n", qualifiedUTF8);
            return result;
        };
        copy = pipeline(@"nm_partitioned_reduction_copy_global_nodes");
        prolongate = pipeline(@"nm_partitioned_reduction_prolongate");
        reduceTiles = pipeline(@"nm_partitioned_reduction_reduce_tiles");
        reducePartials = pipeline(@"nm_partitioned_reduction_reduce_partials");
        invertNodeBlocks = pipeline(@"nm_partitioned_reduction_invert_node_blocks");
        reduceBlockTiles = pipeline(@"nm_partitioned_reduction_reduce_block_tiles");
        reduceBlockPartials = pipeline(@"nm_partitioned_reduction_reduce_block_partials");
        regionValidityTiles = pipeline(@"nm_partitioned_reduction_region_validity_tiles");
        regionValidityPartials = pipeline(@"nm_partitioned_reduction_region_validity_partials");
        factorRegionBlocks = pipeline(@"nm_partitioned_reduction_factor_region_blocks");
        solveRegionBlocks = pipeline(@"nm_partitioned_reduction_solve_region_blocks");
        addRegionCorrection = pipeline(@"nm_partitioned_reduction_add_region_correction");
        reduceSparseModeTiles = pipeline(@"nm_partitioned_reduction_reduce_sparse_mode_tiles");
        reduceSparseModePartials = pipeline(@"nm_partitioned_reduction_reduce_sparse_mode_partials");
        reduceSparseBlockTiles = pipeline(@"nm_partitioned_reduction_reduce_sparse_block_tiles");
        reduceSparseBlockPartials = pipeline(@"nm_partitioned_reduction_reduce_sparse_block_partials");
        regionSparseValidityTiles = pipeline(@"nm_partitioned_reduction_region_sparse_validity_tiles");
        regionSparseValidityPartials = pipeline(@"nm_partitioned_reduction_region_sparse_validity_partials");
        addSparseRegionCorrection = pipeline(@"nm_partitioned_reduction_add_sparse_region_correction");
        femTangentFDCheck = pipeline(@"nm_partitioned_reduction_fem_tangent_fd_check");
        hybridCopy = pipeline(@"nm_partitioned_reduction_hybrid_copy");
        hybridLift = pipeline(@"nm_partitioned_reduction_hybrid_lift");
        hybridZeroSelected = pipeline(@"nm_partitioned_reduction_hybrid_zero_selected");
        hybridPackModes = pipeline(@"nm_partitioned_reduction_hybrid_pack_modes");
        hybridClearInactiveField = pipeline(@"nm_partitioned_reduction_hybrid_clear_inactive_field");
        std::cout << "device=" << name
                  << " projection=matrix_free_partitioned_basis"
                  << " environments=2 fixed_rows=preserved anchors=one_hot\n";
    }

    void dispatchFine(const id<MTLComputePipelineState> pipeline,
                     const MTLSize grid,
                     const std::vector<id<MTLBuffer>>& buffers) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger index = 0u; index < buffers.size(); ++index)
            [encoder setBuffer:buffers[index] offset:0 atIndex:index];
        const MTLSize group = MTLSizeMake(256u, 1u, 1u);
        [encoder dispatchThreads:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        need(command.status == MTLCommandBufferStatusCompleted,
             "projection fine dispatch failed");
    }

    void dispatchGroups(const id<MTLComputePipelineState> pipeline,
                        const MTLSize grid,
                        const std::vector<id<MTLBuffer>>& buffers) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger index = 0u; index < buffers.size(); ++index)
            [encoder setBuffer:buffers[index] offset:0 atIndex:index];
        [encoder dispatchThreadgroups:grid
            threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        need(command.status == MTLCommandBufferStatusCompleted,
             "projection tiled block dispatch failed");
    }

    void dispatchReduction(const std::vector<id<MTLBuffer>>& buffers,
                           const std::uint32_t tileCount,
                           const std::uint32_t modeCount,
                           const std::uint32_t environmentCount) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        const MTLSize groupWidth = MTLSizeMake(256u, 1u, 1u);
        [encoder setComputePipelineState:reduceTiles];
        for (NSUInteger index = 0u; index < buffers.size(); ++index)
            [encoder setBuffer:buffers[index] offset:0 atIndex:index];
        [encoder dispatchThreadgroups:
            MTLSizeMake(tileCount, modeCount, environmentCount)
            threadsPerThreadgroup:groupWidth];
        [encoder setComputePipelineState:reducePartials];
        [encoder setBuffer:buffers[0] offset:0 atIndex:0];
        [encoder setBuffer:buffers[5] offset:0 atIndex:1];
        [encoder setBuffer:buffers[7] offset:0 atIndex:2];
        [encoder dispatchThreadgroups:
            MTLSizeMake(1u, modeCount, environmentCount)
            threadsPerThreadgroup:groupWidth];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        need(command.status == MTLCommandBufferStatusCompleted,
             "projection tiled reduction dispatch failed");
    }
};

std::vector<double> solveDenseReference(std::vector<double> matrix,
                                         std::vector<double> rhs,
                                         const std::size_t n) {
    need(matrix.size() == n * n && rhs.size() == n,
         "invalid dense reference solve dimensions");
    for (std::size_t column = 0u; column < n; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1u; row < n; ++row) {
            if (std::abs(matrix[row * n + column]) >
                std::abs(matrix[pivot * n + column])) pivot = row;
        }
        need(std::abs(matrix[pivot * n + column]) > 1.0e-12,
             "CPU regional reference block is singular");
        if (pivot != column) {
            for (std::size_t j = 0u; j < n; ++j)
                std::swap(matrix[column * n + j], matrix[pivot * n + j]);
            std::swap(rhs[column], rhs[pivot]);
        }
        for (std::size_t row = column + 1u; row < n; ++row) {
            const double factor = matrix[row * n + column] /
                matrix[column * n + column];
            matrix[row * n + column] = 0.0;
            for (std::size_t j = column + 1u; j < n; ++j)
                matrix[row * n + j] -= factor * matrix[column * n + j];
            rhs[row] -= factor * rhs[column];
        }
    }
    std::vector<double> solution(n, 0.0);
    for (std::size_t reverse = 0u; reverse < n; ++reverse) {
        const std::size_t row = n - 1u - reverse;
        double value = rhs[row];
        for (std::size_t column = row + 1u; column < n; ++column)
            value -= matrix[row * n + column] * solution[column];
        solution[row] = value / matrix[row * n + row];
    }
    return solution;
}

using Matrix3 = std::array<double, 9u>;

Matrix3 multiply3(const Matrix3& a, const Matrix3& b) {
    Matrix3 result{};
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            for (std::size_t k = 0; k < 3; ++k)
                result[r * 3 + c] += a[r * 3 + k] * b[k * 3 + c];
    return result;
}

Matrix3 transpose3(const Matrix3& a) {
    Matrix3 result{};
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            result[r * 3 + c] = a[c * 3 + r];
    return result;
}

Vec3 column3(const Matrix3& a, std::size_t c) {
    return {a[c], a[3 + c], a[6 + c]};
}

Vec3 subtract3(const Vec3& a, const Vec3& b) {
    return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}

Vec3 add3(const Vec3& a, const Vec3& b) {
    return {a[0]+b[0], a[1]+b[1], a[2]+b[2]};
}

Vec3 scale3(const Vec3& a, double s) {
    return {a[0]*s, a[1]*s, a[2]*s};
}

double dot3(const Vec3& a, const Vec3& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

std::array<Vec3, 4u> projectedTangentReference(
    const std::array<Vec3, 4u>& u, const Matrix3& inverseRest,
    const Matrix3& frame, const std::array<double, 81u>& tangent,
    double volume, double dt) {
    Matrix3 dp{};
    for (std::size_t r=0; r<3; ++r)
        for (std::size_t c=0; c<3; ++c)
            dp[r*3+c] = u[c+1][r] - u[0][r];
    const Matrix3 local = multiply3(multiply3(dp, inverseRest), frame);
    std::array<double, 9u> response{};
    for (std::size_t r=0; r<9; ++r)
        for (std::size_t c=0; c<9; ++c)
            response[r] += tangent[r*9+c] * local[c];
    Matrix3 stress{};
    std::copy(response.begin(), response.end(), stress.begin());
    Matrix3 h = multiply3(multiply3(stress, transpose3(frame)),
                          transpose3(inverseRest));
    for (double& v : h) v *= -volume * (-dt * dt);
    std::array<Vec3, 4u> result{};
    result[1] = column3(h, 0); result[2] = column3(h, 1);
    result[3] = column3(h, 2);
    result[0] = scale3(add3(add3(result[1], result[2]), result[3]), -1.0);
    return result;
}

void verifyProjectedFEMTangent(DeviceKernels& kernels) {
    constexpr std::uint32_t slotRows[4]{2u, 0u, 3u, 1u};
    constexpr std::uint32_t nodes[4]{83u, 89u, 97u, 103u};
    constexpr double dt=0.037, volume=0.0017, angle=0.43;
    NMTetrahedronGPU tet{};
    tet.nodes={nodes[2],nodes[0],nodes[3],nodes[1]};
    const Matrix3 inverse{1.7,.21,-.13, .08,1.35,.19, -.16,.11,1.52};
    tet.inverseRestRow0={float(inverse[0]),float(inverse[1]),float(inverse[2]),float(volume)};
    tet.inverseRestRow1={float(inverse[3]),float(inverse[4]),float(inverse[5]),0};
    tet.inverseRestRow2={float(inverse[6]),float(inverse[7]),float(inverse[8]),0};
    tet.identity={0u,0u,0u,NM_OBJECT_ACTIVE};
    tet.materialFrameRotation={0,0,float(std::sin(angle/2)),float(std::cos(angle/2))};
    NMContinuumObjectGPU object{}; object.flags=NM_OBJECT_FEM_MATERIAL_FRAME;
    const Matrix3 frame{std::cos(angle),-std::sin(angle),0,
                        std::sin(angle), std::cos(angle),0, 0,0,1};
    std::array<double,81u> lower{}, tangent{};
    std::array<float,81u> tangentF{};
    for (std::size_t r=0;r<9;++r)
        for (std::size_t c=0;c<=r;++c)
            lower[r*9+c] = r==c ? 4.0+.31*r :
                .045*std::sin(.7*(r+1)+.3*(c+1));
    for (std::size_t r=0;r<9;++r)
        for (std::size_t c=0;c<9;++c) {
            for (std::size_t k=0;k<9;++k)
                tangent[r*9+c] += lower[r*9+k]*lower[c*9+k];
            tangentF[r*9+c]=float(tangent[r*9+c]);
            tangent[r*9+c]=double(tangentF[r*9+c]);
        }
    need(std::abs(tangent[0]-tangent[80])>1 && std::abs(tangent[1])>1e-4,
         "tangent fixture must be anisotropic and coupled");
    const std::array<Vec3,4u> mode0{{
        Vec3{.12,-.08,.04},Vec3{-.03,.17,.09},
        Vec3{.11,.02,-.14},Vec3{-.09,-.05,.13}}};
    const std::array<Vec3,4u> mode1{{
        Vec3{-.07,.13,.10},Vec3{.16,-.04,.02},
        Vec3{-.02,-.12,.15},Vec3{.06,.08,-.11}}};
    std::vector<nm_float4> basis(8u);
    for (std::size_t s=0;s<4;++s) {
        const auto& a=mode0[s]; const auto& b=mode1[s];
        basis[slotRows[s]]={float(a[0]),float(a[1]),float(a[2]),0};
        basis[4+slotRows[s]]={float(b[0]),float(b[1]),float(b[2]),0};
    }
    const std::vector<std::uint32_t> globalRows(nodes,nodes+4);
    const std::vector<std::uint32_t> blockModes{0u,1u};
    const std::vector<NMTetrahedronGPU> oneTet{tet};
    const auto tangentSupport=buildPartitionedReductionFEMTangentSupport(
        std::span<const nm_float4>(basis),
        std::span<const std::uint32_t>(globalRows),4u,2u,
        std::span<const std::uint32_t>(blockModes),1u,2u,
        std::span<const NMTetrahedronGPU>(oneTet),0u);
    need(tangentSupport.valid() &&
         tangentSupport.regionTetrahedronIndices==std::vector<std::uint32_t>{0u} &&
         tangentSupport.tetrahedronBasisNodeRows==
             std::vector<std::uint32_t>{2u,0u,3u,1u},
         "FEM tangent support builder lost global-node offset or slot mapping");
    const std::vector<std::uint32_t> rows(
        tangentSupport.tetrahedronBasisNodeRows.begin(),
        tangentSupport.tetrahedronBasisNodeRows.begin()+4);
    const std::vector<float> dtValues{float(dt)};
    std::vector<float> output(4u,std::numeric_limits<float>::quiet_NaN());
    const auto tetBuffer=makeBuffer(kernels.device,std::vector<NMTetrahedronGPU>{tet});
    const auto objectBuffer=makeBuffer(kernels.device,std::vector<NMContinuumObjectGPU>{object});
    const auto tangentBuffer=makeBuffer(kernels.device,std::vector<float>(tangentF.begin(),tangentF.end()));
    const auto basisBuffer=makeBuffer(kernels.device,basis);
    const auto rowsBuffer=makeBuffer(kernels.device,rows);
    const auto dtBuffer=makeBuffer(kernels.device,dtValues);
    const auto outputBuffer=makeBuffer(kernels.device,output);
    kernels.dispatchFine(kernels.femTangentFDCheck,MTLSizeMake(4u,1u,1u),
        {tetBuffer,objectBuffer,tangentBuffer,basisBuffer,rowsBuffer,dtBuffer,outputBuffer});
    const auto* actual=static_cast<const float*>(outputBuffer.contents);
    for (std::size_t left=0;left<2;++left)
        for (std::size_t right=0;right<2;++right) {
            const auto& l=left==0?mode0:mode1;
            const auto& r=right==0?mode0:mode1;
            constexpr double epsilon=0.013;
            std::array<Vec3,4u> plus{}, minus{};
            for (std::size_t node=0;node<4;++node) {
                plus[node]=scale3(r[node],epsilon);
                minus[node]=scale3(r[node],-epsilon);
            }
            const auto fullPlus=projectedTangentReference(
                plus,inverse,frame,tangent,volume,dt);
            const auto fullMinus=projectedTangentReference(
                minus,inverse,frame,tangent,volume,dt);
            double expected=0;
            for (std::size_t node=0;node<4;++node) {
                std::array<Vec3,4u> onlyPlus{}, onlyMinus{};
                onlyPlus[node]=plus[node];
                onlyMinus[node]=minus[node];
                const auto selfPlus=projectedTangentReference(
                    onlyPlus,inverse,frame,tangent,volume,dt);
                const auto selfMinus=projectedTangentReference(
                    onlyMinus,inverse,frame,tangent,volume,dt);
                const Vec3 fullFD=scale3(
                    subtract3(fullPlus[node],fullMinus[node]),1.0/(2.0*epsilon));
                const Vec3 selfFD=scale3(
                    subtract3(selfPlus[node],selfMinus[node]),1.0/(2.0*epsilon));
                expected += dot3(l[node],subtract3(fullFD,selfFD));
            }
            const std::size_t index=left*2+right;
            need(std::isfinite(actual[index]) &&
                 close(actual[index],expected,4.0e-8,1.2e-3),
                 "projected FEM tangent differs from finite-difference reference");
        }
    std::cout << "projected_fem_tangent_fd=pass global_node_offset=83"
              << " anisotropic_material=1 rotated_frame=1 nonidentity_rest=1"
              << " offdiagonal_only=1\n";
}

void verify() {
    constexpr std::uint32_t environmentCount = 2u;
    const Fixture fixture = makeFixture();
    const PartitionedReductionBasis basis = buildBasis(fixture);
    const std::uint32_t globalNodeCount =
        basis.nodeCount + kGlobalNodeOffset + 12u;
    const std::uint32_t tileCount =
        (basis.nodeCount + NM_PARTITIONED_REDUCTION_TILE_WIDTH - 1u) /
        NM_PARTITIONED_REDUCTION_TILE_WIDTH;
    NMPartitionedReductionProjectionDispatchGPU dispatch{
        globalNodeCount, basis.nodeCount, basis.modeCount, tileCount,
        environmentCount, 0u, 0u, 0u};
    need(validPartitionedReductionProjectionDispatch(dispatch),
         "projection dispatch contract rejected valid fixture");

    std::vector<std::uint32_t> globalNodeIndices(basis.nodeCount);
    for (std::uint32_t row = 0u; row < basis.nodeCount; ++row)
        globalNodeIndices[row] = row + kGlobalNodeOffset;

    const std::size_t fineCount =
        static_cast<std::size_t>(environmentCount) * globalNodeCount;
    std::vector<nm_float4> fine(fineCount);
    for (std::uint32_t environment = 0u; environment < environmentCount; ++environment) {
        for (std::uint32_t nodeIndex = 0u; nodeIndex < globalNodeCount; ++nodeIndex) {
            const double n = static_cast<double>(nodeIndex + 1u);
            const double e = static_cast<double>(environment + 1u);
            fine[static_cast<std::size_t>(environment) * globalNodeCount + nodeIndex] = {
                static_cast<float>(0.13 * std::sin(0.17 * n + e)),
                static_cast<float>(0.11 * std::cos(0.09 * n - 0.3 * e)),
                static_cast<float>(0.07 * std::sin(0.03 * n * e)),
                static_cast<float>(1000.0 + 100.0 * e + n)};
        }
    }

    std::vector<float> q(static_cast<std::size_t>(environmentCount) * basis.modeCount);
    std::vector<double> qDouble(basis.modeCount);
    for (std::uint32_t environment = 0u; environment < environmentCount; ++environment) {
        for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
            const float value = static_cast<float>(
                0.01 * std::sin(0.37 * static_cast<double>(mode + 1u) +
                                0.61 * static_cast<double>(environment + 1u)));
            q[static_cast<std::size_t>(environment) * basis.modeCount + mode] = value;
            qDouble[mode] = static_cast<double>(value);
        }
    }

    std::vector<float> basisMass(basis.nodeCount);
    for (std::uint32_t row = 0u; row < basis.nodeCount; ++row)
        basisMass[row] = static_cast<float>(fixture.masses[row]);
    std::vector<nm_float4> projected(fineCount, nm_float4{});
    std::vector<float> reducedForce(
        static_cast<std::size_t>(environmentCount) * basis.modeCount,
        std::numeric_limits<float>::quiet_NaN());
    std::vector<float> reducedMass(reducedForce.size(),
                                   std::numeric_limits<float>::quiet_NaN());
    std::vector<float> partials(
        static_cast<std::size_t>(environmentCount) * basis.modeCount * tileCount,
        0.0f);

    DeviceKernels kernels;
    verifyProjectedFEMTangent(kernels);
    auto dispatchBuffer = makeBuffer(kernels.device,
                                     std::vector<NMPartitionedReductionProjectionDispatchGPU>{dispatch});
    auto sourceBuffer = makeBuffer(kernels.device, fine);
    auto projectedBuffer = makeBuffer(kernels.device, projected);
    auto indexBuffer = makeBuffer(kernels.device, globalNodeIndices);
    auto basisBuffer = makeBuffer(kernels.device, basis.values);
    auto qBuffer = makeBuffer(kernels.device, q);
    auto massBuffer = makeBuffer(kernels.device, basisMass);
    auto partialBuffer = makeBuffer(kernels.device, partials);
    auto forceOutputBuffer = makeBuffer(kernels.device, reducedForce);
    auto massOutputBuffer = makeBuffer(kernels.device, reducedMass);

    kernels.dispatchFine(kernels.copy,
        MTLSizeMake(static_cast<NSUInteger>(globalNodeCount) * environmentCount,
                    1u, 1u),
        {dispatchBuffer, sourceBuffer, projectedBuffer});
    kernels.dispatchFine(kernels.prolongate,
        MTLSizeMake(static_cast<NSUInteger>(basis.nodeCount) * environmentCount,
                    1u, 1u),
        {dispatchBuffer, sourceBuffer, projectedBuffer, indexBuffer,
         basisBuffer, qBuffer});

    const auto* gpuProjected =
        static_cast<const nm_float4*>(projectedBuffer.contents);
    for (std::uint32_t environment = 0u; environment < environmentCount; ++environment) {
        for (std::uint32_t row = 0u; row < basis.nodeCount; ++row) {
            const std::uint32_t global = globalNodeIndices[row];
            const auto& actual = gpuProjected[
                static_cast<std::size_t>(environment) * globalNodeCount + global];
            std::fill(qDouble.begin(), qDouble.end(), 0.0);
            for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode)
                qDouble[mode] = q[static_cast<std::size_t>(environment) *
                                   basis.modeCount + mode];
            std::vector<std::array<double, 3u>> cpuDisplacement(basis.nodeCount);
            need(combinePartitionedPreconditionerModes(
                     basis, qDouble, cpuDisplacement),
                 "CPU prolongation reference rejected provider basis");
            need(close(actual.x, cpuDisplacement[row][0]) &&
                 close(actual.y, cpuDisplacement[row][1]) &&
                 close(actual.z, cpuDisplacement[row][2]),
                 "GPU prolongation differs from the CPU Bq reference");
            need(actual.w == fine[static_cast<std::size_t>(environment) *
                                    globalNodeCount + global].w,
                 "prolongation changed the preserved fourth component");
        }
        for (std::uint32_t nodeIndex = 0u; nodeIndex < globalNodeCount; ++nodeIndex) {
            const bool selected = nodeIndex >= kGlobalNodeOffset &&
                nodeIndex < kGlobalNodeOffset + basis.nodeCount;
            if (selected) continue;
            const auto& actual = gpuProjected[
                static_cast<std::size_t>(environment) * globalNodeCount + nodeIndex];
            const auto& original = fine[
                static_cast<std::size_t>(environment) * globalNodeCount + nodeIndex];
            need(actual.x == original.x && actual.y == original.y &&
                 actual.z == original.z && actual.w == original.w,
                 "copy/prolongation changed a nonselected complement row");
        }

        double fineWork = 0.0;
        double reducedWork = 0.0;
        for (std::uint32_t row = 0u; row < basis.nodeCount; ++row) {
            const std::uint32_t global = globalNodeIndices[row];
            const auto& u = gpuProjected[
                static_cast<std::size_t>(environment) * globalNodeCount + global];
            const auto& f = fine[
                static_cast<std::size_t>(environment) * globalNodeCount + global];
            fineWork += static_cast<double>(u.x) * f.x +
                static_cast<double>(u.y) * f.y +
                static_cast<double>(u.z) * f.z;
        }
        // GPU reduced force is filled below; CPU duality gives a precise
        // invariant independent of the implementation's reduction tree.
        std::vector<std::array<double, 3u>> localForce(basis.nodeCount);
        for (std::uint32_t row = 0u; row < basis.nodeCount; ++row) {
            const auto& f = fine[
                static_cast<std::size_t>(environment) * globalNodeCount +
                globalNodeIndices[row]];
            localForce[row] = {f.x, f.y, f.z};
        }
        std::vector<double> cpuForce(basis.modeCount, 0.0);
        need(projectPartitionedNodalForce(basis, localForce, cpuForce),
             "CPU B^T f reference rejected provider basis");
        for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode)
            reducedWork += q[static_cast<std::size_t>(environment) *
                             basis.modeCount + mode] * cpuForce[mode];
        need(close(fineWork, reducedWork, 4.0e-7, 1.0e-4),
             "CPU work conservation <Bq,f>=<q,B^T f> failed");
    }

    const auto encodeReduction = [&](const id<MTLBuffer> valueBuffer,
                                     const id<MTLBuffer> massOrNil,
                                     const id<MTLBuffer> outputBuffer) {
        std::vector<id<MTLBuffer>> buffers{
            dispatchBuffer, valueBuffer, basisBuffer, indexBuffer,
            massOrNil, partialBuffer, outputBuffer, outputBuffer};
        kernels.dispatchReduction(buffers, tileCount, basis.modeCount,
                                  environmentCount);
    };
    // The reducer uses buffer 7 as the final output; bind a dummy at 6.
    encodeReduction(sourceBuffer, nil, forceOutputBuffer);
    encodeReduction(projectedBuffer, massBuffer, massOutputBuffer);

    const auto* gpuForce =
        static_cast<const float*>(forceOutputBuffer.contents);
    const auto* gpuMass =
        static_cast<const float*>(massOutputBuffer.contents);
    for (std::uint32_t environment = 0u; environment < environmentCount; ++environment) {
        std::vector<std::array<double, 3u>> localForce(basis.nodeCount);
        std::vector<double> cpuForce(basis.modeCount, 0.0);
        for (std::uint32_t row = 0u; row < basis.nodeCount; ++row) {
            const auto& f = fine[
                static_cast<std::size_t>(environment) * globalNodeCount +
                globalNodeIndices[row]];
            localForce[row] = {f.x, f.y, f.z};
        }
        need(projectPartitionedNodalForce(basis, localForce, cpuForce),
             "CPU force projection reference failed");
        std::fill(qDouble.begin(), qDouble.end(), 0.0);
        for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
            const std::size_t out =
                static_cast<std::size_t>(environment) * basis.modeCount + mode;
            need(std::isfinite(gpuForce[out]) &&
                 close(gpuForce[out], cpuForce[mode], 4.0e-6, 8.0e-4),
                 "GPU B^T f differs from CPU provider basis reference");
            qDouble[mode] = q[out];
        }

        double gpuWork = 0.0;
        for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode)
            gpuWork += qDouble[mode] * gpuForce[
                static_cast<std::size_t>(environment) * basis.modeCount + mode];
        double projectedWork = 0.0;
        for (std::uint32_t row = 0u; row < basis.nodeCount; ++row) {
            const auto& u = gpuProjected[
                static_cast<std::size_t>(environment) * globalNodeCount +
                globalNodeIndices[row]];
            const auto& f = fine[
                static_cast<std::size_t>(environment) * globalNodeCount +
                globalNodeIndices[row]];
            projectedWork += static_cast<double>(u.x) * f.x +
                static_cast<double>(u.y) * f.y +
                static_cast<double>(u.z) * f.z;
        }
        need(close(projectedWork, gpuWork, 3.0e-6, 8.0e-4),
             "GPU work conservation <Bq,f>=<q,B^T f> failed");

        for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
            double cpuMassAction = 0.0;
            for (std::uint32_t column = 0u; column < basis.modeCount; ++column)
                cpuMassAction += basis.projectedMass[
                    static_cast<std::size_t>(mode) * basis.modeCount + column] *
                    qDouble[column];
            const std::size_t out =
                static_cast<std::size_t>(environment) * basis.modeCount + mode;
            need(std::isfinite(gpuMass[out]) &&
                 close(gpuMass[out], cpuMassAction, 6.0e-6, 1.2e-3),
                 "GPU B^T M Bq differs from CPU projectedMass action");
        }
    }

    // Exercise the additive regional coarse bridge with diagonal D=diag(2,3,4).
    // The source inverse rows carry diag(1/2,1/3,1/4), matching the production
    // buffer contract while keeping the CPU Kc reference independently clear.
    constexpr std::uint32_t blockModeCount = 6u;
    const std::uint32_t regionCount = static_cast<std::uint32_t>(basis.regions.size());
    need(regionCount == 2u, "GPU block fixture must contain two anatomical regions");
    std::vector<std::uint32_t> blockModeIndices(
        static_cast<std::size_t>(regionCount) * blockModeCount, NM_INVALID_INDEX);
    std::vector<std::uint32_t> modesSeen(regionCount, 0u);
    for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
        const auto& descriptor = basis.modes[mode];
        if (descriptor.kind == PartitionedReductionModeKind::anchorTranslation)
            continue;
        const auto region = std::find_if(basis.regions.begin(), basis.regions.end(),
            [&](const auto& item) { return item.sourceLabel == descriptor.regionLabel; });
        need(region != basis.regions.end(), "regional mode has no retained region label");
        const std::uint32_t regionIndex = static_cast<std::uint32_t>(
            region - basis.regions.begin());
        const std::uint32_t localMode = modesSeen[regionIndex]++;
        need(localMode < blockModeCount, "regional mode count exceeds block fixture width");
        blockModeIndices[static_cast<std::size_t>(regionIndex) * blockModeCount +
                          localMode] = mode;
    }
    for (const auto count : modesSeen)
        need(count == blockModeCount, "regional block fixture omitted a rigid mode");
    NMPartitionedReductionBlockDispatchGPU blockDispatch{
        globalNodeCount, basis.nodeCount, basis.modeCount, tileCount,
        environmentCount, regionCount, blockModeCount, 0u};
    need(validPartitionedReductionBlockDispatch(blockDispatch),
         "block dispatch contract rejected a valid fixture");

    const std::size_t blockCount =
        static_cast<std::size_t>(environmentCount) * regionCount;
    const std::size_t blockPairCount = blockModeCount * blockModeCount;
    std::vector<nm_float4> inverseRows(fineCount * 3u, nm_float4{});
    for (std::uint32_t environment = 0u; environment < environmentCount; ++environment) {
        for (std::uint32_t nodeIndex = 0u; nodeIndex < globalNodeCount; ++nodeIndex) {
            const std::size_t base =
                (static_cast<std::size_t>(environment) * globalNodeCount + nodeIndex) * 3u;
            inverseRows[base] = {0.5f, 0.0f, 0.0f, 1.0f};
            inverseRows[base + 1u] = {0.0f, 1.0f / 3.0f, 0.0f, 0.0f};
            inverseRows[base + 2u] = {0.0f, 0.0f, 0.25f, 0.0f};
        }
    }
    std::vector<nm_float4> selectedD(
        static_cast<std::size_t>(environmentCount) * basis.nodeCount * 3u,
        nm_float4{});
    std::vector<std::uint32_t> rowValid(
        static_cast<std::size_t>(environmentCount) * basis.nodeCount, 0u);
    std::vector<float> blockPartials(blockCount * blockPairCount * tileCount, 0.0f);
    std::vector<float> regionPartials(blockCount * tileCount, 0.0f);
    std::vector<std::uint32_t> regionValidity(blockCount, 0u);
    std::vector<float> reducedBlocks(blockCount * blockPairCount, 0.0f);
    std::vector<std::uint32_t> blockPivots(blockCount * blockModeCount, 0u);
    std::vector<std::uint32_t> factorValid(blockCount, 0u);
    std::vector<float> reducedCoefficients(blockCount * blockModeCount, 0.0f);
    std::vector<std::uint32_t> solveValid(blockCount, 0u);
    std::vector<nm_float4> preconditioned(fine.begin(), fine.end());

    auto blockDispatchBuffer = makeBuffer(kernels.device,
        std::vector<NMPartitionedReductionBlockDispatchGPU>{blockDispatch});
    auto blockModeBuffer = makeBuffer(kernels.device, blockModeIndices);
    auto inverseBuffer = makeBuffer(kernels.device, inverseRows);
    auto selectedDBuffer = makeBuffer(kernels.device, selectedD);
    auto rowValidBuffer = makeBuffer(kernels.device, rowValid);
    auto blockPartialBuffer = makeBuffer(kernels.device, blockPartials);
    auto regionPartialBuffer = makeBuffer(kernels.device, regionPartials);
    auto regionValidityBuffer = makeBuffer(kernels.device, regionValidity);
    auto reducedBlockBuffer = makeBuffer(kernels.device, reducedBlocks);
    auto pivotBuffer = makeBuffer(kernels.device, blockPivots);
    auto factorValidBuffer = makeBuffer(kernels.device, factorValid);
    auto coefficientBuffer = makeBuffer(kernels.device, reducedCoefficients);
    auto solveValidBuffer = makeBuffer(kernels.device, solveValid);
    auto preconditionedBuffer = makeBuffer(kernels.device, preconditioned);

    kernels.dispatchFine(kernels.invertNodeBlocks,
        MTLSizeMake(static_cast<NSUInteger>(basis.nodeCount) * environmentCount, 1u, 1u),
        {blockDispatchBuffer, inverseBuffer, indexBuffer, selectedDBuffer, rowValidBuffer});
    kernels.dispatchGroups(kernels.regionValidityTiles,
        MTLSizeMake(tileCount, regionCount * environmentCount, 1u),
        {blockDispatchBuffer, basisBuffer, blockModeBuffer, rowValidBuffer, regionPartialBuffer});
    kernels.dispatchGroups(kernels.regionValidityPartials,
        MTLSizeMake(1u, regionCount * environmentCount, 1u),
        {blockDispatchBuffer, regionPartialBuffer, regionValidityBuffer});
    kernels.dispatchGroups(kernels.reduceBlockTiles,
        MTLSizeMake(tileCount, static_cast<NSUInteger>(blockPairCount),
                    static_cast<NSUInteger>(regionCount) * environmentCount),
        {blockDispatchBuffer, selectedDBuffer, basisBuffer, blockModeBuffer,
         blockPartialBuffer});
    kernels.dispatchGroups(kernels.reduceBlockPartials,
        MTLSizeMake(1u, static_cast<NSUInteger>(blockPairCount),
                    static_cast<NSUInteger>(regionCount) * environmentCount),
        {blockDispatchBuffer, blockPartialBuffer, reducedBlockBuffer});

    const auto* gpuSelectedD = static_cast<const nm_float4*>(selectedDBuffer.contents);
    const auto* gpuRowValid = static_cast<const std::uint32_t*>(rowValidBuffer.contents);
    const auto* gpuRegionValidity =
        static_cast<const std::uint32_t*>(regionValidityBuffer.contents);
    const auto* gpuReducedBlocks = static_cast<const float*>(reducedBlockBuffer.contents);
    const std::vector<float> denseReducedBlockReference(
        gpuReducedBlocks, gpuReducedBlocks + blockCount * blockPairCount);
    std::vector<double> cpuBlockMatrices(blockCount * blockPairCount, 0.0);
    for (std::uint32_t environment = 0u; environment < environmentCount; ++environment) {
        for (std::uint32_t row = 0u; row < basis.nodeCount; ++row) {
            const std::size_t selectedNode =
                static_cast<std::size_t>(environment) * basis.nodeCount + row;
            need(gpuRowValid[selectedNode] == 1u, "GPU rejected a valid nodal block");
            const nm_float4* d = gpuSelectedD + selectedNode * 3u;
            need(close(d[0].x, 2.0) && close(d[1].y, 3.0) && close(d[2].z, 4.0) &&
                 close(d[0].y, 0.0) && close(d[0].z, 0.0) &&
                 close(d[1].x, 0.0) && close(d[1].z, 0.0) &&
                 close(d[2].x, 0.0) && close(d[2].y, 0.0),
                 "GPU inverse-block recovery differs from CPU diagonal reference");
        }
        for (std::uint32_t regionIndex = 0u; regionIndex < regionCount; ++regionIndex) {
            const std::size_t blockIndex =
                static_cast<std::size_t>(environment) * regionCount + regionIndex;
            need(gpuRegionValidity[blockIndex] == 1u,
                 "GPU marked an all-valid regional block unavailable");
            for (std::uint32_t rowMode = 0u; rowMode < blockModeCount; ++rowMode) {
                const std::uint32_t globalRowMode = blockModeIndices[
                    static_cast<std::size_t>(regionIndex) * blockModeCount + rowMode];
                for (std::uint32_t columnMode = 0u; columnMode < blockModeCount; ++columnMode) {
                    const std::uint32_t globalColumnMode = blockModeIndices[
                        static_cast<std::size_t>(regionIndex) * blockModeCount + columnMode];
                    double expectedBlock = 0.0;
                    for (std::uint32_t row = 0u; row < basis.nodeCount; ++row) {
                        const auto& bRow = basis.values[
                            static_cast<std::size_t>(globalRowMode) * basis.nodeCount + row];
                        const auto& bColumn = basis.values[
                            static_cast<std::size_t>(globalColumnMode) * basis.nodeCount + row];
                        expectedBlock += 2.0 * bRow.x * bColumn.x +
                            3.0 * bRow.y * bColumn.y + 4.0 * bRow.z * bColumn.z;
                    }
                    const std::size_t pair = rowMode * blockModeCount + columnMode;
                    const std::size_t output = blockIndex * blockPairCount + pair;
                    cpuBlockMatrices[output] = expectedBlock;
                    need(close(gpuReducedBlocks[output], expectedBlock, 8.0e-6, 1.0e-3),
                         "GPU regional B^T D B block differs from FP64 CPU reference");
                }
            }
        }
    }

    kernels.dispatchFine(kernels.factorRegionBlocks,
        MTLSizeMake(blockCount, 1u, 1u),
        {blockDispatchBuffer, reducedBlockBuffer, regionValidityBuffer,
         pivotBuffer, factorValidBuffer});
    kernels.dispatchFine(kernels.solveRegionBlocks,
        MTLSizeMake(blockCount, 1u, 1u),
        {blockDispatchBuffer, forceOutputBuffer, blockModeBuffer,
         reducedBlockBuffer, pivotBuffer, regionValidityBuffer, factorValidBuffer,
         coefficientBuffer, solveValidBuffer});
    const auto* gpuFactorValid = static_cast<const std::uint32_t*>(factorValidBuffer.contents);
    const auto* gpuSolveValid = static_cast<const std::uint32_t*>(solveValidBuffer.contents);
    const auto* gpuCoefficients = static_cast<const float*>(coefficientBuffer.contents);
    for (std::uint32_t environment = 0u; environment < environmentCount; ++environment) {
        for (std::uint32_t regionIndex = 0u; regionIndex < regionCount; ++regionIndex) {
            const std::size_t blockIndex =
                static_cast<std::size_t>(environment) * regionCount + regionIndex;
            need(gpuFactorValid[blockIndex] == 1u && gpuSolveValid[blockIndex] == 1u,
                 "GPU regional block factor/solve unexpectedly disabled an admitted block");
            std::vector<double> rhs(blockModeCount);
            for (std::uint32_t localMode = 0u; localMode < blockModeCount; ++localMode) {
                const std::uint32_t globalMode = blockModeIndices[
                    static_cast<std::size_t>(regionIndex) * blockModeCount + localMode];
                rhs[localMode] = gpuForce[
                    static_cast<std::size_t>(environment) * basis.modeCount + globalMode];
            }
            const std::size_t matrixOffset = blockIndex * blockPairCount;
            const std::vector<double> cpuCoefficients = solveDenseReference(
                std::vector<double>(cpuBlockMatrices.begin() + matrixOffset,
                                    cpuBlockMatrices.begin() + matrixOffset + blockPairCount),
                rhs, blockModeCount);
            for (std::uint32_t localMode = 0u; localMode < blockModeCount; ++localMode) {
                const double gpuValue = gpuCoefficients[
                    blockIndex * blockModeCount + localMode];
                need(close(gpuValue, cpuCoefficients[localMode], 1.0e-5, 2.0e-3),
                     "GPU regional coarse solve differs from CPU FP64 reference");
            }
        }
    }

    kernels.dispatchFine(kernels.addRegionCorrection,
        MTLSizeMake(static_cast<NSUInteger>(basis.nodeCount) * environmentCount,
                    1u, 1u),
        {blockDispatchBuffer, indexBuffer, basisBuffer, blockModeBuffer,
         coefficientBuffer, solveValidBuffer, preconditionedBuffer});
    const auto* gpuPreconditioned =
        static_cast<const nm_float4*>(preconditionedBuffer.contents);
    for (std::uint32_t environment = 0u; environment < environmentCount; ++environment) {
        for (std::uint32_t globalNode = 0u; globalNode < globalNodeCount; ++globalNode) {
            nm_float4 expectedValue = fine[
                static_cast<std::size_t>(environment) * globalNodeCount + globalNode];
            if (globalNode >= kGlobalNodeOffset &&
                globalNode < kGlobalNodeOffset + basis.nodeCount) {
                const std::uint32_t row = globalNode - kGlobalNodeOffset;
                float correctionX = 0.0f;
                float correctionY = 0.0f;
                float correctionZ = 0.0f;
                for (std::uint32_t regionIndex = 0u; regionIndex < regionCount; ++regionIndex) {
                    const std::size_t blockIndex =
                        static_cast<std::size_t>(environment) * regionCount + regionIndex;
                    for (std::uint32_t localMode = 0u; localMode < blockModeCount; ++localMode) {
                        const std::uint32_t globalMode = blockModeIndices[
                            static_cast<std::size_t>(regionIndex) * blockModeCount + localMode];
                        const auto& b = basis.values[
                            static_cast<std::size_t>(globalMode) * basis.nodeCount + row];
                        const float coefficient = gpuCoefficients[
                            blockIndex * blockModeCount + localMode];
                        correctionX += b.x * coefficient;
                        correctionY += b.y * coefficient;
                        correctionZ += b.z * coefficient;
                    }
                }
                expectedValue.x += correctionX;
                expectedValue.y += correctionY;
                expectedValue.z += correctionZ;
            }
            const auto& actual = gpuPreconditioned[
                static_cast<std::size_t>(environment) * globalNodeCount + globalNode];
            need(close(actual.x, expectedValue.x, 4.0e-6, 1.0e-3) &&
                 close(actual.y, expectedValue.y, 4.0e-6, 1.0e-3) &&
                 close(actual.z, expectedValue.z, 4.0e-6, 1.0e-3) &&
                 actual.w == expectedValue.w,
                 "GPU additive regional correction changed xyz or packed w incorrectly");
        }
    }

    // Repeat the regional bridge using exact sparse support lists derived from
    // the same exported FP32 B. Compare against dense Metal outputs and the
    // independent FP64 Kc/solve reference above.
    const auto sparseSupport = buildPartitionedReductionSparseSupport(
        basis, std::span<const std::uint32_t>(blockModeIndices),
        regionCount, blockModeCount);
    need(sparseSupport.valid(), "CPU sparse support failed: " + sparseSupport.error);
    need(sparseSupport.sparseTileCount > 0u &&
         sparseSupport.regionSupportRanges.size() == regionCount &&
         sparseSupport.nodeModeRanges.size() == basis.nodeCount,
         "CPU sparse support dimensions are invalid");
    NMPartitionedReductionBlockDispatchGPU sparseBlockDispatch = blockDispatch;
    sparseBlockDispatch.reserved = sparseSupport.sparseTileCount;
    need(validPartitionedReductionBlockDispatch(sparseBlockDispatch),
         "sparse block dispatch rejected compatible dense dimensions");

    auto sparseBlockDispatchBuffer = makeBuffer(kernels.device,
        std::vector<NMPartitionedReductionBlockDispatchGPU>{sparseBlockDispatch});
    auto regionSupportRangeBuffer = makeBuffer(kernels.device,
        sparseSupport.regionSupportRanges);
    auto regionSupportRowsBuffer = makeBuffer(kernels.device,
        sparseSupport.regionSupportRows);
    auto nodeModeRangeBuffer = makeBuffer(kernels.device,
        sparseSupport.nodeModeRanges);
    auto nodeModeSlotsBuffer = makeBuffer(kernels.device,
        sparseSupport.nodeModeSlots);

    std::vector<float> sparseModePartials(
        blockCount * blockModeCount * sparseSupport.sparseTileCount, 0.0f);
    std::vector<float> sparseModeValues(
        static_cast<std::size_t>(environmentCount) * basis.modeCount,
        std::numeric_limits<float>::quiet_NaN());
    auto sparseModePartialBuffer = makeBuffer(kernels.device, sparseModePartials);
    auto sparseModeValueBuffer = makeBuffer(kernels.device, sparseModeValues);
    kernels.dispatchGroups(kernels.reduceSparseModeTiles,
        MTLSizeMake(sparseSupport.sparseTileCount, blockModeCount,
                    static_cast<NSUInteger>(regionCount) * environmentCount),
        {sparseBlockDispatchBuffer, sourceBuffer, basisBuffer, indexBuffer, nil,
         sparseModePartialBuffer, blockModeBuffer,
         regionSupportRangeBuffer, regionSupportRowsBuffer});
    kernels.dispatchGroups(kernels.reduceSparseModePartials,
        MTLSizeMake(1u, blockModeCount,
                    static_cast<NSUInteger>(regionCount) * environmentCount),
        {sparseBlockDispatchBuffer, sparseModePartialBuffer,
         sparseModeValueBuffer, blockModeBuffer});
    const auto* gpuSparseModeValues =
        static_cast<const float*>(sparseModeValueBuffer.contents);
    for (std::uint32_t environment = 0u; environment < environmentCount; ++environment) {
        for (std::uint32_t region = 0u; region < regionCount; ++region) {
            for (std::uint32_t localMode = 0u; localMode < blockModeCount; ++localMode) {
                const std::uint32_t mode = blockModeIndices[
                    static_cast<std::size_t>(region) * blockModeCount + localMode];
                const std::size_t output =
                    static_cast<std::size_t>(environment) * basis.modeCount + mode;
                need(std::isfinite(gpuSparseModeValues[output]) &&
                     close(gpuSparseModeValues[output], gpuForce[output],
                           5.0e-6, 1.0e-3),
                     "sparse GPU B^T f differs from the dense GPU reference");
            }
        }
    }

    std::vector<float> sparseBlockPartials(
        blockCount * blockPairCount * sparseSupport.sparseTileCount, 0.0f);
    std::vector<float> sparseReducedBlocks(blockCount * blockPairCount, 0.0f);
    std::vector<std::uint32_t> sparseValidityPartials(
        blockCount * sparseSupport.sparseTileCount, 0u);
    std::vector<std::uint32_t> sparseRegionValidity(blockCount, 0u);
    auto sparseBlockPartialBuffer = makeBuffer(kernels.device, sparseBlockPartials);
    auto sparseReducedBlockBuffer = makeBuffer(kernels.device, sparseReducedBlocks);
    auto sparseValidityPartialBuffer = makeBuffer(kernels.device, sparseValidityPartials);
    auto sparseRegionValidityBuffer = makeBuffer(kernels.device, sparseRegionValidity);

    kernels.dispatchGroups(kernels.regionSparseValidityTiles,
        MTLSizeMake(sparseSupport.sparseTileCount,
                    static_cast<NSUInteger>(regionCount) * environmentCount, 1u),
        {sparseBlockDispatchBuffer, rowValidBuffer, regionSupportRangeBuffer,
         regionSupportRowsBuffer, sparseValidityPartialBuffer});
    kernels.dispatchGroups(kernels.regionSparseValidityPartials,
        MTLSizeMake(1u, static_cast<NSUInteger>(regionCount) * environmentCount, 1u),
        {sparseBlockDispatchBuffer, sparseValidityPartialBuffer,
         sparseRegionValidityBuffer});
    kernels.dispatchGroups(kernels.reduceSparseBlockTiles,
        MTLSizeMake(sparseSupport.sparseTileCount,
                    static_cast<NSUInteger>(blockPairCount),
                    static_cast<NSUInteger>(regionCount) * environmentCount),
        {sparseBlockDispatchBuffer, selectedDBuffer, basisBuffer, blockModeBuffer,
         sparseBlockPartialBuffer, regionSupportRangeBuffer,
         regionSupportRowsBuffer});
    kernels.dispatchGroups(kernels.reduceSparseBlockPartials,
        MTLSizeMake(1u, static_cast<NSUInteger>(blockPairCount),
                    static_cast<NSUInteger>(regionCount) * environmentCount),
        {sparseBlockDispatchBuffer, sparseBlockPartialBuffer,
         sparseReducedBlockBuffer});

    const auto* gpuSparseReducedBlocks =
        static_cast<const float*>(sparseReducedBlockBuffer.contents);
    const auto* gpuSparseRegionValidity =
        static_cast<const std::uint32_t*>(sparseRegionValidityBuffer.contents);
    for (std::size_t block = 0u; block < blockCount; ++block) {
        need(gpuSparseRegionValidity[block] == gpuRegionValidity[block],
             "sparse regional validity differs from the dense reference");
        for (std::size_t pair = 0u; pair < blockPairCount; ++pair) {
            const std::size_t index = block * blockPairCount + pair;
            if (!close(gpuSparseReducedBlocks[index], denseReducedBlockReference[index],
                       8.0e-6, 1.2e-3)) {
                const std::size_t blockPair = index % blockPairCount;
                const std::size_t blockIndex = index / blockPairCount;
                const std::uint32_t environment = static_cast<std::uint32_t>(
                    blockIndex / regionCount);
                const std::uint32_t region = static_cast<std::uint32_t>(
                    blockIndex % regionCount);
                const auto& support = sparseSupport.regionSupportRanges[region];
                std::fprintf(stderr,
                    "sparse_kc_mismatch env=%u region=%u local_row=%zu local_col=%zu "
                    "dense=%.9g sparse=%.9g cpu=%.17g diff=%.9g support_first=%u "
                    "support_count=%u sparse_tiles=%u dense_tiles=%u\n",
                    environment, region,
                    blockPair / blockModeCount, blockPair % blockModeCount,
                    denseReducedBlockReference[index], gpuSparseReducedBlocks[index],
                    cpuBlockMatrices[index],
                    gpuSparseReducedBlocks[index] - denseReducedBlockReference[index],
                    support.first, support.count, sparseSupport.sparseTileCount,
                    blockDispatch.tileCount);
            }
            need(close(gpuSparseReducedBlocks[index], denseReducedBlockReference[index],
                       8.0e-6, 1.2e-3),
                 "sparse GPU B^T D B differs from the dense GPU reference");
            need(close(gpuSparseReducedBlocks[index], cpuBlockMatrices[index],
                       8.0e-6, 1.2e-3),
                 "sparse GPU B^T D B differs from the FP64 CPU reference");
        }
    }

    std::vector<std::uint32_t> sparsePivots(blockCount * blockModeCount, 0u);
    std::vector<std::uint32_t> sparseFactorValid(blockCount, 0u);
    std::vector<float> sparseCoefficients(blockCount * blockModeCount, 0.0f);
    std::vector<std::uint32_t> sparseSolveValid(blockCount, 0u);
    auto sparsePivotBuffer = makeBuffer(kernels.device, sparsePivots);
    auto sparseFactorValidBuffer = makeBuffer(kernels.device, sparseFactorValid);
    auto sparseCoefficientBuffer = makeBuffer(kernels.device, sparseCoefficients);
    auto sparseSolveValidBuffer = makeBuffer(kernels.device, sparseSolveValid);
    kernels.dispatchFine(kernels.factorRegionBlocks,
        MTLSizeMake(blockCount, 1u, 1u),
        {sparseBlockDispatchBuffer, sparseReducedBlockBuffer,
         sparseRegionValidityBuffer, sparsePivotBuffer, sparseFactorValidBuffer});
    kernels.dispatchFine(kernels.solveRegionBlocks,
        MTLSizeMake(blockCount, 1u, 1u),
        {sparseBlockDispatchBuffer, sparseModeValueBuffer, blockModeBuffer,
         sparseReducedBlockBuffer, sparsePivotBuffer, sparseRegionValidityBuffer,
         sparseFactorValidBuffer, sparseCoefficientBuffer, sparseSolveValidBuffer});
    const auto* gpuSparseFactorValid =
        static_cast<const std::uint32_t*>(sparseFactorValidBuffer.contents);
    const auto* gpuSparseSolveValid =
        static_cast<const std::uint32_t*>(sparseSolveValidBuffer.contents);
    const auto* gpuSparseCoefficients =
        static_cast<const float*>(sparseCoefficientBuffer.contents);
    for (std::size_t block = 0u; block < blockCount; ++block) {
        need(gpuSparseFactorValid[block] == gpuFactorValid[block] &&
             gpuSparseSolveValid[block] == gpuSolveValid[block],
             "sparse regional factor/solve admission differs from dense path");
        for (std::uint32_t localMode = 0u; localMode < blockModeCount; ++localMode) {
            const std::size_t index = block * blockModeCount + localMode;
            need(close(gpuSparseCoefficients[index], gpuCoefficients[index],
                       2.0e-5, 3.0e-3),
                 "sparse regional coarse solve differs from dense GPU reference");
        }
    }

    std::vector<nm_float4> sparsePreconditioned(fine.begin(), fine.end());
    auto sparsePreconditionedBuffer = makeBuffer(kernels.device, sparsePreconditioned);
    kernels.dispatchFine(kernels.addSparseRegionCorrection,
        MTLSizeMake(static_cast<NSUInteger>(basis.nodeCount) * environmentCount,
                    1u, 1u),
        {sparseBlockDispatchBuffer, indexBuffer, basisBuffer, blockModeBuffer,
         sparseCoefficientBuffer, sparseSolveValidBuffer, sparsePreconditionedBuffer,
         nodeModeRangeBuffer, nodeModeSlotsBuffer});
    const auto* gpuSparsePreconditioned =
        static_cast<const nm_float4*>(sparsePreconditionedBuffer.contents);
    for (std::size_t index = 0u; index < fineCount; ++index) {
        const auto& denseValue = gpuPreconditioned[index];
        const auto& sparseValue = gpuSparsePreconditioned[index];
        need(close(sparseValue.x, denseValue.x, 7.0e-6, 1.5e-3) &&
             close(sparseValue.y, denseValue.y, 7.0e-6, 1.5e-3) &&
             close(sparseValue.z, denseValue.z, 7.0e-6, 1.5e-3) &&
             sparseValue.w == denseValue.w,
             "sparse additive correction differs from dense GPU result or changes w");
    }

    // The compiled Human source carries regional modes only. Sensitive anchors
    // are kept as fine complement rows, so build the hybrid fixture with its
    // anchor node fixed in B (the full generic basis above still tests its
    // explicit one-hot anchor modes independently).
    Fixture hybridFixture = fixture;
    for (const std::uint32_t anchor : hybridFixture.anchors)
        hybridFixture.fixed[anchor] = 1u;
    PartitionedReductionBasisInput hybridBasisInput{
        std::span<const Vec3>(hybridFixture.positions),
        std::span<const double>(hybridFixture.masses),
        std::span<const std::uint8_t>(hybridFixture.fixed),
        std::span<const Tet>(hybridFixture.tetrahedra),
        std::span<const std::uint32_t>(hybridFixture.labels),
        std::span<const std::uint32_t>(hybridFixture.anchors)};
    PartitionedReductionBasisOptions hybridBasisOptions;
    hybridBasisOptions.minimumCoordinateReductionFactor = 4u;
    hybridBasisOptions.maximumModeCount = 32u;
    const PartitionedReductionBasis hybridBasis =
        buildPartitionedReductionBasis(hybridBasisInput, hybridBasisOptions);
    need(hybridBasis.valid(),
         "CPU hybrid basis fixture failed: " + hybridBasis.error);
    need(hybridBasis.modeCount == blockModeIndices.size(),
         "hybrid fixture must contain regional modes only");
    const auto hybridSparseSupport = buildPartitionedReductionSparseSupport(
        hybridBasis, std::span<const std::uint32_t>(blockModeIndices),
        regionCount, blockModeCount);
    need(hybridSparseSupport.valid(),
         "CPU hybrid sparse support failed: " + hybridSparseSupport.error);
    for (const std::uint32_t anchor : hybridFixture.anchors) {
        const auto found = std::find(globalNodeIndices.begin(),
                                     globalNodeIndices.end(),
                                     anchor + kGlobalNodeOffset);
        need(found != globalNodeIndices.end(),
             "hybrid fixture anchor is missing from the source-row map");
        const std::size_t row = static_cast<std::size_t>(
            found - globalNodeIndices.begin());
        need(hybridSparseSupport.nodeModeRanges[row].count == 0u,
             "hybrid fixture anchor must remain on the fine complement");
    }
    NMPartitionedReductionBlockDispatchGPU hybridBlockDispatch = blockDispatch;
    hybridBlockDispatch.modeCount = hybridBasis.modeCount;
    hybridBlockDispatch.reserved = hybridSparseSupport.sparseTileCount;
    auto hybridBlockDispatchBuffer = makeBuffer(kernels.device,
        std::vector<NMPartitionedReductionBlockDispatchGPU>{hybridBlockDispatch});
    auto hybridBasisBuffer = makeBuffer(kernels.device, hybridBasis.values);
    auto hybridRegionSupportRangeBuffer = makeBuffer(kernels.device,
        hybridSparseSupport.regionSupportRanges);
    auto hybridRegionSupportRowsBuffer = makeBuffer(kernels.device,
        hybridSparseSupport.regionSupportRows);
    auto hybridNodeModeRangeBuffer = makeBuffer(kernels.device,
        hybridSparseSupport.nodeModeRanges);
    auto hybridNodeModeSlotsBuffer = makeBuffer(kernels.device,
        hybridSparseSupport.nodeModeSlots);

    // Exercise the runtime hybrid Bq map itself: sparse B^T restriction,
    // selected-row zeroing, q packing, full-vector copy/lift, and clearing the
    // inactive second FEM span. This validates adjoint work and preservation of
    // complement xyz, pressure w, and unrelated tail entries.
    constexpr std::uint32_t hybridTailCount = 9u;
    const std::size_t hybridUnknownCount = 2u * fineCount + hybridTailCount;
    need(hybridUnknownCount <= std::numeric_limits<std::uint32_t>::max(),
         "hybrid fixture unknown count exceeds the runtime layout field");
    NMFGMRESLayoutGPU hybridLayout{};
    hybridLayout.unknownCount = static_cast<std::uint32_t>(hybridUnknownCount);
    auto hybridLayoutBuffer = makeBuffer(kernels.device,
        std::vector<NMFGMRESLayoutGPU>{hybridLayout});

    std::vector<nm_float4> hybridInput(hybridUnknownCount, nm_float4{});
    std::copy(fine.begin(), fine.end(), hybridInput.begin());
    for (std::size_t index = 0u; index < fineCount; ++index)
        hybridInput[fineCount + index] = {
            71.0f + static_cast<float>(index), -19.0f, 5.0f, 3.0f};
    for (std::uint32_t tail = 0u; tail < hybridTailCount; ++tail)
        hybridInput[2u * fineCount + tail] = {
            900.0f + static_cast<float>(tail), -2.0f, 7.0f, 13.0f};

    const std::size_t hybridPartialCount =
        static_cast<std::size_t>(environmentCount) * regionCount *
        blockModeCount * hybridSparseSupport.sparseTileCount;
    std::vector<float> hybridProjectionPartials(
        hybridPartialCount, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> hybridProjectedRHS(
        static_cast<std::size_t>(environmentCount) * hybridBasis.modeCount,
        std::numeric_limits<float>::quiet_NaN());
    auto hybridInputBuffer = makeBuffer(kernels.device, hybridInput);
    auto hybridProjectionPartialBuffer = makeBuffer(
        kernels.device, hybridProjectionPartials);
    auto hybridProjectedRHSBuffer = makeBuffer(kernels.device, hybridProjectedRHS);
    kernels.dispatchGroups(kernels.reduceSparseModeTiles,
        MTLSizeMake(hybridSparseSupport.sparseTileCount, blockModeCount,
                    static_cast<NSUInteger>(regionCount) * environmentCount),
        {hybridBlockDispatchBuffer, hybridInputBuffer, hybridBasisBuffer, indexBuffer,
         nil, hybridProjectionPartialBuffer, blockModeBuffer,
         hybridRegionSupportRangeBuffer, hybridRegionSupportRowsBuffer});
    kernels.dispatchGroups(kernels.reduceSparseModePartials,
        MTLSizeMake(1u, blockModeCount,
                    static_cast<NSUInteger>(regionCount) * environmentCount),
        {hybridBlockDispatchBuffer, hybridProjectionPartialBuffer,
         hybridProjectedRHSBuffer, blockModeBuffer});
    kernels.dispatchFine(kernels.hybridZeroSelected,
        MTLSizeMake(static_cast<NSUInteger>(hybridBasis.nodeCount) * environmentCount,
                    1u, 1u),
        {hybridBlockDispatchBuffer, indexBuffer, hybridNodeModeRangeBuffer,
         hybridInputBuffer});
    kernels.dispatchFine(kernels.hybridPackModes,
        MTLSizeMake(static_cast<NSUInteger>(globalNodeCount) * environmentCount,
                    1u, 1u),
        {hybridBlockDispatchBuffer, hybridProjectedRHSBuffer,
         hybridInputBuffer});

    const auto* restricted = static_cast<const nm_float4*>(
        hybridInputBuffer.contents);
    const auto* hybridRHS = static_cast<const float*>(
        hybridProjectedRHSBuffer.contents);
    for (std::uint32_t environment = 0u;
         environment < environmentCount; ++environment) {
        for (std::uint32_t mode = 0u; mode < hybridBasis.modeCount; ++mode) {
            const float packed = restricted[fineCount +
                static_cast<std::size_t>(environment) * globalNodeCount + mode].x;
            need(std::isfinite(hybridRHS[
                     static_cast<std::size_t>(environment) * hybridBasis.modeCount + mode]) &&
                 close(packed, hybridRHS[
                     static_cast<std::size_t>(environment) * hybridBasis.modeCount + mode]),
                 "hybrid restriction did not pack the sparse B^T force coordinates");
        }
        for (std::uint32_t row = 0u; row < hybridBasis.nodeCount; ++row) {
            const std::uint32_t global = globalNodeIndices[row];
            const auto& after = restricted[
                static_cast<std::size_t>(environment) * globalNodeCount + global];
            const auto& before = fine[
                static_cast<std::size_t>(environment) * globalNodeCount + global];
            if (hybridSparseSupport.nodeModeRanges[row].count != 0u)
                need(after.x == 0.0f && after.y == 0.0f && after.z == 0.0f,
                     "hybrid restriction retained selected fine xyz coordinates");
            else
                need(after.x == before.x && after.y == before.y &&
                     after.z == before.z,
                     "hybrid restriction changed a fine-complement row");
            need(after.w == before.w,
                 "hybrid restriction changed the pressure w component");
        }
    }

    // Use independent deterministic pseudo-random q directions for the
    // virtual-work identity q^T(B^T f) = (Bq)^T f.
    std::vector<nm_float4> liftInput(restricted,
        restricted + hybridUnknownCount);
    std::vector<float> randomQ(
        static_cast<std::size_t>(environmentCount) * hybridBasis.modeCount);
    for (std::uint32_t environment = 0u;
         environment < environmentCount; ++environment) {
        for (std::uint32_t mode = 0u; mode < hybridBasis.modeCount; ++mode) {
            const float q = static_cast<float>(0.07 * std::sin(
                0.19 * static_cast<double>(mode + 1u) +
                0.31 * static_cast<double>(environment + 1u)));
            randomQ[static_cast<std::size_t>(environment) *
                hybridBasis.modeCount + mode] = q;
            liftInput[fineCount +
                static_cast<std::size_t>(environment) * globalNodeCount + mode].x = q;
        }
    }
    auto liftInputBuffer = makeBuffer(kernels.device, liftInput);
    std::vector<nm_float4> liftOutput(hybridUnknownCount,
                                      nm_float4{-999.0f, 4.0f, 8.0f, 12.0f});
    auto liftOutputBuffer = makeBuffer(kernels.device, liftOutput);
    kernels.dispatchFine(kernels.hybridCopy,
        MTLSizeMake(hybridUnknownCount, 1u, 1u),
        {hybridLayoutBuffer, liftInputBuffer, liftOutputBuffer});
    kernels.dispatchFine(kernels.hybridLift,
        MTLSizeMake(static_cast<NSUInteger>(hybridBasis.nodeCount) * environmentCount,
                    1u, 1u),
        {hybridBlockDispatchBuffer, indexBuffer, hybridBasisBuffer, blockModeBuffer,
         hybridNodeModeRangeBuffer, hybridNodeModeSlotsBuffer,
         liftInputBuffer, liftOutputBuffer});
    kernels.dispatchFine(kernels.hybridClearInactiveField,
        MTLSizeMake(fineCount, 1u, 1u),
        {hybridBlockDispatchBuffer, liftOutputBuffer});

    const auto* lifted = static_cast<const nm_float4*>(liftOutputBuffer.contents);
    for (std::uint32_t environment = 0u;
         environment < environmentCount; ++environment) {
        double reducedWork = 0.0;
        double fineWork = 0.0;
        for (std::uint32_t mode = 0u; mode < hybridBasis.modeCount; ++mode) {
            reducedWork += static_cast<double>(randomQ[
                static_cast<std::size_t>(environment) * hybridBasis.modeCount + mode]) *
                hybridRHS[static_cast<std::size_t>(environment) *
                    hybridBasis.modeCount + mode];
        }
        for (std::uint32_t row = 0u; row < hybridBasis.nodeCount; ++row) {
            const std::uint32_t global = globalNodeIndices[row];
            const std::size_t fineIndex =
                static_cast<std::size_t>(environment) * globalNodeCount + global;
            const auto& actual = lifted[fineIndex];
            const auto& inputValue = fine[fineIndex];
            if (hybridSparseSupport.nodeModeRanges[row].count != 0u) {
                double bx = 0.0, by = 0.0, bz = 0.0;
                for (std::uint32_t mode = 0u; mode < hybridBasis.modeCount; ++mode) {
                    const auto& b = hybridBasis.values[
                        static_cast<std::size_t>(mode) * hybridBasis.nodeCount + row];
                    const double q = randomQ[
                        static_cast<std::size_t>(environment) * hybridBasis.modeCount + mode];
                    bx += static_cast<double>(b.x) * q;
                    by += static_cast<double>(b.y) * q;
                    bz += static_cast<double>(b.z) * q;
                }
                need(close(actual.x, bx, 4.0e-6, 1.0e-3) &&
                     close(actual.y, by, 4.0e-6, 1.0e-3) &&
                     close(actual.z, bz, 4.0e-6, 1.0e-3),
                     "hybrid Bq lift differs from the CPU basis reference");
                fineWork += static_cast<double>(actual.x) * inputValue.x +
                    static_cast<double>(actual.y) * inputValue.y +
                    static_cast<double>(actual.z) * inputValue.z;
            } else {
                need(actual.x == inputValue.x && actual.y == inputValue.y &&
                     actual.z == inputValue.z,
                     "hybrid lift changed a fine-complement row");
            }
            need(actual.w == inputValue.w,
                 "hybrid lift changed the pressure w component");
        }
        need(close(reducedWork, fineWork, 1.0e-5, 2.0e-4),
             "hybrid Bq/B^T map violates the random-direction virtual-work identity");
    }
    for (std::size_t index = fineCount; index < 2u * fineCount; ++index)
        need(lifted[index].x == 0.0f && lifted[index].y == 0.0f &&
             lifted[index].z == 0.0f && lifted[index].w == 0.0f,
             "hybrid lift did not clear the inactive second FEM span");
    for (std::uint32_t tail = 0u; tail < hybridTailCount; ++tail) {
        const auto& before = liftInput[2u * fineCount + tail];
        const auto& after = lifted[2u * fineCount + tail];
        need(after.x == before.x && after.y == before.y &&
             after.z == before.z && after.w == before.w,
             "hybrid lift changed an unrelated solver tail entry");
    }
    std::cout << "hybrid_Bq_BTf_adjoint=pass random_directions=2"
              << " fine_complement=preserved pressure_w=preserved tail=preserved\n";

    const bool badTileCountRejected = [&] {
        auto invalid = dispatch;
        ++invalid.tileCount;
        return !validPartitionedReductionProjectionDispatch(invalid);
    }();
    need(badTileCountRejected, "host dispatch accepted an inconsistent tile count");
    std::cout << "basis_nodes=" << basis.nodeCount
              << " global_nodes=" << globalNodeCount
              << " modes=" << basis.modeCount
              << " tiles=" << tileCount
              << " work_conservation=pass"
              << " mass_action=pass"
              << " regional_block_factor_solve=pass"
              << " complement_and_pressure_w=pass"
              << " sparse_support_rows=" << sparseSupport.regionSupportRows.size()
              << " sparse_node_modes=" << sparseSupport.nodeModeSlots.size()
              << " sparse_mode_kc_validity_solve_add=pass\n";
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            verify();
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "partitioned reduction projection check failed: "
                      << error.what() << "\n";
            return 1;
        }
    }
}


#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/matter.hpp"
#include "cardboard_expression_cache_fixture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_CARDBOARD_EXPRESSION_CACHE_METALLIB
#define NUMI_CARDBOARD_EXPRESSION_CACHE_METALLIB ""
#endif

namespace {
using namespace numi::matter;
using namespace numi_cardboard_expression_cache;
using Matrix = std::array<double, 9>;
void need(const bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
std::string diagnostics(const std::vector<Diagnostic>& values) {
    std::string out;
    for (const auto& value : values) out += value.message + "; ";
    return out;
}
template<class T>
id<MTLBuffer> makeBuffer(id<MTLDevice> device, const std::vector<T>& values) {
    const T zero{};
    const void* bytes = values.empty() ? static_cast<const void*>(&zero) : values.data();
    return [device newBufferWithBytes:bytes
        length:std::max(std::size_t(1), values.size()) * sizeof(T)
        options:MTLResourceStorageModeShared];
}
struct Response { Matrix p{}, dp{}; bool valid = false; };
struct Instrument {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
    Instrument() {
        device = MTLCreateSystemDefaultDevice();
        need(device != nil, "Metal device is unavailable");
        const std::string name = [[device name] UTF8String];
        need(name.find("Apple") != std::string::npos &&
             name.find("Paravirtual") == std::string::npos,
             "physical Apple Metal device required");
        queue = [device newCommandQueue];
        NSError* error = nil;
        auto library = [device newLibraryWithURL:
            [NSURL fileURLWithPath:@NUMI_CARDBOARD_EXPRESSION_CACHE_METALLIB]
            error:&error];
        need(library != nil, "expression-cache Metal library unavailable");
        auto function = [library newFunctionWithName:@"cardboard_expression_cache_check"];
        need(function != nil, "production expression-cache check kernel missing");
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        need(pipeline != nil, "expression-cache Metal pipeline failed");
        std::cout << "device=" << name
                  << " evaluator=production_constitutive stress_tangent=true physical_steps=0\n";
    }
    Response evaluate(const CompiledWorld& world, const Matrix& f, const Matrix& h,
                      const bool expectValid = true) {
        std::vector<float> inputs;
        for (const double value : f) inputs.push_back(static_cast<float>(value));
        for (const double value : h) inputs.push_back(static_cast<float>(value));
        std::vector<float> parameters;
        for (const auto& parameter : world.parameters)
            parameters.push_back(parameter.valueAndBounds.x);
        std::vector<float> accepted(NM_MAX_MATERIAL_STATE, 0.0f);
        std::array<id<MTLBuffer>, 7> buffers{
            makeBuffer(device, world.materials),
            makeBuffer(device, world.scalarPrograms),
            makeBuffer(device, world.instructions),
            makeBuffer(device, parameters),
            makeBuffer(device, accepted),
            makeBuffer(device, inputs),
            makeBuffer(device, std::vector<float>(20u, 0.0f)),
        };
        auto command = [queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger i = 0; i < buffers.size(); ++i)
            [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        need(command.status == MTLCommandBufferStatusCompleted,
             "production constitutive dispatch failed");
        const auto* output = static_cast<const float*>(buffers[6].contents);
        Response response;
        response.valid = output[0] == 1.0f;
        need(response.valid == expectValid,
             expectValid ? "production evaluator rejected the branch fixture" :
                           "production evaluator accepted an invalid inactive branch");
        need(std::isfinite(output[1]) && (!expectValid || output[1] > 0.0f),
             "production evaluator returned an invalid material-validity margin");
        if (!expectValid) return response;
        for (unsigned i = 0; i < 9u; ++i) {
            need(std::isfinite(output[2u + i]) && std::isfinite(output[11u + i]),
                 "production evaluator returned nonfinite stress or tangent");
            response.p[i] = output[2u + i];
            response.dp[i] = output[11u + i];
        }
        return response;
    }
};
CompiledWorld cook(const MaterialProgram& material) {
    auto source = makeWorld(material);
    CompileOptions options;
    options.maximumRateExponent = 0u;
    options.emitSpecializedMetal = false;
    auto result = compileWorld(source, options);
    need(result.succeeded(), "Metal branch fixture compile: " + diagnostics(result.diagnostics));
    std::string error;
    need(validateCompiledWorldLayout(result.world, &error),
         "Metal branch fixture failed cooked admission: " + error);
    return std::move(result.world);
}
double relativeError(const Matrix& a, const Matrix& b) {
    double error = 0.0, scale = 1.0;
    for (unsigned i = 0; i < 9u; ++i) {
        error = std::max(error, std::abs(a[i] - b[i]));
        scale = std::max({scale, std::abs(a[i]), std::abs(b[i])});
    }
    return error / scale;
}
void check(Instrument& instrument, const double branchSign) {
    auto cachedSource = makeMaterial(true);
    auto uncachedSource = makeMaterial(false);
    const auto cached = cook(cachedSource);
    const auto uncached = cook(uncachedSource);
    Matrix f{1.0 + branchSign * 0.025, 0.13, 0.0,
             0.0, 1.0, -0.07, 0.0, 0.0, 1.0};
    const Matrix h{0.17, -0.21, 0.0, 0.0, 0.0, 0.11, 0.0, 0.0, 0.0};
    const auto c = instrument.evaluate(cached, f, h);
    const auto u = instrument.evaluate(uncached, f, h);
    need(relativeError(c.p, u.p) < 2.0e-6 && relativeError(c.dp, u.dp) < 2.0e-6,
         "cached and uncached production constitutive responses disagree");

    const double k = branchSign > 0.0 ? 4.0e6 : 7.0e6;
    Matrix expectedP{}, expectedDP{};
    expectedP[0] = k * (f[0] - 1.0);
    expectedP[1] = 4.0 * 2.5e5 * (f[1] + 0.2);
    expectedP[5] = 4.0 * 2.5e5 * (f[5] - 0.3);
    expectedDP[0] = k * h[0];
    expectedDP[1] = 4.0 * 2.5e5 * h[1];
    expectedDP[5] = 4.0 * 2.5e5 * h[5];
    const double stressError = relativeError(c.p, expectedP);
    const double tangentError = relativeError(c.dp, expectedDP);
    need(stressError < 2.0e-5,
         "production stress disagrees with active branch/shared-DAG oracle: " +
             std::to_string(stressError));
    need(tangentError < 2.0e-5,
         "production tangent disagrees with active branch/shared-DAG oracle: " +
             std::to_string(tangentError));

    constexpr double epsilon = 1.0e-3;
    Matrix plus = f, minus = f;
    for (unsigned i = 0; i < 9u; ++i) {
        plus[i] += epsilon * h[i];
        minus[i] -= epsilon * h[i];
    }
    const auto cp = instrument.evaluate(cached, plus, h);
    const auto cm = instrument.evaluate(cached, minus, h);
    Matrix fd{};
    for (unsigned i = 0; i < 9u; ++i) fd[i] = (cp.p[i] - cm.p[i]) / (2.0 * epsilon);
    const double fdError = relativeError(fd, c.dp);
    need(fdError < 3.0e-4,
         "production tangent fails same-branch centered stress difference: " +
             std::to_string(fdError));
    std::cout << "branch=" << (branchSign > 0.0 ? "positive" : "negative")
              << " cached_uncached_stress=" << relativeError(c.p, u.p)
              << " cached_uncached_tangent=" << relativeError(c.dp, u.dp)
              << " analytic_stress=" << stressError << " analytic_tangent=" << tangentError
              << " centered_FD=" << fdError << '\n';
}

void checkEagerSelectionDomain(Instrument& instrument) {
    const Matrix positiveF{1.025, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const Matrix negativeF{0.975, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const Matrix h{};
    Matrix cachedResponse{}, uncachedResponse{};
    for (const bool cachedFlag : {false, true}) {
        const auto world = cook(makeMaterial(cachedFlag, true));
        const auto rejected = instrument.evaluate(world, positiveF, h, false);
        need(!rejected.valid,
             "finite selected arm did not reject invalid inactive square root");
        const auto selected = instrument.evaluate(world, negativeF, h);
        need(selected.valid, "valid selected negative arm was unexpectedly rejected");
        const double expectedP00 = -7.0e6 / (2.0 * std::sqrt(0.025));
        need(std::abs(selected.p[0] - expectedP00) <=
                2.0e-5 * std::max(1.0, std::abs(expectedP00)),
             "production evaluator chose the wrong valid negative arm");
        if (cachedFlag) cachedResponse = selected.p;
        else uncachedResponse = selected.p;
    }
    need(relativeError(cachedResponse, uncachedResponse) < 2.0e-6,
         "cached and uncached eager-select domain responses disagree");
    std::cout << "select_nonnegative=eager inactive_invalid_rejected_cached_uncached=true"
              << " selected_valid_arm_preserved=true\n";
}
} // namespace

int main() {
    @autoreleasepool {
        try {
            Instrument instrument;
            check(instrument, -1.0);
            check(instrument, 1.0);
            checkEagerSelectionDomain(instrument);
            std::cout << "cardboard_expression_cache_gpu_check=pass"
                      << " physical_FEM_steps=0 source_material_qualification=false\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "cardboard expression cache GPU check failed: " << error.what() << '\n';
            return 1;
        }
    }
}

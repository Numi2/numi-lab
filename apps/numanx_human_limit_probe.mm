#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/human_limits_gpu.h"
#include "metalrobo/NumiHumanCompliantEquilibrium.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
double number(id x) {
    require([x isKindOfClass:[NSNumber class]], "oracle number missing");
    const double value = [x doubleValue]; require(std::isfinite(value), "oracle number nonfinite"); return value;
}
id<MTLBuffer> buffer(id<MTLDevice> device, std::size_t size) {
    auto result = [device newBufferWithLength:std::max<std::size_t>(size, 16u) options:MTLResourceStorageModeShared];
    require(result != nil, "Metal allocation failed"); std::memset(result.contents, 0, result.length); return result;
}
template<class T> T* data(id<MTLBuffer> b) { return static_cast<T*>(b.contents); }
double maximumError = 0.0;
std::size_t candidateCount = 0u, finiteDifferenceCount = 0u;
void near(double actual, double expected, const char* what, double tolerance = 5e-4) {
    const double error = std::abs(actual - expected) / std::max(1.0, std::abs(expected));
    maximumError = std::max(maximumError, error);
    if (!std::isfinite(actual) || error > tolerance) {
        std::cerr << what << " actual=" << actual << " expected=" << expected << " error=" << error << '\n';
        throw std::runtime_error(what);
    }
}
void run(NSDictionary* fixture, id<MTLDevice> device, id<MTLLibrary> library, id<MTLCommandQueue> queue) {
    NSArray* candidates = fixture[@"candidates"];
    require([candidates isKindOfClass:[NSArray class]] && candidates.count >= 5, "oracle candidates absent");
    const unsigned envs = unsigned(candidates.count), nv = 7u, capacity = 9u;
    NMMatterDispatchGPU dispatch{}; dispatch.environmentCount = envs; dispatch.rigidGeneralizedCapacity = capacity;
    dispatch.femNodeCount = 3u; dispatch.mpmActiveNodeCapacity = 5u;
    const std::size_t base = envs * (2u * dispatch.femNodeCount + dispatch.mpmActiveNodeCapacity);
    const std::size_t total = base + envs * capacity;
    NMHumanLimitDispatchGPU lim{}; lim.count = 1; lim.qCount = 8; lim.dofCount = nv;
    lim.policy = NM_HUMAN_EQUALITY_POLICY_MUJOCO_312_CLASSIC;
    lim.flags = unsigned(number(fixture[@"flags"])); lim.time.x = float(number(fixture[@"h"]));
    NMHumanJointLimitGPU row{}; row.indices = {7u, 6u, 1u, 0u};
    row.rangeMarginInverseWeight = {float(number(fixture[@"range"][0])), float(number(fixture[@"range"][1])),
        float(number(fixture[@"margin"])), float(number(fixture[@"source_inverse_weight"]))};
    row.solref = {float(number(fixture[@"solref"][0])), float(number(fixture[@"solref"][1])), 0.f, 0.f};
    row.solimp0 = {float(number(fixture[@"solimp"][0])), float(number(fixture[@"solimp"][1])),
        float(number(fixture[@"solimp"][2])), float(number(fixture[@"solimp"][3]))};
    row.solimp1 = {float(number(fixture[@"solimp"][4])), 0.f, 0.f, 0.f};
    // Independent MuJoCo fixture linearization supplies a_ref and R.
    // Remove only its known damping/free predictor terms to check the new
    // stationary preparation law against the retained source oracle.
    metalrobo::NumiHumanSourceScalarLaw staticLaw;
    std::memcpy(&staticLaw.solref,&row.solref,16);std::memcpy(&staticLaw.solimp0,&row.solimp0,16);std::memcpy(&staticLaw.solimp1,&row.solimp1,16);
    staticLaw.inverseWeight=row.rangeMarginInverseWeight.w;staticLaw.referenceSafe=(lim.flags&1u)!=0;
    const double dw=std::clamp(double(row.solimp0.y),0.0001,0.9999);
    const double timeConstant=staticLaw.referenceSafe?std::max(double(row.solref.x),2.0*lim.time.x):row.solref.x;
    const double damping=row.solref.x>0?2.0/std::max(1e-15,dw*timeConstant):-row.solref.y/std::max(1e-15,dw);
    for (unsigned side=0;side<2;++side) {
        const double inverseR=number(fixture[@"linearization"][side][2]);
        if (inverseR==0) continue;
        const double sign=side==0?1.0:-1.0;
        const double phi=number(fixture[@"linearization"][side][3]);
        const double freeIncrement=sign*(number(fixture[@"v_free"])-number(fixture[@"v0"]));
        const double aref=(number(fixture[@"linearization"][side][1])+freeIncrement)/lim.time.x;
        // The independent source fixture stores invweight0 in FP64; NHLIM1
        // deliberately transports FP32. Adjust that known serialization
        // factor explicitly rather than loosening the source-law comparison.
        const double expected=(aref+damping*sign*number(fixture[@"v0"]))*inverseR*
            number(fixture[@"source_inverse_weight"])/staticLaw.inverseWeight;
        double actual=0;
        require(metalrobo::evaluateNumiHumanSourceStaticForce(staticLaw,phi,lim.time.x,actual),"static source limit rejected");
        near(actual,expected,"offline static limit differs from MuJoCo source",1e-8);
    }
    const auto rows = buffer(device, sizeof(row)); std::memcpy(rows.contents, &row, sizeof(row));
    const auto q = buffer(device, envs * 8u * 4u), v = buffer(device, envs * nv * 4u), free = buffer(device, v.length);
    const auto lin = buffer(device, envs * 2u * 16u), status = buffer(device, envs * sizeof(NMMatterStatusGPU));
    const auto delta = buffer(device, envs * capacity * 4u), tangent = buffer(device, delta.length);
    const auto residual = buffer(device, total * 16u), direction = buffer(device, residual.length), output = buffer(device, residual.length);
    const auto states = buffer(device, envs * sizeof(NMFGMRESStateGPU));
    const auto factor = buffer(device, envs * nv * nv * 4u), rhs = buffer(device, residual.length), solution = buffer(device, residual.length);
    for (unsigned e = 0; e < envs; ++e) {
        data<float>(q)[e * 8u + 7u] = float(number(fixture[@"q"]));
        data<float>(v)[e * nv + 6u] = float(number(fixture[@"v0"]));
        data<float>(free)[e * nv + 6u] = float(number(fixture[@"v_free"]));
        data<float>(delta)[e * capacity + 6u] = float(number(candidates[e][@"delta"]));
        for (unsigned i = 0; i < nv; ++i) {
            data<nm_float4>(direction)[base + e * capacity + i].x = .3f + .01f * float(i);
            data<nm_float4>(rhs)[base + e * capacity + i].x = .5f + .02f * float(i);
            for (unsigned j = 0; j <= i; ++j)
                data<float>(factor)[e * nv * nv + i * nv + j] = i == j ? .8f + .03f * float(i) : .01f * std::sin(float(i + j));
        }
    }
    std::vector<float> originalFactor(data<float>(factor), data<float>(factor) + envs * nv * nv);
    std::vector<char> originalQ(static_cast<char*>(q.contents), static_cast<char*>(q.contents) + q.length);
    std::vector<char> originalV(static_cast<char*>(v.contents), static_cast<char*>(v.contents) + v.length);
    auto pipeline = [&](NSString* name) {
        NSError* error = nil;
        auto function = [library newFunctionWithName:[@"numi_matter_metal::" stringByAppendingString:name]];
        require(function != nil, "missing limit kernel");
        auto value = [device newComputePipelineStateWithFunction:function error:&error]; require(value != nil, "limit pipeline failed"); return value;
    };
    const auto prepare = pipeline(@"nm_human_limit_prepare"), evaluate = pipeline(@"nm_human_limit_residual"),
        action = pipeline(@"nm_human_limit_operator"), update = pipeline(@"nm_human_limit_factor"),
        precondition = pipeline(@"nm_human_equality_precondition");
    auto encode = [&](bool preparing, bool factoring, bool operating) {
        auto command = [queue commandBuffer]; auto encoder = [command computeCommandEncoder];
        auto start = [&](id<MTLComputePipelineState> p) { [encoder setComputePipelineState:p]; [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0]; };
        auto count = [&](unsigned n) { [encoder dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; };
        if (preparing) {
            start(prepare); [encoder setBytes:&lim length:sizeof(lim) atIndex:1];
            [encoder setBuffer:rows offset:0 atIndex:2]; [encoder setBuffer:q offset:0 atIndex:3];
            [encoder setBuffer:v offset:0 atIndex:4]; [encoder setBuffer:free offset:0 atIndex:5];
            [encoder setBuffer:lin offset:0 atIndex:6]; [encoder setBuffer:status offset:0 atIndex:7]; count(envs * 2u);
        }
        start(evaluate); [encoder setBytes:&lim length:sizeof(lim) atIndex:1];
        [encoder setBuffer:rows offset:0 atIndex:2]; [encoder setBuffer:lin offset:0 atIndex:3];
        [encoder setBuffer:delta offset:0 atIndex:4]; [encoder setBuffer:residual offset:0 atIndex:5];
        [encoder setBuffer:tangent offset:0 atIndex:6]; [encoder setBuffer:status offset:0 atIndex:7]; count(envs * capacity);
        if (operating) {
            start(action); [encoder setBuffer:tangent offset:0 atIndex:1]; [encoder setBuffer:direction offset:0 atIndex:2];
            [encoder setBuffer:output offset:0 atIndex:3]; [encoder setBuffer:states offset:0 atIndex:4]; count(envs * capacity);
        }
        if (factoring) {
            start(update); [encoder setBytes:&lim length:sizeof(lim) atIndex:1];
            [encoder setBuffer:rows offset:0 atIndex:2]; [encoder setBuffer:lin offset:0 atIndex:3];
            [encoder setBuffer:factor offset:0 atIndex:4]; [encoder setBuffer:status offset:0 atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake(envs,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            start(precondition); [encoder setBytes:&lim length:sizeof(lim) atIndex:1];
            [encoder setBuffer:factor offset:0 atIndex:2]; [encoder setBuffer:rhs offset:0 atIndex:3];
            [encoder setBuffer:solution offset:0 atIndex:4]; [encoder setBuffer:states offset:0 atIndex:5];
            [encoder setBuffer:status offset:0 atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake(envs,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        }
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "limit command failed");
    };
    encode(true, true, true);
    for (unsigned e = 0; e < envs; ++e) {
        require(data<NMMatterStatusGPU>(status)[e].code == 0u, "limit GPU status failed");
        for (unsigned side = 0; side < 2; ++side) {
            auto actual = reinterpret_cast<float*>(data<nm_float4>(lin) + 2u * e + side);
            for (unsigned k = 0; k < 4; ++k) near(actual[k], number(fixture[@"linearization"][side][k]), "source linearization");
        }
        for (unsigned i = 0; i < capacity; ++i) {
            const auto index = base + e * capacity + i;
            const double t = i == 6 ? number(candidates[e][@"tangent"]) : 0.;
            const double expectedForce = i == 6 ? number(candidates[e][@"residual"]) : 0.;
            // Removing REFSAFE can put the release threshold thousands of
            // velocity units from zero. Account explicitly for FP32 rounding
            // of the two subtracted operands, rather than scaling tolerance
            // solely by their nearly cancelling result.
            double cancellationBound = 0.;
            if (i == 6) for (unsigned side = 0; side < 2; ++side) {
                const auto l = data<nm_float4>(lin)[2u * e + side];
                cancellationBound += 4. * std::numeric_limits<float>::epsilon() *
                    (std::abs(double(l.x) * number(candidates[e][@"delta"])) + std::abs(double(l.y))) * l.z;
            }
            near(data<nm_float4>(residual)[index].x, expectedForce, "source force",
                5e-4 + cancellationBound / std::max(1., std::abs(expectedForce)));
            near(data<float>(tangent)[e * capacity + i], t, "source tangent");
            near(data<nm_float4>(output)[index].x, t * data<nm_float4>(direction)[index].x, "operator action");
        }
        const double envelope = double(data<nm_float4>(lin)[2u * e].z) + data<nm_float4>(lin)[2u * e + 1u].z;
        for (unsigned i = 0; i < nv; ++i) {
            double applied = 0.;
            for (unsigned j = 0; j < nv; ++j) {
                double expected = i == 6 && j == 6 ? envelope : 0., actual = 0.;
                for (unsigned k = 0; k < nv; ++k) {
                    expected += double(originalFactor[e * nv * nv + i * nv + k]) * originalFactor[e * nv * nv + j * nv + k];
                    actual += double(data<float>(factor)[e * nv * nv + i * nv + k]) * data<float>(factor)[e * nv * nv + j * nv + k];
                }
                near(actual, expected, "envelope factor", 3e-5);
                applied += expected * data<nm_float4>(solution)[base + e * capacity + j].x;
            }
            near(applied, data<nm_float4>(rhs)[base + e * capacity + i].x, "envelope precondition", 3e-5);
        }
    }
    candidateCount += envs;
    // A second residual evaluation must replace the tangent, including rows
    // that release. Replay the exact same immutable preparation and candidate.
    std::vector<char> replay(static_cast<char*>(residual.contents), static_cast<char*>(residual.contents) + residual.length);
    std::memset(residual.contents, 0, residual.length); encode(false, false, false);
    require(std::memcmp(replay.data(), residual.contents, residual.length) == 0, "limit replay drift");
    const float epsilon = .001f;
    for (unsigned e = 0; e < envs; ++e) data<float>(delta)[e * capacity + 6u] += epsilon;
    std::memset(residual.contents, 0, residual.length); encode(false, false, false);
    for (unsigned e = 0; e < envs; ++e) {
        const float d = float(number(candidates[e][@"delta"]));
        bool stable = std::abs(d) < 10.f;
        for (unsigned side = 0; side < 2; ++side) {
            const auto l = data<nm_float4>(lin)[2u * e + side];
            if (l.z != 0.f && std::abs(l.x * d - l.y) < .01f) stable = false;
        }
        if (!stable) continue;
        float prior; std::memcpy(&prior, replay.data() + 16u * (base + e * capacity + 6u), 4u);
        if (32. * std::numeric_limits<float>::epsilon() * std::abs(prior) >
            .005 * epsilon * std::max(1., number(candidates[e][@"tangent"]))) continue;
        const double fd = -(data<nm_float4>(residual)[base + e * capacity + 6u].x - prior) /
            double(data<float>(delta)[e * capacity + 6u] - d);
        near(fd, number(candidates[e][@"tangent"]), "residual finite difference", .005);
        ++finiteDifferenceCount;
    }
    require(std::memcmp(originalQ.data(), q.contents, q.length) == 0 &&
        std::memcmp(originalV.data(), v.contents, v.length) == 0, "limit kernels mutated source state");
    // Invalid source pose poisons only its environment and clears admitted rows.
    data<float>(q)[7] = std::numeric_limits<float>::quiet_NaN();
    std::memset(status.contents, 0, status.length); encode(true, false, false);
    require(data<NMMatterStatusGPU>(status)[0].code != 0u, "nonfinite source pose admitted");
    for (unsigned e = 1; e < envs; ++e) require(data<NMMatterStatusGPU>(status)[e].code == 0u, "failure leaked to another environment");
    std::cout << "limit_case=" << [fixture[@"name"] UTF8String] << " candidates=" << envs << " passed=1\n";
}
}
int main(int argc, char** argv) {
    @autoreleasepool { try {
        require(argc == 2, "usage: numanx_human_limit_probe mujoco-limit-oracle.json");
        NSError* error = nil;
        NSData* bytes = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]];
        require(bytes != nil, "oracle file missing");
        NSDictionary* oracle = [NSJSONSerialization JSONObjectWithData:bytes options:0 error:&error];
        require([oracle isKindOfClass:[NSDictionary class]] &&
            [oracle[@"schema"] isEqual:@"numi.human.mujoco-limit-oracle.v1"] &&
            [oracle[@"mujoco_version"] isEqual:@"3.12.0"], "oracle source identity mismatch");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice(); require(device != nil, "Metal unavailable");
        auto library = [device newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_MATTER_METALLIB] error:&error];
        require(library != nil, "Matter metallib unavailable"); auto queue = [device newCommandQueue];
        NSArray* cases = oracle[@"cases"]; require(cases.count >= 15, "oracle coverage incomplete");
        for (NSDictionary* fixture in cases) run(fixture, device, library, queue);
        require(finiteDifferenceCount >= 30, "finite difference coverage incomplete");
        std::cout << "{\"status\":\"passed\",\"candidates\":" << candidateCount << ",\"finite_differences\":" << finiteDifferenceCount
            << ",\"maximum_scaled_error\":" << maximumError << ",\"runtime_transaction_qualified\":false}\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
}

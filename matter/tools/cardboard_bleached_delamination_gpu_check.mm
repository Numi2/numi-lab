#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "cardboard_bleached_delamination_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_CARDBOARD_BLEACHED_DELAMINATION_METALLIB
#define NUMI_CARDBOARD_BLEACHED_DELAMINATION_METALLIB ""
#endif

namespace {
using namespace numi::matter;
using namespace numi_cardboard::bleached_delamination;
using SourceResponse = numi_cardboard::bleached_delamination::Response;
using Matrix = std::array<double, 9>;
constexpr Matrix identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
unsigned checks = 0u;

void require(const bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void close(const double actual, const double expected,
            const double relativeTolerance, const std::string& message) {
    const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
    require(std::isfinite(actual) && std::isfinite(expected) &&
            std::abs(actual - expected) <= relativeTolerance * scale,
            message + ": actual=" + std::to_string(actual) +
                " expected=" + std::to_string(expected));
}

std::string diagnostics(const std::vector<Diagnostic>& values) {
    std::string result;
    for (const auto& value : values) result += value.message + "; ";
    return result;
}

double parameter(const MaterialProgram& material, const std::string& name) {
    for (const Parameter& value : material.parameters) {
        if (value.name == name) return value.defaultValue;
    }
    throw std::runtime_error("missing material parameter " + name);
}

template<class T>
id<MTLBuffer> buffer(id<MTLDevice> device, const std::vector<T>& values) {
    const T zero{};
    const void* contents = values.empty() ? &zero : values.data();
    const std::size_t count = std::max(std::size_t{1u}, values.size());
    return [device newBufferWithBytes:contents
                                length:sizeof(T) * count
                               options:MTLResourceStorageModeShared];
}

struct Response {
    bool valid = false;
    double validityMargin = 0.0;
    bool validityEvaluated = false;
    bool admitted = false;
    bool projectionSucceeded = false;
    bool stressSucceeded = false;
    Matrix firstPiola{};
    Matrix tangentDirection{};
    std::vector<float> candidateState;
    std::vector<float> acceptedEcho;
};

struct Instrument {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;

    Instrument() {
        device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal device unavailable");
        const std::string deviceName = [[device name] UTF8String];
        require(deviceName.find("Apple") != std::string::npos &&
                    deviceName.find("Paravirtual") == std::string::npos,
                "physical Apple GPU required");
        queue = [device newCommandQueue];
        require(queue != nil, "Metal command queue unavailable");
        NSError* error = nil;
        NSString* path = @NUMI_CARDBOARD_BLEACHED_DELAMINATION_METALLIB;
        id<MTLLibrary> library = [device newLibraryWithURL:
            [NSURL fileURLWithPath:path] error:&error];
        require(library != nil, "bleached-paperboard instrument metallib missing");
        id<MTLFunction> function =
            [library newFunctionWithName:@"cardboard_bleached_delamination_check"];
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        require(pipeline != nil, "bleached-paperboard instrument kernel missing");
        std::cout << "device=" << deviceName
                  << " instrument=production_constitutive_evaluator"
                  << " physical_coupon_steps=0\n";
    }

    Response evaluate(const CompiledWorld& world, const double stretch,
                      const std::vector<float>& acceptedState) {
        Matrix deformation = identity;
        deformation[8] = stretch;
        return evaluate(world, deformation, acceptedState);
    }

    Response evaluate(const CompiledWorld& world, const Matrix& deformation,
                      const std::vector<float>& acceptedState) {
        require(acceptedState.size() == world.materials.at(0).stateCount,
                "accepted state width differs from material state count");
        Matrix direction{};
        direction[8] = 1.0;
        std::vector<float> inputs;
        inputs.reserve(18u);
        for (const double value : deformation) inputs.push_back(static_cast<float>(value));
        for (const double value : direction) inputs.push_back(static_cast<float>(value));
        std::vector<float> parameters;
        parameters.reserve(world.parameters.size());
        for (const NMParameterRangeGPU value : world.parameters)
            parameters.push_back(value.valueAndBounds.x);

        constexpr std::size_t outputCount = 24u + 2u * NM_MAX_MATERIAL_STATE;
        std::vector<float> output(outputCount, 0.0f);
        std::array<id<MTLBuffer>, 7u> buffers{
            buffer(device, std::vector<NMMaterialGPU>{world.materials.at(0)}),
            buffer(device, world.scalarPrograms),
            buffer(device, world.instructions),
            buffer(device, parameters),
            buffer(device, acceptedState),
            buffer(device, inputs),
            buffer(device, output),
        };
        id<MTLCommandBuffer> command = [queue commandBuffer];
        require(command != nil, "Metal command buffer unavailable");
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        require(encoder != nil, "Metal compute encoder unavailable");
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger index = 0u; index < buffers.size(); ++index)
            [encoder setBuffer:buffers[index] offset:0 atIndex:index];
        [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
                "Metal constitutive evaluation failed");

        const auto* values = static_cast<const float*>(buffers[6].contents);
        Response result;
        result.valid = values[0] == 1.0f;
        result.validityMargin = values[1];
        for (unsigned index = 0u; index < 9u; ++index) {
            result.firstPiola[index] = values[2u + index];
            result.tangentDirection[index] = values[11u + index];
        }
        const std::size_t stateCount = world.materials.at(0).stateCount;
        result.candidateState.assign(values + 20u, values + 20u + stateCount);
        const std::size_t acceptedOffset = 20u + NM_MAX_MATERIAL_STATE;
        result.acceptedEcho.assign(values + acceptedOffset,
                                   values + acceptedOffset + stateCount);
        result.validityEvaluated =
            values[20u + 2u * NM_MAX_MATERIAL_STATE] == 1.0f;
        result.admitted =
            values[21u + 2u * NM_MAX_MATERIAL_STATE] == 1.0f;
        result.projectionSucceeded =
            values[22u + 2u * NM_MAX_MATERIAL_STATE] == 1.0f;
        result.stressSucceeded =
            values[23u + 2u * NM_MAX_MATERIAL_STATE] == 1.0f;
        return result;
    }
};

CompiledWorld compileCoupon(const MaterialProgram& material) {
    WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 1.0e-4;
    source.gravity = {0.0, 0.0, 0.0};
    source.materials.push_back(material);

    ObjectSource object;
    object.name = "source_bounded_modeI_evaluator_fixture";
    object.representation = Representation::fem;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.mixedFEM = false;
    object.characteristicLength = 0.01;
    object.femNodes = {
        {{0.0, 0.0, 0.0}}, {{0.01, 0.0, 0.0}},
        {{0.0, 0.01, 0.0}}, {{0.0, 0.0, 0.01}},
    };
    object.tetrahedra = {{{0u, 1u, 2u, 3u}}};
    source.objects.push_back(std::move(object));

    CompileOptions options;
    options.maximumRateExponent = 0;
    options.emitSpecializedMetal = false;
    options.localMaterialNewtonIterations = 16u;
    const CompileResult result = compileWorld(source, options);
    require(result.succeeded(), "coupon material failed compile: " +
            diagnostics(result.diagnostics));
    return result.world;
}

double openingFromDamageMM(const double damage) {
    const Parameters& source = bleachedPaperboard;
    const double xi = damage / (1.0 - damage);
    return source.normalLengthMM * xc(source) * std::pow(xi, source.shape);
}

double stretchFromOpeningMM(const double openingMM, const double thicknessM) {
    return 1.0 + openingMM * 1.0e-3 / thicknessM;
}

void preserveAcceptedState(const Response& response,
                           const std::vector<float>& accepted,
                           const std::string& context) {
    require(response.acceptedEcho == accepted,
            context + " mutated or reordered accepted state");
}

void check(const std::string& path, Instrument& instrument) {
    const ParseResult parsed = parseMatterFile(path);
    require(parsed.succeeded(), "coupon material parse failed: " +
            diagnostics(parsed.diagnostics));
    const MaterialProgram& material = parsed.material;
    require(material.name == "tryding2023_bleached_paperboard_modeI_coupon",
            "wrong material asset loaded");
    require(material.internalState.size() == 1u &&
                material.internalState[0].name == "damage" &&
                material.internalState[0].transfer == InternalState::Transfer::maximum,
            "source damage state layout/transfer policy changed");
    require(material.internalState[0].initialValue == 0.01,
            "source initial damage 0.01 was silently changed");
    close(parameter(material, "mode_i_kinematic_tolerance"), 2.0e-6, 0.0,
          "Mode-I kinematic admission tolerance");
    constexpr double historyFBEpsilon = 1.0e-8;
    close(parameter(material, "history_fb_epsilon"), historyFBEpsilon, 0.0,
          "declared dimensionless history regularizer");
    require(material.validityRoot != NM_INVALID_INDEX,
            "Mode-I-only coupon must have an authored validity gate");

    const CompiledWorld world = compileCoupon(material);
    require(world.materials.size() == 1u && world.materials[0].stateCount == 1u,
            "compiled world changed the single scalar damage state");
    const double thicknessM = parameter(material, "thickness");
    close(parameter(material, "inverse_thickness"), 1.0 / thicknessM, 2.0e-7,
          "source-thickness reciprocal used for area-to-volume energy mapping");
    const double seedDamage = material.internalState[0].initialValue;
    const double seedOpeningMM = openingFromDamageMM(seedDamage);
    const double initialStretch = stretchFromOpeningMM(seedOpeningMM, thicknessM);
    const double representedInitialStretch = static_cast<double>(
        static_cast<float>(initialStretch));
    const double representedInitialOpeningMM =
        (representedInitialStretch - 1.0) * thicknessM * 1.0e3;
    const std::vector<float> initialState = world.stateInitials;
    require(initialState.size() == 1u, "missing initialized coupon damage state");

    State referenceState;
    referenceState.maximumOpeningMM = seedOpeningMM;
    referenceState.openingMM = seedOpeningMM;
    referenceState.tractionMPa = bleachedPaperboard.normalStrengthMPa *
                                 (1.0 - seedDamage);
    referenceState.accumulatedExternalWorkJPerM2 =
        envelopeWorkJPerM2(seedOpeningMM);
    referenceState.dissipatedWorkJPerM2 =
        referenceState.accumulatedExternalWorkJPerM2 -
        freeEnergyJPerM2(referenceState);
    // Evaluate the independent source reference at the exact FP32 stretch
    // delivered to Metal. If that representable value falls just below the
    // exact source seed opening, keep kappa fixed and use source Eq. (43)
    // unloading; do not disguise the quantization effect as a material fit.
    const SourceResponse representedInitial = advance(
        referenceState, representedInitialOpeningMM);
    const double seedEnvelopeTractionPa =
        bleachedPaperboard.normalStrengthMPa * (1.0 - seedDamage) * 1.0e6;
    const double seedTractionQuantizationDeltaPa =
        representedInitial.tractionMPa * 1.0e6 - seedEnvelopeTractionPa;
    const double initialFreeEnergy = representedInitial.freeEnergyJPerM2;

    const Response initial = instrument.evaluate(world, initialStretch, initialState);
    require(initial.valid && initial.validityMargin > 0.0,
            "source-seeded positive-opening state was rejected: margin=" +
                std::to_string(initial.validityMargin) +
                " seed_damage=" + std::to_string(seedDamage) +
                " opening_mm=" + std::to_string(seedOpeningMM) +
                " stretch=" + std::to_string(initialStretch) +
                " validity_evaluated=" +
                std::to_string(initial.validityEvaluated) +
                " admitted=" + std::to_string(initial.admitted) +
                " projection_succeeded=" +
                std::to_string(initial.projectionSucceeded) +
                " stress_succeeded=" +
                std::to_string(initial.stressSucceeded));
    close(initial.candidateState[0], seedDamage, 2.0e-6,
          "native initial projected damage");
    close(initial.firstPiola[8], representedInitial.tractionMPa * 1.0e6, 3.0e-5,
          "native initial onset traction");
    preserveAcceptedState(initial, initialState, "initial projection");

    constexpr unsigned loadingSteps = 512u;
    constexpr double targetOpeningMM = 0.1;
    std::vector<float> accepted = initial.candidateState;
    double previousStretch = representedInitialStretch;
    double previousStress = initial.firstPiola[8];
    double nativeWork = referenceState.accumulatedExternalWorkJPerM2;
    Response loaded = initial;
    double maximumTangentRelativeError = 0.0;
    double worstNativeTangent = 0.0;
    double worstSourceTangent = 0.0;
    unsigned worstTangentStep = 0u;
    double maximumFdRelativeError = 0.0;

    for (unsigned step = 1u; step <= loadingSteps; ++step) {
        const double fraction = static_cast<double>(step) / loadingSteps;
        const double openingMM = seedOpeningMM +
            fraction * (targetOpeningMM - seedOpeningMM);
        const double stretch = stretchFromOpeningMM(openingMM, thicknessM);
        const std::vector<float> acceptedBefore = accepted;
        const Response response = instrument.evaluate(world, stretch, accepted);
        require(response.valid, "native envelope projection failed at loading step " +
                    std::to_string(step));
        preserveAcceptedState(response, acceptedBefore,
                             "loading projection step " + std::to_string(step));
        require(response.candidateState[0] + 3.0e-6f >= accepted[0],
                "native source damage decreased during monotonic opening");
        const double sourceTraction =
            envelopeTractionMPa(openingMM) * 1.0e6;
        close(response.firstPiola[8], sourceTraction, 4.0e-4,
              "native monotonic Mode-I traction at step " +
                  std::to_string(step));
        close(response.candidateState[0],
              1.0 - envelopeTractionMPa(openingMM) /
                        bleachedPaperboard.normalStrengthMPa,
              3.0e-5, "native monotonic damage at step " +
                          std::to_string(step));
        nativeWork += 0.5 * (previousStress + response.firstPiola[8]) *
                      (stretch - previousStretch) * thicknessM;

        if (step == 64u || step == 256u || step == loadingSteps) {
            const double expectedTangent = envelopeTangentMPaPerMM(openingMM) *
                                           1.0e9 * thicknessM;
            const double tangentError = std::abs(
                response.tangentDirection[8] - expectedTangent) /
                std::max(1.0, std::abs(expectedTangent));
            if (tangentError > maximumTangentRelativeError) {
                maximumTangentRelativeError = tangentError;
                worstNativeTangent = response.tangentDirection[8];
                worstSourceTangent = expectedTangent;
                worstTangentStep = step;
            }

            constexpr double fdStep = 1.0e-5;
            const Response plus = instrument.evaluate(
                world, stretch + fdStep, acceptedBefore);
            const Response minus = instrument.evaluate(
                world, stretch - fdStep, acceptedBefore);
            require(plus.valid && minus.valid,
                    "centered tangent perturbation left valid Mode-I domain");
            const double finiteDifference =
                (plus.firstPiola[8] - minus.firstPiola[8]) /
                (2.0 * fdStep);
            const double fdError = std::abs(
                response.tangentDirection[8] - finiteDifference) /
                std::max(1.0, std::abs(finiteDifference));
            maximumFdRelativeError = std::max(maximumFdRelativeError, fdError);
        }
        accepted = response.candidateState;
        previousStretch = stretch;
        previousStress = response.firstPiola[8];
        loaded = response;
    }

    const State referenceStartForRefinement = referenceState;
    SourceResponse referenceLoaded = advance(referenceState, targetOpeningMM);
    close(nativeWork, referenceLoaded.cumulativeWorkJPerM2, 3.0e-4,
          "native stress work versus source envelope work");
    close(loaded.candidateState[0], referenceLoaded.damage, 4.0e-5,
          "native terminal loading damage versus source reference");
    close(loaded.firstPiola[8], referenceLoaded.tractionMPa * 1.0e6,
          4.0e-4, "native terminal load versus source reference");

    // Repeat the native loading path with fewer accepted samples. The
    // 512-step result above must converge toward the same independent source
    // work, rather than passing only at one chosen path resolution.
    double coarseWork =
        referenceStartForRefinement.accumulatedExternalWorkJPerM2;
    double coarsePreviousStress = initial.firstPiola[8];
    double coarsePreviousStretch = representedInitialStretch;
    std::vector<float> coarseAccepted = initialState;
    constexpr unsigned coarseSteps = 32u;
    for (unsigned step = 1u; step <= coarseSteps; ++step) {
        const double fraction = static_cast<double>(step) / coarseSteps;
        const double openingMM = representedInitialOpeningMM +
            fraction * (targetOpeningMM - representedInitialOpeningMM);
        const double requestedStretch =
            stretchFromOpeningMM(openingMM, thicknessM);
        const double representedStretch = static_cast<double>(
            static_cast<float>(requestedStretch));
        const Response response = instrument.evaluate(
            world, requestedStretch, coarseAccepted);
        require(response.valid,
                "coarse native loading path failed at step " +
                    std::to_string(step));
        coarseWork += 0.5 * (coarsePreviousStress + response.firstPiola[8]) *
            (representedStretch - coarsePreviousStretch) * thicknessM;
        coarseAccepted = response.candidateState;
        coarsePreviousStress = response.firstPiola[8];
        coarsePreviousStretch = representedStretch;
    }
    const double coarseWorkError = std::abs(
        coarseWork - referenceLoaded.cumulativeWorkJPerM2);
    const double refinedWorkError = std::abs(
        nativeWork - referenceLoaded.cumulativeWorkJPerM2);
    require(refinedWorkError < coarseWorkError,
            "512-step native path did not improve work error over the 32-step path");

    // Advance to a new maximum, then re-evaluate the identical representable
    // F with that accepted FP32 damage. The source max law has distinct
    // loading and unloading one-sided tangents at this history junction; the
    // regularized implicit tangent must select their midpoint.
    const double tieOpeningMM = seedOpeningMM * 1.10;
    const double tieRequestedStretch =
        stretchFromOpeningMM(tieOpeningMM, thicknessM);
    const Response tiePrime = instrument.evaluate(
        world, tieRequestedStretch, initialState);
    require(tiePrime.valid,
            "could not reach a positive-opening history junction");
    const std::vector<float> tieState = tiePrime.candidateState;
    const Response tie = instrument.evaluate(
        world, tieRequestedStretch, tieState);
    require(tie.valid, "native exact-max junction evaluation failed");
    preserveAcceptedState(tie, tieState, "native history junction");
    close(tie.candidateState[0], tieState[0], 0.0,
          "exact-max junction projection must retain its accepted state");
    const double tieMaximumOpeningMM =
        openingFromDamageMM(static_cast<double>(tieState[0]));
    const double sourceLoadingTangent = envelopeTangentMPaPerMM(
        tieMaximumOpeningMM) * 1.0e9 * thicknessM;
    const double sourceUnloadingTangent = unloadingModulusMPaPerMM(
        tieMaximumOpeningMM) * 1.0e9 * thicknessM;
    const double sourceJunctionMidpoint =
        0.5 * (sourceLoadingTangent + sourceUnloadingTangent);
    close(tie.tangentDirection[8], sourceJunctionMidpoint, 1.0e-4,
          "native history-junction tangent must match source one-sided midpoint");
    const double junctionResidual = std::sqrt(2.0) * historyFBEpsilon;
    const double junctionTolerance = 2.0e-5 *
        (1.0 + std::abs(static_cast<double>(tieState[0])));
    require(junctionResidual < junctionTolerance,
            "history-junction residual must be bounded by native local tolerance");
    std::vector<float> heldState = tieState;
    constexpr unsigned repeatedHolds = 1000u;
    for (unsigned step = 0u; step < repeatedHolds; ++step) {
        const Response held = instrument.evaluate(
            world, tieRequestedStretch, heldState);
        require(held.valid, "native constant-opening hold was rejected");
        preserveAcceptedState(held, heldState,
                              "native repeated constant-opening hold");
        require(held.candidateState[0] == tieState[0],
                "native max-history projection crept during repeated holds");
        heldState = held.candidateState;
    }
    require(maximumTangentRelativeError < 0.04,
            "native algorithmic tangent differs from source envelope derivative: error=" +
                std::to_string(maximumTangentRelativeError) +
                " step=" + std::to_string(worstTangentStep) +
                " native=" + std::to_string(worstNativeTangent) +
                " source=" + std::to_string(worstSourceTangent));
    require(maximumFdRelativeError < 0.04,
            "native algorithmic tangent differs from centered GPU stress FD: error=" +
                std::to_string(maximumFdRelativeError));

    const double loadedFreeEnergy = referenceLoaded.freeEnergyJPerM2;
    const double loadedDissipation = referenceLoaded.cumulativeDissipationJPerM2;
    const std::vector<float> loadedState = loaded.candidateState;
    constexpr double unloadOpeningMM = 0.08;
    const double unloadStretch = stretchFromOpeningMM(unloadOpeningMM, thicknessM);
    const Response unloaded = instrument.evaluate(world, unloadStretch, loadedState);
    require(unloaded.valid, "native elastic unloading projection failed");
    preserveAcceptedState(unloaded, loadedState, "native unloading");
    State referenceUnloadedState = referenceState;
    const SourceResponse referenceUnloaded = advance(referenceUnloadedState,
                                                      unloadOpeningMM);
    close(unloaded.candidateState[0], loadedState[0], 2.0e-6,
          "native damage must remain fixed during unloading");
    close(unloaded.firstPiola[8], referenceUnloaded.tractionMPa * 1.0e6,
          4.0e-4, "native Eq. (43) unloading traction");
    const double expectedUnloadTangent =
        referenceUnloaded.unloadingModulusMPaPerMM * 1.0e9 * thicknessM;
    close(unloaded.tangentDirection[8], expectedUnloadTangent, 4.0e-4,
          "native frozen-damage unloading tangent");
    const double nativeFreeEnergyDensity = 0.5 *
        unloaded.firstPiola[8] * unloaded.firstPiola[8] /
        unloaded.tangentDirection[8];
    close(nativeFreeEnergyDensity,
          referenceUnloaded.freeEnergyJPerM2 / thicknessM, 5.0e-4,
          "unloading stored energy reconstructed from native stress/tangent");
    const double unloadWork = 0.5 * (loaded.firstPiola[8] +
        unloaded.firstPiola[8]) * (unloadStretch -
        stretchFromOpeningMM(targetOpeningMM, thicknessM)) * thicknessM;
    close(unloadWork,
          referenceUnloaded.freeEnergyJPerM2 - loadedFreeEnergy, 5.0e-4,
          "native elastic unloading work/storage balance");
    close(referenceUnloaded.cumulativeDissipationJPerM2,
          loadedDissipation, 2.0e-10,
          "source reference dissipation must not grow on elastic unloading");

    constexpr double reloadOpeningMM = 0.09;
    const Response reloaded = instrument.evaluate(
        world, stretchFromOpeningMM(reloadOpeningMM, thicknessM),
        unloaded.candidateState);
    const Response reclosed = instrument.evaluate(world, unloadStretch,
                                                   reloaded.candidateState);
    require(reloaded.valid && reclosed.valid,
            "native closed elastic unload/reload path was rejected");
    close(reloaded.candidateState[0], loadedState[0], 2.0e-6,
          "native reloading below the previous maximum changed damage");
    const double closedLoopWork = 0.5 * (unloaded.firstPiola[8] +
        reloaded.firstPiola[8]) * (stretchFromOpeningMM(reloadOpeningMM,
        thicknessM) - unloadStretch) * thicknessM +
        0.5 * (reloaded.firstPiola[8] + reclosed.firstPiola[8]) *
        (unloadStretch - stretchFromOpeningMM(reloadOpeningMM,
        thicknessM)) * thicknessM;
    close(closedLoopWork, 0.0, 1.0e-5,
          "native elastic unload/reload loop work");

    const std::vector<float> acceptedBeforeReject = loadedState;
    const Response rejected = instrument.evaluate(world, 0.999,
                                                  acceptedBeforeReject);
    require(!rejected.valid && rejected.validityMargin < 0.0,
            "nonpositive opening was not rejected by material validity");
    preserveAcceptedState(rejected, acceptedBeforeReject,
                          "rejected compression-domain evaluation");
    const Response rejectedZeroOpening = instrument.evaluate(
        world, 1.0, acceptedBeforeReject);
    require(!rejectedZeroOpening.valid &&
                rejectedZeroOpening.validityMargin <= 0.0,
            "zero opening was admitted despite the post-initiation coupon domain");
    preserveAcceptedState(rejectedZeroOpening, acceptedBeforeReject,
                          "rejected zero-opening evaluation");
    Matrix sheared = identity;
    sheared[1] = 0.05;
    const Response rejectedShear = instrument.evaluate(
        world, sheared, acceptedBeforeReject);
    require(!rejectedShear.valid && rejectedShear.validityMargin < 0.0,
            "non-Mode-I shear was admitted by coupon validity");
    preserveAcceptedState(rejectedShear, acceptedBeforeReject,
                          "rejected shear-domain evaluation");
    Matrix biaxial = identity;
    biaxial[0] = 1.01;
    const Response rejectedInPlaneStretch = instrument.evaluate(
        world, biaxial, acceptedBeforeReject);
    require(!rejectedInPlaneStretch.valid &&
                rejectedInPlaneStretch.validityMargin < 0.0,
            "in-plane stretch was admitted by coupon validity");
    preserveAcceptedState(rejectedInPlaneStretch, acceptedBeforeReject,
                          "rejected in-plane-strain evaluation");
    const Response recovered = instrument.evaluate(world, unloadStretch,
                                                    acceptedBeforeReject);
    require(recovered.valid, "coupon did not recover after rejected evaluation");
    close(recovered.candidateState[0], acceptedBeforeReject[0], 2.0e-6,
          "rejected candidate leaked into the next accepted-history evaluation");

    require(initialFreeEnergy > 0.0,
            "explicit post-initiation source seed should carry elastic energy");
    std::cout << std::setprecision(10) << material.name
              << " source_material_fingerprint=" << material.fingerprint
              << " seed_damage=" << seedDamage
              << " seed_opening_mm=" << seedOpeningMM
              << " seed_stretch_fp32=" << representedInitialStretch
              << " represented_seed_opening_mm=" << representedInitialOpeningMM
              << " seed_envelope_traction_Pa=" << seedEnvelopeTractionPa
              << " represented_seed_traction_Pa="
              << representedInitial.tractionMPa * 1.0e6
              << " seed_traction_quantization_delta_Pa="
              << seedTractionQuantizationDeltaPa
              << " loaded_opening_mm=" << targetOpeningMM
              << " loaded_traction_MPa=" << loaded.firstPiola[8] * 1.0e-6
              << " native_work_J_m2=" << nativeWork
              << " source_work_J_m2="
              << referenceLoaded.cumulativeWorkJPerM2
              << " unload_free_energy_J_m2="
              << referenceUnloaded.freeEnergyJPerM2
              << " unload_dissipation_J_m2="
              << referenceUnloaded.cumulativeDissipationJPerM2
              << " coarse_steps=" << coarseSteps
              << " coarse_native_work_J_m2=" << coarseWork
              << " coarse_work_error_J_m2=" << coarseWorkError
              << " refined_steps=" << loadingSteps
              << " refined_work_error_J_m2=" << refinedWorkError
              << " history_fb_epsilon=" << historyFBEpsilon
              << " junction_residual=" << junctionResidual
              << " junction_local_tolerance=" << junctionTolerance
              << " junction_residual_to_tolerance="
              << junctionResidual / junctionTolerance
              << " junction_state=" << tieState[0]
              << " junction_loading_tangent_Pa_per_F33="
              << sourceLoadingTangent
              << " junction_unloading_tangent_Pa_per_F33="
              << sourceUnloadingTangent
              << " junction_native_midpoint_tangent_Pa_per_F33="
              << tie.tangentDirection[8]
              << " junction_source_midpoint_Pa_per_F33="
              << sourceJunctionMidpoint
              << " repeated_holds=" << repeatedHolds
              << " hold_damage_change="
              << static_cast<double>(heldState[0]) - tieState[0]
              << " rejected_shear=true rejected_in_plane=true"
              << " max_tangent_relative_error="
              << maximumTangentRelativeError
              << " max_centered_fd_relative_error="
              << maximumFdRelativeError << '\n';
}
} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            require(argc == 2, "expected the source-bounded coupon material path");
            Instrument instrument;
            check(argv[1], instrument);
            std::cout << "PASS native pure-mode-I constitutive coupon checks="
                      << checks
                      << " zero_thickness_face_law=false"
                      << " corrugated_cardboard_validation=false\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << error.what() << '\n';
            return 1;
        }
    }
}

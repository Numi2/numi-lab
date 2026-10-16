#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"
#include "robertsson2023_material.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#define NUMI_MATTER_METALLIB ""
#endif

using namespace numi::matter;

namespace {
constexpr std::uint32_t kEnvironmentCount = 2u;
constexpr std::uint32_t kNodeCount = 4u;
constexpr std::uint32_t kTetCount = 1u;
constexpr double kThicknessM = 0.00039;
constexpr double kMaximumShear = 0.3;
constexpr std::uint32_t kIncrementsPerLeg = 60u;
constexpr double kTimestepSeconds = 0.005;
constexpr std::uint32_t kStateCount = 27u;
constexpr double kTetVolume = kThicknessM * kThicknessM * kThicknessM / 6.0;

void require(bool condition, const std::string& message);

double evaluateEnergyRoot(const MaterialProgram& material,
                          const std::array<double, 9>& deformation,
                          const double damage) {
    std::vector<double> accepted;
    accepted.reserve(material.internalState.size());
    for (const auto& state : material.internalState)
        accepted.push_back(state.initialValue);
    require(accepted.size() == kStateCount,
            "source material state layout differs from the authored 27-state law");
    std::vector<double> candidate = accepted;
    candidate[26] = damage;
    std::vector<double> cache(material.expressions.nodes.size(),
                              std::numeric_limits<double>::quiet_NaN());
    std::function<double(std::uint32_t)> visit = [&](const std::uint32_t index) {
        const auto& node = material.expressions.nodes.at(index);
        if (std::isfinite(cache.at(index))) return cache.at(index);
        const auto arg = [&](const unsigned slot) {
            return visit(node.arguments.at(slot));
        };
        double value = 0.0;
        switch (node.kind) {
            case ExprKind::constant:
                value = static_cast<float>(node.constant);
                break;
            case ExprKind::parameter:
                value = static_cast<float>(material.parameters.at(node.index).defaultValue);
                break;
            case ExprKind::internalState:
                value = accepted.at(node.index);
                break;
            case ExprKind::candidateState:
                value = candidate.at(node.index);
                break;
            case ExprKind::deformation:
                value = deformation.at(node.index);
                break;
            case ExprKind::deformationDirection:
            case ExprKind::deformationRate:
                value = 0.0;
                break;
            case ExprKind::timeStep:
                value = kTimestepSeconds;
                break;
            case ExprKind::temperature:
                value = 293.15;
                break;
            case ExprKind::add:
                value = arg(0u) + arg(1u);
                break;
            case ExprKind::subtract:
                value = arg(0u) - arg(1u);
                break;
            case ExprKind::multiply:
                value = arg(0u) * arg(1u);
                break;
            case ExprKind::divide:
                value = arg(0u) / arg(1u);
                break;
            case ExprKind::negate:
                value = -arg(0u);
                break;
            case ExprKind::logarithm:
                value = std::log(arg(0u));
                break;
            case ExprKind::exponential:
                value = std::exp(arg(0u));
                break;
            case ExprKind::squareRoot:
                value = std::sqrt(arg(0u));
                break;
            case ExprKind::absolute:
                value = std::abs(arg(0u));
                break;
            case ExprKind::minimum:
                value = std::min(arg(0u), arg(1u));
                break;
            case ExprKind::maximum:
                value = std::max(arg(0u), arg(1u));
                break;
            case ExprKind::integerPower:
                value = std::pow(arg(0u), node.integer);
                break;
            case ExprKind::clamp:
                value = std::clamp(arg(0u), arg(1u), arg(2u));
                break;
            case ExprKind::expm1MinusArgument: {
                const double x = arg(0u);
                value = std::expm1(x) - x;
                break;
            }
            case ExprKind::selectNonnegative:
                value = arg(0u) >= 0.0 ? arg(1u) : arg(2u);
                break;
            case ExprKind::fiberExpLinear:
                throw std::runtime_error(
                    "unexpected fiber expression in the Robertsson source material");
        }
        require(std::isfinite(value), "nonfinite Robertsson energy expression");
        cache[index] = value;
        return value;
    };
    return visit(material.energyRoot);
}

double sourceZDCompressionMPa(const double i13) {
    constexpr double e6MPa = 0.365;
    constexpr double e7 = 11.5;
    const double square = i13 * i13;
    return e6MPa * (square + std::exp(-e7 * (square - 1.0)) / e7 - 2.0);
}

void checkOwnGradeAbsoluteEnergy(const MaterialProgram& paper,
                                 const MaterialProgram& recentered) {
    const auto energyAt = [](const MaterialProgram& material,
                             const double stretch,
                             const double damage) {
        return evaluateEnergyRoot(material,
            {1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, stretch}, damage);
    };
    const auto close = [](const double actual, const double expected,
                          const double absoluteTolerance,
                          const double relativeTolerance) {
        return std::abs(actual - expected) <= std::max(
            absoluteTolerance,
            relativeTolerance * std::max(std::abs(actual), std::abs(expected)));
    };

    const double identityEnergyPa = energyAt(paper, 1.0, 0.0);
    require(std::abs(identityEnergyPa) <= 1.0e-6,
            "undeformed undamaged source reference energy is not zero");

    constexpr double compressionStretch = 0.99;
    constexpr double damage = 0.25;
    const double sourceEnergyMPa = sourceZDCompressionMPa(compressionStretch);
    const double expectedPaperPa = (1.0 - damage) * sourceEnergyMPa * 1.0e6;
    const double paperEnergyPa = energyAt(paper, compressionStretch, damage);
    require(close(paperEnergyPa, expectedPaperPa, 1.0e-3, 2.0e-6),
            "paper_absolute builder differs from the printed Eq. 18 branch");
    require(paperEnergyPa < 0.0,
            "printed Eq. 18 compression branch no longer exposes its negative offset");

    const double offsetMPa = 0.365 * (1.0 - 1.0 / 11.5);
    const double expectedOffsetPa = (1.0 - damage) * offsetMPa * 1.0e6;
    const double recenteredEnergyPa =
        energyAt(recentered, compressionStretch, damage);
    require(close(recenteredEnergyPa - paperEnergyPa,
                  expectedOffsetPa, 1.0e-3, 2.0e-6),
            "recentered convention does not differ by the documented absolute offset");
    require(recenteredEnergyPa > 0.0,
            "recentered comparison energy should be positive at this compressed state");

    const double undamagedPaperPa = energyAt(paper, compressionStretch, 0.0);
    const double damageForcePa =
        (undamagedPaperPa - paperEnergyPa) / damage;
    require(close(damageForcePa, undamagedPaperPa, 1.0e-3, 2.0e-6),
            "damage-conjugate force is not the selected absolute Psi0");

    constexpr double tensionStretch = 1.1;
    const double tensionExpectedPa = 28.4e6 *
        (tensionStretch + 1.0 / tensionStretch - 2.0);
    require(close(energyAt(paper, tensionStretch, 0.0),
                  tensionExpectedPa, 1.0e-2, 2.0e-6),
            "paper_absolute tension branch differs from Eq. 18/Table C1");

    const double leftLimitMPa = 0.365 * (1.0 / 11.5 - 1.0);
    require(close(leftLimitMPa, -0.3332608695652174, 1.0e-14, 1.0e-14),
            "analytic compressed-side limit differs from Eq. 18/Table C1");
    std::cout << std::setprecision(17)
              << "own_grade_cpu_energy_check=PASS\n"
              << "source=Robertsson_2023_single_ply_softwood_paperboard\n"
              << "reference_state_energy_Pa=" << identityEnergyPa << '\n'
              << "compression_I13=" << compressionStretch << '\n'
              << "paper_absolute_energy_at_damage_0.25_Pa=" << paperEnergyPa << '\n'
              << "recentered_minus_paper_absolute_at_damage_0.25_Pa="
              << (recenteredEnergyPa - paperEnergyPa) << '\n'
              << "damage_conjugate_A_at_damage_0_Pa=" << undamagedPaperPa << '\n'
              << "analytic_compressed_side_limit_MPa=" << leftLimitMPa << '\n'
              << "source_assertion_Psi0_nonnegative_at_compression=CONTRADICTED_BY_PRINTED_EQ18\n"
              << "corrugated_kraft_or_NSSC_transfer=NOT_TESTED\n"
              << "qualification=source_equation_CPU_consistency_check_only\n";
}

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
id<MTLBuffer> sharedBuffer(id<MTLDevice> device,
                           const std::vector<T>& values) {
    require(!values.empty(), "cannot allocate an empty Metal buffer");
    return [device newBufferWithBytes:values.data()
                               length:values.size() * sizeof(T)
                              options:MTLResourceStorageModeShared];
}

template <typename T>
bool byteEqual(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
        (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

CompiledWorld compileFixture(
    const numi_cardboard::robertsson2023::CompressionEnergyConvention convention) {
    WorldSource source;
    source.environmentCount = kEnvironmentCount;
    source.gravity = {0.0f, 0.0f, 0.0f};
    source.frameTimestep = kTimestepSeconds;
    source.deterministic = true;
    source.mixedSolver.newtonIterations = 1u;
    source.materials = {
        numi_cardboard::robertsson2023::makePublishedReferenceMaterial(convention)
    };
    require(source.materials[0].name.find(
                numi_cardboard::robertsson2023::conventionName(convention)) !=
                std::string::npos,
            "compiled source material name lost its explicit energy convention");

    ObjectSource object;
    object.name = "robertsson_single_ply_prescribed_F13_tetra";
    object.representation = Representation::fem;
    object.materialIndex = 0u;
    object.mixedFEM = false;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.characteristicLength = kThicknessM;
    object.femNodes = {
        {0.0, 0.0, 0.0},
        {kThicknessM, 0.0, 0.0},
        {0.0, kThicknessM, 0.0},
        {0.0, 0.0, kThicknessM},
    };
    object.tetrahedra = {{{0u, 1u, 2u, 3u}}};
    object.femFixedNodes = {0u, 1u, 2u, 3u};
    object.femMaterialIndices = {0u};
    // Stable synthetic identity for this authored one-region instrument.
    object.femMaterialSourceIdentity = {
        0x726f626572747373ull, 0x6f6e2d323032332dull,
        0x736f6674776f6f64ull, 0x2d74657472612d31ull,
    };
    object.femReferenceNodes = object.femNodes;
    object.femReferenceSourceIdentity = {
        0x7265666572656e63ull, 0x652d73696e676c65ull,
        0x2d7465742d303339ull, 0x6d6d2d7631000000ull,
    };
    object.femReferenceDisplacementGradient = true;
    source.objects = {std::move(object)};

    CompileOptions options;
    options.maximumRateExponent = 0;
    options.emitSpecializedMetal = false;
    options.localMaterialNewtonIterations = 16u;
    const auto compiled = compileWorld(source, options);
    if (!compiled.succeeded()) {
        std::string messages;
        for (const auto& diagnostic : compiled.diagnostics)
            messages += diagnostic.message + "\n";
        throw std::runtime_error("Robertsson FEM fixture compile failed: " + messages);
    }
    require(compiled.world.materials.size() == 1u &&
                compiled.world.materials[0].stateCount == kStateCount,
            "compiled source law does not contain all 27 authored states");
    require(compiled.world.dispatch.environmentCount == kEnvironmentCount &&
                compiled.world.dispatch.tetrahedronCount == kTetCount &&
                compiled.world.dispatch.femNodeCount == kNodeCount,
            "compiled instrument layout differs from its two-environment one-tet contract");
    require(compiled.world.materials[0].localNewtonIterations == 16u,
            "local constitutive Newton budget was changed");
    return compiled.world;
}

struct StepOutput {
    RuntimeStateSnapshot state;
    std::vector<nm_float4> reactions;
};

struct Fixture {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = nil;
    CompiledWorld world;
    Runtime runtime;

    explicit Fixture(
        const numi_cardboard::robertsson2023::CompressionEnergyConvention convention)
        : world(compileFixture(convention)) {
        require(device != nil, "Metal device unavailable");
        require([[device name] rangeOfString:@"Apple"].location != NSNotFound &&
                    [[device name] rangeOfString:@"Paravirtual"].location == NSNotFound,
                "physical Apple GPU required");
        queue = [device newCommandQueue];
        require(queue != nil, "failed to create Metal command queue");

        RuntimeConfiguration configuration;
        configuration.metallib = NUMI_MATTER_METALLIB;
        configuration.environmentCount = kEnvironmentCount;
        configuration.adaptiveTransfer = false;
        configuration.captureDiagnostics = true;
        configuration.captureEvents = false;
        const auto initialized = runtime.initialize(world, configuration);
        require(initialized.encoded, initialized.message);
        require(runtime.setNewtonIterationBudget(1u),
                "could not set global Newton iteration budget to one");
        require(runtime.newtonIterationBudget() == 1u,
                "runtime did not retain global Newton budget one");
    }

    RuntimeStateSnapshot snapshot() {
        auto value = runtime.snapshot();
        require(value.available, value.message);
        return value;
    }

    StepOutput step(std::uint32_t controlStep,
                    const std::vector<nm_float4>& targets,
                    const std::vector<std::uint32_t>& resets = {}) {
        require(targets.size() == kEnvironmentCount * kNodeCount,
                "kinematic target count mismatch");
        require(resets.empty() || resets.size() ==
                    static_cast<std::size_t>(controlStep + 1u) * kEnvironmentCount,
                "reset mask must cover the selected control step and both environments");

        std::vector<MRMetalWorldStatusGPU> statuses(kEnvironmentCount);
        for (std::uint32_t environment = 0u;
             environment < kEnvironmentCount; ++environment) {
            statuses[environment].environment = environment;
        }
        auto statusBuffer = sharedBuffer(device, statuses);
        auto targetBuffer = sharedBuffer(device, targets);
        id<MTLBuffer> resetBuffer = nil;
        if (!resets.empty()) resetBuffer = sharedBuffer(device, resets);

        id<MTLCommandBuffer> command = [queue commandBuffer];
        require(command != nil, "failed to create command buffer");
        EncodeRequest request;
        request.commandBuffer = (__bridge void*)command;
        request.environmentStatuses = (__bridge void*)statusBuffer;
        request.femKinematicTargets = (__bridge void*)targetBuffer;
        request.femKinematicTargetCount =
            static_cast<std::uint32_t>(targets.size());
        request.resetMasks = resetBuffer == nil
            ? nullptr : (__bridge void*)resetBuffer;
        request.resetMaskStepStride = resetBuffer == nil ? 0u : kEnvironmentCount;
        request.controlStep = controlStep;
        request.physicsSubsteps = 1u;
        request.timestepSeconds = runtime.timestepSeconds();
        request.runAdaptiveTransfer = false;

        request.phase = EncodePhase::preDynamics;
        const auto pre = runtime.encode(request);
        require(pre.encoded, pre.message);

        id<MTLBuffer> reactionBuffer = (__bridge id<MTLBuffer>)
            runtime.femConstraintReactionBuffer();
        require(reactionBuffer != nil &&
                    reactionBuffer.length >=
                        kEnvironmentCount * kNodeCount * sizeof(nm_float4),
                "runtime FEM constraint reaction buffer is missing or truncated");
        id<MTLBuffer> reactionReadback = [device
            newBufferWithLength:kEnvironmentCount * kNodeCount * sizeof(nm_float4)
                        options:MTLResourceStorageModeShared];
        require(reactionReadback != nil, "failed to allocate reaction readback");
        id<MTLBlitCommandEncoder> copy = [command blitCommandEncoder];
        require(copy != nil, "failed to allocate reaction readback encoder");
        [copy copyFromBuffer:reactionBuffer
                sourceOffset:0u
                    toBuffer:reactionReadback
           destinationOffset:0u
                        size:kEnvironmentCount * kNodeCount * sizeof(nm_float4)];
        [copy endEncoding];

        request.phase = EncodePhase::postCommit;
        const auto post = runtime.encode(request);
        require(post.encoded, post.message);
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
                "native FEM command buffer failed");

        const auto* reactionValues = static_cast<const nm_float4*>(
            reactionReadback.contents);
        require(reactionValues != nullptr, "reaction readback is unavailable");
        StepOutput result;
        result.reactions.assign(reactionValues,
                                reactionValues + kEnvironmentCount * kNodeCount);
        result.state = snapshot();
        return result;
    }
};

std::vector<nm_float4> targetsForShear(const CompiledWorld& world,
                                       const double gamma) {
    std::vector<nm_float4> targets(kEnvironmentCount * kNodeCount);
    for (std::uint32_t node = 0u; node < kNodeCount; ++node) {
        const auto& authored = world.fem.nodes[node].restAndFixed;
        targets[node] = {
            static_cast<float>(double(authored.x) + gamma * double(authored.z)),
            authored.y,
            authored.z,
            1.0f,
        };
        const auto& control = world.fem.nodes[node].restAndFixed;
        targets[kNodeCount + node] = {
            control.x, control.y, control.z, 0.0f
        };
    }
    return targets;
}

void writeHeader(std::ostream& out) {
    out << "step,command_shear,accepted,status_code,damage_candidate,"
           "reaction_node0_x,reaction_node0_y,reaction_node0_z,"
           "reaction_node1_x,reaction_node1_y,reaction_node1_z,"
           "reaction_node2_x,reaction_node2_y,reaction_node2_z,"
           "reaction_node3_x,reaction_node3_y,reaction_node3_z,"
           "p13_from_reaction_Pa";
    for (std::uint32_t state = 0u; state < kStateCount; ++state)
        out << ",accepted_state_" << state;
    out << '\n';
}

void writeAcceptedRow(std::ostream& out, const std::uint32_t step,
                      const double gamma, const StepOutput& result,
                      const RuntimeStateSnapshot& previous) {
    const std::size_t stateBase = 0u;
    const float damage = result.state.femMaterialState.at(stateBase + 26u);
    const std::size_t nodeBase = 0u;
    const auto& stressNode = result.reactions.at(nodeBase + 3u);
    const auto& acceptedNode = result.state.femNodes.at(3u);
    const auto& previousNode = previous.femNodes.at(3u);
    const double inertialForce = double(acceptedNode.positionAndMass.w) *
        (double(acceptedNode.velocityAndInverseMass.x) -
         double(previousNode.velocityAndInverseMass.x)) / kTimestepSeconds;
    // Node 3 has grad(N3)=(0,0,1/h), so f3x=-V*P13/h.
    const double inferredP13 =
        -kThicknessM * (double(stressNode.x) + inertialForce) / kTetVolume;
    const bool accepted = result.state.statuses.at(0u).code == NM_STATUS_SUCCESS;
    out << step << ',' << std::setprecision(17) << gamma << ',' << accepted << ','
        << result.state.statuses.at(0u).code << ',' << damage;
    for (std::uint32_t node = 0u; node < kNodeCount; ++node) {
        const auto& reaction = result.reactions.at(nodeBase + node);
        out << ',' << reaction.x << ',' << reaction.y << ',' << reaction.z;
    }
    out << ',' << inferredP13;
    for (std::uint32_t state = 0u; state < kStateCount; ++state)
        out << ',' << result.state.femMaterialState.at(stateBase + state);
    out << '\n';
}

std::vector<nm_float4> invalidNegativeJTargets(const CompiledWorld& world) {
    constexpr double f11 = -0.1;
    std::vector<nm_float4> targets(kEnvironmentCount * kNodeCount);
    for (std::uint32_t node = 0u; node < kNodeCount; ++node) {
        const auto& reference = world.fem.nodes[node].restAndFixed;
        targets[node] = {static_cast<float>(f11 * double(reference.x)),
                         reference.y, reference.z, 1.0f};
        const auto& control = world.fem.nodes[node].restAndFixed;
        targets[kNodeCount + node] = {
            control.x, control.y, control.z, 0.0f
        };
    }
    return targets;
}

void checkInitialStates(const RuntimeStateSnapshot& state,
                        const CompiledWorld& world) {
    require(state.materialStateStride >= kStateCount,
            "accepted FEM material-state stride is too short");
    for (std::uint32_t environment = 0u;
         environment < kEnvironmentCount; ++environment) {
        const std::size_t base =
            static_cast<std::size_t>(environment) * kTetCount *
            state.materialStateStride;
        for (std::uint32_t index = 0u; index < kStateCount; ++index) {
            const auto initialIndex = world.materials[0].stateInitialOffset + index;
            require(state.femMaterialState.at(base + index) ==
                        world.stateInitials.at(initialIndex),
                    "reset did not restore Robertsson state " +
                        std::to_string(index) + " in environment " +
                        std::to_string(environment));
        }
    }
}

void run(const std::string& csvPath,
         const numi_cardboard::robertsson2023::CompressionEnergyConvention convention) {
    Fixture fixture(convention);
    const auto initial = fixture.snapshot();
    require(initial.femNodes.size() == kEnvironmentCount * kNodeCount,
            "initial node snapshot has unexpected extent");
    checkInitialStates(initial, fixture.world);

    std::ofstream csv(csvPath);
    require(bool(csv), "cannot create FEM path CSV: " + csvPath);
    csv << std::setprecision(std::numeric_limits<double>::max_digits10);
    writeHeader(csv);

    const std::uint32_t finalStep = 2u * kIncrementsPerLeg;
    RuntimeStateSnapshot finalAccepted;
    RuntimeStateSnapshot previous = initial;
    for (std::uint32_t step = 0u; step <= finalStep; ++step) {
        const double gamma = step <= kIncrementsPerLeg
            ? kMaximumShear * double(step) / double(kIncrementsPerLeg)
            : kMaximumShear * double(finalStep - step) /
                  double(kIncrementsPerLeg);
        const auto output = fixture.step(step, targetsForShear(fixture.world, gamma));
        writeAcceptedRow(csv, step, gamma, output, previous);
        csv.flush();
        require(bool(csv), "failed while writing FEM path CSV");
        std::cout << "step=" << step << " shear=" << gamma
                  << " status_env0=" << output.state.statuses.at(0u).code
                  << " status_env1=" << output.state.statuses.at(1u).code
                  << " damage=" << output.state.femMaterialState.at(26u)
                  << '\n' << std::flush;
        for (std::uint32_t environment = 0u;
             environment < kEnvironmentCount; ++environment) {
            require(output.state.statuses.at(environment).code == NM_STATUS_SUCCESS,
                    "valid prescribed shear failed at step " +
                        std::to_string(step) + " environment " +
                        std::to_string(environment) + " code " +
                        std::to_string(output.state.statuses.at(environment).code));
        }
        for (std::uint32_t node = 0u; node < kNodeCount; ++node) {
            const auto& reference = fixture.world.fem.nodes[node].restAndFixed;
            const auto& accepted = output.state.femNodes[node].positionAndMass;
            const double expectedX = double(reference.x) + gamma * double(reference.z);
            require(std::abs(double(accepted.x) - expectedX) < 1.0e-9 &&
                        accepted.y == reference.y && accepted.z == reference.z,
                    "accepted FEM node differs from prescribed homogeneous F13 at step " +
                        std::to_string(step));
        }
        const std::size_t controlBase = kNodeCount;
        for (std::uint32_t node = 0u; node < kNodeCount; ++node) {
            require(std::memcmp(&output.state.femNodes[controlBase + node],
                                &initial.femNodes[controlBase + node],
                                sizeof(NMFEMNodeStateGPU)) == 0,
                    "held control environment changed during driven path");
        }
        for (const auto& reaction : output.reactions) {
            require(std::isfinite(reaction.x) && std::isfinite(reaction.y) &&
                        std::isfinite(reaction.z),
                    "native FEM reaction contains a nonfinite value");
        }
        finalAccepted = output.state;
        previous = output.state;
    }

    const std::size_t drivenStateBase = 0u;
    double maxStateChange = 0.0;
    for (std::uint32_t state = 0u; state < kStateCount; ++state) {
        const auto initialIndex = fixture.world.materials[0].stateInitialOffset + state;
        maxStateChange = std::max(maxStateChange,
            std::abs(double(finalAccepted.femMaterialState[drivenStateBase + state]) -
                     double(fixture.world.stateInitials[initialIndex])));
    }

    // A negative-J target must reject as an ordinary FEM candidate and leave
    // the last valid material/node state intact for both environments.
    const auto invalid = fixture.step(finalStep + 1u,
                                      invalidNegativeJTargets(fixture.world));
    require(invalid.state.statuses.at(0u).code != NM_STATUS_SUCCESS,
            "negative-J commanded grip was unexpectedly accepted");
    require(invalid.state.statuses.at(1u).code == NM_STATUS_SUCCESS,
            "negative-J rejection contaminated the held control environment");
    require(byteEqual(invalid.state.femMaterialState,
                      finalAccepted.femMaterialState),
            "negative-J transaction failed to roll back accepted FEM material history");
    require(byteEqual(invalid.state.femNodes, finalAccepted.femNodes),
            "negative-J transaction failed to roll back accepted FEM nodes");

    // Reset only the driven environment and prove its 27 source states and
    // prescribed node positions return to the authored start; control persists.
    const std::uint32_t resetStep = finalStep + 2u;
    std::vector<std::uint32_t> resetMasks(
        static_cast<std::size_t>(resetStep + 1u) * kEnvironmentCount, 0u);
    resetMasks[static_cast<std::size_t>(resetStep) * kEnvironmentCount] = 1u;
    std::vector<nm_float4> stationaryTargets(kEnvironmentCount * kNodeCount);
    for (std::uint32_t node = 0u; node < kNodeCount; ++node) {
        const auto& initialNode = initial.femNodes[node].positionAndMass;
        stationaryTargets[node] = {initialNode.x, initialNode.y,
                                   initialNode.z, 0.0f};
        const auto& controlNode = initial.femNodes[kNodeCount + node].positionAndMass;
        stationaryTargets[kNodeCount + node] = {controlNode.x, controlNode.y,
                                                 controlNode.z, 0.0f};
    }
    const auto reset = fixture.step(resetStep, stationaryTargets, resetMasks);
    for (std::uint32_t environment = 0u;
         environment < kEnvironmentCount; ++environment) {
        require(reset.state.statuses.at(environment).code == NM_STATUS_SUCCESS,
                "environment reset step failed");
    }
    checkInitialStates(reset.state, fixture.world);
    for (std::uint32_t node = 0u; node < kNodeCount; ++node) {
        require(std::memcmp(&reset.state.femNodes[node], &initial.femNodes[node],
                            sizeof(NMFEMNodeStateGPU)) == 0,
                "reset did not restore driven FEM node state");
        require(std::memcmp(&reset.state.femNodes[kNodeCount + node],
                            &initial.femNodes[kNodeCount + node],
                            sizeof(NMFEMNodeStateGPU)) == 0,
                "reset changed held control FEM node state");
    }

    const auto restored = fixture.runtime.restore(initial);
    require(restored.encoded, "initial snapshot restore failed: " + restored.message);
    const auto afterRestore = fixture.snapshot();
    require(byteEqual(afterRestore.femNodes, initial.femNodes) &&
                byteEqual(afterRestore.femMaterialState, initial.femMaterialState),
            "initial snapshot restore did not round-trip accepted FEM state");

    std::cout << "instrument=native_fem_prescribed_reference_displacement\n"
              << "energy_convention="
              << numi_cardboard::robertsson2023::conventionName(convention) << '\n'
              << "material=Robertsson2023_single_ply_softwood_full_27_state\n"
              << "material_grade_is_corrugated_kraft=0\n"
              << "material_grade_is_starch_laminated=0\n"
              << "fem_tetrahedra=1\n"
              << "fem_thickness_m=" << kThicknessM << '\n'
              << "environment_count=2\n"
              << "newton_budget_global=1\n"
              << "local_material_newton_iterations=16\n"
              << "timestep_s=" << kTimestepSeconds << '\n'
              << "accepted_path_increments=" << finalStep + 1u << '\n'
              << "path=F13_0_to_0.3_to_0\n"
              << "max_return_state_change_at_zero_shear=" << maxStateChange << '\n'
              << "invalid_negative_J_rollback=1\n"
              << "environment_0_reset_roundtrip=1\n"
              << "snapshot_restore_roundtrip=1\n"
              << "state_csv=" << csvPath << '\n'
              << "qualification=manufactured_FEM_state_force_history_coupling_only;"
                 "not_a_free_fold_mesh_or_objective_result\n";
}
} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc == 2 && std::string(argv[1]) == "--cpu-only") {
                const auto paperAbsolute = compileFixture(
                    numi_cardboard::robertsson2023::CompressionEnergyConvention::paper_absolute);
                const auto recenteredRegression = compileFixture(
                    numi_cardboard::robertsson2023::CompressionEnergyConvention::recentered);
                require(paperAbsolute.materials.size() == 1u &&
                            recenteredRegression.materials.size() == 1u &&
                            paperAbsolute.constitutive.size() == 1u &&
                            recenteredRegression.constitutive.size() == 1u,
                        "source energy convention fixtures did not both compile");
                checkOwnGradeAbsoluteEnergy(
                    paperAbsolute.constitutive[0].material,
                    recenteredRegression.constitutive[0].material);
                std::cout << "cpu_only_compile=1\n"
                          << "paper_absolute_state_count="
                          << paperAbsolute.materials[0].stateCount << '\n'
                          << "recentered_regression_state_count="
                          << recenteredRegression.materials[0].stateCount << '\n'
                          << "paper_absolute_fem_node_count="
                          << paperAbsolute.dispatch.femNodeCount << '\n'
                          << "paper_absolute_fem_tetrahedron_count="
                          << paperAbsolute.dispatch.tetrahedronCount << '\n'
                          << "historical_recentered_branch_compiled=1\n"
                          << "runtime_steps=0\n"
                          << "qualification=CPU_source_equation_consistency_and_layout_compile_only\n";
                return 0;
            }
            require(argc == 2,
                    "usage: numi-matter-robertsson-fem-coupling-check output.csv");
            run(argv[1],
                numi_cardboard::robertsson2023::CompressionEnergyConvention::paper_absolute);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "Robertsson FEM coupling check failed: "
                      << error.what() << '\n';
            return 1;
        }
    }
}
